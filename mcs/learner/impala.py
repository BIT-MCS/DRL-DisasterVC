# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import torch
import numpy as np
from mcs.utils.util import listd_to_dlist, dlist_to_listd
import torch.nn as nn
from .base import LearnerModule
from .base.dm_return_scale import DeepMindReturnScaler


class ImpalaLearner(LearnerModule):
    """
    Reference implementation:
    https://github.com/deepmind/scalable_agent/blob/master/vtrace.py
    """

    args = {
        "discount": 0.99,
        "minimum_importance_value": 1.0,
        "minimum_importance_policy": 1.0,
        "entropy_weight": 0.01,
    }

    def __init__(
            self,
            reward_normalizer,
            minimum_importance_value,
            minimum_importance_policy,
            entropy_weight,
            use_pixel_control,
            cell_size,
            batch_size,
            sequence_len,
            pixel_control_loss_gamma,
            gae_gammma,
            gae_lambda,
            probs_clip,
            target_worker_clip_rho,
            num_actions
    ):
        self.reward_normalizer = reward_normalizer
        self.minimum_importance_value = minimum_importance_value
        self.minimum_importance_policy = minimum_importance_policy
        self.entropy_weight = entropy_weight
        self.use_pixel_control = use_pixel_control
        self.cell_size = cell_size
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.pixel_control_loss_gamma = pixel_control_loss_gamma
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gammma


        self.probs_clip = probs_clip
        self.target_worker_clip_rho = target_worker_clip_rho

        if self.use_pixel_control:
            self.mse = nn.MSELoss()
            self.avgPool = nn.AvgPool2d(kernel_size=cell_size, stride=cell_size)

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer,
            minimum_importance_value=args.minimum_importance_value,
            minimum_importance_policy=args.minimum_importance_policy,
            entropy_weight=args.entropy_weight,
            use_pixel_control=args.use_pixel_control,
            cell_size=args.cell_size,
            batch_size=args.nb_env * args.nb_learn_batch,
            sequence_len=args.rollout_len,
            pixel_control_loss_gamma=args.pixel_control_loss_gamma,
            gae_gammma=args.gae_gamma,
            gae_lambda=args.gae_lambda,
            probs_clip=args.probs_clip,
            target_worker_clip_rho=args.target_worker_clip_rho,
            num_actions=args.action_space
        )

    def learn_step(self, updater, network, target_network, experiences, next_obs, internals):
    

        with torch.no_grad():
            results, _, _ = network(next_obs, internals)

            b_last_values = results["critic"].squeeze(1).data
            q_last = results["pixel_control"]

        if self.use_pixel_control:
            obs = []
            for o in experiences.observations:
                obs.append(o['Box'])
            obs.append(next_obs['Box'])

            q = []
            for p in experiences.pixel_control:
                q.append(p)
            q.append(q_last)
            q = torch.stack(q)
            obs = torch.stack(obs) 
            obs = torch.squeeze(obs, dim=2)  
      
            pc_reward = self.piexl_control_rewards(obs)
            q_t = q[:-1]
            q_tm1 = q[1:,]
            action = []
            for action_key in experiences.actions[0].keys():
                for a in experiences.actions:
                    action.append(a[action_key])
            action = torch.stack(action)
            terminals = torch.stack(experiences.terminals)
            terminal_masks = self.pixel_control_loss_gamma * (1.0 - terminals.float())
            pixel_loss = self.pixel_control_loss(q_t, q_tm1, action, pc_reward, terminal_masks, self.num_actions)

        target_r_log_probs = []
        for b_action, b_log_softs in zip(
                experiences.actions, experiences.target_log_softmaxes
        ):
            # print('b_action', b_action)
            # print('b_log_soft',b_log_softs.shape) b_log_soft  torch.Size([64, 2, 18])
            k_log_probs = []
            for act_tensor, log_soft in zip(
                    b_action.values(), b_log_softs.unbind(1)
            ):
                # print('act_tensor', act_tensor.shape)  act_tensor torch.Size([64])
                # print('log_soft', log_soft.shape) log_soft torch.Size([64, 18])
                log_prob = log_soft.gather(1, act_tensor.unsqueeze(1))  # 64,7  64,1
                k_log_probs.append(log_prob)
            target_r_log_probs.append(torch.cat(k_log_probs, dim=1))
        r_log_probs_target = torch.stack(target_r_log_probs).detach()

        # Gather host log_probs
        r_log_probs = []
        for b_action, b_log_softs in zip(
                experiences.actions, experiences.log_softmaxes
        ):
            # print('b_action', b_action)
            # print('b_log_soft',b_log_softs.shape) b_log_soft  torch.Size([64, 2, 18])
            k_log_probs = []
            for act_tensor, log_soft in zip(
                    b_action.values(), b_log_softs.unbind(1)
            ):
                # print('act_tensor', act_tensor.shape)  act_tensor torch.Size([64])
                # print('log_soft', log_soft.shape) log_soft torch.Size([64, 18])
                log_prob = log_soft.gather(1, act_tensor.unsqueeze(1))  # 64,18  64,1
                k_log_probs.append(log_prob)
            r_log_probs.append(torch.cat(k_log_probs, dim=1))

        r_log_probs_learner = torch.stack(r_log_probs)  

        r_log_probs_actor = torch.stack(experiences.log_probs)
        # print('r_log_probs_actor', r_log_probs_actor.shape)
        r_rewards = self.reward_normalizer(
            torch.stack(experiences.rewards)
        )  # normalize rewards
        r_values = torch.stack(experiences.values)
        r_terminals = torch.stack(experiences.terminals)
        r_entropies = torch.stack(experiences.entropies)
        r_dterminal_masks = self.gae_gamma * (1.0 - r_terminals.float())
        r_lambda_masks = self.gae_lambda * (1.0 - r_terminals.float())

        with torch.no_grad():
            r_log_diffs = r_log_probs_learner - r_log_probs_actor
            # print('r_log_diffs',r_log_diffs.shape) 
            vtrace_target, pg_advantage, importance = self._vtrace_returns(
                r_log_diffs,
                r_dterminal_masks,
                r_lambda_masks,
                r_rewards,
                r_values,
                b_last_values,
                self.minimum_importance_value,
                self.minimum_importance_policy
            )


        min_worker_divide_target_rhp = torch.clamp(torch.exp(r_log_probs_actor - r_log_probs_target), min=0,
                                                   max=self.target_worker_clip_rho)
        upper_theta_divide_worker = min_worker_divide_target_rhp * torch.exp(r_log_probs_learner - r_log_probs_actor)
        surrogate_loss = torch.min(pg_advantage * upper_theta_divide_worker,
                                   pg_advantage * torch.clamp(upper_theta_divide_worker, min=1 - self.probs_clip,
                                                              max=self.probs_clip + 1))
        # surrogate_loss = r_log_probs_learner * pg_advantage

        value_loss = 0.5 * (vtrace_target - r_values).pow(2).mean()
        policy_loss = torch.mean(-surrogate_loss)
        entropy_loss = torch.mean(-r_entropies) * self.entropy_weight

        if self.use_pixel_control:
            updater.step(value_loss + policy_loss + entropy_loss + pixel_loss)
        else:
            updater.step(value_loss + policy_loss + entropy_loss)

        if self.use_pixel_control:
            losses = {
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy_loss": entropy_loss,
                "pixel_loss": pixel_loss,
            }
        else:
            losses = {
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy_loss": entropy_loss,
            }
        metrics = {"importance": importance.mean()}
        return losses, metrics

    @staticmethod
    def _vtrace_returns(
            log_prob_diffs,
            gamma_terminal_mask,
            lambda_terminal_mask,
            r_rewards,
            r_values,
            bootstrap_value,  # [64]
            min_importance_value,
            min_importance_policy
    ):

        rollout_len = log_prob_diffs.shape[0]

        importance = torch.exp(log_prob_diffs)  
        clamped_importance_value = importance.clamp(max=min_importance_value)  
        # if multiple actions take the average, (dim 3 is seq, batch, # actions)
        if clamped_importance_value.dim() == 3:
            clamped_importance_value = clamped_importance_value.mean(-1)  # ([25, 64])

        # create nstep vtrace return
        # first create d_tV of function 1 in the paper
        values_t_plus_1 = torch.cat(
            (r_values[1:], bootstrap_value.unsqueeze(0))  # [20,64]
        )
        diff_value_per_step = clamped_importance_value * (  
                r_rewards + gamma_terminal_mask * values_t_plus_1 - r_values
        )

        # reverse over the values to create the summed importance weighted
        # return everything on the right side of the plus in function 1 of
        # the paper
        advantage = []
        nstep_v = 0.0
        if min_importance_policy != 1 or min_importance_value != 1:
            raise NotImplementedError()

        for i in reversed(range(rollout_len)):
            nstep_v = (
                    diff_value_per_step[i]
                    + gamma_terminal_mask[i]
                    * lambda_terminal_mask[i]
                    * clamped_importance_value[i]
                    * nstep_v
            )
            # print('nstep_v',nstep_v.shape) [64]
            advantage.append(nstep_v)
        # reverse to a forward in time list
        advantage = torch.stack(list(reversed(advantage)))  

        # Add V(s) to finish computation of v_s
        v_s = r_values + advantage  # [25,64]

        # advantage is pg_importance * (v_s of t+1 - values)
        clamped_importance_pg = importance.clamp(max=min_importance_policy)

        v_s_tp1 = torch.cat((v_s[1:], bootstrap_value.unsqueeze(0)))
        advantage = r_rewards + gamma_terminal_mask * v_s_tp1 - r_values

        # if multiple actions broadcast the advantage to be weighted by the
        # different actions importance
        # (dim 3 is seq, batch, # actions)
        if importance.dim() == 3:
            advantage = advantage.unsqueeze(-1)
        # adv = ((adv - adv.mean()) / (adv.std() + 1e-6))
        weighted_advantage = clamped_importance_pg * advantage
        return v_s, weighted_advantage, importance

    def piexl_control_rewards(self, observation):
        # observation [T,B,D,H,W]
        abs_diff = torch.abs(observation[1:] - observation[:-1])
        avg_diff = self.avgPool(abs_diff.float())
        return avg_diff

    def pixel_control_loss(self, q_t, q_tm1, action, reward, discount, num_action):
        # print(q_t.shape) #20 64 441 18
        # print(q_tm1.shape)   #20 64 441 18
        # print(action.shape)   #20 64 
        # print(reward.shape)   #20 64 21 21
        # print(discount.shape)    #20 64 
        # print(num_action) #18

        q_t = torch.reshape(q_t, [self.sequence_len, -1, num_action])  # [T,BHWD,N]
        q_tm1 = torch.reshape(q_tm1, [self.sequence_len, -1, num_action])  # [T,BHWD,N]
        V = torch.max(q_tm1, dim=2)[0]  # [T,BHWD]
        reward = torch.reshape(reward, [self.sequence_len, -1])  # [T,BHWD]

        discount = discount.unsqueeze(-1).unsqueeze(-1)  
        discount = discount.repeat([1, 1] + [21, 21])
        discount = discount.reshape(self.sequence_len, -1)  # [T,BHWD]

        action = action.unsqueeze(-1).unsqueeze(-1)  
        action = action.repeat([1, 1] + [21, 21])
        action = action.reshape(self.sequence_len, -1)  # [T,BHWD]

        q = self.batched_index(q_t, action)  # [T,BHWD]
        #print(reward.shape,discount.shape,V.shape)
        step_value = reward + discount * V
        # print(reward[:,0])
        n_step_v = 0
        n_step_target = []
        for i in reversed(range(self.sequence_len)):
            n_step_v = step_value[i] + discount[i] * n_step_v
            n_step_target.append(n_step_v)
        target = torch.stack(list(reversed(n_step_target)))
        # print(q[:,0])
        # print(target[:,0])
        loss = self.mse(target,q)
        return 0.0001*loss.mean()

    def batched_index(self, values, indices, keepdims=None):
        # values [T,BHWD,N]
        # indices [T,BHWD,1]
        m_zeros = torch.zeros(values.shape).cuda()
        one_hot = m_zeros.scatter_(1, indices.unsqueeze(dim=2), 1)
        return torch.mul(values, one_hot).sum(dim=2)
