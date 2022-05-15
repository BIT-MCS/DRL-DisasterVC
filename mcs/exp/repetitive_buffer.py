from mcs.exp.rollout import Rollout
import copy
import torch


class RepetitiveBuffer(object):


    def __init__(self, inqueue, buffer_size, num_k, exp):  # (q,4,2_)
        self.inqueue = inqueue
        self.size = buffer_size  # 4

        self.max_visit_times = num_k  # 2
        self.cur_max_ttl = num_k  # 2

        self.buffers = [None] * buffer_size
        self.buffer_count = [0] * buffer_size  # [0,0,0,0]
        self.idx = 0
        self.exp = copy.deepcopy(exp)

    def get(self, target_actor, target_network, device):
        terminal_rewards = []
        terminal_infos = []
        if self.buffer_count[self.idx] <= 0:
            self.exp.clear()
            # Get batch from queue
            rollouts, terminal_rewards, terminal_infos = self.inqueue.get()

            # Iterate forward on batch
            self.exp.write_exps(rollouts)

            self.exp.to(device)
            r = self.exp.read()
            internals = {k: ts[0].unbind(0) for k, ts in r.internals.items()}
            with torch.no_grad():
                for obs, rewards in zip(
                        r.observations, r.rewards
                ):
                    _, t_h_exp, t_internals = target_actor.act(
                        target_network, obs, internals
                    )
                    self.exp.write_actor(t_h_exp, no_env=True)
            self.exp.reset_index()
            self.buffers[self.idx] = copy.deepcopy(self.exp)
            self.buffer_count[self.idx] = self.max_visit_times

        buf = self.buffers[self.idx]
        self.buffer_count[self.idx] -= 1
        released = self.buffer_count[self.idx] <= 0
        if released:
            self.buffers[self.idx] = None
        self.idx = (self.idx + 1) % self.size
        return buf, terminal_rewards, terminal_infos

    def setMaxTimes(self, times):
        self.max_visit_times = times
