from collections import OrderedDict

from mcs.actor import ActorModule
from mcs.actor.base.ac_helper import ACActorHelperMixin


class ACActorEval(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, action_space):
        return cls(action_space)

    @staticmethod
    def output_space(action_space):
        head_dict = {"critic": (1,), **action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        actions = OrderedDict()

        for key in self.action_keys:
            logit = self.flatten_logits(preds[key])

            softmax = self.softmax(logit)
            action = self.sample_action(softmax)

            actions[key] = action.cpu()
        return actions, {"value": preds["critic"].squeeze(-1)}

    @classmethod
    def _exp_spec(
        cls, rollout_len, batch_sz, obs_space, act_space, internal_space
    ):
        return {}


class ACActorEvalSample(ACActorEval):
    def compute_action_exp(self, preds, internals, obs, available_actions):
        actions = OrderedDict()

        for key in self.action_keys:
            logit = self.flatten_logits(preds[key])

            softmax = self.softmax(logit)
            action = self.sample_action(softmax)

            actions[key] = action.cpu()
        return actions, {"value": preds["critic"].squeeze(-1)}
