from .base.actor_module import ActorModule
from .ac_rollout import ACRolloutActorTrain
from mcs.actor.ac_eval import ACActorEval, ACActorEvalSample
from .impala import ImpalaHostActor, ImpalaWorkerActor,ImpalaHostTargetActor

ACTOR_REG = [
    ACRolloutActorTrain,
    ACActorEval,
    ACActorEvalSample,
    ImpalaHostActor,
    ImpalaWorkerActor,
    ImpalaHostTargetActor
]
