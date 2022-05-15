from mcs.manager.simple_env_manager import SimpleEnvManager
from mcs.manager.subproc_env_manager import SubProcEnvManager
from mcs.manager.base.manager_module import EnvManagerModule


MANAGER_REG = [SimpleEnvManager, SubProcEnvManager]
