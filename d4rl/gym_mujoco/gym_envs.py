from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv

from offline.d4rl.offline_environment import OfflineEnv


class OfflineAntEnv(OfflineEnv, AntEnv):
    pass


class OfflineHopperEnv(OfflineEnv, HopperEnv):
    pass


class OfflineHalfCheetahEnv(OfflineEnv, HalfCheetahEnv):
    pass


class OfflineWalker2dEnv(OfflineEnv, Walker2dEnv):
    pass
