import os

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv


class UR3Env(MujocoEnv, utils.EzPickle):
    '''Simulation environment made using gymnasium and mujoco.'''

    ##### class variables #####

    # full path to the xml file (MJCF)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/single_ur3_base.xml')
    
    # a frame refers to a step in mojoco simulation
    mujocoenv_frame_skip = 1

    ##### state #####

    # the number of robot and gripper joint coordinates
    ur3_nqpos, gripper_nqpos = 6, 10

    # the number of robot and gripper joint velocities
    ur3_nqvel, gripper_nqvel = 6, 10

    # 4 objects on the table
    objects_nqpos = [7, 7, 7, 7]  # 3 trans, 4 quaternion
    objects_nqvel = [6, 6, 6, 6]  # 3 trans, 3 rotation

    ##### action #####

    # the number of actions for robot and gripper, resp
    ur3_nact, gripper_nact = 6, 2

    # to check collision between objects in simulation, set to True
    ENABLE_COLLISION_CHECKER = False


    def __init__(self):
        if self.ENABLE_COLLISION_CHECKER:
            self._define_collision_checker_variables()


    
