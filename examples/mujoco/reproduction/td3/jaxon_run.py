import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

import os

DEFAULT_CAMERA_CONFIG = {
        'trackbodyid':1,
        'distance': 4.0,
        'lookat': np.array((0.0, 0.0, 2.0)),
        'elevation': -20.0,
        }


def mass_center(model, sim):
    mass = np.expand_dims(mdoel.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis = 0) / np.sim(mass))[0:2].copy()

class JaxonRunEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
            xml_file = 'JAXON_JVRC.xml',
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1, 
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.8, 1.5),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_const_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
                exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.getcwd()+'/JAXON_JVRC_models/', xml_file), 5)
    
    @property
    def healthy_reward(self):
        return float(
                self.is_healthy
                or self._terminate_when_unhealthy
                ) * self._healthy_reward
