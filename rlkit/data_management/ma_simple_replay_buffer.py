from collections import OrderedDict

import numpy as np

# from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class MASimpleReplayBuffer(SimpleReplayBuffer):

    def __init__(
        self,
        num_agent,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        store_raw_action=False,
    ):
        self._num_agent = num_agent
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, num_agent, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, num_agent, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, num_agent, action_dim))
        # Xiaobai: store raw action if set
        self._store_raw_action = store_raw_action
        if store_raw_action:
            self._raw_actions = np.zeros((max_replay_buffer_size, num_agent, action_dim))
        # Make everything a 3D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, num_agent, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, num_agent, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        # self._env_info_keys = env_info_sizes.keys()
        self._env_info_keys = list(env_info_sizes.keys())

        self._top = 0
        self._size = 0
