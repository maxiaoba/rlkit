from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class PrioritizedReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            alpha=0.6,
            beta=0.4,
            env_info_sizes=None,
            **kwargs
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.zeros((max_replay_buffer_size,))

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        if self._size > 0.:
            priority = np.max(self._priorities[:self._size])
        else:
            priority = 1e-3
        self._priorities[self._top] = priority
        if (self._store_raw_action) and ('agent_info' in kwargs):
            if 'raw_action' in kwargs['agent_info']:
                self._raw_actions[self._top] = kwargs['agent_info']['raw_action']

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def random_batch(self, batch_size):
        # indices = np.random.randint(0, self._size, batch_size)
        sample_weights = (self._priorities[:self._size]**self.alpha)/np.sum((self._priorities[:self._size]**self.alpha))
        indices = np.random.choice(np.arange(self._size), batch_size, p=sample_weights)
        importance_weights = (self._size*sample_weights)**(-self.beta)
        importance_weights = importance_weights/np.max(importance_weights)
        importance_weights = importance_weights[indices]
        batch = dict(
            indices=indices,
            importance_weights=importance_weights,
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        if self._store_raw_action:
            batch['raw_actions']=self._raw_actions[indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def update_priority(self, indices, td_errors):
        self._priorities[indices] = np.abs(td_errors)
