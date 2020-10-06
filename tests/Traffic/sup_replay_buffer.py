from collections import OrderedDict

import numpy as np
from rlkit.torch.core import eval_np, np_ify, torch_ify

class SupReplayBuffer:

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        label_dim,
        recurrent=False,
        max_path_length=100,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._label_dim = label_dim
        self._recurrent = recurrent
        if self._recurrent:
            self._max_replay_buffer_size = int(max_replay_buffer_size/max_path_length)
            self._max_path_length = max_path_length
            self._observations = np.zeros((self._max_replay_buffer_size, max_path_length, observation_dim))
            self._actions = np.zeros((self._max_replay_buffer_size, max_path_length, action_dim))
            self._labels = np.zeros((self._max_replay_buffer_size, max_path_length, label_dim))
            self._valids = np.zeros((self._max_replay_buffer_size, max_path_length)).astype(bool)
        else:
            self._max_replay_buffer_size = max_replay_buffer_size
            self._observations = np.zeros((self._max_replay_buffer_size, observation_dim))
            self._actions = np.zeros((self._max_replay_buffer_size, action_dim))
            self._labels = np.zeros((self._max_replay_buffer_size, label_dim))
            self._valids = np.zeros(self._max_replay_buffer_size).astype(bool)
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, label, valid):
        # labes: label_num x label_dim
        self._observations[self._top] = np_ify(observation)
        self._actions[self._top] = np_ify(action)
        self._labels[self._top] = np_ify(label)
        self._valids[self._top] = np_ify(valid)
        self._advance()

    def add_batch(self, batch_obs, bacth_act, batch_label, valid_batch):
        # batch_obs: batch x obs_dim
        # batch_labels: batch x label_dim
        for obs, act, label, valid in zip(batch_obs, bacth_act, batch_label, valid_batch):
            self.add_sample(obs, act, label, valid)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        if self._recurrent:
            batch_size = int(batch_size/self._max_path_length)
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            labels=self._labels[indices],
            valids=self._valids[indices]
        )
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
