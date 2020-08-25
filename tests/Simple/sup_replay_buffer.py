from collections import OrderedDict

import numpy as np


class SupReplayBuffer:

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        label_dim,
    ):
        self._observation_dim = observation_dim
        self._label_dim = label_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._labels = np.zeros((max_replay_buffer_size, label_dim))

        self._top = 0
        self._size = 0

    def add_sample(self, observation, label):
        # labes: label_num x label_dim
        self._observations[self._top] = observation
        self._labels[self._top] = label
        self._advance()

    def add_batch(self, batch_obs, batch_label):
        # batch_obs: batch x obs_dim
        # batch_labels: batch x label_dim
        for obs, label in zip(batch_obs, batch_label):
            self.add_sample(obs, label)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            labels=self._labels[indices],
        )
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
