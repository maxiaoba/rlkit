import abc
import os.path as osp
import numpy as np
import torch
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.samplers.data_collector import PathCollector
from rlkit.core import logger

class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            save_best=False, # only works in single agent RL
            **kwargs,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer=None,
            **kwargs
        )

        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.save_best = save_best

    def _train(self):
        self.training_mode(False)
        best_eval_return = -np.inf
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            if self.save_best:
                eval_returns = [sum(path["rewards"]) for path in eval_paths]
                eval_avg_return = np.mean(eval_returns)
                if eval_avg_return > best_eval_return:
                    best_eval_return = eval_avg_return
                    snapshot = self._get_snapshot()
                    file_name = osp.join(logger._snapshot_dir, 'best.pkl')
                    torch.save(snapshot, file_name)
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=True,
                )
                gt.stamp('exploration sampling', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    self.trainer.train(expl_paths)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
