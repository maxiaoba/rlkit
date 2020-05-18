import copy

from rlkit.data_management.ma_env_replay_buffer import MAEnvReplayBuffer
# from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.ma_path_collector import MAMdpPathCollector
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.policies.gumbel_softmax_policy import GumbelSoftmaxMlpPolicy
from rlkit.torch.maddpg.maddpg import MADDPGTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from log_path import get_ma_path_information

def experiment(variant):
    num_agent = variant['num_agent']
    from rlkit.envs.zmq_env import ZMQEnv
    expl_env = ZMQEnv(variant['port'])
    eval_env = expl_env
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf_n, policy_n, target_qf_n, target_policy_n, eval_policy_n, expl_policy_n = \
        [], [], [], [], [], []
    for i in range(num_agent):
        qf = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            **variant['qf_kwargs']
        )
        policy = GumbelSoftmaxMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_qf = copy.deepcopy(qf)
        target_policy = copy.deepcopy(policy)
        eval_policy = ArgmaxDiscretePolicy(policy,use_preactivation=True)
        expl_policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(expl_env.action_space),
            eval_policy,
        )
        qf_n.append(qf)
        policy_n.append(policy)
        target_qf_n.append(target_qf)
        target_policy_n.append(target_policy)
        eval_policy_n.append(eval_policy)
        expl_policy_n.append(expl_policy)

    eval_path_collector = MAMdpPathCollector(eval_env, eval_policy_n)
    expl_path_collector = MAMdpPathCollector(expl_env, expl_policy_n)
    replay_buffer = MAEnvReplayBuffer(variant['replay_buffer_size'], expl_env, num_agent=num_agent)
    trainer = MADDPGTrainer(
        qf_n=qf_n,
        target_qf_n=target_qf_n,
        policy_n=policy_n,
        target_policy_n=target_policy_n,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        log_path_function=get_ma_path_information,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='BlockedRoad')
    parser.add_argument('--port', type=int, default=9393)
    parser.add_argument('--log_dir', type=str, default='MADDPGGumbel')
    parser.add_argument('--soft', action='store_true', default=False)
    parser.add_argument('--double_q', action='store_true', default=False)
    parser.add_argument('--online_action', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name
    main_dir = args.log_dir\
                +('soft' if args.soft else 'hard')\
                +('double_q' if args.double_q else '')\
                +('online_action' if args.online_action else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        num_agent=2,
        port=args.port,
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 1000),
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=200,
            batch_size=(args.bs if args.bs else 256),
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=(args.lr if args.lr else 1e-3),
            policy_learning_rate=(args.lr if args.lr else 1e-4),
            online_action=args.online_action,
            double_q=args.double_q,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hard=(not args.soft),
            hidden_sizes=[400, 300],
        ),
        replay_buffer_size=int(1E6),
    )
    import os
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir,'variant.json'),'w') as out_json:
        import json
        json.dump(variant,out_json,indent=2)
    import sys
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    setup_logger(args.exp_name+'/'+main_dir, variant=variant,
                snapshot_mode=args.snapshot_mode, snapshot_gap=args.snapshot_gap,
                log_dir=log_dir)
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
