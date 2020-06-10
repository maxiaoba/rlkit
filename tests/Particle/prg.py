import copy

from rlkit.data_management.ma_env_replay_buffer import MAEnvReplayBuffer
# from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.ma_path_collector import MAMdpPathCollector
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.policies.deterministic_policies import TanhMlpPolicy
from rlkit.torch.prg.prg import PRGTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.core.ma_eval_util import get_generic_ma_path_information

def experiment(variant):
    import sys
    sys.path.append("./multiagent-particle-envs")
    from make_env import make_env
    from particle_env_wrapper import ParticleEnv
    expl_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
    eval_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
    num_agent = expl_env.num_agent
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1_n, qf2_n, cactor_n, policy_n, target_qf1_n, target_qf2_n, target_policy_n, expl_policy_n, eval_policy_n = \
        [], [], [], [], [], [], [], [], []
    for i in range(num_agent):
        qf1 = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            **variant['qf_kwargs']
        )
        qf2 = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            **variant['qf_kwargs']
        )
        cactor = TanhMlpPolicy(
            input_size=(obs_dim*num_agent+action_dim*(num_agent-1)),
            output_size=action_dim,
            **variant['cactor_kwargs']
        )
        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_qf1 = copy.deepcopy(qf1)
        target_qf2 = copy.deepcopy(qf2)
        target_policy = copy.deepcopy(policy)
        eval_policy = policy
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=OUStrategy(action_space=expl_env.action_space),
            policy=policy,
        )
        qf1_n.append(qf1)
        qf2_n.append(qf2)
        cactor_n.append(cactor)
        policy_n.append(policy)
        target_qf1_n.append(target_qf1)
        target_qf2_n.append(target_qf2)
        target_policy_n.append(target_policy)
        expl_policy_n.append(expl_policy)
        eval_policy_n.append(eval_policy)
        

    eval_path_collector = MAMdpPathCollector(eval_env, eval_policy_n)
    expl_path_collector = MAMdpPathCollector(expl_env, expl_policy_n)
    replay_buffer = MAEnvReplayBuffer(variant['replay_buffer_size'], expl_env, num_agent=num_agent)
    trainer = PRGTrainer(
        env=expl_env,
        qf1_n=qf1_n,
        target_qf1_n=target_qf1_n,
        qf2_n = qf2_n,
        target_qf2_n = target_qf2_n,
        policy_n=policy_n,
        target_policy_n=target_policy_n,
        cactor_n=cactor_n,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        log_path_function=get_generic_ma_path_information,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='simple')
    parser.add_argument('--log_dir', type=str, default='PRG')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--online_action', action='store_true', default=False)
    parser.add_argument('--target_action', action='store_true', default=False)
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
                +'k'+str(args.k)\
                +('online_action' if args.online_action else '')\
                +('target_action' if args.target_action else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 500),
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=200,
            num_expl_steps_per_train_loop=200,
            min_num_steps_before_training=200,
            max_path_length=100,
            batch_size=(args.bs if args.bs else 256),
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=(args.lr if args.lr else 1e-3),
            cactor_learning_rate=(args.lr if args.lr else 1e-4),
            policy_learning_rate=(args.lr if args.lr else 1e-4),
            logit_level=args.k,
            use_entropy_loss=False,
            online_action=args.online_action,
            target_action=args.target_action,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        cactor_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
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
