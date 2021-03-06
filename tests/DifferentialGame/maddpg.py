import copy
import torch.nn as nn
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.ma_eval_util import get_generic_ma_path_information

def experiment(variant):
    num_agent = variant['num_agent']
    from differential_game import DifferentialGame
    expl_env = DifferentialGame(game_name=args.exp_name)
    eval_env = DifferentialGame(game_name=args.exp_name)
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf_n, policy_n, target_qf_n, target_policy_n, eval_policy_n, expl_policy_n = \
        [], [], [], [], [], []
    qf2_n, target_qf2_n = [], []
    for i in range(num_agent):
        from rlkit.torch.networks import FlattenMlp
        qf = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            hidden_sizes=[variant['qf_kwargs']['hidden_dim']]*2,
        )
        target_qf = copy.deepcopy(qf)
        from rlkit.torch.policies.deterministic_policies import TanhMlpPolicy
        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=[variant['policy_kwargs']['hidden_dim']]*2,
        )
        target_policy = copy.deepcopy(policy)
        eval_policy = policy
        from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
        if variant['random_exploration']:
            from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=EpsilonGreedy(expl_env.action_space, prob_random_action=1.0),
                policy=policy,
            )
        else:
            from rlkit.exploration_strategies.ou_strategy import OUStrategy
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=OUStrategy(action_space=expl_env.action_space),
                policy=policy,
            )
        
        qf_n.append(qf)
        policy_n.append(policy)
        target_qf_n.append(target_qf)
        target_policy_n.append(target_policy)
        eval_policy_n.append(eval_policy)
        expl_policy_n.append(expl_policy)
        if variant['trainer_kwargs']['double_q']:
            qf2 = FlattenMlp(
                input_size=(obs_dim*num_agent+action_dim*num_agent),
                output_size=1,
                hidden_sizes=[variant['qf_kwargs']['hidden_dim']]*2,
            )
            target_qf2 = copy.deepcopy(qf2)
            qf2_n.append(qf2)
            target_qf2_n.append(target_qf2)

    from rlkit.samplers.data_collector.ma_path_collector import MAMdpPathCollector
    eval_path_collector = MAMdpPathCollector(eval_env, eval_policy_n)
    expl_path_collector = MAMdpPathCollector(expl_env, expl_policy_n)

    from rlkit.data_management.ma_env_replay_buffer import MAEnvReplayBuffer
    replay_buffer = MAEnvReplayBuffer(variant['replay_buffer_size'], expl_env, num_agent=num_agent)

    from rlkit.torch.maddpg.maddpg import MADDPGTrainer
    trainer = MADDPGTrainer(
        qf_n=qf_n,
        target_qf_n=target_qf_n,
        policy_n=policy_n,
        target_policy_n=target_policy_n,
        qf2_n = qf2_n,
        target_qf2_n = target_qf2_n,
        **variant['trainer_kwargs']
    )

    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
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
    parser.add_argument('--exp_name', type=str, default='zero_sum')
    parser.add_argument('--log_dir', type=str, default='MADDPG')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--oa', action='store_true', default=False) # online action
    parser.add_argument('--dq', action='store_true', default=False) # doube q
    parser.add_argument('--re', action='store_true', default=False) # random exploration
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
                +('hidden'+str(args.hidden))\
                +('oa' if args.oa else '')\
                +('dq' if args.dq else '')\
                +('re' if args.re else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        num_agent=2,
        random_exploration=args.re,
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 100),
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=(args.bs if args.bs else 256),
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=(args.lr if args.lr else 1e-3),
            policy_learning_rate=(args.lr if args.lr else 1e-4),
            online_action=args.oa,
            double_q=args.dq,
        ),
        qf_kwargs=dict(
            hidden_dim=args.hidden,
        ),
        policy_kwargs=dict(
            hidden_dim=args.hidden,
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
