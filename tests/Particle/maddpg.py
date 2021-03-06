import copy
import torch.nn as nn
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.ma_eval_util import get_generic_ma_path_information

def experiment(variant):
    import sys
    sys.path.append("./multiagent-particle-envs")
    from make_env import make_env
    from particle_env_wrapper import ParticleEnv
    expl_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False,world_args=variant['world_args']))
    eval_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False,world_args=variant['world_args']))
    num_agent = expl_env.num_agent
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if variant['load_kwargs']['load']:
        load_dir = variant['load_kwargs']['load_dir']
        load_epoch = variant['load_kwargs']['load_epoch']
        load_data = torch.load('{}/itr_{}.pkl'.format(load_dir,load_epoch),map_location='cpu')
        qf_n = load_data['trainer/qf_n']
        target_qf_n = load_data['trainer/target_qf_n']
        qf2_n, target_qf2_n = [], []
        policy_n = load_data['trainer/policy_n']
        target_policy_n = load_data['trainer/target_policy_n']

        qf_optimizer_n = load_data['trainer/qf_optimizer_n']
        qf2_optimizer_n = None
        policy_optimizer_n = load_data['trainer/policy_optimizer_n']

        replay_buffer = load_data['replay_buffer']
    else:
        qf_n, policy_n, target_qf_n, target_policy_n, eval_policy_n, expl_policy_n = \
            [], [], [], [], [], []
        qf2_n, target_qf2_n = [], []
        qf_optimizer_n, qf2_optimizer_n, policy_optimizer_n = None, None, None
        for i in range(num_agent):
            from rlkit.torch.networks import FlattenMlp
            qf = FlattenMlp(
                input_size=(obs_dim*num_agent+action_dim*num_agent),
                output_size=1,
                hidden_sizes=[variant['qf_kwargs']['hidden_dim']]*variant['qf_kwargs']['num_layer'],
            )
            target_qf = copy.deepcopy(qf)
            from rlkit.torch.policies.deterministic_policies import TanhMlpPolicy
            policy = TanhMlpPolicy(
                input_size=obs_dim,
                output_size=action_dim,
                hidden_sizes=[variant['policy_kwargs']['hidden_dim']]*variant['policy_kwargs']['num_layer'],
            )
            target_policy = copy.deepcopy(policy)
            
            qf_n.append(qf)
            policy_n.append(policy)
            target_qf_n.append(target_qf)
            target_policy_n.append(target_policy)

            from rlkit.data_management.ma_env_replay_buffer import MAEnvReplayBuffer
            replay_buffer = MAEnvReplayBuffer(variant['replay_buffer_size'], expl_env, num_agent=num_agent)

    from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    expl_policy_n = [PolicyWrappedWithExplorationStrategy(
                            exploration_strategy=OUStrategy(action_space=expl_env.action_space),
                            policy=policy,
                        ) for policy in policy_n]
    eval_policy_n = policy_n

    from rlkit.samplers.data_collector.ma_path_collector import MAMdpPathCollector
    eval_path_collector = MAMdpPathCollector(eval_env, eval_policy_n)
    expl_path_collector = MAMdpPathCollector(expl_env, expl_policy_n)

    from rlkit.torch.maddpg.maddpg import MADDPGTrainer
    trainer = MADDPGTrainer(
        qf_n=qf_n,
        target_qf_n=target_qf_n,
        policy_n=policy_n,
        target_policy_n=target_policy_n,
        qf2_n = qf2_n,
        target_qf2_n = target_qf2_n,
        qf_optimizer_n=qf_optimizer_n,
        qf2_optimizer_n=qf2_optimizer_n,
        policy_optimizer_n=policy_optimizer_n,
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
    parser.add_argument('--exp_name', type=str, default='simple')
    parser.add_argument('--boundary', action='store_true', default=False)
    parser.add_argument('--num_ag', type=int, default=None)
    parser.add_argument('--num_adv', type=int, default=None)
    parser.add_argument('--num_l', type=int, default=None)
    parser.add_argument('--mpl', type=int, default=25) # max path length
    parser.add_argument('--log_dir', type=str, default='MADDPG')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--oa', action='store_true', default=False) # online action
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name\
                +('bd' if args.boundary else '')\
                +(('ag'+str(args.num_ag)) if args.num_ag else '')\
                +(('adv'+str(args.num_adv)) if args.num_adv else '')\
                +(('l'+str(args.num_l)) if args.num_l else '')\
                +'_mpl'+str(args.mpl)
    main_dir = args.log_dir\
                +('layer'+str(args.layer))\
                +('hidden'+str(args.hidden))\
                +('oa' if args.oa else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        world_args=dict(
            num_agents=args.num_ag,
            num_adversaries=args.num_adv,
            num_landmarks=args.num_l,
            boundary=([[-1.,-1.],[1.,1.]] if args.boundary else None)
        ),
        algorithm_kwargs=dict(
            num_epochs=(args.epoch+1 if args.epoch else 1001),
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=(0 if args.load else 1000),
            max_path_length=args.mpl,
            batch_size=(args.bs if args.bs else 256),
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=(args.lr if args.lr else 1e-3),
            policy_learning_rate=(args.lr if args.lr else 1e-4),
            online_action=args.oa,
            double_q=False,
        ),
        qf_kwargs=dict(
            num_layer=args.layer,
            hidden_dim=args.hidden,
        ),
        policy_kwargs=dict(
            num_layer=args.layer,
            hidden_dim=args.hidden,
        ),
        replay_buffer_size=int(1E6),
        load_kwargs=dict(
            load=args.load,
            load_epoch=args.load_epoch,
            load_dir=None,
        ),
    )
    import os
    if args.load:
        while os.path.isdir(log_dir):
            load_dir = log_dir
            log_dir = log_dir + '_load'
        print('log: ',log_dir)
        print('load: ',load_dir)
        variant['load_kwargs']['load_dir'] = load_dir
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
                log_dir=log_dir,text_log_file=None)
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
