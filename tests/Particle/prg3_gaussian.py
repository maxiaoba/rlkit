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
    expl_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
    eval_env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
    num_agent = expl_env.num_agent
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1_n, qf2_n, cactor_n, policy_n = [], [], [], []
    target_qf1_n, target_qf2_n, target_policy_n = [], [], []
    expl_policy_n, eval_policy_n = [], []
    log_alpha_n, log_calpha_n = [], []
    for i in range(num_agent):
        from rlkit.torch.networks import FlattenMlp
        qf1 = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            hidden_sizes=[variant['qf_kwargs']['hidden_dim']]*variant['qf_kwargs']['num_layer'],
        )
        target_qf1 = copy.deepcopy(qf1)
        qf2 = FlattenMlp(
            input_size=(obs_dim*num_agent+action_dim*num_agent),
            output_size=1,
            hidden_sizes=[variant['qf_kwargs']['hidden_dim']]*variant['qf_kwargs']['num_layer'],
        )
        target_qf2 = copy.deepcopy(qf2)
        from rlkit.torch.layers import SplitLayer
        cactor = nn.Sequential(
            FlattenMlp(input_size=obs_dim*num_agent+action_dim*(num_agent-1),
                        output_size=variant['cactor_kwargs']['hidden_dim'],
                        hidden_sizes=[variant['cactor_kwargs']['hidden_dim']]*(variant['cactor_kwargs']['num_layer']-1),
                        ),
            SplitLayer(layers=[nn.Linear(variant['cactor_kwargs']['hidden_dim'],action_dim),
                                nn.Linear(variant['cactor_kwargs']['hidden_dim'],action_dim)])
            )
        from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy
        cactor = TanhGaussianPolicy(module=cactor)

        policy = nn.Sequential(
            FlattenMlp(input_size=obs_dim,
                        output_size=variant['policy_kwargs']['hidden_dim'],
                        hidden_sizes=[variant['policy_kwargs']['hidden_dim']]*(variant['policy_kwargs']['num_layer']-1),
                        ),
            SplitLayer(layers=[nn.Linear(variant['policy_kwargs']['hidden_dim'],action_dim),
                                nn.Linear(variant['policy_kwargs']['hidden_dim'],action_dim)])
            )
        policy = TanhGaussianPolicy(module=policy)
        target_policy = copy.deepcopy(policy)
        from rlkit.torch.policies.make_deterministic import MakeDeterministic
        eval_policy = MakeDeterministic(policy)
        expl_policy = policy
        
        qf1_n.append(qf1)
        qf2_n.append(qf2)
        cactor_n.append(cactor)
        policy_n.append(policy)
        target_qf1_n.append(target_qf1)
        target_qf2_n.append(target_qf2)
        target_policy_n.append(target_policy)
        expl_policy_n.append(expl_policy)
        eval_policy_n.append(eval_policy)
        
    from rlkit.samplers.data_collector.ma_path_collector import MAMdpPathCollector
    eval_path_collector = MAMdpPathCollector(eval_env, eval_policy_n)
    expl_path_collector = MAMdpPathCollector(expl_env, expl_policy_n)

    from rlkit.data_management.ma_env_replay_buffer import MAEnvReplayBuffer
    replay_buffer = MAEnvReplayBuffer(variant['replay_buffer_size'], expl_env, num_agent=num_agent)

    from rlkit.torch.prg.prg3 import PRGTrainer
    trainer = PRGTrainer(
        env=expl_env,
        qf1_n=qf1_n,
        target_qf1_n=target_qf1_n,
        qf2_n = qf2_n,
        target_qf2_n = target_qf2_n,
        policy_n=policy_n,
        target_policy_n=target_policy_n,
        cactor_n=cactor_n,
        log_alpha_n=log_alpha_n,
        log_calpha_n=log_calpha_n,
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
    parser.add_argument('--mpl', type=int, default=25) # max path length
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='PRG3Gaussian')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--oa', action='store_true', default=False) # online action
    parser.add_argument('--ta', action='store_true', default=False) # target action
    parser.add_argument('--ona', action='store_true', default=False) # online next action
    parser.add_argument('--ce', action='store_true', default=False) # cactor entropy
    parser.add_argument('--er', action='store_true', default=False) # entropy reward
    parser.add_argument('--alpha', type=float, default=None) # init alpha
    parser.add_argument('--fa', action='store_true', default=False) # fix alpha
    parser.add_argument('--dcig', action='store_true', default=False) # deterministic cactor in graph
    parser.add_argument('--dna', action='store_true', default=False) # deterministic next action
    parser.add_argument('--pna', action='store_true', default=False) # prg next action
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
    main_dir = args.log_dir\
                +'k'+str(args.k)\
                +('hidden'+str(args.hidden))\
                +('oa' if args.oa else '')\
                +('ta' if args.ta else '')\
                +('ona' if args.ona else '')\
                +('ce' if args.ce else '')\
                +('er' if args.er else '')\
                +(('alpha'+str(args.alpha)) if args.alpha else '')\
                +('fa' if args.fa else '')\
                +('dcig' if args.dcig else '')\
                +('dna' if args.dna else '')\
                +('pna' if args.pna else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    # noinspection PyTypeChecker
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 1000),
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=args.mpl,
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
            use_entropy_loss=True,
            use_entropy_reward=args.er,
            use_cactor_entropy_loss=args.ce,
            online_action=args.oa,
            target_action=args.ta,
            online_next_action=args.ona,
            init_alpha=(args.alpha if args.alpha else 1.),
            use_automatic_entropy_tuning=(not args.fa),
            state_dependent_alpha=False,
            deterministic_cactor_in_graph=args.dcig,
            deterministic_next_action=args.dna,
            prg_next_action=args.pna,
        ),
        qf_kwargs=dict(
            num_layer=args.layer,
            hidden_dim=args.hidden,
        ),
        cactor_kwargs=dict(
            num_layer=args.layer,
            hidden_dim=args.hidden,
        ),
        policy_kwargs=dict(
            num_layer=args.layer,
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
    if args.gpu:
        ptu.set_gpu_mode(True)
    experiment(variant)
