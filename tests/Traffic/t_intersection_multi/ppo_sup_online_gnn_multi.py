import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.networks import Mlp
from combine_net import CombineNet
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm

from log_path import get_traffic_path_information

def experiment(variant):
    from traffic.make_env import make_env
    expl_env = make_env(args.exp_name,**variant['env_kwargs'])
    eval_env = make_env(args.exp_name,**variant['env_kwargs'])
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    label_num = expl_env.label_num
    label_dim = expl_env.label_dim

    if variant['load_kwargs']['load']:
        load_dir = variant['load_kwargs']['load_dir']
        load_data = torch.load(load_dir+'/params.pkl',map_location='cpu')
        policy = load_data['trainer/policy']
        vf = load_data['trainer/value_function']
    else:
        from graph_builder_multi import MultiTrafficGraphBuilder
        gb = MultiTrafficGraphBuilder(input_dim=4, node_num=expl_env.max_veh_num+1,
                                ego_init=torch.tensor([0.,1.]),
                                other_init=torch.tensor([1.,0.]),
                                )
        if variant['gnn_kwargs']['attention']:
            from gnn_attention_net import GNNAttentionNet
            gnn_class = GNNAttentionNet
        else:
            from gnn_net import GNNNet
            gnn_class = GNNNet
        gnn = gnn_class( 
                    pre_graph_builder=gb, 
                    node_dim=variant['gnn_kwargs']['node'],
                    conv_type=variant['gnn_kwargs']['conv_type'],
                    num_conv_layers=variant['gnn_kwargs']['layer'],
                    hidden_activation=variant['gnn_kwargs']['activation'],
                    )
        encoder = gnn
        from layers import FlattenLayer, SelectLayer
        decoder = nn.Sequential(
                    SelectLayer(1,0),
                    FlattenLayer(),
                    nn.ReLU(),
                    nn.Linear(variant['gnn_kwargs']['node'],action_dim)
                )
        from layers import ReshapeLayer
        sup_learner = nn.Sequential(
                SelectLayer(1,np.arange(1,expl_env.max_veh_num+1)),
                nn.ReLU(),
                nn.Linear(variant['gnn_kwargs']['node'], label_dim),
            )
        from sup_softmax_policy import SupSoftmaxPolicy
        policy = SupSoftmaxPolicy(encoder, decoder, sup_learner)
        print('parameters: ',np.sum([p.view(-1).shape[0] for p in policy.parameters()]))
        
        vf = Mlp(
            hidden_sizes=[32, 32],
            input_size=obs_dim,
            output_size=1,
        )
        
    vf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(policy,use_preactivation=True)
    expl_policy = policy

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    from rlkit.torch.vpg.ppo_sup_online import PPOSupOnlineTrainer
    trainer = PPOSupOnlineTrainer(
        policy=policy,
        value_function=vf,
        vf_criterion=vf_criterion,
        **variant['trainer_kwargs']
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        log_path_function = get_traffic_path_information,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='t_intersection_multi')
    parser.add_argument('--nob', action='store_true', default=False)
    parser.add_argument('--obs', type=str, default='full')
    parser.add_argument('--label', type=str, default='full')
    parser.add_argument('--yld', type=float, default=0.5)
    parser.add_argument('--ds', type=float, default=0.1)
    parser.add_argument('--log_dir', type=str, default='PPOSupOnline')
    parser.add_argument('--gnn', type=str, default='GSage')
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--node', type=int, default=16)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--sw', type=float, default=None)
    parser.add_argument('--eb', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name+('nob' if args.nob else '')+'yld'+str(args.yld)+'ds'+str(args.ds)+args.obs+args.label
    main_dir = args.log_dir\
                +args.gnn\
                +('node'+str(args.node))\
                +('layer'+str(args.layer))\
                +('attention' if args.attention else '')\
                +(('act'+args.act) if args.act else '')\
                +(('sw'+str(args.sw)) if args.sw else '')\
                +(('eb'+str(args.eb)) if args.eb else '')\
                +(('ep'+str(args.epoch)) if args.epoch else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    max_path_length = 200
    # noinspection PyTypeChecker
    variant = dict(
        gnn_kwargs=dict(
            conv_type=args.gnn,
            node=args.node,
            layer=args.layer,
            attention=args.attention,
            activation=args.act,
        ),
        env_kwargs=dict(
            num_updates=1,
            normalize_obs=args.nob,
            observe_mode=args.obs,
            label_mode=args.label,
            yld=args.yld,
            driver_sigma=args.ds,
        ),
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 2000),
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=(args.bs if args.bs else 1000),
            max_path_length=max_path_length,
            save_best=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            max_path_length=max_path_length,
            policy_lr=(args.lr if args.lr else 1e-4),
            vf_lr=(args.lr if args.lr else 1e-3),
            exploration_bonus=(args.eb if args.eb else 0.),
            sup_weight=(args.sw if args.sw else 0.1),
        ),
        load_kwargs=dict(
            load=args.load,
            load_dir=log_dir,
        ),
    )
    if args.load:
        log_dir = log_dir + '_load'
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
