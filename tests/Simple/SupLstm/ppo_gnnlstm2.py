import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.vpg.ppo import PPOTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm

def experiment(variant):
    from simple_sup_lstm import SimpleSupLSTMEnv
    expl_env = SimpleSupLSTMEnv(**variant['env_kwargs'])
    eval_env = SimpleSupLSTMEnv(**variant['env_kwargs'])
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
        hidden_dim = variant['lstm_kwargs']['hidden_dim']
        num_lstm_layers = variant['lstm_kwargs']['num_layers']
        node_dim = variant['gnn_kwargs']['node_dim']

        node_num = expl_env.node_num
        input_node_dim = int(obs_dim/node_num)
        a_0 = np.zeros(action_dim)
        h1_0 = np.zeros((node_num, hidden_dim*num_lstm_layers))
        c1_0 = np.zeros((node_num, hidden_dim*num_lstm_layers))
        h2_0 = np.zeros((node_num, hidden_dim*num_lstm_layers))
        c2_0 = np.zeros((node_num, hidden_dim*num_lstm_layers))
        latent_0 = (h1_0, c1_0, h2_0, c2_0)
        from lstm_net import LSTMNet
        lstm1_ego = LSTMNet(input_node_dim, action_dim, hidden_dim, num_lstm_layers)
        lstm1_other = LSTMNet(input_node_dim, 0, hidden_dim, num_lstm_layers)
        lstm2_ego = LSTMNet(node_dim, 0, hidden_dim, num_lstm_layers)
        lstm2_other = LSTMNet(node_dim, 0, hidden_dim, num_lstm_layers)
        from graph_builder import TrafficGraphBuilder
        gb = TrafficGraphBuilder(input_dim=hidden_dim, node_num=node_num,
                                ego_init=torch.tensor([0.,1.]),
                                other_init=torch.tensor([1.,0.]),
                                )
        from gnn_net import GNNNet
        gnn = GNNNet( 
                    pre_graph_builder=gb, 
                    node_dim=variant['gnn_kwargs']['node_dim'],
                    conv_type=variant['gnn_kwargs']['conv_type'],
                    num_conv_layers=variant['gnn_kwargs']['num_layers'],
                    hidden_activation=variant['gnn_kwargs']['activation'],
                    )
        from gnn_lstm2_net import GNNLSTM2Net
        policy_net = GNNLSTM2Net(node_num,gnn,
                                lstm1_ego,lstm1_other,
                                lstm2_ego,lstm2_other)
        from layers import FlattenLayer, SelectLayer
        post_net = nn.Sequential(
                    SelectLayer(-2,0),
                    FlattenLayer(2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim,action_dim)
                )
        from softmax_lstm_policy import SoftmaxLSTMPolicy
        policy = SoftmaxLSTMPolicy(
                    a_0=a_0,
                    latent_0=latent_0,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    lstm_net=policy_net,
                    post_net=post_net,
                    )
        print('parameters: ',np.sum([p.view(-1).shape[0] for p in policy.parameters()]))

        vf = Mlp(
            hidden_sizes=[32, 32],
            input_size=obs_dim,
            output_size=1,
        ) # TODO: id is also an input
        
    vf_criterion = nn.MSELoss()
    from rlkit.torch.policies.make_deterministic import MakeDeterministic
    eval_policy = MakeDeterministic(policy)
    expl_policy = policy

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = PPOTrainer(
        policy=policy,
        value_function=vf,
        vf_criterion=vf_criterion,
        recurrent=True,
        **variant['trainer_kwargs']
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='SimpleSupLSTM')
    parser.add_argument('--node_num', type=int, default=5)
    parser.add_argument('--node_dim', type=int, default=2)
    parser.add_argument('--log_dir', type=str, default='PPOGNN2')
    parser.add_argument('--llayer', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--gnn', type=str, default='GSage')
    parser.add_argument('--node', type=int, default=16)
    parser.add_argument('--glayer', type=int, default=3)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="gap_and_last")
    parser.add_argument('--snapshot_gap', type=int, default=500)
    args = parser.parse_args()
    import os.path as osp
    pre_dir = './Data/'+args.exp_name+'node'+str(args.node_num)+'dim'+str(args.node_dim)
    main_dir = args.log_dir\
                +('llayer'+str(args.llayer))\
                +('hidden'+str(args.hidden))\
                +args.gnn\
                +('node'+str(args.node))\
                +('glayer'+str(args.glayer))\
                +('act'+args.act)\
                +(('ep'+str(args.epoch)) if args.epoch else '')\
                +(('lr'+str(args.lr)) if args.lr else '')\
                +(('bs'+str(args.bs)) if args.bs else '')
    log_dir = osp.join(pre_dir,main_dir,'seed'+str(args.seed))
    max_path_length = 10
    # noinspection PyTypeChecker
    variant = dict(
        lstm_kwargs=dict(
            hidden_dim=args.hidden,
            num_layers=args.llayer,
        ),
        gnn_kwargs=dict(
            conv_type=args.gnn,
            node_dim=args.node,
            num_layers=args.glayer,
            activation=args.act,
        ),
        env_kwargs=dict(
            node_num=args.node_num,
            node_dim=args.node_dim
        ),
        algorithm_kwargs=dict(
            num_epochs=(args.epoch if args.epoch else 1000),
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
