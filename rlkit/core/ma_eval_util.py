"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np

import rlkit.pythonplusplus as ppp
from rlkit.core.eval_util import create_stats_ordered_dict

def get_generic_ma_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    num_agent = paths[0]["rewards"].shape[1]
    statistics = OrderedDict()
    for agent in range(num_agent):
        returns = [sum(path["rewards"][:,agent,:]) for path in paths]

        rewards = np.vstack([path["rewards"][:,agent,:] for path in paths])
        statistics.update(create_stats_ordered_dict('Rewards {}'.format(agent), rewards,
                                                    stat_prefix=stat_prefix))
        statistics.update(create_stats_ordered_dict('Returns {}'.format(agent), returns,
                                                    stat_prefix=stat_prefix))
        actions = [path["actions"][:,agent,:] for path in paths]
        actions = np.vstack(actions)
        statistics.update(create_stats_ordered_dict(
            'Actions {}'.format(agent), actions, stat_prefix=stat_prefix
        ))
        statistics[stat_prefix + 'Average Returns {}'.format(agent)] = get_ma_average_returns(paths, agent)

    statistics['Num Paths'] = len(paths)

    for info_key in ['env_infos', 'agent_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))

    return statistics


def get_ma_average_returns(paths, agent):
    returns = [sum(path["rewards"][:,agent,:]) for path in paths]
    return np.mean(returns)
