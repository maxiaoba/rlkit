"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np
import os.path as osp

import rlkit.pythonplusplus as ppp
from rlkit.core.eval_util import create_stats_ordered_dict, get_average_returns
from rlkit.core import logger

def get_traffic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

    num_collision, num_block, num_outroad, num_success, num_timeout = 0, 0, 0, 0, 0
    log_path = logger.get_snapshot_dir()
    for pid,path in enumerate(paths):
        event = path["env_infos"][-1]['event']
        if event == 'collision':
            num_collision += 1
        elif event == 'block':
            num_block +=1 1
        elif event == 'outroad':
            num_outroad += 1
        elif event == 'goal':
            num_success += 1
        else:
            num_timeout += 1
    statistics['Num Collision'] = num_collision
    statistics['Num Block'] = num_block
    statistics['Num Outroad'] = num_outroad
    statistics['Num Success'] = num_success
    statistics['Num Timeout'] = num_timeout

    for info_key in ['agent_infos']:
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


