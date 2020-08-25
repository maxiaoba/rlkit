import numpy as np


def marollout(
        env,
        agent_n,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    num_agent = len(agent_n)
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    # env_infos = dict()
    o_n = env.reset()
    [agent.reset() for agent in agent_n]
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a_n, agent_info_n = [],{}
        for i,(o, agent) in enumerate(zip(o_n,agent_n)):
            a, agent_info = agent.get_action(o)
            a_n.append(a)
            for key in agent_info.keys():
                agent_info_n[key+' '+str(i)] = agent_info[key]
        next_o_n, r_n, d_n, env_info = env.step(a_n)
        observations.append(o_n)
        rewards.append(r_n)
        terminals.append(d_n)
        actions.append(a_n)
        agent_infos.append(agent_info_n)
        env_infos.append(env_info)
        # for key in env_info.keys():
        #     if key in env_infos.keys():
        #         env_infos[key].append(env_info[key])
        #     else:
        #         env_infos[key] = [env_info[key]]
        path_length += 1
        if d_n.all():
            break
        o_n = next_o_n
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 2:
        actions = np.expand_dims(actions, 2)
    observations = np.array(observations)
    if len(observations.shape) == 2:
        observations = np.expand_dims(observations, 2)
    next_o_n = np.array(next_o_n)
    if len(next_o_n.shape) == 1:
        next_o_n = np.expand_dims(next_o_n, 1)
    next_observations = np.vstack(
        (
            observations[1:, :, :],
            np.expand_dims(next_o_n, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, num_agent, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, num_agent, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )