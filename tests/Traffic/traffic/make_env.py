def make_env(env_name, **kwargs):
	if env_name == 't_intersection':
		from traffic.scenarios.t_intersection import TIntersection
		env = TIntersection(**kwargs)
	elif env_name == 't_intersection_cont':
		from traffic.scenarios.t_intersection_continuous import TIntersectionContinuous
		env = TIntersectionContinuous(**kwargs)
	elif env_name == 't_intersection_multi':
		from traffic.scenarios.t_intersection_multi import TIntersectionMulti
		env = TIntersectionMulti(**kwargs)
	else:
		raise NotImplementedError
	return env