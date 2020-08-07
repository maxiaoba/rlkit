from traffic.scenarios.t_intersection import TIntersection
from traffic.scenarios.t_intersection_continuous import TIntersectionContinuous

def make_env(env_name, **kwargs):
	if env_name == 't_intersection':
		env = TIntersection(**kwargs)
	elif env_name == 't_intersection_cont':
		env = TIntersectionContinuous(**kwargs)
	else:
		raise NotImplementedError
	return env