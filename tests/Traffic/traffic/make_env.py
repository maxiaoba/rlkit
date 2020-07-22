from t_intersection import TIntersection

def make_env(env_name, **kwargs):
	if env_name == 't_intersection':
		env = TIntersection(**kwargs)
	else:
		raise NotImplementedError
	return env