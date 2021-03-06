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
	elif env_name == 't_intersection_multi0':
		from traffic.scenarios.t_intersection_multi_0 import TIntersectionMulti
		env = TIntersectionMulti(**kwargs)
	elif env_name == 't_intersection_extreme':
		from traffic.scenarios.t_intersection_extreme import TIntersectionExtreme
		env = TIntersectionExtreme(**kwargs)
	elif env_name == 't_intersection_lstm':
		from traffic.scenarios.t_intersection_lstm import TIntersectionLSTM
		env = TIntersectionLSTM(**kwargs)
	elif env_name == 't_intersection_lstm2':
		from traffic.scenarios.t_intersection_lstm_2 import TIntersectionLSTM
		env = TIntersectionLSTM(**kwargs)
	elif env_name == 't_intersection_lstm3':
		from traffic.scenarios.t_intersection_lstm_3 import TIntersectionLSTM
		env = TIntersectionLSTM(**kwargs)
	elif env_name == 't_intersection_lstm4':
		from traffic.scenarios.t_intersection_lstm_4 import TIntersectionLSTM
		env = TIntersectionLSTM(**kwargs)
	elif env_name == 'highway':
		from traffic.scenarios.highway import HighWay
		env = HighWay(**kwargs)
	elif env_name == 'highway_extreme':
		from traffic.scenarios.highway_extreme import HighWay
		env = HighWay(**kwargs)
	elif env_name == 'highway_extreme2':
		from traffic.scenarios.highway_extreme_2 import HighWay
		env = HighWay(**kwargs)
	elif env_name == 'highway_extreme3':
		from traffic.scenarios.highway_extreme_3 import HighWay
		env = HighWay(**kwargs)
	else:
		raise NotImplementedError
	return env