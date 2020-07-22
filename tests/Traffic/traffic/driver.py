import numpy as np
import scipy.spatial.distance as ssd

import gym
from gym import spaces
from gym.utils import seeding
# from garage.misc.overrides import overrides

class Driver:
	def __init__(self, idx, car):
		self.idx = idx
		self.car = car

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def observe(self, cars, road):
		pass

	def get_action(self):
		pass

	def reset(self):
		pass

class OneDDriver(Driver):
	def __init__(self, axis, direction=1, **kwargs):
		self.set_axis(axis)
		self.set_direction(direction)
		super(OneDDriver, self).__init__(**kwargs)

	def set_axis(self, axis):
		if axis == 0:
			self.axis0 = 0
			self.axis1 = 1
		else:
			self.axis0 = 1
			self.axis1 = 0

	def set_direction(self,direction):
		self.direction = direction

class TwoDDriver(Driver):
	def __init__(self, x_driver, y_driver, **kwargs):
		self.x_driver = x_driver
		self.y_driver = y_driver
		super(TwoDDriver, self).__init__(**kwargs)
		assert self.x_driver.car is self.car
		assert self.y_driver.car is self.car

	def observe(self, cars, road):
		self.x_driver.observe(cars, road)
		self.y_driver.observe(cars, road)

	def get_action(self):
		a_x = self.x_driver.get_action()
		a_y = self.y_driver.get_action()
		return np.array([a_x, a_y])

	def reset(self):
		self.x_driver.reset()
		self.y_driver.reset()









