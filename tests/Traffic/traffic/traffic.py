import random
import numpy as np
import scipy.spatial.distance as ssd

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from road import Road, RoadSegment
from car import Car
from driver import TwoDDriver
from oned_drivers import IDMDriver, PDDriver

RED_COLORS = [(0.85, 0.12, 0.09), (0.81, 0, 0.058), (0.59, 0.15, 0.105)]
BLUE_COLORS = [(0.27, 0.42, .7), (0.13, 0.65, 0.94)]
GREEN_COLORS = [(0, 0.69, 0.42), (0.18, 0.8, 0.44)]
ROAD_COLOR = (0.18, 0.19, 0.19)
BLACK_COLOR = (0., 0., 0.)


class TrafficEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
            road=Road([RoadSegment([(-20.,-1.5),(100,-1.5),(100,7.5),(-20,7.5)])]),
            n_cars=2,
            dt=0.1,
            car_class=Car,
            driver_class=TwoDDriver,
            x_driver_class=IDMDriver,
            y_driver_class=PDDriver,
            driver_sigma=0.0,
            car_length=5.0,
            car_width=2.0,
            car_max_accel=10.0,
            car_max_speed=40.0,
            car_expose_level=4,
            car_order=2,
            ):

        self.n_cars = n_cars
        self.dt = dt

        self.viewer = None

        self._road = road

        self._cars = [
            car_class(idx=cid, length=car_length, width=car_width, color=random.choice(BLUE_COLORS),
                      max_accel=car_max_accel, max_speed=car_max_speed,
                      expose_level=car_expose_level, order=car_order) for cid in range(self.n_cars)
        ]

        self._drivers = [
            driver_class(idx=did, car=car, 
                x_driver=x_driver_class(idx=did, car=car, sigma=driver_sigma, axis=0), 
                y_driver=y_driver_class(idx=did, car=car, sigma=driver_sigma, axis=1)) 
                for (did, car) in enumerate(self._cars)
        ]

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        [car.seed(seed) for car in self._cars]
        [driver.seed(seed) for driver in self._drivers]
        return [seed]

    @property
    def observation_space(self):
        return self._cars[0].observation_space(self._cars, self._road, include_x=False)

    @property
    def action_space(self):
        return self._cars[0].action_space()

    def reset(self):
        self._reset()
        return self.observe()

    def _reset(self):
        for driver in self._drivers:
            driver.reset()
        self._cars[0].set_position(np.array([0.0, 0.0]))
        self._cars[0].set_velocity(np.array([0.0, 0.0]))
        self._cars[1].set_position(np.array([20.0, 0.0]))
        self._cars[1].set_velocity(np.array([0.0, 0.0]))
        self._drivers[0].x_driver.set_v_des(10.0)
        self._drivers[0].y_driver.set_p_des(3.0)
        self._drivers[1].x_driver.set_v_des(0.0)
        self._drivers[1].y_driver.set_p_des(0.0)
        return None

    def step(self, action):

        self.update(action)

        obs = self.observe()

        reward = self.get_reward()

        done = self.is_terminal()

        info = self.get_info()

        return obs, reward, done, info

    def update(self, action):
        [driver.observe(self._cars, self._road) for driver in self._drivers]
        self._actions = [driver.get_action() for driver in self._drivers]
        [car.step(action, self._road, self.dt) for (car, action) in zip(self._cars, self._actions)]

    def observe(self):
        return self._cars[0].observe(self._cars, self._road, include_x=False)

    def get_reward(self):
        return 0.0

    def get_info(self):
        return {}

    def is_terminal(self):
        return False

    def setup_extra_render(self):
        pass

    def update_extra_render(self):
        pass

    def get_camera_center(self):
        return self._cars[0].position

    def render(self, mode='human', screen_size=800, rate=10):
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_size, screen_size)
            self.viewer.set_bounds(-20.0, 20.0, -20.0, 20.0)

            self.road_xforms = []
            self.road_gemos = []
            for segment in self._road.segments:
                road_poly = segment.vertices
                geom = rendering.make_polygon(road_poly)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*ROAD_COLOR)
                self.viewer.add_geom(geom)
                self.road_xforms.append(xform)
                self.road_gemos.append(geom)

            self.setup_extra_render()

            self.car_xforms = []
            self.car_geoms = []
            self.car_arr_xforms = []
            self.car_arr_geoms = []
            for car in self._cars:  # plot food first, so it is at bottom
                car_poly = [[-car._length / 2.0,
                             -car._width / 2.0], [car._length / 2.0, -car._width / 2.0],
                            [car._length / 2.0,
                             car._width / 2.0], [-car._length / 2.0, car._width / 2.0]]
                arr_poly = [[-car._length / 8.0,
                             -car._width / 4.0], [car._length / 2.0, -car._width / 4.0],
                            [car._length / 2.0,
                             car._width / 4.0], [-car._length / 8.0, car._width / 4.0]]
                geom = rendering.make_polygon(car_poly)
                arr_geom = rendering.make_polygon(arr_poly)
                xform = rendering.Transform()
                geom.set_color(*car._color)
                geom.add_attr(xform)
                self.viewer.add_geom(geom)
                self.car_xforms.append(xform)
                self.car_geoms.append(geom)

                xform = rendering.Transform()
                arr_geom.set_color(0.8, 0.8, 0.8)
                arr_geom.add_attr(xform)
                self.viewer.add_geom(arr_geom)
                self.car_arr_xforms.append(xform)
                self.car_arr_geoms.append(arr_geom)

        for sid, segment in enumerate(self._road.segments):
            self.road_xforms[sid].set_translation(*(- self.get_camera_center()))

        for cid, car in enumerate(self._cars):
            self.car_xforms[cid].set_translation(*(car.position - self.get_camera_center()))
            self.car_xforms[cid].set_rotation(np.arctan2(car.velocity[1],car.velocity[0]))
            self.car_geoms[cid].set_color(*car._color)
            self.car_arr_xforms[cid].set_translation(*(car.position - self.get_camera_center()))
            self.car_arr_xforms[cid].set_rotation(np.arctan2(car.velocity[1],car.velocity[0]))

        self.update_extra_render()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import time
    env = TrafficEnv()
    obs = env.reset()
    img = env.render()
    done = False
    while True:  #not done:
        obs, reward, done, info = env.step(None)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        env.render()
        time.sleep(0.1)
    env.close()
