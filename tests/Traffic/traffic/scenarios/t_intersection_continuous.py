import random
import itertools
import numpy as np
from gym import spaces

from traffic.traffic_env import TrafficEnv
from traffic.road import Road, RoadSegment
from traffic.car import Car
from traffic.drivers.driver import Driver, XYSeperateDriver
from traffic.drivers.oned_drivers import IDMDriver, PDDriver
from traffic.actions.trajectory_accel_action import TrajectoryAccelAction
from traffic.constants import *

class YNYDriver(XYSeperateDriver):
    def __init__(self, yld=True, **kwargs):
        self.yld = yld
        super(YNYDriver, self).__init__(**kwargs)

    def set_yld(self, yld):
        self.yld = yld

    def observe(self, cars, road):
        if self.yld:
            self.x_driver.observe(cars, road)
        else:
            self.x_driver.observe(cars[1:], road)
        self.y_driver.observe(cars, road)

class EgoTrajectory:
    def xy_to_traj(self, pos):
        x, y = pos[0], pos[1]
        r = 6 # 4.5
        if y < 0.:
            s = y
            t = -x
            theta = np.pi/2
            curv = 0.
        elif x > r:
            s = r*np.pi/2. + x - r
            t = y - r
            theta = 0.
            curv = 0.
        else:
            theta = np.arctan2(r-x ,y)
            curv = 1./r
            s = r*(np.pi/2.-theta)
            t = np.sqrt((r-x)**2+(y)**2) - r

        return s, t, theta, curv

    def traj_to_xy(self, pos):
        s, t = pos[0], pos[1]
        r = 6 # 4.5
        if s < 0.:
            x = -t
            y = s
            theta = np.pi/2
            curv = 0.
        elif s > r*np.pi/2.:
            x = r + s - r*np.pi/2.
            y = r + t
            theta = 0.
            curv = 0.
        else:
            theta = np.pi/2 - s/r
            curv = 1./r
            x = r - (r+t)*np.sin(theta)
            y = (r+t)*np.cos(theta)

        return x, y, theta, curv

class EgoDriver(Driver):
    def __init__(self, 
                trajectory=None, 
                as_max=3.0,
                at_max=3.0,
                **kwargs):

        self.trajectory = trajectory
        self.as_max = as_max
        self.at_max = at_max

        self.a_s = None
        self.a_t = None
        super(EgoDriver, self).__init__(**kwargs)

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def observe(self, cars, road):
        pass

    def get_action(self):
        return TrajectoryAccelAction(self.a_s, self.a_t, self.trajectory)

class TIntersectionContinuous(TrafficEnv):
    def __init__(self,
                 yld=1.,
                 desire_speed=3.,
                 speed_cost=0.01,
                 t_cost=0.01,
                 control_cost=0.01,
                 collision_cost=10.,
                 outroad_cost=1.,
                 goal_reward=1.,
                 road=Road([RoadSegment([(-100.,0.),(100.,0.),(100.,8.),(-100.,8.)]),
                             RoadSegment([(-2,-10.),(2,-10.),(2,0.),(-2,0.)])]),
                 n_cars=3,
                 num_updates=1,
                 dt=0.1,
                 **kwargs):

        self.yld = yld
        self.num_updates = num_updates

        self.desire_speed = desire_speed
        self.speed_cost = speed_cost
        self.t_cost = t_cost
        self.control_cost = control_cost
        self.collision_cost = collision_cost
        self.outroad_cost = outroad_cost
        self.goal_reward = goal_reward
        self._collision = False
        self._outroad = False
        self._goal = False

        car_length=5.0
        car_width=2.0
        car_max_accel=10.0
        car_max_speed=40.0
        car_expose_level=4
        cars = [Car(idx=cid, length=car_length, width=car_width, color=random.choice(BLUE_COLORS),
                          max_accel=car_max_accel, max_speed=car_max_speed,
                          expose_level=car_expose_level) for cid in range(n_cars)
                ]

        driver_sigma = 0.0
        s_min = 1.5
        min_overlap = 0.5
        drivers = []
        drivers.append(EgoDriver(trajectory=EgoTrajectory(),idx=0,car=cars[0],dt=dt))
        for did in range(1,n_cars):
            driver = YNYDriver(idx=did, car=cars[did], dt=dt,
                        x_driver=IDMDriver(idx=did, car=cars[did], sigma=driver_sigma, s_min=s_min, axis=0, min_overlap=min_overlap, dt=dt), 
                        y_driver=PDDriver(idx=did, car=cars[did], sigma=driver_sigma, axis=1, dt=dt)) 
            drivers.append(driver)

        super(TIntersectionContinuous, self).__init__(
            road=road,
            cars=cars,
            drivers=drivers,
            dt=dt,
            **kwargs,)

    def update(self, action):
        action = np.clip(action, -1., 1.)
        self._drivers[0].a_s = action[0]*self._drivers[0].as_max
        self._drivers[0].a_t = action[1]*self._drivers[0].at_max

        self._goal = False
        self._collision = False
        self._outroad = False
        for _ in range(self.num_updates):
            for driver in self._drivers:
                driver.observe(self._cars, self._road)
            self._actions = [driver.get_action() for driver in self._drivers]
            [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

            ego_car = self._cars[0]
            for car in self._cars[1:]:
                if ego_car.check_collision(car):
                    self._collision = True
                    return

            if not self._road.is_in(ego_car):
                self._outroad = True
                return

            if (ego_car.position[0] > 8.) \
                and (ego_car.position[1] > 5.) \
                and (ego_car.position[1] < 7.):
                self._goal = True
                return

    def is_terminal(self):
        # print(self._collision, self._outroad, self._goal)
        return (self._collision or self._outroad or self._goal)

    def observe(self):
        return self._cars[0].observe(self._cars, self._road, include_x=True)

    @property
    def observation_space(self):
        return self._cars[0].observation_space(self._cars, self._road, include_x=True)

    @property
    def action_space(self):
        return spaces.Box(low=-np.ones(2),high=np.ones(2))

    def get_reward(self):
        reward = 0.
        action = self._actions[0]
        ego_car = self._cars[0]
        s, t, theta, curv = self._drivers[0].trajectory.xy_to_traj(ego_car.position)
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]
        v_s = v_x*np.cos(theta) + v_y*np.sin(theta)
        v_t = -v_x*np.sin(theta) + v_y*np.cos(theta)

        reward += self.speed_cost*(1.-np.abs(self.desire_speed-v_s)/self.desire_speed)

        reward += self.t_cost*(1.-np.abs(t)/2.0)

        reward -= self.control_cost*0. # TODO

        if self._collision:
            reward -= self.collision_cost

        if self._outroad:
            reward -= self.outroad_cost
            

        if self._goal:
            reward += self.goal_reward

        return reward

    def _reset(self):
        self._collision = False
        self._outroad = False
        self._goal = False
        for i,driver in enumerate(self._drivers):
            driver.reset()
        self._cars[0].set_position(np.array([0., -2.5]))
        self._cars[0].set_velocity(np.array([0., 0.]))
        self._cars[0].set_rotation(np.pi/2.)
        self._cars[1].set_position(np.array([-13., 6.]))
        self._cars[1].set_velocity(np.array([self.desire_speed, 0.]))
        self._cars[1].set_rotation(0.)
        self._cars[2].set_position(np.array([13.0, 2.]))
        self._cars[2].set_velocity(np.array([-self.desire_speed, 0.]))
        self._cars[2].set_rotation(np.pi)

        self._drivers[0].v_des = 0.
        self._drivers[0].t_des = 0.

        if np.random.rand() < self.yld:
            self._drivers[1].set_yld(True)
            self._cars[1]._color = random.choice(GREEN_COLORS)
        else:
            self._drivers[1].set_yld(False)
            self._cars[1]._color = random.choice(RED_COLORS)
        self._drivers[1].x_driver.set_v_des(self.desire_speed)
        self._drivers[1].x_driver.set_direction(1)
        self._drivers[1].y_driver.set_p_des(6.)

        if np.random.rand() < self.yld:
            self._drivers[2].set_yld(True)
            self._cars[2]._color = random.choice(GREEN_COLORS)
        else:
            self._drivers[2].set_yld(False)
            self._cars[2]._color = random.choice(RED_COLORS)
        self._drivers[2].x_driver.set_v_des(self.desire_speed)
        self._drivers[2].x_driver.set_direction(-1)
        self._drivers[2].y_driver.set_p_des(2.)
        return None


if __name__ == '__main__':
    import time
    import pdb
    env = TIntersectionContinuous(num_updates=1)
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 100
    t = 0
    cr = 0.
    # actions = np.load('/Users/xiaobaima/Dropbox/SISL/rlkit/tests/Traffic/Data/t_intersection/MyDQNcg0.1expl0.2/seed0/failure1.npy')
    while True:  #not done: 
        # pdb.set_trace()
        # action = actions[t]
        action = np.random.rand(2)*2.-1.
        # action = np.array([1.,0.])
        t += 1
        obs, reward, done, info = env.step(action)
        print('t: ', t)
        print('action: ',action)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        cr += reward
        env.render()
        time.sleep(0.1)
        if (t > maximum_step) or done:
            print('cr: ',cr)
            pdb.set_trace()
            # if env._collision or env._outroad:
            #     pdb.set_trace()
            t = 0
            cr = 0.
            env.reset()
    env.close()
