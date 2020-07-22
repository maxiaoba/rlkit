import random
import numpy as np

from road import Road, RoadSegment
from traffic import TrafficEnv, GREEN_COLORS, RED_COLORS

from driver import TwoDDriver
from oned_drivers import IDMDriver, PDDriver

class YNYDriver(TwoDDriver):
    def __init__(self, yld=True, **kwargs):
        self.yld = yld
        super(YNYDriver, self).__init__(**kwargs)

    def set_yld(self, yld):
        self.yld = yld

    def observe(self, cars, road):
        if self.yld:
            self.x_driver.observe(cars, road)
        else:
            self.x_driver.observe([], road)
        self.y_driver.observe(cars, road)

class TIntersection(TrafficEnv):
    def __init__(self,
                 action_scale=3.,
                 desire_speed=3.,
                 speed_cost=0.01,
                 y_cost=0.01,
                 control_cost=0.01,
                 collision_cost=1.,
                 outroad_cost=1.,
                 goal_reward=1.,
                 road=Road([RoadSegment([(-10.,0.),(10.,0.),(10.,6.),(-10.,6.)]),
                             RoadSegment([(-1.5,-10.),(1.5,-10.),(1.5,0.),(-1.5,0.)])]),
                 n_cars=3,
                 driver_class=YNYDriver,
                 **kwargs):
        super(TIntersection, self).__init__(
            road=road,
            n_cars=n_cars,
            driver_class=driver_class,
            **kwargs,)
        self.action_scale = action_scale
        self.desire_speed = desire_speed
        self.speed_cost = speed_cost
        self.y_cost = y_cost
        self.control_cost = control_cost
        self.collision_cost = collision_cost
        self.outroad_cost = outroad_cost
        self.goal_reward = goal_reward
        self._collision = False
        self._outroad = False
        self._goal = False

    def update(self, action):
        action = action[0]
        [driver.observe(self._cars, self._road) for driver in self._drivers[1:]]
        self._actions = [driver.get_action() for driver in self._drivers[1:]]
        self._actions = [action*self.action_scale, *self._actions]
        [car.step(action, self._road, self.dt) for (car, action) in zip(self._cars, self._actions)]

        ego_car = self._cars[0]
        self._collision = False
        for car in self._cars[1:]:
            if ego_car.check_collision(car):
                self._collision = True
                break

        self._outroad = False
        if self._road.check_outroad(ego_car):
            self._outroad = True

        self._goal = False
        if (ego_car.position[0] > 8.) \
            and (ego_car.position[1] > 4.) \
            and (ego_car.position[1] < 6.):
            self._goal = True

    def is_terminal(self):
        return np.array([(self._collision or self._outroad or self._goal)])

    def observe(self):
        return np.array([self._cars[0].observe(self._cars, self._road, include_x=True)])

    @property
    def observation_space(self):
        return self._cars[0].observation_space(self._cars, self._road, include_x=True)

    def get_reward(self):
        reward = 0.
        action = self._actions[0]/self.action_scale
        ego_car = self._cars[0]

        vx = ego_car.velocity[0]
        # print('speed_cost: ',self.speed_cost*(np.abs(self.desire_speed-vx)/self.desire_speed) )
        reward -= self.speed_cost*(np.abs(self.desire_speed-vx)/self.desire_speed)

        y = ego_car.position[1]
        # print('y_cost: ',self.y_cost*(np.abs(4.5-y)/4.5))
        reward -= self.y_cost*(np.abs(4.5-y)/4.5)

        # print('control_cost: ',self.control_cost*np.sqrt(action[0]**2+action[1]**2))
        reward -= self.control_cost*np.sqrt(action[0]**2+action[1]**2)

        if self._collision:
            # print('collision_cost: ',self.collision_cost)
            reward -= self.collision_cost

        if self._outroad:
            # print('outroad_cost: ',self.outroad_cost)
            reward -= self.outroad_cost

        if self._goal:
            # print('goal_reward: ',self.goal_reward)
            reward += self.goal_reward

        return np.array([reward])

    def _reset(self):
        self._collision = False
        self._outroad = False
        self._goal = False
        for i,driver in enumerate(self._drivers):
            driver.reset()
        self._cars[0].set_position(np.array([0., -2.5]))
        self._cars[0].set_velocity(np.array([0., 0.]))
        self._cars[1].set_position(np.array([-10., 4.5]))
        self._cars[1].set_velocity(np.array([self.desire_speed, 0.]))
        self._cars[2].set_position(np.array([10.0, 1.5]))
        self._cars[2].set_velocity(np.array([-self.desire_speed, 0.]))
        self._drivers[0].x_driver.set_v_des(3.0)
        self._drivers[0].y_driver.set_p_des(4.5)

        if np.random.rand() < 1.:
            self._drivers[1].set_yld(True)
            self._cars[1]._color = random.choice(GREEN_COLORS)
        else:
            self._drivers[1].set_yld(False)
            self._cars[1]._color = random.choice(RED_COLORS)
        self._drivers[1].x_driver.set_v_des(self.desire_speed)
        self._drivers[1].x_driver.set_direction(1)
        self._drivers[1].y_driver.set_p_des(4.5)

        if np.random.rand() < 1.:
            self._drivers[2].set_yld(True)
            self._cars[2]._color = random.choice(GREEN_COLORS)
        else:
            self._drivers[2].set_yld(False)
            self._cars[2]._color = random.choice(RED_COLORS)
        self._drivers[2].x_driver.set_v_des(self.desire_speed)
        self._drivers[2].x_driver.set_direction(-1)
        self._drivers[2].y_driver.set_p_des(1.5)
        return None


if __name__ == '__main__':
    import time
    import pdb
    env = TIntersection()
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 50
    t = 0
    actions = [*[[0.,.8]]*12,
                *[[.8,0.]]*15,
                *[[0.5,-.8]]*12,
        ]
    while True:  #not done: 
        action = [np.array(actions[t])]
        t += 1
        obs, reward, done, info = env.step(action)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        env.render()
        time.sleep(0.1)
        if (t > maximum_step) or done.all():
            pdb.set_trace()
            t = 0
            env.reset()
    env.close()
