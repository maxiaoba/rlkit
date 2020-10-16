import random
import itertools
import numpy as np
from gym import spaces

from traffic.traffic_env import TrafficEnv
from traffic.road import Road, RoadSegment
from traffic.car import Car
from traffic.drivers.driver import Driver, XYSeperateDriver
from traffic.drivers.oned_drivers import IDMDriver, PDDriver, PDriver
from traffic.actions.trajectory_accel_action import TrajectoryAccelAction
from traffic.constants import *

class EnvDriver(XYSeperateDriver):
    def __init__(self, 
                aggressive,
                x_sigma, y_sigma,
                car,
                **kwargs):
        self.aggressive = aggressive
        self.target_lane = None
        self.car = car
        if self.aggressive:
            v_des = np.random.uniform(0.5, 1.0)
            s_des = np.random.uniform(0.8, 1.0)
            s_min = np.random.uniform(0.4, 0.6)
            min_overlap = -self.car.width/2.
            self.min_front_x = self.car.length + s_min
            self.min_back_x = self.car.length + s_min
            self.min_advantage = self.car.length/2.
        else:
            v_des = np.random.uniform(0.0, 0.5)
            s_des = np.random.uniform(0.9, 1.1)
            s_min = np.random.uniform(0.5, 0.7)
            min_overlap = self.car.width/2.
            self.min_front_x = self.car.length + s_min
            self.min_back_x = self.car.length + s_min
            self.min_advantage = self.car.length
        x_driver =  IDMDriver(sigma=x_sigma, v_des=v_des, s_des=s_des, s_min=s_min, axis=0, min_overlap=min_overlap, car=car, **kwargs)
        y_driver =  PDDriver(sigma=y_sigma, p_des=0., a_max=1.0, axis=1, k_p=2.0, k_d=5.0, car=car, **kwargs)
        self.on_target = True
        super(EnvDriver, self).__init__(x_driver,y_driver,car=car,**kwargs)

    def observe(self, cars, road):
        x, y = self.car.position
        min_front_distance0 = np.inf
        min_back_distance0 = np.inf
        min_front_distance1 = np.inf
        min_back_distance1 = np.inf
        for car in cars:
            if car is self.car:
                continue
            if car.position[1] <= 4.0:
                if (car.position[0] > x) and (car.position[0]-x < min_front_distance0):
                    min_front_distance0 = car.position[0] - x
                elif (car.position[0] < x) and (x-car.position[0] < min_back_distance0):
                    min_back_distance0 = x - car.position[0]
            elif car.position[1] > 4.0:
                if (car.position[0] > x) and (car.position[0]-x < min_front_distance1):
                    min_front_distance1 = car.position[0] - x
                elif (car.position[0] < x) and (x-car.position[0] < min_back_distance1):
                    min_back_distance1 = x - car.position[0]

        if y <= 4.0:
            if (min_front_distance1 - min_front_distance0 > self.min_advantage) \
                and (min_front_distance1 > self.min_front_x) \
                and (min_back_distance1 > self.min_back_x):
                self.y_driver.p_des = 6.0
            else:
                self.y_driver.p_des = 2.0
        else:
            if (min_front_distance0 - min_front_distance1 > self.min_advantage) \
                and (min_front_distance0 > self.min_front_x) \
                and (min_back_distance0 > self.min_back_x):
                self.y_driver.p_des = 2.0
            else:
                self.y_driver.p_des = 6.0

        self.x_driver.observe(cars, road)
        self.y_driver.observe(cars, road)

    def setup_render(self, viewer):
        if not self.aggressive:
            self.car._color = [*GREEN_COLORS[0],0.5]
        else:
            self.car._color = [*RED_COLORS[0],0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

    def update_render(self, camera_center):
        if not self.aggressive:
            self.car._color = [*GREEN_COLORS[0],0.5]
        else:
            self.car._color = [*RED_COLORS[0],0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

class EgoDriver(XYSeperateDriver):
    def __init__(self, 
                x_sigma, y_sigma,
                **kwargs):

        x_driver = IDMDriver(sigma=x_sigma, v_des=0.0, s_des=0.7, s_min=0.5, axis=0, min_overlap=0., **kwargs)
        y_driver =  PDDriver(sigma=y_sigma, p_des=0.0, a_max=1.0, axis=1, **kwargs)
        super(EgoDriver, self).__init__(x_driver,y_driver,**kwargs)

    def apply_action(self, action):
        self.x_driver.v_des = action[0]
        if action[1] == 0:
            self.y_driver.p_des = 2.0
        else:
            self.y_driver.p_des = 6.0

class HighWay(TrafficEnv):
    def __init__(self,
                 obs_noise=0.,
                 x_actions=[0.,0.5,3.],
                 y_actions=[0,1],
                 driver_sigma = 0.,
                 control_cost=0.01,
                 collision_cost=2.,
                 survive_reward=0.01,
                 goal_reward=2.,
                 road=Road([RoadSegment([(-100.,0.),(100.,0.),(100.,8.),(-100.,8.)])]),
                 left_bound = -30.,
                 right_bound = 30.,
                 gap_min = 8.,
                 gap_max = 12.,
                 max_veh_num = 12,
                 num_updates=1,
                 dt=0.1,
                 **kwargs):

        self.obs_noise = obs_noise
        self.x_actions = x_actions
        self.y_actions = y_actions
        # we use target value instead of target change so system is Markovian
        self.rl_actions = list(itertools.product(x_actions,y_actions))
        self.num_updates = num_updates

        self.control_cost = control_cost
        self.collision_cost = collision_cost
        self.survive_reward = survive_reward
        self.goal_reward = goal_reward

        self.left_bound = left_bound
        self.right_bound = right_bound
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.max_veh_num = max_veh_num
        self.label_dim = 2
        self.label_num = self.max_veh_num

        self._collision = False
        self._goal = False
        self._intentions = []
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num/2.)+1

        self.car_length = 5.0
        self.car_width = 2.0
        self.car_max_accel = 5.0
        self.car_max_speed = 5.0
        self.car_max_rotation = 0. #np.pi/18.
        self.car_expose_level = 4
        self.driver_sigma = driver_sigma

        super(HighWay, self).__init__(
            road=road,
            cars=[],
            drivers=[],
            dt=dt,
            **kwargs,)

    def get_sup_labels(self):
        for driver in self._drivers:
            driver.observe(self._cars, self._road)
        labels = np.array([np.nan]*self.label_num)
        for driver in self._drivers[1:]:
            i = driver._idx - 1
            labels[i] = int(driver.aggressive)
        return labels

    def update(self, action):
        # recorder intentios at the begining
        self._sup_labels = self.get_sup_labels()

        rl_action = self.rl_actions[action]
        self._drivers[0].apply_action(rl_action)

        self._goal = False
        self._collision = False
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

            if ego_car.position[0] > self.right_bound-2.:
                self._goal = True
                return

            # add cars when there is enough space
            min_upper_x = np.inf
            min_lower_x = np.inf
            for car in self._cars:
                if (car.position[1] <= 4.) and (car.position[0] < min_lower_x):
                    min_lower_x = car.position[0]
                if (car.position[1] > 4.) and (car.position[0] < min_upper_x):
                    min_upper_x = car.position[0]
            if min_lower_x > (self.left_bound + np.random.uniform(self.gap_min,self.gap_max) + self.car_length):
                x, y = self.left_bound, 2.
                aggressive = np.random.choice([True,False])
                car, driver = self.add_car(x, y, 0., 0., aggressive, 0.)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)
            if min_upper_x > (self.left_bound + np.random.uniform(self.gap_min,self.gap_max) + self.car_length):
                x, y = self.left_bound, 6.
                aggressive = np.random.choice([True,False])
                car, driver = self.add_car(x, y, 0., 0., aggressive, 0.)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)

            # remove cars that are out-of bound
            for car, driver in zip(self._cars[1:],self._drivers[1:]):
                if car.position[0] > self.right_bound:
                    self.remove_car(car, driver)

    def is_terminal(self):
        return (self._collision or self._goal)

    def get_info(self):
        info = {}
        info['sup_labels'] = np.copy(self._sup_labels)

        if self._collision:
            info['event']='collision'
        elif self._goal:
            info['event']='goal'
        else:
            info['event']='nothing'

        return info

    def observe(self):
        # TODO: normalization
        obs = np.zeros(int(4*self.max_veh_num+4))
        for car in self._cars:
            i = int(car._idx*4)
            obs[i:i+2] = car.position + np.random.uniform(-1.,1.,2)*self.obs_noise
            obs[i+2:i+4] = car.velocity + np.random.uniform(-1.,1.,2)*self.obs_noise

        obs = np.copy(obs)
        return obs

    @property
    def observation_space(self):
        low = -np.ones(int(4*self.max_veh_num+4))
        high = np.ones(int(4*self.max_veh_num+4))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.rl_actions))

    def get_reward(self):
        reward = 0.
        action = self._actions[0]
        ego_car = self._cars[0]
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]

        control_cost = 0. # TODO
        reward += self.control_cost*control_cost

        if self._collision:
            reward -= self.collision_cost
        elif self._goal:
            reward += self.goal_reward
        else:
            reward += self.survive_reward
        # print(speed_cost, t_cost, control_cost, reward)
        return reward

    def remove_car(self, car, driver):
        self._cars.remove(car)
        self._drivers.remove(driver)
        if hasattr(self, 'viewer') and self.viewer:
            car.remove_render(self.viewer)
            driver.remove_render(self.viewer)

    def add_car(self, x, y, vx, vy, aggressive, theta):
        if y <= 4.:
            idx = self._lower_lane_next_idx
            self._lower_lane_next_idx += 1
            if self._lower_lane_next_idx > int(self.max_veh_num/2.):
                self._lower_lane_next_idx = 1
        elif y > 4.:
            idx = self._upper_lane_next_idx
            self._upper_lane_next_idx += 1
            if self._upper_lane_next_idx > self.max_veh_num:
                self._upper_lane_next_idx = int(self.max_veh_num/2.)+1
        car = Car(idx=idx, length=self.car_length, width=self.car_width, color=random.choice(RED_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EnvDriver(aggressive=aggressive, 
                            x_sigma=self.driver_sigma, y_sigma=0.,
                            idx=idx, car=car, dt=self.dt
                            ) 
        car.set_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)

        self._cars.append(car)
        self._drivers.append(driver)
        return car, driver

    def _reset(self):
        self._collision = False
        self._goal = False
        self._intentions = []
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num/2.)+1

        self._cars, self._drivers = [], []
        x_0 = self.left_bound
        y_0 = np.random.choice([2.,6.])
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(x_sigma=self.driver_sigma, y_sigma=0.,
                            idx=0,car=car,dt=self.dt)
        car.set_position(np.array([x_0, y_0]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(0.)
        self._cars.append(car)
        self._drivers.append(driver)
        # randomly generate surrounding cars and drivers
        # lower lane 
        x = self.right_bound - np.random.rand()*(self.gap_max-self.gap_min)
        if y_0 == 2.0:
            x_min = x_0 + self.car_length + self.gap_min
        else:
            x_min = self.left_bound
        y = 2.0
        while (x >= x_min):
            aggressive = np.random.choice([True,False])
            self.add_car(x, y, 0., 0., aggressive, 0.)
            x -= (np.random.uniform(self.gap_min,self.gap_max) + self.car_length)

        # upper lane
        x = self.right_bound - np.random.rand()*(self.gap_max-self.gap_min)
        if y_0 == 6.0:
            x_min = x_0 + self.car_length + self.gap_min
        else:
            x_min = self.left_bound
        y = 6.0
        while (x >= x_min):
            aggressive = np.random.choice([True,False])
            self.add_car(x, y, 0., 0., aggressive, 0.)
            x -= (np.random.uniform(self.gap_min,self.gap_max) + self.car_length)

        self._sup_labels = self.get_sup_labels()
        return None

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-40.0, 40.0, -20.0, 20.0)

    def get_camera_center(self):
        return np.array([0.,4.0])

    def update_extra_render(self, extra_input):
        start = np.array([-100.,4.0]) - self.get_camera_center()
        end = np.array([100.,4.0]) - self.get_camera_center()
        attrs = {"color":(1.,1.,1.),"linewidth":4.}
        self.viewer.draw_line(start, end, **attrs)

        if extra_input:
            if ('attention_weight' in extra_input.keys()) and (extra_input['attention_weight'] is not None):
                edge_index = extra_input['attention_weight'][0]
                attention_weight = extra_input['attention_weight'][1]
                upper_indices, lower_indices = self.get_sorted_indices()
                car_indices = [np.nan]*(1+self.max_veh_num)
                car_indices[0] = 0
                car_indices[1:len(lower_indices)+1] = lower_indices[:]
                car_indices[int(self.max_veh_num/2)+1:int(self.max_veh_num/2)+1+len(upper_indices)] = upper_indices[:]
                starts, ends, attentions = [], [], []
                for i in range(edge_index.shape[1]):
                    if np.isnan(car_indices[edge_index[0,i]]) or np.isnan(car_indices[edge_index[1,i]]):
                        pass
                    elif car_indices[edge_index[1,i]] == 0:
                        attention = attention_weight[i].item()
                        attentions.append(attention)
                        car_i = car_indices[edge_index[0,i]]
                        car_j = car_indices[edge_index[1,i]]
                        start = self._cars[car_i].position - self.get_camera_center()
                        end = self._cars[car_j].position - self.get_camera_center()
                        starts.append(start)
                        ends.append(end)
                rank_index = np.argsort(attentions)
                starts = np.array(starts)[rank_index]
                ends = np.array(ends)[rank_index]
                attentions = np.array(attentions)[rank_index]
                assert np.isclose(np.sum(attentions),1.)
                for start, end, attention in zip(starts[-3:],ends[-3:],attentions[-3:]):
                    attrs = {"color":(1.,0.,1.),"linewidth":10.*attention}
                    if (start == end).all():
                        from traffic.rendering import make_circle, _add_attrs
                        circle = make_circle(radius=1., res=15, filled=False, center=start)
                        _add_attrs(circle, attrs)
                        self.viewer.add_onetime(circle)
                    else:
                        self.viewer.draw_line(start, end, **attrs)
            if ('intentions' in extra_input.keys()) and (extra_input['intentions'] is not None):
                for car in self._cars[1:]:
                    from traffic.rendering import make_circle, _add_attrs
                    intention = extra_input['intentions'][car._idx-1]
                    start = car.position - self.get_camera_center()
                    attrs = {"color":(intention[0],intention[1],0.)}
                    circle = make_circle(radius=0.5, res=15, filled=True, center=start)
                    _add_attrs(circle, attrs)
                    self.viewer.add_onetime(circle) 

if __name__ == '__main__':
    import time
    import pdb
    env = HighWay(num_updates=1, driver_sigma=0.1, 
                    obs_noise=0.1,
                    )
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 200
    t = 0
    cr = 0.
    actions = [4]*(2*maximum_step)
    # actions = np.load('/Users/xiaobaima/Dropbox/SISL/rlkit/tests/Traffic/Data/t_intersection/MyDQNcg0.1expl0.2/seed0/failure1.npy')
    while True:  #not done: 
        # pdb.set_trace()
        # action = actions[t][0]
        action = actions[t]
        # action = np.random.randint(env.action_space.n)
        # action = input("Action\n")
        # action = int(action)
        # while action < 0:
        #     t = 0
        #     cr = 0.
        #     env.reset()
        #     env.render()
        #     action = input("Action\n")
        #     action = int(action)
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
            env.render()
    env.close()
