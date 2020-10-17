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

def which_lane(car):
    if car.position[1] <= 4.:
        return 0
    elif car.position[1] <= 8.:
        return 1
    else:
        return 2

class EnvDriver(XYSeperateDriver):
    def __init__(self, 
                aggressive,
                min_x,
                min_y,
                x_sigma, y_sigma,
                **kwargs):
        self.aggressive = aggressive
        self.next_target_lane = None
        self.min_x = min_x
        self.min_y = min_y
        x_driver =  PDDriver(sigma=x_sigma, p_des=0., a_max=1.0, axis=0, k_p=2.0, k_d=5.0, **kwargs)
        y_driver =  PDDriver(sigma=y_sigma,p_des=0., a_max=1.0, axis=1, k_p=2.0, k_d=5.0, **kwargs)
        super(EnvDriver, self).__init__(x_driver,y_driver,**kwargs)

    def observe(self, cars, road):
        x, y = self.car.position
        lane_front_distances = [np.inf]*3
        lane_front_ind = [None]*3
        lane_back_distances = [np.inf]*3
        lane_back_ind = [None]*3
        min_front_distance = np.inf
        min_front_ind = None
        min_back_distance = np.inf
        min_back_ind = None
        min_up_distance = (12.0 + self.min_y) - y
        min_up_ind = None
        min_low_distance = y - (0. - self.min_y)
        min_low_ind = None
        for car in cars:
            if car is self.car:
                continue
            else:
                lane_id = which_lane(car)
                if (car.position[0] > x) and (car.position[0]-x < lane_front_distances[lane_id]):
                    lane_front_distances[lane_id] = car.position[0] - x
                elif (car.position[0] < x) and (x-car.position[0] < lane_back_distances[lane_id]):
                    lane_back_distances[lane_id] = x - car.position[0]
                if (car.position[1] > y) \
                    and (car.position[1] - y < min_up_distance) \
                    and (self.car.get_distance(car,0) < 0.):
                    min_up_distance = car.position[1] - y
                elif (car.position[1] < y) \
                    and (y - car.position[1] < min_low_distance) \
                    and (self.car.get_distance(car,0) < 0.):
                    min_low_distance = y - car.position[1]
                if (car.position[0] > x) \
                    and (car.position[0] - x < min_front_distance) \
                    and (self.car.get_distance(car,1) < 0.):
                    min_front_distance = car.position[0] - x
                elif (car.position[0] < x) \
                    and (x - car.position[0] < min_back_distance) \
                    and (self.car.get_distance(car,1) < 0.):
                    min_back_distance = x - car.position[0]

        current_lane = which_lane(self.car)
        if self.target_lane > current_lane:
            self.next_target_lane = current_lane + 1
        elif self.target_lane < current_lane:
            self.next_target_lane = current_lane - 1
        else:
            self.next_target_lane = current_lane

        if self.next_target_lane == current_lane:
            self.y_driver.p_des = current_lane*4.0 + 2.0
            self.x_driver.p_des = x + 1.0
        else: # need to merge
            if (lane_front_distances[self.next_target_lane] > self.min_x) \
             and (lane_back_distances[self.next_target_lane] > self.min_x):
                self.y_driver.p_des = self.next_target_lane*4.0 + 2.0
                self.x_driver.p_des = x
            else:
                self.y_driver.p_des = current_lane*4.0 + 2.0
                self.x_driver.p_des = x + (lane_front_distances[self.next_target_lane]-lane_back_distances[self.next_target_lane])/2.

        # safety
        # if min_front_distance < min_back_distance:
        #     self.x_driver.p_des = np.minimum(self.x_driver.p_des, x+min_front_distance-self.min_x)
        # else:
        #     self.x_driver.p_des = np.maximum(self.x_driver.p_des, x-min_back_distance+self.min_x)
        self.x_driver.p_des = np.minimum(self.x_driver.p_des, x+min_front_distance-self.min_x)
        if min_up_distance < min_low_distance:
            self.y_driver.p_des = np.minimum(self.y_driver.p_des, y+min_up_distance-self.min_y)
        else:
            self.y_driver.p_des = np.maximum(self.y_driver.p_des, y-min_low_distance+self.min_y)
        self.x_driver.observe(cars, road)
        self.y_driver.observe(cars, road)

    def setup_render(self, viewer):
        if self.next_target_lane == 0:
            self.car._color = [1,0,0,0.5]
        elif self.next_target_lane == 1:
            self.car._color = [0,1,0,0.5]
        elif self.next_target_lane == 2:
            self.car._color = [0,0,1,0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

    def update_render(self, camera_center):
        if self.next_target_lane == 0:
            self.car._color = [1,0,0,0.5]
        elif self.next_target_lane == 1:
            self.car._color = [0,1,0,0.5]
        elif self.next_target_lane == 2:
            self.car._color = [0,0,1,0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

class EgoDriver(XYSeperateDriver):
    def __init__(self, 
                min_x,
                min_y,
                x_sigma, y_sigma,
                **kwargs):

        self.min_x = min_x
        self.min_y = min_y
        x_driver = PDDriver(sigma=x_sigma, p_des=0., a_max=1.0, axis=0, k_p=2.0, k_d=5.0, **kwargs)
        y_driver =  PDDriver(sigma=y_sigma,p_des=0., a_max=1.0, axis=1, k_p=2.0, k_d=5.0, **kwargs)
        super(EgoDriver, self).__init__(x_driver,y_driver,**kwargs)

    def apply_action(self, action):
        self.x_driver.p_des = self.car.position[0] + action[0]
        self.y_driver.p_des = action[1]*4.0 + 2.0

    def observe(self, cars, road):
        x, y = self.car.position
        min_front_distance = np.inf
        min_back_distance = np.inf
        min_up_distance = (12.0 + self.min_y) - y
        min_low_distance = y - (0. - self.min_y)
        for car in cars:
            if car is self.car:
                continue
            else:
                if (car.position[1] > y) \
                    and (car.position[1] - y < min_up_distance) \
                    and (self.car.get_distance(car,0) < 0.):
                    min_up_distance = car.position[1] - y
                elif (car.position[1] < y) \
                    and (y - car.position[1] < min_low_distance) \
                    and (self.car.get_distance(car,0) < 0.):
                    min_low_distance = y - car.position[1]
                if (car.position[0] > x) \
                    and (car.position[0] - x < min_front_distance) \
                    and (self.car.get_distance(car,1) < 0.):
                    min_front_distance = car.position[0] - x
                elif (car.position[0] < x) \
                    and (x - car.position[0] < min_back_distance) \
                    and (self.car.get_distance(car,1) < 0.):
                    min_back_distance = x - car.position[0]

        # safety
        # if min_front_distance < min_back_distance:
        #     self.x_driver.p_des = np.minimum(self.x_driver.p_des, x+min_front_distance-self.min_x)
        # else:
        #     self.x_driver.p_des = np.maximum(self.x_driver.p_des, x-min_back_distance+self.min_x)
        self.x_driver.p_des = np.minimum(self.x_driver.p_des, x+min_front_distance-self.min_x)
        if min_up_distance < min_low_distance:
            self.y_driver.p_des = np.minimum(self.y_driver.p_des, y+min_up_distance-self.min_y)
        else:
            self.y_driver.p_des = np.maximum(self.y_driver.p_des, y-min_low_distance+self.min_y)
        self.x_driver.observe(cars, road)
        self.y_driver.observe(cars, road)

class HighWay(TrafficEnv):
    def __init__(self,
                 obs_noise=0.,
                 x_actions=[-1.,0.,1.],
                 y_actions=[0,1,2],
                 driver_sigma = 0.,
                 x_cost=0.01,
                 y_cost=0.01,
                 control_cost=0.01,
                 collision_cost=2.,
                 survive_reward=0.01,
                 goal_reward=2.,
                 road=Road([RoadSegment([(-100.,0.),(100.,0.),(100.,12.),(-100.,12.)])]),
                 num_updates=1,
                 dt=0.1,
                 **kwargs):

        self.obs_noise = obs_noise
        self.x_actions = x_actions
        self.y_actions = y_actions
        # we use target value instead of target change so system is Markovian
        self.rl_actions = list(itertools.product(x_actions,y_actions))
        self.num_updates = num_updates

        self.x_cost = x_cost
        self.y_cost = y_cost
        self.control_cost = control_cost
        self.collision_cost = collision_cost
        self.survive_reward = survive_reward
        self.goal_reward = goal_reward

        self.left_bound = -30.
        self.right_bound = 30.
        self.x_start = -20.
        self.x_goal = 20.
        self.lane_start = 0
        self.lane_goal = 2
        self.gap_min = 18. # 8.
        self.gap_max = 22. # 12.
        self.max_veh_num = 12 #18
        self.label_dim = 3
        self.label_num = self.max_veh_num

        self._collision = False
        self._goal = False
        self._terminal = False
        self._intentions = []
        self._empty_indices = list(range(1,self.max_veh_num+1))

        self.car_length = 5.0
        self.car_width = 2.0
        self.car_max_accel = 5.0
        self.car_max_speed = 1.0
        self.car_max_rotation = 0.
        self.car_expose_level = 4
        self.driver_sigma = driver_sigma
        self.min_x = 7.
        self.min_y = 3.
        self.ego_min_x = 6.
        self.ego_min_y = 3.

        super(HighWay, self).__init__(
            road=road,
            cars=[],
            drivers=[],
            dt=dt,
            **kwargs,)

    def get_sup_labels(self):
        labels = np.array([np.nan]*self.label_num)
        for driver in self._drivers[1:]:
            i = driver._idx - 1
            labels[i] = int(driver.next_target_lane)
        return labels

    def update(self, action):
        rl_action = self.rl_actions[action]
        self._drivers[0].apply_action(rl_action)

        # recorder intentios at the begining
        for driver in self._drivers:
            driver.observe(self._cars, self._road)
        self._sup_labels = self.get_sup_labels()

        self._goal = False
        self._collision = False
        self._terminal = False
        for i_update in range(self.num_updates):
            if i_update > 0:
                for driver in self._drivers:
                    driver.observe(self._cars, self._road)
            self._actions = [driver.get_action() for driver in self._drivers]
            [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

            ego_car = self._cars[0]
            for car in self._cars[1:]:
                if ego_car.check_collision(car):
                    self._collision = True
                    return

            if (ego_car.position[0] >= self.x_goal) \
                and (which_lane(ego_car) == self.lane_goal):
                self._goal = True
                return

            if ego_car.position[0] >= self.x_goal:
                self._terminal = True
                return

            # add cars when there is enough space
            min_xs = [np.inf]*3
            for car in self._cars:
                # lane_id = which_lane(car)
                # if car.position[0] < min_xs[lane_id]:
                #     min_xs[lane_id] = car.position[0]
                if (car.position[1] < 4.0 + 1.0) \
                    and (car.position[0] < min_xs[0]):
                    min_xs[0] = car.position[0]
                if (car.position[1] > 4.0 - 1.0) and (car.position[1] < 8.0 + 1.0) \
                    and (car.position[0] < min_xs[1]):
                    min_xs[1] = car.position[0]
                if (car.position[1] > 8.0 - 1.0) \
                    and (car.position[0] < min_xs[2]):
                    min_xs[2] = car.position[0]

            for lane_id, min_x in enumerate(min_xs):
                if min_x > (self.left_bound + np.random.uniform(self.gap_min,self.gap_max) + self.car_length):
                    x, y = self.left_bound, lane_id*4.0 + 2.0
                    target_lane = np.random.choice([0,1,2])
                    car, driver = self.add_car(x, y, 0., 0., target_lane, 0.)
                    if hasattr(self, 'viewer') and self.viewer:
                        car.setup_render(self.viewer)
                        driver.setup_render(self.viewer)

            # remove cars that are out-of bound
            for car, driver in zip(self._cars[1:],self._drivers[1:]):
                if car.position[0] > self.right_bound:
                    self.remove_car(car, driver)

    def is_terminal(self):
        return (self._collision or self._goal or self._terminal)

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
        obs = np.zeros(int(4*self.max_veh_num+4))
        for car in self._cars:
            i = int(car._idx*4)
            obs[i] = car.position[0]/self.x_goal + np.random.uniform(-1.,1.)*self.obs_noise
            obs[i+1] = car.position[1]/12. + np.random.uniform(-1.,1.)*self.obs_noise
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
        x, y = ego_car.position[0], ego_car.position[1]
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]

        x_cost = -np.abs(self.x_goal-x)/(self.x_goal-self.x_start)
        reward += self.x_cost*x_cost

        y_start = self.lane_start*4.0+2.0
        y_goal = self.lane_goal*4.0+2.0
        y_cost = -np.abs(y_goal-y)/(y_goal-y_start)
        reward += self.y_cost*y_cost

        control_cost = 0. # TODO
        reward += self.control_cost*control_cost

        if self._collision:
            reward -= self.collision_cost
        elif self._goal:
            reward += self.goal_reward
        else:
            reward += self.survive_reward
        return reward

    def remove_car(self, car, driver):
        self._empty_indices.append(car._idx)
        self._cars.remove(car)
        self._drivers.remove(driver)
        if hasattr(self, 'viewer') and self.viewer:
            car.remove_render(self.viewer)
            driver.remove_render(self.viewer)

    def add_car(self, x, y, vx, vy, target_lane, theta):
        idx = self._empty_indices.pop()
        car = Car(idx=idx, length=self.car_length, width=self.car_width, color=random.choice(RED_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EnvDriver(target_lane=target_lane, 
                            min_x=self.min_x,
                            min_y=self.min_y,
                            x_sigma=self.driver_sigma, y_sigma=0.,
                            idx=idx, car=car, dt=self.dt
                            ) 
        car.set_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)

        self._cars.append(car)
        self._drivers.append(driver)
        driver.observe(self._cars, self._road)
        return car, driver

    def _reset(self):
        self._collision = False
        self._goal = False
        self._terminal = False
        self._intentions = []
        self._empty_indices = list(range(1,self.max_veh_num+1))

        self._cars, self._drivers = [], []
        x_0 = self.x_start
        y_0 = self.lane_start * 4.0 + 2.0
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          max_rotation=self.car_max_rotation,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(min_x=self.min_x, min_y = self.min_y,
                            x_sigma=self.driver_sigma, y_sigma=0.,
                            idx=0,car=car,dt=self.dt)
        car.set_position(np.array([x_0, y_0]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(0.)
        self._cars.append(car)
        self._drivers.append(driver)
        # randomly generate surrounding cars and drivers
        for lane_id in range(3):    
            if lane_id == self.lane_start:
                x = x_0 + np.random.uniform(self.gap_min,self.gap_max) + self.car_length
            else:
                x = self.left_bound + np.random.uniform(self.gap_min,self.gap_max)
            y = lane_id*4.0 + 2.0
            while (x <= self.right_bound):
                target_lane = np.random.choice([0,1,2])
                self.add_car(x, y, 0., 0., target_lane, 0.)
                x += (np.random.uniform(self.gap_min,self.gap_max) + self.car_length)

        for driver in self._drivers[1:]:
            driver.observe(self._cars, self._road)
        self._sup_labels = self.get_sup_labels()
        return None

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-40.0, 40.0, -20.0, 20.0)

    def get_camera_center(self):
        return np.array([0.,6.0])

    def update_extra_render(self, extra_input):
        start = np.array([-100.,4.0]) - self.get_camera_center()
        end = np.array([100.,4.0]) - self.get_camera_center()
        attrs = {"color":(1.,1.,1.),"linewidth":4.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([-100.,8.0]) - self.get_camera_center()
        end = np.array([100.,8.0]) - self.get_camera_center()
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
                    attrs = {"color":(intention[0],intention[1],intention[2])}
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
    actions = [3]*(2*maximum_step)
    # actions = np.load('/Users/xiaobaima/Dropbox/SISL/rlkit/tests/Traffic/Data/t_intersection/MyDQNcg0.1expl0.2/seed0/failure1.npy')
    while True:  #not done: 
        # pdb.set_trace()
        # action = actions[t][0]
        action = actions[t]
        # action = np.random.randint(env.action_space.n)
        # action = input("Action in {}\n".format(env.rl_actions))
        # action = int(action)
        while action < 0:
            t = 0
            cr = 0.
            env.reset()
            env.render()
            action = input("Action\n")
            action = int(action)
        t += 1
        obs, reward, done, info = env.step(action)
        print('t: ', t)
        print('action: ',action)
        print('obs: ', obs)
        print('reward: ', reward)
        print('info: ', info)
        cr += reward
        env.render()
        # pdb.set_trace()
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
