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
    def __init__(self, yld=True, t1=1.0, t2=0., 
                s_min=0., v_min=0.5,
                dv_yld=-1.0, dv_nyld=1.0, 
                ds_yld=1.0, ds_nyld=-1.0,
                **kwargs):
        self.yld = yld
        self.t1 = t1
        self.t2 = t2
        self.s_min = s_min
        self.v_min = v_min
        self.dv_yld = dv_yld
        self.dv_nyld = dv_nyld
        self.ds_yld = ds_yld
        self.ds_nyld = ds_nyld
        self.v_des_0 = None
        self.s_des_0 = None
        self.intention = 0 # 0: noraml drive; 1: yield 2: not yield
        super(YNYDriver, self).__init__(**kwargs)

    def set_yld(self, yld):
        self.yld = yld

    def observe(self, cars, road):
        self.x_driver.observe(cars[1:], road)
        v_front = self.x_driver.front_speed 
        if self.v_des_0 is None:
            self.v_des_0 = self.x_driver.v_des
        if self.s_des_0 is None:
            self.s_des_0 = self.x_driver.front_distance

        s = cars[0].position[0] - self.car.position[0]
        s = s * self.x_driver.direction
        t = self.car.get_distance(cars[0],1)
        ego_vy = cars[0].velocity[1] * np.sign(self.car.position[1]-cars[0].position[1])
        # print("t: ",t, self.t1, self.t2)
        if self.yld:
            if v_front is None:
                self.x_driver.set_v_des(self.v_des_0)
            else:
                self.x_driver.set_v_des(v_front*1.5)
                self.x_driver.s_des = self.s_des_0
        else:
            if v_front is None:
                self.x_driver.set_v_des(self.v_des_0)
            else:
                self.x_driver.set_v_des(v_front*1.5)
                self.x_driver.s_des = self.s_des_0 * 0.5
        if  (s < self.s_min) or (t > self.t1)\
             or ((ego_vy <= self.v_min) and (t > self.t2)): # normal drive
            self.x_driver.observe(cars[1:], road)
            self.intention = 0
        else:
            if self.yld: # yield
                self.x_driver.min_overlap = self.t1
                self.x_driver.observe(cars, road)
                self.intention = 1
            else: # not yield
                self.x_driver.observe(cars[1:], road)
                self.intention = 2

        self.y_driver.observe(cars, road)

    def setup_render(self, viewer):
        if self.yld:
            self.car._color = [*GREEN_COLORS[0],0.5]
        else:
            self.car._color = [*RED_COLORS[0],0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

    def update_render(self, camera_center):
        if self.yld:
            self.car._color = [*GREEN_COLORS[0],0.5]
        else:
            self.car._color = [*RED_COLORS[0],0.5]
        self.car._arr_color = [0.8, 0.8, 0.8, 0.5]

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
                v_des=0.0,
                t_des=0.0,
                k_s_p=2.0,
                k_t_p=2.0,
                k_t_d=2.0,
                sigma=0.0, 
                as_max=3.0,
                at_max=3.0,
                as_max_safe=6.0,
                at_max_safe=6.0,
                concern_distance=1.0,
                safe_distance=0.5,
                safe_speed=1.0,
                **kwargs):

        self.trajectory = trajectory
        self.v_des = v_des
        self.t_des = t_des
        self.k_s_p = k_s_p
        self.k_t_p = k_t_p
        self.k_t_d = k_t_d
        self.as_max = as_max
        self.at_max = at_max
        self.as_max_safe = as_max_safe
        self.at_max_safe = at_max_safe

        self.a_s = None
        self.a_t = None
        super(EgoDriver, self).__init__(**kwargs)
        # np.sqrt(self.car.length**2+self.car.width**2)/2
        self.concern_distance = concern_distance
        self.safe_distance = safe_distance
        self.safe_speed = safe_speed
        self.k_d_safe = 5.0
        self.k_v_safe = 5.0 # 2.0

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def observe(self, cars, road):
        s, t, theta, curv = self.trajectory.xy_to_traj(self.car.position)
        v_x, v_y = self.car.velocity[0], self.car.velocity[1]
        v_s = v_x*np.cos(theta) + v_y*np.sin(theta)
        v_t = -v_x*np.sin(theta) + v_y*np.cos(theta)

        self.a_s = self.k_s_p*(self.v_des-v_s)
        self.a_t = self.k_t_p*(self.t_des-t) - self.k_t_d*v_t
        self.a_s = np.clip(self.a_s,-self.as_max,self.as_max)
        self.a_t = np.clip(self.a_t,-self.at_max,self.at_max)

        # safety check
        a_x_safe = 0.
        a_y_safe = 0.
        unsafe = False
        for cid, car in enumerate(cars):
            if car is self.car:
                continue
            else:
                p1, p2 = self.car.get_closest_points(car)
                distance = np.linalg.norm(p1-p2)
                direction = (p1-p2)/distance
                v_rel = self.car.velocity - car.velocity
                speed_rel = np.sum(v_rel * direction)
                # print(distance)
                if distance < self.concern_distance:
                    if distance < self.safe_distance:
                        unsafe = True
                    elif speed_rel < -self.safe_speed:
                        unsafe = True
        if unsafe:
            # print('unsafe!')
            self.a_s = -self.k_v_safe * v_s
            self.a_t = -self.k_v_safe * v_t
            self.a_s = np.clip(self.a_s,-self.as_max_safe,self.as_max_safe)
            self.a_t = np.clip(self.a_t,-self.at_max_safe,self.at_max_safe)

    def get_action(self):
        return TrajectoryAccelAction(self.a_s, self.a_t, self.trajectory)

class TIntersectionLSTM(TrafficEnv):
    def __init__(self,
                 yld=0.5,
                 obs_noise=0.,
                 v_noise=0.,
                 vs_actions=[0.,0.5,3.],
                 t_actions=[0.],
                 desire_speed=3.,
                 driver_sigma = 0.,
                 speed_cost=0.01,
                 t_cost=0.01,
                 control_cost=0.01,
                 collision_cost=2.,
                 outroad_cost=2.,
                 survive_reward=0.01,
                 goal_reward=2.,
                 road=Road([RoadSegment([(-100.,0.),(100.,0.),(100.,8.),(-100.,8.)]),
                             RoadSegment([(-2,-10.),(2,-10.),(2,0.),(-2,0.)])]),
                 left_bound = -20.,
                 right_bound = 20.,
                 gap_min = 3.,
                 gap_max = 10.,
                 max_veh_num = 12,
                 num_updates=1,
                 dt=0.1,
                 **kwargs):

        self.yld = yld
        self.obs_noise = obs_noise
        self.v_noise = v_noise
        self.vs_actions = vs_actions
        self.t_actions = t_actions
        # we use target value instead of target change so system is Markovian
        self.rl_actions = list(itertools.product(vs_actions,t_actions))
        self.num_updates = num_updates

        self.desire_speed = desire_speed
        self.speed_cost = speed_cost
        self.t_cost = t_cost
        self.control_cost = control_cost
        self.collision_cost = collision_cost
        self.outroad_cost = outroad_cost
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
        self._outroad = False
        self._goal = False
        self._intentions = []
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num/2.)+1

        self.car_length=5.0
        self.car_width=2.0
        self.car_max_accel=10.0
        self.car_max_speed=40.0
        self.car_expose_level=4
        self.driver_sigma = driver_sigma
        self.s_des = 3.0
        self.s_min = 3.0
        self.min_overlap = 1.0

        super(TIntersectionLSTM, self).__init__(
            road=road,
            cars=[],
            drivers=[],
            dt=dt,
            **kwargs,)

    def get_sup_labels(self):
        labels = np.array([np.nan]*self.label_num)
        for driver in self._drivers[1:]:
            i = driver._idx - 1
            labels[i] = int(driver.yld)

        return labels

    def update(self, action):
        # recorder intentios at the begining
        self._sup_labels = self.get_sup_labels()

        rl_action = self.rl_actions[action]
        self._drivers[0].v_des = rl_action[0]
        self._drivers[0].t_des = rl_action[1]

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

            # add cars when there is enough space
            min_upper_x = np.inf
            max_lower_x = -np.inf
            for car in self._cars[1:]:
                if (car.position[1] < 4.) and (car.position[0] > max_lower_x):
                    max_lower_x = car.position[0]
                if (car.position[1] > 4.) and (car.position[0] < min_upper_x):
                    min_upper_x = car.position[0]
            if max_lower_x < (self.right_bound - np.random.uniform(self.gap_min,self.gap_max) - self.car_length):
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 2.
                direction = -1
                x = self.right_bound
                car, driver = self.add_car(x, 2., -v_des, 0., v_des, p_des, direction, np.pi)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)
            if min_upper_x > (self.left_bound + np.random.uniform(self.gap_min,self.gap_max) + self.car_length):
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 6.
                direction = 1
                x = self.left_bound
                car, driver = self.add_car(x, 6., v_des, 0., v_des, p_des, direction, 0.)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)

            # remove cars that are out-of bound
            for car, driver in zip(self._cars[1:],self._drivers[1:]):
                if(car.position[1] < 4.) and (car.position[0] < self.left_bound):
                    self.remove_car(car, driver)
                elif(car.position[1] > 4.) and (car.position[0] > self.right_bound):
                    self.remove_car(car, driver)

    def is_terminal(self):
        return (self._collision or self._outroad or self._goal)

    def get_info(self):
        info = {}
        info['sup_labels'] = np.copy(self._sup_labels)

        if self._collision:
            info['event']='collision'
        elif self._outroad:
            info['event']='outroad'
        elif self._goal:
            info['event']='goal'
        else:
            info['event']='nothing'

        return info

    def observe(self):
        obs = np.zeros(int(4*self.max_veh_num+4))
        for car in self._cars:
            i = int(car._idx*4)
            obs[i:i+2] = car.position/self.right_bound + np.random.uniform(-1.,1.,2)*self.obs_noise
            obs[i+2:i+4] = car.velocity/self.desire_speed + np.random.uniform(-1.,1.,2)*self.obs_noise

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
        s, t, theta, curv = self._drivers[0].trajectory.xy_to_traj(ego_car.position)
        v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]
        v_s = v_x*np.cos(theta) + v_y*np.sin(theta)
        v_t = -v_x*np.sin(theta) + v_y*np.cos(theta)

        speed_cost = -np.abs(self.desire_speed-v_s)/self.desire_speed
        reward += self.speed_cost*speed_cost

        t_cost = -np.abs(t)/(np.max(self.t_actions)+1e-3)
        reward += self.t_cost*t_cost

        control_cost = 0. # TODO
        reward += self.control_cost*control_cost

        if self._collision:
            reward -= self.collision_cost
        elif self._outroad:
            reward -= self.outroad_cost
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

    def add_car(self, x, y, vx, vy, v_des, p_des, direction, theta):
        if y < 4.:
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
                          expose_level=self.car_expose_level)
        driver = YNYDriver(idx=idx, car=car, dt=self.dt,
                    x_driver=IDMDriver(idx=idx, car=car, sigma=self.driver_sigma, s_des=self.s_des, s_min=self.s_min, axis=0, min_overlap=self.min_overlap, dt=self.dt), 
                    y_driver=PDDriver(idx=idx, car=car, sigma=0., axis=1, dt=self.dt)) 
        car.set_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)
        driver.x_driver.set_v_des(v_des)
        driver.x_driver.set_direction(direction)
        driver.y_driver.set_p_des(p_des)
        if np.random.rand() < self.yld:
            driver.set_yld(True)
        else:
            driver.set_yld(False)

        self._cars.append(car)
        self._drivers.append(driver)
        return car, driver

    def _reset(self):
        self._collision = False
        self._outroad = False
        self._goal = False
        self._intentions = []
        self._lower_lane_next_idx = 1
        self._upper_lane_next_idx = int(self.max_veh_num/2.)+1

        self._cars, self._drivers = [], []
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(trajectory=EgoTrajectory(),idx=0,car=car,dt=self.dt)
        car.set_position(np.array([0., -5.0]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(np.pi/2.)
        driver.v_des = 0.
        driver.t_des = 0.
        self._cars.append(car)
        self._drivers.append(driver)
        # randomly generate surrounding cars and drivers
        # lower lane 
        x = self.left_bound + np.random.rand()*(self.gap_max-self.gap_min)
        while (x < self.right_bound):
            v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
            p_des = 2.
            direction = -1
            self.add_car(x, p_des, -v_des, 0., v_des, p_des, direction, np.pi)
            x += (np.random.uniform(self.gap_min,self.gap_max) + self.car_length)

        # upper lane
        x = self.right_bound - np.random.rand()*(self.gap_max-self.gap_min)
        while (x > self.left_bound):
            v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
            p_des = 6.
            direction = 1
            self.add_car(x, p_des, v_des, 0., v_des, p_des, direction, 0.)
            x -= (np.random.uniform(self.gap_min,self.gap_max) + self.car_length)

        self._sup_labels = self.get_sup_labels()
        return None

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-30.0, 30.0, -20.0, 20.0)

    def update_extra_render(self, extra_input):
        t1 = self._drivers[1].t1
        t2 = self._drivers[1].t2
        
        start = np.array([self.left_bound,1.-t1]) - self.get_camera_center()
        end = np.array([self.right_bound,1.-t1]) - self.get_camera_center()
        attrs = {"color":(1.,1.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,1.-t2]) - self.get_camera_center()
        end = np.array([self.right_bound,1.-t2]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,3.+t1]) - self.get_camera_center()
        end = np.array([self.right_bound,3.+t1]) - self.get_camera_center()
        attrs = {"color":(1.,1.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,3.+t2]) - self.get_camera_center()
        end = np.array([self.right_bound,3.+t2]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)

        start = np.array([self.left_bound,5.-t1]) - self.get_camera_center()
        end = np.array([self.right_bound,5.-t1]) - self.get_camera_center()
        attrs = {"color":(1.,1.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,5.-t2]) - self.get_camera_center()
        end = np.array([self.right_bound,5.-t2]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,7.+t1]) - self.get_camera_center()
        end = np.array([self.right_bound,7.+t1]) - self.get_camera_center()
        attrs = {"color":(1.,1.,0.),"linewidth":2.}
        self.viewer.draw_line(start, end, **attrs)
        start = np.array([self.left_bound,7.+t2]) - self.get_camera_center()
        end = np.array([self.right_bound,7.+t2]) - self.get_camera_center()
        attrs = {"color":(1.,0.,0.),"linewidth":2.}
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
    env = TIntersectionLSTM(num_updates=1, yld=0.5, driver_sigma=0.1, 
                            obs_noise=0.1, v_noise=1.0,
                            )
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 200
    t = 0
    cr = 0.
    actions = [0]*(2*maximum_step)
    # actions = np.load('/Users/xiaobaima/Dropbox/SISL/rlkit/tests/Traffic/Data/t_intersection/MyDQNcg0.1expl0.2/seed0/failure1.npy')
    while True:  #not done: 
        # pdb.set_trace()
        # action = actions[t][0]
        action = actions[t]
        # action = np.random.randint(env.action_space.n)
        # action = input("Action\n")
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
