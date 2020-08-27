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
    def __init__(self, yld=True, t=1.5, s_min=0.,
                v_normal=3., v_ny=6., 
                s_normal=3., s_ny=1.,
                **kwargs):
        self.yld = yld
        self.t = t 
        self.s_min = s_min
        self.v_normal = v_normal
        self.v_ny = v_ny
        self.s_normal = s_normal
        self.s_ny = s_ny
        self.intention = 0 # 0: noraml drive; 1: yield 2: not yield
        super(YNYDriver, self).__init__(**kwargs)

    def set_yld(self, yld):
        self.yld = yld

    def observe(self, cars, road):
        s = cars[0].position[0] - self.car.position[0]
        s = s * self.x_driver.direction
        t = self.car.get_distance(cars[0],1)
        ego_vy = cars[0].velocity[1]
        # print("t: ",t, self.t1, self.t2)
        self.x_driver.set_v_des(self.v_normal)
        self.x_driver.s_des = self.s_normal
        if (s < self.s_min) or (t > self.t) or (ego_vy <= 0): # normal drive
            self.x_driver.observe(cars[1:], road)
            self.intention = 0
        else:
            if self.yld: # yield
                self.x_driver.min_overlap = self.t
                self.x_driver.observe(cars, road)
                self.intention = 1
            else: # not yield
                self.x_driver.set_v_des(self.v_ny)
                self.x_driver.s_des = self.s_ny
                self.x_driver.observe(cars[1:], road)
                self.intention = 2
        self.y_driver.observe(cars, road)

    def setup_render(self, viewer):
        if self.yld:
            self.car._color = GREEN_COLORS[0]
        else:
            self.car._color = RED_COLORS[0]

    def update_render(self, camera_center):
        if self.yld:
            self.car._color = GREEN_COLORS[0]
        else:
            self.car._color = RED_COLORS[0]

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
                concern_distance=0.8,
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
            self.a_s = -self.k_v_safe * v_s
            self.a_t = -self.k_v_safe * v_t
            self.a_s = np.clip(self.a_s,-self.as_max_safe,self.as_max_safe)
            self.a_t = np.clip(self.a_t,-self.at_max_safe,self.at_max_safe)

    def get_action(self):
        return TrajectoryAccelAction(self.a_s, self.a_t, self.trajectory)

class TIntersectionExtreme(TrafficEnv):
    def __init__(self,
                 yld=0.5,
                 observe_mode='full',
                 label_mode='full',
                 normalize_obs=False,
                 vs_actions=[0.,3.],
                 t_actions=[0.],
                 desire_speed=3.,
                 driver_sigma = 0.,
                 speed_cost=0.0,
                 t_cost=0.0,
                 control_cost=0.0,
                 collision_cost=1.,
                 outroad_cost=1.,
                 survive_reward=0.0,
                 goal_reward=1.,
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
        self.observe_mode = observe_mode
        self.label_mode = label_mode
        self.normalize_obs = normalize_obs
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
        self._collision = False
        self._outroad = False
        self._goal = False
        self._intentions = []

        self.left_bound = left_bound
        self.right_bound = right_bound
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.max_veh_num = max_veh_num
        self.label_dim = 3
        if self.label_mode == 'full':
            if observe_mode == 'full':
                self.label_num = self.max_veh_num
            elif observe_mode == 'important':
                self.label_num = 4
        elif self.label_mode == 'important':
            self.label_num = 1

        self.car_length=5.0
        self.car_width=2.0
        self.car_max_accel=10.0
        self.car_max_speed=40.0
        self.car_expose_level=4
        self.driver_sigma = driver_sigma
        self.s_des = 3.0
        self.s_min = 3.0
        self.min_overlap = 1.0

        super(TIntersectionExtreme, self).__init__(
            road=road,
            cars=[],
            drivers=[],
            dt=dt,
            **kwargs,)

    def get_sup_labels(self):
        labels = np.array([np.nan]*self.label_num)
        if self.label_mode == 'full':
            i = 0
            if self.observe_mode == 'full':
                upper_indices, lower_indices = self.get_sorted_indices()
                for indx in lower_indices:
                    labels[i] = int(self._drivers[indx].yld)
                    i += 1
                i = int(self.max_veh_num/2)
                for indx in upper_indices:
                    labels[i] = int(self._drivers[indx].yld)
                    i += 1
            elif self.observe_mode == 'important':
                important_indices = self.get_important_indices()
                for indx in important_indices:
                    if indx is None:
                        i += 1
                    else:
                        labels[i] = int(self._drivers[indx].yld)
                        i += 1
        elif self.label_mode == 'important':
            # [ind_ll, ind_lr, ind_ul, ind_ur]
            ind_ll, ind_lr, ind_ul, ind_ur = self.get_important_indices()
            if ind_lr is not None:
                labels[0] = int(self._drivers[ind_lr].yld)
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

            for i,driver in enumerate(self._drivers):
                if i > 0:
                    driver.observe(self._cars, self._road)
            important_indices = self.get_important_indices()
            for indx in important_indices:
                if indx is not None:
                    if self._drivers[indx].intention == 2:
                        self._collision = True
                        return
                    elif self._drivers[indx].intention == 1:
                        self._goal = True
                        return

            # remove cars that are out-of bound
            for car, driver in zip(self._cars[1:],self._drivers[1:]):
                if(car.position[1] < 4.) and (car.position[0] < self.left_bound):
                    self.remove_car(car, driver)
                elif(car.position[1] > 4.) and (car.position[0] > self.right_bound):
                    self.remove_car(car, driver)

            # add cars when there is enough space
            min_upper_x = np.inf
            max_lower_x = -np.inf
            for car in self._cars[1:]:
                if (car.position[1] < 4.) and (car.position[0] > max_lower_x):
                    max_lower_x = car.position[0]
                if (car.position[1] > 4.) and (car.position[0] < min_upper_x):
                    min_upper_x = car.position[0]
            if max_lower_x < (self.right_bound - np.random.rand()*(self.gap_max-self.gap_min) - self.gap_min - self.car_length):
                v_des = self.desire_speed
                p_des = 2.
                direction = -1
                x = self.right_bound
                car, driver = self.add_car(0, x, 2., -self.desire_speed, 0., v_des, p_des, direction, np.pi)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)
            if min_upper_x > (self.left_bound + np.random.rand()*(self.gap_max-self.gap_min) + self.gap_min + self.car_length):
                v_des = self.desire_speed
                p_des = 6.
                direction = 1
                x = self.left_bound
                car, driver = self.add_car(0, x, 6., self.desire_speed, 0., v_des, p_des, direction, 0.)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer)
                    driver.setup_render(self.viewer)

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
        if self.observe_mode == 'full':
            obs = np.zeros(int(4*self.max_veh_num+4))
            obs[:2] = self._cars[0].position
            obs[2:4] = self._cars[0].velocity
            upper_indices, lower_indices = self.get_sorted_indices()
            i = 4
            for indx in lower_indices:
                obs[i:i+2] = self._cars[indx].position - self._cars[0].position
                obs[i+2:i+4] = self._cars[indx].velocity - self._cars[0].velocity
                i += 4
            i = int(4 + self.max_veh_num/2*4)
            for indx in upper_indices:
                obs[i:i+2] = self._cars[indx].position - self._cars[0].position
                obs[i+2:i+4] = self._cars[indx].velocity - self._cars[0].velocity
                i += 4
        elif self.observe_mode == 'important':
            obs = np.zeros(int(4*4+4))
            obs[:2] = self._cars[0].position
            obs[2:4] = self._cars[0].velocity
            important_indices = self.get_important_indices()
            i = 4
            for indx in important_indices:
                if indx is None:
                    obs[i:i+4] = 0.
                else:
                    obs[i:i+2] = self._cars[indx].position - self._cars[0].position
                    obs[i+2:i+4] = self._cars[indx].velocity - self._cars[0].velocity
                i += 4
        if self.normalize_obs:
            obs[0::4] = obs[0::4]/self.right_bound
            obs[1::4] = obs[1::4]/self.right_bound
            obs[2::4] = obs[2::4]/self.desire_speed
            obs[3::4] = obs[3::4]/self.desire_speed
        obs = np.copy(obs)
        return obs

    @property
    def observation_space(self):
        if self.observe_mode == 'full':
            low = -np.ones(int(4*self.max_veh_num+4))
            high = np.ones(int(4*self.max_veh_num+4))
        elif self.observe_mode == 'important':
            low = -np.ones(20)
            high = np.ones(20)
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

    def add_car(self, idx, x, y, vx, vy, v_des, p_des, direction, theta):
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

    def get_important_indices(self):
        # return indices of 4 other vehicles that are closest to ego
        # on 4 directions
        ego_x = self._cars[0].position[0]
        min_ll, min_lr, min_ul, min_ur = np.inf, np.inf, np.inf, np.inf
        ind_ll, ind_lr, ind_ul, ind_ur = None, None, None, None
        for idx,car in enumerate(self._cars[1:]):
            x, y = car.position
            if y < 4.:
                if (x <= ego_x) and (ego_x - x < min_ll):
                    min_ll = ego_x - x
                    ind_ll = idx + 1
                elif (x > ego_x) and (x - ego_x < min_lr):
                    min_lr = x - ego_x
                    ind_lr = idx + 1
            else:
                if (x < ego_x) and (ego_x - x < min_ul):
                    min_ul = ego_x - x
                    ind_ul = idx + 1
                elif (x >= ego_x) and (x - ego_x < min_ur):
                    min_ur = x - ego_x
                    ind_ur = idx + 1
        return [ind_ll, ind_lr, ind_ul, ind_ur]

    def get_sorted_indices(self):
        # return indices of all other vehicles from left to right
        upper_indices, upper_xs = [], []
        lower_indices, lower_xs = [], []
        for indx,car in enumerate(self._cars[1:]):
            if car.position[1] > 4.:
                upper_indices.append(indx+1)
                upper_xs.append(car.position[0])
            else:
                lower_indices.append(indx+1)
                lower_xs.append(car.position[0])
        upper_indices = np.array(upper_indices)[np.argsort(upper_xs)]
        lower_indices = np.array(lower_indices)[np.argsort(lower_xs)]
        return upper_indices, lower_indices

    def _reset(self):
        self._collision = False
        self._outroad = False
        self._goal = False

        self._cars, self._drivers = [], []
        car = Car(idx=0, length=self.car_length, width=self.car_width, color=random.choice(BLUE_COLORS),
                          max_accel=self.car_max_accel, max_speed=self.car_max_speed,
                          expose_level=self.car_expose_level)
        driver = EgoDriver(trajectory=EgoTrajectory(),idx=0,car=car,dt=self.dt)
        car.set_position(np.array([0., -2.5]))
        car.set_velocity(np.array([0., 0.]))
        car.set_rotation(np.pi/2.)
        driver.v_des = 0.
        driver.t_des = 0.
        self._cars.append(car)
        self._drivers.append(driver)
        # randomly generate surrounding cars and drivers
        idx = 1
        # upper lane
        x = self.left_bound + np.random.rand()*(self.gap_max-self.gap_min)
        while (x < self.right_bound):
            v_des = self.desire_speed
            p_des = 6.
            direction = 1
            self.add_car(idx, x, 6., self.desire_speed, 0., v_des, p_des, direction, 0.)
            x += (np.random.rand()*(self.gap_max-self.gap_min) + self.gap_min + self.car_length)
            idx += 1
        # lower lane
        x = self.right_bound - np.random.rand()*(self.gap_max-self.gap_min)
        while (x > self.left_bound):
            v_des = self.desire_speed
            p_des = 2.
            direction = -1
            self.add_car(idx, x, 2., -self.desire_speed, 0., v_des, p_des, direction, np.pi)
            x -= (np.random.rand()*(self.gap_max-self.gap_min) + self.gap_min + self.car_length)
            idx += 1

        self._sup_labels = self.get_sup_labels()
        return None

    def setup_viewer(self):
        from traffic import rendering
        self.viewer = rendering.Viewer(1200, 800)
        self.viewer.set_bounds(-30.0, 30.0, -20.0, 20.0)

    def update_extra_render(self, extra_input):
        if self.observe_mode == 'important':
            important_indices = self.get_important_indices()
            for ind in important_indices:
                if ind is None:
                    pass
                else:
                    center = self._cars[ind].position - self.get_camera_center()
                    attrs = {"color":(1.,0.,0.),"linewidth":1.}
                    from traffic.rendering import make_circle, _add_attrs
                    circle = make_circle(radius=1., res=15, filled=False, center=center)
                    _add_attrs(circle, attrs)
                    self.viewer.add_onetime(circle)
        if self.label_mode == 'important':
            ind_ll, ind_lr, ind_ul, ind_ur = self.get_important_indices()
            for ind in [ind_lr]:
                if ind is None:
                    pass
                else:
                    center = self._cars[ind].position - self.get_camera_center()
                    attrs = {"color":(0.,0.,1.),"linewidth":1.}
                    from traffic.rendering import make_circle, _add_attrs
                    circle = make_circle(radius=0.8, res=15, filled=False, center=center)
                    _add_attrs(circle, attrs)
                    self.viewer.add_onetime(circle)
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
                for start, end, attention in zip(starts[-3:],ends[-3:],attentions[-3:]):
                    attrs = {"color":(1.,0.,1.),"linewidth":10.*attention}
                    if (start == end).all():
                        from traffic.rendering import make_circle, _add_attrs
                        circle = make_circle(radius=1., res=15, filled=False, center=start)
                        _add_attrs(circle, attrs)
                        self.viewer.add_onetime(circle)
                    else:
                        self.viewer.draw_line(start, end, **attrs)
            if ('intention' in extra_input.keys()) and (extra_input['intention'] is not None):
                car_indices = [np.nan]*self.label_num
                if self.label_mode == 'full':
                    upper_indices, lower_indices = self.get_sorted_indices()
                    car_indices[0:len(lower_indices)] = lower_indices[:]
                    car_indices[int(self.max_veh_num/2):int(self.max_veh_num/2)+len(upper_indices)] = upper_indices[:]
                elif self.label_mode == 'important':
                    ind_ll, ind_lr, ind_ul, ind_ur = self.get_important_indices()
                    car_indices[0] = (ind_lr if ind_lr else np.nan)
                for car_ind,intention in zip(car_indices,extra_input['intention']):
                    if not np.isnan(car_ind):
                        from traffic.rendering import make_circle, _add_attrs
                        start = self._cars[car_ind].position - self.get_camera_center()
                        attrs = {"color":(intention[2],intention[0],intention[1])}
                        circle = make_circle(radius=0.5, res=15, filled=True, center=start)
                        _add_attrs(circle, attrs)
                        self.viewer.add_onetime(circle) 

if __name__ == '__main__':
    import time
    import pdb
    env = TIntersectionExtreme(num_updates=1, yld=0.5, driver_sigma=0.1, 
                            normalize_obs=True,
                            observe_mode='important',
                            label_mode='important')
    obs = env.reset()
    img = env.render()
    done = False
    maximum_step = 200
    t = 0
    cr = 0.
    # actions = [*[8]*2,*[8]*4,*[7]*20]
    actions = [*[1]*10,*[1]*20,*[1]*200]
    # actions = np.load('/Users/xiaobaima/Dropbox/SISL/rlkit/tests/Traffic/Data/t_intersection/MyDQNcg0.1expl0.2/seed0/failure1.npy')
    while True:  #not done: 
        # pdb.set_trace()
        # if t >= actions.shape[0]:
        #     action = 7
        # else:
        #     action = actions[t][0]
        # action = actions[t]
        # action = np.random.randint(env.action_space.n)
        action = input("Action\n")
        action = int(action)
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
