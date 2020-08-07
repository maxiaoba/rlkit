import numpy as np
from traffic.actions.action import Action

class XYAccelAction(Action):
    def __init__(self,a_x,a_y):
        self.a_x = a_x
        self.a_y = a_y

    def update(self,car,dt):
        position_old = car.position
        velocity_old = car.velocity
        accel = np.array([self.a_x, self.a_y])
        velocity = car.set_velocity(car.velocity + accel * dt)
        car.set_position(position_old+0.5*(velocity_old+velocity)*dt)
        car.set_rotation(car.heading)