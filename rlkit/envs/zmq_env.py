import zmq
import socket
import gym
from gym.spaces import Box, Discrete
import numpy as np

class ZMQConnection:

    def __init__(self, port, ip=0):
        # hostname = socket.gethostname()    
        # IPAddr = socket.gethostbyname(hostname) 
        # self._ip = IPAddr
        self._ip = ip
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        print("tcp://{}:{}".format(self._ip, self._port))
        self._socket.connect("tcp://{}:{}".format(self._ip, self._port))

    @property
    def socket(self):
        return self._socket

    def sendreq(self, msg):
        self.socket.send_json(msg)
        respmsg = self.socket.recv_json()
        return respmsg


class ZMQEnv:

    def __init__(self, port):
        self._conn = ZMQConnection(port)
        data = self._conn.sendreq({"cmd": "obs_space"})
        assert 'type' in data
        if data['type'] == 'Box':
            assert 'high' in data
            assert 'low' in data
            self.observation_space = Box(low=np.array(data['low']), high=np.array(data['high']), dtype=np.float32)
        data = self._conn.sendreq({"cmd": "act_space"})
        assert 'type' in data
        if data['type'] == 'Box':
            assert 'high' in data
            assert 'low' in data
            self.action_space = Box(low=np.array(data['low']), high=np.array(data['high']), dtype=np.float32)
        elif data['type'] == 'Discrete':
            assert 'n' in data
            self.action_space = Discrete(data['n'])

    def step(self, action):
        if isinstance(self.action_space,Box):
            action = np.array(action).tolist()
        elif isinstance(self.action_space,Discrete):
            action = (np.array(action)+1).tolist()
        data = self._conn.sendreq({"cmd": "step", "action": action})
        assert 'obs' in data
        assert 'rew' in data
        assert 'done' in data
        assert 'info' in data
        obs = np.array(data['obs'])
        reward = np.array(data['rew'])
        done = np.array(data['done'])
        info = data['info']
        if not info:
            info = {}
        return obs, reward, done ,info

    def reset(self):
        data = self._conn.sendreq({"cmd": "reset"})
        assert 'obs' in data
        return np.array(data['obs'])

    def render(self):
        data = self._conn.sendreq({"cmd": "render"})
        return

    def close(self):
        self._conn.socket.close()


if __name__ == '__main__':
    env = ZMQEnv(9393)
    obs = env.reset()
    while True:
        action = np.array(env.action_space.sample())
        print(action)
        ob, reward, done, _ = env.step(action)

        print("s ->{}".format(obs))
        print("a ->{}".format(action))
        print("sp->{}".format(ob))
        print("r ->{}".format(reward))

        obs = ob
        if done:
            break

    env.close()
