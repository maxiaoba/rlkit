from sequential_differential_game import SequentialDifferentialGame

env = SequentialDifferentialGame('max2')
print('o0',env.reset())
print('o1',env.step([0.,0.]))
print('o2',env.step([0.5,-0.5]))
print('o3',env.step([0.5,-1.0]))
print('o4',env.step([0.2,-1.0]))
print('o5',env.step([-1.0,-1.0]))