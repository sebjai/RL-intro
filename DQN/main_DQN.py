# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:35:23 2022

@author: sebja
"""

from Environment import Environment
from DQN import DQN
from DDQN import DDQN

import numpy as np
import matplotlib.pyplot as plt

import copy

#%%
# env = Environment(S0=1.2, kappa=3, lambd=0.01)
env = Environment(S0=1.2, kappa=3, lambd=0.0)

S = env.Sim_donothing(nsims=10)

S = S.detach().numpy()

t = env.dt*np.arange(0,S.shape[0])

plt.plot(t, S)
plt.plot(t, env.theta + (env.S0-env.theta)*np.exp(-env.kappa*t))
plt.axhline(env.theta, linestyle='--', color='k')
plt.show()


#%%
S, I, a, r = env.Sim_blsh(nsims=1)

t = env.dt*np.arange(0,S.shape[0])

plt.subplot(2,2,1)
plt.plot(t, S)
plt.plot(t, env.theta + (env.S0-env.theta)*np.exp(-env.kappa*t))
plt.axhline(env.theta, linestyle='--', color='k')
plt.title(r'$S$')

plt.subplot(2,2,2)
plt.plot(t, np.cumsum(r))
plt.title(r'$\sum r$')

plt.subplot(2,2,3)
plt.plot(t, I)
plt.title(r'$I$')


plt.subplot(2,2,4)
plt.plot(t, a)
plt.title(r'$a$')

plt.tight_layout()

plt.show()

#%%
env = Environment(S0=1, kappa=3, lambd=0.01)
dqn = DQN(env, gamma=0.99, lr=1e-3)
dqn.Train(n_iter = 50_000, mini_batch_size=256, n_plot=5_000)
dqn.PlotPolicy()

#%%
env = Environment(S0=1, kappa=3, lambd=0)
ddqn = DDQN(env, gamma=0.99, lr=1e-3)
ddqn_all = []
for lambd in [0, 0.01, 0.02, 0.03]:
    env = Environment(S0=1, kappa=3, lambd=lambd)
    ddqn.env = env
    ddqn.Train(n_iter = 10_000, mini_batch_size=1024, n_plot= 500)
    ddqn.PlotPolicy()
    ddqn_all.append(copy.deepcopy(ddqn))