# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:09:05 2022

@author: sebja
"""

from Simulator import Simulator
from EpGreedy import EpGreedy
from PolicyIter import PolicyIter

import numpy as np
import matplotlib.pyplot as plt

#%%

mu = [-0.5, -0.2, -0.2]
sigma = [1, 3, 1]

Sim = Simulator(mu, sigma)


def GenerateSims(a):
    r = []
    for i in range(1000):
        r.append(Sim.Sim(a))

    return np.array(r)

r = []
r.append(GenerateSims(0))
r.append(GenerateSims(1))
r.append(GenerateSims(2))
bins = np.linspace(-6, 11, 21)
for i, r_ in enumerate(r):
    print(np.mean(r_), np.std(r_))

    plt.hist(r_, label=str(i), bins=bins, alpha=0.6, density=True)

plt.legend()
plt.show()

#%%
bandit = EpGreedy(Sim)

bandit.run(n_iter=1_000)

#%%
bandit = PolicyIter(Sim)

bandit.run(n_iter=100_000)


















