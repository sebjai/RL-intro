# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:02:19 2022

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class PolicyIter():
    
    def __init__(self, Simulator):
        
        self.Simulator = Simulator
        
        self.Q = []
        
        self.gamma = 0.1*np.random.randn(self.Simulator.N_actions)
        
        # # store the counts for a particular action being taken
        # self.N = np.zeros(self.Simulator.N_actions, int)
        
    def run(self, n_iter=1_000, eta=1e-3):
        
        r = []
        
        Y = np.zeros(self.Simulator.N_actions)
        
        # b = [0,1,2,.. K-1] -- used in update rule below
        b = np.arange(self.Simulator.N_actions)
        
        
        # store the history of gamma
        gamma = np.zeros((n_iter, self.Simulator.N_actions))
        
        for n in tqdm(range(n_iter)):
            
            #
            # pi(a) = exp(gamma_a)/sum_{a'} exp(gamma_a')
            #  ,i.e. softmax
            #
            pi = np.exp(self.gamma)
            pi /= np.sum(pi)
            
            cumsum_pi = np.cumsum(pi)
        
            #
            # choose a random action according to pi
            #
            U = np.random.rand()
            a = int(np.sum( cumsum_pi <= U ))
            
            # get a reward by "pulling this lever"
            #  i.e. take action a
            r.append(self.Simulator.Sim(a))
            
            #
            # Y := E[ R ( I(A=k) -pi(k)) ]
            #
            # Y -> Y + 1/(n+1) *( r *( I(a=k) -pi(k)) - Y)
            
            # for k in range(self.Simulator.N_actions):
            #     Y[k] = Y[k] + 1/(n+1) *(r[-1]* ((a==k) - pi[k]) - Y[k])
            
            # recall b = [0,1,2,.. K-1]
            Y = Y + 1/(n+1) *(r[-1]* ((a==b) - pi) - Y)
            
            gamma[n,:] = self.gamma
            
            self.gamma = self.gamma + eta * Y
            
            
        plt.subplot(1,2,1)
        plt.plot(gamma)
        plt.ylabel(r"$\gamma$")
        
        plt.subplot(1,2,2)
        pi = np.exp(gamma)
        pi /= np.sum(pi, axis=1).reshape(-1,1)
        plt.plot(pi)
        plt.ylabel(r"$\pi$")
        
        plt.tight_layout()
        
        plt.show()
            