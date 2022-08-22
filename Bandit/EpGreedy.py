# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:19:38 2022

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class EpGreedy():
    
    def __init__(self, Simulator):
        
        self.Simulator = Simulator
        
        self.Q = np.random.randn(self.Simulator.N_actions)
        
        # store the counts for a particular action being taken
        self.N = np.zeros(self.Simulator.N_actions, int)
        
    def run(self, n_iter=1_000):
        
        r = []
        
        # store the history of Q's
        Q = np.zeros((n_iter+1, self.Simulator.N_actions) )
        Q[0,:] = self.Q
        
        epsilon = []
        C = 100
        D = 200
        for k in tqdm(range(n_iter)):
            
            epsilon.append(C/(D+k))
            
            # pick an epsilo greedy action
            U = np.random.rand()    
            
            H = (U <= epsilon[-1] )
            
            if H == 1:
                
                # pick action at random
                a = np.random.randint(0, self.Simulator.N_actions)
                
            else:
                
                # pick what is currently optimal
                a = np.argmax(self.Q)
                
                # if there are multiple actions with Q
                # randomly select from them
                idx = np.where(self.Q==self.Q[a])
                if len(idx) > 1 :
                    # j = np.random.randint(0, len(idx))
                    # a = idx[j]
                    # the next line does the above in one shot
                    a = np.random.choice(idx)
                
            r.append(self.Simulator.Sim(a))
            
            # update the Q function
            self.Q[a] = self.Q[a] + (r[-1] - self.Q[a])/(self.N[a]+1)
            
            self.N[a] += 1
            
            Q[k+1, :] = self.Q
            
        plt.plot(Q)
        plt.plot(np.array(epsilon))
        plt.show()
            