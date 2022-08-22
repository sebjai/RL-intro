# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:07:59 2022

@author: sebja
"""

import numpy as np

class Simulator():
    
    def __init__(self, mu, sigma):
        
        assert len(mu) == len(sigma), "len(mu) must be same as len(sigma)"
        
        self.mu = mu
        self.sigma= sigma
        self.N_actions = len(mu)
        
    def Sim(self, a):
        
        return self.mu[a] + self.sigma[a] * np.random.randn()