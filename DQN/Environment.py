# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:23:45 2022

@author: sebja
"""

import numpy as np
import torch

class Environment():
    
    def __init__(self, S0=0.5, kappa=5, theta=1, sigma=0.1, lambd=0.1, dt=0.05):
        
        self.S0 = S0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.lambd = lambd
        
        self.beta = np.exp(-self.kappa*self.dt)
        self.alpha = self.theta*(1-self.beta)
        self.eff_vol = self.sigma*np.sqrt((1-self.beta**2)/(2*self.kappa))
        self.inv_vol = self.sigma/np.sqrt(2*self.kappa)
        
    def randomise_S0(self, nsims=1):
        
        S0 = self.theta + 2*self.inv_vol*torch.randn(nsims)
        
        return S0
        
    def step(self, S, I, a, nsims=1):
        
        Sp = self.alpha + self.beta*S + self.eff_vol * torch.randn(nsims)
        
        Ip = 1.0*a
        
        r = a*(Sp-S) - self.lambd*torch.abs(a-I)*S 
        
        return r, Sp, Ip
    
    # do nothing simulation
    def Sim_donothing(self, nsteps = 100, nsims=1):
        
        S = torch.zeros((nsteps, nsims))
        S[0,:] = self.S0
        
        for i in range(nsteps-1):
            
            _, S[i+1,:], _ = self.step(S[i,:], torch.zeros(nsims), torch.zeros(nsims), nsims)
            
        return S
    
    # simulation buy low, sell high
    def Sim_blsh(self, nsteps = 100, nsims=1):
        
        S = torch.zeros((nsteps, nsims))
        I = torch.zeros((nsteps, nsims))
        a = torch.zeros((nsteps, nsims))
        r = torch.zeros((nsteps, nsims))
        
        S[0,:] = self.S0
        I[0,:] = 0
        
        for i in range(nsteps-1):
            
            a[i,:] = (-1)*(S[i,:] > self.theta) \
                + (+1)*(S[i,:] < self.theta)
            
            r[i,:], S[i+1,:], I[i+1,:] = self.step(S[i,:], I[i,:], a[i,:], nsims)
            
        return S, I, a, r