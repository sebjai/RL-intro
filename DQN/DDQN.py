# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from Environment import Environment 

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

import copy


class ANN(nn.Module):
    
    def __init__(self, n_in, n_out, nNodes, nLayers, activation='silu' ):
        super(ANN, self).__init__()
        
        self.prop_in_to_h = nn.Linear( n_in, nNodes)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])
            
        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation =='relu':
            self.g = nn.ReLU()
            
        # # log(1 + e^x) -- this is not needed (yet)
        # self.softplus = nn.Softplus()

    def forward(self, x):
        
        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))
        
        for prop in self.prop_h_to_h:
            h = self.g(prop(h))
        
        # hidden layer to output layer - no activation
        y = self.prop_h_to_out(h)
        
        return y

class DDQN():
    
    def __init__(self, env : Environment, gamma = 0.99, n_nodes = 20, n_layers = 5, lr=1e-3):
        
        self.env = env
        self.gamma = gamma
        
        # define a network for our Q-function approximation
        #
        # features = asset price, inventory
        # out = Q((S,I), a)  a = short, nothing, long
        #
        self.Q_net = ANN(n_in=2, n_out=3, nNodes=n_nodes, nLayers=n_layers)
        
        self.Q_target_net = copy.deepcopy(self.Q_net)
        
        # define an optimzer
        self.optimizer = optim.AdamW(self.Q_net.parameters(), lr)
        
        self.loss = []
        
        self.S = []
        self.I = []
        self.r = []
        self.epsilon = []
        
    def random_choice(self, a, size):
        
        b = torch.tensor(np.random.choice(a, size)).int()
        
        return b
    
    def action_idx_to_action(self, a_idx):
        
        return a_idx - 1.0
        
    def Train(self, n_iter = 1_000, mini_batch_size=256, n_plot=500):
        
        # ranomly initialize the initial conditions
        
        S = self.env.randomise_S0(mini_batch_size).float()
        I = self.random_choice([-1, 0, 1], (mini_batch_size,)).float()
        
        C = 100
        D = 100
        for i in tqdm(range(n_iter)):
        
            epsilon = C/(D+i)
            
            self.epsilon.append(epsilon)
        
            # this is for proper gradient computations
            self.optimizer.zero_grad()
            
            
            # concatenate states so net can compute properly
            X = torch.cat( (S.reshape(-1,1), I.reshape(-1,1)), axis=1)
            Q = self.Q_net(X)
            
            # find argmax for greedy actions
            _, a_idx = torch.max(Q, dim=1)
            
            # # add in epsilon-greedy actions
            H = (torch.rand(mini_batch_size) < epsilon)
            a_idx[H] = self.random_choice([0, 1, 2], torch.sum(H).numpy()).long()
            
            # import pdb
            # pdb.set_trace()
            
            Q_eval = torch.zeros(mini_batch_size).float()
            for k in [0,1,2]:
                mask = (a_idx == k)
                Q_eval[mask] = Q[mask, k]
            
            # import pdb
            # pdb.set_trace()
            
            a = self.action_idx_to_action(a_idx)
            
            
            # step in the environment
            r, Sp, Ip = self.env.step(S, I, a.detach(), nsims=mini_batch_size )
            
            # compute the Q(S', a*)
            Xp = torch.cat( (Sp.reshape(-1,1), Ip.reshape(-1,1)), axis=1)
            Qp_target = self.Q_target_net(Xp)
            Qp_main = self.Q_net(Xp)
            
            _, ap_idx = torch.max(Qp_main, dim=1)
            
            Qp_eval = torch.zeros(mini_batch_size).float()
            for k in [0,1,2]:
                mask = (ap_idx == k)
                Qp_eval[mask] = Qp_target[mask, k]
            
            # compute the loss
            target = r + self.gamma * Qp_eval
            loss = torch.mean(( target.detach() - Q_eval )**2)
         
            # compute the gradients
            loss.backward()
            
            # perform SGD / Adam  AdamW step using those gradients
            self.optimizer.step()
            
            self.loss.append(loss.item())
            
            # update state
            S = Sp.clone()
            I = Ip.clone()
            
            self.S.append(S.numpy())
            self.I.append(I.numpy())
            self.r.append(r.numpy())
            
            if np.mod(i,100) == 0:
                self.Q_target_net = copy.deepcopy(self.Q_net)
            
            if np.mod(i, n_plot) == 0:
                
                plt.plot(self.loss)
                plt.yscale('log')
                plt.show()
                
                self.RunStrategy(100)
                self.PlotPolicy()
                
        self.Q_target_net = copy.deepcopy(self.Q_net)   
        
        
    def RunStrategy(self, nsteps, nsims=10_000):
        
        
        S = torch.zeros((nsims,nsteps)).float()
        I = torch.zeros((nsims,nsteps)).float()
        a = torch.zeros((nsims,nsteps)).float()
        r = torch.zeros((nsims,nsteps)).float()
        
        S[:,0] = self.env.theta
        I[:,0] = 0
        
        for i in range(nsteps-1):
            
            X = torch.cat((S[:,i].reshape(-1,1),I[:,i].reshape(-1,1)), axis=1)
            Q = self.Q_net(X).detach()
            
            a_idx = torch.argmax(Q, axis=1)
            
            a[:,i] = self.action_idx_to_action(a_idx)
            
            # import pdb
            # pdb.set_trace()
            
            r[:,i], S[:,i+1], I[:,i+1] = self.env.step(S[:,i], I[:,i], a[:,i], nsims=nsims)
            
            
        S = S.detach().numpy()
        a = a.detach().numpy()
        r = r.detach().numpy()
        I = I.detach().numpy()
        
        t = self.env.dt*np.arange(0,S.shape[1])
        
        plt.subplot(2,2,1)
        plt.plot(t, S[0,:])
        plt.plot(t, self.env.theta + (self.env.S0-self.env.theta)*np.exp(-self.env.kappa*t))
        plt.axhline(self.env.theta, linestyle='--', color='k')
        plt.title(r"$S_t$")
        
        plt.subplot(2,2,2)
        plt.plot(t, np.cumsum(r[0,:]))
        plt.title(r"$\sum_{u=1}^t r_u$")
        
        plt.subplot(2,2,3)
        plt.plot(t, I[0,:])
        plt.title(r"$I_t$")
        
        plt.subplot(2,2,4)
        plt.plot(t, a[0,:])
        plt.title(r"$a_t$")
        
        plt.tight_layout()
        
        plt.show()
        
        plt.hist(np.sum(r,axis=1), bins=np.linspace(-0.2,1,51), density=True)
        plt.ylim(0,4)
        plt.title(r"$\sum_t r_t$")
        plt.show()
                
     
    def PlotPolicy(self):
        
        S = self.env.theta + 2*self.env.inv_vol*torch.linspace(-1,1,51).float()
        
        for I in [-1, 0, 1]:
            
            X = torch.cat((S.reshape(-1,1), I * torch.ones((S.shape[0],1)) ), axis=1)
            Q = self.Q_net(X)
            
            a_idx = torch.argmax(Q, axis=1)
            
            a = self.action_idx_to_action(a_idx)
            
            plt.plot(S.detach().numpy(), a.detach().numpy(), label=r"$I=" + str(I) + "$", alpha=0.5, linewidth=(2-I))
            
        plt.legend()
        plt.show()