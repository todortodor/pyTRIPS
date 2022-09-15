#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:39:20 2022

@author: simonl
"""

import numpy as np
from copy import deepcopy
import aa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma
from scipy import optimize
import time
import matplotlib.animation as animation


class parameters:     
    def __init__(self, n=7, s=2):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'][:n]+[i for i in range(n-7)]
        N = len(self.countries)
        self.N = N
        # self.sectors = ['Non patent', 'Patent', 'other', 'other2'][:s]
        self.sectors = ['Non patent', 'Patent']+[str(i) for i in range(s-2)]
        S = len(self.sectors)
        self.S = S
        self.eta = np.ones((N, S))*0.02  # could be over one
        self.eta[:, 0] = 0
        self.labor = np.concatenate(
            (np.array([197426230, 379553032, 84991747, 940817540, 124021697, 717517456, 1758243964])
             ,np.ones(n)*124021697)
            )[:n]
        self.labor = self.labor/self.labor.sum()*30
        # self.labor = np.ones(N)*30
        self.T = np.ones(N)*0.25  # could be anything >0
        self.k = 1.5                  #
        self.rho = 0.02  # 0.001 - 0.02
        self.alpha = np.concatenate((np.array([0.5758, 0.3545]),np.ones(s)*0.5))[:s]
        self.fe = np.ones(S)*2.7  # could be over one
        self.fo = np.ones(S)*2.3  # could be over one
        self.sigma = np.ones(S)*3   #
        self.theta = np.ones(S)*8   #
        self.beta = np.concatenate((np.array([0.74, 0.26]),np.ones(s)*0.5))[:s]
        self.beta = self.beta / self.beta.sum()
        self.zeta = np.ones(S)*0.01
        self.g_0 = 0.01  # makes sense to be low
        # self.tau = 3+np.arange(N*N*S).reshape(N, N, S)/50
        self.tau = np.ones((N, N, S))*4
        for i in range(self.tau.shape[2]):
            np.fill_diagonal(self.tau[:, :, i], 1)
        self.kappa = 0.5            #
        self.gamma = 0.4           #
        self.delta = np.ones((N, S))*0.1
        self.nu = np.ones(S)*0.2    #
        self.nu_tilde = self.nu/2
        self.deficit = np.zeros(N)
        self.price_level_data = np.concatenate(
            (np.array([1, 1.09, 1.18, 0.35, 0.44, 0.24, 0.62])
             ,np.ones(n)*0.8)
            )[:n]
        self.deficit = np.concatenate(
            (np.array([-650210, 359158, 99389, 170021, 36294, -24930, 10277]),np.zeros(n))
              )[:n]  
        self.wage_data = np.concatenate(
            (np.array([66032, 40395, 55951, 2429, 7189, 1143, 5917]),
             np.ones(n)*30000)
            )[:n] 
        self.unit = (self.wage_data*self.labor).sum()
        # self.unit = 1
        self.wage_data = self.wage_data/self.unit
        self.deficit = self.deficit/self.unit**2
        self.deficit[0] = self.deficit[0]-self.deficit.sum()      
        self.output = np.concatenate(
            (np.array([23514908,28011779,8632722,6707045,1634664,1608557,20553953]),
            np.ones(n)*1634664)
            )[:n]/self.unit
        
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

class cobweb:
    def __init__(self, name):
        self.cob_x = []
        self.cob_y = []
        self.name = name
    
    def append_old_new(self, old, new):
        self.cob_x.append(old)
        self.cob_x.append(old)
        self.cob_y.append(new)
        self.cob_y.append(old)
        
    def plot(self, count = None, window = None, pause = 0.1):
        if window is None:
            plt.plot(self.cob_x,self.cob_y)
            plt.plot(np.linspace(min(self.cob_x),max(self.cob_x),1000),
                     np.linspace(min(self.cob_x),max(self.cob_x),1000))
            plt.scatter(self.cob_x[-2],self.cob_y[-2])
            plt.scatter(self.cob_x[-1],self.cob_y[-1],s=5)
        else:
            plt.plot(self.cob_x[-window:],self.cob_y[-window:])
            plt.plot(np.linspace(min(self.cob_x[-window:]),max(self.cob_x[-window:]),1000),
                     np.linspace(min(self.cob_x[-window:]),max(self.cob_x[-window:]),1000))
            plt.scatter(self.cob_x[-2],self.cob_y[-2])
            plt.scatter(self.cob_x[-1],self.cob_y[-1],s=5)
        if count is not None:
            plt.title(self.name+''+str(count))
        plt.show()
        time.sleep(pause)
        
class var:
    def __init__(self):
        pass

    def guess_price_indices(self, price_indices_init):
        self.price_indices = price_indices_init

    def guess_patenting_threshold(self, psi_star_init):
        self.psi_star = psi_star_init

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_expenditure(self, Y_init):
        self.Y = Y_init

    def guess_labor_research(self, l_R_init):
        self.l_R = l_R_init

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        frame = deepcopy(self)
        return frame

    def compute_growth(self, p):
        self.g_s = p.k*np.einsum('is,is -> s',
                                 p.eta,
                                 self.l_R**(1-p.kappa)
                                 )/(p.k-1) - p.zeta
        self.g_s[0] = p.g_0
        # assert np.isnan(self.g_s).sum() == 0, 'nan in g_s'
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()

        self.r = p.rho + self.g/p.gamma

    def compute_aggregate_qualities(self, p):
        A = (self.g_s + p.nu + p.zeta)
        B = np.einsum('s,nis,ns,ns -> nis',
                      p.nu,
                      np.divide(1, self.psi_star**(p.k-1), out=np.zeros_like(
                          self.psi_star), where=self.psi_star != np.inf),
                      1/(self.g_s[None, :]+p.delta+p.zeta[None, :]),
                      1/(self.g_s[None, :]+p.delta +
                         p.nu[None, :]+p.zeta[None, :])
                      )
        prefact = p.k * p.eta * self.l_R**(1-p.kappa)/(p.k-1)
        self.PSI_M = np.einsum('is,nis -> nis',
                               prefact,
                               1/A[None, None, :]+B)
        self.PSI_M[:, :, 0] = 0

        # assert np.isnan(self.psi_star).sum() == 0, 'nan in psi_star'
        # assert np.isnan(self.PSI_M).sum() == 0, 'nan in PSI_M'

        A_tilde = (self.g_s + p.nu_tilde + p.zeta)
        B_tilde = np.einsum('s,nis,ns,ns -> nis',
                            p.nu,
                            np.divide(1, self.psi_star**(p.k-1), out=np.zeros_like(
                                self.psi_star), where=self.psi_star != np.inf),
                            1/(self.g_s[None, :]+p.delta +
                               p.nu_tilde[None, :]+p.zeta[None, :]),
                            1/(self.g_s[None, :]+p.delta +
                               p.nu[None, :]+p.zeta[None, :])
                            )
        self.PSI_CL = np.einsum('is,s,nis -> nis',
                                prefact,
                                1/A_tilde,
                                (p.nu/A)[None, None:]-B_tilde*A_tilde[None, None, :])
        self.PSI_CL[:, :, 0] = 0
        # assert np.isnan(self.PSI_CL).sum() == 0, 'nan in PSI_CL'

        # assert not np.any(np.einsum('njs->ns', self.PSI_M)+np.einsum('njs->ns', self.PSI_CL) > 1),'PSI_M,CL too high'

        self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
                         np.einsum('njs->ns', self.PSI_CL))
        # assert np.isnan(self.PSI_CD).sum() == 0, 'nan in PSI_CD'

    def compute_phi(self, p):
        self.phi = p.T[None, :, None] * np.einsum('nis,is,is->nis',
                                                  p.tau,
                                                  self.w[:, None]**p.alpha[None, :],
                                                  self.price_indices[:, None]**(1-p.alpha[None, :])
                                                  )**(-p.theta[None, None, :])
        # assert np.all(self.w > 0), 'non positive wage'
        # assert np.all(np.einsum('is,is->is',
        #                          self.w[:, None]**p.alpha[None, :],
        #                          self.price_indices[:,
        #                                             None]**(1-p.alpha[None, :])
        #                          ) > 0), 'zero or negative in phi den'
        # assert np.isnan(self.phi).sum() == 0, 'nan in phi'
        # assert np.all(self.phi > 0), 'negative phi'

    def compute_price_indices(self, p):

        power = (p.sigma-1)/p.theta
        
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**power[None, None, :]).sum(axis=1)
        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)
        C = self.PSI_CD*self.phi.sum(axis=1)**power[None, :]
        price = ( (gamma((p.theta+1-p.sigma)/p.sigma)*(A+B+C))**(p.beta[None, :]/(1- p.sigma[None, :])) ).prod(axis=1)
        # assert np.isnan(price).sum() == 0, 'nan in price'
        return price

    def solve_price_ind_and_phi_with_price(self, p, price_init=None, tol_p=1e-10, 
                                           plot_convergence=False, plot_cobweb = False):
        if plot_cobweb:
            cob_price = cobweb('price')
        price_new = None
        if price_init is None:
            price_old = np.ones(p.N)
        else:
            price_old = price_init

        condition = True
        count = 0
        convergence = []

        aa_options = {'dim': p.N,
                      'mem': 10,
                      'type1': False,
                      'regularization': 1e-10,
                      'relaxation': 1,
                      'safeguard_factor': 1,
                      'max_weight_norm': 1e6}
        aa_price = aa.AndersonAccelerator(**aa_options)

        while condition:
            if count != 0:
                aa_price.apply(price_new, price_old)
                price_old = price_new
            
            self.guess_price_indices(price_old)
            self.compute_phi(p)
            price_new = self.compute_price_indices(p)
            condition = np.linalg.norm(
                price_new - price_old)/np.linalg.norm(price_new) > tol_p
            convergence.append(np.linalg.norm(
                price_new - price_old)/np.linalg.norm(price_new))
            count += 1
            
            if plot_cobweb and count%5==0:
                # cob_psi_star.append_old_new(psi_star_old[1,1,1],psi_star_new[1,1,1])
                cob_price.append_old_new(price_old[1],price_new[1])
                cob_price.plot(count=count)
                time.sleep(0.1)
            
            if count>50:
                plot_convergence = True
            
            if plot_convergence:
                plt.semilogy(convergence)
                plt.title('price')
                plt.show()
        self.price_indices = price_new

    def compute_monopolistic_sectoral_prices(self, p):
        power = (p.sigma-1)/p.theta
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M*self.phi**power[None, None, :]).sum(axis=1)

        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)

        C = self.PSI_CD*self.phi.sum(axis=1)**power[None, :]

        self.P_M = np.ones((p.N, p.S))
        self.P_M[:, 0] = np.inf
        self.P_M[:, 1:] = (A[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        # assert np.isnan(self.P_M).sum() == 0, 'nan in P_M'

    def compute_monopolistic_trade_flows(self, p):
        numerator = np.einsum('nis,nis->nis',
                              self.PSI_M,
                              self.phi**((p.sigma-1)/p.theta)[None, None, :])
        self.X_M = np.zeros((p.N, p.N, p.S))
        self.X_M[..., 1:] = np.einsum('nis,ns,ns,s,n->nis',
                                      numerator[..., 1:],
                                      1/(numerator[..., 1:].sum(axis=1)),
                                      self.P_M[..., 1:]**(1-p.sigma[None, 1:]),
                                      p.beta[1:],
                                      self.Y
                                      )
        # assert np.isnan(self.X_M).sum() == 0, 'nan in X_M'

    def compute_psi_star(self, p):
        psi_star = np.einsum('s,i,nis,s,nis,nis,ns,ns -> nis',
                             p.sigma,
                             self.w,
                             self.PSI_M,
                             1/p.nu,
                             np.divide(1, self.X_M, out=np.full_like(
                                 self.X_M, np.inf), where=self.X_M != 0),
                             p.fo[None, None, :]+np.einsum('n,i,s -> nis',
                                                           self.w,
                                                           1/self.w,
                                                           p.fe),
                             self.r+p.zeta[None, :]+p.delta - self.g+self.g_s[None, :],
                             self.r+p.zeta[None, :]+p.nu[None, :] - self.g+self.g_s[None, :]+p.delta
                             )
        psi_star[..., 0] = np.inf

        # assert np.isnan(psi_star).sum() == 0, 'nan in psi_star'
        return psi_star

    def solve_psi_star(self, p, psi_star_init=None, price_init=None, 
                       tol_psi=1e-10, plot_convergence=False, plot_cobweb = False):
        if plot_cobweb:
            cob_psi_star = cobweb('psi star')
        psi_star_new = None
        if psi_star_init is None:
            psi_star_old = np.ones((p.N, p.N, p.S))
        else:
            psi_star_old = psi_star_init

        condition = True
        count = 0
        convergence = []

        aa_options = {'dim': p.N*p.N*p.S,
                      'mem': 5,
                      'type1': False,
                      'regularization': 1e-12,
                      'relaxation': 1,
                      'safeguard_factor': 1,
                      'max_weight_norm': 1e6}
        aa_psi_star = aa.AndersonAccelerator(**aa_options)
        damping = 1.1
        
        while condition:
            # print(count,'psi_star count')
            if count != 0:
                # psi_star_new = psi_star_new.ravel()
                # psi_star_old = psi_star_old.ravel()
                # aa_psi_star.apply(psi_star_new, psi_star_old)
                # psi_star_new = psi_star_new.reshape(p.N,p.N,p.S)
                # psi_star_old = psi_star_old.reshape(p.N,p.N,p.S)
                psi_star_old = (psi_star_new+(damping-1)*psi_star_old)/damping
                # if psi_star_old.min() < 1:
                #     print('correcting psi star')
                #     psi_star_old = psi_star_old/psi_star_old.min()

            self.guess_patenting_threshold(psi_star_old)
            assert np.isnan(self.psi_star).sum() == 0, 'nan in psi_star'
            self.compute_aggregate_qualities(p)
            self.solve_price_ind_and_phi_with_price(
                p, price_init=self.price_indices)
            self.compute_monopolistic_sectoral_prices(p)
            self.compute_monopolistic_trade_flows(p)
            psi_star_new = self.compute_psi_star(p)
            assert np.isnan(psi_star_new).sum() == 0, 'nan in psi_star'

            condition = np.linalg.norm(psi_star_new[..., 1:] - psi_star_old[..., 1:]) /\
                np.linalg.norm(psi_star_new[..., 1:]) > tol_psi
            convergence.append(np.linalg.norm(psi_star_new[..., 1:] - psi_star_old[..., 1:]) /
                               np.linalg.norm(psi_star_new[..., 1:]))
            count += 1
            
            # if count>-50:
            #     plot_convergence = True #!!!!!
            if plot_convergence:
                plt.title('psi star')
                plt.semilogy(convergence)
                plt.show()
            if plot_cobweb:
                # cob_psi_star.append_old_new(psi_star_old[1,1,1],psi_star_new[1,1,1])
                cob_psi_star.append_old_new(psi_star_old.min(),psi_star_new.min())
                cob_psi_star.plot(count=count)
            
        self.psi_star = psi_star_new

    def compute_labor_research(self, p):
        A = np.einsum('nis,s,i,nis,s->is',
                      self.X_M,
                      1/p.sigma,
                      1/self.w,
                      np.divide(1, self.PSI_M, out=np.zeros_like(
                          self.PSI_M), where=self.PSI_M != 0),
                      p.k/(self.r+p.zeta+p.nu-self.g+self.g_s)
                      )
        B = np.einsum('nis,nis->is',
                      p.fo+np.einsum('n,i,s->nis',
                                     self.w,
                                     1/self.w,
                                     p.fe),
                      np.divide(
                          1, self.psi_star**(p.k), out=np.zeros_like(self.psi_star), where=self.psi_star != np.inf)
                      )
        l_R = (p.eta*(A+B)/(p.k-1))**(1/p.kappa)
        # assert np.isnan(l_R).sum() == 0, 'nan in l_R'

        return l_R

    def solve_l_R(self, p, l_R_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                  w_init=None, Y_init=None, plot_convergence=False,plot_cobweb = False):
        if plot_cobweb:
            cob_l_R = cobweb('l_R')
        l_R_new = None
        if l_R_init is None:
            l_R_old = np.ones((p.N, p.S))
        else:
            l_R_old = l_R_init
        condition_l_R = True
        count = 0
        convergence_l_R = []
        aa_options_l_R = {'dim': p.N*p.S,
                          'mem': 5,
                          'type1': False,
                          'regularization': 1e-12,
                          'relaxation': 1,
                          'safeguard_factor': 1,
                          'max_weight_norm': 1e6}
        aa_l_R = aa.AndersonAccelerator(**aa_options_l_R)
        damping = 3
        while condition_l_R:
            # print(count,'l_R count')
            if count != 0:
                l_R_new = l_R_new.ravel()
                l_R_old = l_R_old.ravel()
                aa_l_R.apply(l_R_new, l_R_old)
                l_R_old = (l_R_new.reshape(p.N, p.S) + (damping-1)*l_R_old.reshape(p.N, p.S))/damping
                # while np.any(self.compute_labor_allocations(p, l_R=l_R_old, assign=False)<0):
                #     print('correcting l_R')
                #     l_R_old = l_R_old/2
            self.guess_labor_research(l_R_old)
            self.compute_growth(p)
            self.solve_psi_star(p, psi_star_init=self.psi_star,
                                price_init=self.price_indices)
            l_R_new = self.compute_labor_research(p).astype('double')
            # print(l_R_old[1,1],l_R_new[1,1])
            if np.any(self.compute_labor_allocations(p, l_R=l_R_old, assign=False) < 0):
                print('non positive production labor')
            # print(count,'count l_P',l_P)
            # if np.any(l_P < 0):
            #     print('Negative above')
            #     time.sleep(2)
            # assert np.all(l_P > 0), 'non positive production labor'
            condition_l_R = np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new) > tol_m
            convergence_l_R.append(np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new))
            count += 1
            if plot_cobweb:
                cob_l_R.append_old_new(l_R_old[1,1],l_R_new[1,1])
                cob_l_R.plot(count=count)
            
            if plot_convergence:
                plt.semilogy(convergence_l_R, label='l_R')
                plt.show()

        self.l_R = l_R_new

    def compute_labor_allocations(self, p, l_R=None, assign=True):
        if l_R is None:
            l_R = self.l_R
        l_Ae = np.einsum('s,is,is,nis -> ins',
                         p.fe,
                         p.eta,
                         l_R**(1-p.kappa),
                         np.divide(
                             1, self.psi_star**(p.k), out=np.zeros_like(self.psi_star), where=self.psi_star != np.inf)
                         )
        l_Ao = np.einsum('ins,s,s -> ins',
                         l_Ae,
                         p.fo,
                         p.fe
                         )
        l_P = p.labor - np.einsum('is->i',
                                  np.einsum('ins->is', l_Ao+l_Ae)+l_R)

        # assert np.isnan(l_Ae).sum() == 0, 'nan in l_Ae'
        # assert np.isnan(l_Ao).sum() == 0, 'nan in l_Ao'
        # assert np.isnan(l_P).sum() == 0, 'nan in l_P'
        

        if assign:
            self.l_Ae = l_Ae
            self.l_Ao = l_Ao
            self.l_P = l_P
        else:
            return l_P

    def compute_competitive_sectoral_prices(self, p):
        power = (p.sigma-1)/p.theta
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M*self.phi**power[None, None, :]).sum(axis=1)

        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)

        C = self.PSI_CD*self.phi.sum(axis=1)**power[None, :]

        self.P_CL = np.ones((p.N, p.S))
        self.P_CL[:, 0] = np.inf
        self.P_CL[:, 1:] = (B[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        # assert np.isnan(self.P_CL).sum() == 0, 'nan in P_CL'

        self.P_CD = (C/(A+B+C))**(1/(1-p.sigma))[None, :]
        # assert np.isnan(self.P_CD).sum() == 0, 'nan in P_CD'

    def compute_competitive_trade_flows(self, p):
        numerator = np.einsum('nis,nis->nis',
                              self.PSI_CL,
                              self.phi**((p.sigma-1)/p.theta)[None, None, :])
        self.X_CL = np.zeros((p.N, p.N, p.S))
        self.X_CL[..., 1:] = np.einsum('nis,ns,ns,s,n->nis',
                                       numerator[..., 1:],
                                       1/(numerator[..., 1:].sum(axis=1)),
                                       self.P_CL[...,
                                                 1:]**(1-p.sigma[None, 1:]),
                                       p.beta[1:],
                                       self.Y
                                       )
        # assert np.isnan(self.X_CL).sum() == 0, 'nan in X_CL'
        self.X_CD = np.einsum('nis,ns,ns,s,n->nis',
                              self.phi,
                              1/(self.phi.sum(axis=1)),
                              (self.P_CD**(1-p.sigma[None, :])),
                              p.beta,
                              self.Y
                              )
        # assert np.isnan(self.X_CD).sum() == 0, 'nan in X_CD'

    def compute_wage(self, p):
        wage = (
            p.alpha[None, :] *
            (self.X_CD+self.X_CL +
             (1-1/p.sigma[None, None, :])*self.X_M).sum(axis=0)
        ).sum(axis=1)/self.l_P
        # assert np.isnan(wage).sum() == 0, 'nan in wage'
        # assert np.all(self.l_P > 0), 'non positive production labor'
        # assert np.all(wage > 0), 'non positive wage'
        return wage

    def solve_w(self, p, w_init=None, psi_star_init=None, price_init=None,
                tol_m=1e-8, plot_convergence=False, plot_cobweb = False):
        if plot_cobweb:
            cob_w = cobweb('wage')
        w_new = None
        if w_init is None:
            w_old = np.ones(p.N)
        else:
            w_old = w_init
        condition_w = True
        count = 0
        convergence_w = []
        aa_options_w_Y = {'dim': p.N,
                          'mem': 5,
                          'type1': False,
                          'regularization': 1e-12,
                          'relaxation': 1,
                          'safeguard_factor': 1,
                          'max_weight_norm': 1e6}
        aa_w = aa.AndersonAccelerator(**aa_options_w_Y)
        damping = 1.5
        while condition_w:
            # print(count,'w count')
            if count != 0:
                aa_w.apply(w_new, w_old)
                w_old = (w_new+(damping-1)*w_old)/damping
         
            self.guess_wage(w_old)
            self.solve_l_R(p, l_R_init=self.l_R)
            self.compute_labor_allocations(p)
            self.compute_competitive_sectoral_prices(p)
            self.compute_competitive_trade_flows(p)
            w_new = self.compute_wage(p)
            condition_w = np.linalg.norm(
                w_new - w_old)/np.linalg.norm(w_new) > tol_m
            convergence_w.append(np.linalg.norm(w_new - w_old)/np.linalg.norm(w_new))
            
            count += 1
            if plot_cobweb:
                cob_w.append_old_new(w_old[1],w_new[1])
                cob_w.plot(count=count)
            if plot_convergence:
                plt.semilogy(convergence_w, label='wage')
                plt.show()
        
        self.w = w_new

    def compute_expenditure(self, p):
        A = np.einsum('nis->i', self.X_CD+self.X_CL+self.X_M)
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = np.einsum('i,n,n->i', p.deficit, self.w, p.labor)
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Y = (A+B-(C+D))
        # assert np.isnan(Y).sum() == 0, 'nan in Y'

        return Y

    def solve_Y(self, p, Y_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                plot_convergence=False, plot_cobweb = False):
        if plot_cobweb:
            cob_Y = cobweb('Y')
        Y_new = None
        if Y_init is None:
            Y_old = np.ones(p.N)
        else:
            Y_old = Y_init
        condition_Y = True
        count = 0
        convergence_Y = []
        aa_options_Y = {'dim': p.N,
                        'mem': 5,
                        'type1': False,
                        'regularization': 1e-12,
                        'relaxation': 1,
                        'safeguard_factor': 1,
                        'max_weight_norm': 1e6}
        aa_Y = aa.AndersonAccelerator(**aa_options_Y)
        while condition_Y:
            print(count)
            if count != 0:
                aa_Y.apply(Y_new, Y_old)
                Y_old = Y_new
            self.guess_expenditure(Y_old)
            self.solve_w(p, w_init=self.w)
            Y_new = self.compute_expenditure(p)
            condition_Y = np.linalg.norm(
                Y_new - Y_old)/np.linalg.norm(Y_new) > tol_m
            convergence_Y.append(np.linalg.norm(
                Y_new - Y_old)/np.linalg.norm(Y_new))
            count += 1
            if plot_cobweb:
                cob_Y.append_old_new(Y_old[1],Y_new[1])
                cob_Y.plot(count=count)
            if plot_convergence:
                plt.semilogy(convergence_Y, label='Y')
                plt.show()

        self.Y = Y_new
        self.num_scale_solution(p)

    def num_scale_solution(self, p):
        numeraire = self.price_indices[0]
        
        for qty in ['X_CD','X_CL','X_M','price_indices','w','Y']:
            setattr(self, qty, getattr(self, qty)/numeraire)
        
        self.phi = self.phi*numeraire**p.theta[None,None,:]
    
    def check_solution(self, p, return_checking_copy = False, assertions = False):
        check = self.copy()
        check.compute_growth(p)
        check.compute_aggregate_qualities(p)
        check.compute_phi(p)
        check.price_indices = check.compute_price_indices(p)
        check.compute_monopolistic_sectoral_prices(p)
        check.compute_monopolistic_trade_flows(p)
        check.psi_star = check.compute_psi_star(p)
        check.l_R = check.compute_labor_research(p)
        check.compute_competitive_sectoral_prices(p)
        check.compute_competitive_trade_flows(p)
        check.compute_labor_allocations(p)
        check.w = check.compute_wage(p)
        check.Y = check.compute_expenditure(p)
        if assertions:
            assert np.all(
                np.isclose(check.price_indices, check.compute_price_indices(p))
                ), 'check prices wrong'
            assert np.all(
                np.isclose(check.psi_star, check.compute_psi_star(p))
                ), 'check psi star wrong'
            assert np.all(
                np.isclose(check.l_R, check.compute_labor_research(p))
                ), 'check l_R wrong'
            assert np.all(
                np.isclose(check.w, check.compute_wage(p))
                ), 'check w wrong'
            assert np.all(
                np.isclose(check.Y, check.compute_expenditure(p))
                ), 'check Y wrong'
        if return_checking_copy:
            return check
        
    @staticmethod
    def var_from_vector(vec,p):
        init = var()    
        init.guess_price_indices(vec[0:p.N])
        init.guess_wage(vec[p.N:p.N*2])
        init.guess_expenditure(vec[p.N*2:p.N*3])
        init.guess_labor_research(
            np.insert(vec[p.N*3:p.N*3+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1))
        init.guess_patenting_threshold(
            np.insert(vec[p.N*3+p.N*(p.S-1):].reshape((p.N, p.N, p.S-1)), 0, np.full(p.N,np.inf), axis=2))
        init.compute_growth(p)
        init.compute_aggregate_qualities(p)
        init.compute_phi(p)
        init.compute_monopolistic_sectoral_prices(p)
        init.compute_monopolistic_trade_flows(p)
        init.compute_competitive_sectoral_prices(p)
        init.compute_competitive_trade_flows(p)
        init.compute_labor_allocations(p)        
        return init
    
    def vector_from_var(self):
        price = self.price_indices
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        psi_star = self.psi_star[...,1:].ravel()
        Y = self.Y
        vec = np.concatenate((price,w,Y,l_R,psi_star), axis=0)
        return vec
        
    def compare_two_solutions(self,sol2):
        commonKeys = set(vars(self).keys()) - (set(vars(self).keys()) - set(vars(self).keys()))
        diffs = []
        for k in commonKeys:
            if np.all(np.isclose(vars(self)[k], vars(sol2)[k])):
                print(k, 'identical')
            else:
                diffs.append(k)
        
        for k in diffs:
            print(k, (np.nanmean(vars(self)[k]/vars(sol2)[k])))

def iter_once(x,p, normalize = False):
    init = var.var_from_vector(x,p)
    if normalize:
        init.num_scale_solution(p)
    price = init.compute_price_indices(p)
    w = init.compute_wage(p)
    Y = init.compute_expenditure(p)
    l_R = init.compute_labor_research(p)[...,1:].ravel()
    psi_star = init.compute_psi_star(p)[...,1:].ravel()
    vec = np.concatenate((price,w,Y,l_R,psi_star), axis=0)
    return vec

def deviation(x,p):
    vec = iter_once(x,p)
    return vec - x

def deviation_norm(x,p):
    return np.linalg.norm(deviation(x,p))

def bound_psi_star(x,p,hit_the_bound=None):
    x_psi_star = x[p.N*3+p.N*(p.S-1):]
    if np.any(x_psi_star<1):
        hit_the_bound += 1
        x_psi_star[x_psi_star<1] = 1
    x[p.N*3+p.N*(p.S-1):] = x_psi_star
    return x, hit_the_bound

def bound_research_labor(x,p,hit_the_bound=None):
    x_l_R = x[p.N*3:p.N*3+p.N*(p.S-1)]
    if np.any(x_l_R > p.labor.max()):
        if hit_the_bound is not None:
            hit_the_bound+=1
        x_l_R[x_l_R > p.labor.max()] = p.labor.max()
    x[p.N*3:p.N*3+p.N*(p.S-1)] = x_l_R
    return x,hit_the_bound

def bound_zero(x, cutoff=1e-8, hit_the_bound=None):
    if np.any(x<=0):
        x[x <= 0] = cutoff
        if hit_the_bound is not None:
            hit_the_bound+=1
    return x,hit_the_bound

def smooth_large_jumps(x_new,x_old):
    high_jumps_too_big = x_new > 1000*x_old
    while np.any(high_jumps_too_big):
        # print('high',high_jumps_too_big.sum())
        # x_new[high_jumps_too_big] = x_old[high_jumps_too_big]*1/2+x_new[high_jumps_too_big]*1/2
        x_new = x_old*1/2+x_new*1/2
        high_jumps_too_big = x_new > 1000*x_old
    low_jumps_too_big = x_new < x_old/1000
    while np.any(low_jumps_too_big):
        # print('low',low_jumps_too_big.sum())
        # x_new[low_jumps_too_big] = x_old[low_jumps_too_big]*1/2+x_new[low_jumps_too_big]*1/2
        x_new = x_old*1/2+x_new*1/2
        low_jumps_too_big = x_new < x_old/1000
    return x_new

def guess_from_params(p):
    price_guess = p.price_level_data
    w_guess = p.wage_data*10000
    Y_guess = p.output
    l_R_guess = np.repeat(p.labor[:,None]/100, p.S-1, axis=1).ravel()
    psi_star_guess = np.ones((p.N,p.N,(p.S-1))).ravel()*10
    vec = np.concatenate((price_guess,w_guess,Y_guess,l_R_guess,psi_star_guess), axis=0)
    return vec

p = parameters_julian()
j_res = np.array(
    pd.read_csv('/Users/simonl/Dropbox/TRIPS/Code/new code/temporary_result_lev_levy.csv'
                ,header=None)[5])
j_res = np.insert(j_res, 0, 1)

#%% partial equilibrium solver

p_j = parameters_julian()
x_j = guess_from_params(p_j)
p0 = parameters(n=6,s=3)
x0 = guess_from_params(p0)

x_guess = x_j
p = p_j
partial_equilibriums_sol = var.var_from_vector(x_guess, p)

start = time.perf_counter()

partial_equilibriums_sol.solve_Y(p,
                                 Y_init=partial_equilibriums_sol.Y,
                                  plot_cobweb = True
                                 )

finish = time.perf_counter()
print('Solving time :',finish-start)

#%% full fixed point solver

class sol_class:
    def __init__(self, x_new, p, solving_time, iterations, deviation_norm, 
                 status, hit_the_bound_count, x0=None, tol = 1e-10, 
                 # damping = 5, max_count=1e4,
                 # accelerate = False, safe_convergence=0.1,
                 # accelerate_when_stable=True, plot_cobweb = True, cobweb_anim=False,
                 # plot_convergence = True, apply_bound_zero = True, 
                 # apply_bound_psi_star = False, apply_bound_research_labor = False,
                 # accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                 # accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6
                 ):
        self.x = x_new
        self.p = p
        self.time = solving_time
        self.iter = iterations
        self.dev = deviation_norm
        self.status = status
        self.hit_the_bound_count = hit_the_bound_count
        self.x0 = x0
        self.tol = tol
        # self.damping = damping 
        # self.max_count = max_count
        # self.accelerate = accelerate
        # self.safe_convergence= safe_convergence
        # self.accelerate_when_stable= accelerate_when_stable
        # self.apply_bound_zero = apply_bound_zero
        # self.apply_bound_psi_star = apply_bound_psi_star
        # self.apply_bound_research_labor = apply_bound_research_labor
        # self.accel_memory = accel_memory
        # self.accel_type1 = accel_type1 
        # self.accel_regularization = accel_regularization
        # self.accel_relaxation = accel_relaxation
        # self.accel_safeguard_factor = accel_safeguard_factor 
        # self.accel_max_weight_norm = accel_max_weight_norm
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
    
    def run_summary(self):
        print(self.p.N,' countries, ', self.p.S,' sectors '
              '\nSolving time :',self.time
              ,'\nIterations : ',self.iter
              ,'\nDeviation norm : ',self.dev
              ,'\nStatus : ',self.status
              ,'\nHit the bounds ',self.hit_the_bound_count,' times'
              )
        
p_j = parameters_julian()
p0 = parameters(n=40,s=20)

def fixed_point_solver(p, x0=None, tol = 1e-10, damping = 5, max_count=1e4,
                       accelerate = False, safe_convergence=0.1,
                       accelerate_when_stable=True, plot_cobweb = True, cobweb_anim=False,
                       plot_convergence = True, apply_bound_zero = True, 
                       apply_bound_psi_star = False, apply_bound_research_labor = False,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       ):   
    if x0 is None:
        x0 = guess_from_params(p)
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    x_old, hit_the_bound_count = bound_zero(x_old,1e-8, hit_the_bound_count)
    x_old, hit_the_bound_count = bound_psi_star(x_old, p, hit_the_bound_count)
    x_old, hit_the_bound_count = bound_research_labor(x_old, p, hit_the_bound_count) 
    history_old = []
    history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                    'mem': accel_memory,
                    'type1': accel_type1,
                    'regularization': accel_regularization,
                    'relaxation': accel_relaxation,
                    'safeguard_factor': accel_safeguard_factor,
                    'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    while condition and count < max_count:
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_new = smooth_large_jumps(x_new,x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
            if apply_bound_zero:
                x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
            if apply_bound_psi_star:
                x_old, hit_the_bound_count = bound_psi_star(x_old, p, hit_the_bound_count)
            if apply_bound_research_labor:
                x_old, hit_the_bound_count = bound_research_labor(x_old, p, hit_the_bound_count) 
        x_new = iter_once(x_old, p, normalize = False)
        condition = np.linalg.norm(
            x_new - x_old)/np.linalg.norm(x_new) > tol
        convergence.append(np.linalg.norm(
            x_new - x_old)/np.linalg.norm(x_new))
        count += 1
        if np.all(np.array(convergence[-10:])<safe_convergence) and accelerate_when_stable:
            accelerate = True
            damping = 1
        # history_old.append(x_old.min())
        # history_new.append(x_new.min())
        # history_old.append(x_old[p.N*3+p.N*(p.S-1):].min())
        # history_new.append(x_new[p.N*3+p.N*(p.S-1):].min())
        history_old.append(x_old[25])
        history_new.append(x_new[25])
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = deviation_norm(x_new,p)
    if count < max_count and np.isnan(x_new).sum()==0:
        status = 'successful'
    else:
        status = 'failed'
    
    if plot_cobweb:
        cob = cobweb('all')
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = len(convergence),pause = 0.05) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence)
        plt.show()
    
    sol_inst = sol_class(x_new, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
    
    sol_inst.run_summary()
    
    return sol_inst

sol = fixed_point_solver(p0)
        
# full_fixed_point_sol = var.var_from_vector(x_new,p)
# full_fixed_point_sol.num_scale_solution(p)

#%% least square solver


# sol = optimize.root(fun = function_of_vector, 
#                     x0 = j_res, 
#                     args = p, 
#                     method='krylov',
#                     options={'disp':True})

# x0 = state.vector_from_var()
x0 = j_res
# p_bds = (0,np.inf)
# w_bds = (0,np.inf)
# l_R_bds = (0,np.inf)
# psi_star_bds = (0,np.inf)
# Y_bds = (0,np.inf)
# keep_feasible = [True]*len(x0)


# bounds = list((*(p_bds,)*p.N,
#           *(w_bds,)*p.N,
#           *(l_R_bds,)*(p.N*(p.S-1)),
#           *(psi_star_bds,)*(p.N*p.N*(p.S-1)),
#           *(Y_bds,)*p.N))
# bounds = optimize.Bounds(lb=1e-10,
#                           ub=np.inf)
bounds = (1e-10,np.inf)
lb = np.full_like(x0, 1e-10)
# lb[p.N*3+p.N*(p.S-1):] = 1
ub = np.full_like(x0, np.inf)
ub[p.N*3:p.N*3+p.N*(p.S-1)] = p.labor.max()
keep_feasible=True
bounds = (lb,ub)
sol = optimize.least_squares(fun = deviation, 
                    x0 = x0, 
                    args = (p,), 
                    bounds = bounds,
                    method= 'trf',
                    loss='arctan',
                    max_nfev=1e6,
                    ftol=1e-14, 
                    xtol=0, 
                    gtol=1e-14,
                    verbose = 2)
# sol = optimize.minimize(fun = deviation, 
#                     x0 = sol.x, 
#                     args = p, 
#                     bounds = bounds,
#                     options={'disp':True,
#                               'maxiter':1e6,
#                               'maxfun':1e6,
#                               })

sol_state = var.var_from_vector(sol.x, p)
sol_state.num_scale_solution(p)

# sol_state.solve_Y(p,Y_init = sol_state.Y)

# simon_sol.solve_Y(p,
#                   Y_init=simon_sol.Y,
#                   price_init=simon_sol.price_indices,
#                   psi_star_init=simon_sol.psi_star, plot_convergence=True)
