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


class parameters:
    def __init__(self):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW']
        N = len(self.countries)
        self.N = N
        self.sectors = ['Non patent', 'Patent']
        S = len(self.sectors)
        self.S = S
        self.labor = np.array(
            [197426230, 379553032, 84991747, 940817540, 124021697, 717517456, 1758243964])
        self.labor = self.labor/self.labor.sum()
        self.kappa = 0.5            #
        self.k = 3                  #
        self.fe = np.ones(S)*0.5  # could be over one
        self.fo = np.ones(S)*0.5  # could be over one
        self.eta = np.ones((N, S))*0.25  # could be over one
        self.zeta = np.ones(S)*0.05
        self.beta = np.array([0.74, 0.26])
        self.beta = self.beta / self.beta.sum()
        self.sigma = np.ones(S)*1.5   #
        self.alpha = np.array([0.5758, 0.3545])
        self.g_0 = 0.01  # makes sense to be low
        self.rho = 0.05  # 0.001 - 0.02
        self.gamma = 0.3            #
        self.nu = np.ones(S)*0.2    #
        self.nu_tilde = self.nu
        self.delta = np.ones((N, S))*0.1
        self.T = np.ones(N)*2  # could be anything >0
        self.tau = 1+2+np.arange(N*N*S).reshape(N, N, S)/50
        for i in range(self.tau.shape[2]):
            np.fill_diagonal(self.tau[:, :, i], 1)
        self.theta = np.ones(S)*8   #
        # self.deficit = np.zeros(N)
        self.deficit = np.array(
            [-650210, 359158, 99389, 170021, 36294, -24930, 10277])/1e3
        # self.deficit = self.deficit/self.deficit.max()
        self.wage_data = np.array(
            [66032, 40395, 55951, 2429, 7189, 1143, 5917])/1e3
        self.price_level_data = np.array(
            [1, 1.09, 1.18, 0.35, 0.44, 0.24, 0.62])

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])


class parameters_julian:
    def __init__(self):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN']
        N = len(self.countries)
        self.N = N
        self.sectors = ['Non patent', 'Patent', 'other']
        S = len(self.sectors)
        self.S = S
        self.eta = np.ones((N, S))*0.02  # could be over one
        self.eta[:, 0] = 0
        self.labor = np.array([10, 20, 30, 40])
        # self.labor = np.array([100,200,300,400])
        self.T = np.ones(N)*0.25  # could be anything >0
        self.k = 1.5                  #
        self.rho = 0.02  # 0.001 - 0.02
        self.alpha = np.ones(S)*0.5
        self.fo = np.ones(S)*2.3  # could be over one
        self.fe = np.ones(S)*2.7  # could be over one
        self.sigma = np.array([3.5, 3, 2.8])
        self.theta = np.array([7, 6, 8])   #
        self.beta = np.ones(S)/S
        self.zeta = np.ones(S)*0.01
        self.g_0 = 0.01
        self.tau = np.ones((N, N, S))*4
        for i in range(self.tau.shape[2]):
            np.fill_diagonal(self.tau[:, :, i], 1)
        self.kappa = 0.5
        self.gamma = 0.4            #
        self.delta = np.ones((N, S))
        self.delta[:, 1] = 0.1
        self.nu = np.array([100000, 0.23, 0.2])
        self.nu_tilde = self.nu/2
        self.deficit = np.zeros(N)
        # self.deficit = np.array([0.1,0.1,-0.1,-0.1])

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])


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
        assert np.isnan(self.g_s).sum() == 0, 'nan in g_s'
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

        assert np.isnan(self.psi_star).sum() == 0, 'nan in psi_star'
        assert np.isnan(self.PSI_M).sum() == 0, 'nan in PSI_M'

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
        assert np.isnan(self.PSI_CL).sum() == 0, 'nan in PSI_CL'

        while np.any(np.einsum('njs->ns', self.PSI_M)+np.einsum('njs->ns', self.PSI_CL) > 1):
            print('correcting PSI_M,CL too high')
            self.PSI_CL = self.PSI_CL/2
            self.PSI_M = self.PSI_M/2

        self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
                         np.einsum('njs->ns', self.PSI_CL))
        assert np.isnan(self.PSI_CD).sum() == 0, 'nan in PSI_CD'

    def compute_phi(self, p):
        self.phi = p.T[None, :, None] * np.einsum('nis,is,is->nis',
                                                  p.tau,
                                                  self.w[:,
                                                         None]**p.alpha[None, :],
                                                  self.price_indices[:,
                                                                     None]**(1-p.alpha[None, :])
                                                  )**(-p.theta[None, None, :])
        assert np.all(self.w > 0), 'non positive wage'
        assert np.all((np.einsum('is,is->is',
                                 self.w[:, None]**p.alpha[None, :],
                                 self.price_indices[:,
                                                    None]**(1-p.alpha[None, :])
                                 )) > 0), 'zero in phi den'
        assert np.isnan(self.phi).sum() == 0, 'nan in phi'
        assert np.all(self.phi > 0), 'negative phi'

    def compute_price_indices(self, p):

        power = (p.sigma-1)/p.theta
        
        numeratorA = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**power[None, None, :]).sum(axis=1)
        numeratorB = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)
        numeratorC = self.PSI_CD*self.phi.sum(axis=1)**power[None, :]
        numerator = (numeratorA + numeratorB + numeratorC)
        # price = ( (numerator/numerator[0,:])**(p.beta[None,:]/(1-p.sigma[None,:])) ).prod(axis = 1)
        # P0 = ( (gamma((p.theta+1-p.sigma)/p.sigma)*numerator[0,:].squeeze())**(p.beta/(1-p.sigma)) ).prod()
        price = ((numerator[0, :]/numerator) **
                 (p.beta[None, :]/(p.sigma[None, :]-1))).prod(axis=1)# * P0
        assert np.isnan(price).sum() == 0, 'nan in price'
        return price

    def solve_price_ind_and_phi_with_price(self, p, price_init=None, tol_p=1e-15, plot_convergence=False):

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

        if plot_convergence:
            print(convergence)
            plt.semilogy(convergence)
            plt.show()

        self.price_indices = price_new
        self.compute_phi(p)

    def compute_monopolistic_sectoral_prices(self, p):
        power = (p.sigma-1)/p.theta
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M*self.phi**power[None, None, :]).sum(axis=1)

        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)

        C = self.PSI_CD*self.phi.sum(axis=1)**power[None, :]

        self.P_M = np.ones((p.N, p.S))
        self.P_M[:, 0] = np.inf
        self.P_M[:, 1:] = (A[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        assert np.isnan(self.P_M).sum() == 0, 'nan in P_M'

    def compute_monopolistic_trade_flows(self, p):
        numerator = np.einsum('nis,nis->nis',
                              self.PSI_M,
                              self.phi**((p.sigma-1)/p.theta)[None, None, :])
        self.X_M = np.zeros((p.N, p.N, p.S))
        self.X_M[..., 1:] = np.einsum('nis,ns,ns,s,n,n->nis',
                                      numerator[..., 1:],
                                      1/(numerator[..., 1:].sum(axis=1)),
                                      self.P_M[..., 1:]**(1-p.sigma[None, 1:]),
                                      p.beta[1:],
                                      self.price_indices,
                                      self.Y
                                      )
        assert np.isnan(self.X_M).sum() == 0, 'nan in X_M'

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
                             self.r+p.zeta[None, :]+p.delta -
                             self.g+self.g_s[None, :],
                             self.r+p.zeta[None, :]+p.nu[None, :] -
                             self.g+self.g_s[None, :]+p.delta
                             )
        psi_star[..., 0] = np.inf

        assert np.isnan(psi_star).sum() == 0, 'nan in psi_star'
        return psi_star

    def solve_psi_star(self, p, psi_star_init=None, price_init=None, tol_psi=1e-15, plot_convergence=False):

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

        while condition:
            # print(count)
            if count != 0:
                # psi_star_new = psi_star_new.ravel()
                # psi_star_old = psi_star_old.ravel()
                # aa_psi_star.apply(psi_star_new, psi_star_old)
                # assert np.isnan(psi_star_new).sum()==0, 'nan in psi_star'
                # psi_star_old = psi_star_new.reshape(p.N,p.N,p.S)
                psi_star_old = psi_star_new

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

        if plot_convergence:
            plt.semilogy(convergence)
            plt.show()

        self.psi_star = psi_star_new
        self.compute_aggregate_qualities(p)
        self.solve_price_ind_and_phi_with_price(
            p, price_init=self.price_indices)
        self.compute_monopolistic_sectoral_prices(p)
        self.compute_monopolistic_trade_flows(p)

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
        assert np.isnan(l_R).sum() == 0, 'nan in l_R'

        return l_R

    def solve_l_R(self, p, l_R_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                  w_init=None, Y_init=None, plot_convergence=False):

        l_R_new = None
        if l_R_init is None:
            l_R_old = np.ones((p.N, p.S))
        else:
            l_R_old = l_R_init
        condition_l_R = True
        count = 0
        convergence_l_R = []
        aa_options_l_R = {'dim': p.N*p.S,
                          'mem': 10,
                          'type1': False,
                          'regularization': 1e-12,
                          'relaxation': 1,
                          'safeguard_factor': 1,
                          'max_weight_norm': 1e6}
        aa_l_R = aa.AndersonAccelerator(**aa_options_l_R)
        while condition_l_R:
            if count != 0:
                l_R_new = l_R_new.ravel()
                l_R_old = l_R_old.ravel()
                aa_l_R.apply(l_R_new, l_R_old)
                # l_R_old = ((l_R_new+l_R_old)/2).reshape(p.N,p.S)
                l_R_old = l_R_new.reshape(p.N, p.S)
            self.guess_labor_research(l_R_old)
            self.compute_growth(p)
            self.solve_psi_star(p, psi_star_init=self.psi_star,
                                price_init=self.price_indices)
            l_R_new = self.compute_labor_research(p)
            l_P = self.compute_labor_allocations(p, l_R=l_R_new, assign=False)
            while np.any(l_P) <= 0:  # !!!!!!
                print('negative production labor, correcting')
                l_R_new = l_R_new/2
            condition_l_R = np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new) > tol_m
            convergence_l_R.append(np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new))
            count += 1

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

        assert np.isnan(l_Ae).sum() == 0, 'nan in l_Ae'
        assert np.isnan(l_Ao).sum() == 0, 'nan in l_Ao'
        assert np.isnan(l_P).sum() == 0, 'nan in l_P'
        assert np.all(l_P > 0), 'non positive production labor'

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
        assert np.isnan(self.P_CL).sum() == 0, 'nan in P_CL'

        self.P_CD = (C/(A+B+C))**(1/(1-p.sigma))[None, :]
        assert np.isnan(self.P_CD).sum() == 0, 'nan in P_CD'

    def compute_competitive_trade_flows(self, p):
        numerator = np.einsum('nis,nis->nis',
                              self.PSI_CL,
                              self.phi**((p.sigma-1)/p.theta)[None, None, :])
        self.X_CL = np.zeros((p.N, p.N, p.S))
        self.X_CL[..., 1:] = np.einsum('nis,ns,ns,s,n,n->nis',
                                       numerator[..., 1:],
                                       1/(numerator[..., 1:].sum(axis=1)),
                                       self.P_CL[...,
                                                 1:]**(1-p.sigma[None, 1:]),
                                       p.beta[1:],
                                       self.price_indices,
                                       self.Y
                                       )
        assert np.isnan(self.X_CL).sum() == 0, 'nan in X_CL'
        self.X_CD = np.einsum('nis,ns,ns,s,n,n->nis',
                              self.phi,
                              1/(self.phi.sum(axis=1)),
                              (self.P_CD**(1-p.sigma[None, :])),
                              p.beta,
                              self.price_indices,
                              self.Y
                              )
        assert np.isnan(self.X_CD).sum() == 0, 'nan in X_CD'

    def compute_wage(self, p):
        wage = (
            p.alpha[None, :] *
            (self.X_CD+self.X_CL +
             (1-1/p.sigma[None, None, :])*self.X_M).sum(axis=0)
        ).sum(axis=1)/self.l_P
        assert np.isnan(wage).sum() == 0, 'nan in wage'
        assert np.all(self.l_P > 0), 'non positive production labor'
        assert np.all(wage > 0), 'non positive wage'
        return wage

    def solve_w(self, p, w_init=None, psi_star_init=None, price_init=None,
                tol_m=1e-8, plot_convergence=False):
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
        while condition_w:
            if count != 0:
                aa_w.apply(w_new, w_old)
                w_old = (w_new+w_old)/2
                # w_old = w_new
                # print(np.linalg.norm(w_old))
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

            if plot_convergence:
                plt.semilogy(convergence_w, label='wage')
                plt.show()

        self.w = w_new

    def compute_expenditure(self, p):
        A = np.einsum('nis->i', self.X_CD+self.X_CL+self.X_M)
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = p.deficit + np.einsum('n,ins->i', self.w, self.l_Ae)
        Y = (A+B-C)/self.price_indices
        assert np.isnan(Y).sum() == 0, 'nan in Y'

        return Y

    def solve_Y(self, p, Y_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                plot_convergence=False):
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
                # Y_old = (Y_new+Y_old)/2
                Y_old = Y_new
            self.guess_expenditure(Y_old)
            self.solve_w(p, w_init=self.w)
            Y_new = self.compute_expenditure(p)
            # print(np.linalg.norm(Y_new))
            condition_Y = np.linalg.norm(
                Y_new - Y_old)/np.linalg.norm(Y_new) > tol_m
            convergence_Y.append(np.linalg.norm(
                Y_new - Y_old)/np.linalg.norm(Y_new))
            count += 1

            if plot_convergence:
                plt.semilogy(convergence_Y, label='Y')
                plt.show()

        self.Y = Y_new

    #TODO def make_dataframes(self):


# p = parameters_julian()
# simon_sol = var()

# j_res = np.array(pd.read_csv(
#     '/Users/simonl/Dropbox/TRIPS/Code/new code/temporary_result_lev_levy.csv', header=None)).squeeze()
# j_res = np.insert(j_res, 0, 1)


# test.solve_l_R(p,plot_convergence=True)
# test.solve_w(p,plot_convergence=True)


#
p = parameters_julian()
simon_sol = var()

j_res = np.array(
    pd.read_csv('/Users/simonl/Dropbox/TRIPS/Code/new code/temporary_result_lev_levy.csv'
                ,header=None)[5])
j_res = np.insert(j_res, 0, 1)
# j_res_2 = np.array(pd.read_csv('/Users/simonl/Dropbox/TRIPS/Code/new code/temporary_result_2.csv',header = None)).squeeze()
# j_res_2 = np.insert(j_res_2,0
julian_sol_w = j_res[p.N:p.N*2]
simon_sol.guess_wage(julian_sol_w)

julian_sol_l_R = j_res[p.N*3:p.N*3+p.N*(p.S-1)].reshape((p.N, p.S-1))
julian_sol_l_R = np.insert(julian_sol_l_R, 0, np.zeros(p.N), axis=1)
simon_sol.guess_labor_research(julian_sol_l_R)

julian_sol_price_indices = j_res[0:p.N]
simon_sol.guess_price_indices(julian_sol_price_indices)

julian_sol_psi_star = j_res[p.N*3+p.N*(p.S-1):].reshape((p.N, p.S-1, p.N))
julian_sol_psi_star = np.transpose(julian_sol_psi_star, (0, 2, 1))
julian_sol_psi_star = np.insert(julian_sol_psi_star, 0, np.zeros(p.N), axis=2)
julian_sol_psi_star[julian_sol_psi_star == 0] = np.inf
simon_sol.guess_patenting_threshold(julian_sol_psi_star)

julian_sol_Y = j_res[p.N*2:p.N*3]*5
simon_sol.solve_Y(p,
                  Y_init=julian_sol_Y,
                  price_init=julian_sol_price_indices,
                  psi_star_init=julian_sol_psi_star, plot_convergence=True)
# Y_ones = np.ones(p.N)*5
# simon_sol.solve_Y(p,
#                   Y_init=Y_ones,
#                   price_init=julian_sol_price_indices,
#                   psi_star_init=julian_sol_psi_star,plot_convergence=True)

# simon_sol.solve_w(p,
#                   w_init=julian_sol_w,
#                   price_init=julian_sol_price_indices,
#                   psi_star_init=julian_sol_psi_star,plot_convergence=True)
# simon_sol.solve_l_R(p,
#                     l_R_init=julian_sol_l_R,
#                     price_init=julian_sol_price_indices,
#                     psi_star_init=julian_sol_psi_star,
#                     w_init=julian_sol_w,
#                     Y_init=julian_sol_Y,plot_convergence=True)
