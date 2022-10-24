#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:39:20 2022

@author: simonl
"""

#%% main classes and dependencies
import numpy as np
from copy import deepcopy
import aa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma
from scipy import optimize
import time
import sys
import os
import matplotlib.colors as mcolors
import seaborn as sns

class parameters:     
    def __init__(self, n=7, s=2):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'][:n]+[i for i in range(n-7)]
        N = len(self.countries)
        self.N = N
        # self.sectors = ['Non patent', 'Patent', 'other', 'other2'][:s]
        self.sectors = ['Non patent', 'Patent']+['other'+str(i) for i in range(s-2)]
        S = len(self.sectors)
        self.S = S
        self.eta = np.ones((N, S))*0.02  # could be over one
        self.eta[:, 0] = 0
        self.T = np.ones((N, S))*1.5  # could be anything >0
        # self.T = np.ones(N)*1.5  # could be anything >0
        self.k = 1.33350683                 #
        self.rho = 0.02  # 0.001 - 0.02
        self.alpha = np.concatenate((np.array([0.5758, 0.3545]),np.ones(s)*0.5))[:s]
        self.fe = np.ones(S)  # could be over one
        self.fo = np.ones(S)*0  # could be over one
        self.sigma = np.ones(S)*1.8125  #
        self.theta = np.ones(S)*5   #
        self.beta = np.concatenate((np.array([0.735, 0.265]),np.ones(s)*0.5))[:s]
        self.beta = self.beta / self.beta.sum()
        # self.zeta = np.ones(S)*0.01
        self.zeta = np.ones(S)*0.01
        self.g_0 = 0.01  # makes sense to be low
        self.kappa = 0.5            #
        self.gamma = 0.5       #
        self.delta = np.ones((N, S))*0.05
        self.nu = np.ones(S)*0.1 #
        self.nu_tilde = np.ones(S)*1
        self.nu_R = np.array(1)
        
        # self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        # self.off_diag_mask[np.s_[::(N+1)*S]] = False
        # self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        # self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        # self.diag_mask = np.invert(self.off_diag_mask)
        
        self.unit = 1e6
        
        self.trade_flows = pd.read_csv('data/country_country_sector_moments.csv',index_col=[1,0,2]).sort_index().values.squeeze()/self.unit
        self.trade_flows = self.trade_flows.reshape((N,N,S))
        self.OUT = self.trade_flows.sum()
        # self.trade_shares = (self.trade_flows/(np.diagonal(self.trade_flows).transpose())[:,None,:])
        self.trade_shares = self.trade_flows/self.trade_flows.sum()
        # self.trade_shares = (self.trade_flows).reshape((N,N,S))
        
        self.data = pd.read_csv('data/country_moments.csv',index_col=[0])
        
        self.labor_raw = np.concatenate(
            (self.data.labor.values,np.ones(n)*self.data.labor.values[-1])
            )[:n]
        
        self.gdp_raw = np.concatenate(
            (self.data.gdp.values,np.ones(n)*self.data.gdp.values[-1])
            )[:n]

        # self.unit_labor = self.labor_raw.mean()
        self.unit_labor = 1e9
        self.labor = self.labor_raw/self.unit_labor
        
        self.deficit_raw = np.concatenate(
            (self.data.deficit.values,np.zeros(n))
              )[:n]
        self.deficit_raw[0] = self.deficit_raw[0]-self.deficit_raw.sum()
        
        self.deficit_share_world_output = self.deficit_raw/self.data.output.sum() 
        
        # self.unit = self.gdp_raw.mean()
        
        
        co = 1e-6
        cou = 1e5
        self.lb_dict = {'sigma':1,
                        'theta':4,
                        'rho':co,
                        'gamma':co,
                        'zeta':0,
                        'nu':0,
                        'nu_tilde':0,
                        'kappa':0,
                        'k':1.1,
                        'fe':co,
                        'fo':0,
                        'delta':1e-2,
                        'g_0':0,
                        'alpha':co,
                         'beta':co,
                         'T':co,
                         'eta':co}
        self.ub_dict = {'sigma':cou,
                        'theta':6,
                        'rho':cou,
                        'gamma':cou,
                        'zeta':1,
                        'nu':cou,
                        'nu_tilde':cou,
                        'kappa':1-cou,
                        'k':1.5,
                        'fe':cou,
                        'fo':cou,
                        'delta':cou,
                        'g_0':cou,
                        'alpha':1,
                         'beta':1,
                         'T':np.inf,
                         'eta':cou}
        
        self.idx = {'sigma':pd.Index(self.sectors, name='sector'),
                    'theta':pd.Index(self.sectors, name='sector'),
                    'rho':pd.Index(['scalar']),
                    'gamma':pd.Index(['scalar']),
                    'zeta':pd.Index(self.sectors, name='sector'),
                    'nu':pd.Index(self.sectors, name='sector'),
                    'nu_tilde':pd.Index(self.sectors, name='sector'),
                    'kappa':pd.Index(['scalar']),
                    'k':pd.Index(['scalar']),
                    # 'tau':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                    #                                  , names=['destination','origin','sector']),
                    'fe':pd.Index(self.sectors, name='sector'),
                    'fo':pd.Index(self.sectors, name='sector'),
                    'delta':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                       , names=['country','sector']),
                    'g_0':pd.Index(['scalar']),
                    'alpha':pd.Index(self.sectors, name='sector'),
                    'beta':pd.Index(self.sectors, name='sector'),
                    'T':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                        , names=['country','sector']),
                     # 'T':pd.Index(self.countries, name = 'country'),
                     'eta':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                      , names=['country','sector'])}
        
        sl_non_calib = {'sigma':[np.s_[0]],
                    'theta':[np.s_[0]],
                    'rho':None,
                    'gamma':None,
                    'zeta':[np.s_[0]],
                    'nu':[np.s_[0]],
                    'nu_tilde':[np.s_[0]],
                    'kappa':None,
                    'k':None,
                    # 'tau':[np.s_[::(N+1)*S],np.s_[1::(N+1)*S]],
                    'fe':[np.s_[0]],
                    'fo':[np.s_[0]],
                    'delta':[np.s_[::S]],#,np.s_[S-1]],
                    'g_0':None,
                    'alpha':None,
                    'beta':None,
                      'T':None,
                     'eta':[np.s_[::S]]}
        
        self.mask = {}
        
        for par_name in ['eta','k','rho','alpha','fe','T','fo','sigma','theta','beta','zeta',
                         'g_0','kappa','gamma','delta','nu','nu_tilde']:
            par = getattr(self,par_name)
            if sl_non_calib[par_name] is not None:
                self.mask[par_name] = np.ones_like(par,bool).ravel()
                for slnc in sl_non_calib[par_name]:
                    self.mask[par_name][slnc] = False
                self.mask[par_name] = self.mask[par_name].reshape(par.shape)    
            else:
                self.mask[par_name] = np.ones_like(par,bool)
        
        self.calib_parameters = None
        self.guess = None
        
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
            
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def get_signature_list(self):
        signature_p = []
        for param in self.calib_parameters: 
            signature_p.extend([param]*np.array(getattr(self,param))[self.mask[param]].size)
        return signature_p
    
    @staticmethod
    def get_list_of_params():
        return ['eta','k','rho','alpha','fe','T','fo','sigma','theta','beta','zeta','g_0',
         'kappa','gamma','delta','nu','nu_tilde']
            
    def guess_from_params(self):
        # price_guess = self.data.price_level.values
        Z_guess = self.data.expenditure.values/self.unit
        w_guess = self.data.gdp.values*self.unit_labor/(self.data.labor.values*self.unit)*100
        l_R_guess = np.repeat(self.labor[:,None]/200, self.S-1, axis=1).ravel()
        psi_star_guess = np.ones((self.N,self.N,(self.S-1))).ravel()*100
        phi_guess = np.ones((self.N,self.N,self.S)).ravel()#*0.01
        vec = np.concatenate((w_guess,Z_guess,l_R_guess,psi_star_guess,phi_guess), axis=0)
        return vec
    
    def make_p_vector(self):
        vec = np.concatenate([np.array(getattr(self,p))[self.mask[p]].ravel() for p in self.calib_parameters])
        return vec
    
    def update_parameters(self,vec):
        idx_from = 0
        for p in self.calib_parameters:
            param = np.array(getattr(self,p))
            size = param[self.mask[p]].size
            param[self.mask[p]] = vec[idx_from:idx_from+size]
            setattr(self,p,param)
            idx_from += size
            
    def compare_two_params(self,p2):
        commonKeys = set(vars(self).keys()) - (set(vars(self).keys()) - set(vars(p2).keys()))
        diffs = []
        for k in commonKeys:
            print(k)
            if (isinstance(vars(self)[k], np.ndarray) or isinstance(vars(self)[k], float)):
                try:
                    if np.all(np.isclose(vars(self)[k], vars(p2)[k])):
                        print(k, 'identical')
                    else:
                        diffs.append(k)
                except:
                    pass
        
        for k in diffs:
            print(k, (np.nanmean(vars(self)[k]/vars(p2)[k])))
    
    def write_params(self,path):
        try:
            os.mkdir(path)
        except:
            pass
        for pa_name in self.get_list_of_params():
            par = getattr(p,pa_name)
            df = pd.DataFrame(data = np.array(par).ravel())
            df.to_csv(path+pa_name+'.csv',header=False)
        try:
            df = pd.DataFrame(data = self.guess)
            df.to_csv(path+'guess.csv',index=False,header=None)
        except:
            pass
        if self.calib_parameters is not None:
            df = pd.DataFrame(data = self.calib_parameters)
            df.to_csv(path+'calib_parameters.csv',index=False,header=None)
        
            
    def load_data(self,path,list_of_params = None):
        if list_of_params is None:
            list_of_params = self.get_list_of_params()
        for pa_name in list_of_params:
            df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
            setattr(self,pa_name,df.values.squeeze().reshape(np.array(getattr(self,pa_name)).shape))
        try:
            df = pd.read_csv(path+'guess.csv',header=None)
            setattr(self,'guess',df.values.squeeze())
        except:
            pass
        try:
            df = pd.read_csv(path+'calib_parameters.csv',header=None)
            setattr(self,'calib_parameters',df.values.squeeze())
        except:
            pass
            
    def make_parameters_bounds(self):
        lb = []
        ub = []
        for par in self.calib_parameters:
            lb.append(np.ones(np.array(getattr(self,par))[self.mask[par]].size)*self.lb_dict[par])
            ub.append(np.ones(np.array(getattr(self,par))[self.mask[par]].size)*self.ub_dict[par])
        return (np.concatenate(lb),np.concatenate(ub))

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
        N = 7
        S = 2
        self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        self.off_diag_mask[np.s_[::(N+1)*S]] = False
        self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        self.diag_mask = np.invert(self.off_diag_mask)

    # def guess_price_indices(self, price_indices_init):
    #     self.price_indices = price_indices_init

    def guess_patenting_threshold(self, psi_star_init):
        self.psi_star = psi_star_init

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_expenditure(self, Z_init):
        self.Z = Z_init

    def guess_labor_research(self, l_R_init):
        self.l_R = l_R_init
    
    def guess_phi(self, phi_init):
        self.phi = phi_init

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        frame = deepcopy(self)
        return frame
    
    @staticmethod
    def var_from_vector(vec,p, compute = True):
        # init = var()    
        # init.guess_price_indices(vec[0:p.N])
        # init.guess_wage(vec[p.N:p.N*2])
        # init.guess_expenditure(vec[p.N*2:p.N*3])
        # init.guess_labor_research(
        #     np.insert(vec[p.N*3:p.N*3+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1))
        # init.guess_patenting_threshold(
        #     np.insert(vec[p.N*3+p.N*(p.S-1):p.N*3+p.N*(p.S-1)+p.N**2].reshape((p.N, p.N, p.S-1)), 0, np.full(p.N,np.inf), axis=2))
        # init.guess_phi(vec[p.N*3+p.N*(p.S-1)+p.N**2:].reshape((p.N, p.N, p.S)))
        # if compute:
        #     init.compute_growth(p)
        #     init.compute_aggregate_qualities(p)
        #     init.compute_monopolistic_sectoral_prices(p)
        #     init.compute_monopolistic_trade_flows(p)
        #     init.compute_competitive_sectoral_prices(p)
        #     init.compute_competitive_trade_flows(p)
        #     init.compute_labor_allocations(p)
        # return init
        init = var()    
        init.guess_wage(vec[0:p.N])
        init.guess_expenditure(vec[p.N:p.N*2])
        init.guess_labor_research(
            np.insert(vec[p.N*2:p.N*2+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1))
        init.guess_patenting_threshold(
            np.insert(vec[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2].reshape((p.N, p.N, p.S-1)), 0, np.full(p.N,np.inf), axis=2))
        init.guess_phi(vec[p.N*2+p.N*(p.S-1)+p.N**2:].reshape((p.N, p.N, p.S)))
        if compute:
            init.compute_growth(p)
            init.compute_aggregate_qualities(p)
            init.compute_sectoral_prices(p)
            init.compute_trade_shares(p)
            init.compute_labor_allocations(p)
            init.compute_price_indices(p)
        return init
    
    def vector_from_var(self):
        # price = self.price_indices
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        psi_star = self.psi_star[...,1:].ravel()
        Z = self.Z
        phi = self.phi.ravel()
        vec = np.concatenate((w,Z,l_R,psi_star,phi), axis=0)
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


    def compute_growth(self, p):
        self.g_s = p.k*np.einsum('is,is -> s',
                                 p.eta,
                                 self.l_R**(1-p.kappa)
                                 )/(p.k-1) - p.zeta
        self.g_s[0] = p.g_0
        # assert np.isnan(self.g_s).sum() == 0, 'nan in g_s'
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()
        self.r = p.rho + self.g/p.gamma
        
    def check_PSI_CL(self,p):
        # status = 'ok'
        # self.psi_star[self.psi_star < 1] = 1
        numerator = ((self.g_s + p.nu + p.zeta)*(self.g_s + p.nu_tilde + p.zeta))
        denominator = (self.g_s[None, 1]+p.delta + p.nu_tilde[None, :]+p.zeta[None, :]) \
            * (self.g_s[None, :]+p.delta + p.nu_tilde[None, :]+p.zeta[None, :])
        psi_star_lim = ( (numerator[None, :]/denominator)**(1/(p.k-1)) )[:,None,:]
        psi_star_lim = np.broadcast_to(psi_star_lim,self.psi_star.shape)
        # if np.any(self.psi_star < psi_star_lim):
        #     status = 'corrected'
        #     print('corrected PSI', (self.psi_star[...,1] < psi_star_lim).sum(),self.psi_star.min())
        #     self.psi_star[...,1][self.psi_star[...,1]<psi_star_lim] = \
        #         np.broadcast_to(psi_star_lim,self.psi_star[...,1].shape)[self.psi_star[...,1]<psi_star_lim] + \
                #np.broadcast_to(psi_star_lim,self.psi_star[...,1].shape)[self.psi_star[...,1]<psi_star_lim]/100 
        if np.any(self.psi_star < psi_star_lim):
            print('corrected PSI', (self.psi_star < psi_star_lim).sum(),(self.psi_star - psi_star_lim).min())
            # self.psi_star[...,1][self.psi_star[...,1]<psi_star_lim] = self.psi_star[...,1][self.psi_star[...,1]<psi_star_lim]*2
            self.psi_star[self.psi_star<psi_star_lim] = psi_star_lim[self.psi_star<psi_star_lim]
            print('post_correction', np.any(self.psi_star < psi_star_lim))
        # return status
        
    
    def check_PSI_CD(self,p):
        self.compute_aggregate_qualities(p)
        while np.any(self.PSI_CD<=0):
            plt.plot(self.PSI_CD[...,1].ravel(),label = 'psi CD')
            plt.plot(self.PSI_CL[...,1].sum(axis=1).ravel(),label = 'psi CL')
            plt.plot(self.PSI_M[...,1].sum(axis=1).ravel(),label = 'psi M')
            plt.legend()
            plt.show()
            time.sleep(5)
            print('correcting PSI_CD')
            self.l_R = self.l_R/2
            self.compute_aggregate_qualities(p)
            plt.plot(self.PSI_CD[...,1].ravel(),label = 'psi CD')
            plt.plot(self.PSI_CL[...,1].sum(axis=1).ravel(),label = 'psi CL')
            plt.plot(self.PSI_M[...,1].sum(axis=1).ravel(),label = 'psi M')
            plt.legend()
            plt.title('corrected')
            plt.show()
            time.sleep(5)
    
    # def check_labor(self,p):
    #     status = 'ok'
    #     correction_count_1 = 0
    #     correction_count = 0
    #     l_r = self.l_R[...,1]        
    #     temp = np.einsum('i,i,ni->ni',
    #                     p.eta[...,1],
    #                     l_r,
    #                     self.psi_star[...,1]**(-p.k))
    #     condition1 = temp > p.labor[None,:]
    #     condition2 = temp > p.labor[:,None]
    #     while condition1.any() or condition2.any():
    #         l_r = l_r/2
    #         condition1 = temp > p.labor[None,:]
    #         condition2 = temp > p.labor[:,None]
    #         correction_count += 1
    #         print('corrected labor 1', correction_count1)
        
    #     bound = p.labor - p.fe[1]*temp.sum(axis=0) - p.fo[1]*temp.sum(axis=1)
    #     condition = l_r > 0.9*bound
    #     while np.any(condition):
    #         # l_r[condition] = 0.9*bound[condition]
    #         l_r = l_r/2
    #         temp = np.einsum('i,i,ni->ni',
    #                         p.eta[...,1],
    #                         l_r,
    #                         self.psi_star[...,1]**(-p.k))
    #         bound = p.labor - p.fe[1]*temp.sum(axis=0) - p.fo[1]*temp.sum(axis=1)
    #         condition = l_r[...,1] > 0.9*bound
    #         correction_count += 1
    #         print('corrected labor 2', correction_count)
    #         status = 'corrected labor'
            
    #     if np.any(l_r<0):
    #         l_r[l_r<0] = 0
    #         status = 'corrected labor'
            
    #     self.l_R[...,1] = l_r
    #     return status
        

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
        # self.PSI_CL = np.maximum(np.einsum('is,s,nis -> nis',
        #                         prefact,
        #                         1/A_tilde,
        #                         (p.nu/A)[None, None:]-B_tilde*A_tilde[None, None, :]),0)
        # print(self.psi_star[...,1].min())
        self.PSI_CL[:, :, 0] = 0
        self.PSI_CL[self.PSI_CL<0] = 0
        # print(self.PSI_CL.min())
        # assert np.isnan(self.PSI_CL).sum() == 0, 'nan in PSI_CL'
        # assert not np.any(np.einsum('njs->ns', self.PSI_M)+np.einsum('njs->ns', self.PSI_CL) > 1),'PSI_M,CL too high'
        # print((np.einsum('njs->ns', self.PSI_M) +
        #                  np.einsum('njs->ns', self.PSI_CL)).max())
        self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
                         np.einsum('njs->ns', self.PSI_CL))
        # self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
        #                  np.einsum('njs->ns', self.PSI_CL))
        # print(self.PSI_CD.min())
        # assert np.isnan(self.PSI_CD).sum() == 0, 'nan in PSI_CD'

    def compute_sectoral_prices(self, p):
        power = p.sigma-1
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M*self.phi**power[None, None, :]).sum(axis=1)

        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)

        C = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]

        self.P_M = np.ones((p.N, p.S))
        self.P_M[:, 0] = np.inf
        self.P_M[:, 1:] = (A[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        
        self.P_CL = np.ones((p.N, p.S))
        self.P_CL[:, 0] = np.inf

        self.P_CL[:, 1:] = (B[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]

        self.P_CD = (C/(A+B+C))**(1/(1-p.sigma))[None, :]

    def compute_trade_shares(self, p):
        # numerator_prefact_A = np.einsum('nis,nis->nis',
        #                       self.PSI_M,
        #                       self.phi**(p.sigma-1)[None, None, :])
        denominator_M = np.zeros((p.N, p.N, p.S))
        denominator_M[..., 1:] = np.einsum('nis,ns,ns->nis',
                                self.PSI_M[..., 1:],
                                1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CL = np.zeros((p.N, p.N, p.S))
        denominator_CL[..., 1:] = np.einsum('nis,ns,ns->nis',
                                self.PSI_CL[..., 1:],
                                1/((self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_CL[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CD = np.einsum('nis,ns,ns->nis',
                                   self.phi**(p.theta-(p.sigma-1))[None,None,:],
                                   1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                   self.P_CD**(1-p.sigma[None,:])
                                   )
        self.X_M = denominator_M/(denominator_M + denominator_CL + denominator_CD)
        self.X_CL = denominator_CL/(denominator_M + denominator_CL + denominator_CD)
        self.X_CD = denominator_CD/(denominator_M + denominator_CL + denominator_CD)

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
                         1/p.fe   #!!!!
                         )
        l_P = p.labor - np.einsum('is->i',
                                  np.einsum('ins->is', l_Ao)+np.einsum('nis->is',l_Ae)+l_R)
        
        # l_P_s = p.alpha*(self.X.sum(axis=0)-self.X_M.sum(axis=0)/p.sigma[None,:])/self.w[:,None]
        # l_P = l_P_s.sum(axis=1)
        # assert np.isnan(l_Ae).sum() == 0, 'nan in l_Ae'
        # assert np.isnan(l_Ao).sum() == 0, 'nan in l_Ao'
        # assert np.isnan(l_P).sum() == 0, 'nan in l_P'
        # if np.any(l_P<0):
        #     print(l_R)
        #     print(l_P)
        #     print(p.labor)
        if assign:
            self.l_Ae = l_Ae
            self.l_Ao = l_Ao
            self.l_P = l_P
            # try:
            #     self.l_P_s = l_P_s
            # except:
            #     pass
        else:
            return l_P

    def compute_wage(self, p):
        wage = (p.alpha[None, :] * (p.trade_shares*self.Z.sum()*(1 - (1/p.sigma[None, None, :])*self.X_M)).sum(axis=0)
                ).sum(axis=1)/self.l_P
        # assert np.isnan(wage).sum() == 0, 'nan in wage'
        # assert np.all(self.l_P > 0), 'non positive production labor'
        # assert np.all(wage > 0), 'non positive wage'
        return wage

    def compute_expenditure(self, p):
        A = np.einsum('nis->i', p.trade_shares*self.Z.sum())
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = np.einsum('i,n->i', p.deficit_share_world_output, self.Z)
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Z = (A+B-(C+D))
        # assert np.isnan(Z).sum() == 0, 'nan in Z'
        return Z
    
            
    def compute_labor_research(self, p):
        A = np.einsum('nis,nis,s,i,nis,s->is',
                      p.trade_shares*self.Z.sum(),
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

    def compute_psi_star(self, p):
        psi_star = np.einsum('s,i,nis,s,nis,nis,nis,ns,ns -> nis',
                             p.sigma,
                             self.w,
                             self.PSI_M,
                             1/p.nu,
                             np.divide(1, self.X_M, out=np.full_like(
                                 self.X_M, np.inf), where=self.X_M != 0),
                             1/(p.trade_shares*self.Z.sum()),
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
        
    def compute_phi(self, p):
        denominator_M = np.zeros((p.N, p.N, p.S))
        denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                self.PSI_M[..., 1:],
                                self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CL = np.zeros((p.N, p.N, p.S))
        denominator_CL[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                self.PSI_CL[..., 1:],
                                self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                1/((self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_CL[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CD = np.einsum('ns,ns->ns',
                                   1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                   self.P_CD**(1-p.sigma[None,:])
                                   )
        one_over_denominator = 1/(denominator_M + denominator_CL + denominator_CD[:,None,:])
        phi = np.einsum('nis,,s,n,nis->nis',
                        p.trade_shares,
                        self.Z.sum(),
                        1/p.beta,
                        1/self.Z,
                        one_over_denominator)**(1/p.theta)[None,None,:]
        
        phi = np.einsum('nis,ns,ns,ns,ns->nis',
                phi,
                1/np.diagonal(phi).transpose(),
                p.T**(1/p.theta[None,:]),
                self.w[:,None]**(-p.alpha[None,:]),
                self.price_indices[:,None]**(p.alpha[None,:]-1))
        
        return phi
    
    def compute_price_indices(self, p, assign = True):
        power = (p.sigma-1)
        
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**power[None, None, :]).sum(axis=1)
        B = (self.PSI_CL*self.phi**power[None, None, :]).sum(axis=1)
        C = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]
        price_indices = ( (gamma((p.theta+1-p.sigma)/p.sigma)*(A+B+C))**(p.beta[None, :]/(1- p.sigma[None, :])) ).prod(axis=1)
        if assign:
            self.price_indices = price_indices
        else:
            return price_indices
    
    # def compute_tau(self,p, assign = True, price_to_compute = None):
    #     # self.tau = np.ones((p.N,p.N,p.S))
    #     if price_to_compute is None:
    #         price_to_compute = self.price_indices
    #     if p.T.size == p.N:
    #         tau = np.einsum('nis,is,is,is->nis',
    #                         1/self.phi,
    #                         p.T[:,None]**(1/p.theta[None,:]),
    #                         self.w[:,None]**(-p.alpha[None,:]),
    #                         price_to_compute[:,None]**(p.alpha[None,:]-1))
    #     else:
    #         tau = np.einsum('nis,is,is,is->nis',
    #                         1/self.phi,
    #                         p.T**(1/p.theta[None,:]),
    #                         self.w[:,None]**(-p.alpha[None,:]),
    #                         price_to_compute[:,None]**(p.alpha[None,:]-1))
    #     if assign:
    #         self.tau = tau
    #     else:
    #         return tau
    
    def compute_tau(self,p, assign = True):
        # self.tau = np.ones((p.N,p.N,p.S))
        tau = np.diagonal(self.phi).transpose()[None,:,:]/self.phi
        if assign:
            self.tau = tau
        else:
            return tau
        
    # def compute_T(self,p, assign = True):
    #     # self.tau = np.ones((p.N,p.N,p.S))
    #     T = np.diagonal(self.phi).transpose()*self.w[:,None]**p.alpha[None,:]*self.w[:,None]**(1-p.alpha[None,:])
    #     if assign:
    #         self.T = T
    #     else:
    #         return T
    
    # def scale_tau(self, p):
    #     try:
    #         price_to_compute = self.compute_price_indices(p, assign = False)
    #     except:
    #         self.compute_growth(p)
    #         self.compute_aggregate_qualities(p)
    #         price_to_compute = self.compute_price_indices(p, assign = False)
            
    #     tau = self.compute_tau(p, assign = False, price_to_compute = price_to_compute)

    #     diag_tau = np.diagonal(tau).transpose()
    #     # B = np.ones((p.N,p.S))
    #     # B[:,0] = ( diag_tau[:,0]**(1+p.beta[1]*(p.alpha[1]-1)) / diag_tau[:,1]**(p.beta[1]*(p.alpha[0]-1)) )**(1/(p.alpha*p.beta).sum())
    #     # B[:,1] = ( diag_tau[:,1]**(1+p.beta[0]*(p.alpha[0]-1)) / diag_tau[:,0]**(p.beta[0]*(p.alpha[1]-1)) )**(1/(p.alpha*p.beta).sum())
        
    #     B = diag_tau * ( (diag_tau**p.beta[None,:]).prod(axis=1))[:,None]**((1-p.alpha[None,:])/(p.alpha*p.beta).sum() )
        
    #     self.phi = self.phi*B[:,None,:]
    #     self.compute_sectoral_prices(p)
    #     self.compute_trade_flows(p)
    
    def scale_P(self, p):
        try:
            price_to_compute = self.compute_price_indices(p, assign = False)
        except:
            self.compute_growth(p)
            self.compute_aggregate_qualities(p)
            price_to_compute = self.compute_price_indices(p, assign = False)

        numeraire = price_to_compute[0]
        
        self.w = self.w / numeraire
        self.Z = self.Z / numeraire
        self.phi = self.phi * numeraire
        # p.trade_flows = p.trade_flows / numeraire
        
        self.compute_sectoral_prices(p)
        # self.compute_trade_shares(p)
        
    def scale_Z(self,p):
        fact = p.OUT/self.Z.sum()
        self.w = self.w * fact
        self.Z = self.Z * fact
        
        self.compute_trade_flows(p)
    
    def compute_nominal_value_added(self,p):
        self.nominal_value_added = p.alpha[None, :]*(p.trade_shares*self.Z.sum()*(1-self.X_M/p.sigma[None, None, :])).sum(axis=0)
    
    def compute_nominal_intermediate_input(self,p):
        self.nominal_intermediate_input = np.einsum('s,is->is',
                           (1-p.alpha)/p.alpha,
                           self.nominal_value_added)
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - self.nominal_intermediate_input.sum(axis=1)
        self.cons = self.nominal_final_consumption/self.price_indices
        
    def compute_gdp(self,p):
        self.gdp = self.nominal_final_consumption + p.deficit_share_world_output*self.Z.sum() + self.w*(p.labor - self.l_P)
    
    def compute_profit(self,p):
        self.profit = np.einsum('nis,nis,s,i,nis->nis',
                                self.X_M,
                                p.trade_shares*self.Z.sum(),
                                1/p.sigma,
                                1/self.w,
                                np.divide(1, self.PSI_M, out=np.zeros_like(
                                    self.PSI_M), where=self.PSI_M != 0))
    
    def compute_pflow(self,p):
        self.pflow = np.einsum('nis,is,is->nis',
                              self.psi_star[...,1:]**(-p.k),
                              p.eta[...,1:],
                              self.l_R[...,1:]**(1-p.k)
                              ).squeeze()
        
    def compute_share_of_innovations_patented(self,p):
        self.share_innov_patented = self.psi_star[...,1]**(-p.k)
    
    def compute_welfare(self,p):
        exp = 1-1/p.gamma
        self.U = self.cons**(exp)/(p.rho-self.g*exp)/exp
    
    def compute_consumption_equivalent_welfare(self,p,baseline):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline.g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline.cons
    
    def compute_non_solver_quantities(self,p):
        self.compute_price_indices(p)
        self.compute_tau(p)
        # self.compute_T(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_profit(p)
        self.compute_pflow(p)      
        self.compute_share_of_innovations_patented(p)
        self.compute_welfare(p)

    def check_solution(self, p, return_checking_copy = False, assertions = True):
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
        check.Z = check.compute_expenditure(p)
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
                np.isclose(check.Z, check.compute_expenditure(p))
                ), 'check Z wrong'
            print('is a solution')
        if return_checking_copy:
            return check

def remove_diag(A):
    removed = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], int(A.shape[0])-1, -1)
    return np.squeeze(removed)
    
class moments:
    def __init__(self,list_of_moments = None, n=7, s=2):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'][:n]+[i for i in range(n-7)]
        self.sectors = ['Non patent', 'Patent']+['other'+str(i) for i in range(s-2)]
        if list_of_moments is None:
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'OUT', 'KM', 'RD', 'RP',
                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                               'SINNOVPATEU']
        else:
            self.list_of_moments = list_of_moments
        self.weights_dict = {'GPDIFF':1, 
                             'GROWTH':1, 
                             'KM':5, 
                             'OUT':4, 
                             'RD':3, 
                             'RP':3, 
                             'SPFLOW':1, 
                             'SRDUS':1, 
                             'SRGDP':1, 
                             # 'STFLOW':1,
                             'JUPCOST':1,
                              'TP':1,
                              'Z':1,
                             # 'SDOMTFLOW':1,
                             'SINNOVPATEU':1,
                              'NUR':1
                             }
        
        # self.total_weight = sum([self.weights_dict[mom] for mom in self.list_of_moments])
        
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'OUT':pd.Index(['scalar']), 
                    'RD':pd.Index(self.countries, name='country'), 
                    'RP':pd.Index(self.countries, name='country'), 
                    'SPFLOW':pd.MultiIndex.from_tuples([(c1,c2) for c1 in self.countries for c2 in self.countries if c1 != c2]
                                            , names=['destination','origin']),
                    # 'SPFLOW':pd.MultiIndex.from_product([self.countries,self.countries]
                    #                                   , names=['destination','origin']),
                    'SRDUS':pd.Index(['scalar']), 
                    'JUPCOST':pd.Index(['scalar']), 
                    'SRGDP':pd.Index(self.countries, name='country'), 
                    # 'STFLOW':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                    #                                  , names=['destination','origin','sector']),
                    'TP':pd.Index(['scalar']),
                    'NUR':pd.Index(['scalar']),
                    'Z':pd.Index(self.countries, name='country'),
                    # 'SDOMTFLOW':pd.MultiIndex.from_product([self.countries,self.sectors]
                    #                                  , names=['country','sector']),
                    'SINNOVPATEU':pd.Index(['scalar']),
                    'NUR':pd.Index(['scalar'])}
    
    def get_signature_list(self):
        l = []
        for mom in self.list_of_moments:
            l.extend([mom]*np.array(getattr(self,mom)).size)
        return l
    
    @staticmethod
    def get_list_of_moments():
        # return ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SPFLOW', 
        #         'SRDUS', 'SRGDP', 'STFLOW', 'JUPCOST', 'TP', 'Z', 'SDOMTFLOW', 
        #         'SINNOVPATEU']
        return ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SPFLOW', 
                'SRDUS', 'SRGDP', 'JUPCOST', 'TP', 'Z', 
                'SINNOVPATEU','NUR']
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
    
    def load_data(self):
        self.c_moments = pd.read_csv('data/country_moments.csv',index_col=[0])
        self.cc_moments = pd.read_csv('data/country_country_moments.csv',index_col=[1,0]).sort_index()
        self.ccs_moments = pd.read_csv('data/country_country_sector_moments.csv',index_col=[1,0,2]).sort_index()
        self.moments = pd.read_csv('data/scalar_moments.csv',index_col=[0])
        self.description = pd.read_csv('data/moments_descriptions.csv',sep=';',index_col=[0])
        
        N = len(self.ccs_moments.index.get_level_values(0).drop_duplicates())
        S = len(self.ccs_moments.index.get_level_values(2).drop_duplicates())
        self.unit = 1e6
        
        self.STFLOW_target = (self.ccs_moments.trade/
                              self.ccs_moments.trade.sum()).values.reshape(N,N,S)
        self.SPFLOW_target = self.cc_moments.query("destination_code != origin_code")['patent flows'].values
        self.SPFLOW_target = self.SPFLOW_target.reshape((N,N-1))/self.SPFLOW_target.sum()
        # self.SPFLOW_target = self.cc_moments['patent flows'].values
        # self.SPFLOW_target = self.SPFLOW_target.reshape((N,N))/self.SPFLOW_target.sum()
        self.OUT_target = self.c_moments.expenditure.sum()/self.unit
        self.SRGDP_target = (self.c_moments.gdp/self.c_moments.price_level).values \
                            /(self.c_moments.gdp/self.c_moments.price_level).sum()
        self.RP_target = self.c_moments.price_level.values
        self.RD_target = self.c_moments.rnd_gdp.values
        self.KM_target = self.moments.loc['KM'].value
        self.NUR_target = self.moments.loc['NUR'].value
        self.SRDUS_target = self.moments.loc['SRDUS'].value
        self.GPDIFF_target = self.moments.loc['GPDIFF'].value 
        self.GROWTH_target = self.moments.loc['GROWTH'].value 
        # self.GROWTH_target = self.GROWTH_target*10
        self.Z_target = self.c_moments.expenditure.values/self.unit
        self.JUPCOST_target = self.moments.loc['JUPCOST'].value
        self.TP_target = self.moments.loc['TP'].value
        self.SINNOVPATEU_target = self.moments.loc['SINNOVPATEU'].value
        self.SDOMTFLOW_target = self.ccs_moments.query("destination_code == origin_code").trade.values#/self.ccs_moments.trade.sum()
        self.SDOMTFLOW_target = self.SDOMTFLOW_target.reshape(N,S)/self.unit
    
    def plot_moments(self, list_of_moments, plot = True, save_plot = None):
        scalar_moments = []
        scalar_moments_ratio = []
        for mom in list_of_moments:
            if np.array(getattr(self,mom)).size == 1:
                print(mom+' : ',getattr(self,mom),
                      mom+' target  : ',
                      getattr(self,mom+'_target'))
                scalar_moments.append(mom)
                # scalar_moments_ratio.append(getattr(self,mom)/getattr(self,mom+'_target'))
                scalar_moments_ratio.append(getattr(self,mom+'_deviation'))
            else:
                if plot == True:
                    if mom != 'STFLOW' and mom != 'SDOMTFLOW':
                        fig,ax = plt.subplots(figsize = (12,8))
                        ax.scatter(getattr(self,mom+'_target').ravel(),getattr(self,mom).ravel())
                        ax.plot([0,
                                  getattr(self,mom+'_target').max()]
                                ,[0,
                                  getattr(self,mom+'_target').max()])
                        ax.set_xlabel('target')
                        if mom != 'SPFLOW' :
                            texts = [plt.text(getattr(self,mom+'_target')[i],getattr(self,mom)[i],idx) 
                                     for i,idx in enumerate(self.idx[mom])]   
                        else:
                            texts = [plt.text(getattr(self,mom+'_target').ravel()[i],getattr(self,mom).ravel()[i],idx) 
                                     for i,idx in enumerate(self.idx[mom])]  
                            text = plt.text(getattr(self,mom+'_target').min(),getattr(self,mom).max(),'(Destination,Origin)')
                        
                        plt.title(mom+' targeting')
                        plt.yscale('log')
                        plt.xscale('log')
                        if save_plot is not None:
                            plt.savefig(save_plot+'_'+mom)
                        plt.show()
                    elif mom == 'STFLOW':
                        fig,ax = plt.subplots(figsize = (12,8))
                        ax.scatter(getattr(self,mom+'_target')[...,0].ravel(),
                                   getattr(self,mom)[...,0].ravel(),
                                   label = 'Non patenting sector')
                        ax.plot([0,
                                  getattr(self,mom+'_target').max()]
                                ,[0,
                                  getattr(self,mom+'_target').max()])
                        ax.scatter(getattr(self,mom+'_target')[...,1].ravel(),
                                   getattr(self,mom)[...,1].ravel(),
                                   label = 'Patenting sector')
                        ax.set_xlabel('target')
                        plt.legend()
                        plt.title(mom+' targeting')
                        plt.yscale('log')
                        plt.xscale('log')
                        if save_plot is not None:
                            plt.savefig(save_plot+'_'+mom)
                        plt.show()
                    
                    elif mom == 'SDOMTFLOW':
                        fig,ax = plt.subplots(figsize = (12,8))
                        ax.scatter(getattr(self,mom+'_target')[...,0].ravel(),
                                   getattr(self,mom)[...,0].ravel(),
                                   label = 'Non patenting sector')
                        ax.plot([0,
                                  getattr(self,mom+'_target').max()]
                                ,[0,
                                  getattr(self,mom+'_target').max()])
                        ax.scatter(getattr(self,mom+'_target')[...,1].ravel(),
                                   getattr(self,mom)[...,1].ravel(),
                                   label = 'Patenting sector')
                        ax.set_xlabel('target')
                        texts = [plt.text(getattr(self,mom+'_target').ravel()[i],getattr(self,mom).ravel()[i],idx[0]) 
                                 for i,idx in enumerate(self.idx[mom])]  
                        plt.legend()
                        plt.title(mom+' targeting')
                        plt.yscale('log')
                        plt.xscale('log')
                        if save_plot is not None:
                            plt.savefig(save_plot+'_'+mom)
                        plt.show()
               
        fig,ax = plt.subplots(figsize = (12,8))
        ax.scatter(scalar_moments,scalar_moments_ratio)
        ax.plot(scalar_moments,np.zeros_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        # if np.any(np.array(scalar_moments_ratio)>10):
        #     plt.yscale('log')
        plt.title('scalar moments, deviation')
        if save_plot is not None:
            plt.savefig(save_plot+'_scalar_moments')
        plt.show()
            
    def write_moments(self, path, list_of_moments):
        for mom in list_of_moments:
            df = pd.DataFrame(data = {'target':getattr(m,mom+'_target').ravel(),
                                      'moment':getattr(m,mom).ravel()})
            df.to_csv(path+mom+'.csv',index=False)
        df = pd.DataFrame(data = self.list_of_moments)
        df.to_csv(path+'list_of_moments.csv',index=False)
    
    @staticmethod
    def compare_moments(dic = None, lis = None, save_path = None):
        if dic is None and lis is not None:
            coms = ['m'+str(i) for i,_ in enumerate(lis)]
            moms_c = lis
        elif lis is None and dic is not None:
            coms = [k for k in dic.keys()]
            moms_c = [v for v in dic.values()]
        scalar_moments_collection = [ [] for _ in range(len(coms)) ]
        scalar_moments = []
        for mom in moms_c[0].list_of_moments:
            print(mom)
            if np.array(getattr(moms_c[0],mom)).size == 1:
                scalar_moments.append(mom)
                print('\n'+mom+' target :',getattr(moms_c[0],mom+'_target'))
                for i,mom_c in enumerate(moms_c):
                    # scalar_moments_collection[i].append(getattr(mom_c,mom)/getattr(mom_c,mom+'_target'))
                    scalar_moments_collection[i].append(getattr(mom_c,mom+'_deviation')**2)
                    print(coms[i]+' : ',getattr(mom_c,mom))
            else:
                fig,ax = plt.subplots(figsize = (12,8))
                for i,mom_c in enumerate(moms_c):
                    ax.scatter(getattr(mom_c,mom+'_target').ravel(),getattr(mom_c,mom).ravel()
                               ,label = coms[i],lw=2,marker = 'x')
                ax.plot([getattr(mom_c,mom+'_target').min(),
                          getattr(mom_c,mom+'_target').max()]
                        ,[getattr(mom_c,mom+'_target').min(),
                          getattr(mom_c,mom+'_target').max()], 
                        ls = '--', lw=1, color = 'k')
                if mom != 'SPFLOW' and mom != 'STFLOW' and mom != 'SDOMTFLOW':
                    texts = [plt.text(getattr(mom_c,mom+'_target')[i],getattr(mom_c,mom)[i],idx) 
                              for i,idx in enumerate(mom_c.idx[mom])]
                # ax.plot([0,
                #           getattr(mom_c,mom+'_target').max()]
                #         ,[0,
                #           getattr(mom_c,mom+'_target').max()], ls = '--', lw=0.5)
                ax.set_xlabel('target')
                plt.title(mom+' targeting')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                if save_path is not None:
                    plt.savefig(save_path+mom)
                plt.show()
        fig,ax = plt.subplots(figsize = (12,8))
        for i,scalars_of_one_run in enumerate(scalar_moments_collection):
            ax.scatter(scalar_moments,scalars_of_one_run, label = coms[i])
        # ax.plot(scalar_moments,np.ones_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        ax.plot(scalar_moments,np.zeros_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        # if np.any([np.any(np.array(scalars_of_one_run)>10) 
        #            for scalars_of_one_run in scalar_moments_collection]):
        #     plt.yscale('log')
        plt.title('scalar moments deviation')
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path+'scalar_moments')
        plt.show()
        
    # def compute_STFLOW(self,var,p):
    #     self.STFLOW = (var.X_M+var.X_CL+var.X_CD)/var.Z.sum()
        
    def compute_SPFLOW(self,var,p):
        # numerator = np.einsum('nis,is,is->nis',
        #                       var.psi_star[...,1:]**(-p.k),
        #                       p.eta[...,1:],
        #                       var.l_R[...,1:]**(1-p.k)
        #                       )
        # numerator = remove_diag(numerator)
        pflow = remove_diag(var.pflow)
        # pflow = var.pflow
        self.SPFLOW = pflow/pflow.sum()
        
    def compute_OUT(self,var,p):
        self.OUT = var.Z.sum()
        
    def compute_SRGDP(self,var,p):
        numerator = var.gdp/var.price_indices
        self.SRGDP = numerator/numerator.sum()
        
    def compute_RP(self,var,p):
        self.RP = var.price_indices/var.price_indices[0]
        
    def compute_RD(self,var,p):
        numerator = var.w[:,None]*var.l_R + np.einsum('i,ins->is',var.w,var.l_Ao)\
            + np.einsum('n,ins->is',var.w,var.l_Ao)
        self.RD = np.einsum('is,i->i',
                            numerator,
                            1/var.gdp)
    
    def compute_KM(self,var,p):
        denominatorA = var.r+p.zeta[None,1:]+p.delta[:,1:]-var.g+var.g_s[None,1:]
        bracket = 1/denominatorA - 1/(denominatorA+p.nu[None,1:])
        self.KM = p.k/(p.k-1)*np.einsum('s,s,ns,ns,ns->',
            p.eta[0,1:],
            var.l_R[0,1:]**(1-p.kappa),
            var.psi_star[:,0,1:]**(1-p.k),
            var.profit[:,0,1:],
            bracket,
            )/var.l_R[0,1:].sum()
        
    def compute_SRDUS(self,var,p):
        self.SRDUS = (var.X_M[:,0,1]*p.trade_shares[:,0,1]*var.Z.sum()).sum()/(p.trade_shares[:,0,1]*var.Z.sum()).sum()
    
    def compute_GPDIFF(self,var,p):
        price_index_growth_rate = var.g_s/(1-p.sigma)+p.alpha*var.g
        self.GPDIFF = price_index_growth_rate[0] - price_index_growth_rate[1]
        
    def compute_GROWTH(self,var,p):
        self.GROWTH = var.g    
    
    def compute_Z(self,var,p):
        self.Z = var.Z
    
    def compute_JUPCOST(self,var,p):
        self.JUPCOST = var.pflow[2,0]*(p.fo[1]*var.w[0] + p.fe[1]*var.w[2])
        
    def compute_TP(self,var,p):
        self.TP = var.pflow.sum()
        
    # def compute_SDOMTFLOW(self,var,p):
    #     self.SDOMTFLOW = np.diagonal(var.X).transpose()#/var.X.sum()
    
    def compute_SINNOVPATEU(self,var,p):
        self.SINNOVPATEU = var.share_innov_patented[1,1]
        
    def compute_NUR(self,var,p):
        self.NUR = p.nu[1]
        
    def compute_moments(self,var,p):
        # self.compute_STFLOW(var, p)
        self.compute_SPFLOW(var, p)
        self.compute_OUT(var, p)
        self.compute_SRGDP(var, p)
        self.compute_RP(var, p)
        self.compute_RD(var, p)
        self.compute_KM(var, p)
        self.compute_SRDUS(var, p)
        self.compute_GPDIFF(var, p)
        self.compute_GROWTH(var, p)
        self.compute_JUPCOST(var, p)
        self.compute_TP(var,p)
        self.compute_Z(var,p)
        # self.compute_SDOMTFLOW(var,p)
        self.compute_SINNOVPATEU(var,p)
        self.compute_NUR(var,p)
        
    def compute_moments_deviations(self):
        # for mom in self.get_list_of_moments():
        #     setattr(self,
        #             mom+'_deviation',
        #             self.weights_dict[mom]*((getattr(self,mom) - getattr(self,mom+'_target'))
        #                     /(getattr(self,mom+'_target')*np.sqrt(getattr(self,mom+'_target').size)))
        #             )
        # for mom in self.get_list_of_moments():
        #     setattr(self,
        #             mom+'_deviation',
        #             self.weights_dict[mom]*np.log(np.abs(getattr(self,mom)/getattr(self,mom+'_target')))
        #             /(getattr(self,mom+'_target').size)
        #             )
        for mom in self.get_list_of_moments():
            if mom != 'GPDIFF':
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.log(np.abs(getattr(self,mom)/getattr(self,mom+'_target')))
                #         /np.log(getattr(self,mom+'_target').size+1)
                #         )
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))**(1/2)
                #         )
                setattr(self,
                        mom+'_deviation',
                        self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                        /getattr(self,mom+'_target').size**(1/2)
                        )
            else:
                mo = getattr(self,mom)
                tar = getattr(self,mom+'_target')
                # i = 5 #
                # while mo/(i*tar)+(i-1)/i <= 0:
                #     i += 1
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.log(mo/(i*tar)+(i-1)/i)
                #         /np.log(getattr(self,mom+'_target').size+1)
                #         )
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.log(mo/(i*tar)+(i-1)/i)
                #         )
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.log(mo/(i*tar)+(i-1)/i)
                #         /getattr(self,mom+'_target').size
                #         )
                setattr(self,
                        mom+'_deviation',
                        self.weights_dict[mom]*np.abs(mo-tar)/tar
                        /getattr(self,mom+'_target').size
                        )
                # print(mo,tar,self.weights_dict[mom]*np.abs(mo-tar)/tar)
                
            
    def deviation_vector(self,list_of_moments = None):
        if list_of_moments is None:
            list_of_moments = self.list_of_moments
            
        dev = np.concatenate([getattr(self,mom+'_deviation').ravel() for mom in list_of_moments])
        return dev
    
    def target_vector(self,list_of_moments = None):
        if list_of_moments is None:
            list_of_moments = self.list_of_moments
        dev = np.concatenate([getattr(self,mom+'_target').ravel() for mom in list_of_moments])
        return dev

class sol_class:
    def __init__(self, x_new, p, solving_time, iterations, deviation_norm, 
                 status, hit_the_bound_count, x0=None, tol = 1e-10, 
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

class history:
    def __init__(self,*args):
        self.count = 0
        self.saves = 0
        self.dict = {}
        for a in args:
            self.dict[a] = []
        self.time = 0
    
    def append(self,**kwargs):
        for k,v in kwargs.items():
            self.dict[k].append(v)
            
    def plot(self):
        fig,ax = plt.subplots(figsize = (12,8))     
        ax2 = ax.twiny()
        count = 0
        for k,v in self.dict.items():
            if k != 'objective':
                count += 1
                if count<10:
                    ax.plot(np.linspace(0,self.count,len(v)),v,label=k)
                else:
                    ax.plot(np.linspace(0,self.count,len(v)),v,label=k, ls = '--')
            else:
                ax2.plot(np.linspace(0,self.time/60,len(v)),v,label=k,color='k',lw=2)
        ax.set_xlabel('Number of succesful steady state solving')
        ax.set_ylabel('Loss for each moment')
        ax2.set_xlabel('Time (min)')
        plt.yscale('log')
        ax.legend()
        ax2.legend(loc=(0.85,1.05))
        plt.show() 
    
    def save(self,p,path):
        try:
            os.mkdir(path)
        except:
            pass
        p.write_params(path+str(self.saves)+'/')
        self.saves += 1        

def get_vec_qty(x,p):
    # res = {'price_indices':x[0:p.N],
    #        'w':x[p.N:p.N*2],
    #        'Z':x[p.N*2:p.N*3],
    #        'l_R':x[p.N*3:p.N*3+p.N*(p.S-1)],
    #        'psi_star':x[p.N*3+p.N*(p.S-1):p.N*3+p.N*(p.S-1)+p.N**2],
    #        'phi':x[p.N*3+p.N*(p.S-1)+p.N**2:]
    #        }
    res = {'w':x[0:p.N],
           'Z':x[p.N:p.N*2],
           'l_R':x[p.N*2:p.N*2+p.N*(p.S-1)],
           'psi_star':x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2],
           'phi':x[p.N*2+p.N*(p.S-1)+p.N**2:]
           }
    return res

def deviation(x,p):
    vec, _ = iter_once(x,p)
    return vec - x

def deviation_norm(x,p):
    return np.linalg.norm(deviation(x,p))

def bound_psi_star(x,p,hit_the_bound=None):
    x_psi_star = x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2]
    if np.any(x_psi_star<1):
        hit_the_bound += 1
        x_psi_star[x_psi_star<1] = 1
    x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2] = x_psi_star
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
        x_new = x_old*1/2+x_new*1/2
        high_jumps_too_big = x_new > 1000*x_old
    low_jumps_too_big = x_new < x_old/1000
    while np.any(low_jumps_too_big):
        x_new = x_old*1/2+x_new*1/2
        low_jumps_too_big = x_new < x_old/1000
    return x_new

def write_calibration_results(path,p,m,sol_c,commentary = None):
    writer = pd.ExcelWriter(path+'.xlsx', engine='xlsxwriter')
    # summary = pd.DataFrame({'calibrated parameters':str(calib_parameters),
    #                        'targeted moments':m.list_of_moments,
    #                        'summary':commentary})
    # summary = pd.DataFrame(columns=['summary'])
    workbook = writer.book
    worksheet = workbook.add_worksheet('Summary')
    writer.sheets['Summary'] = worksheet
    df1 = pd.DataFrame(index = p.get_list_of_params(), 
                      columns = ['state', 'lower bound', 'higher bound'])
    for pa_name in p.get_list_of_params():
        if pa_name in p.calib_parameters:
            df1.loc[pa_name] = ['calibrated',p.lb_dict[pa_name],p.ub_dict[pa_name]]
        else:
            df1.loc[pa_name] = ['fixed','','']
    df1.name = 'parameters'
    worksheet.write_string(0, 0, df1.name)
    df1.to_excel(writer,sheet_name='Summary',startrow=1 , startcol=0)
    
    
    df2 = pd.DataFrame(index = m.get_list_of_moments(), columns = ['weight','norm of deviation', 'description'])
    for mom in m.get_list_of_moments():
        df2.loc[mom] = [m.weights_dict[mom],
                        np.linalg.norm(getattr(m,mom+'_deviation')),
                        m.description.loc[mom].description]
    df2.name = 'targeted moments : '+str(m.list_of_moments)
    worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
    df2.to_excel(writer,sheet_name='Summary',startrow=df1.shape[0] + 5 , startcol=0)
    
    worksheet.write_string(df1.shape[0] + df1.shape[0] + 6, 0, commentary)
    
    
    scalar_moments = pd.DataFrame(columns=['model','target'])
    for mom in m.get_list_of_moments():
        print(mom)
        if np.array(getattr(m,mom)).size == 1:
            scalar_moments.loc[mom] = [getattr(m,mom),getattr(m,mom+'_target')]
        else:
            moment = getattr(m,mom)
            moment_target = getattr(m,mom+'_target')
            # df = pd.DataFrame(data = [np.array(moment).ravel(),np.array(moment_target).ravel()],
            #                   index=m.idx[mom], columns = ['model','target'])
            df = pd.DataFrame({'model':np.array(moment).ravel(),'target':np.array(moment_target).ravel()},
                              index=m.idx[mom])
            df.to_excel(writer,sheet_name=mom)
    scalar_moments.to_excel(writer,sheet_name='scalar_moments')
    scalar_parameters = pd.DataFrame(columns=['value'])
    for pa_name in p.get_list_of_params():
        if np.array(getattr(p,pa_name)).size == 1:
            scalar_parameters.loc[pa_name] = getattr(p,pa_name)
        else:
            par = getattr(p,pa_name)
            df = pd.DataFrame({'value':np.array(par).ravel()},index=p.idx[pa_name])
            df.to_excel(writer,sheet_name=pa_name)
    scalar_parameters.to_excel(writer,sheet_name='scalar_parameters')
    
    df_labor = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_labor['non patenting'] = sol_c.nominal_value_added[:,0]/sol_c.w
    df_labor['production patenting sector'] = sol_c.nominal_value_added[:,1]/sol_c.w
    df_labor['RD'] = sol_c.l_R[:,1]
    df_labor['patenting'] = sol_c.l_Ao[:,:,1].sum(axis=1)+sol_c.l_Ae[:,:,1].sum(axis=1)
    df_labor['total'] = df_labor['non patenting']+df_labor['production patenting sector']+df_labor['RD']+df_labor['patenting']
    df_labor['total data'] = p.labor
    df_labor.to_excel(writer,sheet_name='labor')
    
    df_psi_star = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],names=['destination','origin','sector']))
    df_psi_star['psi_star'] = sol_c.psi_star.ravel()
    df_psi_star.to_excel(writer,sheet_name='psi_star')
    
    df_sales = pd.DataFrame(index=pd.MultiIndex.from_product([p.countries, p.countries],names=['destination','origin']))
    df_sales['M share of sales'] = sol_c.X_M[:,:,1].ravel()
    df_sales['CL share of sales'] = sol_c.X_CL[:,:,1].ravel()
    df_sales['CD share of sales'] = sol_c.X_CD[:,:,1].ravel()
    df_sales['total to check'] = df_sales['M share of sales'] + df_sales['CL share of sales'] + df_sales['CD share of sales']
    df_sales.to_excel(writer,sheet_name='monopolistic_competitive_shares')
    
    # df_expenditure = pd.DataFrame(index=pd.MultiIndex.from_product([p.countries, p.countries],names=['destination','origin']))
    # df_expenditure['M share of expenditure'] = sol_c.X_M[:,:,1]
    # df_expenditure['CL share of expenditure'] = sol_c.X_CL[:,:,1]
    # df_expenditure['CD share of expenditure'] = sol_c.X_CD[:,:,1].sum(axis=1)
    # df_expenditure['total check'] = df_expenditure['M share of expenditure']\
    #     + df_expenditure['CL share of expenditure'] + df_expenditure['CD share of expenditure']
    # df_expenditure.to_excel(writer,sheet_name='expenditures_shares')    
    
    df_qualities = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_qualities['PSI_M'] = sol_c.PSI_M[...,1].sum(axis=1)
    df_qualities['PSI_CL'] = sol_c.PSI_CL[...,1].sum(axis=1)
    df_qualities['PSI_CD'] = sol_c.PSI_CD[...,1]
    df_qualities['total check'] = df_qualities['PSI_M']+df_qualities['PSI_CL']+df_qualities['PSI_CD']
    df_qualities.to_excel(writer,sheet_name='aggregate_qualities')    
    
    df_prices = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_prices['P_M'] = sol_c.P_M[...,1]
    df_prices['P_CL'] = sol_c.P_CL[...,1]
    df_prices['P_CD'] = sol_c.P_CD[...,1]
    df_prices['P'] = sol_c.price_indices
    df_prices.to_excel(writer,sheet_name='prices')
    
    df_country = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_country['wage'] = sol_c.w
    df_country['expenditure'] = sol_c.Z
    df_country['gdp'] = sol_c.gdp
    df_country.to_excel(writer,sheet_name='countries_macro_quantities')
    
    df_pflows = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries],names=['destination','origin']))
    df_pflows['pflow'] = sol_c.pflow.ravel()
    df_pflows.to_excel(writer,sheet_name='number of patent flows')
    
    df_tau = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries, p.sectors],names=['destination','origin','sector']))
    df_tau['tau'] = sol_c.tau.ravel()
    df_tau.to_excel(writer,sheet_name='tau')
    
    df_share_patented = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries],names=['destination','origin']))
    df_share_patented['share_innov_patented'] = sol_c.share_innov_patented.ravel()
    df_share_patented.to_excel(writer,sheet_name='share of innovations patented')
    
    writer.save()
    
def full_load_parameters_set(path,list_of_moments = None):
    if list_of_moments is None:
        list_of_moments = moments.get_list_of_moments()
    p = parameters()
    p.load_data(path)
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=True,
                            safe_convergence=0.001,
                            disp_summary=True,
                            damping = 10,
                            max_count = 3e3,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    sol_c.compute_price_indices
    sol_c.compute_non_solver_quantities(p)
    m = moments(list_of_moments)
    m.load_data()
    m.compute_moments(sol_c,p)
    m.compute_Z(sol_c,p)
    m.compute_moments_deviations()
    return p,sol_c,m

def load_collection_of_guess(path = 'calibration_results/'):
    collec = []
    for i in range(len(next(os.walk('calibration_results/'))[1])):
        try:
            guess = pd.read_csv(path+str(i)+'/guess.csv',
                                header = None).values.squeeze()
            collec.append(guess)
        except:
            pass
    return collec

def load_grad_jac(path):
    grad = pd.read_csv(path+'grad.csv',index_col=0).values.squeeze()
    jac = pd.read_csv(path+'jac.csv',index_col=0).values.squeeze().reshape(-1,grad.size)
    return grad, jac

def iter_once(x,p, check_feasibility = False, normalize = False):
    modified_guess = None
    init = var.var_from_vector(x,p,compute=False)
    
    init.compute_growth(p)
    if check_feasibility:
        check = init.check_PSI_CL(p)
        if check == 'corrected':
            modified_guess = init.vector_from_var()
    init.compute_aggregate_qualities(p)
    init.compute_monopolistic_sectoral_prices(p)
    init.compute_monopolistic_trade_flows(p)
    init.compute_competitive_sectoral_prices(p)
    init.compute_competitive_trade_flows(p)
    init.compute_labor_allocations(p)     
    init.compute_total_trade_flows(p)
    init.compute_non_solver_quantities(p)
    if normalize:
        init.num_scale_solution(p)
    # plt.plot(init.X.ravel())
    # plt.show()
    price = init.compute_price_indices(p)
    # numeraire = price[0]
    w = init.compute_wage(p)
    Z = init.compute_expenditure(p)
    l_R = init.compute_labor_research(p)[...,1:].ravel()
    psi_star = init.compute_psi_star(p)[...,1:].ravel()
    phi = init.compute_phi(p).ravel()
    # phi = phi/phi.mean()
    
    # print(numeraire)
    # print(phi.max()/numeraire**p.theta[0])
    vec = np.concatenate((price,w,Z,l_R,psi_star,phi), axis=0)
    return vec, modified_guess            

def fixed_point_solver(p, x0=None, tol = 1e-10, damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, cobweb_anim=False, cobweb_qty='psi_star',
                       cobweb_coord = 1, plot_convergence = True, apply_bound_zero = True, 
                       apply_bound_psi_star = False, apply_bound_research_labor = False,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
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
    cob = cobweb(cobweb_qty)
    norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40):
        # print(count)
        
        
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_new = smooth_large_jumps(x_new,x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        # x_new, modified_x_old  = iter_once(x_old, p, 
        #                                    normalize = False, 
        #                                    check_feasibility = False)
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        if apply_bound_psi_star:
            x_old, hit_the_bound_count = bound_psi_star(x_old, p, hit_the_bound_count)
        if apply_bound_research_labor:
            x_old, hit_the_bound_count = bound_research_labor(x_old, p, hit_the_bound_count) 
            
        init = var.var_from_vector(x_old,p,compute=False)
        init.compute_growth(p)
        # init.psi_star[init.psi_star < 1] = 1 
        # print(init.psi_star.min())
        # init.check_PSI_CL(p)
        # if count>1e3:
        #     # damping = 1
        #     plt.plot(x_old)
        #     plt.show()
        # init.phi = init.phi/np.diagonal(init.phi).transpose()[:,None,:]
        # init.phi = init.phi/init.phi.mean()
        # x_old = init.vector_from_var()
        # if count>1e4:
        #     # damping = 1
        #     plt.plot(x_old)
        #     plt.show()
        # init.scale_P(p)
        # init.scale_Z(p)
        
        init.compute_aggregate_qualities(p)
        init.compute_sectoral_prices(p)
        init.compute_trade_shares(p)
        init.compute_labor_allocations(p)
        init.compute_price_indices(p)
        
        w = init.compute_wage(p)
        Z = init.compute_expenditure(p)
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        psi_star = init.compute_psi_star(p)[...,1:].ravel()
        # psi_star[psi_star < 1] = 1
        phi = init.compute_phi(p).ravel()
        # phi = phi/(np.diagonal(phi).transpose()[:,None,:])
        # phi = (phi/phi.mean())
        # phi = phi.ravel()
        
        x_new = np.concatenate((w,Z,l_R,psi_star,phi), axis=0)
        # temp = var.var_from_vector(x_new,p,compute=False)
        # temp.scale_P(p)
        # temp.scale_Z(p)
        # temp.compute_price_indices(p)
        # x_new = temp.vector_from_var()
        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        # if count>500:
        #     # damping = 1
        #     # plt.scatter(init.X,p.trade_flows)
        #     plt.plot(x_new_decomp[cobweb_qty])
        #     plt.show()
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','psi_star','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        count += 1
        if np.all(np.array(convergence[-10:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
        norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
        history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
        history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
        # print(convergence[-1])
        # print(init.price_indices[0])
    
    finish = time.perf_counter()
    solving_time = finish-start
    # dev_norm = deviation_norm(x_new,p)
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 100,pause = 0.05) 
        cob.plot(count = count, window = 200)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init

#%% fixed point solver
# p = parameters(n=7,s=2)
# p.calib_parameters = ['eta','delta','fe','tau','T','fo','g_0','nu','nu_tilde','zeta']
# p.load_data('calibration_results_matched_economy/14/',p.get_list_of_params())
# p.delta[0,1] = 0.05
# Z_guess = p.data.expenditure.values/p.unit
# w_guess = p.data.gdp.values*p.unit_labor/(p.data.labor.values*p.unit)*100
# l_R_guess = np.repeat(p.labor[:,None]/200, p.S-1, axis=1).ravel()
# psi_star_guess = np.ones((p.N,p.N,(p.S-1))).ravel()*1000
# phi_guess = np.ones((p.N,p.N,p.S)).ravel()#*0.01
# vec = np.concatenate((w_guess,Z_guess,l_R_guess,psi_star_guess,phi_guess), axis=0)
# guess = np.random.rand(p.guess_from_params().size).reshape(p.guess_from_params().shape)
sol, sol_c = fixed_point_solver(p,#x0=p.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='psi_star',
                        plot_convergence=True,
                        plot_cobweb=True,
                        safe_convergence=0.01,
                        disp_summary=True,
                        damping = 10,
                        max_count = 1000,
                        accel_memory = 10, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )

sol_c = var.var_from_vector(sol.x, p)    
# sol_c.scale_tau(p)
sol_c.scale_P(p)
# sol_c.compute_growth(p)
# sol_c.compute_aggregate_qualities(p)
# sol_c.compute_sectoral_prices(p)
# sol_c.compute_trade_shares(p)
# sol_c.compute_labor_allocations(p)
sol_c.compute_price_indices(p)
# # sol_c.scale_Z(p)
# # sol_c.compute_growth(p)
# # sol_c.compute_aggregate_qualities(p)
# # sol_c.compute_sectoral_prices(p)
# # sol_c.compute_trade_flows(p)
# # sol_c.compute_labor_allocations(p)
sol_c.compute_non_solver_quantities(p) 

# # sol_c.compute_non_solver_quantities(p)
list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP',
                    'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                    'SINNOVPATEU']
m = moments(list_of_moments)
m.load_data()
m.compute_moments(sol_c,p)
m.compute_Z(sol_c,p)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)
# p.guess = sol_c.vector_from_var()

#%% calibration func

def calibration_func(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    p.update_parameters(vec_parameters)
    # p.nu_tilde = p.nu
    try:
        v0 = p.guess
    except:
        pass
    sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=1e3,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=5
                                  )
    if sol.status == 'failed': 
        # print('trying less precise 1')
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    
    # if sol.status == 'failed': 
    #     print('trying less precise 2')
    #     sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-12,
    #                                   accelerate=False,
    #                                   accelerate_when_stable=True,
    #                                   plot_cobweb=False,
    #                                   plot_convergence=False,
    #                                   cobweb_qty='phi',
    #                                   disp_summary=False,
    #                                   safe_convergence=0.001,
    #                                   max_count=2e3,
    #                                   accel_memory = 50, 
    #                                   accel_type1=True, 
    #                                   accel_regularization=1e-10,
    #                                   accel_relaxation=0.5, 
    #                                   accel_safeguard_factor=1, 
    #                                   accel_max_weight_norm=1e6,
    #                                   damping_post_acceleration=5
    #                                   )
    
    # if sol.status == 'failed': 
    #     print('trying less precise 3')
    #     sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-12,
    #                                   accelerate=False,
    #                                   accelerate_when_stable=True,
    #                                   plot_cobweb=False,
    #                                   plot_convergence=False,
    #                                   cobweb_qty='phi',
    #                                   disp_summary=False,
    #                                   safe_convergence=0.001,
    #                                   max_count=5e3,
    #                                   accel_memory = 50, 
    #                                   accel_type1=True, 
    #                                   accel_regularization=1e-10,
    #                                   accel_relaxation=0.5, 
    #                                   accel_safeguard_factor=1, 
    #                                   accel_max_weight_norm=1e6,
    #                                   damping_post_acceleration=5
    #                                   )
                            
    # if sol.status == 'failed':
    #     print('trying with good guess without acceleration')
    #     sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-12,
    #                                   accelerate=False,
    #                                   accelerate_when_stable=False,
    #                                   plot_cobweb=False,
    #                                   plot_convergence=False,
    #                                   cobweb_qty='phi',
    #                                   disp_summary=False,
    #                                   safe_convergence=0.001,
    #                                   max_count=5e3)
    if sol.status == 'failed':
        print('trying with standard guess')
        sol, sol_c = fixed_point_solver(p,x0=None,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=1e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
       
    # if sol.status == 'failed':
    #     print('trying with standard guess without acceleration')
    #     sol, sol_c = fixed_point_solver(p,x0=None,tol=1e-12,
    #                                   accelerate=False,
    #                                   accelerate_when_stable=False,
    #                                   plot_cobweb=False,
    #                                   plot_convergence=False,
    #                                   cobweb_qty='phi',
    #                                   disp_summary=False,
    #                                   safe_convergence=0.001,
    #                                   max_count=5e3
    #                                   ) 
    # sol_c = var.var_from_vector(sol.x, sol.p)   
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    sol_c.compute_price_indices(p)
    # sol_c.scale_Z(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    m.compute_Z(sol_c,p)
    m.compute_moments_deviations()
    # print(hist.count)
    if hist is not None:
        if hist.count%1 == 0:
            hist_dic = {mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments}
            hist_dic['objective'] = np.linalg.norm(m.deviation_vector())
            hist.append(**hist_dic)
            hist.time = time.perf_counter() - start_time
        # if hist.count%100 == 0:
        #     m.plot_moments(m.list_of_moments)
        if hist.count%40 == 0:
            hist.plot()
        if hist.count%200==0:
            print('fe : ',p.fe[1],'fo : ',p.fo[1], 'delta : ', p.delta[:,1]
                  , 'nu : ', p.nu[1], 'nu_tilde : ', p.nu_tilde[1], 'k :', p.k)
        if hist.count%200==0:
            hist.save(path = './calibration_results_matched_trade_flows/history105/', p = p)
    hist.count += 1
    p.guess = sol_c.vector_from_var()
    # print(hist.count)
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        # hist.save(path = './calibration_results/fails2/', p = p)
        return np.full_like(m.deviation_vector(),1e10)
        # return m.deviation_vector() 
    else:
        return m.deviation_vector() 

#%% calibration   

new_run = True
if new_run:
    p = parameters(n=7,s=2)
    # p.calib_parameters = ['eta',     'k',     'rho',     'alpha',     'fe',
     # 'T',     'fo',     'sigma',     'theta',     'beta',
     # 'zeta',     'g_0',     'kappa','gamma',
     # 'delta',     'nu',     'nu_tilde']
    p.calib_parameters = ['eta','k','fe','T','theta','zeta','g_0',
                          'delta','nu','nu_tilde']
    # p.calib_parameters = ['T']
    # p.load_data('calibration_results_matched_trade_flows/history27/6/')
    start_time = time.perf_counter()
# p.calib_parameters = ['eta','delta','fe','T',
#                       'g_0','zeta','nu','nu_tilde']
# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                    # 'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                    # 'SINNOVPATEU','NUR']
# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
#                     'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
#                     'SINNOVPATEU']
list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                    'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                    'SINNOVPATEU','NUR']
# list_of_moments = ['TP']
m = moments(list_of_moments)
if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
m.load_data()
# m.SDOMTFLOW_target = np.random.rand(p.N*p.S).reshape(p.N,p.S)*100
bounds = p.make_parameters_bounds()
# collec_of_guess = load_collection_of_guess()
cond = True
iterations = 0
while cond:
    if iterations == 0:
        x0 = p.make_p_vector()
    else:
        x0 = test_ls.x
    test_ls = optimize.least_squares(fun = calibration_func,    
                        x0 = x0, 
                        args = (p,m,p.guess,hist,start_time), 
                        bounds = bounds,
                        # method= 'dogbox',
                        # loss='arctan',
                        # jac='3-point',
                        max_nfev=1e8,
                        ftol=1e-14, 
                        xtol=1e-14, 
                        gtol=1e-14,
                        # f_scale=scale,
                        verbose = 2)
    cond = test_ls.cost>0.0001 and test_ls.nfev>30
    cond= False
    iterations += 1
finish_time = time.perf_counter()
print('minimizing time',finish_time-start_time)

p_sol = p.copy()
p_sol.update_parameters(test_ls.x)

sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                        cobweb_anim=False,tol =1e-15,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=True,
                        plot_cobweb=True,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        # damping=10
                          # apply_bound_psi_star=True
                        )

sol_c = var.var_from_vector(sol.x, p_sol)    
# sol_c.scale_tau(p_sol)
sol_c.scale_P(p_sol)
sol_c.compute_price_indices(p)
sol_c.compute_non_solver_quantities(p_sol) 
m.compute_moments(sol_c,p_sol)
m.compute_Z(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)


#%% load parameters sets

# p2, sol2, m2 = full_load_parameters_set('./calibration_results_matched_trade_flows/2/')
# p3, sol3, m3 = full_load_parameters_set('./calibration_results_matched_trade_flows/3/')
# p4, sol4, m4 = full_load_parameters_set('./calibration_results_matched_trade_flows/4/')
# p5, sol5, m5 = full_load_parameters_set('./calibration_results_matched_trade_flows/5/')
# p6, sol6, m6 = full_load_parameters_set('./calibration_results_matched_trade_flows/6/')
# p7, sol7, m7 = full_load_parameters_set('./calibration_results_matched_trade_flows/7/')
# p8, sol8, m8 = full_load_parameters_set('calibration_results_matched_trade_flows/history3/778/')
# p14, sol14, m14 = full_load_parameters_set('calibration_results_matched_trade_flows/14/')
# p15, sol15, m15 = full_load_parameters_set('calibration_results_matched_trade_flows/15/')
# p16, sol16, m16 = full_load_parameters_set('calibration_results_matched_trade_flows/16/')
# p17, sol17, m17 = full_load_parameters_set('calibration_results_matched_trade_flows/17/')
# p18, sol18, m18 = full_load_parameters_set('calibration_results_matched_trade_flows/18/')
# p27, sol27, m27 = full_load_parameters_set('calibration_results_matched_trade_flows/27/')
# p24, sol24, m24 = full_load_parameters_set('calibration_results_matched_trade_flows/24/')
# p21, sol21, m21 = full_load_parameters_set('calibration_results_matched_trade_flows/21/')
# p22, sol22, m22 = full_load_parameters_set('calibration_results_matched_trade_flows/22/')
# p1, sol1, m1 = full_load_parameters_set('calibration_results_matched_economy/1/')
# p2, sol2, m2 = full_load_parameters_set('calibration_results_matched_economy/2/')
# p3, sol3, m3 = full_load_parameters_set('calibration_results_matched_economy/3/')
p4, sol4, m4 = full_load_parameters_set('calibration_results_matched_economy/4/',
                                        ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                            'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                            'SINNOVPATEU'])
p6, sol6, m6 = full_load_parameters_set('calibration_results_matched_economy/6/',
                                        ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                            'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                            'SINNOVPATEU'])
m6.SRDUS_target = np.array(0.325)
m6.compute_moments(sol6, p6)
m6.compute_moments_deviations()
p7, sol7, m7 = full_load_parameters_set('calibration_results_matched_economy/7/',
                                        ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                            'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                            'SINNOVPATEU'])
# p8, sol8, m8 = full_load_parameters_set('calibration_results_matched_economy/8/')
# p9, sol9, m9 = full_load_parameters_set('calibration_results_matched_economy/9/')
p10, sol10, m10 = full_load_parameters_set('calibration_results_matched_economy/10/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p11, sol11, m11 = full_load_parameters_set('calibration_results_matched_economy/11/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p12, sol12, m12 = full_load_parameters_set('calibration_results_matched_economy/12/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p13, sol13, m13 = full_load_parameters_set('calibration_results_matched_economy/13/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p14, sol14, m14 = full_load_parameters_set('calibration_results_matched_economy/14/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
p15, sol15, m15 = full_load_parameters_set('calibration_results_matched_economy/15/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
p16, sol16, m16 = full_load_parameters_set('calibration_results_matched_economy/16/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
m16.GROWTH_target = m16.GROWTH_target*2
m16.compute_moments(sol16, p16)
m16.compute_moments_deviations()
p17, sol17, m17 = full_load_parameters_set('calibration_results_matched_economy/17/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
m17.GROWTH_target = m17.GROWTH_target*4
m17.compute_moments(sol17, p17)
m17.compute_moments_deviations()
p18, sol18, m18 = full_load_parameters_set('calibration_results_matched_economy/18/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])       
m18.GROWTH_target = m18.GROWTH_target*10
m18.compute_moments(sol18, p18)
m18.compute_moments_deviations()
p19, sol19, m19 = full_load_parameters_set('calibration_results_matched_economy/19/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p20, sol20, m20 = full_load_parameters_set('calibration_results_matched_economy/20/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p21, sol21, m21 = full_load_parameters_set('calibration_results_matched_economy/21/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p22, sol22, m22 = full_load_parameters_set('calibration_results_matched_economy/22/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'TP','SINNOVPATEU'])
p23, sol23, m23 = full_load_parameters_set('calibration_results_matched_economy/23/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'TP','SINNOVPATEU'])
p24, sol24, m24 = full_load_parameters_set('calibration_results_matched_economy/24/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p30, sol30, m30 = full_load_parameters_set('calibration_results_matched_economy/30/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p31, sol31, m31 = full_load_parameters_set('calibration_results_matched_economy/31/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p32, sol32, m32 = full_load_parameters_set('calibration_results_matched_economy/32/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU'])
p33, sol33, m33 = full_load_parameters_set('calibration_results_matched_economy/33/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
p34, sol34, m34 = full_load_parameters_set('calibration_results_matched_economy/34/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
p35, sol35, m35 = full_load_parameters_set('calibration_results_matched_economy/35/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
p36, sol36, m36 = full_load_parameters_set('calibration_results_matched_economy/36/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                                               'SPFLOW', 'SRGDP', 'JUPCOST',
                                                               'SINNOVPATEU','NUR'])
# for m in [m4,m6,m7,m10,m11,m12,m13]:
#     m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
#                         'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
#                         'SINNOVPATEU']
# for m in [m14,m15]:
#     m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
#                         'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
#                         'SINNOVPATEU','NUR']


#%% writing results as excel

commentary = 'calibrated nu higher weight, calibrated k and sigma'
write_calibration_results(
    '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/36',
    p36,m36,sol36,commentary = commentary)
m36.plot_moments(m36.list_of_moments, 
                save_plot = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/36')
# commentary = 'Square root weights on dimension'
# write_calibration_results(
#     '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/3',
#     p3,m3,sol3,commentary = commentary)
# m3.plot_moments(m3.list_of_moments, 
#                 save_plot = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/3')
# commentary = 'Delta US free, SRDUS divided by 2'
# write_calibration_results(
#     '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/6',
#     p6,m6,sol6,commentary = commentary)
# m6.plot_moments(m6.list_of_moments, 
#                 save_plot = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/6')
# commentary = 'Delta US free'
# write_calibration_results(
#     '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/7',
#     p7,m7,sol7,commentary = commentary)
# m7.plot_moments(m7.list_of_moments, 
#                 save_plot = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/7')
# runs = [10,11,12,13,14]
# comments = ['Nu_tilde = nu*5',
#             'Nu_tilde = nu*10',
#             'Nu_tilde = Nu',
#             'Nu_tilde = Nu/5',
#             'Ratio of nu targeted as a moment, target 1']
# runs = [16,17,18]
# ps = [p16,p17,p18]
# ms = [m16,m17,m18]
# solss = [sol16,sol17,sol18]
# comments = ['Growth = growth*2', 'Growth = growth*4','Growth = growth*10']
# path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'

# for i, run in enumerate(runs):
#     # p, sol, m = full_load_parameters_set('calibration_results_matched_economy/'+str(run)+'/')
#     write_calibration_results(
#         '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'+str(run),
#         ps[i],ms[i],solss[i],commentary = comments[i])
#     ms[i].plot_moments(ms[i].list_of_moments, 
#                     save_plot = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'+str(run))


#%% build dic compare moments

dic = {#'3 : lin weights on dim':m3,
       # '4 : sqrt weights on dim':m4,
       # '6 : delta US free, SRDUS divided by 2':m6,
        '7 : free nu, nu ='+str(p7.nu[1]):m7,
       # '10 : Nu_tilde = nu*5':m10,
       # '11 : Nu_tilde = nu*10':m11,
        # '12 : Nu_tilde = nu, growth = 1.69%':m12,
       # '13 : Nu_tilde = nu/5':m13,
       # '14 : Ratio of nu targeted as a moment, target 1':m14,
       # '15 : Ratio of nu targeted as a moment, low weight':m15,
       # '16 : Growth = growth*2':m16,
       # '17 : Growth = growth*4':m17,
       # '18 : Growth = growth*10':m18,
        '19 : Fixed nu = 0.1':m19,
       # '20 : Fixed nu = nu_tilde = 0.1':m20,
       # '21 : domestic flows targeted':m21,
       # '22 : total number of patents targeted':m22,
       # '23 : total number of patents targeted, higher weights on RD':m23,
       # '30 : equivalent to 7, nu = nu_tilde, theta = 8':m30,
       # '31 : equivalent to 7, theta = 8':m31,
       # '32 : calibrated k, theta = 8':m32,
       # '33 : calibrated nu low weight, calibrated k':m33,
       # '34 : calibrated nu higher weight':m34,
       '35 : calibrated nu calibrated k and sigma, nu ='+str(p35.nu[1]):m35,
       '36 : calibrated nu higher weight, calibrated k and sigma, nu ='+str(p36.nu[1]):m36
       }


moments.compare_moments(dic,
    save_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/nu_calibration/')
# moments.compare_moments(dic)

#%% compare params

dic = {#'3 : lin weights on dim':p3,
        # '4 : sqrt weights on dim':p4,
        # '6 : delta US free, SRDUS divided by 2':p6,
        '7 : free nu, nu ='+str(p7.nu[1]):p7,
       #  '10 : Nu_tilde = nu*5':p10,
       #  '11 : Nu_tilde = nu*10':p11,
        # '12 : Nu_tilde = nu, growth = 1.69%':p12,
       #  '13 : Nu_tilde = nu/5':p13,
       #  '14 : Ratio of nu targeted as a moment, target 1':p14,
       #  '15 : Ratio of nu targeted as a moment, low weight':p15,
       # '16 : Growth = growth*2':p16,
       # '17 : Growth = growth*4':p17,
       # '18 : Growth = growth*10':p18,
        '19 : Fixed nu = 0.1':p19,
       # '20 : Fixed nu = nu_tilde = 0.1':p20,
       # '31 : equivalent to 7':p31,
       # '32 : calibrated k, theta = 8':p32,
       # '33 : calibrated nu low weight, calibrated k':p33,
       # '34 : calibrated nu higher weight':p34,
       '35 : calibrated nu calibrated k and sigma, nu ='+str(p35.nu[1]):p35,
       '36 : calibrated nu higher weight, calibrated k and sigma, nu ='+str(p36.nu[1]):p36
       }

save = True
save_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/nu_calibration/'

fig,ax = plt.subplots(figsize = (12,8))
title = 'Delta of the countries' 
for com,par in dic.items():
    ax.plot(par.countries,par.delta[...,1],label=com)
plt.legend()
plt.title(title)
if save:
    plt.savefig(save_path+'delta')
plt.show()    

fig,ax = plt.subplots(figsize = (12,8))
title = 'One over delta of the countries' 
for com,par in dic.items():
    ax.plot(par.countries,1/par.delta[...,1],label=com)
plt.legend()
plt.title(title)
if save:
    plt.savefig(save_path+'one_over_delta')
plt.show()   

fig,ax = plt.subplots(figsize = (12,8))
title = 'T non patenting sector of the countries' 
for com,par in dic.items():
    ax.plot(par.countries,par.T[:,0],label=com)
plt.legend()
plt.yscale('log')
plt.title(title)
if save:
    plt.savefig(save_path+'T_non_patenting')
plt.show()   

fig,ax = plt.subplots(figsize = (12,8))
title = 'T patenting sector of the countries' 
for com,par in dic.items():
    ax.plot(par.countries,par.T[:,1],label=com)
plt.legend()
plt.yscale('log')
plt.title(title)
if save:
    plt.savefig(save_path+'T_patenting')
plt.show() 

fig,ax = plt.subplots(figsize = (12,8))
title = 'Eta of the countries' 
for com,par in dic.items():
    ax.plot(par.countries,par.eta[...,1],label=com)
plt.legend()
plt.title(title)
if save:
    plt.savefig(save_path+'eta')
plt.show() 

fig,ax = plt.subplots(figsize = (12,8))
ax2 = ax.twinx()
title = 'fo, fe, nu, nu_tilde' 
for com,par in dic.items():
    ax.scatter(['fo','fe', 'nu'],
               [par.fo[1], par.fe[1], par.nu[1]],label=com)
    ax2.scatter(['nu_tilde'], par.nu_tilde[1], label = com)
plt.legend()
plt.title(title)
ax.set_ylabel('fo, fe, nu')
ax2.set_yscale('log')
ax2.set_ylabel('nu_tilde')
if save:
    plt.savefig(save_path+'scalar_params')
plt.show()     

#%% counterfactual



p = p36.copy()

sol, baseline = fixed_point_solver(p,x0=p.guess,
                        cobweb_anim=False,tol =1e-15,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        # damping=10
                          # apply_bound_psi_star=True
                        )

baseline.scale_P(p)
baseline.compute_price_indices(p)
baseline.compute_non_solver_quantities(p)

for j,c in enumerate(p.countries):
    p = p14.copy()
    sols_c = []
    deltas = np.logspace(-1,1,11)
    for delt in deltas:
        print(delt)
        p.delta[j,1] = p14.delta[j,1] * delt
        # print(p.guess)
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=True,
                                damping = 10,
                                max_count = 3e3,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        
    
        sol_c = var.var_from_vector(sol.x, p)    
        # sol_c.scale_tau(p)
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p)
        # sol_c.compute_welfare(p)
        sol_c.compute_consumption_equivalent_welfare(p,baseline)
        if sol.status == 'success':
            p.guess = sol_c.vector_from_var()
        else:
            p.guess = None
        
        sols_c.append(sol_c)

# utilities = pd.DataFrame(index = deltas, columns = p.countries, data = np.array([sol.U for sol in sols_c]))
# (utilities/np.abs(utilities.iloc[5])).plot(figsize=(12,8),logx=True)
# consumptions = pd.DataFrame(index = deltas, columns = p.countries, data = np.array([sol.nominal_final_consumption/sol.price_indices for sol in sols_c]))
# (consumptions/consumptions.iloc[5]).plot(figsize=(12,8),logx=True)
    CE = pd.DataFrame(index = deltas, 
                  columns = p.countries, 
                  data = np.array([sol.cons_eq_welfare for sol in sols_c]))
    ax = CE.plot(figsize=(16,12),logx=True,
             fontsize = 20,color=sns.color_palette()[:CE.shape[1]])
    idxmaxes = CE.idxmax()
    maxes = CE.max()
    CE_max = pd.DataFrame({'idx' : idxmaxes, 'values' : maxes})
    CE_max.plot.scatter('idx','values',
                        ax = ax,color=sns.color_palette()[:CE.shape[1]])
    ax.set_xlabel('Change in delta '+c, fontsize = 20)
    ax.set_ylabel('Consumption equivalent welfare', fontsize = 20)
    # ax.scatter(CE.idxmax(),CE.max(),color=[mcolors.BASE_COLORS[k] for k in list(mcolors.BASE_COLORS.keys())[:CE.shape[1]]])
    # CE.to_csv('/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilateral_patent_protection/'+c+'.csv')
    # ax.figure.savefig('/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilateral_patent_protection/'+c)

#%% plot counterfactual

# fig,ax = plt.subplots(figsize = (12,8))
# ax2 = ax.twinx()
# deltas = np.linspace(0.1,0.9,100)
# utilities = np.array([sol.U for sol in sols_c])
utilities = pd.DataFrame(index = deltas, columns = p.countries, data = np.array([sol.U for sol in sols_c]))
# utilities['delta_US'] = deltas
for i in range(7):
    print(max([sol.U[i] for sol in sols_c]),[sol.U[i] for sol in sols_c][-1])
    # ax.plot(deltas, [sol.U[i]/sols_c[5].U[i]/(sol.U[1]/sols_c[5].U[0]) for sol in sols_c], label = p.countries[i])    
    ax.plot(deltas, [sol.U[i]/sols_c[5].U[i] for sol in sols_c], label = p.countries[i])    
ax2.plot(deltas, [sol.g for sol in sols_c], label = 'growth rate',color = 'k')
ax.legend()
ax2.legend()
plt.show()

# fig,ax = plt.subplots(figsize = (12,8))
# for i in range(7):
#     # print(max([sol.U[i] for sol in sols_c]),[sol.U[i] for sol in sols_c][-1])
#     # ax.plot(deltas, [sol.U[i]/sols_c[5].U[i]/(sol.U[1]/sols_c[5].U[0]) for sol in sols_c], label = p.countries[i])    
#     ax.plot(deltas, [sol.l_R[i,1] for sol in sols_c], label = p.countries[i])    
# # ax2.plot(deltas, [sol.g for sol in sols_c], label = 'growth rate',color = 'k')
# ax.legend()
# # ax2.legend()
# plt.show()

# fig,ax = plt.subplots(figsize = (12,8))
# # deltas = np.linspace(0.1,0.9,100)
# for i in range(7):
#     # print(max([sol.U[i] for sol in sols_c]),[sol.U[i] for sol in sols_c][-1])
# ax.plot(deltas, [sol.g[i] for sol in sols_c], label = p.countries[i])    
# plt.legend()
# plt.show()


#%% tau smaller than one discussion

domestic_phi = pd.DataFrame(data = np.diagonal(sol_c.phi).transpose().ravel(), columns = ['phi'], 
                            index = pd.MultiIndex.from_product([p.countries,p.sectors],
                                                                names=['country','sector']))

tau_df = pd.DataFrame(data = sol_c.tau.ravel(), columns = ['tau'],
                      index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],
                                                          names=['destination','origin','sector']))

tau_smaller_than_one = tau_df[tau_df['tau'] < 1-1e-10]

phi_df = pd.DataFrame(data = sol_c.phi.ravel(), columns = ['phi'], 
                    index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],
                                                      names=['destination','origin','sector']))

phi_corresponding = phi_df.loc[[idx for idx in tau_smaller_than_one.index]]

trade_shares_origin = pd.DataFrame(columns = ['trade_shares_origin'], 
                                    index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],
                                                      names=['destination','origin','sector']))
trade_shares_origin['trade_shares_origin'] = (sol_c.X/np.diagonal(sol_c.X).transpose()[None,:,:]).ravel()
corresponding_trade_shares_origin = trade_shares_origin.loc[[idx for idx in tau_smaller_than_one.index]]

trade_shares_destination = pd.DataFrame(columns = ['trade_shares_destination'], 
                                    index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],
                                                      names=['destination','origin','sector']))
trade_shares_destination['trade_shares_destination'] = (sol_c.X/np.diagonal(sol_c.X).transpose()[:,None,:]).ravel()
corresponding_trade_shares_destination = trade_shares_destination.loc[[idx for idx in tau_smaller_than_one.index]]

macro = pd.DataFrame(index = pd.Index(p.countries,name = 'country'))
macro['wage'] = sol_c.w
macro['price_index'] = sol_c.price_indices
macro['productivity'] = p.T

domestic_phi.sort_values('phi',inplace=True)
domestic_phi.to_csv('misc/discussion_tau_smaller_one/domestic_phi.csv')
tau_smaller_than_one.sort_values('tau',inplace=True)
tau_smaller_than_one.to_csv('misc/discussion_tau_smaller_one/tau_smaller_than_one.csv')
phi_corresponding.sort_values('phi',ascending=False,inplace=True)
phi_corresponding.to_csv('misc/discussion_tau_smaller_one/phi_corresponding.csv')
macro.to_csv('misc/discussion_tau_smaller_one/macro.csv')
corresponding_trade_shares_destination.to_csv('misc/discussion_tau_smaller_one/trade_shares_destination.csv')
corresponding_trade_shares_origin.to_csv('misc/discussion_tau_smaller_one/trade_shares_origin.csv')

#%%


grad, jac = load_grad_jac('./calibration_results/34/')
# grad, jac = load_grad_jac('./calibration_results/35/')
fig,ax = plt.subplots(figsize = (16,8))
ax.plot(m34.get_signature_list(), jac[:,7])
# plt.xticks(ticks = np.arange(len(m34.get_signature_list())), labels = m34.get_signature_list())
plt.title('Derivative of moments with respect to deta_US')
plt.show()

#%%
m35.list_of_moments = ['GROWTH','SRDUS', 'KM',
                    'OUT','RD','RP','SPFLOW','SRGDP','STFLOW']

grad, jac = load_grad_jac('./calibration_results/35/')
# grad, jac = load_grad_jac('./calibration_results/35/')
fig,ax = plt.subplots(figsize = (16,8))
ax.plot(m35.get_signature_list(), jac[:,106])
# plt.xticks(ticks = np.arange(len(m34.get_signature_list())), labels = m34.get_signature_list())
plt.title('Derivative of moments with respect to f_o')
plt.show()

#%%
fig,ax = plt.subplots(figsize = (16,8))
ax.plot(m.get_signature_list(), jac[:,107])
# plt.xticks(ticks = np.arange(len(m34.get_signature_list())), labels = m34.get_signature_list())
plt.title('Derivative of moments with respect to f_o')
plt.show()

#%%

pd.DataFrame(test_ls.grad, index = p.get_signature_list()
              ).to_csv('calibration_results_matched_trade_flows/16/grad.csv')
pd.DataFrame(test_ls.jac, 
              columns = p.get_signature_list(), 
              index=m.get_signature_list()
              ).to_csv('calibration_results_matched_trade_flows/16/jac.csv')

#%%
jac = optimize.approx_fprime(p_sol.make_p_vector(), 
                              calibration_func,1e-16,
                              p,m,p.guess,hist,start_time,False)

signature_p = []
for param in p.calib_parameters: 
    signature_p.extend([param]*np.array(getattr(p,param))[p.mask[param]].size)

fig, ax = plt.subplots(figsize = (14,10))
ax.plot(jac[-1,:])
plt.xticks([i for i in range(len(signature_p))])
ax.set_xticklabels(signature_p, fontsize = 20, rotation = 90)
ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
plt.title('Derivatives of the deviation of the moment OUT with respect to the parameters :', fontsize = 20)
plt.savefig('calibration_results_matched_trade_flows/out_study/OUT derivatives')
plt.show()

signature_m = m.get_signature_list()
fig, ax = plt.subplots(figsize = (18,10))
ax.plot(jac[:,14])
plt.xticks([i for i in range(len(signature_m))])
ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
plt.title('Derivatives of the  deviations of the moments with respect to the parameter : fe', fontsize = 20)
plt.show()

signature_m = m.get_signature_list()
fig, ax = plt.subplots(figsize = (18,10))
ax.plot(jac[:,22])
plt.xticks([i for i in range(len(signature_m))])
ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
plt.title('Derivatives of the  deviations of the moments with respect to the parameter : fo', fontsize = 20)
plt.show()

signature_m = m.get_signature_list()
fig, ax = plt.subplots(figsize = (18,10))
ax.plot(jac[:,23])
plt.xticks([i for i in range(len(signature_m))])
ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
plt.title('Derivatives of the  deviations of the moments with respect to the parameter : g_o', fontsize = 20)
plt.show()

for i in range(7):
    signature_m = m.get_signature_list()
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(jac[:,15+i])
    plt.xticks([i for i in range(len(signature_m))])
    ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
    ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
    plt.title('Derivatives of the  deviations of the moments with respect to the parameter : T_'+str(i), fontsize = 20)
    plt.show()
    
for i in range(7):
    signature_m = m.get_signature_list()
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(jac[:,i])
    plt.xticks([i for i in range(len(signature_m))])
    ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
    ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
    plt.title('Derivatives of the  deviations of the moments with respect to the parameter : eta_'+str(i), fontsize = 20)
    plt.show()
    
for i in range(7):
    signature_m = m.get_signature_list()
    fig, ax = plt.subplots(figsize = (18,10))
    ax.plot(jac[:,7+i])
    plt.xticks([i for i in range(len(signature_m))])
    ax.set_xticklabels(signature_m, fontsize = 10, rotation = 90)
    ax.hlines(xmin=0,xmax=25,y=0,ls='--',color='k')
    plt.title('Derivatives of the  deviations of the moments with respect to the parameter : delta_'+str(i), fontsize = 20)
    plt.show()

#%% dependance on T

p = parameters(n=7,s=2)
list_T = []
list_sol = []
x0 = None

for T in np.linspace(1.297,1.3,1001):
    print(T)
    try:
        x0 = p.guess
    except:
        pass
    p.T = np.ones(p.N)*T
    sol = fixed_point_solver(p,x0,cobweb_anim=False,
                              accelerate=False,
                              accelerate_when_stable=True,
                              cobweb_qty='psi_star',
                              plot_cobweb=False,
                              plot_convergence=False,
                              # apply_bound_psi_star=True
                              )
    sol_c = var.var_from_vector(sol.x, sol.p)   
    sol_c.num_scale_solution(p)
    if sol_c.Z.sum() > p.expenditure.sum():
        break
    p.guess = sol_c.vector_from_var()
    list_sol.append(sol_c)
    list_T.append(p.T)
sol_c.compute_non_solver_quantities(p)
fig,ax = plt.subplots(figsize = (12,8))
  
ax.plot([T.mean() for T in list_T],
          [sol.Z.sum()*p.unit for sol in list_sol], lw =3
          )
ax.set_xlabel('Average technology T', fontsize = 20)
ax.set_ylabel('World gross expenditure Z', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=20)

plt.show()

#%% T calibration try

p = parameters(n=7,s=2)

def calibration_func(T,p,x0=None):
    p.T = T
    try:
        x0 = p.guess
    except:
        pass
    sol = fixed_point_solver(p,x0,tol=1e-14,
                              accelerate=False,
                              accelerate_when_stable=True,
                              plot_cobweb=False,
                              plot_convergence=False,
                              disp_summary=False,
                              )
    print(sol.time)
    # p.guess = sol.x
    # try:
    #     print(p.guess.mean())
    # except:
    #     pass
    if sol.status == 'successful':     
        sol_c = var.var_from_vector(sol.x, sol.p)   
        sol_c.num_scale_solution(p)
        p.guess = sol_c.vector_from_var()
        return p.data.expenditure.values/p.data.expenditure.sum() - sol_c.Z/sol_c.Z.sum() 
    else:
        return np.inf

p = parameters(n=7,s=2)

lb = np.full_like(p.T, 1e-12)
ub = np.full_like(p.T, 1e12)
bounds = (lb,ub)

start = time.perf_counter()
test_ls = optimize.least_squares(fun = calibration_func, 
                    x0 = p.T, 
                    args = (p,), 
                    bounds = bounds,
                    # method= 'trf',
                    # loss='arctan',
                    # max_nfev=1e3,
                    # ftol=1e-10, 
                    # xtol=0, 
                    # gtol=1e-10,
                    
                    verbose = 2)
finish = time.perf_counter()
print('minimizing time',finish-start)
p.T = test_ls.x
test_sol = fixed_point_solver(p,x0=p.guess,
                          accelerate=False,
                          accelerate_when_stable=True,
                          plot_cobweb=False,
                          plot_convergence=False,
                          disp_summary=True,
                          )
test_sol_c = var.var_from_vector(test_sol.x, p)
test_sol_c.compute_non_solver_quantities(p)

fig, ax = plt.subplots(figsize = (12,8))
ax2 = ax.twinx()
ax.plot(p.data.expenditure.values/p.data.expenditure.sum(), label = 'Data Z',lw=3)
ax.plot(test_sol_c.Z/test_sol_c.Z.sum() , label = 'Calibrated Z',lw=3)
ax2.semilogy(p.T, label='Technology T', ls = '--', color = 'r',lw=3)
ax.set_xticks([i for i in range(0,p.N)])
ax.set_xticklabels(p.countries,fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

ax.legend(loc=(-0.4,0.8),fontsize=20)
ax2.legend(loc=(1.1,0.8),fontsize=20)

plt.title('Partial calibration of T targeting Z',fontsize=20)

plt.show()

#%%

if np.any(np.isnan(sol.x)):
    sol = fixed_point_solver(p,x0=None,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='l_R',
                                  disp_summary=False,
                                  safe_convergence=0.001
                                  )   
new_guess = 0
while np.any(np.isnan(sol.x)) and new_guess<len(collec_of_guess):
    # print(new_guess)
    sol = fixed_point_solver(p,x0=collec_of_guess[new_guess],tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='l_R',
                                  disp_summary=False,
                                  safe_convergence=0.001
                                  )
    new_guess += 1
if np.any(np.isnan(sol.x)) or sol.status == 'failed':
    sol = fixed_point_solver(p,x0=None,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='psi_star',
                            plot_convergence=False,
                            plot_cobweb=False,
                            disp_summary=False,
                            safe_convergence=0.1)
if np.any(np.isnan(sol.x)) or sol.status == 'failed':
    sol = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='psi_star',
                            plot_convergence=False,
                            plot_cobweb=False,
                            disp_summary=False,
                            safe_convergence=0.001,
                            damping=100
                              # apply_bound_psi_star=True
                            )
