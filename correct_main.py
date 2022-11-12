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
import matplotlib as mpl
import seaborn as sns
# sns.set()
# sns.set_context('talk')
# sns.set_style('white')

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
        self.sigma = np.ones(S)*3  #
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
        self.nu_tilde = np.ones(S)*0.1
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
                        'theta':3,
                        'rho':0,
                        'gamma':co,
                        'zeta':0,
                        'nu':0,
                        'nu_tilde':0,
                        'kappa':co,
                        'k':1+co,
                        'fe':co,
                        'fo':0,
                        'delta':1e-2,
                        'g_0':0,
                        'alpha':co,
                         'beta':co,
                         'T':co,
                         'eta':co}
        self.ub_dict = {'sigma':5,
                        'theta':30,
                        'rho':0.5,
                        'gamma':cou,
                        'zeta':1,
                        'nu':cou,
                        'nu_tilde':cou,
                        'kappa':1-co,
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
            
    def update_sigma_with_SRDUS_target(self,m):
        self.sigma[1] = 1+m.SRDUS_target/(m.sales_mark_up_US_target - 1)
            
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
            setattr(self,'calib_parameters',df[0].to_list())
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
        # self.PSI_CL[self.PSI_CL<0] = 0 #!!!!
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
    
    def compute_non_solver_aggregate_qualities(self,p): 
        prefact = p.k * p.eta * self.l_R**(1-p.kappa)/(p.k-1)
        A = (self.g_s + p.nu + p.zeta)
        A_tilde = (self.g_s + p.nu_tilde + p.zeta)
        self.PSI_MPND = np.einsum('is,nis,ns->nis',
                                  prefact,
                                  np.divide(1, self.psi_star**(p.k-1), out=np.zeros_like(
                                      self.psi_star), where=self.psi_star != np.inf),
                                  1/(A[None,:]+p.delta))
        self.PSI_MPL = np.einsum('s,nis,ns->nis',
                                 p.nu,
                                 self.PSI_MPND,
                                 1/(A_tilde[None,:]+p.delta))
        self.PSI_MPD = np.einsum('s,nis,ns->nis',
                                 p.nu,
                                 self.PSI_MPL,
                                 1/(p.delta+self.g_s[None,:]+p.zeta[None,:]))
        numerator_A = np.einsum('is,nis->nis',
                                prefact,
                                1 - np.divide(1, self.psi_star**(p.k-1), out=np.zeros_like(
                                    self.psi_star), where=self.psi_star != np.inf))
        numerator_B= np.einsum('ns,nis->nis',
                               p.delta,
                               self.PSI_MPND)
        self.PSI_MNP = (numerator_A + numerator_B)/A[None,None,:]
        
    
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
        self.compute_non_solver_aggregate_qualities(p)

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

def eps(x):
    return 1-np.exp(-x)
    
class moments:
    def __init__(self,list_of_moments = None, n=7, s=2):
        self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'][:n]+[i for i in range(n-7)]
        self.sectors = ['Non patent', 'Patent']+['other'+str(i) for i in range(s-2)]
        if list_of_moments is None:
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'OUT', 'KM', 'RD','RD_US','RD_RUS', 'RP',
                               'SRDUS', 'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US',
                               'SPFLOWDOM_RUS', 'SPFLOW_RUS','SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST',
                               'JUPCOSTRD','SINNOVPATUS','TO','TE','DOMPATRATUSEU','DOMPATUS','DOMPATEU',
                               'SPATORIG','SPATDEST','TWSPFLOW','TWSPFLOWDOM']
        else:
            self.list_of_moments = list_of_moments
        # self.weights_dict = {'GPDIFF':1, 
        #                      'GROWTH':1, 
        #                      'KM':5, 
        #                      'OUT':4, 
        #                      'RD':5, 
        #                      'RP':3, 
        #                      'SPFLOW':1, 
        #                      'SRDUS':3, 
        #                      'SRGDP':1, 
        #                      # 'STFLOW':1,
        #                      'JUPCOST':1,
        #                       'TP':1,
        #                       'Z':1,
        #                      # 'SDOMTFLOW':1,
        #                      'SINNOVPATEU':1,
        #                      'SINNOVPATUS':1,
        #                       'NUR':1,
        #                       'TO':3,
        #                       'TE':3
        #                      }
        self.weights_dict = {'GPDIFF':1, 
                             'GROWTH':1, 
                             'KM':5, 
                             'OUT':4, 
                             'RD':3, 
                             'RD_US':3, 
                             'RD_RUS':3, 
                             'RP':3, 
                             'SPFLOW':1, 
                             'SPFLOW_US':1, 
                             'SPFLOW_RUS':1, 
                             'SPFLOWDOM':1, 
                             'SPFLOWDOM_US':1, 
                             'SPFLOWDOM_RUS':1, 
                             'SRDUS':1,
                             'SRGDP':1, 
                             'SRGDP_US':1, 
                             'SRGDP_RUS':1, 
                             # 'STFLOW':1,
                             'JUPCOST':1,
                             'JUPCOSTRD':1,
                              'TP':1,
                              'Z':1,
                             # 'SDOMTFLOW':1,
                             'SINNOVPATEU':1,
                             'SINNOVPATUS':1,
                              'NUR':1,
                              'TO':3,
                              'TE':3,
                              'DOMPATRATUSEU':2,
                              'DOMPATUS':3,
                              'DOMPATEU':3,
                              'SPATORIG':2,
                              'SPATDEST':2,
                              'TWSPFLOW':1,
                              'TWSPFLOWDOM':1,
                             }
        
        # self.total_weight = sum([self.weights_dict[mom] for mom in self.list_of_moments])
        
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'OUT':pd.Index(['scalar']), 
                    'RD':pd.Index(self.countries, name='country'), 
                    'RD_US':pd.Index(['scalar']), 
                    'RD_RUS':pd.Index(self.countries, name='country'), 
                    'RP':pd.Index(self.countries, name='country'), 
                    'SPFLOW':pd.MultiIndex.from_tuples([(c1,c2) for c1 in self.countries for c2 in self.countries if c1 != c2]
                                            , names=['destination','origin']),
                    'SPFLOW_US':pd.Index(self.countries, name='country'),
                    'SPFLOW_RUS':pd.MultiIndex.from_tuples([(c1,c2) for c1 in self.countries for c2 in self.countries if c1 != c2]
                                            , names=['destination','origin']),
                    'SPFLOWDOM':pd.MultiIndex.from_product([self.countries,self.countries]
                                                      , names=['destination','origin']),
                    'SPFLOWDOM_US':pd.Index(['scalar']),
                    'SPFLOWDOM_RUS':pd.MultiIndex.from_product([self.countries,self.countries]
                                                      , names=['destination','origin']),
                    'TWSPFLOW':pd.MultiIndex.from_tuples([(c1,c2) for c1 in self.countries for c2 in self.countries if c1 != c2]
                                            , names=['destination','origin']),
                    'TWSPFLOWDOM':pd.MultiIndex.from_product([self.countries,self.countries]
                                                      , names=['destination','origin']),
                    'SRDUS':pd.Index(['scalar']), 
                    'JUPCOST':pd.Index(['scalar']), 
                    'JUPCOSTRD':pd.Index(['scalar']), 
                    'SRGDP':pd.Index(self.countries, name='country'), 
                    'SRGDP_US':pd.Index(['scalar']), 
                    'SRGDP_RUS':pd.Index(self.countries, name='country'), 
                    # 'STFLOW':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                    #                                  , names=['destination','origin','sector']),
                    'TP':pd.Index(['scalar']),
                    'Z':pd.Index(self.countries, name='country'),
                    'DOMPATRATUSEU':pd.Index(self.countries, name='country'),
                    'SPATORIG':pd.Index(self.countries, name='country'),
                    'SPATDEST':pd.Index(self.countries, name='country'),
                    'turnover':pd.Index(self.countries, name='country'),
                    # 'SDOMTFLOW':pd.MultiIndex.from_product([self.countries,self.sectors]
                    #                                  , names=['country','sector']),
                    'SINNOVPATEU':pd.Index(['scalar']),
                    'SINNOVPATUS':pd.Index(['scalar']),
                    'TO':pd.Index(['scalar']),
                    'TE':pd.Index(['scalar']),
                    'DOMPATUS':pd.Index(['scalar']),
                    'DOMPATEU':pd.Index(['scalar']),
                    'NUR':pd.Index(['scalar'])}
        
        self.shapes = {'SPFLOW':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM':(len(self.countries),len(self.countries)),
                       'SPFLOW_RUS':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM_RUS':(len(self.countries),len(self.countries)),
                       'TWSPFLOW':(len(self.countries),len(self.countries)-1),
                       'TWSPFLOWDOM':(len(self.countries),len(self.countries)),
                       }
        
        self.drop_CHN_IND_BRA_ROW_from_RD = False
        self.add_domestic_US_to_SPFLOW = False
        self.add_domestic_EU_to_SPFLOW = False
    
    def get_signature_list(self):
        l = []
        for mom in self.list_of_moments:
            l.extend([mom]*np.array(getattr(self,mom)).size)
        return l
    
    @staticmethod
    def get_list_of_moments():
        return ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD','RD_US','RD_RUS', 'RP', 
                'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US',
                'SPFLOWDOM_RUS', 'SPFLOW_RUS','DOMPATUS','DOMPATEU',
                'SRDUS', 'SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST','JUPCOSTRD', 'TP', 'Z', 
                'SINNOVPATEU','SINNOVPATUS','TO','TE','NUR','DOMPATRATUSEU',
                'SPATDEST','SPATORIG','TWSPFLOW','TWSPFLOWDOM']
    
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
        # self.STFLOW_target = (self.ccs_moments.trade/
        #                       self.ccs_moments.trade.sum()).values.reshape(N,N,S)
        self.SPFLOW_target = self.cc_moments.query("destination_code != origin_code")['patent flows'].values
        self.SPFLOW_target = self.SPFLOW_target.reshape((N,N-1))/self.SPFLOW_target.sum()
        # self.SPFLOW_US_target = self.SPFLOW_target[0,:]
        # self.SPFLOW_RUS_target = self.SPFLOW_target/self.SPFLOW_US_target
        self.SPFLOW_US_target = self.cc_moments.loc[1]['patent flows'].values/self.cc_moments.query("destination_code != origin_code")['patent flows'].sum()
        self.SPFLOW_RUS_target = (pd.DataFrame(self.cc_moments['patent flows']/self.cc_moments.loc[1]['patent flows']))
        self.SPFLOW_RUS_target = self.SPFLOW_RUS_target.query("destination_code != origin_code")['patent flows'].values.reshape((N,N-1))
        self.SPFLOWDOM_target = self.cc_moments['patent flows'].values
        self.SPFLOWDOM_target = self.SPFLOWDOM_target.reshape((N,N))/self.SPFLOWDOM_target.sum()
        self.SPFLOWDOM_US_target = self.SPFLOWDOM_target[0,0]
        self.SPFLOWDOM_RUS_target = self.SPFLOWDOM_target/self.SPFLOWDOM_US_target
        self.OUT_target = self.c_moments.expenditure.sum()/self.unit
        self.SRGDP_target = (self.c_moments.gdp/self.c_moments.price_level).values \
                            /(self.c_moments.gdp/self.c_moments.price_level).sum()
        self.SRGDP_US_target = self.SRGDP_target[0]
        self.SRGDP_RUS_target = self.SRGDP_target/self.SRGDP_US_target
        self.RP_target = self.c_moments.price_level.values
        self.RD_target = self.c_moments.rnd_gdp.values
        self.RD_US_target = self.RD_target[0]
        self.RD_RUS_target = self.RD_target/self.RD_US_target
        self.KM_target = self.moments.loc['KM'].value
        self.NUR_target = self.moments.loc['NUR'].value
        self.SRDUS_target = self.moments.loc['SRDUS'].value
        self.GPDIFF_target = self.moments.loc['GPDIFF'].value 
        self.GROWTH_target = self.moments.loc['GROWTH'].value 
        self.TE_target = self.moments.loc['TE'].value 
        self.TO_target = np.array(0.05)
        # self.GROWTH_target = self.GROWTH_target*10
        self.Z_target = self.c_moments.expenditure.values/self.unit
        self.JUPCOST_target = self.moments.loc['JUPCOST'].value
        self.JUPCOSTRD_target = self.moments.loc['JUPCOST'].value/(self.c_moments.loc[1,'rnd_gdp']*self.c_moments.loc[1,'gdp'])
        self.TP_target = self.moments.loc['TP'].value
        self.TP_data = self.cc_moments['patent flows'].sum()
        self.DOMPATEU_target = self.cc_moments.loc[(2,2),'patent flows']/self.cc_moments.xs(2,level=1)['patent flows'].sum()
        self.DOMPATUS_target = self.cc_moments.loc[(1,1),'patent flows']/self.cc_moments.xs(1,level=1)['patent flows'].sum()
        self.inter_TP_data = self.cc_moments.query("destination_code != origin_code")['patent flows'].sum()
        self.SINNOVPATEU_target = self.moments.loc['SINNOVPATEU'].value
        self.SINNOVPATUS_target = self.moments.loc['SINNOVPATUS'].value
        # self.SDOMTFLOW_target = self.ccs_moments.query("destination_code == origin_code").trade.values#/self.ccs_moments.trade.sum()
        # self.SDOMTFLOW_target = self.SDOMTFLOW_target.reshape(N,S)/self.unit
        self.sales_mark_up_US = self.moments.loc['sales_mark_up_US'].value
        self.sales_mark_up_US_target = self.moments.loc['sales_mark_up_US'].value
        self.DOMPATRATUSEU_target = (self.cc_moments.query("destination_code == origin_code")['patent flows']\
            /(self.cc_moments.loc[1]['patent flows'].sum() + self.cc_moments.loc[2]['patent flows'].sum())).values
        self.SPATORIG_target = self.cc_moments['patent flows'].groupby('origin_code').sum().values\
            /self.cc_moments['patent flows'].sum()
        self.SPATDEST_target = self.cc_moments['patent flows'].groupby('destination_code').sum().values\
            /self.cc_moments['patent flows'].sum()
        self.TWSPFLOW_target = self.SPFLOW_target*self.ccs_moments.loc[:,:,1].query("destination_code != origin_code")['trade'].values.reshape((N,N-1))\
            /self.ccs_moments.loc[:,:,1].query("destination_code != origin_code")['trade'].sum()
        self.TWSPFLOWDOM_target = self.SPFLOWDOM_target*self.ccs_moments.loc[:,:,1]['trade'].values.reshape((N,N))\
            /self.ccs_moments.loc[:,:,1]['trade'].sum()
    
    def load_run(self,path):
        df = pd.read_csv(path+'list_of_moments.csv')
        self.list_of_moments = df['moments'].tolist()
        df.set_index('moments',inplace=True)
        for mom in self.list_of_moments:
            self.weights_dict[mom] = df.loc[mom, 'weights']
            df_mom = pd.read_csv(path+mom+'.csv')
            if len(df_mom) == 1:
                mom_target = df_mom.iloc[0].target
                mom_value = df_mom.iloc[0].moment
            else:
                mom_target = df_mom['target'].values
                mom_value = df_mom['moment'].values
            try:
                mom_target = mom_target.reshape(self.shapes[mom])
                mom_value = mom_value.reshape(self.shapes[mom])
            except:
                pass
            setattr(self,mom+'_target',mom_target)
            setattr(self,mom,mom_value)
    
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
                scalar_moments_ratio.append(getattr(self,mom+'_deviation')/self.weights_dict[mom])
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
                        if mom not in  ['SPFLOW','SPFLOW_RUS','SPFLOWDOM','SPFLOWDOM_RUS','TWSPFLOW','TWSPFLOWDOM']:
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
            
    def write_moments(self, path):
        for mom in self.list_of_moments:
            df = pd.DataFrame(data = {'target':getattr(m,mom+'_target').ravel(),
                                      'moment':getattr(m,mom).ravel()})
            df.to_csv(path+mom+'.csv',index=False)
        df = pd.DataFrame(data = {'moments':self.list_of_moments,
                                  'weights':[self.weights_dict[mom] for mom in self.list_of_moments]})
        df.to_csv(path+'list_of_moments.csv',index=False)
    
    @staticmethod
    def compare_moments(dic = None, lis = None, save_path = None, contin_cmap = False,list_of_moments=None):
        if dic is None and lis is not None:
            coms = ['m'+str(i) for i,_ in enumerate(lis)]
            moms_c = lis
        elif lis is None and dic is not None:
            coms = [k for k in dic.keys()]
            moms_c = [v for v in dic.values()]
        n_col = int(np.floor(len(dic)/25))+1
        colors = sns.color_palette("Spectral", n_colors = len(dic))
        scalar_moments_collection = [ [] for _ in range(len(coms)) ]
        scalar_moments = []
        if list_of_moments is None:
            list_of_moments = moms_c[0].list_of_moments
        for mom in list_of_moments:
            print(mom)
            if np.array(getattr(moms_c[0],mom)).size == 1:
                scalar_moments.append(mom)
                print('\n'+mom+' target :',getattr(moms_c[0],mom+'_target'))
                for i,mom_c in enumerate(moms_c):
                    # scalar_moments_collection[i].append(getattr(mom_c,mom)/getattr(mom_c,mom+'_target'))
                    scalar_moments_collection[i].append((getattr(mom_c,mom+'_deviation')/mom_c.weights_dict[mom])**2)
                    print(coms[i]+' : ',getattr(mom_c,mom))
            else:
                fig,ax = plt.subplots(figsize = (12,8))
                for i,mom_c in enumerate(moms_c):
                    if contin_cmap:
                        ax.scatter(getattr(mom_c,mom+'_target').ravel(),getattr(mom_c,mom).ravel()
                                   ,label = coms[i],lw=2,marker = 'x',color=colors[i])
                    else:
                        ax.scatter(getattr(mom_c,mom+'_target').ravel(),getattr(mom_c,mom).ravel()
                                   ,label = coms[i],lw=2,marker = 'x')
                ax.plot([getattr(mom_c,mom+'_target').min(),
                          getattr(mom_c,mom+'_target').max()]
                        ,[getattr(mom_c,mom+'_target').min(),
                          getattr(mom_c,mom+'_target').max()], 
                        ls = '--', lw=1, color = 'k')
                if mom not in ['SPFLOWDOM','SPFLOW','SPFLOWDOM_RUS','SPFLOW_RUS',
                               'STFLOW','SDOMTFLOW','TWSPFLOWDOM','TWSPFLOW']:
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
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
                if save_path is not None:
                    plt.tight_layout()
                    plt.savefig(save_path+mom)
                plt.show()
            if mom == 'TO':
                fig,ax = plt.subplots(figsize = (12,8))
                for i,mom_c in enumerate(moms_c):
                    if contin_cmap:
                        ax.plot(mom_c.idx['turnover'],mom_c.turnover[:,1],label = coms[i],lw=2,color=colors[i])
                    else:
                        ax.plot(mom_c.idx['turnover'],mom_c.turnover[:,1],label = coms[i],lw=2)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
                plt.title('turnover')
                if save_path is not None:
                    plt.tight_layout()
                    plt.savefig(save_path+'turnover')
                plt.show()
        fig,ax = plt.subplots(figsize = (12,8))
        for i,scalars_of_one_run in enumerate(scalar_moments_collection):
            if contin_cmap:
                ax.scatter(scalar_moments,scalars_of_one_run, label = coms[i],color=colors[i])
            else:
                ax.scatter(scalar_moments,scalars_of_one_run, label = coms[i])
        # ax.plot(scalar_moments,np.ones_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        ax.plot(scalar_moments,np.zeros_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        # if np.any([np.any(np.array(scalars_of_one_run)>10) 
        #            for scalars_of_one_run in scalar_moments_collection]):
        #     plt.yscale('log')
        plt.title('scalar moments deviation')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
        if save_path is not None:
            plt.tight_layout()
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
        # pflow = remove_diag(var.pflow)
        pflow = var.pflow
        self.SPFLOWDOM = pflow/pflow.sum()
        inter_pflow = remove_diag(var.pflow)
        self.SPFLOW = inter_pflow/inter_pflow.sum()
        
        self.SPFLOW_US = pflow[0,:]/inter_pflow.sum()
        RUS = pflow/pflow[0,:]
        self.SPFLOW_RUS = remove_diag(RUS)
        
        self.SPFLOWDOM_US = self.SPFLOWDOM[0,0]
        self.SPFLOWDOM_RUS = self.SPFLOWDOM/self.SPFLOWDOM_US
        
    def compute_TWSPFLOW(self,var,p):
        # numerator = np.einsum('nis,is,is->nis',
        #                       var.psi_star[...,1:]**(-p.k),
        #                       p.eta[...,1:],
        #                       var.l_R[...,1:]**(1-p.k)
        #                       )
        # numerator = remove_diag(numerator)
        # pflow = remove_diag(var.pflow)
        pflow = var.pflow
        self.TWSPFLOWDOM = pflow*p.trade_flows[...,1]/(pflow.sum()*p.trade_flows[...,1].sum())
        inter_pflow = remove_diag(var.pflow)
        off_diag_trade_flows = remove_diag(p.trade_flows[...,1])
        self.TWSPFLOW = inter_pflow*off_diag_trade_flows/(inter_pflow.sum()*off_diag_trade_flows.sum())
    
    # def compute_SPFLOWRUS(self,var,p):
        
        
    def compute_OUT(self,var,p):
        self.OUT = var.Z.sum()
        
    def compute_SRGDP(self,var,p):
        numerator = var.gdp/var.price_indices
        self.SRGDP = numerator/numerator.sum()
        self.SRGDP_US = self.SRGDP[0]
        self.SRGDP_RUS = self.SRGDP/self.SRGDP_US
        
    def compute_RP(self,var,p):
        self.RP = var.price_indices/var.price_indices[0]
        
    def compute_RD(self,var,p):
        numerator = var.w[:,None]*var.l_R + np.einsum('i,ins->is',var.w,var.l_Ao)\
            + np.einsum('n,ins->is',var.w,var.l_Ao)
        self.RD = np.einsum('is,i->i',
                            numerator,
                            1/var.gdp)
        self.RD_US = self.RD[0]
        self.RD_RUS = self.RD/self.RD_US
    
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
        self.JUPCOSTRD = self.JUPCOST/(self.RD[0]*var.gdp[0])
        
    def compute_TP(self,var,p):
        self.TP = var.pflow.sum()
        inter_pflow = remove_diag(var.pflow)
        self.inter_TP = inter_pflow.sum()
        
    # def compute_SDOMTFLOW(self,var,p):
    #     self.SDOMTFLOW = np.diagonal(var.X).transpose()#/var.X.sum()
    
    def compute_SINNOVPATEU(self,var,p):
        self.SINNOVPATEU = var.share_innov_patented[1,1]
        
    def compute_SINNOVPATUS(self,var,p):
        self.SINNOVPATUS = var.share_innov_patented[0,0]
        
    def compute_TO(self,var,p):
        delt = 5
        self.delta_t = delt
        PHI = var.phi**p.theta[None,None,:]
        
        num_brack_A = var.PSI_CL*eps(p.nu_tilde*delt)[None,None,:]
        num_brack_B = var.PSI_MNP*(eps(p.nu_tilde*delt)*eps(p.nu*delt))[None,None,:]
        num_brack_C = var.PSI_MPND*(eps(p.delta*delt)
                                    *eps(p.nu*delt)[None,:]
                                    *eps(p.nu_tilde*delt)[None,:])[:,None,:]
        num_brack_D = var.PSI_MPL*(eps(p.delta*delt)
                                    *eps(p.nu_tilde*delt)[None,:])[:,None,:]
        num_brack_E = var.PSI_MPD*eps(p.nu*delt)[None,None,:]
        
        num_brack = (num_brack_A +num_brack_B + num_brack_C + num_brack_D + num_brack_E)
        
        num_sum = np.einsum('nis,njs->ns',
                            num_brack,
                            PHI
                            ) - \
                  np.einsum('nis,ns->ns',
                            num_brack,
                            np.diagonal(PHI).transpose()
                            ) - \
                  np.einsum('nis,nis->ns',
                            num_brack,
                            PHI
                            ) + \
                  np.einsum('ns,ns->ns',
                            np.diagonal(num_brack).transpose(),
                            np.diagonal(PHI).transpose(),
                            )
                  
        num = np.einsum('ns,ns->ns',
                        PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:],
                        num_sum
                        )
        
        denom_A = np.einsum('nis,ns,ns->nis',
                                  PHI,
                                  var.PSI_CD,
                                  PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:]
                                  )
        
        denom_B_a = var.PSI_MNP*np.exp(-delt*p.nu)[None,None,:]
        denom_B_b = var.PSI_MPND*(np.exp(-delt*p.nu)[None,:]
                                  +eps(p.nu*delt)[None,:]*np.exp(-delt*p.delta))[:,None,:]
        denom_B_c = (var.PSI_MPL+var.PSI_MPD)*np.exp(-delt*p.delta)[:,None,:]
        denom_B = np.einsum('nis,nis,s->nis',
                            denom_B_a + denom_B_b + denom_B_c,
                            var.phi**(p.sigma-1)[None,None,:],
                            (p.sigma/(p.sigma-1))**(1-p.sigma)
                            )
        
        denom_C_a = var.PSI_CL*np.exp(-delt*p.nu_tilde)[None,None,:]
        denom_C_b = var.PSI_MNP*(np.exp(-delt*p.nu_tilde)*eps(delt*p.nu))[None,None,:]
        denom_C_c = var.PSI_MPND*(eps(p.delta*delt)
                                    *eps(p.nu*delt)[None,:]
                                    *np.exp(-p.nu_tilde*delt)[None,:])[:,None,:]
        denom_C_d = var.PSI_MPL*(eps(p.delta*delt)
                                    *np.exp(-p.nu_tilde*delt)[None,:])[:,None,:]
        denom_C = np.einsum('nis,nis->nis',
                            denom_C_a + denom_C_b + denom_C_c + denom_C_d,
                            var.phi**(p.sigma-1)[None,None,:]
                            )
        
        denom_D_sum = np.einsum('nis,njs->nis',
                                num_brack,
                                PHI
                                ) - \
                      np.einsum('nis,ns->nis',
                                num_brack,
                                np.diagonal(PHI).transpose()
                                )
        
        denom_D = np.einsum('ns,nis->nis',
                        PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:],
                        denom_D_sum
                        )
        
        denom = np.einsum('nis->ns',
                          denom_A + denom_B + denom_C + denom_D
                          )
        
        self.turnover = num/denom
        self.TO = self.turnover[0,1]
        
    def compute_TE(self,var,p):
        out_diag_trade_flows_shares = remove_diag(var.X_M + var.X_CL)
        self.TE = ( (p.theta[None,None,:] - np.einsum('s,nis->nis',
                                                    p.theta-(p.sigma-1),
                                                    out_diag_trade_flows_shares)
                    ).sum(axis=1).sum(axis=0) )[1]/(p.N*(p.N-1))
        
    def compute_NUR(self,var,p):
        self.NUR = p.nu[1]
        
    def get_sales_mark_up_US_from_sigma(self,p):
        self.sales_mark_up_US = 1+m.SRDUS_target/(p.sigma[1] - 1)
        self.sales_mark_up_US_target = 1+m.SRDUS_target/(p.sigma[1] - 1)
        
    def compute_DOMPATRATUSEU(self,var,p):
        self.DOMPATRATUSEU = np.diagonal(var.pflow)/(var.pflow[0,:].sum()+var.pflow[1,:].sum())
        
    def compute_SPATORIG(self,var,p):
        self.SPATORIG = var.pflow.sum(axis=0)/var.pflow.sum()
    
    def compute_SPATDEST(self,var,p):
        self.SPATDEST = var.pflow.sum(axis=1)/var.pflow.sum()
    
    def compute_DOMPATEU(self,var,p):
        self.DOMPATEU = var.pflow[1,1]/var.pflow[:,1].sum()
        
    def compute_DOMPATUS(self,var,p):
        self.DOMPATUS = var.pflow[0,0]/var.pflow[:,0].sum()
        
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
        self.compute_SINNOVPATUS(var,p)
        self.compute_NUR(var,p)
        self.compute_TO(var,p)
        self.compute_TE(var,p)
        self.compute_DOMPATRATUSEU(var,p)
        self.compute_SPATDEST(var,p)
        self.compute_SPATORIG(var,p)
        self.compute_TWSPFLOW(var, p)
        self.compute_DOMPATEU(var, p)
        self.compute_DOMPATUS(var, p)
        
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
            if mom != 'GPDIFF' and mom != 'TO' and mom != 'TE' and mom != 'GROWTH':
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
            # elif mom == 'SPFLOW':
            #     setattr(self,
            #             mom+'_deviation',
            #             self.weights_dict[mom]*np.abs(getattr(self,mom)/getattr(self,mom+'_target')-1)
            #             /getattr(self,mom+'_target').size**(1/2)
            #             )
            else:
                mo = getattr(self,mom)
                tar = getattr(self,mom+'_target')
                setattr(self,
                        mom+'_deviation',
                        self.weights_dict[mom]*np.abs(mo-tar)/tar
                        /getattr(self,mom+'_target').size
                        )
                # print(mo,tar,self.weights_dict[mom]*np.abs(mo-tar)/tar)
        if self.drop_CHN_IND_BRA_ROW_from_RD:
            self.RD_deviation = self.RD_deviation[:3]
            self.RD_RUS_deviation = self.RD_RUS_deviation[:3]    
            
        if self.add_domestic_EU_to_SPFLOW or self.add_domestic_US_to_SPFLOW:
            current_inter_PFLOW = self.SPFLOW.ravel()*self.inter_TP
            current_target = self.SPFLOW_target.ravel()*self.inter_TP_data
            if self.add_domestic_US_to_SPFLOW:
                current_inter_PFLOW = np.append(current_inter_PFLOW,self.SPFLOWDOM[0,0]*self.TP)
                current_target = np.append(current_target,self.SPFLOWDOM_target[0,0]*self.TP_data)
            if self.add_domestic_EU_to_SPFLOW:
                current_inter_PFLOW = np.append(current_inter_PFLOW,self.SPFLOWDOM[1,1]*self.TP)
                current_target = np.append(current_target,self.SPFLOWDOM_target[1,1]*self.TP_data)
            new_SPFLOW = current_inter_PFLOW/current_inter_PFLOW.sum()
            new_target = current_target/current_target.sum()
            self.SPFLOW_deviation = self.weights_dict['SPFLOW']*np.abs(np.log(new_SPFLOW/new_target))/new_target.size**(1/2)
            # print(self.SPFLOW_deviation.shape)
            
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
        if mom in m.list_of_moments:
            df2.loc[mom] = [m.weights_dict[mom],
                            np.linalg.norm(getattr(m,mom+'_deviation')),
                            m.description.loc[mom].description]
        else:
            df2.loc[mom] = [0,
                            np.linalg.norm(getattr(m,mom+'_deviation')),
                            m.description.loc[mom].description]
    df2.name = 'targeted moments : '+str(m.list_of_moments)
    worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
    df2.to_excel(writer,sheet_name='Summary',startrow=df1.shape[0] + 5 , startcol=0)
    
    worksheet.write_string(df1.shape[0] + df2.shape[0] + 6, 0, commentary)
    
    
    scalar_moments = pd.DataFrame(columns=['model','target'])
    for mom in m.get_list_of_moments():
        # print(mom)
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
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
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
    m.load_run(path)
    m.compute_moments(sol_c,p)
    # m.compute_Z(sol_c,p)
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

def compare_params(dic, save=False, save_path=None, color_gradient = True):
    n_col = min(1,round(len(dic)/25)+1)
    if color_gradient:
        colors = sns.color_palette("Spectral", n_colors = len(dic))
    else:
        colors = sns.color_palette()
    fig,ax = plt.subplots(figsize = (12,8))
    title = 'Delta of the countries' 
    for i,(com,par) in enumerate(dic.items()):
        ax.plot(par.countries,par.delta[...,1],label=com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.title(title)
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'delta')
    plt.show()    
    
    fig,ax = plt.subplots(figsize = (12,8))
    title = 'One over delta of the countries' 
    for i,(com,par) in enumerate(dic.items()):
        ax.plot(par.countries,1/par.delta[...,1],label=com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.title(title)
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'one_over_delta')
    plt.show()   
    
    fig,ax = plt.subplots(figsize = (12,8))
    title = 'T non patenting sector of the countries' 
    for i,(com,par) in enumerate(dic.items()):
        ax.plot(par.countries,par.T[:,0],label=com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.yscale('log')
    plt.title(title)
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'T_non_patenting')
    plt.show()   
    
    fig,ax = plt.subplots(figsize = (12,8))
    title = 'T patenting sector of the countries' 
    for i,(com,par) in enumerate(dic.items()):
        ax.plot(par.countries,par.T[:,1],label=com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.yscale('log')
    plt.title(title)
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'T_patenting')
    plt.show() 
    
    fig,ax = plt.subplots(figsize = (12,8))
    title = 'Eta of the countries' 
    for i,(com,par) in enumerate(dic.items()):
        ax.plot(par.countries,par.eta[...,1],label=com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.title(title)
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'eta')
    plt.show() 
    
    fig,ax = plt.subplots(figsize = (12,8))
    ax2 = ax.twinx()
    title = 'fo, fe, nu, nu_tilde' 
    for i,(com,par) in enumerate(dic.items()):
        ax.scatter(['fe', 'nu'],
                   [par.fe[1], par.nu[1]],label=com,color=colors[i])
        ax2.scatter(['nu_tilde'], par.nu_tilde[1], label = com,color=colors[i])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",ncol=n_col)
    plt.title(title)
    ax.set_ylabel('fo, fe, nu')
    ax2.set_yscale('log')
    ax2.set_ylabel('nu_tilde')
    if save:
        plt.tight_layout()
        plt.savefig(save_path+'scalar_params')
    plt.show()            

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
p = parameters(n=7,s=2)
# p.calib_parameters = ['eta','delta','fe','tau','T','fo','g_0','nu','nu_tilde','zeta']
p.load_data('calibration_results_matched_economy/74/')
p.zeta = np.array([0.01 , 0.095])
# Z_guess = p.data.expenditure.values/p.unit
# w_guess = p.data.gdp.values*p.unit_labor/(p.data.labor.values*p.unit)*100
# l_R_guess = np.repeat(p.labor[:,None]/200, p.S-1, axis=1).ravel()
# psi_star_guess = np.ones((p.N,p.N,(p.S-1))).ravel()*1000
# phi_guess = np.ones((p.N,p.N,p.S)).ravel()#*0.01
# vec = np.concatenate((w_guess,Z_guess,l_R_guess,psi_star_guess,phi_guess), axis=0)
# guess = np.random.rand(p.guess_from_params().size).reshape(p.guess_from_params().shape)
sol, sol_c = fixed_point_solver(p,x0=p.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='psi_star',
                        plot_convergence=True,
                        plot_cobweb=True,
                        safe_convergence=0.01,
                        disp_summary=True,
                        damping = 10,
                        max_count = 50000,
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
                    'SRDUS', 'SPFLOWDOM', 'SRGDP', 'JUPCOST',
                    'SINNOVPATUS','TO']
m = moments(list_of_moments)
m.load_data()
m.compute_moments(sol_c,p)
# # m.compute_Z(sol_c,p)
m.compute_moments_deviations()
# m.plot_moments(m.list_of_moments)
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
                                  max_count=2e3,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=5
                                  )
    if sol.status == 'failed': 
        print('trying less precise')
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
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying longer convergence')
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
        
    if sol.status == 'failed':
        print('trying with no acceleration')
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=False,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    sol_c.scale_P(p)
    sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    # m.compute_Z(sol_c,p)
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
        if hist.count%100 == 0:
            hist.plot()
        if hist.count%200==0:
            print('fe : ',p.fe[1],'fo : ',p.fo[1], 'delta : ', p.delta[:,1]
                  , 'nu : ', p.nu[1], 'nu_tilde : ', p.nu_tilde[1], 'k :', p.k
                  , 'theta :', p.theta[1], 'sigma :', p.sigma[1], 'zeta :', p.zeta[1]
                  , 'rho :', p.rho, 'kappa :', p.kappa)
        # if hist.count%200==0:
        #     hist.save(path = './calibration_results_matched_trade_flows/history105/', p = p)
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
baseline_number = '104'
if new_run:
    p = parameters(n=7,s=2)
    # p.calib_parameters = ['eta',     'k',     'rho',     'alpha',     'fe',
     # 'T',     'fo',     'sigma',     'theta',     'beta',
     # 'zeta',     'g_0',     'kappa','gamma',
     # 'delta',     'nu',     'nu_tilde']
    # p.calib_parameters = ['T']
    p.load_data('calibration_results_matched_economy/'+baseline_number+'/')
    # p.calib_parameters = ['eta','k','fe','T','zeta','theta','g_0',
    #                       'delta','nu','nu_tilde']
    start_time = time.perf_counter()

# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOWDOM','SPFLOWDOM_US','SPFLOWDOM_RUS', 'SRGDP',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TE','TO']

# list_of_moments = ['TP']
# p.calib_parameters.append('theta')
m = moments()

m.load_data()
m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
if 'theta' in p.calib_parameters:
    p.update_sigma_with_SRDUS_target(m)
# m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOW_US','SPFLOW_RUS',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TO']

# p.calib_parameters.append('fo')
# p.calib_parameters.remove('nu_tilde')
# m.list_of_moments.remove('SRDUS')
m.list_of_moments.append('DOMPATUS')
m.list_of_moments.append('DOMPATEU')
# p.delta[0,1] = 0.05
# m.list_of_moments.append('JUPCOSTRD')
# m.weights_dict['SPFLOW'] = 10
# m.weights_dict['SRDUS'] = 3
# m.weights_dict['RD_US'] = m.weights_dict['RD_US'] + 2
# m.weights_dict['SPFLOWDOM_US'] = m.weights_dict['RD_US'] + 2
# m.TO_target = np.array(0.02)
# m.KM_target = np.array(0.2)
# m.drop_CHN_IND_BRA_ROW_from_RD = True
# m.add_domestic_US_to_SPFLOW = True
# m.add_domestic_EU_to_SPFLOW = True
if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
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
                        # ftol=1e-14, 
                        xtol=1e-11, 
                        # gtol=1e-14,
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
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

#%% writing results as excel

commentary = 'added DOMPATUS and DOMPATEU moment'
# baseline_number = '102'
run_number = 33
path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
try:
    os.mkdir(path)
except:
    pass
write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
m.plot_moments(m.list_of_moments, save_plot = path+str(run_number))

local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')

#%% make calibration runs for different moment(s) target

baseline = '102'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)

moments_to_change = ['KM','JUPCOST','SINNOVPATUS','TO','GROWTH','SRDUS','sales_mark_up_US']
# moments_to_change = ['SRDUS'] 

dic_runs = dict([(mom, np.linspace(getattr(m_baseline,mom+'_target')*0.5,getattr(m_baseline,mom+'_target')*1.5,11))
                 for mom in moments_to_change])
# dic_runs = {
#             'SRDUS':np.arange(0.1,0.71,0.025),
#             'KM':np.arange(0.03,0.09,0.025),
#             'JUPCOST':np.arange(0.0005,0.001001,0.000025),
#             'SINNOVPATUS':np.arange(0.2,0.65,0.01)
#             }

parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_targets_variations/'
parent_dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_targets_variation/'
try:
    os.mkdir(parent_result_path)
except:
    pass
try:
    os.mkdir(parent_dropbox_path)
except:
    pass

for k, v in dic_runs.items():
    print(k)
    print(v)
    moment_to_change = k
    target_list = v
    result_path = parent_result_path+moment_to_change+'/'
    dropbox_path = parent_dropbox_path+moment_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,target in enumerate(target_list):
        print(k)
        print(v)
        print(target)
        m = moments()
        m.load_data()
        m.load_run(baseline_path)
        p = parameters(n=7,s=2)
        p.load_data(baseline_path)
        setattr(m,moment_to_change+'_target',target)
        # m.sales_mark_up_
        if 'theta' in p_baseline.calib_parameters:
            p.update_sigma_with_SRDUS_target(m)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        test_ls = optimize.least_squares(fun = calibration_func,    
                            x0 = p.make_p_vector(), 
                            args = (p,m,p.guess,hist,start_time), 
                            bounds = bounds,
                            max_nfev=1e8,
                            # ftol=1e-14, 
                            xtol=1e-10, 
                            # gtol=1e-14,
                            verbose = 2)
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
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        m.plot_moments(m.list_of_moments, save_plot = dropbox_path+str(i))

#%% write summaries calibration runs for moment target change

def GetSpacedElements(array, numElems = 13):
    idx = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    out = array[idx]
    return out, idx

baseline = '102'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)
parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_targets_variations/'
parent_dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_targets_variation/'
with_dropped = False
save = True

# moment_to_change = 'SRDUS'
for moment_to_change in ['SRDUS','KM','JUPCOST','SINNOVPATUS','TO','GROWTH']:
# for moment_to_change in ['SRDUS']:
    print(moment_to_change)
    result_path = parent_result_path+moment_to_change+'/'
    dropbox_path = parent_dropbox_path+moment_to_change+'/'
    dropbox_summary_path = dropbox_path+'summary/'
    if with_dropped:
        dropbox_summary_path = dropbox_path+'summary_with_dropped_moment/'
    
    try:
        os.mkdir(result_path+'summary/')
    except:
        pass
    try:
        os.mkdir(dropbox_summary_path)
    except:
        pass
    
    if moment_to_change == 'sales_mark_up_US':
        m_baseline.get_sales_mark_up_US_from_sigma(p_baseline)
    
    baseline_moment = getattr(m_baseline, moment_to_change+'_target')
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            print(p.sigma)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p)    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            run_name = run+': '+moment_to_change+str(getattr(m,moment_to_change+'_target'))
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
        
        elif run == '99' and with_dropped:
            run_list.remove('99')
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            # print(p.sigma)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p)    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            dropped_m = m
            dropped_p = p
            dropped_sol = sol_c
            
        
    targets = np.array([getattr(m,moment_to_change+'_target') for m in dic_m.values()]).squeeze()
    
    # moments.compare_moments(dic_m, contin_cmap=True)
    # compare_params(dic_p)
    
    # moments.compare_moments(dic_m, contin_cmap=True, save_path = dropbox_summary_path)
    # compare_params(dic_p, save = True, save_path = dropbox_summary_path)
    
    # fig,ax = plt.subplots(figsize=(12,8))
    # ax2 = ax.twinx()
    
    # nus = [p.nu[1] for p in dic_p.values()]
    # deltas_US = [p.delta[0,1] for p in dic_p.values()]
    # deltas_mean = [p.delta[:,1].mean() for p in dic_p.values()]
    
    # ax.plot(targets,nus,color=sns.color_palette()[0],label='nu')
    # ax2.plot(targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    # ax2.plot(targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    # ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    # ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    # ax.set_ylabel('Nu',fontsize = 15)
    # ax2.set_ylabel('Delta',fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    
    up_ten_percent_moment_index = np.abs(baseline_moment*1.1-targets).argmin()
    down_ten_percent_moment_index = np.abs(baseline_moment*0.9-targets).argmin()
    
    up_ten_percent_run = runs[up_ten_percent_moment_index]
    down_ten_percent_run = runs[down_ten_percent_moment_index]
    
    down_change_str = '% change for a '+str(((targets[down_ten_percent_moment_index]-baseline_moment)*100/baseline_moment).round(2))+'% change in moment'
    up_change_str = '% change for a '+str(((targets[up_ten_percent_moment_index]-baseline_moment)*100/baseline_moment).round(2))+'% change in moment'
    
    table_10 = pd.DataFrame(columns = [down_change_str,up_change_str,'baseline_value'])
    
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['delta '+c] = [(dic_p[down_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    (dic_p[up_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    p_baseline.delta[i,1]
                                    ]
        
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['eta '+c] = [(dic_p[down_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    (dic_p[up_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    p_baseline.eta[i,1]
                                    ]
    
    table_10.loc['nu'] = [(dic_p[down_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                (dic_p[up_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                p_baseline.nu[1]
                                ]
    table_10.loc['fe'] = [(dic_p[down_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                (dic_p[up_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                p_baseline.fe[1]
                                ]
    table_10.loc['k'] = [(dic_p[down_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                (dic_p[up_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                p_baseline.k
                                ]
    table_10.loc['zeta'] = [(dic_p[down_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                (dic_p[up_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                p_baseline.zeta
                                ]
    table_10.loc['nu_tilde'] = [(dic_p[down_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                (dic_p[up_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                p_baseline.nu_tilde[1]
                                ]
    table_10.loc['theta'] = [(dic_p[down_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                (dic_p[up_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                p_baseline.theta[1]
                                ]
    table_10.loc['sigma'] = [(dic_p[down_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                (dic_p[up_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                p_baseline.sigma[1]
                                ]
    if save:
        table_10.to_csv(dropbox_summary_path +'plus_minus_ten_percent_change_table.csv')
    
    less_targets, less_idx = GetSpacedElements(targets,len(targets))
    less_runs = [runs[i] for i in less_idx]
    less_dic_m = {}
    less_dic_p = {}
    less_dic_sol = {}
    for r in less_runs:
        less_dic_m[r] = dic_m[r]
        less_dic_p[r] = dic_p[r]
        less_dic_sol[r] = dic_sol[r]
        
    temp_m = less_dic_m.copy()    
    try:
        temp_m['99: '+moment_to_change+'dropped'] = dropped_m
    except:
        pass
    temp_p = less_dic_p.copy()   
    try:
        temp_p['99: '+moment_to_change+'dropped'] = dropped_p
    except:
        pass
        
    if save:    
        moments.compare_moments(temp_m, contin_cmap=True, save_path = dropbox_summary_path)
        compare_params(temp_p, save = True, save_path = dropbox_summary_path)
    else:
        moments.compare_moments(temp_m, contin_cmap=True)
        compare_params(temp_p, save = False)    
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    
    nus = [p.nu[1] for p in less_dic_p.values()]
    nus_tilde = [p.nu_tilde[1] for p in less_dic_p.values()]
    thetas = [p.theta[1] for p in less_dic_p.values()]
    deltas_US = [p.delta[0,1] for p in less_dic_p.values()]
    deltas_EU = [p.delta[1,1] for p in less_dic_p.values()]
    deltas_JAP = [p.delta[2,1] for p in less_dic_p.values()]
    deltas_mean = [p.delta[:,1].mean() for p in less_dic_p.values()]
    
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax2.plot(less_targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    ax2.plot(less_targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    try:
        ax2.plot([],[],lw=0,label=f"\n Moment dropped :\n nu :{dropped_p.nu[1]:.3}\n delta_US:{dropped_p.delta[0,1]:.3}\n delta_average:{dropped_p.delta[:,1].mean():.3}")
    except:
        pass
    
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Nu',fontsize = 15)
    ax2.set_ylabel('Delta',fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    
    ax.plot(less_targets,deltas_US,label='US')
    ax.plot(less_targets,deltas_EU,label='EUR')
    ax.plot(less_targets,deltas_JAP,label='JAP')
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Delta',fontsize = 15)
    try:
        ax.plot([],[],lw=0,label=f"\n Moment dropped :\n delta_US:{dropped_p.delta[0,1]:.3}\n delta_EUR:{dropped_p.delta[1,1]:.3}\n delta_JAP:{dropped_p.delta[2,1]:.3}")
    except:
        pass
    ax.legend()
    if save:
        plt.savefig(dropbox_summary_path +'delta_US_EUR_JAP.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax.plot(less_targets,nus_tilde,color=sns.color_palette()[1],label='nu_tilde')
    ax2.plot(less_targets,thetas,color=sns.color_palette()[2],label='theta')    
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Nu and nu_tilde',fontsize = 15)
    ax2.set_ylabel('Theta',fontsize = 15)
    # if not with_dropped:
    try:
        ax2.plot([],[],lw=0,label=f"\n Moment dropped :\n nu :{dropped_p.nu[1]:.3}\n nu_tilde :{dropped_p.nu_tilde[1]:.3}\n theta :{dropped_p.theta[1]:.3}")
    except:
        pass
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_nu_tilde_theta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ratio_delta_US_to_diffusion = np.array(deltas_US)/np.array(nus)
    ax.plot(less_targets,ratio_delta_US_to_diffusion)
    ax.set_ylabel('Delta_US / Nu',fontsize = 15)
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    # if with_dropped:
    try:
        ax.plot([],[],lw=0,label=f" Moment dropped :\n delta_US/nu :{dropped_p.delta[0,1]/dropped_p.nu[1]:.3}")
    except:
        pass
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'patent_protection_US_to_diffusion_ratio.png')
    plt.show()

#%% make calibration runs for different parameters

baseline = '102'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)

parameters_to_change = ['kappa','gamma','rho','zeta']
# parameters_to_change = ['gamma','rho','zeta']

dic_runs = dict([(par, np.linspace(getattr(p_baseline,par)*0.5,getattr(p_baseline,par)*1.5,11))
                 for par in parameters_to_change])
# if 'kappa' in parameters_to_change:
#     dic_runs['kappa'] = np.linspace(getattr(p_baseline,'kappa')*0.8,getattr(p_baseline,'kappa')*1.2,21)
if 'zeta' in parameters_to_change:
    dic_runs['zeta'] = [np.array([p_baseline.zeta[0], i]) for i in np.linspace(0,0.1,21)]
    p_baseline.calib_parameters.remove('zeta')
    

parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_parameters_variations/'
parent_dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_parameters_variation/'
try:
    os.mkdir(parent_result_path)
except:
    pass
try:
    os.mkdir(parent_dropbox_path)
except:
    pass

for k, v in dic_runs.items():
    print(k)
    print(v)
    par_to_change = k
    par_list = v
    result_path = parent_result_path+par_to_change+'/'
    dropbox_path = parent_dropbox_path+par_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,par in enumerate(par_list):
        print(k)
        print(v)
        print(par)
        m = moments()
        m.load_data()
        m.load_run(baseline_path)
        p = parameters(n=7,s=2)
        p.load_data(baseline_path)
        if par_to_change == 'zeta':
            p.calib_parameters.remove('zeta')
        setattr(p,par_to_change,par)
        if 'theta' in p_baseline.calib_parameters:
            p.update_sigma_with_SRDUS_target(m)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        test_ls = optimize.least_squares(fun = calibration_func,    
                            x0 = p.make_p_vector(), 
                            args = (p,m,p.guess,hist,start_time), 
                            bounds = bounds,
                            max_nfev=1e8,
                            # ftol=1e-14, 
                            xtol=1e-11, 
                            # gtol=1e-14,
                            verbose = 2)
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
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        m.plot_moments(m.list_of_moments, save_plot = dropbox_path+str(i))

#%% write summaries calibration runs for parameter change

def GetSpacedElements(array, numElems = 13):
    idx = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    out = array[idx]
    return out, idx

baseline = '102'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)
parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_parameters_variations/'
parent_dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_parameters_variation/'
save = True

# parameter_to_change = 'SRDUS'
for parameter_to_change in ['gamma','kappa','rho','zeta']:
# for parameter_to_change in ['zeta']:
    print(parameter_to_change)
    result_path = parent_result_path+parameter_to_change+'/'
    dropbox_path = parent_dropbox_path+parameter_to_change+'/'
    dropbox_summary_path = dropbox_path+'summary/'
    
    try:
        os.mkdir(result_path+'summary/')
    except:
        pass
    try:
        os.mkdir(dropbox_summary_path)
    except:
        pass
    
    try:
        baseline_param = getattr(p_baseline, parameter_to_change)[1]
    except:
        baseline_param = getattr(p_baseline, parameter_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        print(run)
        run_path = result_path+run+'/'
        p = parameters(n=7,s=2)
        p.load_data(run_path)
        m = moments()
        m.load_data()
        m.load_run(run_path)
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 5e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        sol_c = var.var_from_vector(sol.x, p)    
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p) 
        m.compute_moments(sol_c,p)
        m.compute_moments_deviations()
        
        try:
            run_name = run+': '+parameter_to_change+str(getattr(p,parameter_to_change)[1])
        except:
            run_name = run+': '+parameter_to_change+str(getattr(p,parameter_to_change))
        runs.append(run_name)
        dic_m[run_name] = m
        dic_p[run_name] = p
        dic_sol[run_name] = sol_c
    
    try:
        targets = np.array([getattr(p,parameter_to_change)[1] for p in dic_p.values()]).squeeze()
    except:
        targets = np.array([getattr(p,parameter_to_change) for p in dic_p.values()]).squeeze()
    
    # moments.compare_moments(dic_m, contin_cmap=True)
    # compare_params(dic_p)
    
    # moments.compare_moments(dic_m, contin_cmap=True, save_path = dropbox_summary_path)
    # compare_params(dic_p, save = True, save_path = dropbox_summary_path)
    
    # fig,ax = plt.subplots(figsize=(12,8))
    # ax2 = ax.twinx()
    
    # nus = [p.nu[1] for p in dic_p.values()]
    # deltas_US = [p.delta[0,1] for p in dic_p.values()]
    # deltas_mean = [p.delta[:,1].mean() for p in dic_p.values()]
    
    # ax.plot(targets,nus,color=sns.color_palette()[0],label='nu')
    # ax2.plot(targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    # ax2.plot(targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    # ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    # ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_xlabel(parameter_to_change+'_target',fontsize = 15)
    # ax.set_ylabel('Nu',fontsize = 15)
    # ax2.set_ylabel('Delta',fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    
    up_ten_percent_param_index = np.abs(baseline_param*1.1-targets).argmin()
    down_ten_percent_param_index = np.abs(baseline_param*0.9-targets).argmin()
    
    up_ten_percent_run = runs[up_ten_percent_param_index]
    down_ten_percent_run = runs[down_ten_percent_param_index]
    
    down_change_str = '% change for a '+str(((targets[down_ten_percent_param_index]-baseline_param)*100/baseline_param).round(2))+'% change in parameter'
    up_change_str = '% change for a '+str(((targets[up_ten_percent_param_index]-baseline_param)*100/baseline_param).round(2))+'% change in parameter'
    
    table_10 = pd.DataFrame(columns = [down_change_str,up_change_str,'baseline_value'])
    
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['delta '+c] = [(dic_p[down_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    (dic_p[up_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    p_baseline.delta[i,1]
                                    ]
        
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['eta '+c] = [(dic_p[down_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    (dic_p[up_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    p_baseline.eta[i,1]
                                    ]
    
    table_10.loc['nu'] = [(dic_p[down_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                (dic_p[up_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                p_baseline.nu[1]
                                ]
    table_10.loc['fe'] = [(dic_p[down_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                (dic_p[up_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                p_baseline.fe[1]
                                ]
    table_10.loc['k'] = [(dic_p[down_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                (dic_p[up_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                p_baseline.k
                                ]
    table_10.loc['zeta'] = [(dic_p[down_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                (dic_p[up_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                p_baseline.zeta
                                ]
    table_10.loc['nu_tilde'] = [(dic_p[down_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                (dic_p[up_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                p_baseline.nu_tilde[1]
                                ]
    table_10.loc['theta'] = [(dic_p[down_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                (dic_p[up_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                p_baseline.theta[1]
                                ]
    table_10.loc['sigma'] = [(dic_p[down_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                (dic_p[up_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                p_baseline.sigma[1]
                                ]
    if save:
        table_10.to_csv(dropbox_summary_path +'plus_minus_ten_percent_change_table.csv')
    
    less_targets, less_idx = GetSpacedElements(targets,len(targets))
    less_runs = [runs[i] for i in less_idx]
    less_dic_m = {}
    less_dic_p = {}
    less_dic_sol = {}
    for r in less_runs:
        less_dic_m[r] = dic_m[r]
        less_dic_p[r] = dic_p[r]
        less_dic_sol[r] = dic_sol[r]
        
    if save:    
        moments.compare_moments(less_dic_m, contin_cmap=True, save_path = dropbox_summary_path)
        compare_params(less_dic_p, save = True, save_path = dropbox_summary_path)
    else:
        moments.compare_moments(less_dic_m, contin_cmap=True)
        compare_params(less_dic_p, save = False)
        
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    
    nus = [p.nu[1] for p in less_dic_p.values()]
    nus_tilde = [p.nu_tilde[1] for p in less_dic_p.values()]
    thetas = [p.theta[1] for p in less_dic_p.values()]
    deltas_US = [p.delta[0,1] for p in less_dic_p.values()]
    deltas_EU = [p.delta[1,1] for p in less_dic_p.values()]
    deltas_JAP = [p.delta[2,1] for p in less_dic_p.values()]
    deltas_mean = [p.delta[:,1].mean() for p in less_dic_p.values()]
    
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax2.plot(less_targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    ax2.plot(less_targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Nu',fontsize = 15)
    ax2.set_ylabel('Delta',fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    
    ax.plot(less_targets,deltas_US,label='US')
    ax.plot(less_targets,deltas_EU,label='EUR')
    ax.plot(less_targets,deltas_JAP,label='JAP')
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Delta',fontsize = 15)
    ax.legend()
    if save:
        plt.savefig(dropbox_summary_path +'delta_US_EUR_JAP.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax.plot(less_targets,nus_tilde,color=sns.color_palette()[1],label='nu_tilde')
    ax2.plot(less_targets,thetas,color=sns.color_palette()[2],label='theta')    
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Nu and nu_tilde',fontsize = 15)
    ax2.set_ylabel('Theta',fontsize = 15)
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_nu_tilde_theta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ratio_delta_US_to_diffusion = np.array(deltas_US)/np.array(nus)
    ax.plot(less_targets,ratio_delta_US_to_diffusion)
    ax.set_ylabel('Delta_US / Nu',fontsize = 15)
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'patent_protection_US_to_diffusion_ratio.png')
    plt.show()

#%% Gather all parameters and moments variations

baseline = '102'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)

if baseline == '101':
    moments_to_change = ['SRDUS','KM','JUPCOST','SINNOVPATUS','GROWTH','TO']
    parameters_to_change = ['rho','zeta','kappa','gamma']
if baseline == '102':
    moments_to_change = ['SRDUS','KM','JUPCOST','SINNOVPATUS','GROWTH','TO']
    parameters_to_change = ['rho','zeta','kappa','gamma']
if baseline == '75':
    moments_to_change = ['SRDUS','KM','JUPCOST','SINNOVPATUS','sales_mark_up_US','GROWTH','TO']
    parameters_to_change = ['rho','zeta','kappa','gamma']
if baseline == '74':
    moments_to_change = ['SRDUS','KM','JUPCOST','SINNOVPATUS','sales_mark_up_US']
    parameters_to_change = ['rho','zeta','kappa','gamma']
if baseline == '73':
    moments_to_change = ['SRDUS','KM','JUPCOST','SINNOVPATUS']
    parameters_to_change = ['rho','zeta','kappa','gamma']

parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_targets_variations/'

dic_of_variation_dics = {}

for moment_to_change in moments_to_change:
    variation_dic = {}
    print(moment_to_change)
    result_path = parent_result_path+moment_to_change+'/'
    
    baseline_moment = getattr(m_baseline, moment_to_change+'_target')
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_values = {}
    dic_change = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p)    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            run_name = run+': '+moment_to_change+str(getattr(m,moment_to_change+'_target'))
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(getattr(m,moment_to_change+'_target'))
            dic_change[run_name] = float((getattr(m,moment_to_change+'_target')-baseline_moment)*100/baseline_moment)
        
    variation_dic['changing_quantity'] = moment_to_change+'_target'
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_moment
    
    dic_of_variation_dics[moment_to_change+'_target'] = variation_dic
    
parent_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_parameters_variations/'
for parameter_to_change in parameters_to_change:
    variation_dic = {}
    print(parameter_to_change)
    result_path = parent_result_path+parameter_to_change+'/'
    
    try:
        baseline_param = getattr(p_baseline, parameter_to_change)[1]
    except:
        baseline_param = getattr(p_baseline, parameter_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_change = {}
    dic_values = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p)    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            
            try:
                current_param = getattr(p,parameter_to_change)[1]
            except:
                current_param = getattr(p,parameter_to_change)
            
            run_name = run+': '+parameter_to_change+str(current_param)
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(current_param)
            dic_change[run_name] = float((current_param-baseline_param)*100/baseline_param)
        
    variation_dic['changing_quantity'] = parameter_to_change
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_param
    
    dic_of_variation_dics[parameter_to_change] = variation_dic
    
#%% build a big table and write as excel

df = pd.DataFrame(columns = ['quantity','value','change to baseline']).set_index(['quantity','value','change to baseline'])

for variation_dic in dic_of_variation_dics.values():
    print(variation_dic['changing_quantity'])
    for run in variation_dic['run_names']:
        print(variation_dic['values'][run])
        # value = f'{float(f"{variation_dic["values"][run]:.1g}"):g}'
        value = '{:g}'.format(float('{:.3g}'.format(variation_dic["values"][run])))
        change = '{:g}'.format(float('{:.2g}'.format(variation_dic["change"][run])))
        for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
            if s_spec_par in variation_dic['p'][run].calib_parameters:
                df.loc[(variation_dic['changing_quantity'],value,change),s_spec_par] = getattr(variation_dic['p'][run],s_spec_par)[1]
        for scal_par in ['g_0','k',]:
            df.loc[(variation_dic['changing_quantity'],value,change),scal_par] = getattr(variation_dic['p'][run],scal_par)
        for c_spec_par in ['delta','eta','T']:
            for i,c in enumerate(variation_dic['p'][run].countries):
                df.loc[(variation_dic['changing_quantity'],value,change),c_spec_par+'_'+c] = getattr(variation_dic['p'][run],c_spec_par)[i,1]

dropbox_variation_table_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'+baseline+'_all_variations_table'

writer = pd.ExcelWriter(dropbox_variation_table_path+'.xlsx', engine='xlsxwriter')
workbook = writer.book
# worksheet = workbook.add_worksheet('all_quantities')
df.to_excel(writer,sheet_name='all_quantities',startrow=0 , startcol=0)   

writer.save()

#%% build a sensivity table

for percent_change in [-10,10]:

    df = pd.DataFrame()
    
    # for variation_dic in list_of_variation_dics:
    #     print(variation_dic['changing_quantity'])
    #     ten_percent_idx = np.abs([i-percent_change for i in variation_dic['change'].values()]).argmin()
    #     zero_percent_idx = np.abs([i for i in variation_dic['change'].values()]).argmin()
    #     run = variation_dic['run_names'][ten_percent_idx]
    #     run_baseline = variation_dic['run_names'][zero_percent_idx]
    #     for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
    #         if s_spec_par in variation_dic['p'][run].calib_parameters:
    #             df.loc[variation_dic['changing_quantity'],s_spec_par] = \
    #                 (getattr(variation_dic['p'][run],s_spec_par)[1] - getattr(variation_dic['p'][run_baseline],s_spec_par)[1])*100/getattr(variation_dic['p'][run_baseline],s_spec_par)[1]
    #     for scal_par in ['g_0','k',]:
    #         df.loc[variation_dic['changing_quantity'],scal_par] = \
    #             (getattr(variation_dic['p'][run],scal_par) - getattr(variation_dic['p'][run_baseline],scal_par))*100/getattr(variation_dic['p'][run_baseline],scal_par)
    #     for c_spec_par in ['delta','eta','T']:
    #         for i,c in enumerate(variation_dic['p'][run].countries):
    #             df.loc[variation_dic['changing_quantity'],c_spec_par+'_'+c] = \
    #                 (getattr(variation_dic['p'][run],c_spec_par)[i,1] - getattr(variation_dic['p'][run_baseline],c_spec_par)[i,1])*100/getattr(variation_dic['p'][run_baseline],c_spec_par)[i,1]
    
    for variation_dic in dic_of_variation_dics.values():
        print(variation_dic['changing_quantity'])
        ten_percent_idx = np.abs([i-percent_change for i in variation_dic['change'].values()]).argmin()
        zero_percent_idx = np.abs([i for i in variation_dic['change'].values()]).argmin()
        run = variation_dic['run_names'][ten_percent_idx]
        run_baseline = variation_dic['run_names'][zero_percent_idx]
        for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
            if s_spec_par in variation_dic['p'][run].calib_parameters:
                df.loc[variation_dic['changing_quantity'],s_spec_par] = \
                    (getattr(variation_dic['p'][run],s_spec_par)[1] - getattr(p_baseline,s_spec_par)[1])*100/getattr(p_baseline,s_spec_par)[1]
        for scal_par in ['g_0','k',]:
            df.loc[variation_dic['changing_quantity'],scal_par] = \
                (getattr(variation_dic['p'][run],scal_par) - getattr(p_baseline,scal_par))*100/getattr(p_baseline,scal_par)
        for c_spec_par in ['delta','eta','T']:
            for i,c in enumerate(variation_dic['p'][run].countries):
                df.loc[variation_dic['changing_quantity'],c_spec_par+'_'+c] = \
                    (getattr(variation_dic['p'][run],c_spec_par)[i,1] - getattr(p_baseline,c_spec_par)[i,1])*100/getattr(p_baseline,c_spec_par)[i,1]
                 
    df = df.T
    df = df.round(2)
    
    dropbox_sensitivity_table = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_sensitivities/'+baseline+'_senstivity_'+str(percent_change)+'percent_change'
    
    writer = pd.ExcelWriter(dropbox_sensitivity_table+'.xlsx', engine='xlsxwriter')
    workbook = writer.book
    # worksheet = workbook.add_worksheet('all_quantities')
    df.to_excel(writer,sheet_name='all_quantities',startrow=0 , startcol=0)   
    
    writer.save()

#%% sensitivity graphs

sensitivity_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline+'_sensitivities/'

try:
    os.mkdir(sensitivity_path)
except:
    pass

undisplayed_list = ['sales_mark_up_US_target','zeta']

fig, ax = plt.subplots(figsize = (12,8))

for variation_dic in dic_of_variation_dics.values():
    if variation_dic['changing_quantity'] not in undisplayed_list:
        ax.plot([change for change in variation_dic['change'].values()],[p.delta[0,1]/p.nu[1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
        
ax.set_xlabel('Change in respective moment or parameter')    
ax.set_ylabel('Delta_US / nu',fontsize = 20)    
plt.title('Delta_US / nu',fontsize = 20) 
ax.legend()
plt.savefig(sensitivity_path+baseline+'_patent_protection_to_diffusion_ratio')
plt.show()    

for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
    if s_spec_par in p_baseline.calib_parameters:
        fig, ax = plt.subplots(figsize = (12,8))
        
        for variation_dic in dic_of_variation_dics.values():
            if variation_dic['changing_quantity'] not in undisplayed_list:
                ax.plot([change for change in variation_dic['change'].values()],[getattr(p,s_spec_par)[1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                
        ax.set_xlabel('Change in respective moment or parameter')    
        ax.set_ylabel(s_spec_par,fontsize = 20)
        plt.title(s_spec_par,fontsize = 20)
            
        ax.legend()
        plt.savefig(sensitivity_path+baseline+'_'+s_spec_par)
        plt.show()   
        
for scal_par in ['g_0','k',]:
    if scal_par in p_baseline.calib_parameters:
        fig, ax = plt.subplots(figsize = (12,8))
        
        for variation_dic in dic_of_variation_dics.values():
            if variation_dic['changing_quantity'] not in undisplayed_list:
                ax.plot([change for change in variation_dic['change'].values()],[getattr(p,scal_par) for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                
        ax.set_xlabel('Change in respective moment or parameter')    
        ax.set_ylabel(scal_par,fontsize = 20)
        plt.title(scal_par,fontsize = 20)
            
        ax.legend()
        plt.savefig(sensitivity_path+baseline+'_'+scal_par)
        plt.show()    
        
for c_spec_par in ['delta']:
    for i,c in enumerate(['US']):
        if scal_par in p_baseline.calib_parameters:
            fig, ax = plt.subplots(figsize = (12,8))
            
            for variation_dic in dic_of_variation_dics.values():
                if variation_dic['changing_quantity'] not in undisplayed_list:
                    ax.plot([change for change in variation_dic['change'].values()],[getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                    
            ax.set_xlabel('Change in respective moment or parameter')    
            ax.set_ylabel(c_spec_par+'_'+c,fontsize = 20)
            plt.title(c_spec_par+'_'+c,fontsize = 20)
                
            ax.legend()
            plt.savefig(sensitivity_path+baseline+'_'+c_spec_par+'_'+c)
            plt.show()



#%% writing results as excel

# commentary = 'Dropped CHN IND BRA ROW from RD moment'
# baseline_number = '101'
# run_number = 1
# path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
# try:
#     os.mkdir(path)
# except:
#     pass
# write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
# m.plot_moments(m.list_of_moments, save_plot = path+str(run_number))

# local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
# try:
#     os.mkdir(local_path)
# except:
#     pass
# p_sol.write_params(local_path+str(run_number)+'/')
# m.write_moments(local_path+str(run_number)+'/')

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
p40, sol40, m40 = full_load_parameters_set('calibration_results_matched_economy/40/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m40.TO_target = np.array(0.1)
m40.compute_moments(sol40, p40)
m40.compute_moments_deviations()
p41, sol41, m41 = full_load_parameters_set('calibration_results_matched_economy/41/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m41.TO_target = np.array(0.2)
m41.compute_moments(sol41, p41)
m41.compute_moments_deviations()
p42, sol42, m42 = full_load_parameters_set('calibration_results_matched_economy/42/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m42.TO_target = np.array(0.3)
m42.compute_moments(sol42, p42)
m42.compute_moments_deviations()
p43, sol43, m43 = full_load_parameters_set('calibration_results_matched_economy/43/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m43.TO_target = np.array(0.05)
m43.compute_moments(sol43, p43)
m43.compute_moments_deviations()
p44, sol44, m44 = full_load_parameters_set('calibration_results_matched_economy/44/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m44.TO_target = np.array(0.075)
m44.compute_moments(sol44, p44)
m44.compute_moments_deviations()
p45, sol45, m45 = full_load_parameters_set('calibration_results_matched_economy/45/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m45.TO_target = np.array(0.025)
m45.compute_moments(sol45, p45)
m45.compute_moments_deviations()
p46, sol46, m46 = full_load_parameters_set('calibration_results_matched_economy/46/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m46.TO_target = np.array(0.04)
m46.compute_moments(sol46, p46)
m46.compute_moments_deviations()
p47, sol47, m47 = full_load_parameters_set('calibration_results_matched_economy/47/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m47.TO_target = np.array(0.06)
m47.compute_moments(sol47, p47)
m47.compute_moments_deviations()
p48, sol48, m48 = full_load_parameters_set('calibration_results_matched_economy/48/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m48.TO_target = np.array(0.01)
m48.compute_moments(sol48, p48)
m48.compute_moments_deviations()
p50, sol50, m50 = full_load_parameters_set('calibration_results_matched_economy/48/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS'])
p51, sol51, m51 = full_load_parameters_set('calibration_results_matched_economy/51/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m51.TO_target = np.array(0.05)
m51.compute_moments(sol51, p51)
m51.compute_moments_deviations()
p52, sol52, m52 = full_load_parameters_set('calibration_results_matched_economy/52/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m52.TO_target = np.array(0.1)
m52.compute_moments(sol52, p52)
m52.compute_moments_deviations()
p53, sol53, m53 = full_load_parameters_set('calibration_results_matched_economy/53/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m53.TO_target = np.array(0.2)
m53.compute_moments(sol53, p53)
m53.compute_moments_deviations()
p54, sol54, m54 = full_load_parameters_set('calibration_results_matched_economy/54/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m54.TO_target = np.array(0.075)
m54.compute_moments(sol54, p54)
m54.compute_moments_deviations()
p55, sol55, m55 = full_load_parameters_set('calibration_results_matched_economy/55/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m55.TO_target = np.array(0.025)
m55.compute_moments(sol55, p55)
m55.compute_moments_deviations()
p56, sol56, m56 = full_load_parameters_set('calibration_results_matched_economy/56/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO'])
m56.TO_target = np.array(0.01)
m56.compute_moments(sol56, p56)
m56.compute_moments_deviations()
# for m in [m4,m6,m7,m10,m11,m12,m13]:
#     m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
#                         'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
#                         'SINNOVPATEU']
# for m in [m14,m15]:
#     m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
#                         'SRDUS', 'SPFLOW', 'SRGDP', 'JUPCOST',
#                         'SINNOVPATEU','NUR']
p65, sol65, m65 = full_load_parameters_set('calibration_results_matched_economy/65/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOWDOM', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO','TE'])
m65.TO_target = np.array(0.05)
m65.compute_moments(sol65, p65)
m65.compute_moments_deviations()

p66, sol66, m66 = full_load_parameters_set('calibration_results_matched_economy/66/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO','TE'])
m66.TO_target = np.array(0.05)
m66.compute_moments(sol66, p66)
m66.compute_moments_deviations()
p67, sol67, m67 = full_load_parameters_set('calibration_results_matched_economy/67/',
                                           ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD', 'RP',
                                            'SRDUS','SPFLOW', 'SRGDP', 'JUPCOST',
                                            'SINNOVPATUS','TO','TE'])
m67.TO_target = np.array(0.05)
m67.compute_moments(sol67, p67)
m67.compute_moments_deviations()


#%% build dic compare moments

# dic = {
#         '40 : Turnover US, weight 3, target 0.1':m40,
#         '41 : Turnover US, weight 3, target 0.2':m41,
#         '42 : Turnover US, weight 3, target 0.3':m42,
#         '43 : Turnover US, weight 3, target 0.05':m43,
#         '44 : Turnover US, weight 3, target 0.075':m44,
#         '45 : Turnover US, weight 3, target 0.025':m45,
#         '46 : Turnover US, weight 3, target 0.04':m46,
#         '47 : Turnover US, weight 3, target 0.06':m47,
#         '48 : Turnover US, weight 3, target 0.01':m48,
#         '50 : No turnover target, benchmark':m50
#         }

# dic = {
#         '51 : Turnover US, fixed nu, target 0.05':m51,
#         '52 : Turnover US, fixed nu, target 0.1':m52,
#         '53 : Turnover US, fixed nu, target 0.2':m53,
#         '54 : Turnover US, fixed nu, target 0.075':m54,
#         '55 : Turnover US, fixed nu, target 0.025':m55,
#         '56 : Turnover US, fixed nu, target 0.01':m56
#         }

# dic = {
#         '50 : No turnover target, no TO moment, no TE moment':m50,
#         '43 : Turnover US targeted 0.05, no TE moment, theta fixed':m43,
#         '65 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows targeted, theta calibrated = '+str(p65.theta[1].round(3)):m65,
#         '66 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows not targeted, theta calibrated = '+str(p66.theta[1].round(3)):m66,
#         '67 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows not targeted, theta calibrated = '+str(p67.theta[1].round(3))+'kappa rho calibrated':m67
#         }

# list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD','RD_US',
# 'RD_RUS', 'RP', 'SPFLOWDOM','SPFLOWDOM_US','SPFLOWDOM_RUS','SRDUS',
# 'SRGDP_US','SRGDP_RUS', 'SRGDP', 'JUPCOST','SINNOVPATUS']
# comments = ['1: SPFLOW',
#             '2: SPFLOW weight 10',
#             '3: SPFLOW ratio loss function',
#             '4: SPFLOWDOM ratio loss function',
#             '5: DOMPATRATUSEU',
#             '6: DOMPATRATUSEU ratio loss function',
#             '7: SPATDEST and SPATORIG',
#             '8: Trade weighted SPFLOW',
#             '9: Trade weighted SPFLOWDOM',
#             '11: SPFLOW, no SRDUS',
#             '12: SPFLOW weight 10, no SRDUS',
#             '13: SPFLOW ratio loss function, no SRDUS',
#             '14: SPFLOWDOM ratio loss function, no SRDUS',
#             '15: DOMPATRATUSEU, no SRDUS',
#             '16: DOMPATRATUSEU ratio loss function, no SRDUS',
#             '17: SPATDEST and SPATORIG, no SRDUS',
#             '18: Trade weighted SPFLOW, no SRDUS',
#             '19: Trade weighted SPFLOWDOM, no SRDUS',
#             ]
# comments = ['43 : nu_tilde free, theta fixed',
#             '73 : nu_tilde = 10, theta fixed',
#             '74 : nu_tilde free, theta calibrated',
#             '75 : nu_tilde = nu, theta calibrated',
#             '76 : nu_tilde = nu, theta fixed',
#             '77 : nu_tilde = 10, theta calibrated']
# comments = [
#             '73 : Share of the economy',
#             '83 : Ratio to the US',
#             '84 : Ratio to the US, weight +2',
#             '85 : Ratio to the US, weight +3',
#             ]
# comments = [
#             '73 : with RD moment',
#             '78 : without RD moment',
#             '75 : calibrated theta with RD moment',
#             '79 : calibrated theta without RD moment',
#             '74 : calibrated theta and nu_tilde with RD moment',
#             '80 : calibrated theta and nu_tilde without RD moment'
#             ]
list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
 'RD', 'RD_US', 'RD_RUS', 'RP', 'SPFLOWDOM', 'SPFLOW',
 'SPFLOW_US', 'SPFLOW_RUS', 'SRDUS', 'SRGDP', 'SRGDP_US',
 'SRGDP_RUS', 'JUPCOST','JUPCOSTRD', 'SINNOVPATUS', 'TO',
 'DOMPATUS','DOMPATEU']
comments_dic = {'1':'1 : drop South in RD targeting',
                '21':'21 : added domestic US to patent flow moment',
                '22':'22 : added domestic EU to patent flow moment',
                '23':'23 : added domestic US and EU to patent flow moment',
                '31':'31 : added DOMPATUS',
                '32':'32 : added DOMPATEU',
                '33':'33 : added DOMPATUS and DOMPATUS',
                '41':'41 : 21 and drop South in RD',
                '42':'42 : 22 and drop South in RD',
                '43':'43 : 23 and drop South in RD',
                '5':'5 : patent cost relative to RD_US (JUPCOSTRD)',
                '6':'6 : fix delta_US = 0.05 and drop JUPCOST',
                '7':'7 : drop SRDUS'
                }

# baseline_number = '101'
for baseline_number in ['101','102','104']:
    path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
    dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
    
    moments_runs = []
    params_runs = []
    files_in_dir = next(os.walk(path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    # run_list = ['101','102','103','104']
    comments = []
    for run in run_list:
        p, sol, m = full_load_parameters_set(path+run+'/')
        moments_runs.append(m)
        params_runs.append(p)
        comments.append(comments_dic[run])
    dic_m = dict(zip(comments, moments_runs))
    dic_p = dict(zip(comments, params_runs))
    save_path = dropbox_path+'summary/'
    try:
        os.mkdir(save_path)
    except:
        pass
    moments.compare_moments(dic_m,
        save_path = save_path,list_of_moments = list_of_moments, contin_cmap = True)
    compare_params(dic_p,color_gradient=True,save = True, save_path = save_path)
    # moments.compare_moments(dic_m,list_of_moments = list_of_moments)
    # compare_params(dic_p,color_gradient=False)

#%% compare params

# dic = {#'3 : lin weights on dim':p3,
#         # '4 : sqrt weights on dim':p4,
#         # '6 : delta US free, SRDUS divided by 2':p6,
#         '7 : free nu, nu ='+str(p7.nu[1]):p7,
#        #  '10 : Nu_tilde = nu*5':p10,
#        #  '11 : Nu_tilde = nu*10':p11,
#         # '12 : Nu_tilde = nu, growth = 1.69%':p12,
#        #  '13 : Nu_tilde = nu/5':p13,
#        #  '14 : Ratio of nu targeted as a moment, target 1':p14,
#        #  '15 : Ratio of nu targeted as a moment, low weight':p15,
#        # '16 : Growth = growth*2':p16,
#        # '17 : Growth = growth*4':p17,
#        # '18 : Growth = growth*10':p18,
#         '19 : Fixed nu = 0.1':p19,
#        # '20 : Fixed nu = nu_tilde = 0.1':p20,
#        # '31 : equivalent to 7':p31,
#        # '32 : calibrated k, theta = 8':p32,
#        # '33 : calibrated nu low weight, calibrated k':p33,
#        # '34 : calibrated nu higher weight':p34,
#        '35 : calibrated nu calibrated k and sigma, nu ='+str(p35.nu[1]):p35,
#        '36 : calibrated nu higher weight, calibrated k and sigma, nu ='+str(p36.nu[1]):p36
#        }
# dic = {
#        '40 : Turnover US, weight 3, target 0.1':p40,
#        '41 : Turnover US, weight 3, target 0.2':p41,
#        '42 : Turnover US, weight 3, target 0.3':p42,
#        '43 : Turnover US, weight 3, target 0.05':p43,
#        '44 : Turnover US, weight 3, target 0.075':p44,
#        '45 : Turnover US, weight 3, target 0.025':p45,
#        '46 : Turnover US, weight 3, target 0.04':p46,
#        '47 : Turnover US, weight 3, target 0.06':p47,
#        '48 : Turnover US, weight 3, target 0.01':p48,
#        '50 : No turnover target, benchmark':p50
#        }

# dic = {
#         '51 : Turnover US, fixed nu, target 0.05':p51,
#         '52 : Turnover US, fixed nu, target 0.1':p52,
#         '53 : Turnover US, fixed nu, target 0.2':p53,
#         '54 : Turnover US, fixed nu, target 0.075':p54,
#         '55 : Turnover US, fixed nu, target 0.025':p55,
#         '56 : Turnover US, fixed nu, target 0.01':p56
#         }

# dic = {
#         '50 : No turnover target, no TO moment, no TE moment':p50,
#         '43 : Turnover US targeted 0.05, no TE moment':p43,
#         '65 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows targeted, theta calibrated = '+str(p65.theta[1].round(3)):p65,
#         '66 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows not targeted, theta calibrated = '+str(p66.theta[1].round(3)):p66,
#         '67 : Turnover US targeted 0.05, TE targeted 5, \n domestic pflows not targeted, theta calibrated = '+str(p67.theta[1].round(3))+'kappa rho calibrated':p67
#         }

# save = True
# save_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/trade_elasticity_calibration/'


#%%

# ps = [p40,p41,p42,p43,p44,p45,p46,p47,p48]
# ms = [m40,m41,m42,m43,m44,m45,m46,m47,m48]
# sols = [sol40,sol41,sol42,sol43,sol44,sol45,sol46,sol47,sol48]
ps = [p56,p55,p51,p54,p52,p53]
ms = [m56,m55,m51,m54,m52,m53]
sols = [sol56,sol55,sol51,sol54,sol52,sol53]

fig, ax = plt.subplots(figsize = (12,8))
ax2 = ax.twinx()
ax.plot([m.TO_target for m in ms],[p.nu_tilde[1] for p in ps], label = 'nu_tilde', color = 'r',lw=3)
# ax.plot(sorted([m.TO_target for m in ms]),sorted([p.nu[1] for p in ps]), 
#         label = 'nu', color = 'r',lw=3)
ax2.plot(sorted([m.TO_target for m in ms]),sorted([p.delta[0,1] for p in ps]), label = 'delta_US')
ax2.plot(sorted([m.TO_target for m in ms]),sorted([p.delta[:,1].mean() for p in ps]), label = 'mean_delta')
ax.set_xlabel('Turnover US target',fontsize=20)
ax.set_ylabel('nu_tilde',fontsize=20)
# ax.set_yticks(np.linspace(0,1,11))
ax.set_yscale('log')
ax2.set_ylabel('delta',fontsize=20)
ax2.grid(None)
ax.legend(fontsize=15)
ax2.legend(loc='lower right',fontsize=20)
plt.savefig(save_path+'nu_tilde_function_of_TO_target')
plt.show()

# fig, ax = plt.subplots(figsize = (12,8))
# # ax2 = ax.twinx()

# # ax.plot(sorted([m.TO_target for m in ms]),sorted([p.nu[1] for p in ps]), 
# #         label = 'nu', color = 'r',lw=3)
# ax.plot([m.TO_target for m in ms],[p.nu_tilde[1] for p in ps], label = 'nu_tilde')
# # ax2.plot(sorted([m.TO_target for m in ms]),sorted([p.delta[:,1].mean() for p in ps]), label = 'mean_delta')
# ax.set_xlabel('Turnover US target',fontsize=20)
# ax.set_ylabel('nu_tilde',fontsize=20)
# # ax.set_yticks(np.linspace(0,1,11))
# # ax2.set_ylabel('delta',fontsize=20)
# ax.set_yscale('log')
# # ax2.grid(None)
# ax.legend(fontsize=15)
# # ax2.legend(loc='center right',fontsize=20)
# # plt.savefig(save_path+'nu_and_nu_tilde')
# plt.show()

# for p, m, sol in zip(ps, ms, sols):

#%% counterfactual

baseline_nbr = '85'
baseline_path = 'calibration_results_matched_economy/'+baseline_nbr+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)

sol, baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
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

baseline.scale_P(p_baseline)
baseline.compute_price_indices(p_baseline)
baseline.compute_non_solver_quantities(p_baseline)

counterfactuals_by_country = {}

for c in p_baseline.countries:
    print(c)
    p = p_baseline.copy()
    sols_c = []
    deltas = np.logspace(-1,1,111)
    for delt in deltas:
        print(delt)
        p.delta[p.countries.index(c),1] = p_baseline.delta[p.countries.index(c),1] * delt
        # print(p.guess)
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
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
    counterfactuals_by_country[c] = sols_c

# utilities = pd.DataFrame(index = deltas, columns = p.countries, data = np.array([sol.U for sol in sols_c]))
# (utilities/np.abs(utilities.iloc[5])).plot(figsize=(12,8),logx=True)
# consumptions = pd.DataFrame(index = deltas, columns = p.countries, data = np.array([sol.nominal_final_consumption/sol.price_indices for sol in sols_c]))
# (consumptions/consumptions.iloc[5]).plot(figsize=(12,8),logx=True)
# CE = pd.DataFrame(index = deltas, 
#               columns = p.countries, 
#               data = np.array([sol.cons_eq_welfare for sol in sols_c]))
# CE
# ax = CE.plot(figsize=(16,12),logx=True,
#          fontsize = 20,color=sns.color_palette()[:CE.shape[1]])
# idxmaxes = CE.idxmax()
# maxes = CE.max()
# CE_max = pd.DataFrame({'idx' : idxmaxes, 'values' : maxes})
# CE_max.plot.scatter('idx','values',
#                     ax = ax,color=sns.color_palette()[:CE.shape[1]])
# ax.set_xlabel('Change in delta '+c, fontsize = 20)
# ax.set_ylabel('Consumption equivalent welfare', fontsize = 20)
# ax.scatter(CE.idxmax(),CE.max(),color=[mcolors.BASE_COLORS[k] for k in list(mcolors.BASE_COLORS.keys())[:CE.shape[1]]])
# CE.to_csv('/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilateral_patent_protection_43/'+c+'.csv')
# ax.figure.savefig('/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilateral_patent_protection_43/'+c)

#%% plot counterfactual
save = True
dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilaterat_patent_protection_'+baseline_nbr+'/'
try:
    os.mkdir(dropbox_path)
except:
    pass

for c in p.countries:
    consumption = {}
    consumption_eq_welfare = {}
    for j,c2 in enumerate(p.countries):
        consumption[c2] = [(sol.nominal_final_consumption[j]/sol.price_indices[j])
                           /(baseline.nominal_final_consumption[j]/baseline.price_indices[j])
                           for sol in counterfactuals_by_country[c]]
        consumption_eq_welfare[c2] = [sol.cons_eq_welfare[j]
                           for sol in counterfactuals_by_country[c]] 
    growth_rate = [sol.g for sol in counterfactuals_by_country[c]]
    fig,ax = plt.subplots(2,1,figsize = (10,14),constrained_layout=True)
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax2.plot(deltas,growth_rate, color='k', ls = '--', label = 'Growth rate')
    # ax2.set_ylim(0.016,0.026)
    ax3.plot(deltas,growth_rate, color='k', ls = '--', label = 'Growth rate')
    # ax3.set_ylim(0.016,0.026)
    ax2.set_ylabel('Growth rate')
    ax3.legend(loc = (1,1.05))
    ax[0].set_ylabel('Normalized consumption')
    ax3.set_ylabel('Growth rate')
    ax[1].set_ylabel('Consumption equivalent welfare')
    for j,c2 in enumerate(p.countries):
        ax[0].plot(deltas,consumption[c2],label = c2,color = sns.color_palette()[j])
        ax[1].plot(deltas,consumption_eq_welfare[c2],label = c2,color = sns.color_palette()[j])
    ax[0].legend(loc = 'center right')
    ax[0].scatter([deltas[np.argmax(consumption[c3])] for c3 in p.countries],
                  [max(consumption[c3]) for c3 in p.countries],
                  color=sns.color_palette()[:len(p.countries)])
    ax[1].scatter([deltas[np.argmax(consumption_eq_welfare[c3])] for c3 in p.countries],
                  [max(consumption_eq_welfare[c3]) for c3 in p.countries],
                  color=sns.color_palette()[:len(p.countries)])
    # ax[1].legend()
    plt.title('Counterfactual - patent protection '+c,pad = 20)
    if save:
        plt.savefig(dropbox_path+c)
    plt.show()


#%% plot counterfactual

fig,ax = plt.subplots(figsize = (12,8))
ax2 = ax.twinx()
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
