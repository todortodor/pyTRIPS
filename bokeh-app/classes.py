#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:27:06 2022

@author: simonl
"""
# from os.path import dirname
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma
import time
import os
import seaborn as sns

class parameters:     
    def __init__(self, n=7, s=2, data_path = None):
        if data_path is None:
            data_path = 'data/'
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
        self.d = 1.0
        self.khi = 1.0
        
        # self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        # self.off_diag_mask[np.s_[::(N+1)*S]] = False
        # self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        # self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        # self.diag_mask = np.invert(self.off_diag_mask)
        
        self.unit = 1e6
        # print(data_path)
        self.trade_flows = pd.read_csv(data_path+'country_country_sector_moments.csv',index_col=[1,0,2]).sort_index().values.squeeze()/self.unit
        self.trade_flows = self.trade_flows.reshape((N,N,S))
        self.OUT = self.trade_flows.sum()
        # self.trade_shares = (self.trade_flows/(np.diagonal(self.trade_flows).transpose())[:,None,:])
        self.trade_shares = self.trade_flows/self.trade_flows.sum()
        # self.trade_shares = (self.trade_flows).reshape((N,N,S))
        
        self.data = pd.read_csv(data_path+'country_moments.csv',index_col=[0])
        
        self.labor_raw = np.concatenate(
            (self.data.labor.values,np.ones(n)*self.data.labor.values[-1])
            )[:n]
        
        self.gdp_raw = np.concatenate(
            (self.data.gdp.values,np.ones(n)*self.data.gdp.values[-1])
            )[:n]
        
        self.r_hjort = (self.data.gdp.iloc[0]*np.array(self.data.labor)/(self.data.labor.iloc[0]*np.array(self.data.gdp)))**(1-self.khi)
        
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
                         'eta':co,
                         'd':0.5}
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
                         'eta':cou,
                         'd':cou}
        
        self.idx = {'sigma':pd.Index(self.sectors, name='sector'),
                    'theta':pd.Index(self.sectors, name='sector'),
                    'rho':pd.Index(['scalar']),
                    'gamma':pd.Index(['scalar']),
                    'zeta':pd.Index(self.sectors, name='sector'),
                    'nu':pd.Index(self.sectors, name='sector'),
                    'nu_tilde':pd.Index(self.sectors, name='sector'),
                    'kappa':pd.Index(['scalar']),
                    'd':pd.Index(['scalar']),
                    'khi':pd.Index(['scalar']),
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
                    'd':None,
                    'khi':None,
                    'alpha':None,
                    'beta':None,
                      'T':None,
                     'eta':[np.s_[::S]]}
        
        self.mask = {}
        
        for par_name in ['eta','k','rho','alpha','fe','T','fo','sigma','theta','beta','zeta',
                         'g_0','kappa','gamma','delta','nu','nu_tilde','d','khi']:
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
         'kappa','gamma','delta','nu','nu_tilde','d','khi']
            
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
        for par in self.calib_parameters:
            param = np.array(getattr(self,par))
            size = param[self.mask[par]].size
            param[self.mask[par]] = vec[idx_from:idx_from+size]
            setattr(self,par,param)
            idx_from += size
            
    def update_sigma_with_SRDUS_target(self,m):
        self.sigma[1] = 1+m.SRDUS_target/(m.sales_mark_up_US_target - 1)
        
    def update_khi_and_r_hjort(self, new_khi):
        #new_khi = 1 will remove the hjort factor
        self.khi = new_khi
        self.r_hjort = (self.data.gdp.iloc[0]*np.array(self.data.labor)/(self.data.labor.iloc[0]*np.array(self.data.gdp)))**(1-self.khi)**(1-self.khi)
            
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
            par = getattr(self,pa_name)
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
            if pa_name == 'khi':
                try:
                    df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
                    self.update_khi_and_r_hjort(df.values[0])
                except:
                    pass
            else:
                try:
                    df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
                    setattr(self,pa_name,df.values.squeeze().reshape(np.array(getattr(self,pa_name)).shape))
                except:
                    if pa_name == 'd':
                        self.d = np.array(1.0)
                    else:
                        pass
            
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
        prefact = p.k * p.eta * self.l_R**(1-p.kappa) /(p.k-1)
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
        # if np.any(np.isnan(self.PSI_CL)):
        #     print('prefact',prefact)
        #     print('A_tilde',A_tilde)
        #     print('A',A)
        #     print('term',(p.nu/A)[None, None:]-B_tilde*A_tilde[None, None, :])
        #     time.sleep(30)
        # self.PSI_CL = np.maximum(np.einsum('is,s,nis -> nis',
        #                         prefact,
        #                         1/A_tilde,
        #                         (p.nu/A)[None, None:]-B_tilde*A_tilde[None, None, :]),0)
        # print(self.psi_star[...,1].min())
        self.PSI_CL[:, :, 0] = 0
        self.PSI_CL[self.PSI_CL<0] = 0 #!!!!
        self.PSI_M[self.PSI_M<0] = 0
        
        # print(self.PSI_CL.min())
        # assert np.isnan(self.PSI_CL).sum() == 0, 'nan in PSI_CL'
        # assert not np.any(np.einsum('njs->ns', self.PSI_M)+np.einsum('njs->ns', self.PSI_CL) > 1),'PSI_M,CL too high'
        # print((np.einsum('njs->ns', self.PSI_M) +
        #                  np.einsum('njs->ns', self.PSI_CL)).max())
        self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
                         np.einsum('njs->ns', self.PSI_CL))
        self.PSI_CD[self.PSI_CD<0] = 0
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
        # self.P_M[:, 1:] = (A[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        # print(A.min())
        # print(B.min())
        # print(C.min())
        self.P_M[:, 1:] = np.divide(
            (A+B+C)[:, 1:], A[:, 1:], out=np.full_like(A[:, 1:], np.inf), where=A[:, 1:] > 0)**(1/(p.sigma-1))[None, 1:]
        
        self.P_CL = np.ones((p.N, p.S))
        self.P_CL[:, 0] = np.inf

        # self.P_CL[:, 1:] = (B[:, 1:]/(A+B+C)[:, 1:])**(1/(1-p.sigma))[None, 1:]
        self.P_CL[:, 1:] = np.divide(
            (A+B+C)[:, 1:], B[:, 1:], out=np.full_like(B[:, 1:], np.inf), where=B[:, 1:] > 0)**(1/(p.sigma-1))[None, 1:]
        
        # self.P_CD = (C/(A+B+C))**(1/(1-p.sigma))[None, :]
        self.P_CD = np.divide(
            A+B+C, C, out=np.full_like(C, np.inf), where=C > 0)**(1/(p.sigma-1))[None, :]
        
        self.P_CL[np.isnan(self.P_CL)] = np.inf #!!!
        self.P_CD[np.isnan(self.P_CD)] = np.inf
        self.P_M[np.isnan(self.P_M)] = np.inf

    def compute_trade_shares(self, p):
        # numerator_prefact_A = np.einsum('nis,nis->nis',
        #                       self.PSI_M,
        #                       self.phi**(p.sigma-1)[None, None, :])
        temp = (self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
        denominator_M = np.zeros((p.N, p.N, p.S))
        denominator_M[..., 1:] = np.einsum('nis,ns,ns->nis',
                                self.PSI_M[..., 1:],
                                # 1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                np.divide(1,temp, out=np.full_like(temp, np.inf), where=temp>0),
                                self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CL = np.zeros((p.N, p.N, p.S))
        temp = (self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
        denominator_CL[..., 1:] = np.einsum('nis,ns,ns->nis',
                                self.PSI_CL[..., 1:],
                                # 1/((self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                np.divide(1,temp, out=np.full_like(temp, np.inf), where=temp>0),
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
        
        self.X_M[np.isnan(self.X_M)] = 0 #!!!
        self.X_CD[np.isnan(self.X_CD)] = 0
        self.X_CL[np.isnan(self.X_CL)] = 0

    def compute_labor_allocations(self, p, l_R=None, assign=True):
        if l_R is None:
            l_R = self.l_R
        # print(p.d_np)
        d_np = (p.d-1)*np.identity(p.N)+1
        
        l_Ae = np.einsum('ni,i,s,is,is,nis -> ins',
                         d_np, #!!!
                         p.r_hjort,
                         p.fe,
                         p.eta,
                         l_R**(1-p.kappa),
                         np.divide(
                             1, self.psi_star**(p.k), out=np.zeros_like(self.psi_star), where=self.psi_star != np.inf)
                         )
        l_Ao = np.einsum('ins,s,s -> ins',
                         l_Ae,
                         p.fo,
                         1/p.fe
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
        d_np = (p.d-1)*np.identity(p.N)+1
        # d_np = np.ones((p.N,p.N))
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
                      p.fo+np.einsum('n,i,ni,i,s->nis',
                                     self.w,
                                     1/self.w,
                                     d_np,  #!!!
                                     p.r_hjort,
                                     p.fe),
                      np.divide(
                          1, self.psi_star**(p.k), out=np.zeros_like(self.psi_star), where=self.psi_star != np.inf)
                      )
        l_R = (p.eta*(A+B)/(p.k-1))**(1/p.kappa)
        # assert np.isnan(l_R).sum() == 0, 'nan in l_R'
        return l_R

    def compute_psi_star(self, p):
        d_np = (p.d-1)*np.identity(p.N)+1
        # d_np = np.ones((p.N,p.N))
        psi_star = np.einsum('s,i,nis,s,nis,nis,nis,ns,ns -> nis',
                             p.sigma,
                             self.w,
                             self.PSI_M,
                             1/p.nu,
                             np.divide(1, self.X_M, out=np.full_like(
                                 self.X_M, np.inf), where=self.X_M != 0),
                             1/(p.trade_shares*self.Z.sum()),
                             p.fo[None, None, :]+np.einsum('n,i,ni,i,s -> nis',
                                                           self.w,
                                                           1/self.w,
                                                           d_np,
                                                           p.r_hjort,
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
        temp = (self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
        denominator_CL[..., 1:] = np.einsum('nis,ns,ns->nis',
                                self.PSI_CL[..., 1:],
                                # 1/((self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                np.divide(1,temp, out=np.full_like(temp, np.inf), where=temp>0),
                                self.P_CL[..., 1:]**(1-p.sigma[None, 1:])
                                )
        # denominator_CL[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
        #                         self.PSI_CL[..., 1:],
        #                         self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
        #                         1/((self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
        #                         self.P_CL[..., 1:]**(1-p.sigma[None, 1:])
        #                         )
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
        # if np.isnan(B).sum() > 0:
        #     print(self.PSI_CL)
        #     print(self.phi)
        #     time.sleep(10)
        C = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.sigma)*(A+B+C))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        # price_indices = ( (gamma((p.theta+1-p.sigma)/p.sigma)*(A+B+C))**(p.beta[None, :]/(1- p.sigma[None, :])) ).prod(axis=1)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :]/(p.sigma[None, :]-1)) ).prod(axis=1)
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
        # exp = 1-1/p.gamma
        # self.U = self.cons**(exp)/(p.rho-self.g*exp)/exp
        pass
    
    def compute_semi_elast_RD_delta(self,p):
        self.r_NP = np.zeros(p.S)
        self.r_NP[1:] = self.r + p.zeta[1:] + p.nu[1:] - self.g + self.g_s[1:]
        
        self.DT = np.zeros((p.N,p.S))
        self.DT[:,1:] = np.einsum('s,is,is->is',
                            p.nu[1:],
                            1/(self.r_NP[None,1:]-p.nu[None,1:]+p.delta[:,1:]),
                            1/(self.r_NP[None,1:]+p.delta[:,1:]))
        
        self.semi_elast_RD_delta = np.zeros((p.N,p.S))
        
        numerator_prefact = p.k*np.einsum('is,is,is,is->is',
                                      p.delta[:,1:]**2,
                                      np.diagonal(self.profit[...,1:]).transpose(),
                                      np.diagonal(self.psi_star[...,1:]).transpose()**(1-p.k),
                                      self.DT[:,1:]
                                      )
        numerator_sum = 1/(self.r_NP[None,1:]-p.nu[None,1:]+p.delta[:,1:])\
                            + 1/(self.r_NP[None,1:]+p.delta[:,1:])
        denominator = p.kappa*np.einsum('mis,mis->is',
                                self.profit[...,1:],
                                p.k/self.r_NP[None,None,1:]+np.einsum('mis,ms->mis',
                                                                      self.psi_star[...,1:]**(1-p.k),
                                                                      self.DT[:,1:]),
                                )
        
        self.semi_elast_RD_delta[...,1:] = numerator_prefact*numerator_sum/denominator
    
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
        self.compute_semi_elast_RD_delta(p)

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
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'OUT', 'KM', 'KM_GDP', 'RD','RD_US','RD_RUS', 'RP',
                               'SRDUS', 'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US',
                               'SPFLOWDOM_RUS', 'SPFLOW_RUS','SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST',
                               'JUPCOSTRD','SINNOVPATUS','TO','TE','DOMPATRATUSEU','DOMPATUS','DOMPATEU',
                               'SPATORIG','SPATDEST','TWSPFLOW','TWSPFLOWDOM','ERDUS']
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
                             'KM_GDP':5,
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
                              'DOMPATUS':1,
                              'DOMPATEU':1,
                              'SPATORIG':2,
                              'SPATDEST':2,
                              'TWSPFLOW':1,
                              'TWSPFLOWDOM':1,
                              'ERDUS':3
                             }
        
        # self.total_weight = sum([self.weights_dict[mom] for mom in self.list_of_moments])
        
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'KM_GDP':pd.Index(['scalar']), 
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
                    'NUR':pd.Index(['scalar']),
                    'ERDUS':pd.Index(['scalar'])
                    }
        
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
        return ['GPDIFF', 'GROWTH', 'KM','KM_GDP', 'OUT', 'RD','RD_US','RD_RUS', 'RP', 
                'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US',
                'SPFLOWDOM_RUS', 'SPFLOW_RUS','DOMPATUS','DOMPATEU',
                'SRDUS', 'SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST','JUPCOSTRD', 'TP', 'Z', 
                'SINNOVPATEU','SINNOVPATUS','TO','TE','NUR','DOMPATRATUSEU',
                'SPATDEST','SPATORIG','TWSPFLOW','TWSPFLOWDOM','ERDUS']
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
    
    def load_data(self,data_path = None):
        if data_path is None:
            data_path = 'data/'
        self.c_moments = pd.read_csv(data_path+'country_moments.csv',index_col=[0])
        self.cc_moments = pd.read_csv(data_path+'country_country_moments.csv',index_col=[1,0]).sort_index()
        self.ccs_moments = pd.read_csv(data_path+'country_country_sector_moments.csv',index_col=[1,0,2]).sort_index()
        self.moments = pd.read_csv(data_path+'scalar_moments.csv',index_col=[0])
        self.description = pd.read_csv(data_path+'moments_descriptions.csv',sep=';',index_col=[0])
        
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
        self.KM_GDP_target = self.KM_target*self.RD_US_target
        self.NUR_target = self.moments.loc['NUR'].value
        self.SRDUS_target = self.moments.loc['SRDUS'].value
        self.GPDIFF_target = self.moments.loc['GPDIFF'].value 
        self.GROWTH_target = self.moments.loc['GROWTH'].value 
        self.ERDUS_target = self.moments.loc['ERDUS'].value 
        self.TE_target = self.moments.loc['TE'].value 
        self.TO_target = np.array(0.05)
        # self.GROWTH_target = self.GROWTH_target*10
        self.Z_target = self.c_moments.expenditure.values/self.unit
        self.JUPCOST_target = self.moments.loc['JUPCOST'].value
        self.JUPCOSTRD_target = self.moments.loc['JUPCOST'].value/(self.c_moments.loc[1,'rnd_gdp']*self.c_moments.loc[1,'gdp']/self.unit)
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
                            plt.savefig(save_plot+'_'+mom+'.png')
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
                            plt.savefig(save_plot+'_'+mom+'.png')
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
                            plt.savefig(save_plot+'_'+mom+'.png')
                        plt.show()
               
        fig,ax = plt.subplots(figsize = (12,8))
        ax.scatter(scalar_moments,scalar_moments_ratio)
        ax.plot(scalar_moments,np.zeros_like(scalar_moments,dtype='float'),ls = '--', lw=1, color = 'k')
        # if np.any(np.array(scalar_moments_ratio)>10):
        #     plt.yscale('log')
        plt.title('scalar moments, deviation')
        if save_plot is not None:
            plt.savefig(save_plot+'_scalar_moments'+'.png')
        plt.show()
            
    def write_moments(self, path):
        for mom in self.list_of_moments:
            df = pd.DataFrame(data = {'target':getattr(self,mom+'_target').ravel(),
                                      'moment':getattr(self,mom).ravel()})
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
        # numerator = var.w[:,None]*var.l_R + np.einsum('i,ins->is',var.w,var.l_Ao)\
        #     + np.einsum('n,ins->is',var.w,var.l_Ao)
        numerator = var.w[:,None]*var.l_R + np.einsum('i,ins->is',var.w,var.l_Ao)\
            + np.einsum('n,ins->is',var.w,var.l_Ae)
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
        self.KM_GDP = self.KM*self.RD_US
        
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
        self.JUPCOST = var.pflow[2,0]*(p.fo[1]*var.w[0] + p.r_hjort[2]*p.fe[1]*var.w[2])
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
        self.sales_mark_up_US = 1+self.SRDUS_target/(p.sigma[1] - 1)
        self.sales_mark_up_US_target = 1+self.SRDUS_target/(p.sigma[1] - 1)
        
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
        
    def compute_ERDUS(self,var,p):
        self.ERDUS = var.semi_elast_RD_delta[0,1]
        
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
        self.compute_ERDUS(var, p)
        
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
            if hasattr(self, mom):
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

