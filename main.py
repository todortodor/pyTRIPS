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
        self.T = np.ones(N)*1.5  # could be anything >0
        self.k = 1.33350683                 #
        self.rho = 0.02  # 0.001 - 0.02
        self.alpha = np.concatenate((np.array([0.5758, 0.3545]),np.ones(s)*0.5))[:s]
        self.fe = np.ones(S)*2.7  # could be over one
        self.fo = np.ones(S)*2.3  # could be over one
        self.sigma = np.ones(S)*1.8125  #
        self.theta = np.ones(S)*5   #
        self.beta = np.concatenate((np.array([0.735, 0.265]),np.ones(s)*0.5))[:s]
        self.beta = self.beta / self.beta.sum()
        self.zeta = np.ones(S)*0.01
        self.g_0 = 0.01  # makes sense to be low
        self.tau = np.ones((N, N, S))*4
        for i in range(self.tau.shape[2]):
            np.fill_diagonal(self.tau[:, :, i], 1)
        self.kappa = 0.5            #
        self.gamma = 0.5       #
        self.delta = np.ones((N, S))*0.1
        self.nu = np.ones(S)*0.1   #
        self.nu_tilde = np.ones(S)*0.1
        
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
        
        self.deficit_share_world_gdp = self.deficit_raw/self.data.output.sum() 
        
        # self.unit = self.gdp_raw.mean()
        self.unit = 1e6
        
        co = 1e-3
        cou = 1e5
        self.lb_dict = {'sigma':1,
                        'theta':co,
                        'rho':co,
                        'gamma':co,
                        'zeta':0,
                        'nu':0,
                        'nu_tilde':0,
                        'kappa':0,
                        'k':1+co,
                        'tau':1,
                        'fe':co,
                        'fo':co,
                        'delta':co,
                        'g_0':0,
                        'alpha':co,
                         'beta':co,
                         'T':co,
                         'eta':co}
        self.ub_dict = {'sigma':cou,
                        'theta':cou,
                        'rho':cou,
                        'gamma':cou,
                        'zeta':cou,
                        'nu':cou,
                        'nu_tilde':cou,
                        'kappa':1-co,
                        'k':cou,
                        'tau':cou,
                        'fe':cou,
                        'fo':cou,
                        'delta':cou,
                        'g_0':cou,
                        'alpha':1,
                         'beta':1,
                         'T':cou,
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
                    'tau':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                     , names=['destination','origin','sector']),
                    'fe':pd.Index(self.sectors, name='sector'),
                    'fo':pd.Index(self.sectors, name='sector'),
                    'delta':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                       , names=['country','sector']),
                    'g_0':pd.Index(['scalar']),
                    'alpha':pd.Index(self.sectors, name='sector'),
                    'beta':pd.Index(self.sectors, name='sector'),
                     'T':pd.Index(self.countries, name='country'),
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
                    'tau':[np.s_[::(N+1)*S],np.s_[1::(N+1)*S]],
                    'fe':[np.s_[0]],
                    'fo':[np.s_[0]],
                    'delta':[np.s_[::S]],
                    'g_0':None,
                    'alpha':None,
                    'beta':None,
                     'T':None,
                     'eta':[np.s_[::S]]}
        
        self.mask = {}
        
        for par_name in ['eta','T','k','rho','alpha','fe','fo','sigma','theta','beta','zeta',
                         'g_0','tau','kappa','gamma','delta','nu','nu_tilde']:
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
    
    @staticmethod
    def get_list_of_params():
        return ['eta','T','k','rho','alpha','fe','fo','sigma','theta','beta','zeta','g_0',
         'tau','kappa','gamma','delta','nu','nu_tilde']
            
    def guess_from_params(self):
        price_guess = self.data.price_level.values
        Z_guess = self.data.expenditure.values/self.unit
        w_guess = 10*self.data.gdp.values*self.unit_labor/(self.data.labor.values*self.unit)
        l_R_guess = np.repeat(self.labor[:,None]/200, self.S-1, axis=1).ravel()
        psi_star_guess = np.ones((self.N,self.N,(self.S-1))).ravel()*10
        vec = np.concatenate((price_guess,w_guess,Z_guess,l_R_guess,psi_star_guess), axis=0)
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
            if (isinstance(vars(self)[k], np.ndarray) or isinstance(vars(self)[k], float)):
                if np.all(np.isclose(vars(self)[k], vars(p2)[k])):
                    print(k, 'identical')
                else:
                    diffs.append(k)
        
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
        pass

    def guess_price_indices(self, price_indices_init):
        self.price_indices = price_indices_init

    def guess_patenting_threshold(self, psi_star_init):
        self.psi_star = psi_star_init

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_expenditure(self, Z_init):
        self.Z = Z_init

    def guess_labor_research(self, l_R_init):
        self.l_R = l_R_init

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        frame = deepcopy(self)
        return frame
    
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
        Z = self.Z
        vec = np.concatenate((price,w,Z,l_R,psi_star), axis=0)
        return vec
    
    def make_dataframe_countries(self,p,unit='code'):
        # df = pd.DataFrame(index = p.countries)
        # for qty in ['Z']:
        #     df[qty+'_code_units'] = getattr(self,qty)
        #     df[qty+'_real_units'] = getattr(self,qty)*p.unit
        # for qty in ['l_P']:
        #     df[qty+'_code_units'] = getattr(self,qty)
        #     df[qty+'_real_units'] = getattr(self,qty)*p.unit_labor
        # df['price_indices_code_units'] = self.price_indices
        # # df['price_indices_real_units'] = self.price_indices/p.unit
        # df['w_code_units'] = self.w
        # df['w_real_units'] = self.w*p.unit/p.unit_labor
        self.compute_non_solver_quantities(p)
        df = pd.DataFrame(index = p.countries)
        if unit=='code':
            for qty in ['Z','price_indices','w','l_P','gdp']:
                df[qty] = getattr(self,qty).round(5)
            df['l_R'] = self.l_R.sum(axis=1)
        if unit=='real':
            df['Z real'] = (self.Z*p.unit).round(5)
            df['price_indices'] = (self.price_indices).round(5)
            df['w real'] = (self.w*p.unit_labor/p.unit).round(5)
            df['l_P real'] = (self.l_P*p.unit_labor).round(5)
            df['gdp real'] = (self.gdp*p.unit).round(5)
            df['l_R real'] = (self.l_R.sum(axis=1)*p.unit_labor).round(5)
        return df
            
        
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
        # print(self.psi_star[...,1].min())
        self.PSI_CL[:, :, 0] = 0
        # assert np.isnan(self.PSI_CL).sum() == 0, 'nan in PSI_CL'
        # assert not np.any(np.einsum('njs->ns', self.PSI_M)+np.einsum('njs->ns', self.PSI_CL) > 1),'PSI_M,CL too high'
        # print((np.einsum('njs->ns', self.PSI_M) +
        #                  np.einsum('njs->ns', self.PSI_CL)).max())
        self.PSI_CD = 1-(np.einsum('njs->ns', self.PSI_M) +
                         np.einsum('njs->ns', self.PSI_CL))
        # print(self.PSI_CD.min())
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
                                      self.Z
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
        # print(self.PSI_CL.min())
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
                                       self.Z
                                       )
        # assert np.isnan(self.X_CL).sum() == 0, 'nan in X_CL'
        self.X_CD = np.einsum('nis,ns,ns,s,n->nis',
                              self.phi,
                              1/(self.phi.sum(axis=1)),
                              (self.P_CD**(1-p.sigma[None, :])),
                              p.beta,
                              self.Z
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

    def compute_expenditure(self, p):
        A = np.einsum('nis->i', self.X_CD+self.X_CL+self.X_M)
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = np.einsum('i,n->i', p.deficit_share_world_gdp, self.Z)
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Z = (A+B-(C+D))
        # assert np.isnan(Z).sum() == 0, 'nan in Z'
        return Z
    
    def compute_total_trade_flows(self,p):
        self.X = self.X_CD+self.X_CL+self.X_M
    
    def compute_nominal_value_added(self,p):
        self.nominal_value_added = p.alpha[None, :]*(self.X-self.X_M/p.sigma[None, None, :]).sum(axis=0)
    
    def compute_nominal_intermediate_input(self,p):
        self.nominal_intermediate_input = np.einsum('s,is->is',
                           (1-p.alpha)/p.alpha,
                           self.nominal_value_added)
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - self.nominal_intermediate_input.sum(axis=1)
        
    def compute_gdp(self,p):
        self.gdp = self.nominal_final_consumption + p.deficit_share_world_gdp*self.w*p.labor + self.w*(p.labor - self.l_P)
    
    def compute_profit(self,p):
        self.profit = np.einsum('nis,s,i,nis->nis',
                                self.X_M,
                                1/p.sigma,
                                1/self.w,
                                np.divide(1, self.PSI_M, out=np.zeros_like(
                                    self.PSI_M), where=self.PSI_M != 0))
    
    def compute_non_solver_quantities(self,p):
        self.compute_total_trade_flows(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_profit(p)
    
    def solve_price_indices(self, p, price_init=None, tol_p=1e-10, 
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
    
    def solve_psi_star(self, p, psi_star_init=None, price_init=None, 
                       tol_psi=1e-10, plot_convergence=False
                       , plot_cobweb = False, accelerate_when_stable = True,
                       accelerate = False):
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
        damping = 5        
        while condition:
            if count != 0:
                if accelerate:
                    psi_star_new = psi_star_new.ravel()
                    psi_star_old = psi_star_old.ravel()
                    aa_psi_star.apply(psi_star_new, psi_star_old)
                    psi_star_new = psi_star_new.reshape(p.N,p.N,p.S)
                    psi_star_old = psi_star_old.reshape(p.N,p.N,p.S)
                psi_star_old = (psi_star_new+(damping-1)*psi_star_old)/damping

            self.guess_patenting_threshold(psi_star_old)
            assert np.isnan(self.psi_star).sum() == 0, 'nan in psi_star'
            self.compute_aggregate_qualities(p)
            self.solve_price_indices(
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
            if accelerate_when_stable and np.all(np.array(convergence[-10:])<1e-1):
                damping = 2
            if plot_convergence:
                plt.title('psi star')
                plt.semilogy(convergence)
                plt.show()
            if plot_cobweb:
                # cob_psi_star.append_old_new(psi_star_old[1,1,1],psi_star_new[1,1,1])
                cob_psi_star.append_old_new(psi_star_old.min(),psi_star_new.min())
                cob_psi_star.plot(count=count)
            
        self.psi_star = psi_star_new
    
    def solve_l_R(self, p, l_R_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                  w_init=None, Z_init=None, plot_convergence=False
                  ,plot_cobweb = False):
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
        damping = 5
        while condition_l_R:
            if count != 0:
                l_R_new = l_R_new.ravel()
                l_R_old = l_R_old.ravel()
                # aa_l_R.apply(l_R_new, l_R_old)
                l_R_old = (l_R_new.reshape(p.N, p.S) + (damping-1)*l_R_old.reshape(p.N, p.S))/damping
            self.guess_labor_research(l_R_old)
            self.compute_growth(p)
            self.solve_psi_star(p, psi_star_init=self.psi_star,
                                price_init=self.price_indices)
            if np.any(self.compute_labor_allocations(p, l_R=l_R_old, assign=False) < 0):
                print('non positive production labor')
            l_R_new = self.compute_labor_research(p)
            condition_l_R = np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new) > tol_m
            convergence_l_R.append(np.linalg.norm(
                l_R_new - l_R_old)/np.linalg.norm(l_R_new))
            count += 1
            if np.all(np.array(convergence_l_R[-10:])<0.1):
                damping=1
            if plot_cobweb:
                cob_l_R.append_old_new(l_R_old[1,1],l_R_new[1,1])
                cob_l_R.plot(count=count)
            
            if plot_convergence:
                plt.semilogy(convergence_l_R, label='l_R')
                plt.show()
        self.l_R = l_R_new
    
    def solve_w(self, p, w_init=None, psi_star_init=None, price_init=None,
                tol_m=1e-8, plot_convergence=False, plot_cobweb = False,
                accelerate = True):
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
        aa_options_w_Z = {'dim': p.N,
                          'mem': 5,
                          'type1': False,
                          'regularization': 1e-12,
                          'relaxation': 1,
                          'safeguard_factor': 1,
                          'max_weight_norm': 1e6}
        aa_w = aa.AndersonAccelerator(**aa_options_w_Z)
        damping = 5
        while condition_w:
            # print(count,'wage iteration\nresearch problem solved')
            if count != 0:
                if accelerate:
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
            if np.all(np.array(convergence_w[-10:])<0.1):
                damping=1
            if plot_cobweb:
                cob_w.append_old_new(w_old[1],w_new[1])
                cob_w.plot(count=count)
            if plot_convergence:
                plt.semilogy(convergence_w, label='wage')
                plt.show()       
        self.w = w_new

    def solve_Z(self, p, Z_init=None, psi_star_init=None, price_init=None, tol_m=1e-8,
                plot_convergence=False, plot_cobweb = True):
        if plot_cobweb:
            cob_Z = cobweb('Z')
        Z_new = None
        if Z_init is None:
            Z_old = np.ones(p.N)
        else:
            Z_old = Z_init
        condition_Z = True
        count = 0
        convergence_Z = []
        aa_options_Z = {'dim': p.N,
                        'mem': 5,
                        'type1': False,
                        'regularization': 1e-12,
                        'relaxation': 1,
                        'safeguard_factor': 1,
                        'max_weight_norm': 1e6}
        aa_Z = aa.AndersonAccelerator(**aa_options_Z)
        while condition_Z:
            print(count)
            if count != 0:
                aa_Z.apply(Z_new, Z_old)
                Z_old = Z_new
            self.guess_expenditure(Z_old)
            self.solve_w(p, w_init=self.w)
            Z_new = self.compute_expenditure(p)
            condition_Z = np.linalg.norm(
                Z_new - Z_old)/np.linalg.norm(Z_new) > tol_m
            convergence_Z.append(np.linalg.norm(
                Z_new - Z_old)/np.linalg.norm(Z_new))
            count += 1
            if plot_cobweb:
                cob_Z.append_old_new(Z_old[1],Z_new[1])
                cob_Z.plot(count=count)
            if plot_convergence:
                plt.semilogy(convergence_Z, label='Z')
                plt.show()
        self.Z = Z_new
        self.num_scale_solution(p)

    def num_scale_solution(self, p):
        numeraire = self.price_indices[0]
        
        for qty in ['X_CD','X_CL','X_M','price_indices','w','Z']:
            setattr(self, qty, getattr(self, qty)/numeraire)
        
        self.phi = self.phi*numeraire**p.theta[None,None,:]
    
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
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SPFLOW', 'SRDUS', 'SRGDP', 'STFLOW']
        else:
            self.list_of_moments = list_of_moments
        self.weights_dict = {'GPDIFF':1, 
                             'GROWTH':1, 
                             'KM':1, 
                             'OUT':1, 
                             'RD':1, 
                             'RP':1, 
                             'SPFLOW':1, 
                             'SRDUS':1, 
                             'SRGDP':1, 
                             'STFLOW':1}
        
        # self.total_weight = sum([self.weights_dict[mom] for mom in self.list_of_moments])
        
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'OUT':pd.Index(['scalar']), 
                    'RD':pd.Index(self.countries, name='country'), 
                    'RP':pd.Index(self.countries, name='country'), 
                    'SPFLOW':pd.MultiIndex.from_tuples([(c1,c2) for c1 in self.countries for c2 in self.countries if c1 != c2]
                                           , names=['destination','origin']),
                    'SRDUS':pd.Index(['scalar']), 
                    'SRGDP':pd.Index(self.countries, name='country'), 
                    'STFLOW':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                     , names=['destination','origin','sector'])}
    
    @staticmethod
    def get_list_of_moments():
        return ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SPFLOW', 'SRDUS', 'SRGDP', 'STFLOW']
    
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
        self.OUT_target = self.c_moments.expenditure.sum()/self.unit
        self.SRGDP_target = (self.c_moments.gdp/self.c_moments.price_level).values \
                            /(self.c_moments.gdp/self.c_moments.price_level).sum()
        self.RP_target = self.c_moments.price_level.values
        self.RD_target = self.c_moments.rnd_gdp.values
        self.KM_target = self.moments.loc['KM'].value
        self.SRDUS_target = self.moments.loc['SRDUS'].value
        self.GPDIFF_target = self.moments.loc['GPDIFF'].value 
        self.GROWTH_target = self.moments.loc['GROWTH'].value 
        self.Z_target = self.c_moments.expenditure.values/self.unit
    
    def plot_moments(self, list_of_moments, plot = True):
        for mom in list_of_moments:
            if np.array(getattr(m,mom)).size == 1:
                print(mom+' : ',getattr(m,mom),
                      mom+' target  : ',
                      getattr(m,mom+'_target'))
            else:
                if plot == True:
                    fig,ax = plt.subplots(figsize = (12,8))
                    ax.scatter(getattr(m,mom+'_target').ravel(),getattr(m,mom).ravel())
                    ax.plot([getattr(m,mom+'_target').min(),
                              getattr(m,mom+'_target').max()]
                            ,[getattr(m,mom+'_target').min(),
                              getattr(m,mom+'_target').max()])
                    ax.plot([0,
                              getattr(m,mom+'_target').max()]
                            ,[0,
                              getattr(m,mom+'_target').max()])
                    ax.set_xlabel('target')
                    # plt.xlim((0.9*getattr(m,mom+'_target').min(),
                    #           1.1*getattr(m,mom+'_target').max()))
                    # plt.ylim((0.9*getattr(m,mom+'_target').min(),
                    #           1.1*getattr(m,mom+'_target').max()))
                    # plt.xlim((0,
                    #          1.1*getattr(m,mom+'_target').max()))
                    # plt.ylim((0,
                    #          1.1*getattr(m,mom+'_target').max()))
                    plt.title(mom+' targeting')
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.show()
            
    def write_moments(self, path, list_of_moments):
        for mom in list_of_moments:
            df = pd.DataFrame(data = {'target':getattr(m,mom+'_target').ravel(),
                                      'moment':getattr(m,mom).ravel()})
            df.to_csv(path+mom+'.csv',index=False)
        df = pd.DataFrame(data = self.list_of_moments)
        df.to_csv(path+'list_of_moments.csv',index=False)
        
    def compute_STFLOW(self,var,p):
        self.STFLOW = (var.X_M+var.X_CL+var.X_CD)/var.Z.sum()
        
    def compute_SPFLOW(self,var,p):
        numerator = np.einsum('nis,is,is->nis',
                              var.psi_star[...,1:]**(-p.k),
                              p.eta[...,1:],
                              var.l_R[...,1:]**(1-p.k)
                              )
        numerator = remove_diag(numerator)
        self.SPFLOW = numerator/numerator.sum()
        
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
        self.SRDUS = var.X_M[:,0,1].sum()/var.X[:,0,1].sum()
    
    def compute_GPDIFF(self,var,p):
        price_index_growth_rate = var.g_s/(1-p.sigma)+p.alpha*var.g
        self.GPDIFF = price_index_growth_rate[0] - price_index_growth_rate[1]
        
    def compute_GROWTH(self,var,p):
        self.GROWTH = var.g    
    
    def compute_Z(self,var,p):
        self.Z = var.Z
        
    def compute_moments(self,var,p):
        self.compute_STFLOW(var, p)
        self.compute_SPFLOW(var, p)
        self.compute_OUT(var, p)
        self.compute_SRGDP(var, p)
        self.compute_RP(var, p)
        self.compute_RD(var, p)
        self.compute_KM(var, p)
        self.compute_SRDUS(var, p)
        self.compute_GPDIFF(var, p)
        self.compute_GROWTH(var, p)
        
    def compute_moments_deviations(self):
        for mom in self.list_of_moments:
            setattr(self,
                    mom+'_deviation',
                    self.weights_dict[mom]*((getattr(self,mom) - getattr(self,mom+'_target'))
                            /(getattr(self,mom+'_target')*np.sqrt(getattr(self,mom+'_target').size)))
                    )
            
    def deviation_vector(self,list_of_moments = None):
        if list_of_moments is None:
            list_of_moments = self.list_of_moments
            
        dev = np.concatenate([getattr(self,mom+'_deviation').ravel() for mom in self.list_of_moments])
        return dev
    
    def target_vector(self):
        dev = np.concatenate([getattr(self,mom+'_target').ravel() for mom in self.list_of_moments])
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
        for k,v in self.dict.items():
            if k != 'objective':
                ax.plot(np.linspace(0,self.count,len(v)),v,label=k)
            else:
                ax2.plot(np.linspace(0,self.time/60,len(v)),v,label=k,color='k',lw=2)
        ax.set_xlabel('Number of succesful steady state solving')
        ax.set_ylabel('L2 norm of the deviation from target (m/target - 1)/sqrt(dim(m))')
        ax2.set_xlabel('Time (min)')
        plt.yscale('log')
        ax.legend()
        ax2.legend(loc=(0.85,1.05))
        plt.show()    
        

def get_vec_qty(x,p):
    res = {'price_indices':x[0:p.N],
           'w':x[p.N:p.N*2],
           'Z':x[p.N*2:p.N*3],
           'l_R':x[p.N*3:p.N*3+p.N*(p.S-1)],
           'psi_star':x[p.N*3+p.N*(p.S-1):]
           }
    return res

def iter_once(x,p, normalize = False):
    init = var.var_from_vector(x,p)
    if normalize:
        init.num_scale_solution(p)
    price = init.compute_price_indices(p)
    w = init.compute_wage(p)
    Z = init.compute_expenditure(p)
    l_R = init.compute_labor_research(p)[...,1:].ravel()
    psi_star = init.compute_psi_star(p)[...,1:].ravel()
    vec = np.concatenate((price,w,Z,l_R,psi_star), axis=0)
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
        x_new = x_old*1/2+x_new*1/2
        high_jumps_too_big = x_new > 1000*x_old
    low_jumps_too_big = x_new < x_old/1000
    while np.any(low_jumps_too_big):
        x_new = x_old*1/2+x_new*1/2
        low_jumps_too_big = x_new < x_old/1000
    return x_new

def write_calibration_results(path,p,m,commentary = None):
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
    
    df2 = pd.DataFrame(index = m.list_of_moments, columns = ['weight','norm of deviation', 'description'])
    for mom in m.get_list_of_moments():
        df2.loc[mom] = [m.weights_dict[mom],
                        np.linalg.norm(getattr(m,mom+'_deviation')),
                        m.description.loc[mom].description]
    df2.name = 'targeted moments'
    worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
    df2.to_excel(writer,sheet_name='Summary',startrow=df1.shape[0] + 5 , startcol=0)
    
    worksheet.write_string(df1.shape[0] + df1.shape[0] + 2, 0, commentary)
    
    
    scalar_moments = pd.DataFrame(columns=['model','target'])
    for mom in m.get_list_of_moments():
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
    writer.save()


def fixed_point_solver(p, x0=None, tol = 1e-10, damping = 5, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, cobweb_anim=False, cobweb_qty='psi_star',
                       cobweb_coord = 1, plot_convergence = True, apply_bound_zero = True, 
                       apply_bound_psi_star = False, apply_bound_research_labor = False,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True):   
    if x0 is None:
        x0 = p.guess_from_params()
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
    cob = cobweb(cobweb_qty)
    damping = 10
    while condition and count < max_count:
        # if count%1000 ==0:
        #     print(count)
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
        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        # if np.any(x_new<0):
        #     print(count, 'negative')
        # if count == 1e3:
        #     x_old[p.N:p.N*2] = 2*x_old[p.N:p.N*2]
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['price_indices','w','Z','psi_star','l_R']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(
            x_new - x_old)/np.linalg.norm(x_new))
        # print(convergence[-1])
        count += 1
        if np.all(np.array(convergence[-10:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
            damping = 2
        history_old.append(get_vec_qty(x_old,p)[cobweb_qty].min())
        history_new.append(get_vec_qty(x_new,p)[cobweb_qty].min())
        # history_old.append(get_vec_qty(x_old,p)[cobweb_qty][1,...])
        # history_new.append(get_vec_qty(x_new,p)[cobweb_qty][1,...])
        # if count > 1e4 and count%1000==0:
        #     damping = 3
        #     # plt.semilogy(convergence)
        #     plt.plot(history_old)
        #     plt.title(count)
        #     plt.show()
            # x_new = x_new*(0.5+np.random.rand(x_new.size))**10
        # if count > 5 and count%2==0:
        #     cob.append_old_new(history_old[-1],history_new[-1])
        #     pause = 0.1
        #     # if count == 20:
        #     #     pause = 1
        #     cob.plot(count=count, window = None,pause = pause)
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = deviation_norm(x_new,p)
    if count < max_count and np.isnan(x_new).sum()==0:
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
    # temp_for_scaling = var.var_from_vector(x_new, p)
    # temp_for_scaling.num_scale_solution(p)     
    # x_sol = temp_for_scaling.vector_from_var()
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = None,pause = 0.05) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence)
        plt.show()
    return sol_inst

#%% fixed point solver
    
# p = parameters(n=7,s=2)
# calib_parameters = parameters.get_list_of_params()
# p.calib_parameters = ['sigma','theta','nu','k','nu_tilde','eta',
#                     'delta','fe','tau','T','fo','g_0']

# p.load_data('calibration_results/15/',['eta','delta','fe','tau','T','fo','g_0'])
# p.delta[4,1] = 50*7.93126207e-04
# p.delta[:,1] = 0.1
sol = fixed_point_solver(p,#x0=p.guess,
                        cobweb_anim=False,tol =1e-13,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='psi_star',
                        plot_convergence=True,
                        plot_cobweb=True,
                        safe_convergence=0.001
                         # apply_bound_psi_star=True
                        )

sol_c = var.var_from_vector(sol.x, p)     
sol_c.num_scale_solution(p)   
sol_c.compute_non_solver_quantities(p)
# list_of_moments = ['GPDIFF','GROWTH','KM','OUT','RD',
#                    'RP','SPFLOW','SRGDP','STFLOW']
# m = moments(list_of_moments)
# m.load_data()
# m.compute_moments(sol_c,p)
# m.compute_Z(sol_c,p)
# m.compute_moments_deviations()
# m.plot_moments(m.get_list_of_moments())
p.guess = sol_c.vector_from_var()

#%% calibration

def calibration_func(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    p.update_parameters(vec_parameters)
    try:
        v0 = p.guess
    except:
        pass
    sol = fixed_point_solver(p,v0,tol=1e-13,
                                 accelerate=False,
                                 accelerate_when_stable=True,
                                 plot_cobweb=False,
                                 plot_convergence=False,
                                 cobweb_qty='l_R',
                                 disp_summary=False,
                                 safe_convergence=0.001
                                 )
    # if sol.status == 'successful':     
    sol_c = var.var_from_vector(sol.x, sol.p)   
    sol_c.num_scale_solution(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    m.compute_Z(sol_c,p)
    m.compute_moments_deviations()
    hist.count += 1
    # print({mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments})
    if hist is not None:
        if hist.count%10 == 0:
            hist_dic = {mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments}
            hist_dic['objective'] = np.linalg.norm(m.deviation_vector())
            hist.append(**hist_dic)
            hist.time = time.perf_counter() - start_time
        if hist.count%1000 == 0:
            hist.plot()
    p.guess = sol_c.vector_from_var()
    # print(p.fe[1],p.fo[1])
    return m.deviation_vector() 
    
# p = parameters(n=7,s=2)
# p.calib_parameters = ['eta','delta','fe','tau','T','fo','g_0','nu','nu_tilde']
# p.load_data('calibration_results/16/',list_of_params=p.calib_parameters)
# p.guess = None

list_of_moments = ['GPDIFF','GROWTH','KM','SRDUS','OUT','RD','RP','SPFLOW','SRGDP','STFLOW']
m = moments(list_of_moments)
# hist = history(*tuple(m.list_of_moments+['objective']))
m.load_data()
bounds = p.make_parameters_bounds()
# start_time = time.perf_counter()

# hist.append({mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments})
# p.guess=None    

test_ls = optimize.least_squares(fun = calibration_func,    
                    x0 = p.make_p_vector(), 
                    args = (p,m,p.guess,hist,start_time), 
                    bounds = bounds,
                    # method= 'trf',
                    # loss='arctan',
                    max_nfev=1e8,
                    ftol=1e-14, 
                    xtol=1e-15, 
                    gtol=1e-14,
                    # f_scale=scale,
                    verbose = 2)
finish_time = time.perf_counter()
print('minimizing time',finish_time-start_time)

plt.plot(test_ls.x)
plt.show()

#%%

commentary = 'Same situation as 3, but adding the SRDUS moment'
write_calibration_results('/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results/3',p,m,
                          commentary = commentary)

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
