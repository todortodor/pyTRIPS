#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:27:06 2022

@author: simonl
"""
from os.path import dirname, join
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma, gammaincc, gammainc#, beta, betainc, hyp2f1
from scipy.optimize import root
import scipy.integrate as integrate
import time
import os
import seaborn as sns
import warnings
from mpmath import betainc
# warnings.simplefilter('ignore', np.RankWarning)

class parameters:     
    def __init__(self):   
        co = 1e-6
        cou = 1e5
        self.lb_dict = {'sigma':1.5,
                        'theta':3,
                        'rho':0,
                        'gamma':co,
                        'zeta':0,
                        'nu':0,
                        'nu_tilde':0,
                        'kappa':co,
                        'k':1+co,
                        'a':co,
                        'fe':co,
                        'fo':co,
                        'delta':co,
                        # 'delta':0.05,
                        'delta_dom':co,
                        'delta_int':co,
                        # 'delta_dom':0.05,
                        # 'delta_int':0.05,
                        'g_0':0,
                        'alpha':co,
                         'beta':co,
                         'T':co,
                         'eta':co,
                         'khi':0,
                         'r_hjort':co,
                         'd':co}
        self.ub_dict = {'sigma':5,
                        'theta':12,
                        'rho':0.5,
                        'gamma':cou,
                        'zeta':1,
                        'nu':100,
                        'nu_tilde':cou,
                        'kappa':1-co,
                        'k':2,
                        'a':10,
                        'fe':cou,
                        'fo':cou,
                        'delta':10,
                        'delta_dom':10,
                        'delta_int':10,
                        'g_0':cou,
                        'alpha':1,
                         'beta':1,
                         'T':np.inf,
                         'eta':cou,
                         'khi':1,
                         'r_hjort':cou,
                         'd':10}
        
        self.calib_parameters = None
        self.guess = None
        self.dyn_guess = None
        
        self.correct_eur_patent_cost = True
        self.fix_fe_across_sectors = False
        
        self.g_0 = 0.01
        self.kappa = 0.5
        self.gamma = 0.5 
        self.k = np.array([1.3,1.3])
        self.a = np.float64(0.0)
        self.rho = 0.02
        self.d = np.float64(1.0)
        self.data_path = None
        self.unit = 1e6
    
    def load_data(self,data_path=None,keep_already_calib_params=False,dir_path=None,nbr_sectors=2):
        if dir_path is None:
            dir_path = './'
        if data_path is None:
            data_path = 'data/data_leg/'
        
        self.data_path = data_path
        
        data_path = dir_path+data_path
        
        self.data = pd.read_csv(data_path+'country_moments.csv',index_col=[0])
        N = len(self.data.index)
        self.N = N
        
        if nbr_sectors == 2:
            self.sectors = ['Non patent', 'Patent']
        if nbr_sectors == 3:
            self.sectors = ['Non patent', 'Patent', 'Pharma Chemicals']
        if nbr_sectors == 4:
            self.sectors = ['Non patent', 'Patent', 'Pharmaceuticals', 'Chemicals']
        S = len(self.sectors)
        self.S = S
        
        if N==7:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW']
        if N==13:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'AUS', 'MEX', 'IDN', 'ROW']
        if N==12:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ZAF', 'ROW']
        if N==11:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ROW']
            
        self.data_sectors = pd.read_csv(data_path+'sector_moments.csv',index_col=[0])
        self.alpha = self.data_sectors['alpha'].values
        # self.beta = self.data_sectors['beta'].values
        
        self.trade_flows = pd.read_csv(data_path+'country_country_sector_moments.csv',index_col=[1,0,2]).sort_index().values.squeeze()/self.unit
        self.trade_flows = self.trade_flows.reshape((N,N,S))
        self.trade_shares = self.trade_flows/self.trade_flows.sum()
        try:
            self.tariff = pd.read_csv(data_path+'tariff.csv',index_col=[1,0,2]).sort_index().values.squeeze().reshape((N,N,S))
        except:
            self.tariff = np.zeros_like(self.trade_flows)
        self.beta = np.einsum('nis->s',self.trade_shares)
        self.deficit_raw = self.data.deficit.values
        self.deficit_raw[0] = self.deficit_raw[0]-self.deficit_raw.sum()
        self.deficit_share_world_output = self.deficit_raw/self.data.output.sum()
        self.unit_labor = 1e9
        self.unit = 1e6
        self.khi = 0.16
        self.labor_raw = self.data.labor.values
        self.labor = self.labor_raw/self.unit_labor
        self.r_hjort = ((self.data.gdp.iloc[0]*np.array(self.data.labor)*self.data.price_level
                        /(self.data.labor.iloc[0]*self.data.price_level.iloc[0]*np.array(self.data.gdp))
                        )**(1-self.khi)).values
        
        if self.correct_eur_patent_cost:
            # self.r_hjort[1] = self.r_hjort[1]*pd.read_csv(
            #     data_path+'final_pat_fees.csv',index_col=0).loc[2,'fee']/pd.read_csv(
            #         data_path+'final_pat_fees.csv',index_col=0).loc[1,'fee']
            # self.r_hjort[1] = self.r_hjort[1]*3.6/1.8
            # self.r_hjort[1] = self.r_hjort[1]*3.6
            self.r_hjort[1] = self.r_hjort[1]*3.872572
            # self.r_hjort[1] = self.r_hjort[1]*43730.23/0.71388/pd.read_csv(
            #         data_path+'final_pat_fees.csv',index_col=0).loc[1,'fee']
        
        if not keep_already_calib_params:
            self.eta = np.ones((N, S))*0.02
            self.eta[:, 0] = 0
            self.sigma = np.ones(S)*2.9
            self.theta = np.ones(S)*5
            self.zeta = np.ones(S)*0.01
            self.T = np.ones((N, S))*1.5
            self.fe = np.ones(S)
            self.fo = np.ones(S)
            self.delta = np.ones((N, S))*0.05
            self.delta_dom = np.ones_like(self.delta)*0.05
            self.delta_int = np.ones_like(self.delta)*0.05
            self.delta_eff = np.where(
                                    np.eye(self.delta_dom.shape[0], dtype=bool)[:, :, None],
                                    self.delta_dom[:, None, :],
                                    self.delta_int[:, None, :]
                                )
            self.nu = np.ones(S)*0.1 #
            self.nu_tilde = np.ones(S)*0.1
        
        self.tau = np.full(N*N*S, np.nan).reshape((N,N,S))
        
        
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
                    # 'k':pd.Index(['scalar']),
                    'k':pd.Index(self.sectors, name='sector'),
                    'a':pd.Index(['scalar']),
                    'tau':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                      , names=['destination','origin','sector']),
                    'tariff':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                      , names=['destination','origin','sector']),
                    'fe':pd.Index(self.sectors, name='sector'),
                    'r_hjort':pd.Index(self.countries, name='country'),
                    'fo':pd.Index(self.sectors, name='sector'),
                    'delta':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                       , names=['country','sector']),
                    'delta_dom':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                       , names=['country','sector']),
                    'delta_int':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                       , names=['country','sector']),
                    'g_0':pd.Index(['scalar']),
                    'alpha':pd.Index(self.sectors, name='sector'),
                    'beta':pd.Index(self.sectors, name='sector'),
                    'T':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                        , names=['country','sector']),
                     'eta':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                      , names=['country','sector'])}
        
        sl_non_calib = {
                    # 'sigma':[np.s_[0],np.s_[1]],
                    'sigma':[np.s_[0]],
                    'theta':[np.s_[0]],
                    'rho':None,
                    'gamma':None,
                    'zeta':[np.s_[0]],
                    'nu':[np.s_[0]],
                    'nu_tilde':[np.s_[0]],
                    'kappa':None,
                    'k':None,
                    'a':None,
                    'fe':[np.s_[0]],
                    'fo':[np.s_[0]],
                    'delta':[np.s_[::S]],#,np.s_[S-1]],
                    'delta_dom':[np.s_[::S]],#,np.s_[S-1]],
                    'delta_int':[np.s_[::S]],#,np.s_[S-1]],
                    'g_0':None,
                    'd':None,
                    'khi':None,
                    'alpha':None,
                    'beta':None,
                     'T':None,
                     'r_hjort':None,
                     'eta':[np.s_[::S]]}
        
        # if nbr_sectors == 4:
        #     sl_non_calib['delta'] = [np.s_[::S],np.s_[2::S],np.s_[3::S]]
        
        if self.fix_fe_across_sectors:
            sl_non_calib['fe'] = [np.s_[0],np.s_[2:]]
        
        self.mask = {}
        
        for par_name in ['eta','k','rho','alpha','fe','T','fo','sigma','theta','beta','zeta',
                         'g_0','kappa','gamma','delta','delta_dom','delta_int','nu','nu_tilde','d','khi',
                         'r_hjort','a']:
            par = getattr(self,par_name)
            if sl_non_calib[par_name] is not None:
                self.mask[par_name] = np.ones_like(par,bool).ravel()
                for slnc in sl_non_calib[par_name]:
                    self.mask[par_name][slnc] = False
                self.mask[par_name] = self.mask[par_name].reshape(par.shape)    
            else:
                self.mask[par_name] = np.ones_like(par,bool)
        
    
    def load_run(self,path,list_of_params = None,dir_path=None):
        if dir_path is None:
            dir_path = './'
        try:
            df = pd.read_csv(path+'calib_parameters.csv',header=None)
            setattr(self,'calib_parameters',df[0].to_list())
        except:
            pass
            
        try:
            df = pd.read_csv(path+'guess.csv',header=None)
            setattr(self,'guess',df.values.squeeze())
        except:
            pass
        
        try:
            df = pd.read_csv(path+'dyn_guess.csv',header=None)
            setattr(self,'dyn_guess',df.values.squeeze())
        except:
            pass
        
        try:
            # df = pd.read_csv(path+'data_path.csv',header=None)
            df = pd.read_csv(path+'data_path.csv',index_col=0)
            # print(df)
            setattr(self,'N',df.loc['nbr_of_countries','run'])
            setattr(self,'S',df.loc['nbr_of_sectors','run'])
            setattr(self,'data_path',df.loc['data_path','run'])
        except:
            setattr(self,'N',7)
            setattr(self,'S',2)
            setattr(self,'data_path','data/data_leg/')
        
        
        self.load_data(self.data_path,dir_path=dir_path,nbr_sectors=int(df.loc['nbr_of_sectors','run']))

        if int(df.loc['nbr_of_sectors','run']) == 2:
            self.k = np.array([1.0])

        if list_of_params is None:
            list_of_params = self.get_list_of_params()
        for pa_name in list_of_params:
            # if pa_name == 'k':
            #     df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
            #     print(pa_name,df.values.squeeze())
            #     print(np.array(getattr(self,pa_name)).shape)
            try:
                df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
                if pa_name != 'k':
                    setattr(self,pa_name,df.values.squeeze().reshape(np.array(getattr(self,pa_name)).shape))
                else:
                    setattr(self,pa_name,df.values.squeeze())
            except:
                # if pa_name == 'd':
                #     self.d = np.array(1.0)
                if pa_name == 'tariff':
                    self.tariff = np.zeros_like(self.trade_flows)
                else:
                    pass   
                if pa_name == 'delta_dom':
                    self.delta_dom = self.delta.copy()
                else:
                    pass
                if pa_name == 'delta_int':
                    self.delta_int = self.delta.copy()
                else:
                    pass
                
                self.delta_eff = np.where(
                                        np.eye(self.delta_dom.shape[0], dtype=bool)[:, :, None],
                                        self.delta_dom[:, None, :],
                                        self.delta_int[:, None, :]
                                    )

        
        if self.k.shape == ():
            self.k = np.repeat(self.k,self.S)
        
        self.update_delta_eff()            
        
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
         'kappa','gamma','delta','delta_dom','delta_int','nu','nu_tilde','d','khi','r_hjort',
         'tau','tariff','a']
            
    def guess_from_params(self,for_solver_with_entry_costs=False):
        Z_guess = self.data.expenditure.values/self.unit
        w_guess = self.data.gdp.values*self.unit_labor/(self.data.labor.values*self.unit)/1e6
        l_R_guess = np.repeat(self.labor[:,None]/200, self.S-1, axis=1).ravel()
        profit_guess = np.ones((self.N,self.N,(self.S-1))).ravel()*0.001
        phi_guess = np.ones((self.N,self.N,self.S)).ravel()*0.3
        vec = np.concatenate((w_guess,Z_guess,l_R_guess,profit_guess,phi_guess), axis=0)
        if for_solver_with_entry_costs:
            price_indices_guess = np.ones(self.N)
            vec = np.concatenate(
                (w_guess,Z_guess,l_R_guess,profit_guess,phi_guess,price_indices_guess)
                , axis=0)
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
        self.delta_eff = np.where(
                                np.eye(self.delta_dom.shape[0], dtype=bool)[:, :, None],
                                self.delta_dom[:, None, :],
                                self.delta_int[:, None, :]
                            )
        
    def update_delta_eff(self):
        self.delta_eff = np.where(
                                np.eye(self.delta_dom.shape[0], dtype=bool)[:, :, None],
                                self.delta_dom[:, None, :],
                                self.delta_int[:, None, :]
                            )
            
    def update_sigma_with_SRDUS_target(self,m):
        self.sigma[1] = 1+m.SRDUS_target/(m.sales_mark_up_US_target - 1)
        
    def update_khi_and_r_hjort(self, new_khi):
        #new_khi = 1 will remove the hjort factor
        self.khi = new_khi
        self.r_hjort = ((self.data.gdp.iloc[0]*np.array(self.data.labor)*self.data.price_level
                        /(self.data.labor.iloc[0]*self.data.price_level.iloc[0]*np.array(self.data.gdp))
                        )**(1-self.khi)).values
            
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
        try:
            df = pd.DataFrame(data = self.dyn_guess)
            df.to_csv(path+'dyn_guess.csv',index=False,header=None)
        except:
            pass
        if self.calib_parameters is not None:
            df = pd.DataFrame(data = self.calib_parameters)
            df.to_csv(path+'calib_parameters.csv',index=False,header=None)
            
        df = pd.DataFrame(index=['data_path','nbr_of_countries',
                                 'nbr_of_sectors'])
        df['run'] = [self.data_path,self.N,self.S]
        df.to_csv(path+'data_path.csv')
            
    def make_parameters_bounds(self):
        lb = []
        ub = []
        for par in self.calib_parameters:
            lb.append(np.ones(np.array(getattr(self,par))[self.mask[par]].size)*self.lb_dict[par])
            ub.append(np.ones(np.array(getattr(self,par))[self.mask[par]].size)*self.ub_dict[par])
        return (np.concatenate(lb),np.concatenate(ub))
    
    def make_one_country_parameters(self,country):
        country_i = self.countries.index(country)
        one_country_p = self.copy()
        one_country_p.N = 1
        one_country_p.eta = one_country_p.eta[country_i:country_i+1,:]
        one_country_p.T = one_country_p.T[country_i:country_i+1,:]
        one_country_p.delta = one_country_p.delta[country_i:country_i+1,:]
        one_country_p.tau = one_country_p.tau[country_i:country_i+1,country_i:country_i+1,:]
        one_country_p.tariff = one_country_p.tariff[country_i:country_i+1,country_i:country_i+1,:]
        one_country_p.trade_flows = one_country_p.trade_flows[country_i:country_i+1,country_i:country_i+1,:]
        one_country_p.trade_shares = one_country_p.trade_flows/one_country_p.trade_flows.sum()
        one_country_p.r_hjort = one_country_p.r_hjort[country_i:country_i+1]
        one_country_p.countries = one_country_p.countries[country_i:country_i+1]
        one_country_p.labor = one_country_p.labor[country_i:country_i+1]
        one_country_p.labor_raw = one_country_p.labor_raw[country_i:country_i+1]
        one_country_p.data = one_country_p.data.iloc[country_i:country_i+1]
        one_country_p.deficit_raw = 0
        one_country_p.deficit_share_world_output = 0
        one_country_p.guess = None
        one_country_p.dyn_guess = None
        return one_country_p
        
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

class var_with_entry_costs:
    def __init__(self, context, N = 7, S = 2):
        self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        self.off_diag_mask[np.s_[::(N+1)*S]] = False
        self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        self.diag_mask = np.invert(self.off_diag_mask)
        self.context = context

    def guess_profit(self, profit_init):
        self.profit = profit_init    

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_Z(self, Z_init):
        self.Z = Z_init

    def guess_labor_research(self, l_R_init):
        self.l_R = l_R_init
    
    def guess_phi(self, phi_init):
        self.phi = phi_init
        
    def guess_price_indices(self, price_indices_init):
        self.price_indices = price_indices_init

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        frame = deepcopy(self)
        return frame
    
    @staticmethod
    def var_from_vector(vec,p,context,compute = True):
        init = var_with_entry_costs(context=context)    
        init.guess_wage(vec[0:p.N])
        init.guess_Z(vec[p.N:p.N+p.N])
        init.guess_labor_research(
            np.insert(vec[p.N+p.N:p.N+p.N+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1))
        init.guess_profit(
            np.insert(vec[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2].reshape((p.N, p.N, p.S-1)), 0, np.zeros(p.N), axis=2))
        init.guess_phi(vec[p.N+p.N+p.N*(p.S-1)+p.N**2:p.N+p.N+p.N*(p.S-1)+p.N**2+p.N**2*p.S].reshape((p.N, p.N, p.S)))
        init.guess_price_indices(vec[p.N+p.N+p.N*(p.S-1)+p.N**2+p.N**2*p.S:])
        if compute:
            init.compute_solver_quantities(p)
        return init
    
    @staticmethod
    def var_from_vector_no_price_indices(vec,p,context,compute = True):
        init = var_with_entry_costs(context=context)    
        init.guess_wage(vec[0:p.N])
        init.guess_Z(vec[p.N:p.N+p.N])
        init.guess_labor_research(
            np.insert(vec[p.N+p.N:p.N+p.N+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1))
        init.guess_profit(
            np.insert(vec[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2].reshape((p.N, p.N, p.S-1)), 0, np.zeros(p.N), axis=2))
        init.guess_phi(vec[p.N+p.N+p.N*(p.S-1)+p.N**2:p.N+p.N+p.N*(p.S-1)+p.N**2+p.N**2*p.S].reshape((p.N, p.N, p.S)))
        if compute:
            init.compute_solver_quantities(p)
        return init

    def vector_from_var(self):
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        profit = self.profit[...,1:].ravel()
        Z = self.Z
        phi = self.phi.ravel()
        price_indices = self.price_indices
        vec = np.concatenate((w,Z,l_R,profit,phi,price_indices), axis=0)
        return vec
    
    def vector_from_var_no_price_indices(self):
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        profit = self.profit[...,1:].ravel()
        Z = self.Z
        phi = self.phi.ravel()
        vec = np.concatenate((w,Z,l_R,profit,phi), axis=0)
        return vec
    
    @staticmethod
    def hypergeometric_integral(lb, ub, alpha, beta, y, z):
        
        # self.hypergeometric_integral(
        #                             lb = self.psi_m_star[...,1],
        #                             ub = self.psi_MP_star[...,1],
        #                             alpha = self.V_P[...,1],
        #                             # b = self.w[:,None]*p.fe[1]*p.r_hjort[:,None],
        #                             beta = self.w*p.fe[1]*p.r_hjort,
        #                             y = p.k,
        #                             z = p.d)
        
        # def integrand(psi, alpha, beta, z, y):
        #     # print((alpha * psi - beta[:, None]).shape)
        #     res = (alpha * psi - beta) ** z * (psi) ** (-y)
        #     return res
        
        # def integrate_func():
        #     res = np.zeros_like(lb)
        #     for i in range(lb.shape[0]):
        #         for j in range(lb.shape[0]):
        #             # print(lb[i,j], ub[i,j])
        #             res[i,j] = integrate.quad(integrand, 
        #                                           lb[i,j], 
        #                                           ub[i,j],
        #                                           args=(alpha[i,j],beta[i],z,y)
        #                                           )[0]
        #     return res
        
        # # Perform the integration
        # integral_0 = integrate_func()
        
        # Calculate t_ub and t_lb
        t_ub = 1 - beta[:, None] / ub / alpha
        t_lb = 1 - beta[:, None] / lb / alpha
        
        # print(alpha)
        # print(beta)
        
        # Calculate the second integral using the betainc function
        integral = (beta[:, None] ** (1 - y + z) / alpha ** (1 - y)
                    ) * np.vectorize(
                        lambda a, b, x1, x2: float(betainc(a, b, x1, x2, regularized=False))
                        )(z + 1, y - z - 1, t_lb, t_ub)
        
        # print((integral - integral_0)[integral>0])
        
        return integral

    
    def compute_growth(self, p):
        self.g_s = p.k*np.einsum('is,is -> s',
                                 p.eta,
                                 self.l_R**(1-p.kappa)
                                 )/(p.k-1) - p.zeta
        self.g_s[0] = p.g_0
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()
        self.r = p.rho + self.g/p.gamma
        self.G = self.r+p.zeta-self.g+self.g_s+p.nu
        
    def compute_entry_costs(self,p):
        if self.context == 'calibration':
            # self.a = p.a * np.maximum(
            #     np.einsum('is,nis,nis,is,is->nis',
            #                     p.T**(1/p.theta[None,:]),
            #                     1/self.phi,
            #                     1/(1+p.tariff),
            #                     self.w[:,None]**-p.alpha[None,:],
            #                     self.price_indices[:,None]**(p.alpha[None,:]-1),
            #                     ) - 1,
            #     0
            #     )
            self.a = p.a * np.einsum('is,nis,nis,is,is->nis',
                                p.T**(1/p.theta[None,:]),
                                1/self.phi,
                                1/(1+p.tariff),
                                self.w[:,None]**-p.alpha[None,:],
                                self.price_indices[:,None]**(p.alpha[None,:]-1),
                                )
            np.einsum('nns->ns',self.a)[:] = 0
        
        elif self.context == 'counterfactual':
            # self.a = p.a * np.maximum(p.tau-1,0)
            self.a = p.a * p.tau
            np.einsum('nns->ns',self.a)[:] = 0
        
    def compute_V(self,p):
        self.V_NP = np.einsum('nis,i,s->nis',
                              self.profit,
                              self.w,
                              1/self.G
                              )
        
        self.V_P = np.einsum('nis,i,ns->nis',
                             self.profit,
                             self.w,
                             1/(self.G[None,:]-p.nu[None,:]+p.delta)-1/(self.G[None,:]+p.delta)+1/(self.G[None,:])
                             )
        
        # assert np.all(self.V_P[...,1:] >= self.V_NP[...,1:])
        
    def compute_patenting_thresholds(self, p):
        self.psi_star = np.full((p.N,p.N,p.S),np.inf)
        self.psi_star[...,1:] = np.maximum(
            np.einsum('n,s,n,nis->nis',
                      self.w,
                      p.fe[1:],
                      p.r_hjort,
                      1/(self.V_P[...,1:] - self.V_NP[...,1:])
                      ),
            1
            )
        
        self.a_NP_star = np.ones((p.N,p.N,p.S))
        self.a_NP_star[...,1:] = np.maximum(
            np.einsum('i,nis,nis->nis',
                      self.w,
                      self.a[...,1:],
                      1/self.V_NP[...,1:]
                      ),
            1
            )
        
        self.a_P_star = np.ones((p.N,p.N,p.S))
        self.a_P_star[...,1:] = np.maximum(
            np.einsum('i,nis,nis->nis',
                      self.w,
                      self.a[...,1:],
                      1/(self.V_P[...,1:]-np.einsum('n,s,n->ns',
                                                    self.w,
                                                    p.fe[1:],
                                                    p.r_hjort)[:,None,:])
                      ),
            1
            )
        
        self.psi_o_star = np.full((p.N,p.S),np.inf)
        
        def aleph_P_star(psi_o_star):
            res = np.maximum(
                            np.einsum('i,nis,nis->nis',
                                      self.w,
                                      self.a[...,1:],
                                      1/(psi_o_star[None,...]*self.V_P[...,1:]-np.einsum('n,s,n->ns',
                                                                    self.w,
                                                                    p.fe[1:],
                                                                    p.r_hjort)[:,None,:])
                                      ),
                            1
                            )
            return res
        
        def aleph_NP_star(psi_o_star):
            res = np.maximum(
                            np.einsum('i,nis,is,nis->nis',
                                      self.w,
                                      self.a[...,1:],
                                      1/psi_o_star,
                                      1/self.V_NP[...,1:]
                                      ),
                            1
                            )
            return res
        
        # this will need to be updated if more than one patenting sector
        def func_to_solve(psi_o_star):
            psi_o_star = psi_o_star[:,None]
            signature = psi_o_star[None,...] >= self.psi_star[...,1:]
            
            A = np.einsum('nis,nis->nis',
                          np.einsum('is,nis->nis',
                                    psi_o_star,
                                    self.V_P[...,1:],
                                    ) - np.einsum('n,s,n->ns',
                                                self.w,
                                                p.fe[1:],
                                                p.r_hjort)[:,None,:] ,
                          aleph_P_star(psi_o_star)**(-p.d)
                )
                                                  
            B = np.einsum('i,nis,,nis->nis',
                          self.w,
                          self.a[...,1:],
                          p.d/(p.d+1),
                          aleph_P_star(psi_o_star)**(-p.d-1)
                          )
            
            C = np.einsum('nis,nis->nis',
                          np.einsum('is,nis->nis',
                                    psi_o_star,
                                    self.V_NP[...,1:],
                                    ),
                          aleph_NP_star(psi_o_star)**(-p.d)
                )
            
            D = np.einsum('i,nis,,nis->nis',
                          self.w,
                          self.a[...,1:],
                          p.d/(p.d+1),
                          aleph_NP_star(psi_o_star)**(-p.d-1)
                          )
            
            to_sum = A - B - (C - D)
            
            res = (signature * to_sum).sum(axis=0) - self.w[:,None]*p.fo[None,1:]*p.r_hjort[:,None]
            
            return res.ravel()/psi_o_star.ravel()
            # print(res)
            # return res.ravel()
        
        x0 = np.min(self.psi_star[...,1],axis=0)
        roots = root(func_to_solve,x0=x0,tol=1e-15)
        
        self.psi_o_star[:,1] = roots.x
        # print(roots.x)
        
        # check where the equality condition is satisfied and will replace in psi_o_star
        signature = np.isclose(self.psi_star[...,1:],1)
        
        A = np.einsum('nis,nis->nis',
                      self.V_P[...,1:] - np.einsum('n,s,n->ns',
                                            self.w,
                                            p.fe[1:],
                                            p.r_hjort)[:,None,:],
                      self.a_P_star[...,1:]**-p.d
                      )
        
        B = np.einsum('i,nis,,nis->nis',
                      self.w,
                      self.a[...,1:],
                      p.d/(p.d+1),
                      self.a_P_star[...,1:]**(-p.d-1)
                      )
        
        C = np.einsum('nis,nis->nis',
                      self.V_NP[...,1:],
                      self.a_NP_star[...,1:]**-p.d
                      )
        
        D = np.einsum('i,nis,,nis->nis',
                      self.w,
                      self.a[...,1:],
                      p.d/(p.d+1),
                      self.a_NP_star[...,1:]**(-p.d-1)
                      )
        
        to_sum = A - B - (C - D)
        res = (signature * to_sum).sum(axis=0) - self.w[:,None]*p.fo[None,1:]*p.r_hjort[:,None]
        
        self.psi_o_star[...,1:][res>0] = 1
        
        self.psi_m_star = np.maximum(self.psi_star,self.psi_o_star[None,:,:])
        
        # careful not to confuse the following with self.PSI_M.
        # small psi are patenting thresholds, the two quantities are not linked
        self.psi_MP_star = np.full((p.N,p.N,p.S),np.inf)
        self.psi_MP_star[...,1:] = np.maximum(
            self.psi_m_star[...,1:],
            (self.w[None,:,None]*self.a[...,1:]
             +self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None]
             ) / self.V_P[...,1:]
            )
        
        self.psi_MNP_star = np.full((p.N,p.N,p.S),np.inf)
        self.psi_MNP_star[...,1:] = np.maximum(
            self.psi_m_star[...,1:],
            (self.w[None,:,None]*self.a[...,1:]) / self.V_NP[...,1:]
            )
        
        # x = np.linspace(1,100,100)
        # l_y = [func_to_solve( np.ones((p.N,p.S))[...,1]*x_i ) for x_i in x]
        # print(x)
        # y = np.array([[l[i] for l in l_y] for i,c in enumerate(p.countries)]).T
        # print(y)
        # fig = plt.figure(dpi=188)
        # plt.axhline(0,color='grey',label='Zero')
        # plt.plot(x,y,label=p.countries)
        # plt.scatter(y=np.zeros_like(roots.x),x=roots.x,color='red',marker='+',zorder=100,label='roots')
        # # if thresholds_to_compare is not None:
        # #     plt.scatter(y=np.zeros_like(roots.x),
        # #                 x=thresholds_to_compare,
        # #                 color='green',
        # #                 marker='+',
        # #                 zorder=100,
        # #                 label='Original thresholds')
        # # plt.xscale('log')
        # # plt.yscale('symlog')
        # plt.legend(loc=(1.01,0))
        # plt.show()
        
    def compute_mass_innovations(self,p):
        # this would have to be updated for additional sectors
        
        # print(self.psi_m_star[...,1].max())
        # print(self.psi_MP_star[...,1].max())
        
        # second way
        integral_k_d = self.hypergeometric_integral(
                                    lb = self.psi_m_star[...,1],
                                    ub = self.psi_MP_star[...,1],
                                    alpha = self.V_P[...,1],
                                    # b = self.w[:,None]*p.fe[1]*p.r_hjort[:,None],
                                    beta = self.w*p.fe[1]*p.r_hjort,
                                    y = p.k,
                                    z = p.d)
        
        # print(integral_k_d)
        
        self.integral_k_d = integral_k_d
        
        temp_w_a_power_minus_d = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )
        
        A = p.k*(
            1
            - np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(1-p.k)
            + self.psi_m_star[...,1:]**(1-p.k)
            - self.psi_MP_star[...,1:]**(1-p.k)
            )/(p.k-1)
        
        
        B = p.k*np.einsum('nis,nis->nis',
                      temp_w_a_power_minus_d/self.V_NP[...,1:]**(-p.d),
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k+1) - 1
                      )/(p.d-p.k+1)
        
        C = p.k*np.einsum('nis,ni->nis',
                          temp_w_a_power_minus_d,
                          integral_k_d
                          )
        
        self.mu_MNE = np.zeros((p.N,p.N,p.S))
        self.mu_MNE[...,1:] = A - B - C
        
        # print('mu_MNE')
        
        self.mu_MPND = np.zeros((p.N,p.N,p.S))
        self.mu_MPND[...,1:] = C + (p.k*self.psi_MP_star[...,1:]**(1-p.k))/(p.k-1)
        
        D = p.k*(
            np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(1-p.k)
            - self.psi_m_star[...,1:]**(1-p.k)
            )/(p.k-1)
        
        self.mu_MNP = np.zeros((p.N,p.N,p.S))
        self.mu_MNP[...,1:] = B + D
        
        # print(np.allclose(self.mu_MNE+self.mu_MPND+self.mu_MNP,p.k/(p.k-1)))
    
    def compute_aggregate_qualities(self,p):
        self.PSI_MNE = np.zeros((p.N,p.N,p.S))
        self.PSI_MPND = np.zeros((p.N,p.N,p.S))
        self.PSI_MPD = np.zeros((p.N,p.N,p.S))
        self.PSI_MNP = np.zeros((p.N,p.N,p.S))
        self.PSI_M = np.zeros((p.N,p.N,p.S))
        self.PSI_ME = np.zeros((p.N,p.N,p.S))
        self.PSI_CD = np.ones((p.N,p.S))
        prefact = p.eta[None,:,1:] * self.l_R[None,:,1:]**(1-p.kappa)
        A = self.g_s[1:] + p.nu[1:] + p.zeta[1:]
        self.PSI_MNE[...,1:] = np.einsum('nis,nis,s->nis',
                                  prefact,
                                  self.mu_MNE[...,1:],
                                  1/A
                                  )
        self.PSI_MPND[...,1:] = np.einsum('nis,nis,ns->nis',
                                  prefact,
                                  self.mu_MPND[...,1:],
                                  1/(A[None,:]+p.delta[:,1:])
                                  )
        self.PSI_MPD[...,1:] = np.einsum('s,nis,ns->nis',
                                 p.nu[1:],
                                 self.PSI_MPND[...,1:],
                                 1/(p.delta[...,1:]+self.g_s[None,1:]+p.zeta[None,1:]))
        numerator_A = np.einsum('nis,nis->nis',
                                prefact,
                                self.mu_MNP[...,1:])
        numerator_B= np.einsum('ns,nis->nis',
                               p.delta[...,1:],
                               self.PSI_MPND[...,1:])
        self.PSI_MNP[...,1:] = (numerator_A + numerator_B)/A[None,None,:]
        self.PSI_M[...,1:] = self.PSI_MNE[...,1:]+self.PSI_MPND[...,1:]\
                            +self.PSI_MPD[...,1:]+self.PSI_MNP[...,1:]
        self.PSI_ME[...,1:] = self.PSI_MPND[...,1:]+self.PSI_MPD[...,1:]+self.PSI_MNP[...,1:]
        self.PSI_CD[:,1:] = 1-self.PSI_M[...,1:].sum(axis=1)
        
    def compute_sectoral_prices(self, p):
        power = p.sigma-1
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, 1:] \
            * (self.PSI_ME[...,1:]*self.phi[...,1:]**power[None, None, 1:]).sum(axis=1)

        B = self.PSI_CD[...,1:]*(self.phi[...,1:]**p.theta[None,None,1:]).sum(axis=1)**(power/p.theta)[None, 1:]

        self.P_M = np.full((p.N, p.S),np.inf)
        self.P_M[:,1:] = (A/(A+B))**(1/(1-p.sigma))[None, 1:]
        
        self.P_CD = np.ones((p.N, p.S))
        self.P_CD[:,1:] = (B/(A+B))**(1/(1-p.sigma))[None, 1:]
        
    def compute_labor_allocations(self, p):
        self.l_Ao = np.zeros((p.N,p.S))
        self.l_Ao[...,1:] = np.einsum('i,s,is,is,is -> is',
                         p.r_hjort,
                         p.fo[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         self.psi_o_star[...,1:]**-p.k
                         )
        
        self.l_Ae = np.zeros((p.N,p.N,p.S))
        temp_w_a_power_minus_d = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )
        temp_w_a_power_minus_d_minus_1 = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d+1), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )

        integral_k_plus_un_d = self.hypergeometric_integral(
                                    lb = self.psi_m_star[...,1],
                                    ub = self.psi_MP_star[...,1],
                                    alpha = self.V_P[...,1],
                                    # b = self.w[:,None]*p.fe[1]*p.r_hjort[:,None],
                                    beta = self.w*p.fe[1]*p.r_hjort,
                                    y = p.k+1,
                                    z = p.d)
        self.integral_k_plus_un_d = integral_k_plus_un_d
        self.l_Ae[...,1:] = np.einsum('n,s,is,is,nis -> ins',
                         p.r_hjort,
                         p.fe[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         p.k*temp_w_a_power_minus_d*integral_k_plus_un_d[...,None]+self.psi_MP_star[...,1:]**-p.k
                         )
        
        self.l_Aa = np.zeros((p.N,p.N,p.S))
        A = p.k*np.einsum('nis,nis,nis -> nis',
                      temp_w_a_power_minus_d_minus_1,
                      1/self.V_NP[...,1:]**(-p.d-1),
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k+1) - 1
                      )/(p.d-p.k+1)
        B = self.psi_MP_star[...,1:]**(-p.k)\
            -self.psi_m_star[...,1:]**(-p.k)\
            +np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(-p.k)

        lb = self.psi_m_star[...,1]
        ub = self.psi_MP_star[...,1]
        alpha = self.V_P[...,1]
        # b = self.w[:,None]*p.fe[1]*p.r_hjort[:,None],
        beta = self.w*p.fe[1]*p.r_hjort
        y = p.k
        z = p.d
        
        # by part integration to use the previous calculation of the integral
        term_lb = lb**(-y)*(alpha*lb-beta[:,None])**(z+1)/y
        term_ub = ub**(-y)*(alpha*ub-beta[:,None])**(z+1)/y
        
        integral_k_plus_un_d_plus_un = term_lb-term_ub + alpha*(z+1)*self.integral_k_d/y
        
        self.integral_k_plus_un_d_plus_un = integral_k_plus_un_d_plus_un
        C = p.k*temp_w_a_power_minus_d_minus_1*integral_k_plus_un_d_plus_un[...,None]
        self.l_Aa[...,1:] = p.d*np.einsum('is,is,nis,nis->nis',
                                      p.eta[:,1:],
                                      self.l_R[...,1:]**(1-p.kappa),
                                      self.a[...,1:],
                                      A+B+C
                                      )/(p.d+1)
        self.l_P = p.labor-(self.l_Ao+self.l_R+self.l_Ae.sum(axis=0)+self.l_Aa.sum(axis=0)).sum(axis=1)
        
    def compute_trade_flows_and_shares(self, p, assign = True):
            temp = (self.PSI_ME[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
            X_M = np.zeros((p.N, p.N, p.S))
            X_M[...,1:] = np.einsum('nis,nis,ns,ns,s,n->nis',
                                    self.phi[..., 1:]**(p.sigma-1)[None, None, 1:],
                                    self.PSI_ME[..., 1:],
                                    1/temp,
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:]),
                                    p.beta[1:],
                                    self.Z
                                    )
            X_CD = np.einsum('nis,ns,ns,s,n->nis',
                                        self.phi**(p.theta)[None,None,:],
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:]),
                                        p.beta,
                                        self.Z
                                        )
            X = X_M+X_CD
            if assign:
                self.X_M = X_M
                self.X_CD = X_CD
                self.X = X
            else:
                return X_M,X_CD,X
            
    def compute_solver_quantities(self,p):
        self.compute_growth(p)
        # print('growth done')
        self.compute_entry_costs(p)
        # print('entry costs done')
        self.compute_V(p)
        self.compute_patenting_thresholds(p)
        # print('patenting thresholds done')
        self.compute_mass_innovations(p)
        # print('mass innovations done')
        self.compute_aggregate_qualities(p)
        self.compute_sectoral_prices(p)
        self.compute_labor_allocations(p)
        self.compute_trade_flows_and_shares(p)

    def compute_price_indices(self,p):
        power = (p.sigma-1)
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_ME * self.phi**power[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :]/(p.sigma[None, :]-1)) ).prod(axis=1)
        return price_indices
    
    def compute_labor_research(self,p):
        A_1 = p.k*np.einsum('nis,i,nis->nis',
                        self.V_NP[...,1:],
                        1/self.w,
                        self.a_NP_star[...,1:]
                        )/(p.k-1)
        A_2 = p.d*self.a[...,1:]/(p.d+1)
        A = np.einsum('nis,nis->nis',
                      A_1 - A_2,
                      self.a_NP_star[...,1:]**(-p.k)
                      )
        
        B_1 = p.k*np.einsum('nis,i,nis->nis',
                        self.V_NP[...,1:],
                        1/self.w,
                        self.psi_MNP_star[...,1:]
                        )/(p.k-1)
        B_2 = p.d*self.a[...,1:]/(p.d+1)
        B = np.einsum('nis,nis->nis',
                      B_1 - B_2,
                      self.psi_MNP_star[...,1:]**(-p.k)
                      )
            
        temp_w_a_power_minus_d = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )
        
        signature_C = np.einsum('i,nis,nis->nis',
            self.w,
            self.a[...,1:],
            1/self.V_NP[...,1:]
            )  > 1
        
        # C = p.k*np.einsum('nis,i,nis,nis,nis->nis',
        #               temp_w_a_power_minus_d,
        #               1/self.w,
        #               self.V_NP[...,1:]**(p.d+1),
        #               signature_C,
        #               np.maximum(
        #                   np.einsum('i,nis,nis->nis',
        #                   self.w,
        #                   self.a[...,1:],
        #                   1/self.V_NP[...,1:]),
        #                   0)**(p.d-p.k+1)-1,
        #               )/((p.d+1)*(p.d-p.k+1))
        
        C = p.k*np.einsum('nis,i,nis,nis,nis->nis',
                      temp_w_a_power_minus_d,
                      1/self.w,
                      self.V_NP[...,1:]**(p.d+1),
                      signature_C,
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k+1) - 1
                      )/((p.d+1)*(p.d-p.k+1))
        
        # print(C)
        
        # C[np.isnan(C)] = 0
        # C[C < 0] = 0
        
        D_1 = p.k*np.einsum('nis,i,nis->nis',
                        self.V_P[...,1:],
                        1/self.w,
                        self.psi_MP_star[...,1:]
                        )/(p.k-1)
        D_2 = self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None]/self.w[None,:,None]\
                +p.d*self.a[...,1:]/(p.d+1)
        D = np.einsum('nis,nis->nis',
                      D_1 - D_2,
                      self.psi_MP_star[...,1:]**(-p.k)
                      )
        
        signature_E = np.einsum('nis,nis->nis',
            self.w[None,:,None]*self.a[...,1:]+self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None],
            1/self.V_P[...,1:]
            )  > self.psi_o_star[None,:,1:]
        
        integral_k_plus_un_d_plus_un = self.integral_k_plus_un_d_plus_un
        
        E = p.k*np.einsum('nis,nis,i,ni->nis',
                      signature_E,
                      temp_w_a_power_minus_d,
                      1/self.w,
                      integral_k_plus_un_d_plus_un
                      )/(p.d+1)
        
        l_R = np.zeros((p.N,p.S))
        temp = (A - B + C + D +E).sum(axis=0) - p.fe[None,1:]*p.r_hjort[:,None]*self.psi_o_star[...,1:]**(-p.k)
        l_R[...,1:] = (temp*p.eta[...,1:])**(1/p.kappa)
        return l_R
    
    def compute_profit(self,p):
        profit = np.zeros((p.N,p.N,p.S))
        profit[...,1:] = np.einsum('nis,s,i,nis,nis->nis',
                                self.X_M[...,1:],
                                1/p.sigma[1:],
                                1/self.w,
                                1/self.PSI_ME[...,1:],
                                1/(1+p.tariff[...,1:]))
        return profit
    
    def compute_expenditure(self, p):
        A1 = np.einsum('nis,nis->i', 
                      self.X,
                      1/(1+p.tariff))
        A2 = np.einsum('ins,ins,ins->i', 
                      self.X,
                      p.tariff,
                      1/(1+p.tariff))
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = p.deficit_share_world_output*np.einsum('nis,nis->', 
                      self.X,
                      1/(1+p.tariff))
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Z = (A1+A2+B-(C+D))
        return Z
    
    def compute_phi(self, p):
        if self.context == 'calibration':
            denominator_M = np.zeros((p.N, p.N, p.S))
            denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                    self.PSI_ME[..., 1:],
                                    self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                    1/((self.PSI_ME[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                    )
            denominator_CD = np.einsum('ns,ns->ns',
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:])
                                        )
            f_phi = np.einsum('nis,nis,nis->nis',
                            p.trade_shares,
                            1+p.tariff,
                            1/(denominator_M + denominator_CD[:,None,:]))
            
            phi = np.einsum('nis,nns,ns,ns,ns->nis',
                    f_phi**(1/p.theta)[None,None,:],
                    f_phi**(-1/p.theta)[None,None,:],
                    p.T**(1/p.theta[None,:]),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
    
            return phi
        
        elif self.context == 'counterfactual':
            # phi = np.einsum('is,nis,is,is->nis',
            #         p.T**(1/p.theta[None,:]),
            #         1/p.tau,
            #         self.w[:,None]**(-p.alpha[None,:]),
            #         self.price_indices[:,None]**(p.alpha[None,:]-1))
            # return phi
            phi = np.einsum('is,nis,nis,is,is->nis',
                    p.T**(1/p.theta[None,:]),
                    1/p.tau,
                    1/(1+p.tariff),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
            return phi
        
        else:
            print('context attribute needs to be either "calibration" or "counterfactual"')
            
    def compute_wage(self, p):
        wage = (p.alpha[None, :] * ((self.X - self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
                ).sum(axis=1)/self.l_P
        return wage
    
    def compute_tau(self,p, assign = True):
        tau = np.einsum('is,nis,nis,is,is->nis',
                        p.T**(1/p.theta[None,:]),
                        1/self.phi,
                        1/(1+p.tariff),
                        self.w[:,None]**-p.alpha[None,:],
                        self.price_indices[:,None]**(p.alpha[None,:]-1),
                        )
        if assign:
            self.tau = tau
        else:
            return tau
    
    def scale_P(self, p):
        try:
            numeraire = self.price_indices[0]
        except:
            self.compute_solver_quantities(p)
            numeraire = self.price_indices[0]
        
        self.w = self.w / numeraire
        self.Z = self.Z / numeraire
        self.X = self.X / numeraire
        self.X_CD = self.X_CD / numeraire
        self.X_M = self.X_M / numeraire
        self.phi = self.phi * numeraire
        self.price_indices = self.price_indices / numeraire
        self.compute_sectoral_prices(p)
        
    def compute_nominal_value_added(self,p):
        self.nominal_value_added = p.alpha[None, :]*((self.X-self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
    
    def compute_nominal_intermediate_input(self,p):
        self.nominal_intermediate_input = np.einsum('s,is->is',
                           (1-p.alpha)/p.alpha,
                           self.nominal_value_added)
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - self.nominal_intermediate_input.sum(axis=1)
        self.cons = self.nominal_final_consumption/self.price_indices
        
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        self.sectoral_price_indices = one_over_price_indices_no_pow_no_prod**(1/(p.sigma[None, :]-1))
        self.sectoral_cons = np.einsum('s,n,ns->ns',
                                  p.beta,
                                  self.Z,
                                  1/self.sectoral_price_indices
                                  )
        
    def compute_gdp(self,p):
        self.gdp = self.nominal_final_consumption + \
            p.deficit_share_world_output*np.einsum('nis,nis->',
                                                   self.X,
                                                   1/(1+p.tariff)
                                                   ) + \
            self.w*np.einsum('is->i',
                             self.l_R + self.l_Ao
                             ) + \
            np.einsum('n,ins->i',
                      self.w,
                      self.l_Ae) + \
            np.einsum('n,nis->i',
                       self.w,
                       self.l_Aa)

    def compute_pflow(self,p):
        temp_w_a_power_minus_d = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )
        
        bracket = p.k*np.einsum('nis,ni->nis',
                              temp_w_a_power_minus_d,
                              self.integral_k_plus_un_d
                              ) + self.psi_MP_star[...,1:]**(-p.k)
        
        self.pflow = np.einsum('nis,is,is->nis',
                              bracket,
                              p.eta[...,1:],
                              self.l_R[...,1:]**(1-p.kappa)
                              ).squeeze()
        
    def compute_share_of_innovations_patented(self,p):
        # this will only be valid for domestic quantities, we only use it as such
        self.share_innov_patented = self.psi_m_star[...,1:]**(-p.k)
        
    def compute_semi_elast_patenting_delta(self,p):
        # This is not updated with entry costs
        self.r_NP = np.zeros(p.S)
        self.r_NP[1:] = self.r + p.zeta[1:] + p.nu[1:] - self.g + self.g_s[1:]
        
        self.DT = np.zeros((p.N,p.S))
        self.DT[:,1:] = np.einsum('s,is,is->is',
                            p.nu[1:],
                            1/(self.r_NP[None,1:]-p.nu[None,1:]+p.delta[:,1:]),
                            1/(self.r_NP[None,1:]+p.delta[:,1:]))
        
        self.semi_elast_patenting_delta = np.zeros((p.N,p.S))
        A = (
            (1-p.kappa)*p.k/(p.kappa*(p.k-1))
              )*np.einsum('is,is,s,i,is,is->is',
                      p.eta[...,1:],
                      1/self.l_R[...,1:]**p.kappa,
                      p.fe[1:]+p.fo[1:],
                      p.r_hjort,
                      self.psi_o_star[...,1:]**(-p.k),
                      1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:])
                      )
                         
        B = p.k*(1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:]))
        
        self.semi_elast_patenting_delta[...,1:] = p.delta[...,1:]**2*(A+B)
    
    def compute_non_solver_quantities(self,p):
        self.compute_tau(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_pflow(p)      
        self.compute_semi_elast_patenting_delta(p)
        self.compute_share_of_innovations_patented(p)
        
    def compute_average_ratio_entry_costs_exports(self,p):
        self.total_entry_costs_by_innovator = np.einsum('i,nis->i',
            self.w,
            self.l_Aa[...,1:]
            )
        
        self.ratio_total_entry_costs_by_innovator_over_exports = (
            self.total_entry_costs_by_innovator
            / np.einsum('nis->i',
                        self.X_M[..., 1:]
                        )
        )
        
        self.mass_enters = p.k/(p.k-1) - self.mu_MNE
        self.sales_innovators = np.einsum('is,is,nis,nis,nis->i',
                                          p.eta[...,1:],
                                          self.l_R[...,1:]**(1-p.kappa),
                                          self.mass_enters[...,1:],
                                          1/self.PSI_M[...,1:],
                                          self.X_M[...,1:]
                                          )
        self.ratio_total_entry_costs_by_innovator_over_sales_innovators = (
            self.total_entry_costs_by_innovator
            / self.sales_innovators
            )
        
        df = pd.DataFrame(index=p.countries)
        df['Total entry costs by innovator'] = self.total_entry_costs_by_innovator
        df['GDP'] = self.gdp
        df['Ratio to exports'] = self.ratio_total_entry_costs_by_innovator_over_exports
        df['Ratio to exports at entry'] = self.ratio_total_entry_costs_by_innovator_over_sales_innovators
        
        self.summary_entry_costs_quantities = df
        #!!!
        # df = pd.DataFrame(index=pd.MultiIndex.from_product([self.countries,self.countries]
        #                                   , names=['destination','origin']))
        # df['Ratio total entry costs by innovator over sales innovators'] = 

    def compute_consumption_equivalent_welfare(self,p,baseline):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline.g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline.cons
                
    def compute_world_welfare_changes(self,p,baseline):
        one_ov_gamma = 1/p.gamma
        numerator = (p.labor**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = (p.labor**one_ov_gamma*baseline.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_pop_average_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
        numerator = (baseline.cons**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = baseline.cons.sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_negishi_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
class var:
    def __init__(self, context, N = 7, S = 2):
        self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        self.off_diag_mask[np.s_[::(N+1)*S]] = False
        self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        self.diag_mask = np.invert(self.off_diag_mask)
        self.context = context

    def guess_profit(self, profit_init):
        self.profit = profit_init    

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_Z(self, Z_init):
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
    def var_from_vector(vec,p,context,compute = True):
        init = var(context=context)    
        init.guess_wage(vec[0:p.N])
        init.guess_Z(vec[p.N:p.N+p.N])
        init.guess_labor_research(
            np.insert(vec[p.N+p.N:p.N+p.N+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1)
            )
        init.guess_profit(
            np.insert(vec[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2*(p.S-1)].reshape((p.N, p.N, p.S-1)), 0, np.zeros(p.N), axis=2)
            )
        init.guess_phi(vec[p.N+p.N+p.N*(p.S-1)+p.N**2*(p.S-1):].reshape((p.N, p.N, p.S)))
        if compute:
            init.compute_solver_quantities(p)
        return init

    def vector_from_var(self):
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        profit = self.profit[...,1:].ravel()
        Z = self.Z
        phi = self.phi.ravel()
        vec = np.concatenate((w,Z,l_R,profit,phi), axis=0)
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
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()
        self.r = p.rho + self.g/p.gamma
        self.G = self.r+p.zeta-self.g+self.g_s+p.nu
        
    def compute_patenting_thresholds(self, p):
        A = np.einsum('n,n,s,i,i->nis',
                               self.w,
                               p.r_hjort,
                               p.fe[1:],
                               1/self.w,
                               1/p.r_hjort,
                               )
        
        denom_bracket = 1/(self.G[None,:]+p.delta-p.nu[None,:])-1/(self.G[None,:]+p.delta)
        self.psi_C = np.full((p.N,p.N,p.S),np.inf)
        self.psi_C[...,1:] = A*p.r_hjort[None,:,None]/(self.profit[...,1:]*denom_bracket[:,None,1:])
        self.psi_star = np.maximum(self.psi_C,1)
        psi_star_n_star = np.min(self.psi_star,axis=0)
        
        x_old = np.max(self.psi_C[...,1:], axis=0)
        x_new = None
        cond = True
        it = 0
        while cond:
            if it>0:
                x_old = x_new
            mask = x_old[None,:,:]>=self.psi_C[...,1:]
            x_new = (np.sum(A,axis=0,where=mask)+p.fo[None,1:])/np.sum(A/self.psi_C[...,1:],axis=0,where=mask)
            cond = np.any(x_old != x_new)
            it+=1

        condition = np.maximum(A*(psi_star_n_star[None,:,1:]/self.psi_C[...,1:]-1),0).sum(axis=0)>=p.fo[None,1:]
        x_new[condition] = psi_star_n_star[...,1:][condition]
        self.psi_o_star = np.full((p.N,p.S),np.inf)
        self.psi_o_star[...,1:] = x_new
        self.psi_m_star = np.full((p.N,p.N,p.S),np.inf)
        self.psi_m_star[...,1:] = np.maximum(self.psi_o_star[None,:,1:],self.psi_star[...,1:])

    def compute_aggregate_qualities(self, p):
        prefact = p.k * p.eta * self.l_R**(1-p.kappa) /(p.k-1)
        A = (self.g_s[1:] + p.nu[1:] + p.zeta[1:])
        B = np.einsum('s,nis,ns -> nis',
                      p.nu[1:],
                      self.psi_m_star[...,1:]**(1-p.k[None,None,1:]),
                      1/((self.g_s[None, 1:]+p.delta[...,1:]+p.zeta[None, 1:])
           *(self.g_s[None, 1:]+p.delta[...,1:]+p.nu[None,1:]+p.zeta[None,1:]))
                      )
        self.PSI_M = np.zeros((p.N,p.N,p.S))
        self.PSI_M[...,1:] = np.einsum('is,nis -> nis',
                               prefact[...,1:],
                               1/A[None, None, :]+B)
        
        self.PSI_CD = np.ones((p.N,p.S))
        self.PSI_CD[...,1:] = 1-self.PSI_M[...,1:].sum(axis=1)

    def compute_sectoral_prices(self, p):
        power = p.sigma-1
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, 1:] \
            * (self.PSI_M[...,1:]*self.phi[...,1:]**power[None, None, 1:]).sum(axis=1)

        B = self.PSI_CD[...,1:]*(self.phi[...,1:]**p.theta[None,None,1:]).sum(axis=1)**(power/p.theta)[None, 1:]

        self.P_M = np.full((p.N, p.S),np.inf)
        self.P_M[:,1:] = (A/(A+B))**(1/(1-p.sigma))[None, 1:]
        
        self.P_CD = np.ones((p.N, p.S))
        self.P_CD[:,1:] = (B/(A+B))**(1/(1-p.sigma))[None, 1:]
        
    def compute_labor_allocations(self, p):
        self.l_Ae = np.zeros((p.N,p.N,p.S))
        self.l_Ae[...,1:] = np.einsum('n,s,is,is,nis -> ins',
                         p.r_hjort,
                         p.fe[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         self.psi_m_star[...,1:]**-p.k[None,None,1:]
                         )
        self.l_Ao = np.zeros((p.N,p.S))
        self.l_Ao[...,1:] = np.einsum('i,s,is,is,is -> is',
                         p.r_hjort,
                         p.fo[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         self.psi_o_star[...,1:]**-p.k[None,1:]
                         )
        self.l_P = p.labor-(self.l_Ao+self.l_R+self.l_Ae.sum(axis=0)).sum(axis=1)
        
    def compute_price_indices(self, p, assign = True):
        power = (p.sigma-1)
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**power[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :]/(p.sigma[None, :]-1)) ).prod(axis=1)
        if assign:
            self.price_indices = price_indices
        else:
            return price_indices
        
    def compute_trade_flows_and_shares(self, p, assign = True):
            temp = (self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
            X_M = np.zeros((p.N, p.N, p.S))
            X_M[...,1:] = np.einsum('nis,nis,ns,ns,s,n->nis',
                                    self.phi[..., 1:]**(p.sigma-1)[None, None, 1:],
                                    self.PSI_M[..., 1:],
                                    1/temp,
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:]),
                                    p.beta[1:],
                                    self.Z
                                    )
            X_CD = np.einsum('nis,ns,ns,s,n->nis',
                                        self.phi**(p.theta)[None,None,:],
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:]),
                                        p.beta,
                                        self.Z
                                        )
            X = X_M+X_CD
            if assign:
                self.X_M = X_M
                self.X_CD = X_CD
                self.X = X
            else:
                return X_M,X_CD,X
        
    def compute_solver_quantities(self,p):
        self.compute_growth(p)
        self.compute_patenting_thresholds(p)
        self.compute_aggregate_qualities(p)
        self.compute_sectoral_prices(p)
        self.compute_labor_allocations(p)
        self.compute_trade_flows_and_shares(p)
        self.compute_price_indices(p)

    def compute_wage(self, p):
        wage = (p.alpha[None, :] * ((self.X - self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
                ).sum(axis=1)/self.l_P
        return wage
            
    def compute_labor_research(self, p):
        A1 = ((p.k[None,None,1:]/(p.k[None,None,1:]-1))*self.profit[...,1:]/self.G[None,None,1:]).sum(axis=0)
        A2 = np.einsum('nis,n,s,n,i,nis->is',
                       self.psi_m_star[...,1:]**-p.k[None,None,1:],
                       self.w,
                       p.fe[1:],
                       p.r_hjort,
                       1/self.w,
                       p.k[None,None,1:]*self.psi_m_star[...,1:]/(self.psi_C[...,1:]*(p.k[None,None,1:]-1))-1
                       )
        B = self.psi_o_star[:,1:]**-p.k[None,1:]*p.fo[None,1:]*p.r_hjort[:,None]
        l_R = np.zeros((p.N,p.S))
        l_R[...,1:] = (p.eta[...,1:]*(A1+A2-B))**(1/p.kappa)
        # assert np.isnan(l_R).sum() == 0, 'nan in l_R'
        return l_R
    
    def compute_profit(self,p):
        profit = np.zeros((p.N,p.N,p.S))
        profit[...,1:] = np.einsum('nis,s,i,nis,nis->nis',
                                self.X_M[...,1:],
                                1/p.sigma[1:],
                                1/self.w,
                                1/self.PSI_M[...,1:],
                                1/(1+p.tariff[...,1:]))
        return profit
    
    def compute_expenditure(self, p):
        A1 = np.einsum('nis,nis->i', 
                      self.X,
                      1/(1+p.tariff))
        A2 = np.einsum('ins,ins,ins->i', 
                      self.X,
                      p.tariff,
                      1/(1+p.tariff))
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = p.deficit_share_world_output*np.einsum('nis,nis->', 
                      self.X,
                      1/(1+p.tariff))
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Z = (A1+A2+B-(C+D))
        return Z
    
    def compute_phi(self, p):
        if self.context == 'calibration':
            denominator_M = np.zeros((p.N, p.N, p.S))
            denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                    self.PSI_M[..., 1:],
                                    self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                    1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                    )
            denominator_CD = np.einsum('ns,ns->ns',
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:])
                                        )
            f_phi = np.einsum('nis,nis,nis->nis',
                            p.trade_shares,
                            1+p.tariff,
                            1/(denominator_M + denominator_CD[:,None,:]))
            
            phi = np.einsum('nis,nns,ns,ns,ns->nis',
                    f_phi**(1/p.theta)[None,None,:],
                    f_phi**(-1/p.theta)[None,None,:],
                    p.T**(1/p.theta[None,:]),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
    
            return phi
        
        elif self.context == 'counterfactual':
            # phi = np.einsum('is,nis,is,is->nis',
            #         p.T**(1/p.theta[None,:]),
            #         1/p.tau,
            #         self.w[:,None]**(-p.alpha[None,:]),
            #         self.price_indices[:,None]**(p.alpha[None,:]-1))
            # return phi
            phi = np.einsum('is,nis,nis,is,is->nis',
                    p.T**(1/p.theta[None,:]),
                    1/p.tau,
                    1/(1+p.tariff),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
            return phi
        
        else:
            print('context attribute needs to be either "calibration" or "counterfactual"')
    
    def check_phi(self,p):
        denominator_M = np.zeros((p.N, p.N, p.S))
        denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                self.PSI_M[..., 1:],
                                self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CD = np.einsum('ns,ns->ns',
                                    1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                    self.P_CD**(1-p.sigma[None,:])
                                    )
        one_over_denominator = 1/(denominator_M + denominator_CD[:,None,:])
        phi = np.einsum('nis,s,n,nis->nis',
                        self.X,
                        1/p.beta,
                        1/self.Z,
                        one_over_denominator)**(1/p.theta)[None,None,:]
        return self.phi/phi
    
    # def scale_tau(self,p):
        
    #     self.phi = self.phi\
    #         *np.einsum('ns,ns,ns->ns',
    #             p.T**(1/p.theta[None,:]),
    #             self.w[:,None]**(-p.alpha[None,:]),
    #             self.price_indices[:,None]**(p.alpha[None,:]-1))[:,None,:]\
    #         /np.einsum('nns->ns',self.phi)[:,None,:]
    
    def compute_tau(self,p, assign = True):
        tau = np.einsum('is,nis,nis,is,is->nis',
                        p.T**(1/p.theta[None,:]),
                        1/self.phi,
                        1/(1+p.tariff),
                        self.w[:,None]**-p.alpha[None,:],
                        self.price_indices[:,None]**(p.alpha[None,:]-1),
                        )
        if assign:
            self.tau = tau
        else:
            return tau
    
    def scale_P(self, p):
        try:
            numeraire = self.price_indices[0]
        except:
            self.compute_solver_quantities(p)
            numeraire = self.price_indices[0]
        
        self.w = self.w / numeraire
        self.Z = self.Z / numeraire
        self.X = self.X / numeraire
        self.X_CD = self.X_CD / numeraire
        self.X_M = self.X_M / numeraire
        self.phi = self.phi * numeraire
        self.price_indices = self.price_indices / numeraire
        self.compute_sectoral_prices(p)
    
    def compute_nominal_value_added(self,p):
        self.nominal_value_added = p.alpha[None, :]*((self.X-self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
    
    def compute_nominal_intermediate_input(self,p):
        self.nominal_intermediate_input = np.einsum('s,is->is',
                           (1-p.alpha)/p.alpha,
                           self.nominal_value_added)
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - self.nominal_intermediate_input.sum(axis=1)
        self.cons = self.nominal_final_consumption/self.price_indices
        
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        self.sectoral_price_indices = one_over_price_indices_no_pow_no_prod**(1/(p.sigma[None, :]-1))
        self.sectoral_cons = np.einsum('s,n,ns->ns',
                                  p.beta,
                                  self.Z,
                                  1/self.sectoral_price_indices
                                  )
        
    def compute_gdp(self,p):
        self.gdp = self.nominal_final_consumption + \
            p.deficit_share_world_output*np.einsum('nis,nis->',
                                                   self.X,
                                                   1/(1+p.tariff)
                                                   ) + \
            self.w*np.einsum('is->i',
                             self.l_R + self.l_Ao
                             ) + \
            np.einsum('n,ins->i',
                      self.w,
                      self.l_Ae)

    def compute_pflow(self,p):
        self.pflow = np.einsum('nis,is,is->nis',
                              self.psi_m_star[...,1:]**(-p.k[None,None,1:]),
                              p.eta[...,1:],
                              self.l_R[...,1:]**(1-p.kappa)
                              ).squeeze()
        
    def compute_share_of_innovations_patented(self,p):
        self.share_innov_patented = self.psi_m_star[...,1:]**(-p.k[None,None,1:])
    
    def compute_welfare(self,p):
        # exp = 1-1/p.gamma
        # self.U = self.cons**(exp)/(p.rho-self.g*exp)/exp
        pass
    
    def compute_semi_elast_patenting_delta(self,p):
        self.r_NP = np.zeros(p.S)
        self.r_NP[1:] = self.r + p.zeta[1:] + p.nu[1:] - self.g + self.g_s[1:]
        
        self.DT = np.zeros((p.N,p.S))
        self.DT[:,1:] = np.einsum('s,is,is->is',
                            p.nu[1:],
                            1/(self.r_NP[None,1:]-p.nu[None,1:]+p.delta[:,1:]),
                            1/(self.r_NP[None,1:]+p.delta[:,1:]))
        
        # self.semi_elast_patenting_delta = np.zeros((p.N,p.S))
        
        # numerator_prefact = p.k*np.einsum('is,is,is,is->is',
        #                               p.delta[:,1:]**2,
        #                               np.diagonal(self.profit[...,1:]).transpose(),
        #                               np.diagonal(self.psi_m_star[...,1:]).transpose()**(1-p.k),
        #                               self.DT[:,1:]
        #                               )
        # numerator_sum = 1/(self.r_NP[None,1:]-p.nu[None,1:]+p.delta[:,1:])\
        #                     + 1/(self.r_NP[None,1:]+p.delta[:,1:])
        # denominator = p.kappa*np.einsum('mis,mis->is',
        #                         self.profit[...,1:],
        #                         p.k/self.r_NP[None,None,1:]+np.einsum('mis,ms->mis',
        #                                                               self.psi_m_star[...,1:]**(1-p.k),
        #                                                               self.DT[:,1:]),
        #                         )
        
        # self.semi_elast_patenting_delta[...,1:] = numerator_prefact*numerator_sum/denominator
        
        # self.G = self.r+p.zeta-self.g+self.g_s+p.nu
        self.semi_elast_patenting_delta = np.zeros((p.N,p.S))
        A = (
            (1-p.kappa)*p.k[None,1:]/(p.kappa*(p.k[None,1:]-1))
              )*np.einsum('is,is,s,i,is,is->is',
                      p.eta[...,1:],
                      1/self.l_R[...,1:]**p.kappa,
                      p.fe[1:]+p.fo[1:],
                      p.r_hjort,
                      self.psi_o_star[...,1:]**(-p.k[None,1:]),
                      1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:])
                      )
                         
        B = p.k[None,1:]*(1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:]))
        
        self.semi_elast_patenting_delta[...,1:] = p.delta[...,1:]**2*(A+B)

    def compute_non_solver_aggregate_qualities(self,p): 
        self.PSI_MPND = np.zeros((p.N,p.N,p.S))
        self.PSI_MPD = np.zeros((p.N,p.N,p.S))
        self.PSI_MPL = np.zeros((p.N,p.N,p.S))
        self.PSI_MNP = np.zeros((p.N,p.N,p.S))
        prefact = p.k[None,1:] * p.eta[...,1:] * self.l_R[...,1:]**(1-p.kappa)/(p.k[None,1:]-1)
        A = (self.g_s[1:] + p.nu[1:] + p.zeta[1:])
        self.PSI_MPND[...,1:] = np.einsum('is,nis,ns->nis',
                                  prefact,
                                  self.psi_m_star[...,1:]**(1-p.k[None,None,1:]),
                                  1/(A[None,:]+p.delta[...,1:]))
        self.PSI_MPD[...,1:] = np.einsum('s,nis,ns->nis',
                                 p.nu[1:],
                                 self.PSI_MPND[...,1:],
                                 1/(p.delta[...,1:]+self.g_s[None,1:]+p.zeta[None,1:]))
        numerator_A = np.einsum('is,nis->nis',
                                prefact,
                                1-self.psi_m_star[...,1:]**(1-p.k[None,None,1:]))
        numerator_B= np.einsum('ns,nis->nis',
                               p.delta[...,1:],
                               self.PSI_MPND[...,1:])
        self.PSI_MNP[...,1:] = (numerator_A + numerator_B)/A[None,None,:]
    
    def compute_V(self,p):
        self.V_NP = np.einsum('nis,i,s->nis',
                              self.profit,
                              self.w,
                              1/self.G
                              )
        self.V_PD = np.einsum('nis,i,ns->nis',
                              self.profit,
                              self.w,
                              1/(self.G[None,:]-p.nu[None,:]+p.delta)
                              )
        
        self.V_P = np.einsum('nis,i,ns->nis',
                             self.profit,
                             self.w,
                             1/(self.G[None,:]-p.nu[None,:]+p.delta)-1/(self.G[None,:]+p.delta)+1/(self.G[None,:])
                             )
        
        self.V = np.zeros((p.N,p.S))
        
        A1 = ((p.k[None,None,1:]/(p.k[None,None,1:]-1))*self.V_NP[...,1:]).sum(axis=0)
        A2 = np.einsum('nis,n,s,n,nis->is',
                       self.psi_m_star[...,1:]**-p.k[None,None,1:],
                       self.w,
                       p.fe[1:],
                       p.r_hjort,
                       p.k[None,None,1:]*self.psi_m_star[...,1:]/(self.psi_C[...,1:]*(p.k[None,None,1:]-1))-1
                       )
        B = self.psi_o_star[:,1:]**-p.k[None,1:]*p.fo[None,1:]*p.r_hjort[:,None]*self.w[:,None]
        self.V[...,1:] = A1+A2-B
        
    def compute_quantities_with_prod_patents(self,p,upper_bound_integral = np.inf):
        
        def incomplete_sum_with_exponent(matrix,exponent):
            res = np.full_like(matrix,np.nan)
            for i in range(matrix.shape[1]):
                res[:,i] = (np.delete(matrix,i,axis=1)**exponent).sum(axis=1)
            return res
        
        def upper_inc_gamma(a,x):
            return gamma(a)*gammaincc(a,x)
            # return gammaincc(a,x)
        
        def sim(z):
            bound_A = np.einsum('i,ni->ni',
                                p.T[..., 1],
                                incomplete_sum_with_exponent(
                                    self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1]
                                )/(z**p.theta[1])

            A_bracket_1 = upper_inc_gamma(
                (p.theta[1]+1-p.sigma[1])/p.theta[1],
                bound_A
            )
            A_bracket_2 = upper_inc_gamma(
                (p.theta[1]+1-p.sigma[1])/p.theta[1],
                bound_A*(p.sigma[1]/(p.sigma[1] - 1))**p.theta[1]
            )

            A = np.einsum('ni,i,ni->ni',
                          (incomplete_sum_with_exponent(self.phi[..., 1], p.theta[1])
                           / self.phi[..., 1]**p.theta[1])**((p.sigma[1]-1)/p.theta[1]),
                          p.T[..., 1]**(-1/p.theta[1]),
                          A_bracket_1 - A_bracket_2
                          )

            bound_B = np.einsum('i,ni->ni',
                                p.T[..., 1],
                                incomplete_sum_with_exponent(
                                    self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1]
                                )/(z**p.theta[1])

            B_bracket_1 = upper_inc_gamma(
                (p.theta[1]-p.sigma[1])/p.theta[1],
                bound_B
            )
            B_bracket_2 = upper_inc_gamma(
                (p.theta[1]-p.sigma[1])/p.theta[1],
                bound_B*(p.sigma[1]/(p.sigma[1] - 1))**p.theta[1]
            )

            B = np.einsum('ni,ni->ni',
                          (incomplete_sum_with_exponent(
                              self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1])**(p.sigma[1]/p.theta[1]),
                          B_bracket_1 - B_bracket_2
                          )/z

            return A-B
        
        # integral_calculated = np.full_like(self.phi[...,1],np.nan)
        
        # def integrand(z,i,j):
        #     return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*sim(z) )[i,j]
        
        # for i in range(p.N):
        #     for j in range(p.N):
        #         print(integrate.quad(lambda x: integrand(x,i,j), 0, upper_bound_integral,full_output=1))
        #         integral_calculated[i,j] = integrate.quad(lambda x: integrand(x,i,j), 0, upper_bound_integral,full_output=1).y
        
        def integrand(z):
            return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*sim(z) )
            # return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][:,None]*z**(-p.theta[1]))*sim(z) )
        
        # c1 = 0
        # # c2 = 0
        # fig,ax=plt.subplots(figsize = (12,8))
        # for c2 in range(11):
        #     ax.plot(np.logspace(0,2,1001),[sim(z)[c1,c2] for z in np.logspace(0,2,1001)],
        #              label=p.countries[c1]+'_'+p.countries[c2])
        # plt.legend()
        # plt.title('sim(z))')
        # plt.xscale('log')
        # plt.show()
        
        self.integral_result = integrate.quad_vec(lambda x: integrand(x), 0, upper_bound_integral,full_output=1)
        integral_calculated = self.integral_result[0]
        
        A = (1 +
             (incomplete_sum_with_exponent(
                 self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1])
             * (p.sigma[1]/(p.sigma[1]-1))**p.theta[1]
             )**((p.sigma[1] - p.theta[1] - 1)/p.theta[1])
        
        B = np.einsum('i,ni->ni',
                        p.T[...,1]**((1+p.theta[1])/p.theta[1]),
                      # p.T[...,1]**((1)/p.theta[1]),
                      integral_calculated
                      )*p.theta[1]*p.sigma[1]**p.sigma[1]/(
                          (p.sigma[1]-1)**(p.sigma[1]-1)*gamma((p.theta[1]+1-p.sigma[1])/p.theta[1])
                          )
        
        self.profit_with_prod_patent = np.zeros_like(self.profit)
        self.profit_with_prod_patent[...,1] = self.profit[...,1]*(A+B)
        
        for i,country in enumerate(p.countries):
            self.profit_with_prod_patent[i,i,1] = 0
        
        self.profit_with_prod_patent_D = np.zeros_like(self.profit)
        self.profit_with_prod_patent_D[...,1] = self.profit[...,1]*(
            self.phi[...,1]**p.theta[1]/(self.phi[...,1]**p.theta[1]).sum(axis=1)[:,None]
            )**((-p.sigma[1]+p.theta[1]+1)/p.theta[1])
        
        # self.profit_with_prod_patent_D_bis = np.zeros_like(self.profit)
        # self.profit_with_prod_patent_D_bis[...,1] = ((p.sigma[1]-1)**(p.sigma[1]-1)/p.sigma[1]**p.sigma[1]
        #                                              )*np.einsum('ni,n,i->ni',
        #                                                          self.X_CD[...,1],
        #                                                          1/self.PSI_CD[...,1],
        #                                                          1/self.w
        #                                                          )
        
        # # alternative way of computing with direct integration on p
        
        # A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
        #     * (self.PSI_M * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
        # B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        # temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        # one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        # sectoral_price_indices = one_over_price_indices_no_pow_no_prod**(1/(p.sigma[None, :]-1))
        
        # sectoral_cons = np.einsum('s,n,ns->ns',
        #                           p.beta,
        #                           self.Z,
        #                           1/sectoral_price_indices
        #                           )
                                                                 
        # self.Pr_E_1_over_psi = np.zeros_like(self.profit)
        # self.Pr_E_1_over_psi[...,1] = np.einsum('i,,,n,n,ni,,ni,i,i->ni',
        #                                   p.T[...,1],
        #                                   (p.sigma[1]-1)**p.theta[1],
        #                                   1/(p.sigma[1])**(p.theta[1]+1),
        #                                   sectoral_price_indices[...,1]**p.sigma[1],
        #                                   sectoral_cons[...,1],
        #                                   ( self.phi[...,1]**p.theta[1] * ((p.sigma[1]-1)/p.sigma[1])**p.theta[1] + incomplete_sum_with_exponent(self.phi[...,1],p.theta[1])
        #                                       )**( (p.sigma[1] - p.theta[1] - 1)/p.theta[1] ),
        #                                   gamma( (p.theta[1]+1-p.sigma[1])/p.theta[1] ),
        #                                   p.tau[...,1]**-p.theta[1],
        #                                   self.w**-(p.theta[1]*p.alpha[1]),
        #                                   self.price_indices**(-p.theta[1]*(1-p.alpha[1]))
        #                                   )
        
        # def p_integrand_for_Pr_E_2_over_psi(x,z,i,j):
        #     # A = p**(p.theta[1]-p.sigma[1])*np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * p**p.theta[1])
        #     # B = np.einsum('',
        #     #                 p.tau[...,1],
        #     #                 p.w**p.alpha[1],
        #     #                 self.price_indices**(1-p.alpha[1]),
        #     #                 p**(p.theta[1]-p.sigma[1]-1),
        #     #                 np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * p**p.theta[1])
        #     #                 )/z
        #     # return (A-B)[i,j]
        #     res = np.einsum('ni,,ni->ni',
        #                     x - np.einsum('ni,i,i->ni',
        #                             p.tau[...,1],
        #                             self.w**p.alpha[1],
        #                             self.price_indices**(1-p.alpha[1]))/z,
        #                     x**(p.theta[1]-p.sigma[1]-1),
        #                     np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * x**p.theta[1])
        #                     )*p.theta[1]/z
        #     return res[i,j]
        
        # # from tqdm import tqdm
        
        # def p_integral_for_Pr_E_2_over_psi(z):
        #     p_lb = np.einsum('ni,i,i->ni',
        #                     p.tau[...,1],
        #                     self.w**p.alpha[1],
        #                     self.price_indices**(1-p.alpha[1]),
        #                     )/z
        #     p_ub = p.sigma[1]*p_lb/(p.sigma[1]-1)
        #     p_integral_calculated = np.zeros_like(self.profit)
        #     for i in range(p.N):
        #         for j in range(p.N):
        #             p_integral_calculated[i,j,1] = integrate.quad(p_integrand_for_Pr_E_2_over_psi, p_lb[i,j], p_ub[i,j], args=(z,i,j))[0]
        #     return p_integral_calculated
        
        # # def second_p_integrand_for_Pr_E_2_over_psi(z):
        # #     pass
        
        # # def second_p_integral_for_Pr_E_2_over_psi():
        # #     pass
            
        # def z_integrand_for_Pr_E_2_over_psi(z):
        #     return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*p_integral_for_Pr_E_2_over_psi(z)[...,1] )
        
        # def z_integral_for_Pr_E_2_over_psi():
        #     return integrate.quad_vec(z_integrand_for_Pr_E_2_over_psi, 0, np.inf, full_output=1)[0]
            
                                                         
        # self.Pr_E_2_over_psi = np.zeros_like(self.profit)
        # self.Pr_E_2_over_psi[...,1] = np.einsum('i,,n,n,ni,ni->ni',
        #                                   p.T[...,1],
        #                                   p.theta[1],
        #                                   sectoral_price_indices[...,1]**p.sigma[1],
        #                                   sectoral_cons[...,1],
        #                                   z_integral_for_Pr_E_2_over_psi(),
        #                                   incomplete_sum_with_exponent(self.phi[...,1],p.theta[1])
        #                                   )
                                                                 
        # self.profit_with_prod_patent_with_p_integral = (self.Pr_E_1_over_psi+self.Pr_E_2_over_psi)/self.w[None,:,None]
        
        # # end alternative way of computing with direct integration on p
        
        self.V_NP_P_minus_V_NP_NP_with_prod_patent = np.zeros_like(self.profit)
        self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1] = \
            self.profit_with_prod_patent[...,1]*(
                1/(self.G[1]+p.delta[:,1]-p.nu[1])-1/(self.G[1]+p.delta[:,1])
                )[None,:]*self.w[None,:]
            
        for i,country in enumerate(p.countries):
            self.V_NP_P_minus_V_NP_NP_with_prod_patent[i,i,1] = 0
        
        self.V_P_P_minus_V_P_NP_with_prod_patent = np.zeros_like(self.profit)
        self.V_P_P_minus_V_P_NP_with_prod_patent[...,1] = \
            self.profit_with_prod_patent[...,1]*(
                1/(self.G[1]+p.delta[None,:,1]-p.nu[1])-1/(self.G[1]+p.delta[None,:,1]) \
                    - 1/(self.G[1]+p.delta[None,:,1]+p.delta[:,None,1]-p.nu[1]) + 1/(self.G[1]+p.delta[None,:,1]+p.delta[:,None,1])
                )*self.w[None,:]
        
        for i,country in enumerate(p.countries):
            self.V_P_P_minus_V_P_NP_with_prod_patent[i,i,1] = 0
        
        # i)
        # case a
        
        self.psi_o_star_with_prod_patent_a = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_a = np.full_like(self.psi_o_star,np.inf)
        
        denom_A = np.diagonal(self.V_P[...,1]-self.V_NP[...,1])/self.w
        denom_B = (self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]).sum(axis=0)-np.diagonal(
            self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:])
        denom = denom_A + denom_B
        
        self.psi_o_star_with_prod_patent_a[...,1] = p.r_hjort*(p.fe[1] + p.fo[1])/denom
        self.psi_o_star_without_prod_patent_a[...,1] = p.r_hjort*(p.fe[1] + p.fo[1])/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_a = self.psi_o_star_with_prod_patent_a**-p.k
        self.share_innov_patented_dom_without_prod_patent_a = self.psi_o_star_without_prod_patent_a**-p.k
        
        # case b
        
        self.psi_o_star_with_prod_patent_b = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_b = np.full_like(self.psi_o_star,np.inf)
        
        mask_B = np.diagonal(self.psi_m_star[...,1])[None,:]<self.psi_m_star[...,1]
        mask_C = np.diagonal(self.psi_m_star[...,1])[None,:]>self.psi_m_star[...,1]
        
        denom_A = np.diagonal(self.V_P[...,1]-self.V_NP[...,1])/self.w
        denom_B = (mask_B*(self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:])).sum(axis=0)
        denom_C = (mask_C*(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:])).sum(axis=0)
        denom = denom_A + denom_B + denom_C
        
        self.psi_o_star_with_prod_patent_b[...,1] = p.r_hjort*p.fe[1]/denom
        self.psi_o_star_without_prod_patent_b[...,1] = p.r_hjort*p.fe[1]/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_b = self.psi_o_star_with_prod_patent_b**-p.k
        self.share_innov_patented_dom_without_prod_patent_b = self.psi_o_star_without_prod_patent_b**-p.k
        
        # check on ib)
        
        # check_b_lhs = self.psi_o_star_with_prod_patent_b[...,1]
        # check_b_rhs = np.min(self.psi_m_star[...,1],axis=0)
        
        for i,country in enumerate(p.countries):
            if np.argmin(self.psi_m_star[:,i,1]) != i and (
                    self.psi_m_star[:,i,1]==np.min(self.psi_m_star[:,i,1])).sum() == 1:
                print(f'check b for {country}')
                if self.psi_o_star_with_prod_patent_b[i,1] > np.min(self.psi_m_star[:,i,1]):
                    print('passed')
                else:
                    print('not passed')    
                    
                print(f'check b for order for {country}')
                if len([p.countries[x]
                         for x in np.where(self.psi_m_star[:, i, 1] < self.psi_m_star[i, i, 1])[0]
                         if x != i]
                        ) == len([p.countries[x]
                                 for x in np.where(self.psi_m_star[:, i, 1] < self.psi_o_star_with_prod_patent_b[i, 1])[0]
                                 if x != i]
                                ):
                    print('passed, patents in before :',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_m_star[i,i,1])[0] if x!=i],
                          'to after:',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_o_star_with_prod_patent_b[i,1])[0] if x!=i],
                          'countries patent before origin') 
                else:
                    print('not passed, patents in before :',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_m_star[i,i,1])[0] if x!=i],
                          'to after:',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_o_star_with_prod_patent_b[i,1])[0] if x!=i],
                          'countries patent before origin') 
        
        # case c
        
        self.psi_o_star_with_prod_patent_c = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_c = np.full_like(self.psi_o_star,np.inf)
        
        mask_is_n_in_n_star_of_i = np.isclose(self.psi_m_star[...,1],np.min(self.psi_m_star[...,1],axis=0))
        
        denom_A = (mask_is_n_in_n_star_of_i
                   *(self.V_P[...,1]-self.V_NP[...,1])
                   /self.w[None,:]
                   ).sum(axis=0)
        denom_B = (mask_is_n_in_n_star_of_i
                   *(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1])
                   /self.w[None,:]
                   ).sum(axis=0)
        denom_C = (~mask_is_n_in_n_star_of_i*self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
                   ).sum(axis=0)
        denom = denom_A + denom_B + denom_C
        num = (
            (self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:])*mask_is_n_in_n_star_of_i
            ).sum(axis=0) + p.r_hjort*p.fo[1]
    
        self.psi_o_star_with_prod_patent_c[...,1] = num/denom
        self.psi_o_star_without_prod_patent_c[...,1] = num/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_c = self.psi_o_star_with_prod_patent_c**-p.k
        self.share_innov_patented_dom_without_prod_patent_c = self.psi_o_star_without_prod_patent_c**-p.k
        
        # check on ic)
        
        for n, destination in enumerate(p.countries):
            for i, origin in enumerate(p.countries):
                if mask_is_n_in_n_star_of_i[n,i] and mask_is_n_in_n_star_of_i.sum(axis=0)[i]>1 and i!=n:
                    test_mask = mask_is_n_in_n_star_of_i.copy()
                    test_mask = False
                    denom_A = (test_mask
                               *(self.V_P[...,1]-self.V_NP[...,1])
                               /self.w[None,:]
                               ).sum(axis=0)
                    denom_B = (test_mask
                               *(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1])
                               /self.w[None,:]
                               ).sum(axis=0)
                    denom_C = (~test_mask*self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
                               ).sum(axis=0)
                    denom = denom_A + denom_B + denom_C
                    num = (
                        (self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:])*test_mask
                        ).sum(axis=0) + p.r_hjort*p.fo[1]
                    
                    print(f'check c when excluding destination {destination} from origin {origin}')
                    
                    if (self.psi_o_star_with_prod_patent_c[...,1]*denom)[i] > num[i]:
                        print('not passed')
                    else:
                        print('passed')
                        
            
        # iaa)
        
        self.psi_m_star_without_prod_patent_aa = np.full_like(self.psi_m_star,np.inf)
        self.psi_m_star_with_prod_patent_aa = np.full_like(self.psi_m_star,np.inf)
        
        denom_A = (self.V_P[...,1]-self.V_NP[...,1])/self.w[None,:]
        denom_B = self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
        denom_C = self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:]
        denom = denom_A - denom_B + denom_C
        
        self.psi_m_star_without_prod_patent_aa[...,1] = self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:]/denom_A
        self.psi_m_star_with_prod_patent_aa[...,1] = self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:]/denom
         
        # icc)
        
        self.psi_o_star_without_prod_patent_cc = self.psi_o_star_without_prod_patent_c.copy()
        
        # icc2)
        
        self.psi_o_star_with_prod_patent_cc2 = np.full_like(self.psi_o_star,np.inf)
        
        def compute_threshold_if_only_patent_domestically_first(country_index):
            res = np.full_like(self.psi_o_star,np.inf)
            
            denom_A = np.diagonal(
                (self.V_P[..., 1]-self.V_NP[..., 1])/self.w[None, :]
            )
            denom_B = ((self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1])
                       /self.w[None,:]
                       ).sum(axis=0)-np.diagonal((self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1])
                                                 /self.w[None,:])
            denom = denom_A + denom_B
            num =  p.r_hjort*(p.fo[1]+p.fe[1])
        
            res = num/denom
            
            return res[country_index]
        
        for i, country in enumerate(p.countries):
            self.psi_o_star_with_prod_patent_cc2[i,1] = compute_threshold_if_only_patent_domestically_first(i)
        
        # icc1)
        
        self.psi_o_star_with_prod_patent_cc1 = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_with_prod_patent_cc = np.full_like(self.psi_o_star,np.inf)
        
        def compute_threshold_if_patent_domestically_and_foreign_simultaneously(mask_is_n_in_n_star_of_i,country_index):
            res = np.full_like(self.psi_o_star,np.inf)
            
            # denom_A = np.diagonal(
            #     (self.V_P[..., 1]-self.V_NP[..., 1])/self.w[None, :]
            # )
            # denom_A = 0
            denom_B = (mask_is_n_in_n_star_of_i
                       *(self.V_P_P_minus_V_P_NP_with_prod_patent[:,country_index,1]+self.V_P[:,country_index, 1]-self.V_NP[:,country_index,1])
                       /self.w[country_index]
                       ).sum()
            denom_C = (~mask_is_n_in_n_star_of_i*self.V_NP_P_minus_V_NP_NP_with_prod_patent[:,country_index,1]/self.w[country_index]
                       ).sum()
            denom = denom_B + denom_C
            num = (
                (self.w*p.r_hjort*p.fe[1]/self.w[country_index])*mask_is_n_in_n_star_of_i
                ).sum() + p.r_hjort[country_index]*p.fo[1]
            
            res = num/denom
            
            return res
        
        def subsets(s):
            x = len(s)
            masks = [1 << i for i in range(x)]
            for i in range(1,1 << x):
                yield [ss for mask, ss in zip(masks, s) if i & mask]
                
        self.cc1_min_patenting_combination_by_origin = [[x] for x in p.countries]
                    
        for i, origin in enumerate(p.countries):
            initial_mask_is_n_in_n_star_of_i = self.psi_m_star[:,i,1] == np.min(self.psi_m_star[:,i,1])
            
            countries_to_test = [p.countries[k] for k in np.where(initial_mask_is_n_in_n_star_of_i)[0] if k!=i]
            combinations_of_countries_to_test = list(subsets(countries_to_test))
            self.psi_o_star_with_prod_patent_cc[i,1] = self.psi_o_star_with_prod_patent_cc2[i,1]
            self.psi_o_star_with_prod_patent_cc1[i,1] = np.inf
            self.cc1_min_patenting_combination_by_origin[i] = [origin]
            
            for combination_of_countries in combinations_of_countries_to_test:
                new_mask = np.array([c in combination_of_countries or c==origin for c in p.countries])
                new_threshold = compute_threshold_if_patent_domestically_and_foreign_simultaneously(new_mask,i)
                print(origin,combination_of_countries,new_mask,new_threshold)
                if new_threshold < self.psi_o_star_with_prod_patent_cc1[i,1]:
                    self.psi_o_star_with_prod_patent_cc1[i,1] = new_threshold
                if new_threshold  < self.psi_o_star_with_prod_patent_cc[i,1]:
                    self.psi_o_star_with_prod_patent_cc[i,1] = new_threshold
                    self.cc1_min_patenting_combination_by_origin[i] = combination_of_countries
            
        # gather every change of patenting threshold in one array
        
        self.psi_m_star_with_prod_patent = np.full_like(self.psi_m_star,np.inf)
        self.case_marker = np.empty(self.psi_m_star[...,1].shape, dtype="<U20")
        
        for i,origin in enumerate(p.countries):
            if self.psi_m_star[i,i,1] == np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] == 1:
                    # case a where the domestic threshold is the smallest one for the origin, and it is the only smallest one
                    # country i patents only at home first
                    print(f'{origin},domestic case a')
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_a[i,1]
                    self.case_marker[i,i] = 'a'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                            self.case_marker[n,i] = 'aa'
            
            elif self.psi_m_star[i,i,1] == np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] > 1:
                    # case c where the domestic threshold is the smallest one for the origin, but it is not the only smallest one
                    # country i patents first at home and abroad at the same time
                    print(f'{origin},case cc')
                    # self.psi_m_star_with_prod_patent[i,i,1] = np.minimum(self.psi_o_star_with_prod_patent_cc1[i,1],
                    #                                                       self.psi_o_star_with_prod_patent_cc2[i,1])
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_cc[i,1]
                    self.case_marker[i,i] = 'cc'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            if self.psi_m_star[n,i,1] == np.min(self.psi_m_star[:,i,1]):
                                #case cc1)
                                if p.countries[n] in self.cc1_min_patenting_combination_by_origin[i]:
                                    self.psi_m_star_with_prod_patent[n,i,1] = self.psi_o_star_with_prod_patent_cc1[i,1]
                                    self.case_marker[n,i] = 'cc1'
                                    
                                #case cc2)
                                else:
                                    self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                    self.case_marker[n,i] = 'cc2'
                                    
                            elif self.psi_m_star[n,i,1] > np.min(self.psi_m_star[:,i,1]):
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                self.case_marker[n,i] = 'aa'
                                
            elif self.psi_m_star[i,i,1] != np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] == 1:
                    # case b where the domestic threshold is not the smallest one for the origin, and the smallest one is unique
                    # country i patents first abroad
                    print(f'{origin},case b')
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_b[i,1]
                    self.case_marker[i,i] = 'b'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            if self.psi_m_star[n,i,1] < self.psi_o_star_with_prod_patent_b[i,1]:
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star[n,i,1]
                                self.case_marker[n,i] = 'bb1'
                                
                            elif self.psi_m_star[n,i,1] >= self.psi_o_star_with_prod_patent_b[i,1]:
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                self.case_marker[n,i] = 'bb2'


        # ii)
        num_bracket = self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]*(
            1-np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k)
            ) + self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:]*(
                np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k)
                )
        
        self.mult_val_pat = 1 + (
            num_bracket.sum(axis=0) - np.diagonal(num_bracket)
            )/( np.diagonal(self.profit[...,1]) * (1/(self.G[1]+p.delta[:,1]-p.nu[1])-1/(self.G[1]+p.delta[:,1])) )
        
        
        # iii)
        
        self.V_with_prod_patent = np.zeros((p.N,p.S))
        
        A1 = ((p.k/(p.k-1))*self.V_NP[...,1]/self.w[None,:]).sum(axis=0)
        A2 = np.einsum('ni,ni,i->i',
                        self.V_P[...,1]/self.w[None,:] - self.V_NP[...,1]/self.w[None,:],
                        self.psi_m_star[...,1]**(1-p.k),
                        self.mult_val_pat
                        )*(p.k/(p.k-1))
        A3 = - np.einsum('ni,n,n,i->i',
                          self.psi_m_star[...,1]**-p.k,
                          self.w,
                          p.r_hjort,
                          1/self.w
                          )*p.fe[1]
        B = self.psi_o_star[:,1]**-p.k*p.fo[1]*p.r_hjort
        self.V_with_prod_patent[...,1] = (A1+A2+A3-B)*self.w[None,:]
        
        self.mult_val_all_innov = self.V_with_prod_patent[...,1]/self.V[...,1]

    def compute_share_of_exports_patented(self,p):
        A = np.einsum('ni,ni,ni,ni->ni',
                        self.PSI_MPD[...,1]+self.PSI_MPND[...,1],
                        1/self.PSI_M[...,1],
                        1/(1+p.tariff[...,1]),
                        self.X_M[...,1]
                        )
        B = np.einsum('ni,ni->ni',
                        1/(1+p.tariff[...,1]),
                        self.X[...,1]
                        )
        self.share_of_exports_patented = (A.sum(axis=0)-np.einsum('ii->i',
                                                                  A)
                                          )/(B.sum(axis=0)-np.einsum('ii->i',
                                                                     B))
    
    def compute_average_mark_up(self,p):
        prefactor = p.sigma[1:]/(p.sigma[1:]-1)
        A = self.X_M[:,:,1:]/(1+p.tariff[:,:,1:])
        B = self.X_CD[:,:,1:]/(1+p.tariff[:,:,1:])
        
        self.sectoral_average_markup = np.einsum(
            's,is,is->is',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
            ) / np.einsum(
                's,is,is->is',
                prefactor,
                np.einsum('nis->is',B),
                1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
                )
        
        self.aggregate_average_markup = np.einsum(
            's,is,i->i',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
            ) / np.einsum(
                's,is,i->i',
                prefactor,
                np.einsum('nis->is',B),
                1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
                )                                                                 
                                                                     
    def compute_non_solver_quantities(self,p):
        self.compute_tau(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_pflow(p)      
        self.compute_share_of_innovations_patented(p)
        self.compute_welfare(p)
        self.compute_non_solver_aggregate_qualities(p)
        self.compute_semi_elast_patenting_delta(p)
        self.compute_V(p)
        self.compute_average_mark_up(p)
        
    def compute_consumption_equivalent_welfare(self,p,baseline):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline.g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline.cons
                
    def compute_world_welfare_changes(self,p,baseline):
        one_ov_gamma = 1/p.gamma
        numerator = (p.labor**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = (p.labor**one_ov_gamma*baseline.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_pop_average_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
        numerator = (baseline.cons**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = baseline.cons.sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_negishi_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
    def compute_world_welfare_changes_custom_weights(self,p,baseline,weights):
        one_ov_gamma = 1/p.gamma
        numerator = (weights**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = (weights**one_ov_gamma*baseline.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_custom_weights_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
    def compute_one_country_welfare_change(self,p,baseline_cons_country,baseline_g):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline_g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline_cons_country
                
class var_double_diff_double_delta:
    def __init__(self, context, N = 7, S = 2):
        self.off_diag_mask = np.ones((N,N,S),bool).ravel()
        self.off_diag_mask[np.s_[::(N+1)*S]] = False
        self.off_diag_mask[np.s_[1::(N+1)*S]] = False
        self.off_diag_mask = self.off_diag_mask.reshape((N,N,S))
        self.diag_mask = np.invert(self.off_diag_mask)
        self.context = context

    def guess_profit(self, profit_init):
        self.profit = profit_init    

    def guess_wage(self, w_init):
        self.w = w_init

    def guess_Z(self, Z_init):
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
    def var_from_vector(vec,p,context,compute = True):
        init = var_double_diff_double_delta(context=context)    
        init.guess_wage(vec[0:p.N])
        init.guess_Z(vec[p.N:p.N+p.N])
        init.guess_labor_research(
            np.insert(vec[p.N+p.N:p.N+p.N+p.N*(p.S-1)].reshape((p.N, p.S-1)), 0, np.zeros(p.N), axis=1)
            )
        init.guess_profit(
            np.insert(vec[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2*(p.S-1)].reshape((p.N, p.N, p.S-1)), 0, np.zeros(p.N), axis=2)
            )
        init.guess_phi(vec[p.N+p.N+p.N*(p.S-1)+p.N**2*(p.S-1):].reshape((p.N, p.N, p.S)))
        if compute:
            init.compute_solver_quantities(p)
        return init

    def vector_from_var(self):
        w = self.w
        l_R = self.l_R[...,1:].ravel()
        profit = self.profit[...,1:].ravel()
        Z = self.Z
        phi = self.phi.ravel()
        vec = np.concatenate((w,Z,l_R,profit,phi), axis=0)
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
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()
        self.r = p.rho + self.g/p.gamma
        self.G = self.r+p.zeta-self.g+self.g_s+p.nu+p.nu_tilde
        
    def compute_patenting_thresholds(self, p):
        A = np.einsum('n,n,s,i,i->nis',
                               self.w,
                               p.r_hjort,
                               p.fe[1:],
                               1/self.w,
                               1/p.r_hjort,
                               )
        
        denom_bracket = 1/(self.G[None,None,:]+p.delta_eff-p.nu[None,None,:]-p.nu_tilde[None,None,:])-1/(self.G[None,None,:]+p.delta_eff)
        self.psi_C = np.full((p.N,p.N,p.S),np.inf)
        self.psi_C[...,1:] = A*p.r_hjort[None,:,None]/(self.profit[...,1:]*denom_bracket[...,1:])
        self.psi_star = np.maximum(self.psi_C,1)
        psi_star_n_star = np.min(self.psi_star,axis=0)
        
        x_old = np.max(self.psi_C[...,1:], axis=0)
        x_new = None
        cond = True
        it = 0
        while cond:
            if it>0:
                x_old = x_new
            mask = x_old[None,:,:]>=self.psi_C[...,1:]
            x_new = (np.sum(A,axis=0,where=mask)+p.fo[None,1:])/np.sum(A/self.psi_C[...,1:],axis=0,where=mask)
            cond = np.any(x_old != x_new)
            it+=1

        condition = np.maximum(A*(psi_star_n_star[None,:,1:]/self.psi_C[...,1:]-1),0).sum(axis=0)>=p.fo[None,1:]
        x_new[condition] = psi_star_n_star[...,1:][condition]
        self.psi_o_star = np.full((p.N,p.S),np.inf)
        self.psi_o_star[...,1:] = x_new
        self.psi_m_star = np.full((p.N,p.N,p.S),np.inf)
        self.psi_m_star[...,1:] = np.maximum(self.psi_o_star[None,:,1:],self.psi_star[...,1:])

    def compute_aggregate_qualities(self, p):
        prefact = p.k * p.eta * self.l_R**(1-p.kappa) /(p.k-1)
        A = 1/(self.g_s[1:] + p.nu[1:] + p.nu_tilde[1:] + p.zeta[1:])
        B = self.psi_m_star[...,1:]**(1-p.k[None,None,1:])/(self.g_s[None, None, 1:]+p.delta_eff[...,1:]+p.zeta[None,None,1:]+ p.nu[None,None,1:] + p.nu_tilde[None,None,1:])
        C1 = p.nu[None,None,1:]/(self.g_s[None,None, 1:]+p.delta_eff[...,1:]+p.zeta[None,None, 1:]+ p.nu_tilde[None,None,1:])
        C2 = p.nu_tilde[None,None,1:]/(self.g_s[None,None, 1:]+p.delta_eff[...,1:]+p.zeta[None,None, 1:])
        self.PSI_M = np.zeros((p.N,p.N,p.S))
        self.PSI_M[...,1:] = np.einsum('is,nis -> nis',
                               prefact[...,1:],
                               A[None, None, :]+B*(C1+C2+C1*C2))
        
        prefact_CL = (p.k * p.eta * self.l_R**(1-p.kappa) /(p.k-1))*(p.nu / (self.g_s + p.nu_tilde + p.zeta))

        D = (1 - self.psi_m_star[...,1:]**(1 - p.k[None, None, 1:])) / \
            (self.g_s[None, None, 1:] + p.nu[None, None, 1:] + p.nu_tilde[None, None, 1:] + p.zeta[None, None, 1:])
        
        E1 = 1/(self.g_s[None, None, 1:] + p.nu[None, None, 1:] + p.nu_tilde[None, None, 1:] + p.zeta[None, None, 1:])
        E2 = 1/(self.g_s[None, None, 1:] + p.delta_eff[...,1:] + p.nu_tilde[None, None, 1:] + p.zeta[None, None, 1:])
        E3 = p.delta_eff[...,1:] * self.psi_m_star[...,1:]**(1 - p.k[None, None, 1:]) / \
             (self.g_s[None, None, 1:] + p.delta_eff[...,1:] + p.nu[None, None, 1:] + p.nu_tilde[None, None, 1:] + p.zeta[None, None, 1:])
        
        self.PSI_CL = np.zeros((p.N, p.N, p.S))
        self.PSI_CL[...,1:] = np.einsum('is,nis -> nis',
                                        prefact_CL[...,1:],
                                        D + (E1 + E2) * E3)

        
        self.PSI_CD = np.ones((p.N,p.S))
        self.PSI_CD[...,1:] = 1-(self.PSI_M[...,1:]+self.PSI_CL[...,1:]).sum(axis=1)

    def compute_sectoral_prices(self, p):
        power = p.sigma-1
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, 1:] \
            * (self.PSI_M[...,1:]*self.phi[...,1:]**power[None, None, 1:]).sum(axis=1)

        B = self.PSI_CD[...,1:]*(self.phi[...,1:]**p.theta[None,None,1:]).sum(axis=1)**(power/p.theta)[None, 1:]
        
        C  = (self.PSI_CL[...,1:]*self.phi[...,1:]**power[None, None, 1:]).sum(axis=1)

        self.P_M = np.full((p.N, p.S),np.inf)
        self.P_M[:,1:] = (A/(A+B+C))**(1/(1-p.sigma))[None, 1:]
        
        self.P_CD = np.ones((p.N, p.S))
        self.P_CD[:,1:] = (B/(A+B+C))**(1/(1-p.sigma))[None, 1:]
        
        self.P_CL = np.full((p.N, p.S),np.inf)
        self.P_CL[:,1:] = (C/(A+B+C))**(1/(1-p.sigma))[None, 1:]
        
    def compute_labor_allocations(self, p):
        self.l_Ae = np.zeros((p.N,p.N,p.S))
        self.l_Ae[...,1:] = np.einsum('n,s,is,is,nis -> ins',
                         p.r_hjort,
                         p.fe[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         self.psi_m_star[...,1:]**-p.k[None,None,1:]
                         )
        self.l_Ao = np.zeros((p.N,p.S))
        self.l_Ao[...,1:] = np.einsum('i,s,is,is,is -> is',
                         p.r_hjort,
                         p.fo[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         self.psi_o_star[...,1:]**-p.k[None,1:]
                         )
        self.l_P = p.labor-(self.l_Ao+self.l_R+self.l_Ae.sum(axis=0)).sum(axis=1)
        
    def compute_price_indices(self, p, assign = True):
        power = (p.sigma-1)
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**power[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(power/p.theta)[None, :]
        C = (self.PSI_CL * self.phi**power[None, None, :]).sum(axis=1)
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B+C))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :]/(p.sigma[None, :]-1)) ).prod(axis=1)
        if assign:
            self.price_indices = price_indices
        else:
            return price_indices
        
    def compute_trade_flows_and_shares(self, p, assign = True):
            temp = (self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
            X_M = np.zeros((p.N, p.N, p.S))
            X_M[...,1:] = np.einsum('nis,nis,ns,ns,s,n->nis',
                                    self.phi[..., 1:]**(p.sigma-1)[None, None, 1:],
                                    self.PSI_M[..., 1:],
                                    1/temp,
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:]),
                                    p.beta[1:],
                                    self.Z
                                    )
            
            X_CL = np.zeros((p.N, p.N, p.S))
            X_CL[...,1:] = np.einsum('nis,nis,ns,ns,s,n->nis',
                                    self.phi[..., 1:]**(p.sigma-1)[None, None, 1:],
                                    self.PSI_CL[..., 1:],
                                    1/temp,
                                    self.P_CL[..., 1:]**(1-p.sigma[None, 1:]),
                                    p.beta[1:],
                                    self.Z
                                    )
            
            X_CD = np.einsum('nis,ns,ns,s,n->nis',
                                        self.phi**(p.theta)[None,None,:],
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:]),
                                        p.beta,
                                        self.Z
                                        )
            X = X_M+X_CD+X_CL
            if assign:
                self.X_M = X_M
                self.X_CD = X_CD
                self.X_CL = X_CL
                self.X = X
            else:
                return X_M,X_CD,X_CL
        
    def compute_solver_quantities(self,p):
        self.compute_growth(p)
        self.compute_patenting_thresholds(p)
        self.compute_aggregate_qualities(p)
        self.compute_sectoral_prices(p)
        self.compute_labor_allocations(p)
        self.compute_trade_flows_and_shares(p)
        self.compute_price_indices(p)

    def compute_wage(self, p):
        wage = (p.alpha[None, :] * ((self.X - self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
                ).sum(axis=1)/self.l_P
        return wage
            
    def compute_labor_research(self, p):
        A1 = ((p.k[None,None,1:]/(p.k[None,None,1:]-1))*self.profit[...,1:]/self.G[None,None,1:]).sum(axis=0)
        A2 = np.einsum('nis,n,s,n,i,nis->is',
                       self.psi_m_star[...,1:]**-p.k[None,None,1:],
                       self.w,
                       p.fe[1:],
                       p.r_hjort,
                       1/self.w,
                       p.k[None,None,1:]*self.psi_m_star[...,1:]/(self.psi_C[...,1:]*(p.k[None,None,1:]-1))-1
                       )
        B = self.psi_o_star[:,1:]**-p.k[None,1:]*p.fo[None,1:]*p.r_hjort[:,None]
        l_R = np.zeros((p.N,p.S))
        l_R[...,1:] = (p.eta[...,1:]*(A1+A2-B))**(1/p.kappa)
        # assert np.isnan(l_R).sum() == 0, 'nan in l_R'
        return l_R
    
    def compute_profit(self,p):
        profit = np.zeros((p.N,p.N,p.S))
        profit[...,1:] = np.einsum('nis,s,i,nis,nis->nis',
                                self.X_M[...,1:],
                                1/p.sigma[1:],
                                1/self.w,
                                1/self.PSI_M[...,1:],
                                1/(1+p.tariff[...,1:]))
        return profit
    
    def compute_expenditure(self, p):
        A1 = np.einsum('nis,nis->i', 
                      self.X,
                      1/(1+p.tariff))
        A2 = np.einsum('ins,ins,ins->i', 
                      self.X,
                      p.tariff,
                      1/(1+p.tariff))
        B = np.einsum('i,nis->i', self.w, self.l_Ae)
        C = p.deficit_share_world_output*np.einsum('nis,nis->', 
                      self.X,
                      1/(1+p.tariff))
        D = np.einsum('n,ins->i', self.w, self.l_Ae)
        Z = (A1+A2+B-(C+D))
        return Z
    
    def compute_phi(self, p):
        if self.context == 'calibration':
            denominator_M = np.zeros((p.N, p.N, p.S))
            denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                    self.PSI_M[..., 1:],
                                    self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                    1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                    self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                    )
            denominator_CD = np.einsum('ns,ns->ns',
                                        1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                        self.P_CD**(1-p.sigma[None,:])
                                        )
            f_phi = np.einsum('nis,nis,nis->nis',
                            p.trade_shares,
                            1+p.tariff,
                            1/(denominator_M + denominator_CD[:,None,:]))
            
            phi = np.einsum('nis,nns,ns,ns,ns->nis',
                    f_phi**(1/p.theta)[None,None,:],
                    f_phi**(-1/p.theta)[None,None,:],
                    p.T**(1/p.theta[None,:]),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
    
            return phi
        
        elif self.context == 'counterfactual':
            # phi = np.einsum('is,nis,is,is->nis',
            #         p.T**(1/p.theta[None,:]),
            #         1/p.tau,
            #         self.w[:,None]**(-p.alpha[None,:]),
            #         self.price_indices[:,None]**(p.alpha[None,:]-1))
            # return phi
            phi = np.einsum('is,nis,nis,is,is->nis',
                    p.T**(1/p.theta[None,:]),
                    1/p.tau,
                    1/(1+p.tariff),
                    self.w[:,None]**(-p.alpha[None,:]),
                    self.price_indices[:,None]**(p.alpha[None,:]-1))
            return phi
        
        else:
            print('context attribute needs to be either "calibration" or "counterfactual"')
    
    def check_phi(self,p):
        denominator_M = np.zeros((p.N, p.N, p.S))
        denominator_M[..., 1:] = np.einsum('nis,nis,ns,ns->nis',
                                self.PSI_M[..., 1:],
                                self.phi[..., 1:]**((p.sigma-1)-p.theta)[None, None, 1:],
                                1/((self.PSI_M[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)),
                                self.P_M[..., 1:]**(1-p.sigma[None, 1:])
                                )
        denominator_CD = np.einsum('ns,ns->ns',
                                    1/(self.phi**(p.theta)[None,None,:]).sum(axis=1),
                                    self.P_CD**(1-p.sigma[None,:])
                                    )
        one_over_denominator = 1/(denominator_M + denominator_CD[:,None,:])
        phi = np.einsum('nis,s,n,nis->nis',
                        self.X,
                        1/p.beta,
                        1/self.Z,
                        one_over_denominator)**(1/p.theta)[None,None,:]
        return self.phi/phi
    
    # def scale_tau(self,p):
        
    #     self.phi = self.phi\
    #         *np.einsum('ns,ns,ns->ns',
    #             p.T**(1/p.theta[None,:]),
    #             self.w[:,None]**(-p.alpha[None,:]),
    #             self.price_indices[:,None]**(p.alpha[None,:]-1))[:,None,:]\
    #         /np.einsum('nns->ns',self.phi)[:,None,:]
    
    def compute_tau(self,p, assign = True):
        tau = np.einsum('is,nis,nis,is,is->nis',
                        p.T**(1/p.theta[None,:]),
                        1/self.phi,
                        1/(1+p.tariff),
                        self.w[:,None]**-p.alpha[None,:],
                        self.price_indices[:,None]**(p.alpha[None,:]-1),
                        )
        if assign:
            self.tau = tau
        else:
            return tau
    
    def scale_P(self, p):
        try:
            numeraire = self.price_indices[0]
        except:
            self.compute_solver_quantities(p)
            numeraire = self.price_indices[0]
        
        self.w = self.w / numeraire
        self.Z = self.Z / numeraire
        self.X = self.X / numeraire
        self.X_CD = self.X_CD / numeraire
        self.X_M = self.X_M / numeraire
        self.phi = self.phi * numeraire
        self.price_indices = self.price_indices / numeraire
        self.compute_sectoral_prices(p)
    
    def compute_nominal_value_added(self,p):
        self.nominal_value_added = p.alpha[None, :]*((self.X-self.X_M/p.sigma[None, None, :])/(1+p.tariff)).sum(axis=0)
    
    def compute_nominal_intermediate_input(self,p):
        self.nominal_intermediate_input = np.einsum('s,is->is',
                           (1-p.alpha)/p.alpha,
                           self.nominal_value_added)
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - self.nominal_intermediate_input.sum(axis=1)
        self.cons = self.nominal_final_consumption/self.price_indices
        
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
            * (self.PSI_M * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
        B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        self.sectoral_price_indices = one_over_price_indices_no_pow_no_prod**(1/(p.sigma[None, :]-1))
        self.sectoral_cons = np.einsum('s,n,ns->ns',
                                  p.beta,
                                  self.Z,
                                  1/self.sectoral_price_indices
                                  )
        
    def compute_gdp(self,p):
        self.gdp = self.nominal_final_consumption + \
            p.deficit_share_world_output*np.einsum('nis,nis->',
                                                   self.X,
                                                   1/(1+p.tariff)
                                                   ) + \
            self.w*np.einsum('is->i',
                             self.l_R + self.l_Ao
                             ) + \
            np.einsum('n,ins->i',
                      self.w,
                      self.l_Ae)

    def compute_pflow(self,p):
        self.pflow = np.einsum('nis,is,is->nis',
                              self.psi_m_star[...,1:]**(-p.k[None,None,1:]),
                              p.eta[...,1:],
                              self.l_R[...,1:]**(1-p.kappa)
                              ).squeeze()
        
    def compute_share_of_innovations_patented(self,p):
        self.share_innov_patented = self.psi_m_star[...,1:]**(-p.k[None,None,1:])
    
    def compute_welfare(self,p):
        # exp = 1-1/p.gamma
        # self.U = self.cons**(exp)/(p.rho-self.g*exp)/exp
        pass

    def compute_non_solver_aggregate_qualities(self,p): 
        self.PSI_MPND = np.zeros((p.N,p.N,p.S))
        self.PSI_MPD = np.zeros((p.N,p.N,p.S))
        self.PSI_MPL = np.zeros((p.N,p.N,p.S))
        self.PSI_MNP = np.zeros((p.N,p.N,p.S))
        prefact = p.k[None,1:] * p.eta[...,1:] * self.l_R[...,1:]**(1-p.kappa)/(p.k[None,1:]-1)
        A = (self.g_s[1:] + p.nu[1:] + p.nu_tilde[1:] + p.zeta[1:])
        self.PSI_MPND[...,1:] = np.einsum('is,nis,nis->nis',
                                  prefact,
                                  self.psi_m_star[...,1:]**(1-p.k[None,None,1:]),
                                  1/(A[None,None,:]+p.delta_eff[...,1:]))
        self.PSI_MPL[...,1:] = np.einsum('s,nis,nis->nis',
                                 p.nu[1:],
                                 self.PSI_MPND[...,1:],
                                 1/(p.delta_eff[...,1:]+self.g_s[None,1:]+p.zeta[None,1:]+p.nu_tilde[None,None,1:]))
        self.PSI_MPD[...,1:] = np.einsum('s,nis,nis->nis',
                                 p.nu_tilde[1:],
                                 self.PSI_MPND[...,1:]+self.PSI_MPL[...,1:],
                                 1/(p.delta_eff[...,1:]+self.g_s[None,None,1:]+p.zeta[None,None,1:]))
        numerator_A = np.einsum('is,nis->nis',
                                prefact,
                                1-self.psi_m_star[...,1:]**(1-p.k[None,None,1:]))
        numerator_B= np.einsum('nis,nis->nis',
                               p.delta_eff[...,1:],
                               self.PSI_MPND[...,1:])
        self.PSI_MNP[...,1:] = (numerator_A + numerator_B)/A[None,None,:]
    
    def compute_V(self,p):
        self.V_NP = np.einsum('nis,i,s->nis',
                              self.profit,
                              self.w,
                              1/self.G
                              )
        self.V_PD = np.einsum('nis,i,nis->nis',
                              self.profit,
                              self.w,
                              1/(self.G[None,None,:]-p.nu[None,None,:]-p.nu_tilde[None,None,:]+p.delta_eff)
                              )
        
        self.V_P = np.einsum('nis,i,nis->nis',
                             self.profit,
                             self.w,
                             1/(self.G[None,None,:]-p.nu[None,None,:]-p.nu_tilde[None,None,:]+p.delta_eff)\
                                 -1/(self.G[None,None,:]+p.delta_eff)+1/(self.G[None,None,:])
                             )
        
        self.V = np.zeros((p.N,p.S))
        
        A1 = ((p.k[None,None,1:]/(p.k[None,None,1:]-1))*self.V_NP[...,1:]).sum(axis=0)
        A2 = np.einsum('nis,n,s,n,nis->is',
                       self.psi_m_star[...,1:]**-p.k[None,None,1:],
                       self.w,
                       p.fe[1:],
                       p.r_hjort,
                       p.k[None,None,1:]*self.psi_m_star[...,1:]/(self.psi_C[...,1:]*(p.k[None,None,1:]-1))-1
                       )
        B = self.psi_o_star[:,1:]**-p.k[None,1:]*p.fo[None,1:]*p.r_hjort[:,None]*self.w[:,None]
        self.V[...,1:] = A1+A2-B
        
    def compute_quantities_with_prod_patents(self,p,upper_bound_integral = np.inf):
        
        def incomplete_sum_with_exponent(matrix,exponent):
            res = np.full_like(matrix,np.nan)
            for i in range(matrix.shape[1]):
                res[:,i] = (np.delete(matrix,i,axis=1)**exponent).sum(axis=1)
            return res
        
        def upper_inc_gamma(a,x):
            return gamma(a)*gammaincc(a,x)
            # return gammaincc(a,x)
        
        def sim(z):
            bound_A = np.einsum('i,ni->ni',
                                p.T[..., 1],
                                incomplete_sum_with_exponent(
                                    self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1]
                                )/(z**p.theta[1])

            A_bracket_1 = upper_inc_gamma(
                (p.theta[1]+1-p.sigma[1])/p.theta[1],
                bound_A
            )
            A_bracket_2 = upper_inc_gamma(
                (p.theta[1]+1-p.sigma[1])/p.theta[1],
                bound_A*(p.sigma[1]/(p.sigma[1] - 1))**p.theta[1]
            )

            A = np.einsum('ni,i,ni->ni',
                          (incomplete_sum_with_exponent(self.phi[..., 1], p.theta[1])
                           / self.phi[..., 1]**p.theta[1])**((p.sigma[1]-1)/p.theta[1]),
                          p.T[..., 1]**(-1/p.theta[1]),
                          A_bracket_1 - A_bracket_2
                          )

            bound_B = np.einsum('i,ni->ni',
                                p.T[..., 1],
                                incomplete_sum_with_exponent(
                                    self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1]
                                )/(z**p.theta[1])

            B_bracket_1 = upper_inc_gamma(
                (p.theta[1]-p.sigma[1])/p.theta[1],
                bound_B
            )
            B_bracket_2 = upper_inc_gamma(
                (p.theta[1]-p.sigma[1])/p.theta[1],
                bound_B*(p.sigma[1]/(p.sigma[1] - 1))**p.theta[1]
            )

            B = np.einsum('ni,ni->ni',
                          (incomplete_sum_with_exponent(
                              self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1])**(p.sigma[1]/p.theta[1]),
                          B_bracket_1 - B_bracket_2
                          )/z

            return A-B
        
        # integral_calculated = np.full_like(self.phi[...,1],np.nan)
        
        # def integrand(z,i,j):
        #     return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*sim(z) )[i,j]
        
        # for i in range(p.N):
        #     for j in range(p.N):
        #         print(integrate.quad(lambda x: integrand(x,i,j), 0, upper_bound_integral,full_output=1))
        #         integral_calculated[i,j] = integrate.quad(lambda x: integrand(x,i,j), 0, upper_bound_integral,full_output=1).y
        
        def integrand(z):
            return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*sim(z) )
            # return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][:,None]*z**(-p.theta[1]))*sim(z) )
        
        # c1 = 0
        # # c2 = 0
        # fig,ax=plt.subplots(figsize = (12,8))
        # for c2 in range(11):
        #     ax.plot(np.logspace(0,2,1001),[sim(z)[c1,c2] for z in np.logspace(0,2,1001)],
        #              label=p.countries[c1]+'_'+p.countries[c2])
        # plt.legend()
        # plt.title('sim(z))')
        # plt.xscale('log')
        # plt.show()
        
        self.integral_result = integrate.quad_vec(lambda x: integrand(x), 0, upper_bound_integral,full_output=1)
        integral_calculated = self.integral_result[0]
        
        A = (1 +
             (incomplete_sum_with_exponent(
                 self.phi[..., 1], p.theta[1])/self.phi[..., 1]**p.theta[1])
             * (p.sigma[1]/(p.sigma[1]-1))**p.theta[1]
             )**((p.sigma[1] - p.theta[1] - 1)/p.theta[1])
        
        B = np.einsum('i,ni->ni',
                        p.T[...,1]**((1+p.theta[1])/p.theta[1]),
                      # p.T[...,1]**((1)/p.theta[1]),
                      integral_calculated
                      )*p.theta[1]*p.sigma[1]**p.sigma[1]/(
                          (p.sigma[1]-1)**(p.sigma[1]-1)*gamma((p.theta[1]+1-p.sigma[1])/p.theta[1])
                          )
        
        self.profit_with_prod_patent = np.zeros_like(self.profit)
        self.profit_with_prod_patent[...,1] = self.profit[...,1]*(A+B)
        
        for i,country in enumerate(p.countries):
            self.profit_with_prod_patent[i,i,1] = 0
        
        self.profit_with_prod_patent_D = np.zeros_like(self.profit)
        self.profit_with_prod_patent_D[...,1] = self.profit[...,1]*(
            self.phi[...,1]**p.theta[1]/(self.phi[...,1]**p.theta[1]).sum(axis=1)[:,None]
            )**((-p.sigma[1]+p.theta[1]+1)/p.theta[1])
        
        # self.profit_with_prod_patent_D_bis = np.zeros_like(self.profit)
        # self.profit_with_prod_patent_D_bis[...,1] = ((p.sigma[1]-1)**(p.sigma[1]-1)/p.sigma[1]**p.sigma[1]
        #                                              )*np.einsum('ni,n,i->ni',
        #                                                          self.X_CD[...,1],
        #                                                          1/self.PSI_CD[...,1],
        #                                                          1/self.w
        #                                                          )
        
        # # alternative way of computing with direct integration on p
        
        # A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] \
        #     * (self.PSI_M * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
        # B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        # temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A+B))
        # one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        # sectoral_price_indices = one_over_price_indices_no_pow_no_prod**(1/(p.sigma[None, :]-1))
        
        # sectoral_cons = np.einsum('s,n,ns->ns',
        #                           p.beta,
        #                           self.Z,
        #                           1/sectoral_price_indices
        #                           )
                                                                 
        # self.Pr_E_1_over_psi = np.zeros_like(self.profit)
        # self.Pr_E_1_over_psi[...,1] = np.einsum('i,,,n,n,ni,,ni,i,i->ni',
        #                                   p.T[...,1],
        #                                   (p.sigma[1]-1)**p.theta[1],
        #                                   1/(p.sigma[1])**(p.theta[1]+1),
        #                                   sectoral_price_indices[...,1]**p.sigma[1],
        #                                   sectoral_cons[...,1],
        #                                   ( self.phi[...,1]**p.theta[1] * ((p.sigma[1]-1)/p.sigma[1])**p.theta[1] + incomplete_sum_with_exponent(self.phi[...,1],p.theta[1])
        #                                       )**( (p.sigma[1] - p.theta[1] - 1)/p.theta[1] ),
        #                                   gamma( (p.theta[1]+1-p.sigma[1])/p.theta[1] ),
        #                                   p.tau[...,1]**-p.theta[1],
        #                                   self.w**-(p.theta[1]*p.alpha[1]),
        #                                   self.price_indices**(-p.theta[1]*(1-p.alpha[1]))
        #                                   )
        
        # def p_integrand_for_Pr_E_2_over_psi(x,z,i,j):
        #     # A = p**(p.theta[1]-p.sigma[1])*np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * p**p.theta[1])
        #     # B = np.einsum('',
        #     #                 p.tau[...,1],
        #     #                 p.w**p.alpha[1],
        #     #                 self.price_indices**(1-p.alpha[1]),
        #     #                 p**(p.theta[1]-p.sigma[1]-1),
        #     #                 np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * p**p.theta[1])
        #     #                 )/z
        #     # return (A-B)[i,j]
        #     res = np.einsum('ni,,ni->ni',
        #                     x - np.einsum('ni,i,i->ni',
        #                             p.tau[...,1],
        #                             self.w**p.alpha[1],
        #                             self.price_indices**(1-p.alpha[1]))/z,
        #                     x**(p.theta[1]-p.sigma[1]-1),
        #                     np.exp(-incomplete_sum_with_exponent(self.phi[...,1],p.theta[1]) * x**p.theta[1])
        #                     )*p.theta[1]/z
        #     return res[i,j]
        
        # # from tqdm import tqdm
        
        # def p_integral_for_Pr_E_2_over_psi(z):
        #     p_lb = np.einsum('ni,i,i->ni',
        #                     p.tau[...,1],
        #                     self.w**p.alpha[1],
        #                     self.price_indices**(1-p.alpha[1]),
        #                     )/z
        #     p_ub = p.sigma[1]*p_lb/(p.sigma[1]-1)
        #     p_integral_calculated = np.zeros_like(self.profit)
        #     for i in range(p.N):
        #         for j in range(p.N):
        #             p_integral_calculated[i,j,1] = integrate.quad(p_integrand_for_Pr_E_2_over_psi, p_lb[i,j], p_ub[i,j], args=(z,i,j))[0]
        #     return p_integral_calculated
        
        # # def second_p_integrand_for_Pr_E_2_over_psi(z):
        # #     pass
        
        # # def second_p_integral_for_Pr_E_2_over_psi():
        # #     pass
            
        # def z_integrand_for_Pr_E_2_over_psi(z):
        #     return ( z**(-p.theta[1]-1)*np.exp(-p.T[...,1][None,:]*z**(-p.theta[1]))*p_integral_for_Pr_E_2_over_psi(z)[...,1] )
        
        # def z_integral_for_Pr_E_2_over_psi():
        #     return integrate.quad_vec(z_integrand_for_Pr_E_2_over_psi, 0, np.inf, full_output=1)[0]
            
                                                         
        # self.Pr_E_2_over_psi = np.zeros_like(self.profit)
        # self.Pr_E_2_over_psi[...,1] = np.einsum('i,,n,n,ni,ni->ni',
        #                                   p.T[...,1],
        #                                   p.theta[1],
        #                                   sectoral_price_indices[...,1]**p.sigma[1],
        #                                   sectoral_cons[...,1],
        #                                   z_integral_for_Pr_E_2_over_psi(),
        #                                   incomplete_sum_with_exponent(self.phi[...,1],p.theta[1])
        #                                   )
                                                                 
        # self.profit_with_prod_patent_with_p_integral = (self.Pr_E_1_over_psi+self.Pr_E_2_over_psi)/self.w[None,:,None]
        
        # # end alternative way of computing with direct integration on p
        
        self.V_NP_P_minus_V_NP_NP_with_prod_patent = np.zeros_like(self.profit)
        self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1] = \
            self.profit_with_prod_patent[...,1]*(
                1/(self.G[1]+p.delta[:,1]-p.nu[1])-1/(self.G[1]+p.delta[:,1])
                )[None,:]*self.w[None,:]
            
        for i,country in enumerate(p.countries):
            self.V_NP_P_minus_V_NP_NP_with_prod_patent[i,i,1] = 0
        
        self.V_P_P_minus_V_P_NP_with_prod_patent = np.zeros_like(self.profit)
        self.V_P_P_minus_V_P_NP_with_prod_patent[...,1] = \
            self.profit_with_prod_patent[...,1]*(
                1/(self.G[1]+p.delta[None,:,1]-p.nu[1])-1/(self.G[1]+p.delta[None,:,1]) \
                    - 1/(self.G[1]+p.delta[None,:,1]+p.delta[:,None,1]-p.nu[1]) + 1/(self.G[1]+p.delta[None,:,1]+p.delta[:,None,1])
                )*self.w[None,:]
        
        for i,country in enumerate(p.countries):
            self.V_P_P_minus_V_P_NP_with_prod_patent[i,i,1] = 0
        
        # i)
        # case a
        
        self.psi_o_star_with_prod_patent_a = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_a = np.full_like(self.psi_o_star,np.inf)
        
        denom_A = np.diagonal(self.V_P[...,1]-self.V_NP[...,1])/self.w
        denom_B = (self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]).sum(axis=0)-np.diagonal(
            self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:])
        denom = denom_A + denom_B
        
        self.psi_o_star_with_prod_patent_a[...,1] = p.r_hjort*(p.fe[1] + p.fo[1])/denom
        self.psi_o_star_without_prod_patent_a[...,1] = p.r_hjort*(p.fe[1] + p.fo[1])/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_a = self.psi_o_star_with_prod_patent_a**-p.k
        self.share_innov_patented_dom_without_prod_patent_a = self.psi_o_star_without_prod_patent_a**-p.k
        
        # case b
        
        self.psi_o_star_with_prod_patent_b = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_b = np.full_like(self.psi_o_star,np.inf)
        
        mask_B = np.diagonal(self.psi_m_star[...,1])[None,:]<self.psi_m_star[...,1]
        mask_C = np.diagonal(self.psi_m_star[...,1])[None,:]>self.psi_m_star[...,1]
        
        denom_A = np.diagonal(self.V_P[...,1]-self.V_NP[...,1])/self.w
        denom_B = (mask_B*(self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:])).sum(axis=0)
        denom_C = (mask_C*(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:])).sum(axis=0)
        denom = denom_A + denom_B + denom_C
        
        self.psi_o_star_with_prod_patent_b[...,1] = p.r_hjort*p.fe[1]/denom
        self.psi_o_star_without_prod_patent_b[...,1] = p.r_hjort*p.fe[1]/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_b = self.psi_o_star_with_prod_patent_b**-p.k
        self.share_innov_patented_dom_without_prod_patent_b = self.psi_o_star_without_prod_patent_b**-p.k
        
        # check on ib)
        
        # check_b_lhs = self.psi_o_star_with_prod_patent_b[...,1]
        # check_b_rhs = np.min(self.psi_m_star[...,1],axis=0)
        
        for i,country in enumerate(p.countries):
            if np.argmin(self.psi_m_star[:,i,1]) != i and (
                    self.psi_m_star[:,i,1]==np.min(self.psi_m_star[:,i,1])).sum() == 1:
                print(f'check b for {country}')
                if self.psi_o_star_with_prod_patent_b[i,1] > np.min(self.psi_m_star[:,i,1]):
                    print('passed')
                else:
                    print('not passed')    
                    
                print(f'check b for order for {country}')
                if len([p.countries[x]
                         for x in np.where(self.psi_m_star[:, i, 1] < self.psi_m_star[i, i, 1])[0]
                         if x != i]
                        ) == len([p.countries[x]
                                 for x in np.where(self.psi_m_star[:, i, 1] < self.psi_o_star_with_prod_patent_b[i, 1])[0]
                                 if x != i]
                                ):
                    print('passed, patents in before :',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_m_star[i,i,1])[0] if x!=i],
                          'to after:',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_o_star_with_prod_patent_b[i,1])[0] if x!=i],
                          'countries patent before origin') 
                else:
                    print('not passed, patents in before :',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_m_star[i,i,1])[0] if x!=i],
                          'to after:',
                          [p.countries[x] for x in np.where(self.psi_m_star[:,i,1]<self.psi_o_star_with_prod_patent_b[i,1])[0] if x!=i],
                          'countries patent before origin') 
        
        # case c
        
        self.psi_o_star_with_prod_patent_c = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_without_prod_patent_c = np.full_like(self.psi_o_star,np.inf)
        
        mask_is_n_in_n_star_of_i = np.isclose(self.psi_m_star[...,1],np.min(self.psi_m_star[...,1],axis=0))
        
        denom_A = (mask_is_n_in_n_star_of_i
                   *(self.V_P[...,1]-self.V_NP[...,1])
                   /self.w[None,:]
                   ).sum(axis=0)
        denom_B = (mask_is_n_in_n_star_of_i
                   *(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1])
                   /self.w[None,:]
                   ).sum(axis=0)
        denom_C = (~mask_is_n_in_n_star_of_i*self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
                   ).sum(axis=0)
        denom = denom_A + denom_B + denom_C
        num = (
            (self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:])*mask_is_n_in_n_star_of_i
            ).sum(axis=0) + p.r_hjort*p.fo[1]
    
        self.psi_o_star_with_prod_patent_c[...,1] = num/denom
        self.psi_o_star_without_prod_patent_c[...,1] = num/denom_A
        
        self.share_innov_patented_dom_with_prod_patent_c = self.psi_o_star_with_prod_patent_c**-p.k
        self.share_innov_patented_dom_without_prod_patent_c = self.psi_o_star_without_prod_patent_c**-p.k
        
        # check on ic)
        
        for n, destination in enumerate(p.countries):
            for i, origin in enumerate(p.countries):
                if mask_is_n_in_n_star_of_i[n,i] and mask_is_n_in_n_star_of_i.sum(axis=0)[i]>1 and i!=n:
                    test_mask = mask_is_n_in_n_star_of_i.copy()
                    test_mask = False
                    denom_A = (test_mask
                               *(self.V_P[...,1]-self.V_NP[...,1])
                               /self.w[None,:]
                               ).sum(axis=0)
                    denom_B = (test_mask
                               *(self.V_P_P_minus_V_P_NP_with_prod_patent[...,1])
                               /self.w[None,:]
                               ).sum(axis=0)
                    denom_C = (~test_mask*self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
                               ).sum(axis=0)
                    denom = denom_A + denom_B + denom_C
                    num = (
                        (self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:])*test_mask
                        ).sum(axis=0) + p.r_hjort*p.fo[1]
                    
                    print(f'check c when excluding destination {destination} from origin {origin}')
                    
                    if (self.psi_o_star_with_prod_patent_c[...,1]*denom)[i] > num[i]:
                        print('not passed')
                    else:
                        print('passed')
                        
            
        # iaa)
        
        self.psi_m_star_without_prod_patent_aa = np.full_like(self.psi_m_star,np.inf)
        self.psi_m_star_with_prod_patent_aa = np.full_like(self.psi_m_star,np.inf)
        
        denom_A = (self.V_P[...,1]-self.V_NP[...,1])/self.w[None,:]
        denom_B = self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]
        denom_C = self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:]
        denom = denom_A - denom_B + denom_C
        
        self.psi_m_star_without_prod_patent_aa[...,1] = self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:]/denom_A
        self.psi_m_star_with_prod_patent_aa[...,1] = self.w[:,None]*p.r_hjort[:,None]*p.fe[1]/self.w[None,:]/denom
         
        # icc)
        
        self.psi_o_star_without_prod_patent_cc = self.psi_o_star_without_prod_patent_c.copy()
        
        # icc2)
        
        self.psi_o_star_with_prod_patent_cc2 = np.full_like(self.psi_o_star,np.inf)
        
        def compute_threshold_if_only_patent_domestically_first(country_index):
            res = np.full_like(self.psi_o_star,np.inf)
            
            denom_A = np.diagonal(
                (self.V_P[..., 1]-self.V_NP[..., 1])/self.w[None, :]
            )
            denom_B = ((self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1])
                       /self.w[None,:]
                       ).sum(axis=0)-np.diagonal((self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1])
                                                 /self.w[None,:])
            denom = denom_A + denom_B
            num =  p.r_hjort*(p.fo[1]+p.fe[1])
        
            res = num/denom
            
            return res[country_index]
        
        for i, country in enumerate(p.countries):
            self.psi_o_star_with_prod_patent_cc2[i,1] = compute_threshold_if_only_patent_domestically_first(i)
        
        # icc1)
        
        self.psi_o_star_with_prod_patent_cc1 = np.full_like(self.psi_o_star,np.inf)
        self.psi_o_star_with_prod_patent_cc = np.full_like(self.psi_o_star,np.inf)
        
        def compute_threshold_if_patent_domestically_and_foreign_simultaneously(mask_is_n_in_n_star_of_i,country_index):
            res = np.full_like(self.psi_o_star,np.inf)
            
            # denom_A = np.diagonal(
            #     (self.V_P[..., 1]-self.V_NP[..., 1])/self.w[None, :]
            # )
            # denom_A = 0
            denom_B = (mask_is_n_in_n_star_of_i
                       *(self.V_P_P_minus_V_P_NP_with_prod_patent[:,country_index,1]+self.V_P[:,country_index, 1]-self.V_NP[:,country_index,1])
                       /self.w[country_index]
                       ).sum()
            denom_C = (~mask_is_n_in_n_star_of_i*self.V_NP_P_minus_V_NP_NP_with_prod_patent[:,country_index,1]/self.w[country_index]
                       ).sum()
            denom = denom_B + denom_C
            num = (
                (self.w*p.r_hjort*p.fe[1]/self.w[country_index])*mask_is_n_in_n_star_of_i
                ).sum() + p.r_hjort[country_index]*p.fo[1]
            
            res = num/denom
            
            return res
        
        def subsets(s):
            x = len(s)
            masks = [1 << i for i in range(x)]
            for i in range(1,1 << x):
                yield [ss for mask, ss in zip(masks, s) if i & mask]
                
        self.cc1_min_patenting_combination_by_origin = [[x] for x in p.countries]
                    
        for i, origin in enumerate(p.countries):
            initial_mask_is_n_in_n_star_of_i = self.psi_m_star[:,i,1] == np.min(self.psi_m_star[:,i,1])
            
            countries_to_test = [p.countries[k] for k in np.where(initial_mask_is_n_in_n_star_of_i)[0] if k!=i]
            combinations_of_countries_to_test = list(subsets(countries_to_test))
            self.psi_o_star_with_prod_patent_cc[i,1] = self.psi_o_star_with_prod_patent_cc2[i,1]
            self.psi_o_star_with_prod_patent_cc1[i,1] = np.inf
            self.cc1_min_patenting_combination_by_origin[i] = [origin]
            
            for combination_of_countries in combinations_of_countries_to_test:
                new_mask = np.array([c in combination_of_countries or c==origin for c in p.countries])
                new_threshold = compute_threshold_if_patent_domestically_and_foreign_simultaneously(new_mask,i)
                print(origin,combination_of_countries,new_mask,new_threshold)
                if new_threshold < self.psi_o_star_with_prod_patent_cc1[i,1]:
                    self.psi_o_star_with_prod_patent_cc1[i,1] = new_threshold
                if new_threshold  < self.psi_o_star_with_prod_patent_cc[i,1]:
                    self.psi_o_star_with_prod_patent_cc[i,1] = new_threshold
                    self.cc1_min_patenting_combination_by_origin[i] = combination_of_countries
            
        # gather every change of patenting threshold in one array
        
        self.psi_m_star_with_prod_patent = np.full_like(self.psi_m_star,np.inf)
        self.case_marker = np.empty(self.psi_m_star[...,1].shape, dtype="<U20")
        
        for i,origin in enumerate(p.countries):
            if self.psi_m_star[i,i,1] == np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] == 1:
                    # case a where the domestic threshold is the smallest one for the origin, and it is the only smallest one
                    # country i patents only at home first
                    print(f'{origin},domestic case a')
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_a[i,1]
                    self.case_marker[i,i] = 'a'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                            self.case_marker[n,i] = 'aa'
            
            elif self.psi_m_star[i,i,1] == np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] > 1:
                    # case c where the domestic threshold is the smallest one for the origin, but it is not the only smallest one
                    # country i patents first at home and abroad at the same time
                    print(f'{origin},case cc')
                    # self.psi_m_star_with_prod_patent[i,i,1] = np.minimum(self.psi_o_star_with_prod_patent_cc1[i,1],
                    #                                                       self.psi_o_star_with_prod_patent_cc2[i,1])
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_cc[i,1]
                    self.case_marker[i,i] = 'cc'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            if self.psi_m_star[n,i,1] == np.min(self.psi_m_star[:,i,1]):
                                #case cc1)
                                if p.countries[n] in self.cc1_min_patenting_combination_by_origin[i]:
                                    self.psi_m_star_with_prod_patent[n,i,1] = self.psi_o_star_with_prod_patent_cc1[i,1]
                                    self.case_marker[n,i] = 'cc1'
                                    
                                #case cc2)
                                else:
                                    self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                    self.case_marker[n,i] = 'cc2'
                                    
                            elif self.psi_m_star[n,i,1] > np.min(self.psi_m_star[:,i,1]):
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                self.case_marker[n,i] = 'aa'
                                
            elif self.psi_m_star[i,i,1] != np.min(self.psi_m_star[:,i,1]) \
                and np.where(self.psi_m_star[:,i,1] == self.psi_m_star[:,i,1].min())[0].shape[0] == 1:
                    # case b where the domestic threshold is not the smallest one for the origin, and the smallest one is unique
                    # country i patents first abroad
                    print(f'{origin},case b')
                    self.psi_m_star_with_prod_patent[i,i,1] = self.psi_o_star_with_prod_patent_b[i,1]
                    self.case_marker[i,i] = 'b'
                    for n,destination in enumerate(p.countries):
                        if i!=n:
                            if self.psi_m_star[n,i,1] < self.psi_o_star_with_prod_patent_b[i,1]:
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star[n,i,1]
                                self.case_marker[n,i] = 'bb1'
                                
                            elif self.psi_m_star[n,i,1] >= self.psi_o_star_with_prod_patent_b[i,1]:
                                self.psi_m_star_with_prod_patent[n,i,1] = self.psi_m_star_with_prod_patent_aa[n,i,1]
                                self.case_marker[n,i] = 'bb2'


        # ii)
        num_bracket = self.V_NP_P_minus_V_NP_NP_with_prod_patent[...,1]/self.w[None,:]*(
            1-np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k)
            ) + self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:]*(
                np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k)
                )
        
        self.mult_val_pat = 1 + (
            num_bracket.sum(axis=0) - np.diagonal(num_bracket)
            )/( np.diagonal(self.profit[...,1]) * (1/(self.G[1]+p.delta[:,1]-p.nu[1])-1/(self.G[1]+p.delta[:,1])) )
        
        
        # iii)
        
        self.V_with_prod_patent = np.zeros((p.N,p.S))
        
        A1 = ((p.k/(p.k-1))*self.V_NP[...,1]/self.w[None,:]).sum(axis=0)
        A2 = np.einsum('ni,ni,i->i',
                        self.V_P[...,1]/self.w[None,:] - self.V_NP[...,1]/self.w[None,:],
                        self.psi_m_star[...,1]**(1-p.k),
                        self.mult_val_pat
                        )*(p.k/(p.k-1))
        A3 = - np.einsum('ni,n,n,i->i',
                          self.psi_m_star[...,1]**-p.k,
                          self.w,
                          p.r_hjort,
                          1/self.w
                          )*p.fe[1]
        B = self.psi_o_star[:,1]**-p.k*p.fo[1]*p.r_hjort
        self.V_with_prod_patent[...,1] = (A1+A2+A3-B)*self.w[None,:]
        
        self.mult_val_all_innov = self.V_with_prod_patent[...,1]/self.V[...,1]

    def compute_share_of_exports_patented(self,p):
        A = np.einsum('ni,ni,ni,ni->ni',
                        self.PSI_MPD[...,1]+self.PSI_MPND[...,1],
                        1/self.PSI_M[...,1],
                        1/(1+p.tariff[...,1]),
                        self.X_M[...,1]
                        )
        B = np.einsum('ni,ni->ni',
                        1/(1+p.tariff[...,1]),
                        self.X[...,1]
                        )
        self.share_of_exports_patented = (A.sum(axis=0)-np.einsum('ii->i',
                                                                  A)
                                          )/(B.sum(axis=0)-np.einsum('ii->i',
                                                                     B))
    
    def compute_average_mark_up(self,p):
        prefactor = p.sigma[1:]/(p.sigma[1:]-1)
        A = self.X_M[:,:,1:]/(1+p.tariff[:,:,1:])
        B = self.X_CD[:,:,1:]/(1+p.tariff[:,:,1:])
        
        self.sectoral_average_markup = np.einsum(
            's,is,is->is',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
            ) / np.einsum(
                's,is,is->is',
                prefactor,
                np.einsum('nis->is',B),
                1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
                )
        
        self.aggregate_average_markup = np.einsum(
            's,is,i->i',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
            ) / np.einsum(
                's,is,i->i',
                prefactor,
                np.einsum('nis->is',B),
                1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
                )                                                                 
                                                                     
    def compute_non_solver_quantities(self,p):
        self.compute_tau(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_pflow(p)      
        self.compute_share_of_innovations_patented(p)
        self.compute_welfare(p)
        self.compute_non_solver_aggregate_qualities(p)
        self.compute_V(p)
        self.compute_average_mark_up(p)
        
    def compute_consumption_equivalent_welfare(self,p,baseline):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline.g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline.cons
                
    def compute_world_welfare_changes(self,p,baseline):
        one_ov_gamma = 1/p.gamma
        numerator = (p.labor**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = (p.labor**one_ov_gamma*baseline.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_pop_average_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
        numerator = (baseline.cons**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = baseline.cons.sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_negishi_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
    def compute_world_welfare_changes_custom_weights(self,p,baseline,weights):
        one_ov_gamma = 1/p.gamma
        numerator = (weights**one_ov_gamma*self.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-baseline.g*(1-one_ov_gamma))
        denominator = (weights**one_ov_gamma*baseline.cons**((p.gamma-1)*one_ov_gamma)).sum()*(p.rho-self.g*(1-one_ov_gamma))
        self.cons_eq_custom_weights_welfare_change = (numerator/denominator)**(p.gamma/(p.gamma-1))
        
    def compute_one_country_welfare_change(self,p,baseline_cons_country,baseline_g):
        self.cons_eq_welfare = self.cons*\
            ((p.rho-baseline_g*(1-1/p.gamma))/(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))\
                /baseline_cons_country

def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**i)
    return np.array(alt)

def cheb(N):
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D, x

def cheb_neuman_right(N):
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    D[0:int(N/2),:] = 0
    return D, x

class dynamic_var:
    def __init__(self, t_inf = 200, nbr_of_time_points = 1001, 
                 N = 7, S = 2, sol_init = None, sol_fin = None):
        self.t_inf = t_inf
        self.Nt = nbr_of_time_points
        self.t = np.linspace(-1,1,self.Nt)
        self.D,self.t_cheby = cheb(self.Nt-1)
        self.D_neuman,self.t_cheby = cheb_neuman_right(self.Nt-1)
        self.t_real = (self.t_cheby+1)*self.t_inf/2
        # print(self.t_real)
        self.sol_init = sol_init
        self.sol_fin = sol_fin
        if N == 7:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW']
        if N==13:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'AUS', 'MEX', 'IDN', 'ROW']
        if N==12:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ZAF', 'ROW']
        if N==11:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ROW']
        self.map_parameter = 32
        
    def elements(self): 
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        frame = deepcopy(self)
        return frame
            
    def vector_from_var(self):
        price_indices = self.price_indices.ravel()
        w = self.w.ravel()
        Z = self.Z.ravel()
        list_of_raveled_vectors = [getattr(self,qty)[...,1:,:].ravel()
                                   for qty in ['PSI_CD','PSI_MNP','PSI_MPND',
                                               # 'PSI_MPD','V_PD','V_P','V_NP']]
                                               'PSI_MPD','V_PD','DELTA_V','V_NP']]
        vec = np.concatenate([price_indices,w,Z]+list_of_raveled_vectors, axis=0)
        return vec
    
    
    @staticmethod
    def var_from_vector(vec,p,compute = True,sol_init=None,Nt=25,t_inf=500,sol_fin=None):
        init = dynamic_var(sol_init=sol_init,nbr_of_time_points =Nt,t_inf=t_inf,sol_fin=sol_fin)
        init.initiate_state_variables_0(sol_init)
        dic_of_guesses = {'price_indices':np.zeros((p.N,Nt)),
                        'w':np.zeros((p.N,Nt)),
                        'Z':np.zeros((p.N,Nt)),
                        'PSI_CD':np.zeros((p.N,p.S,Nt))[...,1:,:],
                        'PSI_MNP':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'PSI_MPND':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'PSI_MPD':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
                        # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
                        # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
                        # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
                        'V_PD':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'DELTA_V':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'V_NP':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:]}
        init.guess_from_dic(dic_of_guesses)
        init.guess_from_vector(vec)
        if compute:
            init.compute_solver_quantities(p)
        return init
    
    def guess_from_vector(self, x_old):
        idx = 0
        idx_end = 0
        
        idx_end += self.price_indices.size
        self.guess_price_indices(x_old[idx:idx_end].reshape(self.price_indices.shape))
        idx = idx_end
        
        idx_end += self.w.size
        self.guess_wage(x_old[idx:idx_end].reshape(self.w.shape))
        idx = idx_end
        
        idx_end += self.Z.size
        self.guess_Z(x_old[idx:idx_end].reshape(self.Z.shape))
        idx = idx_end
        
        idx_end += self.PSI_CD[...,1:,:].size
        self.guess_PSI_CD(x_old[idx:idx_end].reshape(self.PSI_CD[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
        idx_end += self.PSI_MNP[...,1:,:].size
        self.guess_PSI_MNP(x_old[idx:idx_end].reshape(self.PSI_MNP[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
        idx_end += self.PSI_MPND[...,1:,:].size
        self.guess_PSI_MPND(x_old[idx:idx_end].reshape(self.PSI_MPND[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
        idx_end += self.PSI_MPD[...,1:,:].size
        self.guess_PSI_MPD(x_old[idx:idx_end].reshape(self.PSI_MPD[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
        idx_end += self.V_PD[...,1:,:].size
        self.guess_V_PD(x_old[idx:idx_end].reshape(self.V_PD[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end        
        
        idx_end += self.DELTA_V[...,1:,:].size
        self.guess_DELTA_V(x_old[idx:idx_end].reshape(self.DELTA_V[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
        idx_end += self.V_NP[...,1:,:].size
        self.guess_V_NP(x_old[idx:idx_end].reshape(self.V_NP[...,1:,:].shape)
                          ,only_patenting_sectors=True)
        idx = idx_end
        
    def guess_price_indices(self,price_indices_init):
        self.price_indices = price_indices_init
        
    def guess_wage(self,w_init):
        self.w = w_init
        
    def guess_Z(self,Z_init):
        self.Z = Z_init
        
    def guess_PSI_CD(self,PSI_CD_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_CD_init.shape)
            shape[1] = shape[1]+1
            self.PSI_CD = np.zeros(shape)
            self.PSI_CD[...,1:,:] = PSI_CD_init
        else:
            self.PSI_CD = PSI_CD_init
        
    def guess_PSI_MNP(self,PSI_MNP_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_MNP_init.shape)
            shape[2] = shape[2]+1
            self.PSI_MNP = np.zeros(shape)
            self.PSI_MNP[...,1:,:] = PSI_MNP_init
        else:
            self.PSI_MNP = PSI_MNP_init
        
    def guess_PSI_MPND(self,PSI_MPND_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_MPND_init.shape)
            shape[2] = shape[2]+1
            self.PSI_MPND = np.zeros(shape)
            self.PSI_MPND[...,1:,:] = PSI_MPND_init
        else:
            self.PSI_MPND = PSI_MPND_init
        
    def guess_PSI_MPD(self,PSI_MPD_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_MPD_init.shape)
            shape[2] = shape[2]+1
            self.PSI_MPD = np.zeros(shape)
            self.PSI_MPD[...,1:,:] = PSI_MPD_init
        else:
            self.PSI_MPD = PSI_MPD_init
        
    def guess_V_PD(self, V_PD_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(V_PD_init.shape)
            shape[2] = shape[2]+1
            self.V_PD = np.zeros(shape)
            self.V_PD[...,1:,:] = V_PD_init
        else:
            self.V_PD = V_PD_init
        
    def guess_V_P(self, V_P_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(V_P_init.shape)
            shape[2] = shape[2]+1
            self.V_P = np.zeros(shape)
            self.V_P[...,1:,:] = V_P_init
        else:
            self.V_P = V_P_init
        
    def guess_V_NP(self, V_NP_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(V_NP_init.shape)
            shape[2] = shape[2]+1
            self.V_NP = np.zeros(shape)
            self.V_NP[...,1:,:] = V_NP_init
        else:
            self.V_NP = V_NP_init
            
    def guess_DELTA_V(self, DELTA_V_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(DELTA_V_init.shape)
            shape[2] = shape[2]+1
            self.DELTA_V = np.zeros(shape)
            self.DELTA_V[...,1:,:] = DELTA_V_init
        else:
            self.DELTA_V = DELTA_V_init
        
    def guess_from_dic(self, dic_of_guesses):
        self.guess_price_indices(dic_of_guesses['price_indices'])
        self.guess_wage(dic_of_guesses['w'])
        self.guess_Z(dic_of_guesses['Z'])
        self.guess_PSI_CD(dic_of_guesses['PSI_CD'],only_patenting_sectors=True)
        self.guess_PSI_MNP(dic_of_guesses['PSI_MNP'],only_patenting_sectors=True)
        self.guess_PSI_MPND( dic_of_guesses['PSI_MPND'],only_patenting_sectors=True)
        self.guess_PSI_MPD( dic_of_guesses['PSI_MPD'],only_patenting_sectors=True)
        self.guess_V_PD( dic_of_guesses['V_PD'],only_patenting_sectors=True)
        self.guess_V_NP( dic_of_guesses['V_NP'],only_patenting_sectors=True)
        try:
            self.guess_V_P( dic_of_guesses['V_P'],only_patenting_sectors=True)
        except:
            pass
        try:
            self.guess_DELTA_V( dic_of_guesses['DELTA_V'],only_patenting_sectors=True)
        except:
            pass
        
    def initiate_state_variables_0(self,var):
        self.PSI_CD_0 = var.PSI_CD
        self.PSI_MNP_0 = var.PSI_MNP
        self.PSI_MPND_0 = var.PSI_MPND
        self.PSI_MPD_0 = var.PSI_MPD
        self.PSI_M_0 = self.PSI_MNP_0+self.PSI_MPND_0+self.PSI_MPD_0
    
    def compute_phi(self, p):
        self.phi = np.einsum('is,nis,nis,ist,ist->nist',
                p.T**(1/p.theta[None,:]),
                1/p.tau,
                1/(1+p.tariff),
                self.w[:,None,:]**(-p.alpha[None,:,None]),
                self.price_indices[:,None,:]**(p.alpha[None,:,None]-1))
        
    def compute_PSI_M(self,p):
        self.PSI_M = self.PSI_MNP + self.PSI_MPND + self.PSI_MPD
    
    def compute_sectoral_prices(self, p):
        power = p.sigma-1

        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, 1:, None] \
            * ((self.PSI_M[...,1:,:]+self.PSI_M_0[...,1:,None])*self.phi[...,1:,:]**power[None, None, 1:,None]).sum(axis=1)

        B = (self.PSI_CD[...,1:,:]+self.PSI_CD_0[...,1:,None])*(
            self.phi[...,1:,:]**p.theta[None,None,1:,None]).sum(axis=1)**(power/p.theta)[None, 1:,None]

        self.P_M = np.full((p.N, p.S, self.Nt),np.inf)
        self.P_M[:,1:,:] = (A/(A+B))**(1/(1-p.sigma))[None, 1:,None]
   
        self.P_CD = np.ones((p.N, p.S, self.Nt))
        self.P_CD[:,1:,:] = (B/(A+B))**(1/(1-p.sigma))[None, 1:,None]
        
    def compute_patenting_thresholds(self, p):
        A = np.einsum('nt,n,s,it,i->nist',
                               self.w,
                               p.r_hjort,
                               p.fe[1:],
                               1/self.w,
                               1/p.r_hjort,
                               )
        
        self.psi_C = np.full((p.N,p.N,p.S,self.Nt),np.inf)
        self.psi_C[...,1:,:] = np.einsum('nt,s,n,nist->nist',
                                         self.w,
                                         p.fe[1:],
                                         p.r_hjort,
                                         1/(self.DELTA_V[...,1:,:])
                                         )
        self.psi_star = np.maximum(self.psi_C,1)
        psi_star_n_star = np.min(self.psi_star,axis=0)
        
        x_old = np.max(self.psi_C[...,1:,:], axis=0)
        mask = x_old[None,:,:,:]>=self.psi_C[...,1:,:]
        
        condition = np.maximum(A*(psi_star_n_star[None,:,1:,:]/self.psi_C[...,1:,:]-1),0).sum(axis=0)>=p.fo[None,1:,None]
        
        x_new = None
        cond = True
        it = 0
        print_once= True
        
        while cond:
            if it>0:
                x_old = x_new
            mask = x_old[None,:,:,:]>=self.psi_C[...,1:,:]
            x_new = (np.sum(A,axis=0,where=mask)+p.fo[None,1:,None])/np.sum(A/self.psi_C[...,1:,:],axis=0,where=mask)
            cond = np.any(x_old[~condition] != x_new[~condition])
            if it>40 and print_once:
                print('stuck')
                print_once = False
                self.plot_numerical_derivatives()
                cond = False
            it+=1
            
        x_new[condition] = psi_star_n_star[...,1:,:][condition]
        self.psi_o_star = np.full((p.N,p.S,self.Nt),np.inf)
        self.psi_o_star[...,1:,:] = x_new
        self.psi_m_star = np.full((p.N,p.N,p.S,self.Nt),np.inf)
        self.psi_m_star[...,1:,:] = np.maximum(self.psi_o_star[None,:,1:,:],self.psi_star[...,1:,:])
        
    def compute_V(self,p):
        self.V = np.zeros((p.N,p.S,self.Nt))        
        A1 = ((p.k[None,None,1:,None]/(p.k[None,None,1:,None]-1))*self.V_NP[...,1:,:]).sum(axis=0)
        A2 = np.einsum('nist,nt,s,n,nist->ist',
                       self.psi_m_star[...,1:,:]**-p.k[None,None,1:,None],
                       self.w,
                       p.fe[1:],
                       p.r_hjort,
                       p.k[None,None,1:,None]*self.psi_m_star[...,1:,:]/(self.psi_C[...,1:,:]*(p.k[None,None,1:,None]-1))-1
                       )
        B = self.psi_o_star[:,1:,:]**-p.k[None,1:,None]*p.fo[None,1:,None]*p.r_hjort[:,None,None]*self.w[:,None]
        self.V[...,1:,:] = A1+A2-B
        
    def compute_labor_research(self, p):
        self.l_R = np.zeros((p.N,p.S,self.Nt))
        self.l_R[...,1:,:] = np.einsum('is,ist,it->ist',
                                     p.eta[...,1:],
                                     self.V[...,1:,:],
                                     1/self.w)**(1/p.kappa)
    
    def compute_growth(self, p):
        self.g_s = p.k[:,None]*np.einsum('is,ist -> st',
                                 p.eta,
                                 self.l_R**(1-p.kappa)
                                 )/(p.k[:,None]-1) - p.zeta[:,None]
        self.g_s[0,:] = p.g_0
        self.g = (p.beta[:,None]*self.g_s/(p.sigma[:,None]-1)).sum(axis=0) / (p.beta*p.alpha).sum()
        
    def compute_labor_allocations(self, p):
        self.l_Ae = np.zeros((p.N,p.N,p.S,self.Nt))
        self.l_Ae[...,1:,:] = np.einsum('n,s,is,ist,nist -> inst',
                         p.r_hjort,
                         p.fe[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:,:]**(1-p.kappa),
                         self.psi_m_star[...,1:,:]**-p.k[None,None,1:,None]
                         )
        self.l_Ao = np.zeros((p.N,p.S,self.Nt))
        self.l_Ao[...,1:,:] = np.einsum('i,s,is,ist,ist -> ist',
                         p.r_hjort,
                         p.fo[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:,:]**(1-p.kappa),
                         self.psi_o_star[...,1:,:]**-p.k[None,1:,None]
                         )
        self.l_P = p.labor[:,None]-(self.l_Ao+self.l_R+self.l_Ae.sum(axis=0)).sum(axis=1)
        
    def compute_trade_flows_and_shares(self, p, assign = True):
            temp = ((self.PSI_M+self.PSI_M_0[...,None])[...,1:,:]*self.phi[...,1:,:]**(p.sigma-1)[None, None, 1:,None]).sum(axis=1)
            self.X_M = np.zeros((p.N, p.N, p.S, self.Nt))
            self.X_M[...,1:,:] = np.einsum('nist,nist,nst,nst,s,nt->nist',
                                    self.phi[..., 1:,:]**(p.sigma-1)[None, None, 1:,None],
                                    (self.PSI_M+self.PSI_M_0[...,None])[...,1:,:],
                                    1/temp,
                                    self.P_M[...,1:,:]**(1-p.sigma[None,1:,None]),
                                    p.beta[1:],
                                    self.Z
                                    )
            self.X_CD = np.einsum('nist,nst,nst,s,nt->nist',
                                    self.phi**(p.theta)[None,None,:,None],
                                    1/(self.phi**(p.theta)[None,None,:,None]).sum(axis=1),
                                    self.P_CD**(1-p.sigma[None,:,None]),
                                    p.beta,
                                    self.Z
                                    )

            self.X = self.X_M+self.X_CD
            
    def compute_profit(self,p):
        self.profit = np.zeros((p.N,p.N,p.S,self.Nt))
        self.profit[...,1:,:] = np.einsum('nist,s,nist,nis->nist',
                                self.X_M[...,1:,:],
                                1/p.sigma[1:],
                                1/(self.PSI_M+self.PSI_M_0[...,None])[...,1:,:],
                                1/(1+p.tariff[...,1:]))
    
    def compute_nominal_final_consumption(self,p):
        self.nominal_final_consumption = self.Z - np.einsum('s,nist,nis->it',
                                                            1-p.alpha,
                                                            self.X - self.X_M/p.sigma[None,None,:,None],
                                                            1/(1+p.tariff))
    
    def compute_interest_rate(self,p):
        self.CP_growth_rate = 2*np.einsum('tu,nu->nt',self.D_neuman,self.nominal_final_consumption)\
            /(self.t_inf*self.nominal_final_consumption)
        
        A = p.rho + (self.g[None,:]+self.CP_growth_rate)/p.gamma
        
        self.inflation = 2*np.einsum('tu,nu->nt',self.D_neuman,self.price_indices)\
            /(self.t_inf*self.price_indices)
            
        self.r = A + (1-1/p.gamma)*self.inflation
        
    def compute_solver_quantities(self,p):
        self.compute_phi(p)
        self.compute_PSI_M(p)
        self.compute_sectoral_prices(p)
        self.compute_patenting_thresholds(p)
        self.compute_V(p)
        self.compute_labor_research(p)
        self.compute_growth(p)
        self.compute_labor_allocations(p)
        self.compute_trade_flows_and_shares(p)
        self.compute_profit(p)
        self.compute_nominal_final_consumption(p)
        self.compute_interest_rate(p)
        
    def compute_wage(self,p):
        wage = (p.alpha[None, :, None] * ((self.X - self.X_M/p.sigma[None, None, :, None])/(1+p.tariff[...,None])).sum(axis=0)
                ).sum(axis=1)/self.l_P
        return wage
        
    def compute_expenditure(self,p):
        A = np.einsum('nist->it', self.X)
        B = np.einsum('it,nist->it', self.w, self.l_Ae)
        C = np.einsum('i,kt->it',p.deficit_share_world_output,self.Z)
        D = np.einsum('nt,inst->it', self.w, self.l_Ae)
        Z = (A+B-(C+D))
        A1 = np.einsum('nist,nis->it', 
                      self.X,
                      1/(1+p.tariff))
        A2 = np.einsum('inst,ins,ins->it', 
                      self.X,
                      p.tariff,
                      1/(1+p.tariff))
        B = np.einsum('it,nist->it', self.w, self.l_Ae)
        C = np.einsum('i,t->it',
                      p.deficit_share_world_output,
                      np.einsum('nist,nis->t', 
                                  self.X,
                                  1/(1+p.tariff)
                                  )
                      )
        D = np.einsum('nt,inst->it', self.w, self.l_Ae)
        Z = (A1+A2+B-(C+D))
        return Z
        
    def compute_price_indices(self,p):
        power = (p.sigma-1)
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :, None] \
            * ((self.PSI_M + self.PSI_M_0[...,None]) * self.phi**power[None, None, :, None]).sum(axis=1)
        B = (self.PSI_CD + self.PSI_CD_0[...,None])*(self.phi**p.theta[None,None,:,None]).sum(axis=1)**(power/p.theta)[None, :, None]

        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:,None]*(A+B))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :, None]/(p.sigma[None, :, None]-1)) ).prod(axis=1)
        return price_indices
        
    def compute_PSI_CD(self,p):
        self.PSI_CD_dot = 2*np.einsum('tu,nsu->nst',self.D_neuman,self.PSI_CD[...,1:,:])/self.t_inf
        PSI_CD = np.zeros((p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nst',
            p.nu[1:],
            self.PSI_MNP[...,1:,:]+self.PSI_MNP_0[...,1:,None],
            )
        numB = np.einsum('ns,nist->nst',
            p.delta[:,1:],
            self.PSI_MPD[...,1:,:]+self.PSI_MPD_0[...,1:,None],
            )
        numC = self.PSI_CD_dot
        PSI_CD[...,1:,:] = np.einsum('nst,st->nst',
                           numA+numB-numC,
                           1/(self.g_s[1:,:]+p.zeta[1:,None])
                           )-self.PSI_CD_0[...,1:,None]
        PSI_CD[...,-1] = 0
        
        return PSI_CD
        
    def compute_PSI_MNP(self,p):
        self.PSI_MNP_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_MNP[...,1:,:])/self.t_inf
        PSI_MNP = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = p.k[None,None,1:,None]*np.einsum('is,ist,nist->nist',
            p.eta[:,1:],
            self.l_R[...,1:,:]**(1-p.kappa),
            1-self.psi_m_star[...,1:,:]**(1-p.k[None,None,1:,None]),
            )/(p.k[None,None,1:,None]-1)
        numB = np.einsum('ns,nist->nist',
            p.delta[:,1:],
            self.PSI_MPND[...,1:,:]+self.PSI_MPND_0[...,1:,None],
            )
        numC = self.PSI_MNP_dot 
        PSI_MNP[...,1:,:] = np.einsum('nist,st->nist',
                           numA+numB-numC,
                           1/(self.g_s[1:,:]+p.zeta[1:,None]+p.nu[1:,None])
                           )-self.PSI_MNP_0[...,1:,None]
        
        PSI_MNP[...,-1] = 0
        
        return PSI_MNP
        
        
    def compute_PSI_MPND(self,p):
        self.PSI_MPND_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_MPND[...,1:,:])/self.t_inf
        PSI_MPND = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = p.k[None,None,1:,None]*np.einsum('is,ist,nist->nist',
            p.eta[:,1:],
            self.l_R[...,1:,:]**(1-p.kappa),
            self.psi_m_star[...,1:,:]**(1-p.k[None,None,1:,None]),
            )/(p.k[None,None,1:,None]-1)
        numB = self.PSI_MPND_dot
        PSI_MPND[...,1:,:] = np.einsum('nist,nst->nist',
                           numA-numB,
                           1/(self.g_s[None,1:,:]+p.zeta[None,1:,None]+p.nu[None,1:,None]+p.delta[:,1:,None])
                           )-self.PSI_MPND_0[...,1:,None]
        
        PSI_MPND[...,-1] = 0
        
        return PSI_MPND
        
        
    def compute_PSI_MPD(self,p):
        self.PSI_MPD_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_MPD[...,1:,:])/self.t_inf
        PSI_MPD = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nist',
            p.nu[1:],
            self.PSI_MPND[...,1:,:]+self.PSI_MPND_0[...,1:,None],
            )
        numB = self.PSI_MPD_dot
        PSI_MPD[...,1:,:] = np.einsum('nist,nst->nist',
                           numA-numB,
                           1/(self.g_s[None,1:,:]+p.zeta[None,1:,None]+p.delta[:,1:,None])
                           )-self.PSI_MPD_0[...,1:,None]
        
        PSI_MPD[...,-1] = 0
        
        return PSI_MPD
        
    def compute_V_PD(self,p):
        self.V_PD_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_PD[...,1:,:])/self.t_inf
        V_PD = np.zeros((p.N,p.N,p.S,self.Nt))
        V_PD[...,1:,:] = np.einsum('nist,nist->nist',
                                   self.profit[...,1:,:]+self.V_PD_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.delta[:,None,1:,None]+self.g_s[None,None,1:,:]-self.g[None,None,None,:])
                                   )
        return V_PD
        
    def compute_V_NP(self,p):
        self.V_NP_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_NP[...,1:,:])/self.t_inf
        V_NP = np.zeros((p.N,p.N,p.S,self.Nt))
        V_NP[...,1:,:] = np.einsum('nist,ist->nist',
                                   self.profit[...,1:,:]+self.V_NP_dot,
                                   1/(self.r[:,None,:]+p.zeta[None,1:,None]+p.nu[None,1:,None]+self.g_s[None,1:,:]-self.g[None,None,:])
                                   )
        return V_NP
        
    def compute_V_P(self,p):
        #Not used, replaced by DELTA_V = V_P - V_NP
        self.V_P_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_P[...,1:,:])/self.t_inf
        V_P = np.zeros((p.N,p.N,p.S,self.Nt))
        V_P[...,1:,:] = np.einsum('nist,nist->nist',
                                   self.profit[...,1:,:]+p.delta[:,None,1:,None]*self.V_NP[...,1:,:]\
                                       +p.nu[None,None,1:,None]*self.V_PD[...,1:,:]+self.V_P_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.nu[None,None,1:,None]+
                                      p.delta[:,None,1:,None]+self.g_s[None,None,1:,:]-self.g[None,None,None,:])
                                   )
        return V_P
    
    def compute_DELTA_V(self,p):
        self.DELTA_V_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.DELTA_V[...,1:,:])/self.t_inf
        DELTA_V = np.zeros((p.N,p.N,p.S,self.Nt))
        DELTA_V[...,1:,:] = np.einsum('nist,nist->nist',
                                   p.nu[None,None,1:,None]*self.V_PD[...,1:,:]+self.DELTA_V_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.nu[None,None,1:,None]+
                                      p.delta[:,None,1:,None]+self.g_s[None,None,1:,:]-self.g[None,None,None,:])
                                   )
        return DELTA_V
    
    def compute_A(self,p):
        self.A = np.exp(np.polyval(np.polyint(np.polyfit(self.t_real,
                            self.g,
                            self.Nt)),self.t_real))
        
    def compute_PSI_S(self,p):
        self.PSI_S = np.zeros((p.S,self.Nt))
        for s,_ in enumerate(p.sectors):
            self.PSI_S[s,:] = np.exp(np.polyval(np.polyint(np.polyfit(self.t_real,
                            self.g_s[s,:],
                            self.Nt)),self.t_real))
            
    def compute_pflow(self,p):
        self.pflow = np.einsum('nist,is,ist->nist',
                              self.psi_m_star[...,1:,:]**(-p.k[None,None,1:,None]),
                              p.eta[...,1:],
                              self.l_R[...,1:,:]**(1-p.kappa)
                              ).squeeze()
            
    def compute_consumption_equivalent_welfare(self,p):
        power = 1-1/p.gamma

        self.integrand_welfare = np.einsum(',t,nt->nt',
                              p.rho-self.sol_init.g*power,
                              np.exp(-p.rho*self.t_real),
                              (self.ratios_of_consumption_levels_change_not_normalized*np.exp(self.sol_init.g*self.t_real)[None,:])**power)
        
        self.second_term_sum_welfare = self.integrand_welfare/(p.rho-self.g[None,:]*power)
        
        self.integral_welfare = np.zeros((p.N,self.Nt))
        for i in range(p.N):
            self.integral_welfare[i,:] = np.polyval(
                np.polyint(np.polyfit(self.t_real,
                            self.integrand_welfare[i,:],
                            self.Nt)),self.t_real
                )

        self.cons_eq_welfare = (self.integral_welfare[:,0]+self.second_term_sum_welfare[:,0])**(1/power)
        
        # population-weighted world welfare change
        bracketA_integrand = np.einsum('t,t,t->t',
                                       np.exp(-p.rho*self.t_real),
                                       self.A**power,
                                       (p.labor[:,None]**(1/p.gamma)
                                       *(self.nominal_final_consumption/self.price_indices)**power).sum(axis=0)
                                       )
        bracket_A = np.polyval(
            np.polyint(np.polyfit(self.t_real,
                        bracketA_integrand,
                        self.Nt)),self.t_real
            )[0]
        
        bracket_B = self.A[0]**power\
                *np.exp(-p.rho*self.t_real[0])\
                *(p.labor**(1/p.gamma)*(self.nominal_final_consumption[:,0]/self.price_indices[:,0])**power).sum()\
                /(p.rho-self.g[0]*power)
                
        self.cons_eq_pop_average_welfare_change = np.einsum(',,->',
                                (p.rho-self.sol_init.g*power),
                                1/(p.labor**(1/p.gamma)*self.sol_init.cons**power).sum(),
                                bracket_A+bracket_B
                                )**(1/power)
        
        # negishi-weighted world welfare change
        bracketA_integrand = np.einsum('t,t,t->t',
                                       np.exp(-p.rho*self.t_real),
                                       self.A**power,
                                       (self.sol_init.cons[:,None]**(1/p.gamma)
                                       *(self.nominal_final_consumption/self.price_indices)**power).sum(axis=0)
                                       )
        bracket_A = np.polyval(
            np.polyint(np.polyfit(self.t_real,
                        bracketA_integrand,
                        self.Nt)),self.t_real
            )[0]
        
        bracket_B = self.A[0]**power\
                *np.exp(-p.rho*self.t_real[0])\
                *(self.sol_init.cons**(1/p.gamma)*(self.nominal_final_consumption[:,0]/self.price_indices[:,0])**power).sum()\
                /(p.rho-self.g[0]*power)
                
        self.cons_eq_negishi_welfare_change = np.einsum(',,->',
                                (p.rho-self.sol_init.g*power),
                                1/(self.sol_init.cons).sum(),
                                bracket_A+bracket_B
                                )**(1/power)
    
    def compute_consumption_equivalent_welfare_for_subset_of_countries(self,p,countries):
        countries_indices = [p.countries.index(c) for c in countries]
        power = 1-1/p.gamma
        # population-weighted world welfare change
        bracketA_integrand = np.einsum('t,t,t->t',
                                       np.exp(-p.rho*self.t_real),
                                       self.A**power,
                                       np.take((p.labor[:,None]**(1/p.gamma)
                                       *(self.nominal_final_consumption/self.price_indices)**power),countries_indices,axis=0).sum(axis=0)
                                       )
        bracket_A = np.polyval(
            np.polyint(np.polyfit(self.t_real,
                        bracketA_integrand,
                        self.Nt)),self.t_real
            )[0]
        
        bracket_B = self.A[0]**power\
                *np.exp(-p.rho*self.t_real[0])\
                *np.take((p.labor**(1/p.gamma)*(self.nominal_final_consumption[:,0]/self.price_indices[:,0])**power),countries_indices,axis=0).sum()\
                /(p.rho-self.g[0]*power)
                
        cons_eq_pop_average_welfare_change_subset_countries = np.einsum(',,->',
                                (p.rho-self.sol_init.g*power),
                                1/np.take((p.labor**(1/p.gamma)*self.sol_init.cons**power),countries_indices,axis=0).sum(),
                                bracket_A+bracket_B
                                )**(1/power)
        
        # negishi-weighted world welfare change
        bracketA_integrand = np.einsum('t,t,t->t',
                                       np.exp(-p.rho*self.t_real),
                                       self.A**power,
                                       np.take((self.sol_init.cons[:,None]**(1/p.gamma)
                                       *(self.nominal_final_consumption/self.price_indices)**power),countries_indices,axis=0).sum(axis=0)
                                       )
        bracket_A = np.polyval(
            np.polyint(np.polyfit(self.t_real,
                        bracketA_integrand,
                        self.Nt)),self.t_real
            )[0]
        
        bracket_B = self.A[0]**power\
                *np.exp(-p.rho*self.t_real[0])\
                *np.take((self.sol_init.cons**(1/p.gamma)*(self.nominal_final_consumption[:,0]
                                                   /self.price_indices[:,0])**power),countries_indices,axis=0).sum()\
                /(p.rho-self.g[0]*power)
                
        cons_eq_negishi_welfare_change_subset_countries = np.einsum(',,->',
                                (p.rho-self.sol_init.g*power),
                                1/np.take((self.sol_init.cons),countries_indices,axis=0).sum(),
                                bracket_A+bracket_B
                                )**(1/power)
        
        return {'pop_weighted':cons_eq_pop_average_welfare_change_subset_countries,'negishi':cons_eq_negishi_welfare_change_subset_countries}
        
    def compute_ratios_of_consumption_levels_change_not_normalized(self,p):
        self.ratios_of_consumption_levels_change_not_normalized = \
            (self.nominal_final_consumption/self.price_indices)*self.A[None,:]*np.exp(-self.sol_init.g*self.t_real)[None,:]/self.sol_init.cons[:,None]
    
    def compute_non_solver_quantities(self,p):
        self.compute_A(p)
        self.compute_PSI_S(p)
        self.compute_ratios_of_consumption_levels_change_not_normalized(p)
        self.compute_consumption_equivalent_welfare(p)
        self.compute_pflow(p)
    
    def plot_country(self,country_idx,title=None,initial=False,history=False):
        fig,ax = plt.subplots(5,2,figsize = (15,10),layout = 'constrained')
        if country_idx == 'all':
            for i,c in enumerate(self.countries):
                fit = np.polyfit(self.t_cheby,
                                  self.w[i,:]/self.w[i,0],
                                  self.Nt)
                ax[0,0].scatter(self.t_real,self.w[i,:]/self.w[i,0],label=c,zorder=-i*10)
                ax[0,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                
                fit = np.polyfit(self.t_cheby,
                                  self.price_indices[i,:]/self.price_indices[i,0],
                                  self.Nt)
                ax[1,0].scatter(self.t_real,self.price_indices[i,:]/self.price_indices[i,0],label=str(i),zorder=-i*10)
                ax[1,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                
                fit = np.polyfit(self.t_cheby,
                                  self.Z[i,:]/self.Z[i,0],
                                  self.Nt)
                ax[0,1].scatter(self.t_real,self.Z[i,:]/self.Z[i,0],label=str(i),zorder=-i*10)
                ax[0,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                
                fit = np.polyfit(self.t_cheby,
                                  self.PSI_CD[i,1,:],
                                  self.Nt)
                ax[1,1].scatter(self.t_real,self.PSI_CD[i,1,:],label=str(i),zorder=-i*10)
                ax[1,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                if not initial:
                    fit = np.polyfit(self.t_cheby,
                                      self.psi_o_star[i,1,:],
                                      self.Nt)
                    ax[2,0].scatter(self.t_real,self.psi_o_star[i,1,:],zorder=-i*10)
                    ax[2,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                            np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                    
                    fit = np.polyfit(self.t_cheby,
                                      self.g,
                                      self.Nt)
                    ax[2,1].scatter(self.t_real,self.g,zorder=-i*10)
                    ax[2,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                            np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                try:
                    fit = np.polyfit(self.t_cheby,
                                      self.l_R[i,1,:]/self.l_R[i,1,0],
                                      self.Nt)
                    ax[3,0].scatter(self.t_real,self.l_R[i,1,:]/self.l_R[i,1,0],zorder=-i*10)
                    ax[3,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                            np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                except:
                    pass
                
                try:
                    fit = np.polyfit(self.t_cheby,
                                      self.welfare[i,:],
                                      self.Nt)
                    ax[3,1].scatter(self.t_real,self.welfare[i,:],zorder=-i*10)
                    ax[3,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                            np.polyval(fit, cheb(500)[1]),zorder=-i*10)
                except:
                    pass
                
                try:
                    fit = np.polyfit(self.t_cheby,
                                      self.cons_eq_welfare[i,:],
                                      self.Nt)
                    ax[4,1].scatter(self.t_real,self.cons_eq_welfare[i,:],marker='*',zorder=-i*10)
                    ax[4,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                            np.polyval(fit, cheb(500)[1]),ls='--',zorder=-i*10)
                except:
                    pass
                
        else:
            fit = np.polyfit(self.t_cheby,
                              self.w[country_idx,:],
                              self.Nt)
            ax[0,0].scatter(self.t_real,self.w[country_idx,:])
            ax[0,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                    np.polyval(fit, cheb(500)[1]))
            
            fit = np.polyfit(self.t_cheby,
                              self.price_indices[country_idx,:],
                              self.Nt)
            ax[1,0].scatter(self.t_real,self.price_indices[country_idx,:])
            ax[1,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                    np.polyval(fit, cheb(500)[1]))
            
            fit = np.polyfit(self.t_cheby,
                              self.Z[country_idx,:],
                              self.Nt)
            ax[0,1].scatter(self.t_real,self.Z[country_idx,:])
            ax[0,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                    np.polyval(fit, cheb(500)[1]))
            
            fit = np.polyfit(self.t_cheby,
                              self.PSI_CD[country_idx,1,:],
                              self.Nt)
            ax[1,1].scatter(self.t_real,self.PSI_CD[country_idx,1,:])
            ax[1,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                    np.polyval(fit, cheb(500)[1]))
            
            if not initial:
                fit = np.polyfit(self.t_cheby,
                                  self.psi_o_star[country_idx,1,:],
                                  self.Nt)
                ax[2,0].scatter(self.t_real,self.psi_o_star[country_idx,1,:])
                ax[2,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]))
                
                fit = np.polyfit(self.t_cheby,
                                  self.g,
                                  self.Nt)
                ax[2,1].scatter(self.t_real,self.g)
                ax[2,1].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]))
                
                fit = np.polyfit(self.t_cheby,
                                  self.r[0,:],
                                  self.Nt)
                ax[4,0].scatter(self.t_real,self.r[0,:])
                ax[4,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]))
            
            try:
                fit = np.polyfit(self.t_cheby,
                                  self.l_R[country_idx,1,:],
                                  self.Nt)
                ax[3,0].scatter(self.t_real,self.l_R[country_idx,1,:])
                ax[3,0].plot((cheb(500)[1]+1)*self.t_inf/2, 
                        np.polyval(fit, cheb(500)[1]))
            except:
                pass
            
            if self.sol_init is not None:
                ax[0,0].scatter([0],[self.sol_init.w[country_idx]],color='red')
                ax[0,1].scatter([0],[self.sol_init.Z[country_idx]],color='red')
                ax[1,0].scatter([0],[self.sol_init.price_indices[country_idx]],color='red')
                ax[1,1].scatter([0],[self.sol_init.PSI_CD[country_idx,1]-self.PSI_CD_0[country_idx,1]],color='red')
                if not initial:
                    ax[2,0].scatter([0],[self.sol_init.psi_o_star[country_idx,1]],color='red')
                    ax[2,1].scatter([0],[self.sol_init.g],color='red')
                    ax[3,0].scatter([0],[self.sol_init.l_R[country_idx,1]],color='red')
            
            if self.sol_fin is not None:
                ax[0,0].scatter([self.t_inf],[self.sol_fin.w[country_idx]],color='red')
                ax[0,1].scatter([self.t_inf],[self.sol_fin.Z[country_idx]],color='red')
                ax[1,0].scatter([self.t_inf],[self.sol_fin.price_indices[country_idx]],color='red')
                ax[1,1].scatter([self.t_inf],[self.sol_fin.PSI_CD[country_idx,1]-self.PSI_CD_0[country_idx,1]],color='red')
                if not initial:
                    ax[2,0].scatter([self.t_inf],[self.sol_fin.psi_o_star[country_idx,1]],color='red')
                    ax[2,1].scatter([self.t_inf],[self.sol_fin.g],color='red')
                    ax[3,0].scatter([self.t_inf],[self.sol_fin.l_R[country_idx,1]],color='red')
                    
        if title is not None:
            plt.suptitle(title)
        ax[0,0].set_title('w')
        ax[1,0].set_title('P')
        ax[0,1].set_title('Z')
        ax[1,1].set_title('PSI_CD')
        ax[2,1].set_title('g')
        ax[2,0].set_title('psi_o_star')
        ax[3,1].set_title('welfares')
        ax[4,1].set_title('cons equivalent welfares')
        ax[3,0].set_title('l_R')
        ax[4,0].set_title('r')

        if not initial:
            ax[4,1].plot(self.DELTA_V[...,1,:].ravel())
            ax[4,1].set_title('DELTA V')
        if country_idx == 'all':
            ax[0,0].legend()
        plt.show()
        
    def plot_all_countries(self):
        for i,c in enumerate(self.countries):
            self.plot_country(i,title=c)
        
    def plot_numerical_derivatives(self,title=None):
        fig,ax = plt.subplots(2,2,figsize = (15,10),layout="constrained")
        ax[0,0].plot(self.nominal_final_consumption.ravel())
        ax[0,0].set_title('PC')
        ax1 = ax[0,0].twinx()
        ax1.plot(self.CP_growth_rate.ravel(),color='r')
        ax[1,0].plot(self.PSI_CD[...,1:,:].ravel())
        ax[1,0].set_title('PSI_CD')
        ax1 = ax[1,0].twinx()
        ax1.plot(self.PSI_CD_dot.ravel(),color='r')
        # ax[0,1].plot(self.V_P[...,1:,:].ravel())
        # ax[0,1].set_title('V_P')
        # ax1 = ax[0,1].twinx()
        # ax1.plot(self.V_P_dot.ravel(),color='r')
        ax[0,1].plot(self.DELTA_V[...,1:,:].ravel())
        ax[0,1].set_title('DELTA_V')
        ax1 = ax[0,1].twinx()
        ax1.plot(self.DELTA_V_dot.ravel(),color='r')
        ax[1,1].plot(self.r.ravel())
        ax[1,1].set_title('r')
        
        if title is not None:
            plt.suptitle(title)
        
        plt.show()
    
    def get_jump(self,qty):
        if qty == 'profit':
            jump = (getattr(self,qty)[...,-1]-np.einsum('nis,i->nis',
                                  self.sol_init.profit,
                                  self.sol_init.w)
                    )/(np.einsum('nis,i->nis',
                                self.sol_fin.profit,
                                self.sol_fin.w)
                        -
                        np.einsum('nis,i->nis',
                                self.sol_init.profit,
                                self.sol_init.w)
                        )
            return np.nanmean(jump)*100,np.nanmedian(jump)*100
        jump = (getattr(self,qty)[...,-1]-getattr(self.sol_init,qty)
                )/(getattr(self.sol_fin,qty)-getattr(self.sol_init,qty))
        return np.nanmean(jump)*100,np.nanmedian(jump)*100
    
    def get_typical_time_evolution(self,qty):
        #!!! to improve for dimensions
        try:
            origin_deriv = (2*np.einsum('tu,...u->...t',
                                    self.D_neuman,
                                    getattr(self,qty)
                                    )/self.t_inf)[...,-1]
        except:
            pass
        time_evol = np.abs(getattr(self,qty)[...,-1]-getattr(self,qty)[...,0])/np.abs(origin_deriv)
        return np.nanmean(time_evol),np.nanmedian(time_evol)
    
def remove_diag(A):
    removed = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], int(A.shape[0])-1, -1)
    return np.squeeze(removed)

def eps(x):
    return 1-np.exp(-x)
    
class moments:
    def __init__(self,list_of_moments = None):
        if list_of_moments is None:
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'OUT', 'KM','KM_DD_DD','KMCHEM','KMPHARMA','KMPHARMACHEM', 'KM_GDP', 
                                    'RD','RDPHARMA','RDCHEM','RDPHARMACHEM','RD_US','RD_RUS', 'RP',
                               'SRDUS', 'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US','SDOMTFLOW','STFLOW',
                               'STFLOWSDOM','SPFLOWDOM_RUS', 'SPFLOW_RUS','SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST',
                               'UUPCOST','UUPCOSTS','PCOST','PCOSTINTER','PCOSTNOAGG','PCOSTINTERNOAGG',
                               'JUPCOSTRD','SINNOVPATUS','TO','TO_DD_DD','TOCHEM','TOPHARMA','TOPHARMACHEM','TE','TECHEM','TEPHARMA','TEPHARMACHEM',
                               'DOMPATRATUSEU','DOMPATUS','DOMPATEU','AGGAVMARKUP','AVMARKUPPHARCHEM',
                               'DOMPATINUS','DOMPATINEU','SPATORIG','SPATDEST','TWSPFLOW','TWSPFLOWDOM','ERDUS',
                               'PROBINNOVENT','SHAREEXPMON','SGDP','RGDPPC','SDFLOW']
        else:
            self.list_of_moments = list_of_moments
        self.weights_dict = {'GPDIFF': 1,
                             'GROWTH': 5,
                             'KM': 1,
                             'KM_DD_DD': 1,
                             'KMCHEM': 1,
                             'KMPHARMA': 1,
                             'KMPHARMACHEM': 1,
                             'KM_GDP': 5,
                             'OUT': 5,
                             'RD': 10,
                             'RDPHARMACHEM': 3,
                             'RDPHARMA': 3,
                             'RDCHEM': 3,
                             'RD_US': 3,
                             'RD_RUS': 3,
                             'RP': 1,
                             'SPFLOW': 1,
                             'SPFLOW_US': 1,
                             'SPFLOW_RUS': 1,
                             'SPFLOWDOM': 1,
                             'SPFLOWDOM_US': 1,
                             'SPFLOWDOM_RUS': 1,
                             'AGGAVMARKUP':5,
                             'AVMARKUPPHARCHEM':5,
                             'SRDUS': 1,
                             'SRGDP': 1,
                             'SGDP': 1,
                             'RGDPPC': 1,
                             'SRGDP_US': 1,
                             'SRGDP_RUS': 1,
                             'STFLOW': 1,
                             'SDOMTFLOW': 1,
                             'JUPCOST': 1,
                             'UUPCOST': 1,
                             'UUPCOSTS': 1,
                             'PCOSTNOAGG': 1,
                             'PCOSTINTERNOAGG': 1,
                             'PCOST': 1,
                             'PCOSTINTER': 1,
                             'JUPCOSTRD': 1,
                             'TP': 1,
                             'inter_TP': 3,
                             'Z': 1,
                             'SDFLOW':1,
                             'STFLOWSDOM': 1,
                             'SINNOVPATEU': 1,
                             'SINNOVPATUS': 1,
                             'NUR': 1,
                             'TO': 5,
                             'TO_DD_DD': 5,
                             'TOCHEM': 5,
                             'TOPHARMA': 5,
                             'TOPHARMACHEM': 5,
                             'TECHEM': 5,
                             'TEPHARMA': 5,
                             'TEPHARMACHEM': 5,
                             'TE': 5,
                             'DOMPATRATUSEU': 2,
                             'DOMPATUS': 1,
                             'DOMPATEU': 1,
                             'DOMPATINUS': 1,
                             'DOMPATINEU': 1,
                             'SPATORIG': 2,
                             'SPATDEST': 2,
                             'TWSPFLOW': 1,
                             'TWSPFLOWDOM': 1,
                             'ERDUS': 3,
                             'PROBINNOVENT': 5,
                             'SHAREEXPMON': 5
                             }

        self.drop_CHN_IND_BRA_ROW_from_RD = True
        self.add_domestic_US_to_SPFLOW = False
        self.add_domestic_EU_to_SPFLOW = False
        self.aggregate_moments = False
        
        self.loss = 'log'
        self.dim_weight = 'lin'
    
    def get_signature_list(self, list_of_moments = None):
        if list_of_moments is None:
            list_of_moments = self.list_of_moments
        l = []
        for mom in list_of_moments:
            # if self.idx[mom][0] == 'scalar':
            #     l.extend([mom])
            # else:
            if mom == 'RD' or mom =='RD_RDUS':
                if self.drop_CHN_IND_BRA_ROW_from_RD:
                    # l.extend([mom+' '+str(x) for x in list(self.idx[mom])[:3]])
                    l.extend([mom+' '+str(x) for x in [list(self.idx[mom])[i] for i in [0,1,2,6,7,9]]])
            else:        
                l.extend([mom+' '+str(x) for x in list(self.idx[mom])])
        return l
    
    @staticmethod
    def get_list_of_moments():
        return ['GPDIFF', 'GROWTH', 'KM','KM_DD_DD','KMCHEM','KMPHARMA','KMPHARMACHEM','KM_GDP', 'OUT', 'RD',
                'RDPHARMA','RDCHEM','RDPHARMACHEM','RD_US','RD_RUS', 'RP', 
                'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US','SDOMTFLOW','STFLOW','STFLOWSDOM',
                'SPFLOWDOM_RUS', 'SPFLOW_RUS','DOMPATUS','DOMPATEU','DOMPATINUS','DOMPATINEU',
                'SRDUS', 'SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST','UUPCOST','UUPCOSTS','PCOST','PCOSTINTER',
                'PCOSTNOAGG','PCOSTINTERNOAGG','JUPCOSTRD', 'TP', 'Z','inter_TP', 
                'SINNOVPATEU','SINNOVPATUS','TO','TO_DD_DD','TOCHEM','TOPHARMA','TOPHARMACHEM',
                'TE','TECHEM','TEPHARMA','TEPHARMACHEM','NUR','DOMPATRATUSEU','AGGAVMARKUP','AVMARKUPPHARCHEM',
                'SPATDEST','SPATORIG','TWSPFLOW','TWSPFLOWDOM','ERDUS','PROBINNOVENT',
                'SHAREEXPMON','SGDP','RGDPPC','SDFLOW']
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
            
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def load_data(self,data_path = None,dir_path=None):
        
        if dir_path is None:
            dir_path = './'
        if data_path is None:
            data_path = 'data/data_leg/'
        
        self.data_path = data_path
        
        data_path = dir_path+data_path
        
        self.ccs_moments = pd.read_csv(data_path+'country_country_sector_moments.csv',index_col=[1,0,2]).sort_index()
        
        N = len(self.ccs_moments.index.get_level_values(0).drop_duplicates())
        S = len(self.ccs_moments.index.get_level_values(2).drop_duplicates())
        
        self.c_moments = pd.read_csv(data_path+'country_moments.csv',index_col=[0])
        if S == 2:
            self.cc_moments = pd.read_csv(data_path+'country_country_moments.csv',index_col=[1,0]).sort_index()
        if S > 2:
            self.cc_moments = pd.read_csv(data_path+'country_country_moments.csv',index_col=[1,0,2]).sort_index()
        self.moments = pd.read_csv(data_path+'scalar_moments.csv',index_col=[0])
        self.sector_moments = pd.read_csv(data_path+'sector_moments.csv',index_col=[0])
        
        self.description = pd.read_csv(data_path+'moments_descriptions.csv',sep=';',index_col=[0])
        self.pat_fees = pd.read_csv(data_path+'final_pat_fees.csv',index_col=[0])
        
        if N==7:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW']
        if N==13:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'AUS', 'MEX', 'IDN', 'ROW']
        if N==12:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ZAF', 'ROW']
        if N==11:
            self.countries = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN',
                              'KOR', 'RUS', 'MEX', 'ROW']
        self.N = N
        if S == 2:
            self.sectors = ['Non patent', 'Patent']
        if S == 3:
            self.sectors = ['Non patent', 'Patent', 'Pharma Chemicals']
        if S == 4:
            self.sectors = ['Non patent', 'Patent', 'Pharmaceuticals', 'Chemicals']
        
        self.unit = 1e6
        self.STFLOW_target = (self.ccs_moments.trade/
                              self.ccs_moments.trade.sum()).values.reshape(N,N,S)
        self.STFLOWSDOM_target = self.ccs_moments.trade.values.reshape(N,N,S)\
            /np.einsum('nns->ns',self.ccs_moments.trade.values.reshape(N,N,S))[:,None,:]
        if S > 2:
            self.SDFLOW_target = np.einsum('nns->ns',self.STFLOW_target[:,:,2:]
                                           )/np.einsum('nn->n',
                                                       self.STFLOW_target[:,:,1])[:,None]
        if S == 2:
            self.SDFLOW_target = np.array([np.nan])
        self.SPFLOW_target = self.cc_moments.query("destination_code != origin_code")['patent flows'].values
        if S == 2:
            self.SPFLOW_target = self.SPFLOW_target.reshape((N,N-1))/self.SPFLOW_target.sum()
        if S > 2:
            self.SPFLOW_target = self.SPFLOW_target.reshape((N,N-1,S-1))/self.SPFLOW_target.sum()
        if S == 2:
            self.SPFLOW_US_target = self.cc_moments.loc[1]['patent flows'].values/self.cc_moments.query("destination_code != origin_code")['patent flows'].sum()
            self.SPFLOW_RUS_target = (pd.DataFrame(self.cc_moments['patent flows']/self.cc_moments.loc[1]['patent flows']))
            self.SPFLOW_RUS_target = self.SPFLOW_RUS_target.query("destination_code != origin_code")['patent flows'].values.reshape((N,N-1))
        self.SPFLOWDOM_target = self.cc_moments['patent flows'].values
        if S == 2:
            self.SPFLOWDOM_target = self.SPFLOWDOM_target.reshape((N,N))/self.SPFLOWDOM_target.sum()
        if S > 2:
            self.SPFLOWDOM_target = self.SPFLOWDOM_target.reshape((N,N,S-1))/self.SPFLOWDOM_target.sum()
        if S == 2:
            self.SPFLOWDOM_US_target = self.SPFLOWDOM_target[0,0]
            self.SPFLOWDOM_RUS_target = self.SPFLOWDOM_target/self.SPFLOWDOM_US_target
        self.OUT_target = self.c_moments.expenditure.sum()/self.unit
        self.SRGDP_target = (self.c_moments.gdp/self.c_moments.price_level).values \
                            /(self.c_moments.gdp/self.c_moments.price_level).sum()
        self.SGDP_target = (self.c_moments.gdp).values \
                            /(self.c_moments.gdp).sum()
        self.RGDPPC_target = (self.c_moments.gdp/self.c_moments.price_level).values \
            / (self.c_moments.labor.values)
        self.RGDPPC_target = self.RGDPPC_target/self.RGDPPC_target[0]
        self.SRGDP_US_target = self.SRGDP_target[0]
        self.SRGDP_RUS_target = self.SRGDP_target/self.SRGDP_US_target
        self.RP_target = self.c_moments.price_level.values
        self.RD_target = self.c_moments.rnd_gdp.values
        if S == 4:
            self.country_sector_moments = pd.read_csv(data_path+'country_sector_moments.csv',index_col=[0])
            self.RDPHARMA_target = self.country_sector_moments['RD ratio pharma'].loc[[1,2,3,7,8]].values
            self.RDCHEM_target = self.country_sector_moments['RD ratio chemicals'].loc[[1,2,3,7,8]].values
        if S == 3:
            self.country_sector_moments = pd.read_csv(data_path+'country_sector_moments.csv',index_col=[0])
            self.RDPHARMACHEM_target = self.country_sector_moments['RD ratio pharma'].loc[[1,2,3,7,8]].values \
                                        + self.country_sector_moments['RD ratio chemicals'].loc[[1,2,3,7,8]].values
        self.RD_US_target = self.RD_target[0]
        self.RD_RUS_target = self.RD_target/self.RD_US_target
        self.KM_target = self.moments.loc['KM'].value
        self.KM_DD_DD_target = self.moments.loc['KM'].value
        if S == 4:
            self.KMPHARMA_target = self.moments.loc['KMPHARMA'].value
            self.KMCHEM_target = self.moments.loc['KMCHEM'].value
        if S == 3:
            self.KMPHARMACHEM_target = (self.moments.loc['KMPHARMA'].value + self.moments.loc['KMCHEM'].value)/2
        self.KM_GDP_target = self.KM_target*self.RD_US_target
        self.NUR_target = self.moments.loc['NUR'].value
        self.SRDUS_target = self.moments.loc['SRDUS'].value
        self.GPDIFF_target = self.moments.loc['GPDIFF'].value
        if S == 2:
            self.GPDIFF_target = 0.0242481 - np.array([0.0154756,0.0401137,0.0340597])
        if S == 3:
            self.GPDIFF_target = 0.0242481 - np.array([0.0154756,0.0370867])
        self.GROWTH_target = self.moments.loc['GROWTH'].value 
        self.ERDUS_target = self.moments.loc['ERDUS'].value 
        self.PROBINNOVENT_target = self.moments.loc['PROBINNOVENT'].value 
        self.SHAREEXPMON_target = self.moments.loc['SHAREEXPMON'].value 
        self.TE_target = self.moments.loc['TE'].value 
        self.TO_target = self.moments.loc['TO'].value
        self.TO_DD_DD_target = self.moments.loc['TO'].value
        if S == 4:
            self.TOPHARMACHEM_target = np.array([np.nan])
            self.TEPHARMACHEM_target = np.array([np.nan])
            self.TOCHEM_target = self.moments.loc['TOCHEM'].value 
            self.TOPHARMA_target = self.moments.loc['TOPHARMA'].value 
            self.TECHEM_target = self.moments.loc['TECHEM'].value 
            self.TEPHARMA_target = self.moments.loc['TEPHARMA'].value 
        elif S == 3:
            self.TOPHARMACHEM_target = (self.moments.loc['TOPHARMA'].value + self.moments.loc['TOCHEM'].value)/2
            self.TEPHARMACHEM_target = (self.moments.loc['TEPHARMA'].value + self.moments.loc['TECHEM'].value)/2
            self.TOCHEM_target = np.array([np.nan])
            self.TOPHARMA_target = np.array([np.nan])
            self.TECHEM_target = np.array([np.nan])
            self.TEPHARMA_target = np.array([np.nan])
        else:
            self.TOPHARMACHEM_target = np.array([np.nan])
            self.TEPHARMACHEM_target = np.array([np.nan])
            self.TOCHEM_target = np.array([np.nan])
            self.TOPHARMA_target = np.array([np.nan])
            self.TECHEM_target = np.array([np.nan])
            self.TEPHARMA_target = np.array([np.nan])
        try:
            self.PCOSTINTER_target = (self.pat_fees['fee'].values*self.cc_moments.query(
                "destination_code != origin_code"
                )['patent flows'].groupby('destination_code').sum().values).sum()/1e12
        except:
            self.PCOSTINTER_target = (self.pat_fees['fee'].values*self.cc_moments.query(
                "destination_code != origin_code"
                )['patent flows'].groupby('destination_code').sum().values[:self.pat_fees['fee'].values.shape[0]]).sum()/1e12
        self.PCOST_target = self.PCOSTINTER_target+\
            self.pat_fees.loc[1,'fee']*self.cc_moments.loc[(1,1),'patent flows']/1e12+\
            self.pat_fees.loc[2,'fee']*self.cc_moments.loc[(2,2),'patent flows']/1e12
        self.PCOSTINTERNOAGG_target = self.PCOSTINTER_target\
            -self.pat_fees.loc[2,'fee']*self.cc_moments.query(
                "destination_code != origin_code"
                ).loc[2,'patent flows'].sum()/1e12\
            -self.pat_fees.loc[7,'fee']*self.cc_moments.query(
                        "destination_code != origin_code"
                ).loc[7,'patent flows'].sum()/1e12
        self.PCOSTNOAGG_target = self.PCOSTINTERNOAGG_target+\
            self.pat_fees.loc[1,'fee']*self.cc_moments.loc[(1,1),'patent flows']/1e12
        self.Z_target = self.c_moments.expenditure.values/self.unit
        self.JUPCOST_target = self.moments.loc['JUPCOST'].value
        self.UUPCOST_target = self.moments.loc['UUPCOST'].value
        if S>2:
            self.UUPCOSTS_target = self.sector_moments.UUPCOSTS.values[1:]
        if S==3:
            self.UUPCOSTS_target = self.sector_moments.UUPCOSTS.values[1:S+1]
            self.UUPCOSTS_target[1] = self.sector_moments.UUPCOSTS.values[2::].sum()
        self.JUPCOSTRD_target = self.moments.loc['JUPCOST'].value/(self.c_moments.loc[1,'rnd_gdp']*self.c_moments.loc[1,'gdp']/self.unit)
        self.TP_target = self.moments.loc['TP'].value
        try:
            self.AGGAVMARKUP_target = self.moments.loc['AGGAVMARKUP'].value
            self.AVMARKUPPHARCHEM_target = self.moments.loc['AVMARKUPPHARCHEM'].value
        except:
            pass
        self.inter_TP_target = np.array(0.00117416)
        self.TP_data = self.cc_moments['patent flows'].sum()
        self.DOMPATEU_target = self.cc_moments.loc[(2,2),'patent flows']/self.cc_moments.xs(2,level=1)['patent flows'].sum()
        self.DOMPATUS_target = self.cc_moments.loc[(1,1),'patent flows']/self.cc_moments.xs(1,level=1)['patent flows'].sum()
        self.DOMPATINEU_target = self.cc_moments.loc[(2,2),'patent flows']/self.cc_moments.xs(2,level=0)['patent flows'].sum()
        self.DOMPATINUS_target = self.cc_moments.loc[(1,1),'patent flows']/self.cc_moments.xs(1,level=0)['patent flows'].sum()
        self.inter_TP_data = self.cc_moments.query("destination_code != origin_code")['patent flows'].sum()
        self.SINNOVPATEU_target = self.moments.loc['SINNOVPATEU'].value
        self.SINNOVPATUS_target = np.array([self.moments.loc['SINNOVPATUS'].value])[0]#*(S-1))
        self.SDOMTFLOW_target = self.ccs_moments.query("destination_code == origin_code").trade.values/self.ccs_moments.trade.sum()
        self.SDOMTFLOW_target = self.SDOMTFLOW_target.reshape(N,S)#/self.unit
        self.sales_mark_up_US = self.moments.loc['sales_mark_up_US'].value
        self.sales_mark_up_US_target = self.moments.loc['sales_mark_up_US'].value
        self.DOMPATRATUSEU_target = (self.cc_moments.query("destination_code == origin_code")['patent flows']\
            /(self.cc_moments.loc[1]['patent flows'].sum() + self.cc_moments.loc[2]['patent flows'].sum())).values
        self.SPATORIG_target = self.cc_moments['patent flows'].groupby('origin_code').sum().values\
            /self.cc_moments['patent flows'].sum()
        self.SPATDEST_target = self.cc_moments['patent flows'].groupby('destination_code').sum().values\
            /self.cc_moments['patent flows'].sum()
        if S == 2:
            self.TWSPFLOW_target = self.SPFLOW_target*self.ccs_moments.loc[:,:,1].query("destination_code != origin_code")['trade'].values.reshape((N,N-1))\
                /self.ccs_moments.loc[:,:,1].query("destination_code != origin_code")['trade'].sum()
            self.TWSPFLOWDOM_target = self.SPFLOWDOM_target*self.ccs_moments.loc[:,:,1]['trade'].values.reshape((N,N))\
                /self.ccs_moments.loc[:,:,1]['trade'].sum()
            
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'KM_DD_DD':pd.Index(['scalar']), 
                    'KMCHEM':pd.Index(['scalar']), 
                    'KMPHARMA':pd.Index(['scalar']), 
                    'KMPHARMACHEM':pd.Index(['scalar']), 
                    'KM_GDP':pd.Index(['scalar']), 
                    'OUT':pd.Index(['scalar']), 
                    'RD':pd.Index(self.countries, name='country'), 
                    'RDCHEM':pd.Index(['USA', 'EUR', 'JAP', 'CAN', 'KOR'], name='country'), 
                    'RDPHARMA':pd.Index(['USA', 'EUR', 'JAP', 'CAN', 'KOR'], name='country'), 
                    'RDPHARMACHEM':pd.Index(['USA', 'EUR', 'JAP', 'CAN', 'KOR'], name='country'), 
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
                    'UUPCOST':pd.Index(['scalar']), 
                    'AGGAVMARKUP':pd.Index(['scalar']), 
                    'AVMARKUPPHARCHEM':pd.Index(['scalar']), 
                    'UUPCOSTS':pd.Index(self.sectors[1:],name='sector'), 
                    'PCOSTNOAGG':pd.Index(['scalar']), 
                    'PCOSTINTERNOAGG':pd.Index(['scalar']), 
                    'PCOST':pd.Index(['scalar']), 
                    'PCOSTINTER':pd.Index(['scalar']), 
                    'JUPCOSTRD':pd.Index(['scalar']), 
                    'SRGDP':pd.Index(self.countries, name='country'), 
                    'SGDP':pd.Index(self.countries, name='country'), 
                    'RGDPPC':pd.Index(self.countries, name='country'), 
                    'SRGDP_US':pd.Index(['scalar']), 
                    'SRGDP_RUS':pd.Index(self.countries, name='country'), 
                    'STFLOW':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                      , names=['destination','origin','sector']),
                    'SDFLOW':pd.MultiIndex.from_product([self.countries,self.sectors[2:]]
                                                      , names=['country','sector']),
                    'STFLOWSDOM':pd.MultiIndex.from_product([self.countries,self.countries,self.sectors]
                                                      , names=['destination','origin','sector']),
                    'SDOMTFLOW':pd.MultiIndex.from_product([self.countries,self.sectors]
                                                      , names=['country','sector']),
                    'TP':pd.Index(['scalar']),
                    'inter_TP':pd.Index(['scalar']),
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
                    'TO_DD_DD':pd.Index(['scalar']),
                    'TE':pd.Index(['scalar']),
                    'TOPHARMACHEM':pd.Index(['scalar']),
                    'TEPHARMACHEM':pd.Index(['scalar']),
                    'TOPHARMA':pd.Index(['scalar']),
                    'TEPHARMA':pd.Index(['scalar']),
                    'TOCHEM':pd.Index(['scalar']),
                    'TECHEM':pd.Index(['scalar']),
                    'DOMPATUS':pd.Index(['scalar']),
                    'DOMPATEU':pd.Index(['scalar']),
                    'DOMPATINUS':pd.Index(['scalar']),
                    'DOMPATINEU':pd.Index(['scalar']),
                    'NUR':pd.Index(['scalar']),
                    'ERDUS':pd.Index(['scalar']),
                    'PROBINNOVENT':pd.Index(['scalar']),
                    'SHAREEXPMON':pd.Index(['scalar'])
                    }
        
        if S>2:
            # self.idx['SPFLOW'] = pd.MultiIndex.from_product([self.countries,self.countries,self.sectors[1:]]
            #                                   , names=['destination','origin','sector'])
            self.idx['SPFLOW'] = pd.MultiIndex.from_tuples(
                                                                                [
                                                                                    (dest, orig, sector)
                                                                                    for dest in self.countries
                                                                                    for orig in self.countries
                                                                                    for sector in self.sectors[1:]
                                                                                    if dest != orig
                                                                                ],
                                                                                names=['destination', 'origin', 'sector']
                                                                            )
            self.idx['GPDIFF'] = pd.Index(self.sectors[1:], name='sector')
            self.idx['DOMPATINUS'] = pd.Index(self.sectors[1:], name='sector')
        
        self.shapes = {'SPFLOW':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM':(len(self.countries),len(self.countries)),
                       'SPFLOW_RUS':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM_RUS':(len(self.countries),len(self.countries)),
                       'SDOMTFLOW':(len(self.countries),len(self.sectors)),
                       'STFLOWSDOM':(len(self.countries),len(self.countries),len(self.sectors)),
                       'TWSPFLOW':(len(self.countries),len(self.countries)-1),
                       'TWSPFLOWDOM':(len(self.countries),len(self.countries)),
                       }
        
        if S>2:
            self.shapes['SPFLOW'] = (len(self.countries),len(self.countries)-1,len(self.sectors)-1)
            self.shapes['SDFLOW'] = (len(self.countries),len(self.sectors)-2)
    
    def load_run(self,path,dir_path=None):
        if dir_path is None:
            dir_path = './'
        try:
            # df = pd.read_csv(path+'data_path.csv',header=None)
            df = pd.read_csv(path+'data_path.csv',index_col=0)
            setattr(self,'N',int(df.loc['nbr_of_countries','run']))
            setattr(self,'S',int(df.loc['nbr_of_sectors','run']))
            setattr(self,'data_path',df.loc['data_path','run'])
        except:
            setattr(self,'N',7)
            setattr(self,'S',2)
            setattr(self,'data_path','data/data_leg/')
        
        self.load_data(self.data_path,dir_path=dir_path)
            
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
        
    def compute_STFLOW(self,var,p):
        self.STFLOW = (var.X/(1+p.tariff))/(var.X/(1+p.tariff)).sum()
        
    def compute_SDFLOW(self,var,p):
        if p.S > 2:
            self.SDFLOW = np.einsum('nns->ns',var.X[:,:,2:]/(1+p.tariff[:,:,2:])
                                    )/np.einsum('nn->n',var.X[:,:,1]/(1+p.tariff[:,:,1]))[:,None]
        if p.S == 2:
            self.SDFLOW = np.nan
        
    def compute_STFLOWSDOM(self,var,p):
        self.STFLOWSDOM = (var.X/(1+p.tariff))/np.einsum('nns->ns',var.X/(1+p.tariff))[:,None,:]
        
    def compute_SPFLOW(self,var,p):
        if p.S == 2:
            pflow = var.pflow
            self.SPFLOWDOM = pflow/(pflow).sum()
            inter_pflow = remove_diag(var.pflow)
            self.SPFLOW = inter_pflow/inter_pflow.sum()
            
            self.SPFLOW_US = pflow[0,:]/inter_pflow.sum()
            RUS = pflow/pflow[0,:]
            self.SPFLOW_RUS = remove_diag(RUS)
            
            self.SPFLOWDOM_US = self.SPFLOWDOM[0,0]
            self.SPFLOWDOM_RUS = self.SPFLOWDOM/self.SPFLOWDOM_US
            
        if p.S > 2:
            pflow = var.pflow
            self.SPFLOWDOM = pflow/(pflow).sum()
            inter_pflow = remove_diag(var.pflow)
            self.SPFLOW = inter_pflow/inter_pflow.sum()
        
    def compute_TWSPFLOW(self,var,p):
        pflow = var.pflow/(1+p.tariff[...,1])
        self.TWSPFLOWDOM = pflow*p.trade_flows[...,1]/(pflow.sum()*p.trade_flows[...,1].sum())
        inter_pflow = remove_diag(var.pflow/(1+p.tariff[...,1]))
        off_diag_trade_flows = remove_diag(p.trade_flows[...,1]/(1+p.tariff[...,1]))
        self.TWSPFLOW = inter_pflow*off_diag_trade_flows/(inter_pflow.sum()*off_diag_trade_flows.sum())        
        
    def compute_OUT(self,var,p):
        self.OUT = (var.X/(1+p.tariff)).sum()
        
    def compute_SRGDP(self,var,p):
        numerator = var.gdp/var.price_indices
        self.SRGDP = numerator/numerator.sum()
        self.SRGDP_US = self.SRGDP[0]
        self.SRGDP_RUS = self.SRGDP/self.SRGDP_US
        
    def compute_SGDP(self,var,p):
        numerator = var.gdp
        self.SGDP = numerator/numerator.sum()
        
    def compute_RGDPPC(self,var,p):
        self.RGDPPC = var.gdp / var.price_indices / p.labor
        self.RGDPPC = self.RGDPPC/self.RGDPPC[0]
        
    def compute_RP(self,var,p):
        self.RP = var.price_indices/var.price_indices[0]
        
    def compute_RD(self,var,p):
        numerator = var.w[:,None]*var.l_R + np.einsum('i,is->is',var.w,var.l_Ao)\
            + np.einsum('n,ins->is',var.w,var.l_Ae)
        self.RD = np.einsum('is,i->i',
                            numerator,
                            1/var.gdp)
        self.RD_US = self.RD[0]
        self.RD_RUS = self.RD/self.RD_US
        
        if p.S == 4:
            self.RDPHARMA = np.einsum('is,i->is',
                                numerator,
                                1/var.gdp)[:,2][[1,2,3,7,8]] / \
                            np.einsum('is,i->i',
                                numerator,
                                1/var.gdp)[[1,2,3,7,8]]
            self.RDCHEM = np.einsum('is,i->is',
                                numerator,
                                1/var.gdp)[:,3][[1,2,3,7,8]] / \
                            np.einsum('is,i->i',
                                numerator,
                                1/var.gdp)[[1,2,3,7,8]]
        if p.S == 3:
            self.RDPHARMACHEM = np.einsum('is,i->is',
                                numerator,
                                1/var.gdp)[:,2][[1,2,3,7,8]] / \
                            np.einsum('is,i->i',
                                numerator,
                                1/var.gdp)[[1,2,3,7,8]]
    
    def compute_KM(self,var,p):
        # bracket = 1/(var.G[None,1:]+p.delta[:,1:]-p.nu[1:]) - 1/(var.G[None,1:]+p.delta[:,1:])
        # self.KM = p.k/(p.k-1)*np.einsum('s,s,ns,ns,ns->',
        #     p.eta[0,1:],
        #     var.l_R[0,1:]**(1-p.kappa),
        #     var.psi_m_star[:,0,1:]**(1-p.k),
        #     var.profit[:,0,1:],
        #     bracket,
        #     )/(var.l_R[0,1:].sum()+var.l_Ao[0,1:].sum()+(var.w[:,None]*var.l_Ae[0,:,1:]/var.w[0]).sum())
        bracket = 1/(var.G[None,1:]+p.delta[:,1:]-p.nu[None,1:]) - 1/(var.G[None,1:]+p.delta[:,1:])
        KM = np.einsum('s,is,is,nis,nis,ns,i->ni',
            p.k[1:]/(p.k[1:]-1),
            p.eta[:,1:],
            var.l_R[:,1:]**(1-p.kappa),
            var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
            var.profit[:,:,1:],
            bracket,
            1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
            )
        self.KM = KM[0,0]
        self.KM_GDP = self.KM*self.RD_US
        
        G = var.r+p.zeta-var.g+var.g_s+p.nu+p.nu_tilde
        bracket = 1/(G[None,None,1:]+p.delta_eff[:,:,1:]-p.nu[None,None,1:]-p.nu_tilde[None,None,1:]) \
            - 1/(G[None,None,1:]+p.delta_eff[:,:,1:])
        KM_DD_DD = np.einsum('s,is,is,nis,nis,nis,i->ni',
            p.k[1:]/(p.k[1:]-1),
            p.eta[:,1:],
            var.l_R[:,1:]**(1-p.kappa),
            var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
            var.profit[:,:,1:],
            bracket,
            1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
            )
        self.KM_DD_DD = KM_DD_DD[0,0]
        
        if p.S==4:
            KM = np.einsum('s,is,is,nis,nis,ns,i->nis',
                p.k[1:]/(p.k[1:]-1),
                p.eta[:,1:],
                var.l_R[:,1:]**(1-p.kappa),
                var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                var.profit[:,:,1:],
                bracket,
                1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                )
            self.KM = KM[0,0,0]
            self.KMPHARMA = KM[0,0,1]
            self.KMCHEM = KM[0,0,2]
            
            if self.aggregate_moments:
                self.KM = np.einsum('s,is,is,nis,nis,ns,i->ni',
                    p.k[1:]/(p.k[1:]-1),
                    p.eta[:,1:],
                    var.l_R[:,1:]**(1-p.kappa),
                    var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                    var.profit[:,:,1:],
                    bracket,
                    1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                    )[0,0]
        if p.S==3:
            KM = np.einsum('s,is,is,nis,nis,ns,i->nis',
                p.k[1:]/(p.k[1:]-1),
                p.eta[:,1:],
                var.l_R[:,1:]**(1-p.kappa),
                var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                var.profit[:,:,1:],
                bracket,
                1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                )
            self.KM = KM[0,0,0]
            self.KMPHARMACHEM = KM[0,0,1]
            
            if self.aggregate_moments:
                self.KM = np.einsum('s,is,is,nis,nis,ns,i->ni',
                    p.k[1:]/(p.k[1:]-1),
                    p.eta[:,1:],
                    var.l_R[:,1:]**(1-p.kappa),
                    var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                    var.profit[:,:,1:],
                    bracket,
                    1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                    )[0,0]
        
    def compute_SRDUS(self,var,p):
        self.SRDUS = (var.X_M[:,0,1]/(1+p.tariff[:,0,1])).sum()/(var.X[:,0,1]/(1+p.tariff[:,0,1])).sum()
    
    def compute_GPDIFF(self,var,p):
        price_index_growth_rate = var.g_s/(1-p.sigma)+p.alpha*var.g
        if p.S == 2:
            self.GPDIFF = price_index_growth_rate[0] - price_index_growth_rate[1]
        if p.S > 2:
            self.GPDIFF = price_index_growth_rate[0] - price_index_growth_rate[1:]
        
    def compute_GROWTH(self,var,p):
        self.GROWTH = var.g    
    
    def compute_Z(self,var,p):
        self.Z = var.Z
    
    def compute_JUPCOST(self,var,p):
        self.JUPCOST = var.pflow[2,0]*p.r_hjort[2]*p.fe[1]*var.w[2]
        self.JUPCOSTRD = self.JUPCOST/(self.RD[0]*var.gdp[0])
        
    def compute_UUPCOST(self,var,p):
        if p.S == 2:
            self.UUPCOST = var.pflow[0,0]*p.r_hjort[0]*p.fe[1]*var.w[0]
        if p.S > 2:
            self.UUPCOST = var.pflow[0,0].sum()*p.r_hjort[0]*p.fe[1]*var.w[0]
            self.UUPCOSTS = var.pflow[0,0,:]*p.r_hjort[0]*p.fe[1]*var.w[0]
        
    def compute_PCOSTINTER(self,var,p):
        off_diag_pflow = var.pflow.copy()
        np.einsum('nn->n',off_diag_pflow)[...] = 0
        self.PCOSTINTER = np.einsum('ni,n,,n->',
                               off_diag_pflow,
                               p.r_hjort,
                               p.fe[1],
                               var.w)
        
    def compute_PCOST(self,var,p):
        self.PCOST = self.PCOSTINTER + var.pflow[0,0]*p.r_hjort[0]*p.fe[1]*var.w[0]\
            + var.pflow[1,1]*p.r_hjort[1]*p.fe[1]*var.w[1]
            
    def compute_PCOSTINTERNOAGG(self,var,p):
        off_diag_pflow = var.pflow.copy()
        np.einsum('nn->n',off_diag_pflow)[...] = 0
        self.PCOSTINTERNOAGG = self.PCOSTINTER - off_diag_pflow[1,:].sum()*p.r_hjort[1]*p.fe[1]*var.w[1]\
            - off_diag_pflow[6,:].sum()*p.r_hjort[6]*p.fe[1]*var.w[6]
        
    def compute_PCOSTNOAGG(self,var,p):
        self.PCOSTNOAGG = self.PCOSTINTERNOAGG + var.pflow[0,0]*p.r_hjort[0]*p.fe[1]*var.w[0]
        
    def compute_TP(self,var,p):
        self.TP = var.pflow.sum()
        inter_pflow = remove_diag(var.pflow)
        self.inter_TP = inter_pflow.sum()
        
    def compute_SDOMTFLOW(self,var,p):
        self.SDOMTFLOW = np.diagonal(var.X).transpose()/var.X.sum()
    
    def compute_SINNOVPATEU(self,var,p):
        self.SINNOVPATEU = var.share_innov_patented[1,1]
        
    def compute_SINNOVPATUS(self,var,p):
        # self.SINNOVPATUS = var.share_innov_patented[0,0,:]
        self.SINNOVPATUS = var.share_innov_patented[0,0,0]
        if p.S > 2:
            if self.aggregate_moments:
                self.SINNOVPATUS = np.einsum('ns,ns,nns->n',
                    p.eta[:,1:],
                    var.l_R[:,1:]**(1-p.kappa),
                    var.share_innov_patented
                    )[0] / np.einsum('ns,ns->n',
                        p.eta[:,1:],
                        var.l_R[:,1:]**(1-p.kappa),
                        )[0]
                
        
    def compute_TO(self,var,p):
        delt = 5
        self.delta_t = delt
        PHI = var.phi**p.theta[None,None,:]
        
        num_brack_B = var.PSI_MNP*eps(p.nu*delt)[None,None,:]
        num_brack_C = var.PSI_MPND*(eps(p.delta*delt)*eps(p.nu*delt)[None,:])[:,None,:]
        num_brack_E = var.PSI_MPD*eps(p.nu*delt)[None,None,:]
        
        num_brack = (num_brack_B + num_brack_C + num_brack_E)
        
        num_sum = np.einsum('nis,njs->ns',
                            num_brack,
                            PHI
                            ) - \
                  np.einsum('ns,njs->ns',
                            np.diagonal(num_brack).transpose(),
                            PHI
                            ) - \
                  np.einsum('nis,ns->ns',
                            num_brack,
                            np.diagonal(PHI).transpose()
                            ) + \
                  np.einsum('ns,ns->ns',
                            np.diagonal(num_brack).transpose(),
                            np.diagonal(PHI).transpose()
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
        denom_B_c = var.PSI_MPD*np.exp(-delt*p.delta)[:,None,:]
        denom_B = np.einsum('nis,nis,s->nis',
                            denom_B_a + denom_B_b + denom_B_c,
                            var.phi**(p.sigma-1)[None,None,:],
                            (p.sigma/(p.sigma-1))**(1-p.sigma)
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
                          denom_A + denom_B + denom_D
                          ) - np.einsum('nns->ns',
                                            denom_A + denom_B + denom_D
                                            ) 
        
        self.turnover = num/denom
        self.TO = self.turnover[0,1]
        if p.S==4:
            self.TOPHARMA = self.turnover[0,2]
            self.TOCHEM = self.turnover[0,3]
            self.TOPHARMACHEM = np.nan
        elif p.S==3:
            self.TOPHARMA = np.nan
            self.TOCHEM = np.nan
            self.TOPHARMACHEM = self.turnover[0,2]
        else:
            self.TOPHARMA = np.nan
            self.TOCHEM = np.nan
            self.TOPHARMACHEM = np.nan
            
    def compute_TO_DD_DD(self,var,p):
        delt = 5
        self.delta_t = delt
        PHI = var.phi**p.theta[None,None,:]
        
        num_brack_B = var.PSI_MNP*eps(p.nu_tilde*delt)[None,None,:]
        num_brack_C = var.PSI_MPND*(eps(p.delta_eff*delt)*eps(p.nu_tilde*delt)[None,None,:])
        num_brack_D = var.PSI_MPD*eps(p.delta_eff*delt)
        num_brack_E = var.PSI_MPL*(eps(p.delta_eff*delt)*eps(p.nu_tilde*delt)[None,None,:])
        num_brack_F = var.PSI_CL*eps(p.nu_tilde*delt)[None,None,:]
        
        num_brack = (num_brack_B + num_brack_C + num_brack_D + num_brack_E + num_brack_F)
        
        num_sum = np.einsum('nis,njs->ns',
                            num_brack,
                            PHI
                            ) - \
                  np.einsum('ns,njs->ns',
                            np.diagonal(num_brack).transpose(),
                            PHI
                            ) - \
                  np.einsum('nis,ns->ns',
                            num_brack,
                            np.diagonal(PHI).transpose()
                            ) + \
                  np.einsum('ns,ns->ns',
                            np.diagonal(num_brack).transpose(),
                            np.diagonal(PHI).transpose()
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
        
        denom_B_a = var.PSI_MNP*np.exp(-delt*(p.nu+p.nu_tilde))[None,None,:]
        denom_B_b = var.PSI_MPND*(np.exp(-delt*p.delta_eff)
                                  +np.exp(-(p.nu+p.nu_tilde)*delt)[None,None,:]*eps(delt*p.delta_eff))
        denom_B_c = (var.PSI_MPD+var.PSI_MPL)*np.exp(-delt*p.delta_eff)
        denom_B = np.einsum('nis,nis,s->nis',
                            denom_B_a + denom_B_b + denom_B_c,
                            var.phi**(p.sigma-1)[None,None,:],
                            (p.sigma/(p.sigma-1))**(1-p.sigma)
                            )
        
        denom_C_a = var.PSI_MNP*eps(p.nu)[None,None,:]
        denom_C_b = var.PSI_MPND*(eps(p.delta_eff*delt)*eps(p.nu*delt)[None,None,:])
        denom_C_c = var.PSI_MPL*eps(p.delta_eff*delt)
        denom_C = np.einsum('s,nis,nis,s->nis',
                            np.exp(-delt*p.nu_tilde),
                            denom_C_a + denom_C_b + denom_C_c,
                            var.phi**(p.sigma-1)[None,None,:],
                            (p.sigma/(p.sigma-1))**(1-p.sigma)
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
                          ) - np.einsum('nns->ns',
                                            denom_A + denom_B + denom_C + denom_D
                                            ) 
        
        self.turnover_DD_DD = num/denom
        self.TO_DD_DD = self.turnover_DD_DD[0,1]
        
    def compute_TE(self,var,p):
        out_diag_trade_flows_shares = remove_diag(var.X_M/var.X)
        self.TE = ( (p.theta[None,None,:] - np.einsum('s,nis->nis',
                                                    p.theta-(p.sigma-1),
                                                    out_diag_trade_flows_shares)
                    ).sum(axis=1).sum(axis=0) )[1]/(p.N*(p.N-1))
        if p.S==4:
            self.TEPHARMA = ( (p.theta[None,None,:] - np.einsum('s,nis->nis',
                                                        p.theta-(p.sigma-1),
                                                        out_diag_trade_flows_shares)
                        ).sum(axis=1).sum(axis=0) )[2]/(p.N*(p.N-1))
            self.TECHEM = ( (p.theta[None,None,:] - np.einsum('s,nis->nis',
                                                        p.theta-(p.sigma-1),
                                                        out_diag_trade_flows_shares)
                        ).sum(axis=1).sum(axis=0) )[3]/(p.N*(p.N-1))
            self.TEPHARMACHEM = np.nan
        if p.S==3:
            self.TEPHARMA = np.nan
            self.TECHEM = np.nan
            self.TEPHARMACHEM = ( (p.theta[None,None,:] - np.einsum('s,nis->nis',
                                                        p.theta-(p.sigma-1),
                                                        out_diag_trade_flows_shares)
                        ).sum(axis=1).sum(axis=0) )[2]/(p.N*(p.N-1))
            # if self.aggregate_moments:
            #     weights = remove_diag(var.X / var.X.sum(axis=-1)[:,:,None])
            #     self.TE = ( 
            #         (weights *
            #         (p.theta[None,None,:] - np.einsum('s,nis->nis',
            #                                                 p.theta-(p.sigma-1),
            #                                                 out_diag_trade_flows_shares)
            #                 )
            #         ).sum(axis=1).sum(axis=0) 
            #         )[1]/(p.N*(p.N-1))
                
        else:
            self.TEPHARMACHEM = np.nan
            self.TEPHARMA = np.nan
            self.TECHEM = np.nan
        
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
        
    def compute_DOMPATINEU(self,var,p):
        self.DOMPATINEU = var.pflow[1,1]/var.pflow[1,:].sum()
        
    def compute_DOMPATINUS(self,var,p):
        self.DOMPATINUS = var.pflow[0,0]/var.pflow[0,:].sum()
        
    def compute_ERDUS(self,var,p):
        self.ERDUS = var.semi_elast_patenting_delta[0,1]
        
    def compute_SHAREEXPMON(self,var,p):
        numerator = ((var.X_M[...,1]/(1+p.tariff[...,1])).sum(axis=0) - np.diagonal(var.X_M[...,1]/(1+p.tariff[...,1])))
        denominator = (var.X_M[...,1]/(1+p.tariff[...,1])).sum(axis=0)
        self.SHAREEXPMON = numerator[0] / denominator[0]
        
    def compute_AGGAVMARKUP(self,var,p):
        self.AGGAVMARKUP = var.aggregate_average_markup[0]
        self.AVMARKUPPHARCHEM = var.sectoral_average_markup[0,1] / var.aggregate_average_markup[0]
        
    def compute_PROBINNOVENT(self,var,p):
        self.PROBINNOVENT = np.nan
        try:
            # to be updated for other sectors
            def aleph_P_star(psi):
                res = np.maximum(
                                np.einsum('i,ni,ni->ni',
                                          var.w,
                                          var.a[...,1],
                                          1/(psi*var.V_P[...,1]-np.einsum('n,,n->n',
                                                                        var.w,
                                                                        p.fe[1],
                                                                        p.r_hjort)[:,None])
                                          ),
                                1
                                )
                return res
            
            def aleph_NP_star(psi):
                res = np.maximum(
                                np.einsum('i,ni,,ni->ni',
                                          var.w,
                                          var.a[...,1],
                                          1/psi,
                                          1/var.V_NP[...,1]
                                          ),
                                1
                                )
                return res
            
            def integrand_US(psi):
                inside_min = (aleph_P_star(psi) * (psi >= var.psi_m_star[...,1])) + (aleph_NP_star(psi) * (psi <= var.psi_m_star[...,1]))
                res = ( p.k*psi**(-p.k-1)*np.min( inside_min[1:,0] )**(-p.d) )
                return res
            
            self.PROBINNOVENT = integrate.quad(integrand_US,1,np.inf)[0]
            
            def integrand_JAP(psi):
                inside_min = (aleph_P_star(psi) * (psi >= var.psi_m_star[...,1])) + (aleph_NP_star(psi) * (psi <= var.psi_m_star[...,1]))
                mask = np.ones(p.N)
                mask = (mask == 1)
                mask[2]=False
                res = ( p.k*psi**(-p.k-1)*np.min( inside_min[mask,2] )**(-p.d) )
                return res
            
            self.PROBINNOVENT_JAP = integrate.quad(integrand_JAP,1,np.inf)[0]
            
            # def integrand_JAP(psi):
            #     # signature_NP = (psi <= var.psi_m_star[...,1])
            #     term_1 = np.zeros_like(var.psi_m_star[...,1])
            #     term_1[psi <= var.psi_m_star[...,1]] = aleph_NP_star(psi)[psi <= var.psi_m_star[...,1]]
            #     term_2 = np.zeros_like(var.psi_m_star[...,1])
            #     term_2[psi >= var.psi_m_star[...,1]] = aleph_P_star(psi)[psi >= var.psi_m_star[...,1]]
            #     # signature_P = (psi >= var.psi_m_star[...,1])
            #     inside_min = (aleph_P_star(psi) * (psi >= var.psi_m_star[...,1])) + (aleph_NP_star(psi) * (psi <= var.psi_m_star[...,1]))
            #     # inside_min = np.maximum(aleph_P_star(psi) * (psi >= var.psi_m_star[...,1]),1)
            #     # res = p.k*psi**(-p.k-1)*np.min(inside_min[:,0])
            #     # res = ( p.k*psi**(-p.k-1)*np.min( inside_min[1:,2] )**(-p.d) )
            #     mask = np.ones(p.N)
            #     mask = (mask == 1)
            #     mask[2]=False
            #     res = ( p.k*psi**(-p.k-1)*np.min( inside_min[mask,2] )**(-p.d) )
            #     # res = ( p.k*psi**(-p.k-1)*np.min( (term_1[1:,0] + term_2[1:,0]) )**(-p.d) )
            #     # res = ( p.k*psi**(-p.k-1)*np.min( inside_min[1:,0] )**(-p.d) )
            #     return res
            
            # self.PROBINNOVENT_JAP = integrate.quad(integrand_JAP,1,np.inf)[0]
        except:
            pass
        
        
    def compute_moments(self,var,p):
        if p.S == 2:
            self.compute_STFLOW(var, p)
            self.compute_STFLOWSDOM(var, p)
            self.compute_SPFLOW(var, p)
            self.compute_OUT(var, p)
            self.compute_SRGDP(var, p)
            self.compute_SGDP(var, p)
            self.compute_RGDPPC(var, p)
            self.compute_RP(var, p)
            self.compute_RD(var, p)
            self.compute_KM(var, p)
            self.compute_SRDUS(var, p)
            self.compute_GPDIFF(var, p)
            self.compute_GROWTH(var, p)
            self.compute_JUPCOST(var, p)
            self.compute_UUPCOST(var, p)
            self.compute_PCOSTINTER(var,p)
            self.compute_PCOST(var,p)
            self.compute_PCOSTINTERNOAGG(var,p)
            self.compute_PCOSTNOAGG(var,p)
            self.compute_TP(var,p)
            self.compute_Z(var,p)
            self.compute_SDOMTFLOW(var,p)
            self.compute_SINNOVPATEU(var,p)
            self.compute_SINNOVPATUS(var,p)
            self.compute_NUR(var,p)
            self.compute_TO(var,p)
            self.compute_TO_DD_DD(var,p)
            self.compute_TE(var,p)
            self.compute_DOMPATRATUSEU(var,p)
            self.compute_SPATDEST(var,p)
            self.compute_SPATORIG(var,p)
            self.compute_TWSPFLOW(var, p)
            self.compute_DOMPATEU(var, p)
            self.compute_DOMPATUS(var, p)
            self.compute_DOMPATINEU(var, p)
            self.compute_DOMPATINUS(var, p)
            # self.compute_ERDUS(var, p)
            self.compute_SDFLOW(var, p)
            # self.compute_PROBINNOVENT(var, p)
            # self.compute_SHAREEXPMON(var, p)
        if p.S > 2:
            self.compute_SPFLOW(var, p)
            self.compute_OUT(var, p)
            self.compute_SRGDP(var, p)
            self.compute_RP(var, p)
            self.compute_RD(var, p)
            self.compute_KM(var, p)
            self.compute_GPDIFF(var, p)
            self.compute_GROWTH(var, p)
            self.compute_UUPCOST(var, p)
            self.compute_Z(var,p)
            self.compute_SINNOVPATUS(var,p)
            self.compute_TO(var,p)
            self.compute_TE(var,p)
            self.compute_DOMPATINUS(var, p)
            self.compute_SDFLOW(var, p)
            self.compute_AGGAVMARKUP(var, p)
        
    def compute_moments_deviations(self):

        for mom in self.get_list_of_moments():
            if hasattr(self, mom):
                # print(mom)
                distort_for_large_pflows_fac = 6
                # if mom != 'GPDIFF' and mom != 'TO' and mom != 'TE' and mom != 'GROWTH' and mom != 'OUT':
                if mom != 'GPDIFF' and mom != 'TO' and mom != 'TE' and mom != 'GROWTH' and mom != 'OUT' and mom != 'SPFLOW' and mom != 'SPFLOWDOM':
                    # setattr(self,
                    #         mom+'_deviation',
                    #         self.weights_dict[mom]*np.log(np.abs(getattr(self,mom)/getattr(self,mom+'_target')))
                    #         /np.log(getattr(self,mom+'_target').size+1)
                    #         )
                    # setattr(self,
                    #         mom+'_deviation',
                    #         self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))**(1/2)
                    #         )
                    if self.loss == 'log':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    /getattr(self,mom+'_target').size
                                    )
                    if self.loss == 'ratio':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))/getattr(self,mom+'_target')
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))/getattr(self,mom+'_target')
                                    /getattr(self,mom+'_target').size
                                    )
                
                
                elif mom == 'SPFLOW' or mom == 'SPFLOWDOM':
                    if self.loss == 'log':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    *(1+distort_for_large_pflows_fac/np.abs(np.log(getattr(self,mom+'_target'))))
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    *(1+distort_for_large_pflows_fac/np.abs(np.log(getattr(self,mom+'_target'))))
                                    /getattr(self,mom+'_target')
                                    )
                    if self.loss == 'ratio':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))
                                    *(1+distort_for_large_pflows_fac/np.abs(np.log(getattr(self,mom+'_target'))))
                                    /getattr(self,mom+'_target')
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))
                                    *(1+distort_for_large_pflows_fac/np.abs(np.log(getattr(self,mom+'_target'))))
                                    /getattr(self,mom+'_target')
                                    /getattr(self,mom+'_target').size
                                    )
                            
                else:
                    mo = getattr(self,mom)
                    tar = getattr(self,mom+'_target')
                    setattr(self,
                            mom+'_deviation',
                            self.weights_dict[mom]*np.abs(mo-tar)/tar
                            /getattr(self,mom+'_target').size
                            )
                # mo = getattr(self,mom)
                # tar = getattr(self,mom+'_target')
                # setattr(self,
                #         mom+'_deviation',
                #         self.weights_dict[mom]*np.abs(mo-tar)/tar
                #         /np.log(getattr(self,mom+'_target')+1)
                #         )
                    # print(mo,tar,self.weights_dict[mom]*np.abs(mo-tar)/tar)
        
        # self.SPFLOW_deviation[0,0] = self.SPFLOW_deviation[0,0]*10
        
        if self.drop_CHN_IND_BRA_ROW_from_RD:
            if self.N == 7:
            # if self.N == 7 or self.N ==12:
                self.RD_deviation = self.RD_deviation[:3]
                try:
                    self.RD_RUS_deviation = self.RD_RUS_deviation[:3]   
                except:
                    pass
            if self.N == 12:
                    self.RD_deviation = np.array([self.RD_deviation[i] for i in [0,1,2,6,7]])
            if self.N == 11:
                    self.RD_deviation = np.array([self.RD_deviation[i] for i in [0,1,2,6,7]])
            if self.N == 13:
                    self.RD_deviation = np.array(
                        [self.RD_deviation[i] for i in [0,1,2,6,7,9]]
                                                 )
            #     # self.RD_deviation = np.array(self.RD_deviation[:3].tolist()+self.RD_deviation[7:-1].tolist())
            #     self.RD_deviation = np.concatenate([self.RD_deviation[:3],self.RD_deviation[6:-1]],axis=0)
                
            #     try:
            #         self.RD_RUS_deviation = np.array(self.RD_RUS_deviation[:3].tolist()+self.RD_RUS_deviation[6:-1].tolist())
            #     except:
            #         pass
                
            
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
    
    def objective_function(self):
        return (self.deviation_vector()**2).sum()/sum([self.weights_dict[mom] for mom in self.list_of_moments])

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
        ax.legend(loc='center left')
        ax2.legend(loc=(0.85,1.05))
        plt.show() 
    
    def save(self,p,path):
        try:
            os.mkdir(path)
        except:
            pass
        p.write_params(path+str(self.saves)+'/')
        self.saves += 1  
        
class history_nash:
    def __init__(self):
        self.count = 0
        self.make_a_pause = True
        self.delta = []
        self.welfare = []
        self.current_deltas = None
        self.current_welfare = None
        self.expected_welfare = np.full(7,np.nan)
        self.expected_deltas = np.full(7,np.nan)
    def update_current_deltas(self,new_deltas):
        self.current_deltas = new_deltas
    def update_current_welfare(self,new_welfare):
        self.current_welfare = new_welfare

