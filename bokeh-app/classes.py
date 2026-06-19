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
                        'power_fdi':co,
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
                         'd':co,
                         'd_frac':co}
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
                        'power_fdi':cou,
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
                         'd':10,
                         'd_frac':1-co}
        
        self.calib_parameters = None
        self.guess = None
        self.dyn_guess = None
        
        self.correct_eur_patent_cost = True
        self.fix_fe_across_sectors = False
        self.fix_delta_across_sectors = False

        self.g_0 = 0.01
        self.kappa = 0.5
        self.gamma = 0.5 
        self.power_fdi = 1.0
        self.k = np.array([1.3,1.3])
        # FDI setup cost magnitude.  Two modes:
        #   - SCALAR (default): self.a is a scalar; broadcasts to (N,N,S) in
        #     compute_entry_costs via   var.a[n,i,s] = p.a * supply_potential.
        #     Single free parameter for the entire FDI cost matrix.
        #   - BILATERAL: self.a is a (N,N,S) array.  Each off-diagonal sector-1
        #     entry is an independent calibration parameter, allowing the FDI
        #     cost to vary by origin-destination pair.  Sector 0 and diagonal
        #     entries are held at zero.  Activated by parameters.enable_bilateral_a().
        # Both modes use the same multiplicative form in compute_entry_costs;
        # numpy broadcasting handles scalar transparently.  See compute_entry_costs.
        self.a = np.float64(0.0)
        self.bilateral_a = False  # default: scalar a
        self.rho = 0.02
        self.d = np.float64(1.0)
        # d_frac: reparameterization for FDI-model calibration enforcing k > d + 1.
        # When 'd_frac' is in calib_parameters, sync rule is:
        #     self.d = self.d_frac * (self.k[1] - 1)   (k[1] is the FDI sector's
        #                                                Pareto tail index)
        # so 0 < d_frac < 1 implies 0 < d < k[1] - 1, automatically respecting
        # the constraint that scoping the gamma and Lambda^F denominators
        # k - d - 1 stay positive. See update_parameters / make_p_vector for
        # the bidirectional sync.
        # var_with_entry_costs (a different model) reads p.d directly and is
        # not affected unless 'd_frac' is the calibrated parameter.
        self.d_frac = np.float64(0.5)
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

        # FDI affiliate sales (model X^{M,F,data}_{nis}, sector 1 only):
        #   - CSV uses 1-based integer country codes positionally matching
        #     self.countries (code k → self.countries[k-1]).
        #   - Diagonal entries (e.g. US affiliates in US) zeroed — the model
        #     X^{M,F} is foreign affiliate sales only (n != j).
        # Result stored at p.X_F_data with shape (N, N, S), [dest, origin, s],
        # in the same units as p.trade_flows.
        self.X_F_data = np.zeros((N, N, S))
        try:
            _fdi = pd.read_csv(dir_path+'data/fdi_longformat_2015_AAMNE.csv')
            _piv = (_fdi.pivot_table(index='Rep_ccode', columns='File_ccode',
                                     values='FileToRep_Flow', aggfunc='sum')
                       .reindex(index=range(1, N+1),
                                columns=range(1, N+1))
                       .fillna(0.0))
            _fdi_mat = _piv.values / self.unit
            np.fill_diagonal(_fdi_mat, 0.0)
            if S >= 2:
                self.X_F_data[..., 1] = _fdi_mat
        except FileNotFoundError:
            pass  # leave self.X_F_data as zeros

        try:
            self.tariff = pd.read_csv(data_path+'tariff.csv',index_col=[1,0,2]).sort_index().values.squeeze().reshape((N,N,S))
        except:
            self.tariff = np.zeros_like(self.trade_flows)
        self.beta = np.einsum('nis->s',self.trade_shares)
        self.deficit_raw = self.data.deficit.values.copy()
        self.deficit_raw[0] = self.deficit_raw[0]-self.deficit_raw.sum()
        self.deficit_share_world_output = self.deficit_raw/self.data.output.sum()
        self.unit_labor = 1e9
        self.unit = 1e6
        self.khi = 0.16
        self.labor_raw = self.data.labor.values
        self.labor = self.labor_raw/self.unit_labor
        self.r_hjort = ((self.data.gdp.iloc[0]*np.array(self.data.labor)*self.data.price_level
                        /(self.data.labor.iloc[0]*self.data.price_level.iloc[0]*np.array(self.data.gdp))
                        )**(1-self.khi)).values.copy()
        
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
                    'd_frac':pd.Index(['scalar']),
                    'khi':pd.Index(['scalar']),
                    # 'k':pd.Index(['scalar']),
                    'k':pd.Index(self.sectors, name='sector'),
                    'a':pd.Index(['scalar']),
                    'power_fdi':pd.Index(['scalar']),
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
                    'sigma':[np.s_[0],np.s_[1]],
                    # 'sigma':[np.s_[0]],
                    'theta':[np.s_[0]],
                    'rho':None,
                    'gamma':None,
                    'zeta':[np.s_[0]],
                    'nu':[np.s_[0]],
                    'nu_tilde':[np.s_[0]],
                    'kappa':None,
                    'k':None,
                    'a':None,
                    'power_fdi':None,
                    'fe':[np.s_[0]],
                    'fo':[np.s_[0]],
                    'delta':[np.s_[::S]],
                    # 'delta':[np.s_[::S],np.s_[2::S]],
                    'delta_dom':[np.s_[::S]],#,np.s_[S-1]],
                    # 'delta_dom':[np.s_[np.r_[0:7, 8:N*S]]],#,np.s_[S-1]],
                    'delta_int':[np.s_[::S]],#,np.s_[S-1]],
                    # 'delta_int':[np.s_[np.r_[0:7, 8:N*S]]],#,np.s_[S-1]],
                    'g_0':None,
                    'd':None,
                    'd_frac':None,
                    'khi':None,
                    'alpha':None,
                    'beta':None,
                     'T':None,
                     'r_hjort':None,
                     'eta':[np.s_[::S]]
                     }
        
        # if nbr_sectors == 4:
        #     sl_non_calib['delta'] = [np.s_[::S],np.s_[2::S],np.s_[3::S]]
        
        if self.fix_fe_across_sectors:
            sl_non_calib['fe'] = [np.s_[0],np.s_[2:]]

        if self.fix_delta_across_sectors:
            # calibrate delta only in the first patenting sector (sector 1);
            # sectors >=2 are tied to it (see calibration_func)
            sl_non_calib['delta'] = [np.s_[::S], np.s_[2::S]]

        self.mask = {}
        
        for par_name in ['eta','k','rho','alpha','fe','T','fo','sigma','theta','beta','zeta',
                         'g_0','kappa','gamma','delta','delta_dom','delta_int','nu','nu_tilde','d','d_frac','khi',
                         'r_hjort','a','power_fdi']:
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
        _d_frac_loaded = False  # track for backward-compat sync at end
        for pa_name in list_of_params:
            # if pa_name == 'k':
            #     df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
            #     print(pa_name,df.values.squeeze())
            #     print(np.array(getattr(self,pa_name)).shape)
            try:
                df = pd.read_csv(path+pa_name+'.csv',header=None,index_col=0)
                if pa_name == 'a':
                    # Backwards-compat: detect whether this checkpoint stored
                    # scalar 'a' or bilateral (N, N, S) 'a'.  If bilateral and
                    # current self.a is scalar, auto-promote before reshaping.
                    n_vals = df.values.size
                    if n_vals == self.N * self.N * self.S and not self.bilateral_a:
                        self.enable_bilateral_a(init_value=0.0)  # placeholder; values come from disk
                    if self.bilateral_a:
                        setattr(self, 'a',
                                df.values.squeeze().reshape((self.N, self.N, self.S)))
                    else:
                        # scalar mode — extract first value
                        setattr(self, 'a', np.float64(df.values.squeeze()))
                elif pa_name != 'k':
                    setattr(self,pa_name,df.values.squeeze().reshape(np.array(getattr(self,pa_name)).shape))
                else:
                    setattr(self,pa_name,df.values.squeeze())
                if pa_name == 'd_frac':
                    _d_frac_loaded = True
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
            
        if self.k.shape[0] != self.mask['k'].shape[0]:
            self.mask['k'] = np.array([True]*self.S)
        
        self.update_delta_eff()
        # Backward compatibility: if the checkpoint pre-dates d_frac (no
        # d_frac.csv on disk), derive d_frac from the loaded d and current
        # k[1] so the FDI calibration can still warm-start cleanly. If the
        # checkpoint did save d_frac, trust the saved value as authoritative.
        if not _d_frac_loaded:
            self._sync_dfrac_from_d()            
        
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
         'kappa','gamma','delta','delta_dom','delta_int','nu','nu_tilde','d','d_frac','khi','r_hjort',
         'tau','tariff','a','power_fdi']
            
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
                ,axis=0)
        return vec
    
    # ──────────────────────────────────────────────────────────────────────
    # d <-> d_frac sync helpers
    # ──────────────────────────────────────────────────────────────────────
    # When the FDI model is calibrated, 'd_frac' is the free parameter and
    #   d = d_frac * (k[1] - 1 - D_MARGIN)
    # so that k - d - 1 >= D_MARGIN always. The margin must be large enough
    # to keep the gamma and Lambda^F denominators 1/(k-d-1) well-bounded;
    # with margin = 0.05, those terms are at most 20, which is fine. The
    # previous tiny margin (1e-6) let the optimizer drift to k-d-1 ~ 0.07
    # where the model became numerically fragile and the solver started
    # failing systematically. See _D_MARGIN below.
    _D_MARGIN = 0.05
    def _d_safe_range(self):
        """Returns max admissible d so that k - d - 1 >= _D_MARGIN."""
        return max(float(self.k[1]) - 1.0 - self._D_MARGIN, 1e-12)

    def _sync_d_from_dfrac(self):
        """Set self.d from self.d_frac and current k[1]."""
        self.d = np.float64(float(self.d_frac) * self._d_safe_range())

    def _sync_dfrac_from_d(self):
        """Set self.d_frac from self.d and current k[1] (inverse map, clamped)."""
        safe = self._d_safe_range()
        if safe > 0:
            frac = float(self.d) / safe
            self.d_frac = np.float64(min(max(frac, 1e-6), 1.0 - 1e-6))

    def make_p_vector(self):
        # NOTE: when 'd_frac' is in calib_parameters, the *caller* is expected
        # to set self.d_frac to a sensible value (e.g. p.d_frac = 0.6) before
        # the first least_squares iteration. We do NOT auto-sync from self.d
        # here, because that would silently overwrite the user's d_frac seed
        # with a value derived from p.d (which may be stale or zero in a
        # freshly-loaded checkpoint).
        # If you only have self.d and want d_frac to track it, call
        # self._sync_dfrac_from_d() explicitly before make_p_vector.
        vec = np.concatenate([np.array(getattr(self,p))[self.mask[p]].ravel() for p in self.calib_parameters])
        return vec

    def update_parameters(self,vec):
        idx_from = 0
        # Track whether we need a final d<->d_frac sync.
        # If both 'd' and 'd_frac' are in calib_parameters, 'd_frac' wins
        # (i.e. the FDI calibration is taking precedence).
        sync_d_from_dfrac = False
        for par in self.calib_parameters:
            param = np.array(getattr(self,par))
            size = param[self.mask[par]].size
            param[self.mask[par]] = vec[idx_from:idx_from+size]
            setattr(self,par,param)
            idx_from += size
            if par == 'd_frac':
                sync_d_from_dfrac = True
        # If 'd_frac' is being calibrated (or 'k' moved while 'd_frac' is the
        # underlying free var), recompute d from d_frac so the model sees the
        # constraint-respecting d. Skipped when only 'd' is calibrated, to
        # preserve backward compatibility with var_with_entry_costs runs.
        if sync_d_from_dfrac or ('d_frac' in self.calib_parameters and 'k' in self.calib_parameters):
            self._sync_d_from_dfrac()
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
                        )**(1-self.khi)).values.copy()

    def enable_bilateral_a(self, init_value=None):
        """
        Promote self.a from scalar to a bilateral (N, N, S) array, with
        per-pair calibration of the FDI setup cost.

        Each off-diagonal sector-1 entry becomes an independent free parameter:
            self.a[n, i, 1]  for all n != i,   n, i = 0..N-1
        Diagonal entries (n == i) and sector-0 entries are held at 0
        (they have no effect since FDI applies only to sector 1 and only
        across distinct origin/destination pairs).

        After this call:
          - self.a has shape (N, N, S).
          - self.mask['a'] selects only off-diagonal sector-1 entries.
          - self.shapes['a'] becomes a (dest, origin, sector) MultiIndex.
          - self.bilateral_a = True.
          - The corresponding moment FDI_FLOW (bilateral) should be added
            to moments.list_of_moments instead of (or in addition to)
            FDI_FLOW_N (destination-aggregated).

        Parameters
        ----------
        init_value : float or None
            Value to broadcast into all off-diagonal sector-1 entries.
            If None, uses the current scalar self.a (or 0.1 if a is zero/missing).
            Typical: 0.1 (the same starting value used in scalar-a runs).

        Notes
        -----
        Idempotent: calling twice is harmless (preserves current values).
        Backwards-compatible: existing code paths that read self.a continue
        to work via numpy broadcasting (scalar -> broadcasts; array ->
        element-wise).  compute_entry_costs needs no change.
        """
        N, S = self.N, self.S
        # If already bilateral with the correct shape, preserve loaded values
        # (truly idempotent). Just refresh the mask and idx in case those
        # weren't set yet (e.g., after a bare load_run that promoted a but
        # didn't rebuild the mask). Reshape-safe: a (1,)->shape() arrays
        # are treated as scalar.
        cur = np.asarray(self.a)
        if self.bilateral_a and cur.shape == (N, N, S):
            # Already bilateral: keep values. Just (re)build mask + idx.
            pass
        else:
            # Promote scalar (or wrong-shape) self.a to bilateral.
            if init_value is None:
                scalar = float(cur.reshape(-1)[0]) if cur.size > 0 else 0.0
                init_value = scalar if scalar > 0 else 0.1
            a_arr = np.zeros((N, N, S))
            if S >= 2:
                a_arr[..., 1] = init_value
                np.fill_diagonal(a_arr[..., 1], 0.0)
            self.a = a_arr
            self.bilateral_a = True
        # Update idx (used for serialization) for bilateral shape
        self.idx['a'] = pd.MultiIndex.from_product(
            [self.countries, self.countries, self.sectors],
            names=['destination', 'origin', 'sector'])
        # Build calib mask: True only on off-diagonal sector-1 entries
        mask = np.zeros((N, N, S), bool)
        if S >= 2:
            mask[..., 1] = True
            np.fill_diagonal(mask[..., 1], False)
        self.mask['a'] = mask

    def mask_a_from_fdi_flow_mask(self, m):
        """
        Synchronize self.mask['a'] (bilateral FDI cost calibration mask)
        with moments.FDI_FLOW_mask: drop a^s_{ni} entries whose FDI_FLOW
        cell is excluded from the residual.

        Rationale: if we don't target FDI_FLOW_{ni} (because the data is at
        the noise floor), the bilateral cost a^s_{ni} for that pair is
        unidentified and should be held fixed rather than calibrated as a
        free direction the optimizer is free to wander in.

        After this call:
          - self.mask['a'][n, i, 1] = True   iff   moments.FDI_FLOW_mask[n, i]
            (and the pair is off-diagonal).
          - sector 0 and diagonal stay False as before.

        The held-fixed a values keep whatever value they had when the call
        was made (typically the warm-start values from the previous run).
        If you want them at a neutral starting value, set p.a[mask_False]
        before calling.

        Parameters
        ----------
        m : moments
            Instance with FDI_FLOW_mask attribute populated (typically by
            m.set_fdi_flow_zero_threshold).
        """
        if not self.bilateral_a:
            raise RuntimeError("mask_a_from_fdi_flow_mask requires "
                               "bilateral_a; call enable_bilateral_a first.")
        if not hasattr(m, 'FDI_FLOW_mask') or m.FDI_FLOW_mask is None:
            raise RuntimeError("moments.FDI_FLOW_mask is not set; call "
                               "m.set_fdi_flow_zero_threshold(...) first.")
        N, S = self.N, self.S
        if m.FDI_FLOW_mask.shape != (N, N):
            raise ValueError(f"FDI_FLOW_mask shape {m.FDI_FLOW_mask.shape} "
                             f"!= expected ({N}, {N})")
        mask = np.zeros((N, N, S), bool)
        if S >= 2:
            mask[..., 1] = m.FDI_FLOW_mask
            np.fill_diagonal(mask[..., 1], False)  # belt and suspenders
        self.mask['a'] = mask
        n_free = int(mask.sum())
        n_max = N * (N - 1)
        print(f"[a] mask synced with FDI_FLOW_mask: {n_free} of {n_max} "
              f"off-diagonal entries free ({n_max - n_free} held fixed)")
            
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
            # print(par)
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
        # ── integral and shared denominators (unchanged) ─────────────────────
# integral_k_d = self.hypergeometric_integral(
#     lb=self.psi_m_star_O[...,1], ub=self.psi_MP_star[...,1],
#     alpha=self.V_P[...,1], beta=self.w*p.fe[1]*p.r_hjort, y=k, z=d)

# def _hypergeometric_integral(lb, ub, alpha, beta, y, z):
#     """
#     Compute  ∫_{lb}^{ub}  x^{-y} (alpha*x - beta)^z  dx
#     using the incomplete Beta function identity.
 
#     Parameters
#     ----------
#     lb, ub : array_like   — lower / upper integration limits (shapes broadcastable)
#     alpha  : array_like   — coefficient of x inside the bracket  (shape n,n or n,n,s)
#     beta   : array_like   — offset (shape n or n,s depending on context)
#     y      : float        — power of x
#     z      : float        — power of the bracket
#     """
#     t_ub = 1 - beta[:, None] / ub / alpha
#     t_lb = 1 - beta[:, None] / lb / alpha
#     integral = (beta[:, None] ** (1 - y + z) / alpha ** (1 - y)
#                 ) * np.vectorize(
#                     lambda a, b, x1, x2: float(betainc(a, b, x1, x2, regularized=False))
#                 )(z + 1, y - z - 1, t_lb, t_ub)
#     return integral
 
 
# # def _betainc_vec(a, b, t1, t2):
# #     """Vectorised non-regularised incomplete Beta function."""
# #     return np.vectorize(
# #         lambda aa, bb, x1, x2: float(betainc(aa, bb, max(0.0,min(1.0,float(x1))), max(0.0,min(1.0,float(x2))), regularized=False).real)
# #     )(a, b, t1, t2)

# def _betainc_vec(a, b, t1, t2):
#     """Vectorised non-regularised incomplete Beta function."""
#     return np.vectorize(
#         lambda aa, bb, x1, x2: float(betainc(aa, bb, 
#                                               max(0.0, min(1.0, float(x1))), 
#                                               max(0.0, min(1.0, float(x2))), 
#                                               regularized=False).real)
#     )(a, b, t1, t2)

# # ─────────────────────────────────────────────────────────────────────────────
# # var_with_fdi
# # ─────────────────────────────────────────────────────────────────────────────

# class var_with_fdi:
#     """
#     Variable container for the steady-state solver with entry costs AND FDI.
 
#     Methods identical to var_with_entry_costs are reproduced verbatim.
#     Methods that differ are annotated with  # [FDI].
#     """
 
#     # ── construction ────────────────────────────────────────────────────────
 
#     def __init__(self, context, N=7, S=2):
#         m = np.ones((N, N, S), bool).ravel()
#         m[np.s_[::(N + 1) * S]] = False
#         m[np.s_[1::(N + 1) * S]] = False
#         self.off_diag_mask = m.reshape((N, N, S))
#         self.diag_mask = ~self.off_diag_mask
#         self.context = context
 
#     # ── guess setters ────────────────────────────────────────────────────────
 
#     def guess_profit(self, v):        self.profit = v
#     def guess_wage(self, v):          self.w = v
#     def guess_Z(self, v):             self.Z = v
#     def guess_labor_research(self, v):self.l_R = v
#     def guess_phi(self, v):           self.phi = v
#     def guess_price_indices(self, v): self.price_indices = v
#     def guess_pi_F(self, v):          self.pi_F = v   # [FDI]
 
#     def elements(self):
#         for key, item in sorted(self.__dict__.items()):
#             print(key, ',', str(type(item))[8:-2])
 
#     def copy(self):
#         return deepcopy(self)
 
#     # ── vector ↔ var  [FDI: adds pi_F block] ────────────────────────────────
 
#     @staticmethod
#     def var_from_vector(vec, p, context, compute=True):
#         """
#         Vector layout (lengths):
#             w             N
#             Z             N
#             l_R           N*(S-1)          sector 0 excluded
#             profit        N*N*(S-1)        sector 0 excluded
#             phi           N*N*S
#             price_indices N
#             pi_F          N*N*(S-1)        sector 0 excluded   [FDI]
#         """
#         N, S = p.N, p.S
#         v = var_with_fdi(context=context, N=N, S=S)
#         i0 = 0
 
#         v.guess_wage(vec[i0:i0+N]);                                        i0 += N
#         v.guess_Z(vec[i0:i0+N]);                                           i0 += N
#         v.guess_labor_research(
#             np.insert(vec[i0:i0+N*(S-1)].reshape((N, S-1)),
#                       0, np.zeros(N), axis=1));                            i0 += N*(S-1)
#         v.guess_profit(
#             np.insert(vec[i0:i0+N*N*(S-1)].reshape((N, N, S-1)),
#                       0, np.zeros(N), axis=2));                            i0 += N*N*(S-1)
#         v.guess_phi(vec[i0:i0+N*N*S].reshape((N, N, S)));                 i0 += N*N*S
#         v.guess_price_indices(vec[i0:i0+N]);                               i0 += N
 
#         # [FDI] pi_F block — may be absent if vec comes from a legacy
#         # var_with_entry_costs guess (e.g. loaded via p.guess).
#         # In that case default to zeros (Case 1 starting point).
#         remaining = vec[i0:]
#         pi_F_size = N * N * (S - 1)
#         if len(remaining) >= pi_F_size:
#             pi_F_flat = remaining[:pi_F_size]
#         else:
#             pi_F_flat = np.zeros(pi_F_size)
#         v.guess_pi_F(
#             np.insert(pi_F_flat.reshape((N, N, S-1)),
#                       0, np.zeros(N), axis=2))
 
#         if compute:
#             v.compute_solver_quantities(p)
#         return v
 
#     def vector_from_var(self):
#         return np.concatenate([
#             self.w,
#             self.Z,
#             self.l_R[..., 1:].ravel(),
#             self.profit[..., 1:].ravel(),
#             self.phi.ravel(),
#             self.price_indices,
#             self.pi_F[..., 1:].ravel(),    # [FDI]
#         ])
 
#     # ── hypergeometric integral (unchanged) ─────────────────────────────────
 
#     @staticmethod
#     def hypergeometric_integral(lb, ub, alpha, beta, y, z):
#         t_ub = 1 - beta[:, None] / ub / alpha
#         t_lb = 1 - beta[:, None] / lb / alpha
#         # print(type(t_lb), type(t_ub))
#         return (beta[:, None]**(1-y+z) / alpha**(1-y)) * _betainc_vec(
#             z+1, y-z-1, t_lb, t_ub)
 
#     # ── compute_growth (unchanged) ───────────────────────────────────────────
 
#     def compute_growth(self, p):
#         self.g_s = (p.k * np.einsum('is,is->s', p.eta, self.l_R**(1-p.kappa))
#                     / (p.k-1) - p.zeta)
#         self.g_s[0] = p.g_0
#         self.g = (p.beta * self.g_s / (p.sigma-1)).sum() / (p.beta * p.alpha).sum()
#         self.r = p.rho + self.g / p.gamma
#         self.G = self.r + p.zeta - self.g + self.g_s + p.nu
 
#     # ── compute_entry_costs (unchanged) ─────────────────────────────────────
 
#     def compute_entry_costs(self, p):
#         if self.context == 'calibration':
#             self.a = p.a *np.maximum( 
#                 np.einsum(
#                 'is,nis,nis,is,is->nis',
#                 p.T**(1/p.theta[None,:]), 1/self.phi, 1/(1+p.tariff),
#                 self.w[:,None]**-p.alpha[None,:],
#                 self.price_indices[:,None]**(p.alpha[None,:]-1))
#                 ,1)**p.power_fdi
#             np.einsum('nns->ns', self.a)[:] = 0
#         elif self.context == 'counterfactual':
#             self.a = p.a * np.maximum(p.tau,1)**p.power_fdi
#             np.einsum('nns->ns', self.a)[:] = 0
            
#     # def compute_entry_costs(self, p):
#     #     if self.context == 'calibration':
#     #         # Export entry costs (unchanged)
#     #         self.a = p.a * np.einsum(
#     #             'is,nis,nis,is,is->nis',
#     #             p.T**(1/p.theta[None,:]), 1/self.phi, 1/(1+p.tariff),
#     #             self.w[:,None]**-p.alpha[None,:],
#     #             self.price_indices[:,None]**(p.alpha[None,:]-1))
#     #         np.einsum('nns->ns', self.a)[:] = 0
    
#     #         # Point 15: solve for a_{nis} to match FDI share data
#     #         if hasattr(p, 'fdi_flow') and p.fdi_flow is not None:
#     #             for s in range(1, p.S):
#     #                 # Data: within-destination FDI shares by origin
#     #                 fdi_sum_n = p.fdi_flow.sum(axis=1)             # (N,)
#     #                 safe_sum  = np.where(fdi_sum_n > 0, fdi_sum_n, 1.0)
#     #                 fdi_share_data = p.fdi_flow / safe_sum[:, None] # (N,N)
    
#     #                 def model_fdi_shares(log_a_flat):
#     #                     # Set a, recompute the FDI quality stocks, return shares
#     #                     a_trial = self.a[:, :, s].copy()
#     #                     a_trial[self.off_diag_mask[:, :, s]] = np.exp(log_a_flat)
#     #                     self.a[:, :, s] = a_trial
    
#     #                     # Recompute the FDI-relevant quantities for sector s only
#     #                     self.compute_V(p)
#     #                     self.compute_case_indicator(p)
#     #                     self.compute_auxiliary_thresholds(p)
#     #                     self.compute_patenting_thresholds(p)
#     #                     self.compute_mass_innovations(p)
#     #                     self.compute_aggregate_qualities(p)
    
#     #                     PSI_F = self.PSI_M_F[:, :, s]              # (N,N)
#     #                     PSI_F_sum = PSI_F.sum(axis=1)               # (N,)
#     #                     safe = np.where(PSI_F_sum > 0, PSI_F_sum, 1.0)
#     #                     share_model = PSI_F / safe[:, None]          # (N,N)
    
#     #                     # Residual on off-diagonal pairs only
#     #                     residual = (share_model - fdi_share_data)[self.off_diag_mask[:, :, s]]
#     #                     return residual
    
#     #                 # Initial guess: log of current a, off-diagonal only
#     #                 a0 = self.a[:, :, s][self.off_diag_mask[:, :, s]]
#     #                 a0 = np.where(a0 > 0, a0, 1e-4)
#     #                 log_a0 = np.log(a0)
    
#     #                 sol = root(model_fdi_shares, x0=log_a0, tol=1e-10)
#     #                 self.a[:, :, s][self.off_diag_mask[:, :, s]] = np.exp(sol.x)
#     #                 np.einsum('nns->ns', self.a[:, :, s:s+1])[:] = 0
    
#     #     elif self.context == 'counterfactual':
#     #         self.a = p.a * p.tau
#     #         np.einsum('nns->ns', self.a)[:] = 0
 
#     # ── compute_V  [FDI: adds V_NP_F, V_P_F] ───────────────────────────────
 
#     def compute_V(self, p):
#         # Export value functions (unchanged)
#         self.V_NP = np.einsum('nis,i,s->nis', self.profit, self.w, 1/self.G)
#         self.V_P  = np.einsum(
#             'nis,i,ns->nis', self.profit, self.w,
#             1/(self.G[None,:]-p.nu[None,:]+p.delta)
#             - 1/(self.G[None,:]+p.delta) + 1/self.G[None,:])
 
#         # FDI value functions: driven by pi_F, wage is w_n  [FDI]
#         self.V_NP_F = np.einsum('nis,n,s->nis', self.pi_F, self.w, 1/self.G)
#         self.V_P_F  = np.einsum(
#             'nis,n,ns->nis', self.pi_F, self.w,
#             1/(self.G[None,:]-p.nu[None,:]+p.delta)
#             - 1/(self.G[None,:]+p.delta) + 1/self.G[None,:])
 
#     # ── compute_case_indicator  [FDI] ───────────────────────────────────────
 
#     # def compute_case_indicator(self, p):
#     #     """Case 2: w_i*pi^w < w_n*Pi^{w,F}, off-diagonal, patenting sectors."""
#     #     self.case2 = np.zeros((p.N, p.N, p.S), bool)
#     #     self.case2[..., 1:] = (
#     #         (self.w[None,:,None] * self.profit[...,1:]
#     #          < self.w[:,None,None] * self.pi_F[...,1:])
#     #         & self.off_diag_mask[...,1:])
        
#     def compute_case_indicator(self, p):
#         self.case2 = np.zeros((p.N, p.N, p.S), bool)
#         self.case2[..., 1:] = (
#             (self.w[None,:,None] * self.profit[...,1:]
#              < self.w[:,None,None] * self.pi_F[...,1:])
#             & (self.pi_F[...,1:] > 0)
#             & self.off_diag_mask[...,1:]
#         )
        
#     # def compute_case_indicator(self, p):
#     #     """
#     #     In the solver we always treat off-diagonal patenting sector pairs as
#     #     Case 2 (potential FDI). The gamma_PF / gamma_NPF fractions naturally
#     #     go to zero when FDI is dominated by exports (large a_{nis} or small
#     #     pi_F), so there is no need to hard-gate on the profit comparison.
#     #     The profit comparison case2: w_i*pi^w < w_n*Pi^{w,F} is only
#     #     meaningful once the fixed point is solved; using it as a gate during
#     #     iteration prevents Case 2 from ever being triggered from a cold start.
#     #     """
#     #     self.case2 = np.zeros((p.N, p.N, p.S), bool)
#     #     self.case2[..., 1:] = self.off_diag_mask[..., 1:]
 
#     # # ── compute_auxiliary_thresholds  [FDI] ─────────────────────────────────
 
#     def compute_auxiliary_thresholds(self, p):
#         """Eqs (6)-(7) of main_3_.tex.  All np.inf in Case 1."""
#         # w_n * h_n * fe_s  shape (N, S-1)
#         w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])
#         eps = 1e-30
 
#         dNP     = self.V_NP_F[...,1:] - self.V_NP[...,1:]
#         dP      = self.V_P_F[...,1:]  - self.V_P[...,1:]
#         dPF_NPO = self.V_P_F[...,1:]  - self.V_NP[...,1:]
#         mask    = self.case2[...,1:]
#         w_a     = self.w[:,None,None] * self.a[...,1:]
 
#         def _thresh(num, den):
#             out = np.full(num.shape, np.inf)
#             out[mask] = num[mask] / (den[mask] + eps)
#             return out
 
#         self.a_NPF_NPO = np.full((p.N,p.N,p.S), np.inf)
#         self.a_NPF_NPO[...,1:] = _thresh(w_a, dNP)
 
#         self.a_PF_PO = np.full((p.N,p.N,p.S), np.inf)
#         self.a_PF_PO[...,1:] = _thresh(w_a, dP)
 
#         self.a_PF_NPO = np.full((p.N,p.N,p.S), np.inf)
#         self.a_PF_NPO[...,1:] = _thresh(w_a + w_fe_h[:,None,:], dPF_NPO)
 
#         self.psi_bar_NPO_PF = np.full((p.N,p.N,p.S), np.inf)
#         self.psi_bar_NPO_PF[...,1:] = _thresh(
#             np.broadcast_to(w_fe_h[:,None,:], w_a.shape).copy(), dPF_NPO)
 
#     # ── compute_patenting_thresholds  [FDI: extended] ───────────────────────
 
#     def compute_patenting_thresholds(self, p):
#         """
#         Adds psi^{*,F} (FDI threshold, Case 2) and computes separate
#         effective thresholds psi^{m*,O} and psi^{m*,F}.
#         psi_star and psi_m_star are kept as aliases for backward compatibility.
#         """
#         w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])  # (N,S-1)
 
#         # ── a_NP_star, a_P_star (unchanged) ─────────────────────────────────
#         self.a_NP_star = np.ones((p.N,p.N,p.S))
#         self.a_NP_star[...,1:] = np.maximum(
#             np.einsum('i,nis,nis->nis', self.w, self.a[...,1:], 1/self.V_NP[...,1:]), 1)
 
#         self.a_P_star = np.ones((p.N,p.N,p.S))
#         self.a_P_star[...,1:] = np.maximum(
#             np.einsum('i,nis,nis->nis',
#                       self.w, self.a[...,1:],
#                       1/(self.V_P[...,1:] - w_fe_h[:,None,:])), 1)
 
#         # ── psi^{*,O} ────────────────────────────────────────────────────────
#         # Case 1: same formula as var_with_entry_costs
#         psi_C1 = np.full((p.N,p.N,p.S), np.inf)
#         psi_C1[...,1:] = (
#             w_fe_h[:,None,:] /
#             (self.profit[...,1:] * self.w[None,:,None]
#              * (1/(self.G[None,:]+p.delta-p.nu[None,:])
#                 - 1/(self.G[None,:]+p.delta))[:,None,1:]))
#         # Case 2: psi^{C,O} = w_n h_n fe_s / (V_P_O - V_NP_O)
#         psi_C2_O = np.full((p.N,p.N,p.S), np.inf)
#         psi_C2_O[...,1:] = np.where(
#             self.case2[...,1:],
#             w_fe_h[:,None,:] / (self.V_P[...,1:] - self.V_NP[...,1:] + 1e-30),
#             np.inf)
#         self.psi_star_O = np.maximum(np.where(self.case2, psi_C2_O, psi_C1), 1)
#         self.psi_star   = self.psi_star_O   # backward-compat alias
 
#         # ── psi^{*,F}  [FDI] ────────────────────────────────────────────────
#         self.psi_star_F = np.full((p.N,p.N,p.S), np.inf)
#         self.psi_star_F[...,1:] = np.where(
#             self.case2[...,1:],
#             w_fe_h[:,None,:] / (self.V_P_F[...,1:] - self.V_NP_F[...,1:] + 1e-30),
#             np.inf)
#         self.psi_star_F = np.maximum(self.psi_star_F, 1)
 
#         # ── psi^{o*} (unchanged logic, uses psi_star_O) ─────────────────────
#         self.psi_o_star = np.full((p.N,p.S), np.inf)
 
#         def aleph_P_star(pso):
#             return np.maximum(
#                 np.einsum('i,nis,nis->nis', self.w, self.a[...,1:],
#                           1/(pso[None,...]*self.V_P[...,1:] - w_fe_h[:,None,:])), 1)
 
#         def aleph_NP_star(pso):
#             return np.maximum(
#                 np.einsum('i,nis,is,nis->nis',
#                           self.w, self.a[...,1:], 1/pso, 1/self.V_NP[...,1:]), 1)
 
#         def func_to_solve(pso):
#             pso = pso[:,None]
#             sig = pso[None,...] >= self.psi_star_O[...,1:]
#             A = (np.einsum('is,nis->nis', pso, self.V_P[...,1:]) - w_fe_h[:,None,:]
#                  ) * aleph_P_star(pso)**(-p.d)
#             B = np.einsum('i,nis,,nis->nis',
#                           self.w, self.a[...,1:], p.d/(p.d+1),
#                           aleph_P_star(pso)**(-p.d-1))
#             C = np.einsum('is,nis->nis', pso, self.V_NP[...,1:]
#                           ) * aleph_NP_star(pso)**(-p.d)
#             D = np.einsum('i,nis,,nis->nis',
#                           self.w, self.a[...,1:], p.d/(p.d+1),
#                           aleph_NP_star(pso)**(-p.d-1))
#             res = ((sig*(A-B-(C-D))).sum(axis=0)
#                    - self.w[:,None]*p.fo[None,1:]*p.r_hjort[:,None])
#             return res.ravel() / pso.ravel()
 
#         x0 = np.min(self.psi_star_O[...,1], axis=0)
#         roots = root(func_to_solve, x0=x0, tol=1e-15)
#         self.psi_o_star[:,1] = roots.x
 
#         # equality condition
#         sig = np.isclose(self.psi_star_O[...,1:], 1)
#         A = (self.V_P[...,1:] - w_fe_h[:,None,:]) * self.a_P_star[...,1:]**(-p.d)
#         B = np.einsum('i,nis,,nis->nis', self.w, self.a[...,1:],
#                       p.d/(p.d+1), self.a_P_star[...,1:]**(-p.d-1))
#         C = self.V_NP[...,1:] * self.a_NP_star[...,1:]**(-p.d)
#         D = np.einsum('i,nis,,nis->nis', self.w, self.a[...,1:],
#                       p.d/(p.d+1), self.a_NP_star[...,1:]**(-p.d-1))
#         res = ((sig*(A-B-(C-D))).sum(axis=0)
#                - self.w[:,None]*p.fo[None,1:]*p.r_hjort[:,None])
#         self.psi_o_star[...,1:][res > 0] = 1
 
#         # ── effective thresholds  [FDI] ──────────────────────────────────────
#         self.psi_m_star_O = np.maximum(self.psi_star_O, self.psi_o_star[None,:,:])
#         self.psi_m_star_F = np.maximum(self.psi_star_F, self.psi_o_star[None,:,:])
#         self.psi_m_star   = self.psi_m_star_O   # backward-compat alias
 
#         self.psi_MP_star = np.full((p.N,p.N,p.S), np.inf)
#         self.psi_MP_star[...,1:] = np.maximum(
#             self.psi_m_star_O[...,1:],
#             (self.w[None,:,None]*self.a[...,1:]
#              + self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None]
#              ) / self.V_P[...,1:])
 
#         self.psi_MNP_star = np.full((p.N,p.N,p.S), np.inf)
#         self.psi_MNP_star[...,1:] = np.maximum(
#             self.psi_m_star_O[...,1:],
#             self.w[None,:,None]*self.a[...,1:] / self.V_NP[...,1:])
 
#     # ── compute_mass_innovations  [FDI: adds gamma fractions] ───────────────
 
#     def compute_mass_innovations(self, p):
#         """
#         mu_MNE/MPND/MNP: unchanged from var_with_entry_costs.
#         gamma_PO/PF/NPO/NPF: quality fractions from eqs (21)-(26) of main_3_.tex.
#         In Case 1 all gamma_?_F = 0 and the _O fractions reduce to the
#         no-FDI k/(k-1) formulas.
#         """
#         k = p.k[1]; d = p.d
        
 
#         # ── integral and shared denominators (unchanged) ─────────────────────
#         integral_k_d = self.hypergeometric_integral(
#             lb=self.psi_m_star_O[...,1], ub=self.psi_MP_star[...,1],
#             alpha=self.V_P[...,1], beta=self.w*p.fe[1]*p.r_hjort, y=k, z=d)
        
        
#         self.integral_k_d = integral_k_d
 
#         temp_w_a_d = np.divide(
#             1, (self.w[None,:,None]*self.a[...,1:])**d,
#             out=np.zeros_like(self.a[...,1:]), where=self.a[...,1:]!=0)
        
        
#         # ── mu quantities (unchanged) ────────────────────────────────────────
#         A = k*(1 - np.minimum(self.psi_m_star_O[...,1:],self.a_NP_star[...,1:])**(1-k)
#                + self.psi_m_star_O[...,1:]**(1-k)
#                - self.psi_MP_star[...,1:]**(1-k)) / (k-1)
#         B = k*np.einsum('nis,nis->nis',
#                         temp_w_a_d/self.V_NP[...,1:]**(-d),
#                         np.minimum(self.psi_m_star_O[...,1:],
#                                    self.a_NP_star[...,1:])**(d-k+1)-1) / (d-k+1)
#         C = k*np.einsum('nis,ni->nis', temp_w_a_d, integral_k_d)
 
#         self.mu_MNE  = np.zeros((p.N,p.N,p.S)); self.mu_MNE[...,1:]  = A-B-C
#         self.mu_MPND = np.zeros((p.N,p.N,p.S))
#         self.mu_MPND[...,1:] = C + k*self.psi_MP_star[...,1:]**(1-k)/(k-1)
#         D_ = k*(np.minimum(self.psi_m_star_O[...,1:],self.a_NP_star[...,1:])**(1-k)
#                 - self.psi_m_star_O[...,1:]**(1-k)) / (k-1)
#         self.mu_MNP  = np.zeros((p.N,p.N,p.S)); self.mu_MNP[...,1:]  = B+D_
        
 
#         # ── gamma fractions  [FDI] ────────────────────────────────────────────
#         psi_mO = self.psi_m_star_O[...,1:]
#         psi_mF = self.psi_m_star_F[...,1:]
#         a_po   = self.a_PF_PO[...,1:]
#         a_npo  = self.a_PF_NPO[...,1:]
#         a_npf  = self.a_NPF_NPO[...,1:]
#         psi_bar= self.psi_bar_NPO_PF[...,1:]
#         a_nis  = self.a[...,1:]
#         c2     = self.case2[...,1:]
        
#         a_po   = np.where(np.isfinite(a_po)   & (a_po   > 0), a_po,   np.inf)
#         a_npo  = np.where(np.isfinite(a_npo)  & (a_npo  > 0), a_npo,  np.inf)
#         a_npf  = np.where(np.isfinite(a_npf)  & (a_npf  > 0), a_npf,  np.inf)
#         psi_bar= np.where(np.isfinite(psi_bar) & (psi_bar > 0), psi_bar, 0.0)
#         psi_mO = np.where(np.isfinite(psi_mO) & (psi_mO > 1), psi_mO, 1.0)
#         psi_mF = np.where(np.isfinite(psi_mF) & (psi_mF > 1), psi_mF, 1.0)
        
        
#         # w_n * h_n * fe_s with shape (N,1,S-1) for broadcasting
#         w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])[:,None,:]
 
#         # safe a_nis for division
#         a_safe = np.where(a_nis > 0, a_nis, 1.0)
 
#         # ---------- gamma^{P,O}  (eq 21) ----------
#         # = k/(k-1)*psi_mO^{1-k}
#         #   - k/(k-d-1)*a_PF_PO^d * [psi_mO^{1-k-d} - max(a_PF_PO,psi_mO)^{1-k-d}]
#         g_PO_c2 = (
#             k/(k-1) * psi_mO**(1-k)
#             - k/(k-d-1) * a_po**d * (
#                 psi_mO**(1-k-d) - np.maximum(a_po, psi_mO)**(1-k-d)))
#         g_PO_c1 = k/(k-1) * psi_mO**(1-k)
#         self.gamma_PO = np.zeros((p.N,p.N,p.S))
#         self.gamma_PO[...,1:] = np.where(c2, g_PO_c2, g_PO_c1)
 
#         # ---------- gamma^{P,F}  (eq 22) — zero in Case 1 ----------
#         # = k/(k-d-1)*a_po^d*[psi_mO^{1-k-d}-max(a_po,psi_mO)^{1-k-d}]
#         #   + k/(k-1)*max(a_po,psi_mO)^{1-k}
#         #   + 1[a_PF_NPO<psi_mO] * k*(w_n h_n fe_s / a_nis)^d * psi_bar^k
#         #     * B(1-psibar/psi_mO, 1-psibar/a_PF_NPO ; d+2, d+k-1)
#         ind_PF = c2 & (a_npo < psi_mO)
#         g_PF = np.where(c2,
#             k/(k-d-1)*a_po**d*(psi_mO**(1-k-d)-np.maximum(a_po,psi_mO)**(1-k-d))
#             + k/(k-1)*np.maximum(a_po,psi_mO)**(1-k),
#             0.0)
#         t1_PF = np.where(ind_PF, 1-psi_bar/psi_mO,  0.0)
#         t2_PF = np.where(ind_PF, 1-psi_bar/a_npo,   0.0)
#         B_PF  = np.where(ind_PF, _betainc_vec(d+2, d+k-1, t1_PF, t2_PF), 0.0)
#         pre_PF = np.where(ind_PF & (a_nis>0),
#                           k*(w_fe_h/a_safe)**d * psi_bar**k, 0.0)
#         self.gamma_PF = np.zeros((p.N,p.N,p.S))
#         self.gamma_PF[...,1:] = g_PF + pre_PF*B_PF
 
#         # ---------- gamma^{NP,O}  (eq 24) ----------
#         # Case 2:
#         # = k/(k-1)*[1-psi_mF^{1-k}]
#         #   - 1[a_NPF_NPO>1]*k*a_npf^d/(k-d-1)*[1-min(a_npf,psi_mF)^{d+1-k}]
#         #   + 1[a_PF_NPO>psi_mF]*{k/(k-1)*[psi_mF^{1-k}-min(a_npo,psi_mO)^{1-k}]
#         #     - k*(w_n h_n fe_s/a_nis)^d*psi_bar^k
#         #       * B(1-psibar/psi_mF, 1-psibar/min(a_npo,psi_mO); d+2, d+k-1)}
#         g_NPO_c2 = (
#             k/(k-1)*(1-psi_mF**(1-k))
#             - np.where(a_npf > 1,
#                        k*a_npf**d/(k-d-1)*(1-np.minimum(a_npf,psi_mF)**(d+1-k)), 0.0))
#         ind_NPO  = c2 & (a_npo > psi_mF)
#         min_apo_mO = np.minimum(a_npo, psi_mO)
#         t1_NPO = np.where(ind_NPO, 1-psi_bar/psi_mF,     0.0)
#         t2_NPO = np.where(ind_NPO, 1-psi_bar/min_apo_mO, 0.0)
#         B_NPO  = np.where(ind_NPO, _betainc_vec(d+2, d+k-1, t1_NPO, t2_NPO), 0.0)
#         pre_NPO = np.where(ind_NPO & (a_nis>0),
#                            k*(w_fe_h/a_safe)**d * psi_bar**k, 0.0)
#         g_NPO_c2 += np.where(ind_NPO,
#             k/(k-1)*(psi_mF**(1-k)-min_apo_mO**(1-k)) - pre_NPO*B_NPO, 0.0)
#         # Case 1: no-FDI formula
#         g_NPO_c1 = (
#             k/(k-1)*(1-psi_mO**(1-k))
#             - k*temp_w_a_d/self.V_NP[...,1:]**(-d)
#               * (np.minimum(psi_mO,self.a_NP_star[...,1:])**(d-k+1)-1) / (d-k+1))
#         self.gamma_NPO = np.zeros((p.N,p.N,p.S))
#         self.gamma_NPO[...,1:] = np.where(c2, g_NPO_c2, g_NPO_c1)
 
#         # ---------- gamma^{NP,F}  (eq 26) — zero in Case 1 ----------
#         # = 1[a_npf>=psi_mF] * k*a_npf^d/(k-d-1)*[1-psi_mF^{d+1-k}]
#         #   + 1[1<=a_npf<psi_mF] * {k*a_npf^d/(k-d-1)*[1-a_npf^{d+1-k}]
#         #                           + k/(k-1)*[a_npf^{1-k}-psi_mF^{1-k}]}
#         #   + 1[a_npf<1] * k/(k-1)*[1-psi_mF^{1-k}]
#         g_NPF = (
#             np.where(c2 & (a_npf >= psi_mF),
#                      k*a_npf**d/(k-d-1)*(1-psi_mF**(d+1-k)), 0.0)
#             + np.where(c2 & (a_npf >= 1) & (a_npf < psi_mF),
#                        k*a_npf**d/(k-d-1)*(1-a_npf**(d+1-k))
#                        + k/(k-1)*(a_npf**(1-k)-psi_mF**(1-k)), 0.0)
#             + np.where(c2 & (a_npf < 1),
#                        k/(k-1)*(1-psi_mF**(1-k)), 0.0))
#         self.gamma_NPF = np.zeros((p.N,p.N,p.S))
#         self.gamma_NPF[...,1:] = g_NPF
 
#     # ── compute_aggregate_qualities  [FDI] ──────────────────────────────────
 
#     def compute_aggregate_qualities(self, p):
#         """
#         Eqs (27)-(35) of main_3_.tex.
#         Six quality stocks (O and F versions of P_ND, P_D, NP) instead of four.
#         In Case 1 all F stocks are zero and the O stocks match var_with_entry_costs.
#         """
#         prefact = p.eta[None,:,1:] * self.l_R[None,:,1:]**(1-p.kappa)  # (N,N,S-1)
#         gs  = self.g_s[1:]; nu = p.nu[1:]; ze = p.zeta[1:]
#         de  = p.delta[:,1:]   # (N,S-1) destination-specific obsolescence
#         A_NE  = gs + nu + ze
#         A_PND = gs[None,:] + de + nu + ze    # (N,S-1)
#         A_PD  = gs[None,:] + de + ze         # (N,S-1)
 
#         # Export stocks
#         self.PSI_M_P_ND_O = np.zeros((p.N,p.N,p.S))
#         self.PSI_M_P_ND_O[...,1:] = np.einsum(
#             'nis,nis->nis', prefact * self.gamma_PO[...,1:], 1/A_PND[:,None,:])
 
#         self.PSI_M_P_D_O = np.zeros((p.N,p.N,p.S))
#         self.PSI_M_P_D_O[...,1:] = np.einsum(
#             'nis,ns->nis', self.PSI_M_P_ND_O[...,1:], nu/A_PD)
 
#         self.PSI_M_NP_O = np.zeros((p.N,p.N,p.S))
#         num_O = (np.einsum('nis,nis->nis', prefact, self.gamma_NPO[...,1:])
#                  + np.einsum('ns,nis->nis', de, self.PSI_M_P_ND_O[...,1:]))
#         self.PSI_M_NP_O[...,1:] = np.einsum('nis,s->nis', num_O, 1/A_NE)
 
#         # FDI stocks  [FDI]
#         self.PSI_M_P_ND_F = np.zeros((p.N,p.N,p.S))
#         self.PSI_M_P_ND_F[...,1:] = np.einsum(
#             'nis,nis->nis', prefact * self.gamma_PF[...,1:], 1/A_PND[:,None,:])
 
#         self.PSI_M_P_D_F = np.zeros((p.N,p.N,p.S))
#         self.PSI_M_P_D_F[...,1:] = np.einsum(
#             'nis,ns->nis', self.PSI_M_P_ND_F[...,1:], nu/A_PD)
 
#         self.PSI_M_NP_F = np.zeros((p.N,p.N,p.S))
#         num_F = (np.einsum('nis,nis->nis', prefact, self.gamma_NPF[...,1:])
#                  + np.einsum('ns,nis->nis', de, self.PSI_M_P_ND_F[...,1:]))
#         self.PSI_M_NP_F[...,1:] = np.einsum('nis,s->nis', num_F, 1/A_NE)
 
#         # Aggregates
#         self.PSI_M_O = self.PSI_M_P_ND_O + self.PSI_M_P_D_O + self.PSI_M_NP_O
#         self.PSI_M_F = self.PSI_M_P_ND_F + self.PSI_M_P_D_F + self.PSI_M_NP_F
 
#         # Backward-compat aliases (PSI_ME = export mass, used for export trade flows)
#         self.PSI_ME   = self.PSI_M_O
#         self.PSI_MPND = self.PSI_M_P_ND_O
#         self.PSI_MPD  = self.PSI_M_P_D_O
#         self.PSI_MNP  = self.PSI_M_NP_O
 
#         # Non-entering stock (same formula as var_with_entry_costs)
#         self.PSI_MNE = np.zeros((p.N,p.N,p.S))
#         self.PSI_MNE[...,1:] = prefact * self.mu_MNE[...,1:] / A_NE[None,None,:]
 
#         self.PSI_M  = self.PSI_M_O + self.PSI_M_F + self.PSI_MNE
 
#         self.PSI_CD = np.ones((p.N,p.S))
#         self.PSI_CD[:,1:] = 1 - self.PSI_M[...,1:].sum(axis=1)
 
#     # ── compute_sectoral_prices  [FDI: FDI mass in denominator D] ───────────
 
#     def compute_sectoral_prices(self, p):
#         """
#         Eqs (36)-(39) of main_3_.tex.  The common denominator D_ns now
#         includes the FDI mass PSI_M_F weighted by phi_{nns} (destination's
#         own productivity).
#         """
#         power = p.sigma - 1
#         phi_nn = np.einsum('nns->ns', self.phi)   # (N,S) diagonal phi
 
#         A_exp = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,1:] * (
#             self.PSI_ME[...,1:] * self.phi[...,1:]**power[None,None,1:]
#         ).sum(axis=1)    # (N,S-1)
 
#         # FDI: affiliate sells in n using destination's technology phi_{nns}  [FDI]
#         A_fdi = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,1:] * (
#             self.PSI_M_F[...,1:] * phi_nn[:,None,1:]**power[None,None,1:]
#         ).sum(axis=1)    # (N,S-1)
 
#         B_cd = self.PSI_CD[:,1:] * (
#             self.phi[...,1:]**p.theta[None,None,1:]
#         ).sum(axis=1)**((power/p.theta)[None,1:])
 
#         D = A_exp + A_fdi + B_cd
 
#         self.P_M = np.full((p.N,p.S), np.inf)
#         self.P_M[:,1:] = (A_exp/D)**(1/(1-p.sigma[None,1:]))
 
#         self.P_M_F = np.full((p.N,p.S), np.inf)    # [FDI]
#         self.P_M_F[:,1:] = (A_fdi/D)**(1/(1-p.sigma[None,1:]))
 
#         self.P_CD = np.ones((p.N,p.S))
#         self.P_CD[:,1:] = (B_cd/D)**(1/(1-p.sigma[None,1:]))
 
#     # ── compute_labor_allocations  [FDI: adds l_F] ──────────────────────────
 
#     def compute_labor_allocations(self, p):
#         """
#         l_Ao, l_Ae, l_Aa: unchanged from var_with_entry_costs.
#         l_F: FDI setup labour, eq (41) of main_3_.tex.  [FDI]
#         """
#         k = p.k[1]; d = p.d
#         w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])  # (N,S-1)
 
#         # l_Ao (unchanged)
#         self.l_Ao = np.zeros((p.N,p.S))
#         self.l_Ao[...,1:] = np.einsum(
#             'i,s,is,is,is->is',
#             p.r_hjort, p.fo[1:], p.eta[...,1:],
#             self.l_R[...,1:]**(1-p.kappa),
#             self.psi_o_star[...,1:]**(-k))
 
#         # Shared integrals
#         temp_w_a_d = np.divide(
#             1, (self.w[None,:,None]*self.a[...,1:])**d,
#             out=np.zeros_like(self.a[...,1:]), where=self.a[...,1:]!=0)
#         temp_w_a_d1 = np.divide(
#             1, (self.w[None,:,None]*self.a[...,1:])**(d+1),
#             out=np.zeros_like(self.a[...,1:]), where=self.a[...,1:]!=0)
 
#         integral_k1_d = self.hypergeometric_integral(
#             lb=self.psi_m_star_O[...,1], ub=self.psi_MP_star[...,1],
#             alpha=self.V_P[...,1], beta=self.w*p.fe[1]*p.r_hjort, y=k+1, z=d)
#         self.integral_k_plus_un_d = integral_k1_d
 
#         # l_Ae (unchanged)
#         self.l_Ae = np.zeros((p.N,p.N,p.S))
#         self.l_Ae[...,1:] = np.einsum(
#             'n,s,is,is,nis->ins',
#             p.r_hjort, p.fe[1:], p.eta[...,1:],
#             self.l_R[...,1:]**(1-p.kappa),
#             k*temp_w_a_d*integral_k1_d[...,None] + self.psi_MP_star[...,1:]**(-k))
 
#         # l_Aa (unchanged)
#         A_aa = k*np.einsum('nis,nis,nis->nis',
#                            temp_w_a_d1, 1/self.V_NP[...,1:]**(-d-1),
#                            np.minimum(self.psi_m_star_O[...,1:],
#                                       self.a_NP_star[...,1:])**(d-k+1)-1) / (d-k+1)
#         B_aa = (self.psi_MP_star[...,1:]**(-k)
#                 - self.psi_m_star_O[...,1:]**(-k)
#                 + np.minimum(self.psi_m_star_O[...,1:],
#                              self.a_NP_star[...,1:])**(-k))
#         lb_ = self.psi_m_star_O[...,1]; ub_ = self.psi_MP_star[...,1]
#         al_ = self.V_P[...,1];          be_ = self.w*p.fe[1]*p.r_hjort
#         tlb = lb_**(-k)*(al_*lb_-be_[:,None])**(d+1)/k
#         tub = ub_**(-k)*(al_*ub_-be_[:,None])**(d+1)/k
#         integral_k1_d1 = tlb - tub + al_*(d+1)*self.integral_k_d/k
#         self.integral_k_plus_un_d_plus_un = integral_k1_d1
#         C_aa = k*temp_w_a_d1*integral_k1_d1[...,None]
#         self.l_Aa = np.zeros((p.N,p.N,p.S))
#         self.l_Aa[...,1:] = d*np.einsum(
#             'is,is,nis,nis->nis',
#             p.eta[:,1:], self.l_R[...,1:]**(1-p.kappa),
#             self.a[...,1:], A_aa+B_aa+C_aa) / (d+1)
 
#         # l_F: FDI setup labour  [FDI]
#         # L^F_{ins} = eta_is*(L^R_is)^{1-kappa} * a_nis^d/(d+1) * Lambda^F_nis
#         # Lambda^F using the gamma route (equivalent to the piecewise integral):
#         #   Lambda^F = (gamma_NPF + gamma_PF) * (k-1)/k
#         Lambda_F = ((self.gamma_NPF[...,1:] + self.gamma_PF[...,1:])
#                     * (k-1)/k)
#         self.l_F = np.zeros((p.N,p.N,p.S))
#         self.l_F[...,1:] = np.where(
#             self.case2[...,1:],
#             np.einsum('is,is,nis,nis->nis',
#                       p.eta[:,1:], self.l_R[...,1:]**(1-p.kappa),
#                       self.a[...,1:]**d/(d+1), Lambda_F),
#             0.0)
 
#         # l_P includes l_F  [FDI]
#         self.l_P = p.labor - (
#             self.l_Ao + self.l_R
#             + self.l_Ae.sum(axis=0)
#             + self.l_Aa.sum(axis=0)
#             + self.l_F.sum(axis=0)
#         ).sum(axis=1)
 
#     # ── compute_trade_flows_and_shares  [FDI: adds X_M_F] ───────────────────
 
#     def compute_trade_flows_and_shares(self, p, assign=True):
#         """
#         X_M, X_CD unchanged (use PSI_ME = PSI_M_O).
#         X_M_F: FDI affiliate sales, uses PSI_M_F and phi_{nns}.  [FDI]
#         """
#         # Export monopolist flows (unchanged)
#         temp_exp = (self.PSI_ME[...,1:]*self.phi[...,1:]**(p.sigma-1)[None,None,1:]
#                     ).sum(axis=1)
#         X_M = np.zeros((p.N,p.N,p.S))
#         X_M[...,1:] = (
#             self.phi[...,1:]**(p.sigma-1)[None,None,1:]
#             * self.PSI_ME[...,1:]
#             / temp_exp[:,None,:]
#             * self.P_M[:,None,1:]**(1-p.sigma[None,1:])
#             * p.beta[None,None,1:]
#             * self.Z[None,:,None])
 
#         # FDI affiliate flows  [FDI]
#         # Affiliates in n from origin i sell using phi_{nns}
#         phi_nn = np.einsum('nns->ns', self.phi)     # (N,S)
#         temp_fdi = (self.PSI_M_F[...,1:]
#                     * phi_nn[:,None,1:]**(p.sigma-1)[None,None,1:]
#                     ).sum(axis=1)                    # (N,S-1)
#         X_M_F = np.zeros((p.N,p.N,p.S))
#         safe_fdi = np.where(temp_fdi > 0, temp_fdi, 1.0)
#         X_M_F[...,1:] = np.where(
#             temp_fdi[:,None,:] > 0,
#             (self.PSI_M_F[...,1:]
#              * phi_nn[:,None,1:]**(p.sigma-1)[None,None,1:]
#              / safe_fdi[:,None,:]
#              * p.beta[None,None,1:]
#              * self.Z[None,:,None])
#             * self.P_M_F[:,None,1:]**(1-p.sigma[None,1:]),
#             0.0)
 
#         # Competitive goods (unchanged)
#         X_CD = (
#             self.phi**p.theta[None,None,:]
#             / (self.phi**p.theta[None,None,:]).sum(axis=1)[:,None,:]
#             * self.P_CD[:,None,:]**(1-p.sigma[None,None,:])
#             * p.beta[None,None,:]
#             * self.Z[None,:,None])
 
#         X = X_M + X_CD
#         if assign:
#             self.X_M=X_M; self.X_M_F=X_M_F; self.X_CD=X_CD; self.X=X
#         else:
#             return X_M, X_M_F, X_CD, X
 
#     # ── solver pipeline ──────────────────────────────────────────────────────
 
#     def compute_solver_quantities(self, p):
#         # print('start')
#         self.compute_growth(p)
#         self.compute_entry_costs(p)
#         self.compute_V(p)
#         self.compute_case_indicator(p)           # [FDI]
#         self.compute_auxiliary_thresholds(p)     # [FDI]
#         self.compute_patenting_thresholds(p)
#         # print('here')
#         self.compute_mass_innovations(p)
#         self.compute_aggregate_qualities(p)
#         self.compute_sectoral_prices(p)
#         self.compute_labor_allocations(p)
#         self.compute_trade_flows_and_shares(p)
 
#     # ── update equations ─────────────────────────────────────────────────────
 
#     def compute_price_indices(self, p):
#         """Eq (29) extended with FDI mass.  [FDI]"""
#         power = p.sigma-1
#         phi_nn = np.einsum('nns->ns', self.phi)
#         A_exp = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,:] * (
#             self.PSI_ME * self.phi**power[None,None,:]).sum(axis=1)
#         A_fdi = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,:] * (
#             self.PSI_M_F * phi_nn[:,None,:]**power[None,None,:]).sum(axis=1)
#         B = self.PSI_CD*(self.phi**p.theta[None,None,:]).sum(axis=1)**(
#             power/p.theta)[None,:]
#         temp = gamma((p.theta+1-p.sigma)/p.theta)[None,:]*(A_exp+A_fdi+B)
#         one_over = np.divide(1,temp,out=np.full_like(temp,np.inf),where=temp>0)
#         return (one_over**(p.beta[None,:]/(p.sigma[None,:]-1))).prod(axis=1)
 
#     def compute_wage(self, p):
#         """Eq (32) — unchanged."""
#         return (p.alpha[None,:] * (
#             (self.X - self.X_M/p.sigma[None,None,:])/(1+p.tariff)
#         ).sum(axis=0)).sum(axis=1) / self.l_P
 
#     def compute_expenditure(self, p):
#         """Eq (33) extended with FDI labour and profit flows.  [FDI]"""
#         A1 = np.einsum('nis,nis->i', self.X, 1/(1+p.tariff))
#         A2 = np.einsum('ins,ins,ins->i', self.X, p.tariff, 1/(1+p.tariff))
#         B  = np.einsum('i,nis->i', self.w, self.l_Ae)
#         C  = p.deficit_share_world_output*np.einsum('nis,nis->',self.X,1/(1+p.tariff))
#         D  = np.einsum('n,ins->i', self.w, self.l_Ae)
#         # FDI labour: borne by innovator i, paid to workers in n  [FDI]
#         B_F = np.einsum('i,nis->i', self.w, self.l_F)
#         D_F = np.einsum('n,ins->i', self.w, self.l_F)
#         # FDI profit repatriation  [FDI]
#         FDI_in  = np.einsum('nis,s->i', self.X_M_F[...,1:], 1/p.sigma[1:])
#         FDI_out = np.einsum('ins,s->i', self.X_M_F[...,1:], 1/p.sigma[1:])
#         return A1+A2+B+B_F+FDI_in - (C+D+D_F+FDI_out)
 
#     def compute_profit(self, p):
#         """Eq (31) export normalised profits — unchanged."""
#         profit = np.zeros((p.N,p.N,p.S))
#         profit[...,1:] = np.einsum(
#             'nis,s,i,nis,nis->nis',
#             self.X_M[...,1:], 1/p.sigma[1:], 1/self.w,
#             1/self.PSI_ME[...,1:], 1/(1+p.tariff[...,1:]))
#         return profit
 
#     # def compute_pi_F(self, p):
#     #     """[FDI] Pi^{w,F}_{nis} = X_M_F_{nis} / (sigma_s * Psi_M_F_{nis} * w_n)"""
#     #     pi_F = np.zeros((p.N,p.N,p.S))
#     #     safe = np.where(self.PSI_M_F[...,1:]>0, self.PSI_M_F[...,1:], 1.0)
#     #     pi_F[...,1:] = np.einsum(
#     #         'nis,s,n,nis->nis',
#     #         self.X_M_F[...,1:], 1/p.sigma[1:], 1/self.w, 1/safe)
#     #     self.safe=safe
#     #     pi_F[self.PSI_M_F==0] = 0.0
#     #     return pi_F
    
#     # def compute_pi_F(self, p):
#     #     pi_F = self.pi_F.copy()  # start from current guess, not zero
#     #     mask = self.PSI_M_F[...,1:] > 0
#     #     safe = np.where(mask, self.PSI_M_F[...,1:], 1.0)
#     #     pi_F[...,1:] = np.where(
#     #         mask,
#     #         np.einsum('nis,s,n,nis->nis',
#     #                   self.X_M_F[...,1:], 1/p.sigma[1:], 1/self.w, 1/safe),
#     #         self.pi_F[...,1:])  # keep current value where PSI_M_F = 0
#     #     return pi_F
    
#     def compute_pi_F(self, p):
#         pi_F = np.zeros((p.N, p.N, p.S))
#         mask = self.PSI_M_F[...,1:] > 0
#         safe = np.where(mask, self.PSI_M_F[...,1:], 1.0)
#         pi_F[...,1:] = np.where(
#             mask,
#             np.einsum('nis,s,n,nis->nis',
#                       self.X_M_F[...,1:], 1/p.sigma[1:], 1/self.w, 1/safe),
#             0.0)
#         return pi_F
 
#     def compute_labor_research(self, p):
#         """Eq (30) — identical to var_with_entry_costs."""
#         k=p.k[1]; d=p.d
#         w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])
#         temp_w_a_d = np.divide(
#             1,(self.w[None,:,None]*self.a[...,1:])**d,
#             out=np.zeros_like(self.a[...,1:]),where=self.a[...,1:]!=0)
#         A1=k*np.einsum('nis,i,nis->nis',self.V_NP[...,1:],1/self.w,
#                        self.a_NP_star[...,1:])/(k-1)
#         A =np.einsum('nis,nis->nis',A1-d*self.a[...,1:]/(d+1),
#                      self.a_NP_star[...,1:]**(-k))
#         B1=k*np.einsum('nis,i,nis->nis',self.V_NP[...,1:],1/self.w,
#                        self.psi_MNP_star[...,1:])/(k-1)
#         B =np.einsum('nis,nis->nis',B1-d*self.a[...,1:]/(d+1),
#                      self.psi_MNP_star[...,1:]**(-k))
#         sigC=np.einsum('i,nis,nis->nis',self.w,self.a[...,1:],
#                        1/self.V_NP[...,1:])>1
#         C=k*np.einsum('nis,i,nis,nis,nis->nis',
#                       temp_w_a_d,1/self.w,self.V_NP[...,1:]**(d+1),sigC,
#                       np.minimum(self.psi_m_star_O[...,1:],
#                                  self.a_NP_star[...,1:])**(d-k+1)-1
#                       )/((d+1)*(d-k+1))
#         D1=k*np.einsum('nis,i,nis->nis',self.V_P[...,1:],1/self.w,
#                        self.psi_MP_star[...,1:])/(k-1)
#         D2=(self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None]
#             /self.w[None,:,None] + d*self.a[...,1:]/(d+1))
#         D =np.einsum('nis,nis->nis',D1-D2,self.psi_MP_star[...,1:]**(-k))
#         sigE=((self.w[None,:,None]*self.a[...,1:]
#                +self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None])
#               /self.V_P[...,1:] > self.psi_o_star[None,:,1:])
#         E=k*np.einsum('nis,nis,i,ni->nis',sigE,temp_w_a_d,1/self.w,
#                       self.integral_k_plus_un_d_plus_un)/(d+1)
#         l_R=np.zeros((p.N,p.S))
#         temp=((A-B+C+D+E).sum(axis=0)
#               - p.fo[None,1:]*p.r_hjort[:,None]*self.psi_o_star[...,1:]**(-k))
#         l_R[...,1:]=(temp*p.eta[...,1:])**(1/p.kappa)
#         return l_R
 
#     def compute_phi(self, p):
#         """Calibration / counterfactual phi — unchanged."""
#         if self.context=='calibration':
#             denom_M=np.zeros((p.N,p.N,p.S))
#             # print((1/((self.PSI_ME[...,1:]*self.phi[...,1:]**(p.sigma-1)[None,None,1:]
#             #     ).sum(axis=1))).shape)
#             denom_M[...,1:]=np.einsum(
#                 'nis,nis,ns,ns->nis',
#                 self.PSI_ME[...,1:],
#                 self.phi[...,1:]**((p.sigma-1)-p.theta)[None,None,1:],
#                 1/((self.PSI_ME[...,1:]*self.phi[...,1:]**(p.sigma[None,None,1:]-1)
#                     ).sum(axis=1)),
#                 self.P_M[:,1:]**(1-p.sigma[None,1:])
#                 )
#             denom_CD=np.einsum(
#                 'ns,ns->ns',
#                 1/(self.phi**p.theta[None,None,:]).sum(axis=1),
#                 self.P_CD[:,:]**(1-p.sigma[None,:]).squeeze())
#             f_phi=np.einsum(
#                 'nis,nis,nis->nis',
#                 p.trade_shares,1+p.tariff,
#                 1/(denom_M+denom_CD[:,None,:]))
#             return np.einsum(
#                 'nis,nns,ns,ns,ns->nis',
#                 f_phi**(1/p.theta)[None,None,:],
#                 f_phi**(-1/p.theta)[None,None,:],
#                 p.T**(1/p.theta[None,:]),
#                 self.w[:,None]**(-p.alpha[None,:]),
#                 self.price_indices[:,None]**(p.alpha[None,:]-1))
#         elif self.context=='counterfactual':
#             return np.einsum(
#                 'is,nis,nis,is,is->nis',
#                 p.T**(1/p.theta[None,:]),
#                 1/p.tau, 1/(1+p.tariff),
#                 self.w[:,None]**(-p.alpha[None,:]),
#                 self.price_indices[:,None]**(p.alpha[None,:]-1))
 
#     # ── non-solver quantities ────────────────────────────────────────────────
 
#     def scale_P(self, p):
#         """Normalise all nominal quantities.  [FDI: adds X_M_F]"""
#         num=self.price_indices[0]
#         self.w/=num; self.Z/=num; self.X/=num
#         self.X_CD/=num; self.X_M/=num; self.X_M_F/=num   # [FDI]
#         self.phi*=num; self.price_indices/=num
#         self.compute_sectoral_prices(p)
 
#     def compute_tau(self, p, assign=True):
#         tau=np.einsum('is,nis,nis,is,is->nis',
#                       p.T**(1/p.theta[None,:]),1/self.phi,1/(1+p.tariff),
#                       self.w[:,None]**-p.alpha[None,:],
#                       self.price_indices[:,None]**(p.alpha[None,:]-1))
#         if assign: self.tau=tau
#         else: return tau
 
#     def compute_nominal_value_added(self,p):
#         self.nominal_value_added=(
#             p.alpha[None,:]*((self.X-self.X_M/p.sigma[None,None,:])
#                              /(1+p.tariff)).sum(axis=0))
 
#     def compute_nominal_intermediate_input(self,p):
#         self.nominal_intermediate_input=np.einsum(
#             's,is->is',(1-p.alpha)/p.alpha,self.nominal_value_added)
 
#     def compute_nominal_final_consumption(self,p):
#         self.nominal_final_consumption=(
#             self.Z-self.nominal_intermediate_input.sum(axis=1))
#         self.cons=self.nominal_final_consumption/self.price_indices
 
#     def compute_gdp(self,p):
#         """Eq (37) extended with l_F.  [FDI]"""
#         self.gdp=(
#             self.nominal_final_consumption
#             +p.deficit_share_world_output*np.einsum('nis,nis->',self.X,1/(1+p.tariff))
#             +self.w*np.einsum('is->i',self.l_R+self.l_Ao)
#             +np.einsum('n,ins->i',self.w,self.l_Ae)
#             +np.einsum('n,nis->i',self.w,self.l_Aa)
#             +np.einsum('n,nis->i',self.w,self.l_F))    # [FDI]
 
#     def compute_pflow(self,p):
#         k=p.k[1]; d=p.d
#         temp_w_a_d=np.divide(1,(self.w[None,:,None]*self.a[...,1:])**d,
#                               out=np.zeros_like(self.a[...,1:]),
#                               where=self.a[...,1:]!=0)
#         bracket=(k*np.einsum('nis,ni->nis',temp_w_a_d,self.integral_k_plus_un_d)
#                  +self.psi_MP_star[...,1:]**(-k))
#         self.pflow=np.einsum('nis,is,is->nis',bracket,p.eta[...,1:],
#                              self.l_R[...,1:]**(1-p.kappa)).squeeze()
 
#     def compute_share_of_innovations_patented(self,p):
#         self.share_innov_patented=self.psi_m_star_O[...,1:]**(-p.k[1])
 
#     def compute_non_solver_quantities(self,p):
#         self.compute_tau(p)
#         self.compute_nominal_value_added(p)
#         self.compute_nominal_intermediate_input(p)
#         self.compute_nominal_final_consumption(p)
#         self.compute_gdp(p)
#         self.compute_pflow(p)
#         self.compute_share_of_innovations_patented(p)
 
#     def compute_consumption_equivalent_welfare(self,p,baseline):
#         self.cons_eq_welfare=(
#             self.cons
#             *((p.rho-baseline.g*(1-1/p.gamma))
#               /(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))
#             /baseline.cons)
 
#     def compute_world_welfare_changes(self,p,baseline):
#         one_ov_g=1/p.gamma
#         n=(p.labor**one_ov_g*self.cons**((p.gamma-1)*one_ov_g)).sum()*(
#             p.rho-baseline.g*(1-one_ov_g))
#         d=(p.labor**one_ov_g*baseline.cons**((p.gamma-1)*one_ov_g)).sum()*(
#             p.rho-self.g*(1-one_ov_g))
#         self.cons_eq_pop_average_welfare_change=(n/d)**(p.gamma/(p.gamma-1))
#         n2=(baseline.cons**one_ov_g*self.cons**((p.gamma-1)*one_ov_g)).sum()*(
#             p.rho-baseline.g*(1-one_ov_g))
#         d2=baseline.cons.sum()*(p.rho-self.g*(1-one_ov_g))
#         self.cons_eq_negishi_welfare_change=(n2/d2)**(p.gamma/(p.gamma-1))

def _betainc_vec(a, b, t1, t2):
    """Vectorised non-regularised incomplete Beta function.

    Computed as: B_unreg(a, b, t1, t2) = [I_reg(a,b,t2) - I_reg(a,b,t1)] * B(a,b)
    where I_reg is scipy's regularized incomplete beta (fast C code).
    """
    import scipy.special as sp
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    t1 = np.clip(np.asarray(t1, dtype=float), 0.0, 1.0)
    t2 = np.clip(np.asarray(t2, dtype=float), 0.0, 1.0)
    # Broadcast scalars to array shapes
    if np.ndim(a) == 0 and np.ndim(t1) > 0:
        a = np.broadcast_to(a, t1.shape)
    if np.ndim(b) == 0 and np.ndim(t1) > 0:
        b = np.broadcast_to(b, t1.shape)
    # Need a > 0 and b > 0 for scipy's betainc/beta to be well-defined.
    # If a<=0 or b<=0, fall back to mpmath element-wise (slow path).
    bad = (a <= 0) | (b <= 0)
    if bad.any():
        return _betainc_vec_mpmath(a, b, t1, t2)
    # Fast path: regularized betainc * full beta
    res = (sp.betainc(a, b, t2) - sp.betainc(a, b, t1)) * sp.beta(a, b)
    res = np.where(np.isfinite(res), res, 0.0)
    return res


def _betainc_vec_mpmath(a, b, t1, t2):
    """Slow mpmath fallback for non-positive shape parameters."""
    return np.vectorize(
        lambda aa, bb, x1, x2: float(betainc(aa, bb,
            max(0.0, min(1.0, float(x1))),
            max(0.0, min(1.0, float(x2))), regularized=False).real)
    )(a, b, t1, t2)
 
 
# def _betainc_vec(a, b, t1, t2):
#     """Vectorised non-regularised incomplete Beta function."""
#     return np.vectorize(
#         lambda aa, bb, x1, x2: float(betainc(aa, bb, 
#                                               max(0.0, min(1.0, float(x1))), 
#                                               max(0.0, min(1.0, float(x2))), 
#                                               regularized=False).real)
#     )(a, b, t1, t2)
# ─────────────────────────────────────────────────────────────────────────────
# var_with_fdi
# ─────────────────────────────────────────────────────────────────────────────
 
# ─────────────────────────────────────────────────────────────────────────────
# var_with_fdi  (CORRECTED VERSION — May 2026)
#
# This file rewrites the `var_with_fdi` class to match algorithm_corrected.tex.
# To use it: replace the existing var_with_fdi class definition (lines
# 1595-2753 of classes.py) with this entire block. Imports (numpy, betainc,
# _betainc_vec, etc.) must already be in scope from the surrounding classes.py.
#
# === KEY CORRECTIONS APPLIED (vs the previous version) ======================
#  1.  ValmidPs / ValnoorigPs (eqs 15', 17') now include the missing factor
#      psi multiplying V^{NP,O} and V^{NP,F}.  Previous code had
#      V_NPO*(1 - e^{-d}) and V_NPF*e^{-d} without psi.
#  2.  The original-patent threshold psi^{o*} solves a SINGLE MIXED Case 1 /
#      Case 2 equation (eqs 14', 18', 19'). For each origin i the sum runs
#      over destinations n, each in its own case (diagonal n=i is always
#      Case 1).
#  3.  gamma quality shares now use beta-function shape (d+1, k-d-1) and
#      prefactor psi_bar^{1-k} (eqs 21'-26'). Previous code had
#      (d+2, d+k-1) and psi_bar^k.
#  4.  Aggregate qualities (eqs 28', 31', 32') drop the extra k/(k-1)
#      multiplying gamma (already done correctly in the previous code, kept).
#  5.  Patent-application labor L^e is the mixed Case 1 / Case 2 expression
#      (eq 40') with beta-shape (d+1, k-d). Previously L^e (alias l_Ae) used
#      the Case-1-only formula (k * w_a^{-d} * integral_{k+1,d} + psi_MP^{-k}).
#  6.  Affiliate-labor Lambda^F (eq 42) uses beta-shape (d+2, k-d-1).
#      Previously (d+2, d+k-1).
#  7.  Innovation-value RHS (eq 49') has NO leading no-FDI sum. The full
#      IntVNorigP + IntVorigP^{A,B,C,D} block (eqs 50'-54') is the complete
#      value of innovation. Previous code disabled this block and used the
#      old Case-1-only formula for all pairs.
#  8.  IntV blocks (50'-54') use negative-exponent probability terms psi^{-k}
#      (not psi^k); denominator (d-k+1) where appropriate; beta-shape
#      (d+2, k-d-1) in IntVorigP^C.
#  9.  Wage equation (eq 57') divides X by (1+b) AND subtracts
#      -sigma^{-1} sum_j X^{M,F}_{ijs}. Previous code missed the FDI-profit
#      term in compute_wage (returned per-origin wage without it).
# 10.  Income/spending (eq 58') now includes L^F service flows on BOTH sides
#      (exports and imports); affiliate-profit signs corrected (FDI_in
#      = profits earned ABROAD by i is POSITIVE; FDI_out = profits earned
#      in i by foreign affiliates is NEGATIVE).
#
# === REMOVED (unnecessary in the corrected formulation) =====================
#   - mu_MNE, mu_MPND, mu_MNP  (replaced by gamma's which already are mu)
#   - PSI_MNE
#   - psi_MP_star, psi_MNP_star, a_NP_star, a_P_star  (Case-1 aleph thresholds
#     used by the old Case-1-only labor formulas)
#   - integral_k_d, integral_k_plus_un_d, integral_k_plus_un_d_plus_un
#   - hypergeometric_integral static method
#   - l_Aa  (legacy patent-application "additional" labor; only L^e, L^F,
#     L^o, L^R survive in the corrected algorithm)
#
# === KEPT ALIASES (used by moments code in classes.py lines 6301-6303,
#                   7292-7295, 8979-9018) =====================================
#   - PSI_ME   = PSI_M_O      (alias for export mass, used internally)
#   - PSI_MPND = PSI_M_P_ND_O
#   - PSI_MPD  = PSI_M_P_D_O
#   - PSI_MNP  = PSI_M_NP_O
# ─────────────────────────────────────────────────────────────────────────────

class var_with_fdi:
    """
    Variable container for the steady-state solver with entry costs AND FDI.
    Implements algorithm_corrected.tex equations.
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(self, context, N=7, S=2):
        m = np.ones((N, N, S), bool).ravel()
        m[np.s_[::(N + 1) * S]] = False
        m[np.s_[1::(N + 1) * S]] = False
        self.off_diag_mask = m.reshape((N, N, S))
        self.diag_mask = ~self.off_diag_mask
        self.context = context

    # ── guess setters ────────────────────────────────────────────────────────

    def guess_profit(self, v):        self.profit = v
    def guess_wage(self, v):          self.w = v
    def guess_Z(self, v):             self.Z = v
    def guess_labor_research(self, v):self.l_R = v
    def guess_phi(self, v):           self.phi = v
    def guess_price_indices(self, v): self.price_indices = v
    def guess_pi_F(self, v):          self.pi_F = v

    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])

    def copy(self):
        return deepcopy(self)

    # ── vector ↔ var ────────────────────────────────────────────────────────

    @staticmethod
    def var_from_vector(vec, p, context, compute=True):
        """
        Vector layout (lengths):
            w             N
            Z             N
            l_R           N*(S-1)
            profit        N*N*(S-1)
            phi           N*N*S
            price_indices N
            pi_F          N*N*(S-1)
        """
        N, S = p.N, p.S
        v = var_with_fdi(context=context, N=N, S=S)
        i0 = 0
        v.guess_wage(vec[i0:i0+N]);                                        i0 += N
        v.guess_Z(vec[i0:i0+N]);                                           i0 += N
        v.guess_labor_research(
            np.insert(vec[i0:i0+N*(S-1)].reshape((N, S-1)),
                      0, np.zeros(N), axis=1));                            i0 += N*(S-1)
        v.guess_profit(
            np.insert(vec[i0:i0+N*N*(S-1)].reshape((N, N, S-1)),
                      0, np.zeros(N), axis=2));                            i0 += N*N*(S-1)
        v.guess_phi(vec[i0:i0+N*N*S].reshape((N, N, S)));                 i0 += N*N*S
        v.guess_price_indices(vec[i0:i0+N]);                               i0 += N
        # pi_F: may be absent in legacy guesses; default to zero (no-FDI start).
        remaining = vec[i0:]
        pi_F_size = N * N * (S - 1)
        if len(remaining) >= pi_F_size:
            pi_F_flat = remaining[:pi_F_size]
        else:
            pi_F_flat = np.zeros(pi_F_size)
        v.guess_pi_F(
            np.insert(pi_F_flat.reshape((N, N, S-1)),
                      0, np.zeros(N), axis=2))
        if compute:
            v.compute_solver_quantities(p)
        return v

    def vector_from_var(self):
        return np.concatenate([
            self.w,
            self.Z,
            self.l_R[..., 1:].ravel(),
            self.profit[..., 1:].ravel(),
            self.phi.ravel(),
            self.price_indices,
            self.pi_F[..., 1:].ravel(),
        ])

    # ── compute_growth ──────────────────────────────────────────────────────

    def compute_growth(self, p):
        self.g_s = (p.k * np.einsum('is,is->s', p.eta, self.l_R**(1-p.kappa))
                    / (p.k-1) - p.zeta)
        self.g_s[0] = p.g_0
        self.g = (p.beta * self.g_s / (p.sigma-1)).sum() / (p.beta * p.alpha).sum()
        self.r = p.rho + self.g / p.gamma
        self.G = self.r + p.zeta - self.g + self.g_s + p.nu

    # ── compute_entry_costs ─────────────────────────────────────────────────

    def compute_entry_costs(self, p):
        # FDI setup cost is the bilateral calibrated parameter a^s_{ni} directly:
        #     var.a[n, i, s] = p.a[n, i, s]
        # No supply-potential proxy, no power_fdi.  The previous formula
        #   self.a = p.a * max(supply_potential, 1)**p.power_fdi
        # was a parsimonious proxy used when p.a was a single scalar; it
        # served to spread one number across (n, i, s) using trade-resistance
        # information. With bilateral calibration, p.a has N*N*S entries and
        # IS the structural FDI cost matrix — no further multiplicative
        # wrapping is appropriate. p.power_fdi is therefore unused.
        # The diagonal (n == i) is enforced to zero (no FDI to self).
        # Sector-0 entries are also zero since FDI applies only to sector 1.
        # Both contexts share the same formula.
        self.a = np.broadcast_to(np.asarray(p.a), (p.N, p.N, p.S)).copy()
        np.einsum('nns->ns', self.a)[:] = 0
        if p.S >= 2:
            self.a[..., 0] = 0

    # ── compute_V ───────────────────────────────────────────────────────────

    def compute_V(self, p):
        """Export and FDI normalised value functions."""
        self.V_NP = np.einsum('nis,i,s->nis', self.profit, self.w, 1/self.G)
        self.V_P  = np.einsum(
            'nis,i,ns->nis', self.profit, self.w,
            1/(self.G[None,:]-p.nu[None,:]+p.delta)
            - 1/(self.G[None,:]+p.delta) + 1/self.G[None,:])
        # FDI value functions: driven by pi_F, wage is w_n
        self.V_NP_F = np.einsum('nis,n,s->nis', self.pi_F, self.w, 1/self.G)
        self.V_P_F  = np.einsum(
            'nis,n,ns->nis', self.pi_F, self.w,
            1/(self.G[None,:]-p.nu[None,:]+p.delta)
            - 1/(self.G[None,:]+p.delta) + 1/self.G[None,:])

    # ── compute_case_indicator ──────────────────────────────────────────────

    def compute_case_indicator(self, p):
        """Case 2_{nis}: w_i * pi^w_{nis} < w_n * Pi^{w,F}_{nis}  (n != i)."""
        self.case2 = np.zeros((p.N, p.N, p.S), bool)
        self.case2[..., 1:] = (
            (self.w[None,:,None] * self.profit[...,1:]
             < self.w[:,None,None] * self.pi_F[...,1:])
            & self.off_diag_mask[...,1:])

    # ── compute_auxiliary_thresholds  (eqs 6-7) ─────────────────────────────

    def compute_auxiliary_thresholds(self, p):
        """Auxiliary FDI-cost thresholds. +inf in Case 1."""
        w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])
        eps = 1e-30
        dNP     = self.V_NP_F[...,1:] - self.V_NP[...,1:]
        dP      = self.V_P_F[...,1:]  - self.V_P[...,1:]
        dPF_NPO = self.V_P_F[...,1:]  - self.V_NP[...,1:]
        mask    = self.case2[...,1:]
        w_a     = self.w[:,None,None] * self.a[...,1:]

        def _thresh(num, den):
            out = np.full(num.shape, np.inf)
            out[mask] = num[mask] / (den[mask] + eps)
            return out

        self.a_NPF_NPO = np.full((p.N,p.N,p.S), np.inf)
        self.a_NPF_NPO[...,1:] = _thresh(w_a, dNP)
        self.a_PF_PO = np.full((p.N,p.N,p.S), np.inf)
        self.a_PF_PO[...,1:] = _thresh(w_a, dP)
        self.a_PF_NPO = np.full((p.N,p.N,p.S), np.inf)
        self.a_PF_NPO[...,1:] = _thresh(w_a + w_fe_h[:,None,:], dPF_NPO)
        self.psi_bar_NPO_PF = np.full((p.N,p.N,p.S), np.inf)
        self.psi_bar_NPO_PF[...,1:] = _thresh(
            np.broadcast_to(w_fe_h[:,None,:], w_a.shape).copy(), dPF_NPO)

    # ── compute_patenting_thresholds  (eqs 11-13, 14', 18'-19') ─────────────

    def compute_patenting_thresholds(self, p):
        """
        psi^{*,O}, psi^{*,F}, and the original-patent threshold psi^{o*}_{is}
        which solves the MIXED Case 1 / Case 2 equation (19').
        """
        w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])

        # ── psi^{*,O} (Case 1 eq 11, Case 2 eq 12) ──────────────────────────
        psi_C1 = np.full((p.N,p.N,p.S), np.inf)
        psi_C1[...,1:] = (
            w_fe_h[:,None,:] /
            (self.profit[...,1:] * self.w[None,:,None]
             * (1/(self.G[None,:]+p.delta-p.nu[None,:])
                - 1/(self.G[None,:]+p.delta))[:,None,1:]))
        psi_C2_O = np.full((p.N,p.N,p.S), np.inf)
        psi_C2_O[...,1:] = np.where(
            self.case2[...,1:],
            w_fe_h[:,None,:] / (self.V_P[...,1:] - self.V_NP[...,1:] + 1e-30),
            np.inf)
        self.psi_star_O = np.maximum(np.where(self.case2, psi_C2_O, psi_C1), 1)
        self.psi_star   = self.psi_star_O   # alias

        # ── psi^{*,F} (eq 13, Case 2 only) ──────────────────────────────────
        self.psi_star_F = np.full((p.N,p.N,p.S), np.inf)
        self.psi_star_F[...,1:] = np.where(
            self.case2[...,1:],
            w_fe_h[:,None,:] / (self.V_P_F[...,1:] - self.V_NP_F[...,1:] + 1e-30),
            np.inf)
        self.psi_star_F = np.maximum(self.psi_star_F, 1)

        # ── psi^{o*}_{is}: MIXED Case 1 / Case 2 root solve (eq 18'/19') ────
        # Fully vectorized implementation:
        # For each (i, s) the LHS is sum_n {...} where each term depends only
        # on a single scalar psi = psi_o[i, s] and per-(n,i,s) constants.
        # We broadcast psi to shape (N, N, S-1) using psi_b[n, i, s] = psi[i, s].
        d = p.d; k = p.k[1]
        self.psi_o_star = np.full((p.N, p.S), 1.0)

        # Precompute per-(n,i,s) constants (slice to sector >= 1)
        c2_b    = self.case2[..., 1:]                        # (N, N, S-1)
        psi_O_b = self.psi_star_O[..., 1:]
        psi_F_b = self.psi_star_F[..., 1:]
        a_npf_b = self.a_NPF_NPO[..., 1:]
        a_pf_npo_b = self.a_PF_NPO[..., 1:]
        psi_bar_b  = self.psi_bar_NPO_PF[..., 1:]
        a_pf_po_b  = self.a_PF_PO[..., 1:]
        V_NP_b   = self.V_NP[..., 1:]
        V_NP_F_b = self.V_NP_F[..., 1:]
        V_P_b    = self.V_P[..., 1:]
        V_P_F_b  = self.V_P_F[..., 1:]
        w_a_b    = (self.w[:, None, None] * self.a[..., 1:])      # (N, N, S-1)
        w_n_fe_h_b = np.einsum('n,n,s->ns', self.w, p.r_hjort,
                                p.fe[1:])[:, None, :]              # (N, 1, S-1)

        # epsilon helpers as fully vectorized arrays of psi
        def _eps_NPO_NPF_v(psi_b):
            # eps = max(1, a_npf/psi); a_npf=inf in Case1 -> use 1
            out = np.ones_like(a_npf_b)
            mask = np.isfinite(a_npf_b)
            out = np.where(mask & (psi_b > 0),
                            np.maximum(1.0, a_npf_b / np.where(psi_b > 0, psi_b, 1.0)),
                            1.0)
            return out

        def _eps_NPO_PF_v(psi_b):
            # eps = max(1, (a_pf_npo - psi_bar)/(psi - psi_bar))
            denom = psi_b - psi_bar_b
            num = a_pf_npo_b - psi_bar_b
            ok = np.isfinite(a_pf_npo_b) & np.isfinite(psi_bar_b) & (denom > 0)
            ratio = np.where(ok, num / np.where(denom > 0, denom, 1.0), 1.0)
            return np.where(ok, np.maximum(1.0, ratio), 1.0)

        def _eps_PO_PF_v(psi_b):
            ok = np.isfinite(a_pf_po_b)
            out = np.where(ok & (psi_b > 0),
                            np.maximum(1.0, a_pf_po_b / np.where(psi_b > 0, psi_b, 1.0)),
                            1.0)
            return out

        def _ValmidPs_v(psi_b):
            e = _eps_NPO_PF_v(psi_b)
            return (psi_b * V_NP_b * (1 - e**(-d))
                    + (psi_b * V_P_F_b - w_n_fe_h_b) * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def _ValhighPs_v(psi_b):
            e = _eps_PO_PF_v(psi_b)
            return ((psi_b * V_P_b - w_n_fe_h_b) * (1 - e**(-d))
                    + (psi_b * V_P_F_b - w_n_fe_h_b) * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def _ValnoorigPs_v(psi_b):
            e = _eps_NPO_NPF_v(psi_b)
            return (psi_b * V_NP_b * (1 - e**(-d))
                    + psi_b * V_NP_F_b * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def lhs_minus_rhs_vec(psi_arr):
            """
            Fully vectorized LHS - RHS.
            psi_arr: shape (N, S-1) — psi candidate per (i, s).
            Returns: shape (N, S-1) of LHS - RHS.
            """
            # Broadcast psi to (N, N, S-1): psi_b[n, i, s] = psi_arr[i, s]
            psi_b = psi_arr[None, :, :]                          # (1, N, S-1)
            # Case 1 contribution: 1[psi >= psi_O*] * (psi (V_P - V_NP) - w_n_fe_h_s)
            case1_active = (~c2_b) & (psi_b >= psi_O_b)
            case1_term = case1_active * (psi_b * (V_P_b - V_NP_b) - w_n_fe_h_b)

            # Case 2 contributions:
            c2_mid_active  = c2_b & (psi_b >= psi_F_b) & (psi_b < psi_O_b)
            c2_high_active = c2_b & (psi_b >= psi_O_b)
            c2_noorig_active = c2_b & (psi_b >= psi_F_b)

            # Compute Val* using vectorized helpers (safe: returns finite even
            # in branches where the indicator is 0; we mask after).
            # We need to be careful: psi_b may equal 0 etc.
            psi_b_safe = np.where(psi_b > 0, psi_b, 1e-30)
            psi_b3 = np.broadcast_to(psi_b_safe, (p.N, p.N, p.S-1))

            mid_v   = _ValmidPs_v(psi_b3)
            high_v  = _ValhighPs_v(psi_b3)
            noorig_v= _ValnoorigPs_v(psi_b3)

            mid_term   = c2_mid_active * mid_v
            high_term  = c2_high_active * high_v
            noorig_term= c2_noorig_active * noorig_v

            terms = case1_term + mid_term + high_term - noorig_term
            terms = np.where(np.isfinite(terms), terms, 0.0)
            lhs = terms.sum(axis=0)                              # (N, S-1)

            rhs = (self.w * p.r_hjort)[:, None] * p.fo[None, 1:]  # (N, S-1)
            return lhs - rhs

        # Step 1: check whether psi^{o*}=1 (eq 18'): LHS at psi=1 >= RHS
        eq1 = lhs_minus_rhs_vec(np.ones((p.N, p.S-1)))
        need_solve = (eq1 < 0)

        # Step 2: where eq1 < 0, solve LHS - RHS = 0 numerically (eq 19')
        if need_solve.any():
            x0 = np.maximum(np.min(self.psi_star_O[...,1:], axis=0), 1.0)
            x0 = np.where(np.isfinite(x0), x0, 1.0).ravel()

            def func_to_solve(psi_flat):
                psi_arr = psi_flat.reshape(p.N, p.S-1).copy()
                psi_arr = np.where(need_solve, psi_arr, 1.0)
                psi_arr = np.maximum(psi_arr, 1.0)  # keep in domain
                res = lhs_minus_rhs_vec(psi_arr)
                res = np.where(need_solve, res, 0.0)
                return res.ravel()
            try:
                sol = root(func_to_solve, x0=x0, tol=1e-10)
                psi_sol = sol.x.reshape(p.N, p.S-1)
                psi_sol = np.where(need_solve, np.maximum(psi_sol, 1.0), 1.0)
            except Exception:
                psi_sol = np.where(need_solve, x0.reshape(p.N, p.S-1), 1.0)
            self.psi_o_star[:,1:] = psi_sol
        # else: psi^{o*} = 1 everywhere (already initialised)

        # ── effective patenting thresholds (eq 20) ──────────────────────────
        self.psi_m_star_O = np.maximum(self.psi_star_O, self.psi_o_star[None,:,:])
        self.psi_m_star_F = np.maximum(self.psi_star_F, self.psi_o_star[None,:,:])
        self.psi_m_star   = self.psi_m_star_O   # alias

    # ── compute_mass_innovations  (gamma quality shares, eqs 21'-26') ───────

    def compute_mass_innovations(self, p):
        """
        gamma^{P,O}, gamma^{P,F}, gamma^{NP,O}, gamma^{NP,F}.

        CORRECTED per FDI_algorithm_remaining_errors_report.tex (Edit 1):
          - All a-thresholds enter as a^{-d}  (NOT a^d)
          - psi exponents in high regions are d-k+1  (NOT 1-k-d)
          - gamma^{P,O} subtracts max(a^{PF,PO}, psi^{m*,O})^{1-k}
          - gamma^{P,F} has explicit 4-branch indicator structure on a^{PF_NPO}
          - The four shares partition the Pareto mean: 
            gamma^{P,O} + gamma^{P,F} + gamma^{NP,O} + gamma^{NP,F} = k/(k-1)
        Beta shape is (d+1, k-d-1), prefactor (h_n f^e_s / a_nis)^d * psi_bar^{1-k}.
        """
        k = p.k[1]; d = p.d
        psi_mO = self.psi_m_star_O[...,1:]
        psi_mF = self.psi_m_star_F[...,1:]
        a_po   = self.a_PF_PO[...,1:]
        a_npo  = self.a_PF_NPO[...,1:]
        a_npf  = self.a_NPF_NPO[...,1:]
        psi_bar= self.psi_bar_NPO_PF[...,1:]
        a_nis  = self.a[...,1:]
        c2     = self.case2[...,1:]
        self.c2 = c2

        # h_n * fe_s (no w_n — w_n absorbed into psi_bar definition)
        h_fe = np.einsum('n,s->ns', p.r_hjort, p.fe[1:])[:,None,:]
        a_safe = np.where(a_nis > 0, a_nis, 1.0)

        # Safe versions for use in branches where the threshold may be inf
        max_apo_mO = np.maximum(a_po, psi_mO)

        # ──────────────────────────────────────────────────────────────────
        # gamma^{P,O}  (eq 21' of corrected algorithm)
        # gamma^{P,O} = k/(k-1) [psi^{m*,O}^{1-k} - max(a^{PF,PO}, psi^{m*,O})^{1-k}]
        #             - k a^{-d}/(k-d-1) [psi^{m*,O}^{d-k+1} - max(...)^{d-k+1}]
        # In Case 1 (no FDI), gamma^{P,O} = k/(k-1) psi^{m*,O}^{1-k}
        # (the second bracket vanishes because a_po -> inf -> max = a_po,
        # and the resulting expression collapses).
        # ──────────────────────────────────────────────────────────────────
        g_PO_c2 = (
            k/(k-1) * (psi_mO**(1-k) - max_apo_mO**(1-k))
            - k * a_po**(-d) / (k-d-1) * (
                psi_mO**(d-k+1) - max_apo_mO**(d-k+1)))
        g_PO_c1 = k/(k-1) * psi_mO**(1-k)
        self.gamma_PO = np.zeros((p.N,p.N,p.S))
        self.gamma_PO[...,1:] = np.where(c2, g_PO_c2, g_PO_c1)

        # ──────────────────────────────────────────────────────────────────
        # gamma^{P,F}  (eq 22') — zero in Case 1
        #   Term A:  k a^{-d}/(k-d-1) [psi^{m*,O}^{d-k+1} - max(...)^{d-k+1}]
        #            + k/(k-1) max(...)^{1-k}
        #   Plus three indicator terms over a^{PF_NPO}:
        #   Branch 1 [a^{PF_NPO} <= psi^{m*,F}]:  k/(k-1) [psi^{m*,F}^{1-k} - psi^{m*,O}^{1-k}]
        #   Branch 2 [psi^{m*,F} < a^{PF_NPO} < psi^{m*,O}]:
        #       k(h_n fe_s/a)^d psi_bar^{1-k}
        #         * B(1 - psi_bar/psi^{m*,F}, 1 - psi_bar/a^{PF_NPO}; d+1, k-d-1)
        #       + k/(k-1) [a^{PF_NPO}^{1-k} - psi^{m*,O}^{1-k}]
        #   Branch 3 [a^{PF_NPO} >= psi^{m*,O}]:
        #       k(h_n fe_s/a)^d psi_bar^{1-k}
        #         * B(1 - psi_bar/psi^{m*,F}, 1 - psi_bar/psi^{m*,O}; d+1, k-d-1)
        # ──────────────────────────────────────────────────────────────────
        g_PF_base = np.where(c2,
            k * a_po**(-d) / (k-d-1) * (
                psi_mO**(d-k+1) - max_apo_mO**(d-k+1))
            + k/(k-1) * max_apo_mO**(1-k),
            0.0)

        # Branch 1: a^{PF_NPO} <= psi^{m*,F}
        ind_PF_b1 = c2 & (a_npo <= psi_mF)
        g_PF_b1 = np.where(ind_PF_b1,
            k/(k-1) * (psi_mF**(1-k) - psi_mO**(1-k)),
            0.0)

        # Branch 2: psi^{m*,F} < a^{PF_NPO} < psi^{m*,O}
        ind_PF_b2 = c2 & (a_npo > psi_mF) & (a_npo < psi_mO)
        psi_bar_safe = np.where((psi_bar > 0) & np.isfinite(psi_bar), psi_bar, 1.0)
        t1_PFb2 = np.where(ind_PF_b2, 1 - psi_bar_safe/psi_mF, 0.0)
        t2_PFb2 = np.where(ind_PF_b2, 1 - psi_bar_safe/np.where(a_npo>0,a_npo,1.0), 0.0)
        B_PF_b2 = np.where(ind_PF_b2,
                           _betainc_vec(d+1, k-d-1, t1_PFb2, t2_PFb2), 0.0)
        pre_PF = np.where(c2 & (a_nis > 0),
                          k * (h_fe / a_safe)**d * psi_bar_safe**(1-k),
                          0.0)
        g_PF_b2 = np.where(ind_PF_b2,
            pre_PF * B_PF_b2
            + k/(k-1) * (a_npo**(1-k) - psi_mO**(1-k)),
            0.0)

        # Branch 3: a^{PF_NPO} >= psi^{m*,O}
        ind_PF_b3 = c2 & (a_npo >= psi_mO)
        t1_PFb3 = np.where(ind_PF_b3, 1 - psi_bar_safe/psi_mF, 0.0)
        t2_PFb3 = np.where(ind_PF_b3, 1 - psi_bar_safe/psi_mO, 0.0)
        B_PF_b3 = np.where(ind_PF_b3,
                           _betainc_vec(d+1, k-d-1, t1_PFb3, t2_PFb3), 0.0)
        g_PF_b3 = np.where(ind_PF_b3, pre_PF * B_PF_b3, 0.0)

        self.gamma_PF = np.zeros((p.N,p.N,p.S))
        self.gamma_PF[...,1:] = g_PF_base + g_PF_b1 + g_PF_b2 + g_PF_b3

        # ──────────────────────────────────────────────────────────────────
        # gamma^{NP,O}  (eq 24')
        #   Term 1 [a^{NPF_NPO} > 1]:
        #       k/(k-1) [1 - min(a^{NPF_NPO}, psi^{m*,F})^{1-k}]
        #       - k a^{-d}/(k-d-1) [1 - min(...)^{d-k+1}]
        #   Term 2 [a^{PF_NPO} > psi^{m*,F}]:
        #       k/(k-1) [psi^{m*,F}^{1-k} - min(a^{PF_NPO}, psi^{m*,O})^{1-k}]
        #       - k (h_n fe_s/a)^d psi_bar^{1-k}
        #         * B(1 - psi_bar/psi^{m*,F}, 1 - psi_bar/min(a^{PF_NPO}, psi^{m*,O}); d+1, k-d-1)
        # In Case 1: gamma^{NP,O} = k/(k-1) [1 - psi^{m*,O}^{1-k}]
        # ──────────────────────────────────────────────────────────────────
        ind_NPO_t1 = c2 & (a_npf > 1)
        min_anpf_mF = np.minimum(a_npf, psi_mF)
        g_NPO_t1 = np.where(ind_NPO_t1,
            k/(k-1) * (1 - min_anpf_mF**(1-k))
            - k * a_npf**(-d) / (k-d-1) * (1 - min_anpf_mF**(d-k+1)),
            0.0)

        ind_NPO_t2 = c2 & (a_npo > psi_mF)
        min_apo_mO = np.minimum(a_npo, psi_mO)
        t1_NPO = np.where(ind_NPO_t2, 1 - psi_bar_safe/psi_mF, 0.0)
        t2_NPO = np.where(ind_NPO_t2, 1 - psi_bar_safe/min_apo_mO, 0.0)
        B_NPO = np.where(ind_NPO_t2, _betainc_vec(d+1, k-d-1, t1_NPO, t2_NPO), 0.0)
        g_NPO_t2 = np.where(ind_NPO_t2,
            k/(k-1) * (psi_mF**(1-k) - min_apo_mO**(1-k))
            - pre_PF * B_NPO,
            0.0)

        g_NPO_c2 = g_NPO_t1 + g_NPO_t2
        g_NPO_c1 = k/(k-1) * (1 - psi_mO**(1-k))
        self.gamma_NPO = np.zeros((p.N,p.N,p.S))
        self.gamma_NPO[...,1:] = np.where(c2, g_NPO_c2, g_NPO_c1)

        # ──────────────────────────────────────────────────────────────────
        # gamma^{NP,F}  (eq 26') — zero in Case 1
        # Branch 1 [a^{NPF_NPO} < 1]: k/(k-1) [1 - psi^{m*,F}^{1-k}]
        # Branch 2 [1 <= a^{NPF_NPO} < psi^{m*,F}]:
        #     k a^{-d}/(k-d-1) [1 - a^{NPF_NPO}^{d-k+1}]
        #     + k/(k-1) [a^{NPF_NPO}^{1-k} - psi^{m*,F}^{1-k}]
        # Branch 3 [a^{NPF_NPO} >= psi^{m*,F}]:
        #     k a^{-d}/(k-d-1) [1 - psi^{m*,F}^{d-k+1}]
        # ──────────────────────────────────────────────────────────────────
        g_NPF_b1 = np.where(c2 & (a_npf < 1),
                            k/(k-1) * (1 - psi_mF**(1-k)),
                            0.0)
        g_NPF_b2 = np.where(c2 & (a_npf >= 1) & (a_npf < psi_mF),
                            k * a_npf**(-d) / (k-d-1) * (1 - a_npf**(d-k+1))
                            + k/(k-1) * (a_npf**(1-k) - psi_mF**(1-k)),
                            0.0)
        g_NPF_b3 = np.where(c2 & (a_npf >= psi_mF),
                            k * a_npf**(-d) / (k-d-1) * (1 - psi_mF**(d-k+1)),
                            0.0)
        self.gamma_NPF = np.zeros((p.N,p.N,p.S))
        self.gamma_NPF[...,1:] = g_NPF_b1 + g_NPF_b2 + g_NPF_b3

        # Sanitise: gamma's must be non-negative and finite
        for _g in ['gamma_PO','gamma_PF','gamma_NPO','gamma_NPF']:
            arr = getattr(self, _g)
            arr[~np.isfinite(arr)] = 0.0
            arr[arr < 0] = 0.0

        # Enforce overall Pareto partition: the four gamma's sum to at most k/(k-1)
        bound_total = k/(k-1)
        g_sum = (self.gamma_PO[...,1:] + self.gamma_PF[...,1:]
                 + self.gamma_NPO[...,1:] + self.gamma_NPF[...,1:])
        over = g_sum > bound_total
        if over.any():
            scale = np.where(over & (g_sum > 0), bound_total / (g_sum + 1e-30), 1.0)
            self.gamma_PO[...,1:]  *= scale
            self.gamma_PF[...,1:]  *= scale
            self.gamma_NPO[...,1:] *= scale
            self.gamma_NPF[...,1:] *= scale

    # ── compute_aggregate_qualities  (eqs 28', 31', 32') ────────────────────

    def compute_aggregate_qualities(self, p):
        """
        Six quality stocks (O & F versions of P_ND, P_D, NP).
        NO extra k/(k-1) factor — gamma's already contain it (eqs 28', 31', 32').
        """
        prefact = p.eta[None,:,1:] * self.l_R[None,:,1:]**(1-p.kappa)
        gs  = self.g_s[1:]; nu = p.nu[1:]; ze = p.zeta[1:]
        de  = p.delta[:,1:]
        A_NE  = gs + nu + ze
        A_PND = gs[None,:] + de + nu + ze
        A_PD  = gs[None,:] + de + ze

        # Export (O) stocks
        self.PSI_M_P_ND_O = np.zeros((p.N,p.N,p.S))
        self.PSI_M_P_ND_O[...,1:] = np.einsum(
            'nis,nis,nis->nis',
            prefact, self.gamma_PO[...,1:], 1/A_PND[:,None,:])

        self.PSI_M_P_D_O = np.zeros((p.N,p.N,p.S))
        self.PSI_M_P_D_O[...,1:] = np.einsum(
            's,nis,nis->nis', nu, self.PSI_M_P_ND_O[...,1:], 1/A_PD[:,None,:])

        self.PSI_M_NP_O = np.zeros((p.N,p.N,p.S))
        num_O = (np.einsum('nis,nis->nis', prefact, self.gamma_NPO[...,1:])
                 + np.einsum('ns,nis->nis', de, self.PSI_M_P_ND_O[...,1:]))
        self.PSI_M_NP_O[...,1:] = np.einsum('nis,s->nis', num_O, 1/A_NE)

        # FDI (F) stocks
        self.PSI_M_P_ND_F = np.zeros((p.N,p.N,p.S))
        self.PSI_M_P_ND_F[...,1:] = np.einsum(
            'nis,nis,nis->nis', prefact, self.gamma_PF[...,1:], 1/A_PND[:,None,:])

        self.PSI_M_P_D_F = np.zeros((p.N,p.N,p.S))
        self.PSI_M_P_D_F[...,1:] = np.einsum(
            's,nis,nis->nis', nu, self.PSI_M_P_ND_F[...,1:], 1/A_PD[:,None,:])

        self.PSI_M_NP_F = np.zeros((p.N,p.N,p.S))
        num_F = (np.einsum('nis,nis->nis', prefact, self.gamma_NPF[...,1:])
                 + np.einsum('ns,nis->nis', de, self.PSI_M_P_ND_F[...,1:]))
        self.PSI_M_NP_F[...,1:] = np.einsum('nis,s->nis', num_F, 1/A_NE)

        # Aggregates (eqs 33-34)
        self.PSI_M_O = self.PSI_M_P_ND_O + self.PSI_M_P_D_O + self.PSI_M_NP_O
        self.PSI_M_F = self.PSI_M_P_ND_F + self.PSI_M_P_D_F + self.PSI_M_NP_F
        self.PSI_M   = self.PSI_M_O + self.PSI_M_F

        # PSI_CD (eq 35)
        self.PSI_CD = np.ones((p.N,p.S))
        self.PSI_CD[:,1:] = 1 - self.PSI_M[...,1:].sum(axis=1)
        # Clamp PSI_CD to [0,1]: if PSI_M sum > 1 due to numerical overflow,
        # scale PSI_M stocks proportionally
        overflow_mask = self.PSI_CD[:,1:] < 0
        if overflow_mask.any():
            PSI_M_total = self.PSI_M[...,1:].sum(axis=1)
            safe_total = np.where(PSI_M_total > 1, PSI_M_total, 1.0)
            scale = np.where(overflow_mask, 1.0 / safe_total, 1.0)
            for attr in ['PSI_M_O','PSI_M_F','PSI_M_P_ND_O','PSI_M_P_D_O',
                         'PSI_M_NP_O','PSI_M_P_ND_F','PSI_M_P_D_F','PSI_M_NP_F',
                         'PSI_M']:
                getattr(self, attr)[...,1:] *= scale[:,None,:]
            self.PSI_CD[:,1:] = np.maximum(1 - self.PSI_M[...,1:].sum(axis=1), 0.0)

        # Backward-compat aliases (used by moments code)
        self.PSI_ME   = self.PSI_M_O
        self.PSI_MPND = self.PSI_M_P_ND_O
        self.PSI_MPD  = self.PSI_M_P_D_O
        self.PSI_MNP  = self.PSI_M_NP_O

    # ── compute_sectoral_prices  (eqs 36-39) ────────────────────────────────

    def compute_sectoral_prices(self, p):
        """D_ns includes export, FDI (with phi_nn), and CD masses."""
        power = p.sigma - 1
        phi_nn = np.einsum('nns->ns', self.phi)

        A_exp = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,1:] * (
            self.PSI_M_O[...,1:] * self.phi[...,1:]**power[None,None,1:]
        ).sum(axis=1)

        A_fdi = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,1:] * (
            self.PSI_M_F[...,1:] * phi_nn[:,None,1:]**power[None,None,1:]
        ).sum(axis=1)

        B_cd = self.PSI_CD[:,1:] * (
            self.phi[...,1:]**p.theta[None,None,1:]
        ).sum(axis=1)**((power/p.theta)[None,1:])

        D = A_exp + A_fdi + B_cd
        safe_D = np.where(D > 0, D, 1.0)

        self.P_M = np.zeros((p.N,p.S))
        self.P_M[:,1:] = np.where(
            (D > 0) & (A_exp > 0),
            (A_exp/safe_D)**(1/(1-p.sigma[None,1:])), 0.0)

        self.P_M_F = np.zeros((p.N,p.S))
        self.P_M_F[:,1:] = np.where(
            (D > 0) & (A_fdi > 0),
            (A_fdi/safe_D)**(1/(1-p.sigma[None,1:])), 0.0)

        self.P_CD = np.ones((p.N,p.S))
        self.P_CD[:,1:] = np.where(
            (D > 0) & (B_cd > 0),
            (B_cd/safe_D)**(1/(1-p.sigma[None,1:])), 0.0)

    # ── compute_labor_allocations  (eqs 40', Lo, 42) ────────────────────────

    def compute_labor_allocations(self, p):
        """
        L^o, L^e (mixed Case 1/Case 2 — eq 40'), L^F (eq 42 with corrected
        beta-shape (d+2, k-d-1)). Aggregates to L^P.
        """
        k = p.k[1]; d = p.d
        psi_mO = self.psi_m_star_O[...,1:]
        psi_mF = self.psi_m_star_F[...,1:]
        a_po   = self.a_PF_PO[...,1:]
        a_npo  = self.a_PF_NPO[...,1:]
        a_npf  = self.a_NPF_NPO[...,1:]
        psi_bar= self.psi_bar_NPO_PF[...,1:]
        a_nis  = self.a[...,1:]
        c2     = self.case2[...,1:]
        a_safe = np.where(a_nis > 0, a_nis, 1.0)
        w_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort, p.fe[1:])[:,None,:]

        # ── L^o_is ──────────────────────────────────────────────────────────
        self.l_Ao = np.zeros((p.N,p.S))
        self.l_Ao[...,1:] = np.einsum(
            'i,s,is,is,is->is',
            p.r_hjort, p.fo[1:], p.eta[...,1:],
            self.l_R[...,1:]**(1-p.kappa),
            self.psi_o_star[...,1:]**(-k))

        # ── L^e_{ins}  (eq 40' — MIXED Case 1 / Case 2) ─────────────────────
        # Case 1: psi^{m*,O}^{-k}
        Le_c1 = psi_mO**(-k)

        # Case 2 sub-branches:
        # (i)   a^{PF,NPO} <= psi^{m*,F}            -> psi^{m*,F}^{-k}
        # (ii)  psi^{m*,F} <= a^{PF,NPO} <= psi^{m*,O} -> a^{PF,NPO}^{-k}
        # (iii) a^{PF,NPO} > psi^{m*,O}             -> psi^{m*,O}^{-k}
        # (iv)  psi^{m*,F} < a^{PF,NPO}             -> k*(h_n fe_s/a)^d * psi_bar^{-k}
        #                                              * B(...; d+1, k-d)
        Le_c2_i   = np.where(a_npo <= psi_mF, psi_mF**(-k), 0.0)
        Le_c2_ii  = np.where((a_npo > psi_mF) & (a_npo <= psi_mO),
                             a_npo**(-k), 0.0)
        Le_c2_iii = np.where(a_npo > psi_mO, psi_mO**(-k), 0.0)

        ind_iv = c2 & (psi_mF < a_npo)
        t1_iv = np.where(ind_iv, 1 - psi_bar/psi_mF, 0.0)
        t2_iv = np.where(ind_iv,
                         1 - psi_bar/np.minimum(a_npo, psi_mO), 0.0)
        # CORRECTED beta-shape (d+1, k-d)
        B_iv = np.where(ind_iv, _betainc_vec(d+1, k-d, t1_iv, t2_iv), 0.0)
        # FIX 1.5: per algorithm eq (38) and RomerEK eq (88), the inner Pareto
        # ratio is (h_n f^e_s / a_{nis})^d with NO w_n inside the d-th power.
        # The wage is already embodied in psi_bar and the thresholds; (h f^e
        # / a) is dimensionally a ratio of labor quantities and is wage-free.
        # Previously: (w_fe_h / a)^d  =  (w_n h_n f^e_s / a)^d, which inserted
        # an extra w_n^d factor into the patent-application probability.
        h_fe_only = np.einsum('n,s->ns', p.r_hjort, p.fe[1:])[:,None,:]
        pre_iv = np.where(ind_iv & (a_nis > 0),
                          k * (h_fe_only/a_safe)**d
                          / np.where(np.isfinite(psi_bar) & (psi_bar > 0),
                                     psi_bar, 1.0)**k,
                          0.0)
        Le_c2_iv = pre_iv * B_iv

        Le_c2 = Le_c2_i + Le_c2_ii + Le_c2_iii + Le_c2_iv
        Le_total = np.where(c2, Le_c2, Le_c1)
        Le_total = np.where(np.isfinite(Le_total), Le_total, 0.0)
        Le_total = np.maximum(Le_total, 0.0)

        # Axis convention: l_Ae[i, n, s]  (origin i, destination n)
        # → labor paid by destination n on origin i's innovations.
        # Sum into (i, n, s) using einsum
        self.l_Ae = np.zeros((p.N,p.N,p.S))
        self.l_Ae[...,1:] = np.einsum(
            'n,s,is,is,nis->ins',
            p.r_hjort, p.fe[1:], p.eta[...,1:],
            self.l_R[...,1:]**(1-p.kappa),
            Le_total)

        # ────────────────────────────────────────────────────────────────────
        # L^F_{ins}  (eq LF/LambdaF of corrected algorithm — Edit 2 of report)
        #
        # CORRECTED:
        #   - Outside prefactor is d * a_nis / (d+1)  (NOT a_nis^d / (d+1))
        #   - All a-thresholds enter as a^{-(d+1)}  (NOT a^{d-1})
        #   - psi exponents in low/high regions are d-k+1  (NOT -dk+1)
        #   - Middle region (a^{PF_NPO} <= psi^{m*,F}) simplifies to
        #     (psi^{m*,F})^{-k} - (psi^{m*,O})^{-k}  (the spurious psi_mF^{-k}
        #     factor is gone).
        #   - Beta shape is (d+2, k-d-1).
        #
        # Lambda^F has 8 terms (matches algorithm lines 299-305):
        #   T1 [a^{NPF_NPO} < 1]: 1 - psi^{m*,F}^{-k}
        #   T2 [1 <= a^{NPF_NPO} < psi^{m*,F}]:
        #         k a^{-(d+1)}/(k-d-1) [1 - a^{NPF_NPO}^{d-k+1}]
        #         + a^{NPF_NPO}^{-k} - psi^{m*,F}^{-k}
        #   T3 [a^{NPF_NPO} >= psi^{m*,F}]:
        #         k a^{-(d+1)}/(k-d-1) [1 - psi^{m*,F}^{d-k+1}]
        #   T4 [a^{PF_NPO} <= psi^{m*,F}]:
        #         psi^{m*,F}^{-k} - psi^{m*,O}^{-k}
        #   T5 [psi^{m*,F} < a^{PF_NPO} < psi^{m*,O}]:
        #         k (h_n fe_s / a)^{d+1} psi_bar^{-k}
        #           * B(1 - psi_bar/psi^{m*,F}, 1 - psi_bar/a^{PF_NPO}; d+2, k-d-1)
        #         + a^{PF_NPO}^{-k} - psi^{m*,O}^{-k}
        #   T6 [a^{PF_NPO} >= psi^{m*,O}]:
        #         k (h_n fe_s / a)^{d+1} psi_bar^{-k}
        #           * B(1 - psi_bar/psi^{m*,F}, 1 - psi_bar/psi^{m*,O}; d+2, k-d-1)
        #   T7: max(a^{PF,PO}, psi^{m*,O})^{-k}
        #   T8 [a^{PF,PO} > psi^{m*,O}]:
        #         k a^{PF,PO,-(d+1)}/(k-d-1) [psi^{m*,O}^{d-k+1} - a^{PF,PO,d-k+1}]
        # ────────────────────────────────────────────────────────────────────
        psi_bar_safe = np.where((psi_bar > 0) & np.isfinite(psi_bar), psi_bar, 1.0)
        h_fe_only = np.einsum('n,s->ns', p.r_hjort, p.fe[1:])[:,None,:]

        T1 = np.where(c2 & (a_npf < 1), 1 - psi_mF**(-k), 0.0)
        T2 = np.where(c2 & (a_npf >= 1) & (a_npf < psi_mF),
                      k * a_npf**(-(d+1)) / (k-d-1) * (1 - a_npf**(d-k+1))
                      + a_npf**(-k) - psi_mF**(-k), 0.0)
        T3 = np.where(c2 & (a_npf >= psi_mF),
                      k * a_npf**(-(d+1)) / (k-d-1) * (1 - psi_mF**(d-k+1)), 0.0)
        T4 = np.where(c2 & (a_npo <= psi_mF),
                      psi_mF**(-k) - psi_mO**(-k), 0.0)

        # T5: middle region (psi_mF < a_npo < psi_mO)
        ind_t5 = c2 & (a_npo > psi_mF) & (a_npo < psi_mO)
        t1_lF5 = np.where(ind_t5, 1 - psi_bar_safe/psi_mF, 0.0)
        t2_lF5 = np.where(ind_t5, 1 - psi_bar_safe/np.where(a_npo>0,a_npo,1.0), 0.0)
        B_lF5 = np.where(ind_t5, _betainc_vec(d+2, k-d-1, t1_lF5, t2_lF5), 0.0)
        pre_lF = np.where(c2 & (a_nis > 0),
                          k * (h_fe_only / a_safe)**(d+1) * psi_bar_safe**(-k),
                          0.0)
        T5 = np.where(ind_t5,
                      pre_lF * B_lF5 + a_npo**(-k) - psi_mO**(-k),
                      0.0)

        # T6: high region (a_npo >= psi_mO)
        ind_t6 = c2 & (a_npo >= psi_mO)
        t1_lF6 = np.where(ind_t6, 1 - psi_bar_safe/psi_mF, 0.0)
        t2_lF6 = np.where(ind_t6, 1 - psi_bar_safe/psi_mO, 0.0)
        B_lF6 = np.where(ind_t6, _betainc_vec(d+2, k-d-1, t1_lF6, t2_lF6), 0.0)
        T6 = np.where(ind_t6, pre_lF * B_lF6, 0.0)

        T7 = np.where(c2, np.maximum(a_po, psi_mO)**(-k), 0.0)
        T8 = np.where(c2 & (a_po > psi_mO),
                      k * a_po**(-(d+1)) / (k-d-1)
                      * (psi_mO**(d-k+1) - a_po**(d-k+1)), 0.0)

        Lambda_F = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8
        Lambda_F = np.where(c2, Lambda_F, 0.0)
        Lambda_F = np.where(np.isfinite(Lambda_F), Lambda_F, 0.0)
        Lambda_F = np.maximum(Lambda_F, 0.0)

        # Axis convention: l_F[i, n, s]  (origin i, destination n)
        # CORRECTED: outside prefactor is d*a/(d+1)  (NOT a^d/(d+1))
        self.l_F = np.zeros((p.N,p.N,p.S))
        self.l_F[...,1:] = np.einsum(
            'is,is,nis,nis->ins',
            p.eta[:,1:], self.l_R[...,1:]**(1-p.kappa),
            d * self.a[...,1:] / (d+1), Lambda_F)
        self.l_F = np.maximum(self.l_F, 0.0)

        # ── L^P_i  (l_Aa dropped — not in the corrected algorithm) ──────────
        # FIX 1.1: l_Ae and l_F are indexed [origin i, destination n, sector s].
        # L^P_i needs labor USED IN country i = sum over origins n at fixed
        # destination=i.  That is sum(axis=0), NOT sum(axis=1).  The previous
        # code summed destinations at fixed origin, which gave each country's
        # firms' labor abroad rather than each country's labor stock.
        self.l_P = p.labor - (
            self.l_Ao + self.l_R
            + self.l_Ae.sum(axis=0)
            + self.l_F.sum(axis=0)
        ).sum(axis=1)

    # ── compute_trade_flows_and_shares  (eqs 43-47) ─────────────────────────

    def compute_trade_flows_and_shares(self, p, assign=True):
        """X^{M,O}, X^{M,F}, X^{CD}, X (with FDI added on diagonal)."""
        # X^{M,O}
        temp_exp = (self.PSI_M_O[...,1:]
                    * self.phi[...,1:]**(p.sigma-1)[None,None,1:]
                    ).sum(axis=1)
        X_M = np.zeros((p.N,p.N,p.S))
        safe_exp = np.where(temp_exp > 0, temp_exp, 1.0)
        X_M[...,1:] = np.where(
            temp_exp[:,None,:] > 0,
            np.einsum(
                'nis,nis,ns,ns,s,n->nis',
                self.phi[...,1:]**(p.sigma-1)[None,None,1:],
                self.PSI_M_O[...,1:],
                1/safe_exp,
                self.P_M[:,1:]**(1-p.sigma[None,1:]),
                p.beta[1:], self.Z),
            0.0)

        # X^{M,F} (affiliates use phi_nn)
        phi_nn = np.einsum('nns->ns', self.phi)
        temp_fdi = (self.PSI_M_F[...,1:]
                    * phi_nn[:,None,1:]**(p.sigma-1)[None,None,1:]
                    ).sum(axis=1)
        X_M_F = np.zeros((p.N,p.N,p.S))
        safe_fdi = np.where(temp_fdi > 0, temp_fdi, 1.0)
        X_M_F[...,1:] = np.where(
            temp_fdi[:,None,:] > 0,
            np.einsum(
                'nis,nis,nis,s,n->nis',
                self.PSI_M_F[...,1:],
                phi_nn[:,None,1:]**(p.sigma-1)[None,None,1:],
                1/safe_fdi[:,None,:],
                p.beta[1:], self.Z)
            * self.P_M_F[:,None,1:]**(1-p.sigma[None,1:]),
            0.0)

        # X^{CD}
        safe_P_CD = np.where(self.P_CD > 0, self.P_CD, 1.0)
        X_CD = np.where(
            self.P_CD[:,None,:] > 0,
            np.einsum(
                'nis,ns,nis,s,n->nis',
                self.phi**p.theta[None,None,:],
                1/(self.phi**p.theta[None,None,:]).sum(axis=1),
                safe_P_CD[:,None,:]**(1-p.sigma[None,:]),
                p.beta, self.Z),
            0.0)

        # Total X: diagonal includes sum_j X^{M,F}_{njs} (eq 47)
        X = X_M + X_CD
        np.einsum('nns->ns', X)[...] += X_M_F.sum(axis=1)

        if assign:
            self.X_M = X_M; self.X_M_F = X_M_F; self.X_CD = X_CD; self.X = X
        else:
            return X_M, X_M_F, X_CD, X

    # ── compute_solver_quantities ───────────────────────────────────────────

    def compute_solver_quantities(self, p):
        self.compute_growth(p)
        self.compute_entry_costs(p)
        self.compute_V(p)
        self.compute_case_indicator(p)
        self.compute_auxiliary_thresholds(p)
        self.compute_patenting_thresholds(p)
        self.compute_mass_innovations(p)
        self.compute_aggregate_qualities(p)
        self.compute_sectoral_prices(p)
        self.compute_labor_allocations(p)
        self.compute_trade_flows_and_shares(p)

    # ── update equations (RHS of solver loop) ───────────────────────────────

    def compute_price_indices(self, p):
        """Aggregate price index incorporating FDI mass."""
        power = p.sigma-1
        phi_nn = np.einsum('nns->ns', self.phi)
        A_exp = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,:] * (
            self.PSI_M_O * self.phi**power[None,None,:]).sum(axis=1)
        A_fdi = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None,:] * (
            self.PSI_M_F * phi_nn[:,None,:]**power[None,None,:]).sum(axis=1)
        B = self.PSI_CD * (self.phi**p.theta[None,None,:]).sum(axis=1)**(
            (power/p.theta)[None,:])
        temp = gamma((p.theta+1-p.sigma)/p.theta)[None,:] * (A_exp+A_fdi+B)
        one_over = np.divide(1, temp, out=np.full_like(temp, np.inf), where=temp>0)
        return (one_over**(p.beta[None,:]/(p.sigma[None,:]-1))).prod(axis=1)

    def compute_wage(self, p):
        """
        Wage equation (eq 57'):
        w_i L^P_i = sum_s alpha_s [sum_n X_{nis}/(1+b) - sum_n X^{M,O}_{nis}/(sigma(1+b))
                                   - sum_j X^{M,F}_{ijs}/sigma]
        """
        # sum_n X_{nis} / (1+b)
        A1 = (self.X / (1 + p.tariff)).sum(axis=0)              # (i, s)
        # sum_n X^{M,O}_{nis} / (sigma (1+b))
        A2 = (self.X_M / (1 + p.tariff)).sum(axis=0) / p.sigma[None,:]
        # sum_j X^{M,F}_{ijs} / sigma   (affiliates LOCATED in country i)
        # FIX 1.2: X_M_F is indexed [destination n, origin i, sector s].  We
        # need sales LOCATED in destination=i (affiliates produce where they
        # sell), summed over origin axis.  That is axis=1.  The previous
        # axis=0 summed destinations at fixed origin, giving country-i firms'
        # affiliate sales abroad — wrong for production-labor accounting.
        A3 = self.X_M_F.sum(axis=1) / p.sigma[None,:]           # (i, s)
        per_sector = A1 - A2 - A3                                # (i, s)
        return (p.alpha[None,:] * per_sector).sum(axis=1) / self.l_P

    def compute_expenditure(self, p):
        """
        Income/spending (eq Z of FDI_algorithm_fixed.tex, new derivation):

          Z_i = sum_{s,n} X_{nis}/(1+b_{nis})                  # tariff-net sales
              + sum_{s,n} X_{ins} b_{ins}/(1+b_{ins})          # tariff revenue
              - tb_i * sum_{n,i,s} X_{nis}/(1+b_{nis})         # trade balance,
                                                                 # scaled by GLOBAL
                                                                 # tariff-net X^W
              + sum_{s,n} w_i (L^e_{nis} + L^F_{nis})          # i's L receipts
              - sum_{s,n} w_n (L^e_{ins} + L^F_{ins})          # i's L payments

        with tb_i = (TB_i / X^W)^{DATA} = p.deficit_share_world_output.

        NO separate FDI-profit term: trade flows X already enter net of
        tariffs, and affiliate sales are subsumed into X^{M,F} components
        of the bilateral X used above (see compute_trade_flows_and_shares,
        which folds X^{M,F}.sum(axis=1) into the diagonal of X).

        Axis conventions:
          self.X[n, i, s]:  destination n, origin i  -> X_{nis}
          self.l_Ae, self.l_F[i, n, s]: origin i, destination n  -> L^e_{nis}

        Notation in the eq above: X_{nis} has n=destination, i=origin (doc
        convention), so:
          - X_{nis} / (1+b_{nis}) summed over (n,s) = i's tariff-net SALES
            received  (i is origin, n is destination)
          - X_{ins} b_{ins}/(1+b_{ins}) summed over (n,s) = i's tariff
            revenue on its IMPORTS  (i is destination, n is origin)
          - L^e_{nis} with l_Ae[origin, destination, s]: workers in country i
            employed by foreign/domestic firms = sum over origin axis at
            destination=i, paid at local wage w_i.  This is i's RECEIPTS.
          - L^e_{ins}: workers in destination n employed by i's firms = sum
            over destination axis at origin=i, paid at w_n.  This is i's
            PAYMENTS abroad.
        """
        # i's tariff-net SALES received: sum over (n=dest, s) of X[n, i, s]/(1+b[n,i,s])
        A1 = np.einsum('nis,nis->i', self.X, 1/(1+p.tariff))

        # i's tariff revenue on its IMPORTS: sum over (n=origin, s) of
        # X[i, n, s] * b[i, n, s] / (1+b[i, n, s])
        # In numpy axis order ('i' is destination here, 'n' is origin):
        A2 = np.einsum('ins,ins,ins->i', self.X, p.tariff, 1/(1+p.tariff))

        # FIX 1.3: Trade balance is tb_i (per-country share) * GLOBAL X^W,
        # not tb_i * country-i's own tariff-net sales A1[i].  Global X^W is
        # exactly A1.sum() (sum over all (i,s) of country-i's tariff-net sales
        # = sum over all (n,i,s) of X/(1+b)).  Previously: TB = p.deficit_*
        # A1, which scaled by each country's own A1 — inconsistent with the
        # algorithm doc eq (60) which uses world output.
        TB = p.deficit_share_world_output * A1.sum()

        # FIX 1.4: Service flows.
        # Convention: l_Ae[i_origin, n_destination, s], wages paid at the
        # destination's wage rate (workers paid where they work).
        # RECEIPTS by country i = workers IN country i paid by any firm,
        # i.e. destination=i, summed over origins (axis=0 of l_Ae), times w_i.
        B_e = np.einsum('i,jis->i', self.w, self.l_Ae)
        B_F = np.einsum('i,jis->i', self.w, self.l_F)
        # PAYMENTS by country i = country-i firms paying workers in n,
        # i.e. origin=i, summed over destinations (axis=1 of l_Ae), each term
        # weighted by destination's w_n.
        D_e = np.einsum('n,ins->i', self.w, self.l_Ae)
        D_F = np.einsum('n,ins->i', self.w, self.l_F)

        return A1 + A2 - TB + B_e + B_F - D_e - D_F

    def compute_profit(self, p):
        """Export profits: pi^w_{nis} = X^{M,O}_{nis} / (sigma * PSI_M_O * w_i * (1+b))."""
        profit = np.zeros((p.N,p.N,p.S))
        safe_psi = np.where(self.PSI_M_O[...,1:] > 0, self.PSI_M_O[...,1:], 1.0)
        profit[...,1:] = np.where(
            self.PSI_M_O[...,1:] > 0,
            np.einsum('nis,s,i,nis,nis->nis',
                      self.X_M[...,1:], 1/p.sigma[1:], 1/self.w,
                      1/safe_psi, 1/(1+p.tariff[...,1:])),
            0.0)
        return profit

    def compute_pi_F(self, p):
        """
        FDI profits: Pi^{w,F} is a DESTINATION-SECTOR object (not origin-specific).

        Per algorithm_corrected.tex eq (\\ref{eq:Piupd}):
          Primary (when sum_i Psi^{M,F}_{nis} > 0):
              Pi^{w,F}_{ns} = sum_i X^{M,F}_{nis} / (sigma_s * w_n * sum_i Psi^{M,F}_{nis})
          Fallback (sum_i Psi^{M,F}_{nis} = 0; e.g. Case 1, or zero-FDI start):
              Pi^{w,F}_{ns} = X^{M,O}_{nns} / (sigma_s * Psi^{M,O}_{nns} * w_n)
          Then broadcast to (n, i, s):
              Pi^{w,F}_{nis} = Pi^{w,F}_{ns}  for all i.

        The fallback uses the domestic monopolist's per-unit profit as the
        natural extrapolation when there are no affiliates: a hypothetical FDI
        unit would face the same domestic demand and capture the same per-unit
        profit as the domestic non-FDI monopolist.
        """
        pi_F = np.zeros((p.N, p.N, p.S))

        # Sums over origins (axis=1 in (n, i, s))
        psi_F_sum_i = self.PSI_M_F[..., 1:].sum(axis=1)             # (N, S-1)
        X_F_sum_i   = self.X_M_F[..., 1:].sum(axis=1)               # (N, S-1)

        # Primary: aggregate FDI-sales ratio
        mask_primary = psi_F_sum_i > 0
        safe_psi = np.where(mask_primary, psi_F_sum_i, 1.0)
        pi_F_ns_primary = np.einsum(
            'ns,s,n,ns->ns',
            X_F_sum_i, 1/p.sigma[1:], 1/self.w, 1/safe_psi)

        # Fallback: domestic monopolist sales (n == n diagonal of X^{M,O})
        # Note: self.X_M is the export (origin-mode) flow X^{M,O}; the
        # affiliate (FDI) flow is self.X_M_F.
        X_M_O_diag   = np.einsum('nns->ns', self.X_M[..., 1:])      # (N, S-1)
        PSI_M_O_diag = np.einsum('nns->ns', self.PSI_M_O[..., 1:])  # (N, S-1)
        mask_dom = PSI_M_O_diag > 0
        safe_psi_dom = np.where(mask_dom, PSI_M_O_diag, 1.0)
        pi_F_ns_fallback = np.where(
            mask_dom,
            np.einsum('ns,s,ns,n->ns',
                      X_M_O_diag, 1/p.sigma[1:], 1/safe_psi_dom, 1/self.w),
            0.0)

        # Choose primary where it's defined; otherwise fallback
        pi_F_ns = np.where(mask_primary, pi_F_ns_primary, pi_F_ns_fallback)

        # Broadcast to (n, i, s): same value for all origins i
        pi_F[..., 1:] = pi_F_ns[:, None, :].repeat(p.N, axis=1)

        return pi_F

    def compute_labor_research(self, p):
        """
        Eq (49'):  (L^R_is)^kappa / eta_is
                   = (1/w_i) * sum_n [IntVNorigP + IntVorigP^A + B + C + D]
                     - h_i f^o_s (psi^{o*}_{is})^{-k}

        Fully vectorized: broadcasts psi_o[i,s] -> psi_o_b[n,i,s] and computes
        IntVN/A/B/C/D over (N, N, S-1) using masked np.where operations.
        """
        import scipy.special as _sp
        k = p.k[1]; d = p.d
        N, S = p.N, p.S

        # Per-(n, i, s>=1) constants
        c2_full = self.case2[..., 1:]                 # (N, N, S-1)
        # Diagonal n==i is Case 1 by construction (case2 already excludes diag)
        c2 = c2_full
        c1 = ~c2

        a_nis      = self.a[..., 1:]
        psi_bar    = self.psi_bar_NPO_PF[..., 1:]
        V_NPO      = self.V_NP[..., 1:]
        V_NPF      = self.V_NP_F[..., 1:]
        V_PO       = self.V_P[..., 1:]
        V_PF       = self.V_P_F[..., 1:]
        psi_O      = self.psi_star_O[..., 1:]
        psi_F      = self.psi_star_F[..., 1:]
        a_bar      = self.a_NPF_NPO[..., 1:]
        a_hat      = self.a_PF_NPO[..., 1:]
        a_chk      = self.a_PF_PO[..., 1:]

        # In Case 1, V_NPF and V_PF are meaningless; substitute 0 so they
        # never enter any active branch (we only use them where c2 is True).
        V_NPF_safe = np.where(c2, V_NPF, 0.0)
        V_PF_safe  = np.where(c2, V_PF,  0.0)

        # Broadcast psi_o[i, s] to (n, i, s) so we can do element-wise math.
        psi_o = np.broadcast_to(self.psi_o_star[None, :, 1:], (N, N, S-1))
        psi_o = np.maximum(psi_o, 1.0)   # safety

        w_n   = np.broadcast_to(self.w[:, None, None], (N, N, S-1))
        w_n_a = w_n * a_nis                                       # (N,N,S-1)
        w_n_fe_h = np.einsum('n,n,s->ns', self.w, p.r_hjort,
                              p.fe[1:])[:, None, :]                # (N,1,S-1)
        w_n_fe_h_b = np.broadcast_to(w_n_fe_h, (N, N, S-1))

        # =============================================================
        # IntVNorigP  (eq 50')
        # =============================================================
        # Case 1: V_NPO * k/(k-1) * (1 - psi_o^{1-k})
        IntVN_c1 = V_NPO * k/(k-1) * (1 - psi_o**(1-k))

        # Case 2, sub-branch a: 1 >= a_bar
        IntVN_c2a = (V_NPF_safe * k/(k-1) * (1 - psi_o**(1-k))
                     - (1 - psi_o**(-k)) * w_n_a * d/(d+1))
        # Case 2, sub-branch b: 1 < a_bar
        # Use safe a_bar > 0 in division
        a_bar_safe = np.where(np.isfinite(a_bar) & (a_bar > 0), a_bar, 1.0)
        mb = np.minimum(psi_o, a_bar_safe)
        mb = np.maximum(mb, 1e-30)
        t1 = k/(k-1) * (1 - mb**(1-k)) * V_NPO
        t2 = (k * (mb**(d-k+1) - 1) / (d-k+1)
              * a_bar_safe**(-d) / (d+1)
              * (V_NPF_safe - V_NPO))
        t3 = V_NPF_safe * k/(k-1) * (mb**(1-k) - psi_o**(1-k))
        t4 = -w_n_a * d/(d+1) * (mb**(-k) - psi_o**(-k))
        IntVN_c2b = t1 + t2 + t3 + t4

        # Combine: Case 1 if not c2; Case 2a if c2 and 1>=a_bar; else 2b
        IntVN = np.where(c1, IntVN_c1,
                np.where(1 >= a_bar, IntVN_c2a, IntVN_c2b))

        # =============================================================
        # IntVorigP^A  (eq 51')  — Case 1 only
        # =============================================================
        psi_O_safe = np.where(np.isfinite(psi_O) & (psi_O > 0), psi_O, np.inf)
        IntVA_lo = (k/(k-1) * psi_o**(1-k) * V_NPO
                    + k/(k-1) * np.where(np.isfinite(psi_O_safe),
                                          psi_O_safe**(1-k), 0.0)
                      * (V_PO - V_NPO)
                    - np.where(np.isfinite(psi_O_safe),
                               psi_O_safe**(-k), 0.0) * w_n_fe_h_b)
        IntVA_hi = (k/(k-1) * psi_o**(1-k) * V_PO
                    - w_n_fe_h_b * psi_o**(-k))
        IntVA = np.where(c1,
                          np.where(psi_o < psi_O_safe, IntVA_lo, IntVA_hi),
                          0.0)

        # =============================================================
        # IntVorigP^B  (eq 52')  — Case 2, psi_o < psi_F
        # =============================================================
        psi_F_safe = np.where(np.isfinite(psi_F) & (psi_F > 0), psi_F, np.inf)
        active_B = c2 & (psi_o < psi_F_safe)
        # sub a) a_bar < psi_o
        IntVB_a = (k/(k-1) * V_NPF_safe * (psi_o**(1-k) - psi_F_safe**(1-k))
                   - w_n_a * d/(d+1) * (psi_o**(-k) - psi_F_safe**(-k)))
        # sub b) psi_o <= a_bar < psi_F
        IntVB_b = (V_NPO * k/(k-1) * (psi_o**(1-k) - a_bar_safe**(1-k))
                   + w_n_a / (d+1) * a_bar_safe**(-(d+1)) * k / (k-1-d)
                     * (psi_o**(d-k+1) - a_bar_safe**(d-k+1))
                   + V_NPF_safe * k/(k-1) * (a_bar_safe**(1-k) - psi_F_safe**(1-k))
                   - w_n_a * d/(d+1) * (a_bar_safe**(-k) - psi_F_safe**(-k)))
        # sub c) a_bar >= psi_F
        IntVB_c = (V_NPO * k/(k-1) * (psi_o**(1-k) - psi_F_safe**(1-k))
                   + w_n_a / (d+1) * a_bar_safe**(-(d+1)) * k / (k-1-d)
                     * (psi_o**(d-k+1) - psi_F_safe**(d-k+1)))
        IntVB = np.where(
            active_B,
            np.where(a_bar_safe < psi_o, IntVB_a,
            np.where(a_bar_safe < psi_F_safe, IntVB_b, IntVB_c)),
            0.0)

        # =============================================================
        # IntVorigP^C  (eq 53')  — Case 2 only
        # =============================================================
        psib       = psi_bar
        psib_safe  = np.where(np.isfinite(psib) & (psib > 0), psib, 1.0)
        a_safe     = np.where(a_nis > 0, a_nis, 1.0)
        a_hat_safe = np.where(np.isfinite(a_hat) & (a_hat > 0), a_hat, np.inf)
        active_C   = c2 & (a_nis > 0) & np.isfinite(psib) & (psib > 0)

        max_o_F   = np.maximum(psi_o, psi_F_safe)
        max_F_O_o = np.maximum(np.maximum(psi_F_safe, psi_O_safe), psi_o)

        # sub a) a_hat <= max_o_F
        IntVC_a = (V_PF_safe * k/(k-1)
                   * (max_o_F**(1-k) - max_F_O_o**(1-k))
                   - (w_n_fe_h_b + w_n_a * d/(d+1))
                   * (max_o_F**(-k) - max_F_O_o**(-k)))

        # Vectorized incomplete beta for (d+2, k-d-1)
        # Compute once for both sub-branches b and c using their respective bounds
        if (d + 2) > 0 and (k - d - 1) > 0:
            x1_b = np.clip(1 - psib_safe / np.maximum(max_o_F, 1e-30), 0.0, 1.0)
            x2_b = np.clip(1 - psib_safe / np.maximum(a_hat_safe, 1e-30), 0.0, 1.0)
            beta_full = float(_sp.beta(d+2, k-d-1))
            B_C_b = (_sp.betainc(d+2, k-d-1, x2_b)
                     - _sp.betainc(d+2, k-d-1, x1_b)) * beta_full

            x1_c = x1_b
            x2_c = np.clip(1 - psib_safe / np.maximum(max_F_O_o, 1e-30), 0.0, 1.0)
            B_C_c = (_sp.betainc(d+2, k-d-1, x2_c)
                     - _sp.betainc(d+2, k-d-1, x1_c)) * beta_full
        else:
            # mpmath fallback (rarely needed)
            x1_b = np.clip(1 - psib_safe / np.maximum(max_o_F, 1e-30), 0.0, 1.0)
            x2_b = np.clip(1 - psib_safe / np.maximum(a_hat_safe, 1e-30), 0.0, 1.0)
            x2_c = np.clip(1 - psib_safe / np.maximum(max_F_O_o, 1e-30), 0.0, 1.0)
            B_C_b = _betainc_vec_mpmath(d+2, k-d-1, x1_b, x2_b)
            B_C_c = _betainc_vec_mpmath(d+2, k-d-1, x1_b, x2_c)

        # FIX 1.6: per algorithm eq (53), the middle sub-case's prefactor is
        #     (k/(d+1)) * (h_n f^e_s / a_{nis})^d * (h_n f^e_s w_n) * psi_bar^{-k}
        # i.e. w_n appears EXACTLY ONCE (in the outer patent-cost factor),
        # not inside the d-th power.  Previously the code used
        #     (w_n_fe_h_b / a)^d  =  (w_n h_n f^e_s / a)^d
        # giving w_n^{d+1} total instead of w_n^1.  The high sub-case
        # IntVC_c (below) already has the correct (h_fe)^{d+1} * w_n
        # structure; this fix brings the middle sub-case in line.
        h_fe_b = np.where(w_n > 0,
                          w_n_fe_h_b / np.where(w_n > 0, w_n, 1.0), 0.0)
        IntVC_b = (V_NPO * k/(k-1) * (max_o_F**(1-k) - a_hat_safe**(1-k))
                   + k/(d+1) * (h_fe_b / a_safe)**d
                     * w_n_fe_h_b * psib_safe**(-k) * B_C_b
                   + V_PF_safe * k/(k-1) * (a_hat_safe**(1-k) - max_F_O_o**(1-k))
                   - (w_n_fe_h_b + w_n_a * d/(d+1)) * (
                       a_hat_safe**(-k) - max_F_O_o**(-k)))

        # sub c) a_hat >= max_F_O_o
        # (w_n_fe_h_b / w_n) = h_n * fe_s
        h_fe = np.where(w_n > 0, w_n_fe_h_b / np.where(w_n > 0, w_n, 1.0), 0.0)
        IntVC_c = (V_NPO * k/(k-1) * (max_o_F**(1-k) - max_F_O_o**(1-k))
                   + k * h_fe**(d+1) / (d+1)
                     * w_n * a_safe**(-d) * psib_safe**(-k) * B_C_c)

        IntVC = np.where(
            active_C,
            np.where(a_hat_safe <= max_o_F, IntVC_a,
            np.where(a_hat_safe < max_F_O_o, IntVC_b, IntVC_c)),
            0.0)

        # =============================================================
        # IntVorigP^D  (eq 54')  — Case 2 only
        # =============================================================
        a_chk_safe = np.where(np.isfinite(a_chk) & (a_chk > 0), a_chk, np.inf)
        max_o_O = np.maximum(psi_o, psi_O_safe)

        IntVD_a = max_o_O**(-k) * (
            V_PF_safe * k/(k-1) * max_o_O
            - (w_n_fe_h_b + w_n_a * d/(d+1)))
        IntVD_b = (V_PO * k/(k-1) * (max_o_O**(1-k) - a_chk_safe**(1-k))
                   - w_n_fe_h_b * (max_o_O**(-k) - a_chk_safe**(-k))
                   + w_n_a / (d+1) * a_chk_safe**(-d-1) * k / (k-1-d)
                     * (max_o_O**(d-k+1) - a_chk_safe**(d-k+1))
                   + a_chk_safe**(-k) * (V_PF_safe * k/(k-1) * a_chk_safe
                                          - (w_n_fe_h_b + w_n_a * d/(d+1))))
        IntVD = np.where(c2,
                          np.where(a_chk_safe <= max_o_O, IntVD_a, IntVD_b),
                          0.0)

        # Aggregate
        contrib = IntVN + IntVA + IntVB + IntVC + IntVD
        contrib = np.where(np.isfinite(contrib), contrib, 0.0)
        IntV_sum = contrib.sum(axis=0)                            # (N, S-1)

        # Eq (49'): divide by w_i, subtract patenting-cost
        l_R = np.zeros((p.N, p.S))
        rhs = (IntV_sum / self.w[:, None]
               - p.fo[None, 1:] * p.r_hjort[:, None]
                 * self.psi_o_star[:, 1:]**(-k))
        rhs = np.maximum(rhs, 0.0)
        l_R[..., 1:] = (rhs * p.eta[..., 1:])**(1/p.kappa)
        return l_R

    def compute_phi(self, p):
        """
        phi calibration update — algorithm eq (57) of FDI_algorithm_fixed.tex.

          phi^theta_{nis} = T_n * (w_n^alpha P_n^{1-alpha})^{-theta_s}
                          * R_{nis} * (1 + b_{nis})
                          * X^data_{nis} / (X^data_{nn,s}
                                            - sum_j X^{M,F,data}_{nj,s})

        R_{nis} is the price/quality bracket ratio (eq 58), structurally
        (A + B_{nn,s}) / (A + B_{ni,s}) where A is the (P^CD/P) common term
        and B = PSI^{M,O} * phi^{sigma-1-theta} * (P^{M,O}/P)^{1-sigma}
              / sum_j PSI^{M,O}_{njs} phi^{sigma-1}_{njs}.

        Sector 0 handles itself: PSI^{M,O}_{*,*,0} = 0 ⇒ B = 0 ⇒ R = 1, and
        the FDI sum is zero, so the formula collapses to the standard EK
        inversion.

        Counterfactual branch unchanged.
        """
        if self.context != 'calibration':
            return np.einsum(
                'is,nis,nis,is,is->nis',
                p.T**(1/p.theta[None,:]),
                1/p.tau, 1/(1+p.tariff),
                self.w[:,None]**(-p.alpha[None,:]),
                self.price_indices[:,None]**(p.alpha[None,:]-1))

        # Data ratio: X_{nis} / (X_{nn,s} - sum_j X^{M,F,data}_{nj,s})
        denom_data = (np.einsum('nns->ns', p.trade_flows)
                      - p.X_F_data.sum(axis=1))                    # (N, S)
        data_ratio = p.trade_flows / denom_data[:, None, :]        # (N, N, S)

        # R_{nis} bracket — only sector >= 1 has monopolistic mass; for
        # sector 0, PSI^{M,O} = 0 and P_M = 0, so R collapses to 1.
        sigma_m1   = (p.sigma - 1)[None, None, :]
        sigma_m1_t = (p.sigma - 1 - p.theta)[None, None, :]
        R = np.ones((p.N, p.N, p.S))
        s1 = slice(1, None)
        A = (self.P_CD[:, s1]**(1 - p.sigma[None, s1])
             / (self.phi[..., s1]**p.theta[None, None, s1]).sum(axis=1))  # (N, S-1)
        B = (self.PSI_M_O[..., s1] * self.phi[..., s1]**sigma_m1_t[..., s1]
             * (self.P_M[:, s1]**(1 - p.sigma[None, s1]))[:, None, :]
             / (self.PSI_M_O[..., s1] * self.phi[..., s1]**sigma_m1[..., s1]
                ).sum(axis=1)[:, None, :])                                # (N,N,S-1)
        B_diag = np.einsum('nns->ns', B)                                  # (N, S-1)
        R[..., s1] = (A[:, None, :] + B_diag[:, None, :]) / (A[:, None, :] + B)

        # Assemble phi^theta and take the theta-th root
        wP = (self.w[:, None]**p.alpha[None, :]
              * self.price_indices[:, None]**(1 - p.alpha[None, :]))
        phi_theta_new = ((p.T * wP**(-p.theta[None, :]))[:, None, :]
                         * R * (1 + p.tariff) * data_ratio)
        return phi_theta_new**(1 / p.theta[None, None, :])

    # ── non-solver quantities ───────────────────────────────────────────────

    def scale_P(self, p):
        """Normalise nominal quantities by p_1."""
        num = self.price_indices[0]
        self.w /= num; self.Z /= num; self.X /= num
        self.X_CD /= num; self.X_M /= num; self.X_M_F /= num
        self.phi *= num; self.price_indices /= num
        self.compute_sectoral_prices(p)

    def compute_tau(self, p, assign=True):
        tau = np.einsum(
            'is,nis,nis,is,is->nis',
            p.T**(1/p.theta[None,:]), 1/self.phi, 1/(1+p.tariff),
            self.w[:,None]**-p.alpha[None,:],
            self.price_indices[:,None]**(p.alpha[None,:]-1))
        if assign: self.tau = tau
        else:      return tau

    def compute_nominal_value_added(self, p):
        """
        Eq (59'): w_i L^P_is = alpha_s [sum_n X_{nis}/(1+b)
                                        - sum_n X^{M,O}_{nis}/(sigma(1+b))
                                        - sum_j X^{M,F}_{ijs}/sigma]
        """
        A1 = (self.X / (1+p.tariff)).sum(axis=0)
        A2 = (self.X_M / (1+p.tariff)).sum(axis=0) / p.sigma[None,:]
        # FIX 1.2 (mirror of compute_wage): X_M_F is [n_dest, i_origin, s];
        # the production-labor equation needs affiliate sales LOCATED in
        # destination=i, so sum the origin axis (axis=1), not the destination
        # axis (axis=0).
        A3 = self.X_M_F.sum(axis=1) / p.sigma[None,:]
        self.nominal_value_added = p.alpha[None,:] * (A1 - A2 - A3)

    def compute_nominal_intermediate_input(self, p):
        self.nominal_intermediate_input = np.einsum(
            's,is->is', (1-p.alpha)/p.alpha, self.nominal_value_added)

    def compute_nominal_final_consumption(self, p):
        self.nominal_final_consumption = (
            self.Z - self.nominal_intermediate_input.sum(axis=1))
        self.cons = self.nominal_final_consumption / self.price_indices

        # Sectoral price indices (with FDI mass) and sectoral consumption
        phi_nn = np.einsum('nns->ns', self.phi)
        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, :] * (
            (self.PSI_M_O * self.phi**(p.sigma-1)[None, None, :]).sum(axis=1)
            + self.PSI_M_F.sum(axis=1) * phi_nn**(p.sigma-1)[None, :])
        B = self.PSI_CD * (self.phi**p.theta[None, None, :]
                           ).sum(axis=1)**((p.sigma-1)/p.theta)[None, :]
        temp = gamma((p.theta+1-p.sigma)/p.theta)[None, :] * (A + B)
        one_over = np.divide(1, temp, out=np.full_like(temp, np.inf), where=temp > 0)
        self.sectoral_price_indices = one_over**(1/(p.sigma[None, :]-1))
        self.sectoral_cons = np.einsum('s,n,ns->ns',
                                       p.beta, self.Z,
                                       1/self.sectoral_price_indices)

    def compute_gdp(self, p):
        """GDP with FDI service flows. Axis convention: l_Ae, l_F are (i,n,s)."""
        self.gdp = (
            self.nominal_final_consumption
            + p.deficit_share_world_output
              * np.einsum('nis,nis->', self.X, 1/(1+p.tariff))
            + self.w * np.einsum('is->i', self.l_R + self.l_Ao)
            + np.einsum('n,ins->i', self.w, self.l_Ae)
            + np.einsum('n,ins->i', self.w, self.l_F))

    def compute_pflow(self, p):
        """
        Aggregate flow of patent applications (used by moments).

        FIX 1.8: was previously a Case-1-only proxy
            psi_m_star_O^{-k} * eta * L_R^{1-kappa},
        but pflow IS consumed by moments (SPFLOW, TP, JUPCOST, PCOST,
        DOMPATRATUSEU, ...), so the proxy biased every patent-flow moment.

        Per algorithm eq (38), the patent-application labor is
            L^e_{ins} = h_n * f^e_s * eta_{is} * L_R^{1-kappa} *
                        [full Case 1/Case 2 patent probability].
        The number of patent applications by origin i in destination n is
            P^s_{ni} = L^e_{ins} / (h_n * f^e_s)
                     = eta_{is} * L_R^{1-kappa} * [full patent probability].

        In the code, l_Ae[i_origin, n_destination, s] already contains the
        full mixed expression (constructed in compute_labor_allocations).
        Divide by h_n * f^e_s and transpose to (n_dest, i_origin, s) to
        match the moments-class indexing convention.
        """
        N, S = p.N, p.S
        # h_n * f^e_s for s >= 1 (sector 0 excluded from patenting)
        h_fe = np.einsum('n,s->ns', p.r_hjort, p.fe[1:])  # (N, S-1)
        safe_hfe = np.where(h_fe > 0, h_fe, 1.0)

        # l_Ae axes: [origin i, destination n, sector s]
        # P^s_{ni}: index [destination n, origin i, sector s]
        # So divide l_Ae by h_n * f^e_s broadcasting on (i_origin, n_dest, s),
        # then transpose (i, n) -> (n, i).
        # h_fe is (n, s); broadcast as (1, n, s) along origin axis.
        pflow_ins = np.where(
            h_fe[None, :, :] > 0,
            self.l_Ae[..., 1:] / safe_hfe[None, :, :],
            0.0)  # shape (N, N, S-1), indexed [i_origin, n_dest, s]
        pflow_nis = np.transpose(pflow_ins, (1, 0, 2))  # -> [n_dest, i_origin, s]
        # Preserve previous behavior: squeeze trailing length-1 sector axis
        # (for S=2 the moments code expects a 2D (N,N) array).
        self.pflow = pflow_nis.squeeze()

    def compute_share_of_innovations_patented(self, p):
        self.share_innov_patented = self.psi_m_star_O[...,1:]**(-p.k[1])

    def compute_non_solver_quantities(self, p):
        self.compute_tau(p)
        self.compute_nominal_value_added(p)
        self.compute_nominal_intermediate_input(p)
        self.compute_nominal_final_consumption(p)
        self.compute_gdp(p)
        self.compute_pflow(p)
        self.compute_share_of_innovations_patented(p)

    def compute_consumption_equivalent_welfare(self, p, baseline):
        self.cons_eq_welfare = (
            self.cons
            * ((p.rho-baseline.g*(1-1/p.gamma))
               /(p.rho-self.g*(1-1/p.gamma)))**(p.gamma/(p.gamma-1))
            / baseline.cons)

    def compute_world_welfare_changes(self, p, baseline):
        one_ov_g = 1/p.gamma
        n = (p.labor**one_ov_g * self.cons**((p.gamma-1)*one_ov_g)).sum() * (
            p.rho - baseline.g*(1-one_ov_g))
        d = (p.labor**one_ov_g * baseline.cons**((p.gamma-1)*one_ov_g)).sum() * (
            p.rho - self.g*(1-one_ov_g))
        self.cons_eq_pop_average_welfare_change = (n/d)**(p.gamma/(p.gamma-1))
        n2 = (baseline.cons**one_ov_g * self.cons**((p.gamma-1)*one_ov_g)).sum() * (
            p.rho - baseline.g*(1-one_ov_g))
        d2 = baseline.cons.sum() * (p.rho - self.g*(1-one_ov_g))
        self.cons_eq_negishi_welfare_change = (n2/d2)**(p.gamma/(p.gamma-1))


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
                                    y = p.k[1],
                                    z = p.d)
        
        # print(integral_k_d)
        
        self.integral_k_d = integral_k_d
        
        temp_w_a_power_minus_d = np.divide(
            1, 
            (self.w[None,:,None]*self.a[...,1:])**(p.d), 
            out=np.zeros_like(self.a[...,1:]),
            where=self.a[...,1:]!=0
            )
        
        A = p.k[1]*(
            1
            - np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(1-p.k[1])
            + self.psi_m_star[...,1:]**(1-p.k[1])
            - self.psi_MP_star[...,1:]**(1-p.k[1])
            )/(p.k[1]-1)
        
        
        B = p.k[1]*np.einsum('nis,nis->nis',
                      temp_w_a_power_minus_d/self.V_NP[...,1:]**(-p.d),
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k[1]+1) - 1
                      )/(p.d-p.k[1]+1)
        
        C = p.k[1]*np.einsum('nis,ni->nis',
                          temp_w_a_power_minus_d,
                          integral_k_d
                          )
        
        self.mu_MNE = np.zeros((p.N,p.N,p.S))
        self.mu_MNE[...,1:] = A - B - C
        
        # print('mu_MNE')
        
        self.mu_MPND = np.zeros((p.N,p.N,p.S))
        self.mu_MPND[...,1:] = C + (p.k[1]*self.psi_MP_star[...,1:]**(1-p.k[1]))/(p.k[1]-1)
        
        D = p.k[1]*(
            np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(1-p.k[1])
            - self.psi_m_star[...,1:]**(1-p.k[1])
            )/(p.k[1]-1)
        
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
                         self.psi_o_star[...,1:]**-p.k[1]
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
                                    y = p.k[1]+1,
                                    z = p.d)
        self.integral_k_plus_un_d = integral_k_plus_un_d
        self.l_Ae[...,1:] = np.einsum('n,s,is,is,nis -> ins',
                         p.r_hjort,
                         p.fe[1:],
                         p.eta[...,1:],
                         self.l_R[...,1:]**(1-p.kappa),
                         p.k[1]*temp_w_a_power_minus_d*integral_k_plus_un_d[...,None]+self.psi_MP_star[...,1:]**-p.k[1]
                         )
        
        self.l_Aa = np.zeros((p.N,p.N,p.S))
        A = p.k[1]*np.einsum('nis,nis,nis -> nis',
                      temp_w_a_power_minus_d_minus_1,
                      1/self.V_NP[...,1:]**(-p.d-1),
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k[1]+1) - 1
                      )/(p.d-p.k[1]+1)
        B = self.psi_MP_star[...,1:]**(-p.k[1])\
            -self.psi_m_star[...,1:]**(-p.k[1])\
            +np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(-p.k[1])

        lb = self.psi_m_star[...,1]
        ub = self.psi_MP_star[...,1]
        alpha = self.V_P[...,1]
        # b = self.w[:,None]*p.fe[1]*p.r_hjort[:,None],
        beta = self.w*p.fe[1]*p.r_hjort
        y = p.k[1]
        z = p.d
        
        # by part integration to use the previous calculation of the integral
        term_lb = lb**(-y)*(alpha*lb-beta[:,None])**(z+1)/y
        term_ub = ub**(-y)*(alpha*ub-beta[:,None])**(z+1)/y
        
        integral_k_plus_un_d_plus_un = term_lb-term_ub + alpha*(z+1)*self.integral_k_d/y
        
        self.integral_k_plus_un_d_plus_un = integral_k_plus_un_d_plus_un
        C = p.k[1]*temp_w_a_power_minus_d_minus_1*integral_k_plus_un_d_plus_un[...,None]
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
        A_1 = p.k[1]*np.einsum('nis,i,nis->nis',
                        self.V_NP[...,1:],
                        1/self.w,
                        self.a_NP_star[...,1:]
                        )/(p.k[1]-1)
        A_2 = p.d*self.a[...,1:]/(p.d+1)
        A = np.einsum('nis,nis->nis',
                      A_1 - A_2,
                      self.a_NP_star[...,1:]**(-p.k[1])
                      )
        
        B_1 = p.k[1]*np.einsum('nis,i,nis->nis',
                        self.V_NP[...,1:],
                        1/self.w,
                        self.psi_MNP_star[...,1:]
                        )/(p.k[1]-1)
        B_2 = p.d*self.a[...,1:]/(p.d+1)
        B = np.einsum('nis,nis->nis',
                      B_1 - B_2,
                      self.psi_MNP_star[...,1:]**(-p.k[1])
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
        
        C = p.k[1]*np.einsum('nis,i,nis,nis,nis->nis',
                      temp_w_a_power_minus_d,
                      1/self.w,
                      self.V_NP[...,1:]**(p.d+1),
                      signature_C,
                      np.minimum(self.psi_m_star[...,1:],self.a_NP_star[...,1:])**(p.d-p.k[1]+1) - 1
                      )/((p.d+1)*(p.d-p.k[1]+1))
        
        # print(C)
        
        # C[np.isnan(C)] = 0
        # C[C < 0] = 0
        
        D_1 = p.k[1]*np.einsum('nis,i,nis->nis',
                        self.V_P[...,1:],
                        1/self.w,
                        self.psi_MP_star[...,1:]
                        )/(p.k[1]-1)
        D_2 = self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None]/self.w[None,:,None]\
                +p.d*self.a[...,1:]/(p.d+1)
        D = np.einsum('nis,nis->nis',
                      D_1 - D_2,
                      self.psi_MP_star[...,1:]**(-p.k[1])
                      )
        
        signature_E = np.einsum('nis,nis->nis',
            self.w[None,:,None]*self.a[...,1:]+self.w[:,None,None]*p.fe[None,None,1:]*p.r_hjort[:,None,None],
            1/self.V_P[...,1:]
            )  > self.psi_o_star[None,:,1:]
        
        integral_k_plus_un_d_plus_un = self.integral_k_plus_un_d_plus_un
        
        E = p.k[1]*np.einsum('nis,nis,i,ni->nis',
                      signature_E,
                      temp_w_a_power_minus_d,
                      1/self.w,
                      integral_k_plus_un_d_plus_un
                      )/(p.d+1)
        
        l_R = np.zeros((p.N,p.S))
        temp = (A - B + C + D +E).sum(axis=0) - p.fe[None,1:]*p.r_hjort[:,None]*self.psi_o_star[...,1:]**(-p.k[1])
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
        
        bracket = p.k[1]*np.einsum('nis,ni->nis',
                              temp_w_a_power_minus_d,
                              self.integral_k_plus_un_d
                              ) + self.psi_MP_star[...,1:]**(-p.k[1])
        
        self.pflow = np.einsum('nis,is,is->nis',
                              bracket,
                              p.eta[...,1:],
                              self.l_R[...,1:]**(1-p.kappa)
                              ).squeeze()
        
    def compute_share_of_innovations_patented(self,p):
        # this will only be valid for domestic quantities, we only use it as such
        self.share_innov_patented = self.psi_m_star[...,1:]**(-p.k[1])
        
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
            (1-p.kappa)*p.k[1]/(p.kappa*(p.k[1]-1))
              )*np.einsum('is,is,s,i,is,is->is',
                      p.eta[...,1:],
                      1/self.l_R[...,1:]**p.kappa,
                      p.fe[1:]+p.fo[1:],
                      p.r_hjort,
                      self.psi_o_star[...,1:]**(-p.k[1]),
                      1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:])
                      )
                         
        B = p.k[1]*(1/(self.G[None,1:]+p.delta[...,1:])+1/(self.G[None,1:]+p.delta[...,1:]-p.nu[None,1:]))
        
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
        
        self.mass_enters = p.k[1]/(p.k[1]-1) - self.mu_MNE
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
        # print(self.l_R.shape)
        self.g_s = p.k*np.einsum('is,is -> s',
                                 p.eta,
                                 self.l_R**(1-p.kappa)
                                 )/(p.k-1) - p.zeta
        self.g_s[0] = p.g_0
        self.g = (p.beta*self.g_s/(p.sigma-1)).sum() / (p.beta*p.alpha).sum()
        self.r = p.rho + self.g/p.gamma
        self.G = self.r+p.zeta-self.g+self.g_s+p.nu
        
    def compute_patenting_thresholds(self, p,exog_patent_thresholds=False):
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
        if not exog_patent_thresholds:
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
        
    def compute_solver_quantities(self,p,exog_patent_thresholds=False):
        self.compute_growth(p)
        self.compute_patenting_thresholds(p,exog_patent_thresholds=exog_patent_thresholds)
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
            1-np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k[1])
            ) + self.V_P_P_minus_V_P_NP_with_prod_patent[...,1]/self.w[None,:]*(
                np.maximum(self.psi_m_star[...,1]/np.diagonal(self.psi_m_star[...,1])[None,:],1)**(1-p.k[1])
                )
        
        self.mult_val_pat = 1 + (
            num_bracket.sum(axis=0) - np.diagonal(num_bracket)
            )/( np.diagonal(self.profit[...,1]) * (1/(self.G[1]+p.delta[:,1]-p.nu[1])-1/(self.G[1]+p.delta[:,1])) )
        
        
        # iii)
        
        self.V_with_prod_patent = np.zeros((p.N,p.S))
        
        A1 = ((p.k[1]/(p.k[1]-1))*self.V_NP[...,1]/self.w[None,:]).sum(axis=0)
        A2 = np.einsum('ni,ni,i->i',
                        self.V_P[...,1]/self.w[None,:] - self.V_NP[...,1]/self.w[None,:],
                        self.psi_m_star[...,1]**(1-p.k[1]),
                        self.mult_val_pat
                        )*(p.k[1]/(p.k[1]-1))
        A3 = - np.einsum('ni,n,n,i->i',
                          self.psi_m_star[...,1]**-p.k[1],
                          self.w,
                          p.r_hjort,
                          1/self.w
                          )*p.fe[1]
        B = self.psi_o_star[:,1]**-p.k[1]*p.fo[1]*p.r_hjort
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
        
        # TYPO
        # self.sectoral_average_markup = np.einsum(
        #     's,is,is->is',
        #     prefactor,
        #     np.einsum('nis->is',A),
        #     1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
        #     ) + np.einsum(
        #         's,is,is->is',
        #         prefactor,
        #         np.einsum('nis->is',B),
        #         1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
        #         )
        self.sectoral_average_markup = np.einsum(
            's,is,is->is',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
            ) + np.einsum(
                's,nis,is->is',
                prefactor,
                np.einsum('nis->nis',B),
                1/np.einsum('nis->is',A+prefactor[None,None,:]*B)
                )
        
        # TYPO
        # self.aggregate_average_markup = np.einsum(
        #     's,is,i->i',
        #     prefactor,
        #     np.einsum('nis->is',A),
        #     1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
        #     ) + np.einsum(
        #         's,is,i->i',
        #         prefactor,
        #         np.einsum('nis->is',B),
        #         1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
        #         )
        self.aggregate_average_markup = np.einsum(
            's,is,i->i',
            prefactor,
            np.einsum('nis->is',A),
            1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
            ) + np.einsum(
                's,nis,i->i',
                prefactor,
                np.einsum('nis->nis',B),
                1/np.einsum('nis->i',A+prefactor[None,None,:]*B)
                )

    # def compute_export_price_index(self,p)  :
    #     numeratorA = np.einsum('s,nis,nis,s->nis',
    #         gamma((p.theta+2-p.sigma)/p.theta)[1:],
    #         self.PSI_M[...,1:],
    #         self.phi[...,1:]**(p.sigma[None,None,1:]-2),
    #         ((p.sigma/(p.sigma-1))**(2-p.sigma))[1:]
    #         )          
    #     numeratorB = np.einsum('nis,ns,ns->nis',
    #         self.phi[...,1:]**(p.theta[None,None,1:]),
    #         self.PSI_CD[...,1:],
    #         (
    #             (self.phi[...,1:]**(p.theta[None,1:])).sum(axis=1)
    #         )**((p.sigma[None,1:]-2)/p.theta[None,1:]-1)
    #         )
        
    #     denominatorA = np.einsum('s,nis,nis,s->nis',
    #         gamma((p.theta+1-p.sigma)/p.theta)[1:],
    #         self.PSI_M[...,1:],
    #         self.phi[...,1:]**(p.sigma[None,None,1:]-1),
    #         ((p.sigma/(p.sigma-1))**(1-p.sigma))[1:]
    #         )          
    #     denominatorB = np.einsum('nis,ns,ns->nis',
    #         self.phi[...,1:]**(p.theta[None,None,1:]),
    #         self.PSI_CD[...,1:],
    #         (
    #             (self.phi[...,1:]**(p.theta[None,1:])).sum(axis=1)
    #         )**((p.sigma[None,1:]-1)/p.theta[None,1:]-1)
    #         ) 
        
    #     self.export_price_index = (numeratorA + numeratorB) / (denominatorA + denominatorB)
    def compute_export_price_index(self,p)  :
        # Gamma((theta+2-sigma)/theta) / Gamma((theta+1-sigma)/theta)
        # multiplies the WHOLE ratio (both monopolistic and competitive
        # parts of the avg-price formula). The previous code applied the
        # numerator's Gamma only to the monopolistic term and similarly
        # for the denominator, so the competitive-only limit dropped the
        # Gamma factor. Apply both Gammas to both A and B contributions.
        gamma_num = gamma((p.theta+2-p.sigma)/p.theta)[1:]   # (S-1,)
        gamma_den = gamma((p.theta+1-p.sigma)/p.theta)[1:]   # (S-1,)

        numeratorA = np.einsum('s,nis,nis,s->nis',
            gamma_num,
            self.PSI_M[...,1:],
            self.phi[...,1:]**(p.sigma[None,None,1:]-2),
            ((p.sigma/(p.sigma-1))**(2-p.sigma))[1:]
            )
        numeratorB = np.einsum('s,nis,ns,ns->nis',
            gamma_num,
            self.phi[...,1:]**(p.theta[None,None,1:]),
            self.PSI_CD[...,1:],
            (
                (self.phi[...,1:]**(p.theta[None,1:])).sum(axis=1)
            )**((p.sigma[None,1:]-2)/p.theta[None,1:]-1)
            )

        denominatorA = np.einsum('s,nis,nis,s->nis',
            gamma_den,
            self.PSI_M[...,1:],
            self.phi[...,1:]**(p.sigma[None,None,1:]-1),
            ((p.sigma/(p.sigma-1))**(1-p.sigma))[1:]
            )
        denominatorB = np.einsum('s,nis,ns,ns->nis',
            gamma_den,
            self.phi[...,1:]**(p.theta[None,None,1:]),
            self.PSI_CD[...,1:],
            (
                (self.phi[...,1:]**(p.theta[None,1:])).sum(axis=1)
            )**((p.sigma[None,1:]-1)/p.theta[None,1:]-1)
            )

        raw_pi = (numeratorA + numeratorB) / (denominatorA + denominatorB)

        # Normalize each destination's bilateral price index by the
        # trade-weighted average across origins (i != n), using total
        # bilateral trade X = X_M + X_CD as weights. Excluding own
        # destination (i = n) from the weighted average matches the
        # definition in the tex note.
        weights = (self.X_M[..., 1:] + self.X_CD[..., 1:])  # (N, N, S-1)
        N = weights.shape[0]
        off_diag = (~np.eye(N, dtype=bool))[..., None]      # (N, N, 1)
        w_off = weights * off_diag
        num_avg = (w_off * raw_pi).sum(axis=1)              # (N, S-1)
        den_avg = w_off.sum(axis=1)                         # (N, S-1)
        avg_pi = num_avg / np.where(den_avg > 0, den_avg, 1.0)  # (N, S-1)

        self.export_price_index = raw_pi / avg_pi[:, None, :]                                 
                                                                     
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
            temp = (self.PSI_CL[..., 1:]*self.phi[..., 1:]**(p.sigma-1)[None, None, 1:]).sum(axis=1)
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
        
    def compute_patenting_thresholds(self, p, exog_patent_thresholds=False):
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
        if not exog_patent_thresholds:
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
                    # self.plot_numerical_derivatives()
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
        B = self.psi_o_star[:,1:,:]**-p.k[None,1:,None]*p.fo[None,1:,None]*p.r_hjort[:,None,None]*self.w[:,None,:]
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
        
    def compute_solver_quantities(self,p,exog_lr=False,exog_patent_thresholds=False):
        self.compute_phi(p)
        self.compute_PSI_M(p)
        self.compute_sectoral_prices(p)
        self.compute_patenting_thresholds(p,exog_patent_thresholds=exog_patent_thresholds)
        self.compute_V(p)
        if not exog_lr:
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
        # A = np.einsum('nist->it', self.X)
        # B = np.einsum('it,nist->it', self.w, self.l_Ae)
        # C = np.einsum('i,kt->it',p.deficit_share_world_output,self.Z)
        # D = np.einsum('nt,inst->it', self.w, self.l_Ae)
        # Z = (A+B-(C+D))
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

class dynamic_var_double_diff_double_delta:
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
                                   for qty in ['PSI_CL','PSI_CD','PSI_MNP','PSI_MPND',
                                               # 'PSI_MPD','V_PD','V_P','V_NP']]
                                               'PSI_MPL','PSI_MPD','V_PD','DELTA_V','V_NP']]
        vec = np.concatenate([price_indices,w,Z]+list_of_raveled_vectors, axis=0)
        return vec
    
    
    @staticmethod
    def var_from_vector(vec,p,compute = True,sol_init=None,Nt=25,t_inf=500,sol_fin=None):
        init = dynamic_var_double_diff_double_delta(sol_init=sol_init,nbr_of_time_points =Nt,t_inf=t_inf,sol_fin=sol_fin)
        init.initiate_state_variables_0(sol_init)
        dic_of_guesses = {'price_indices':np.zeros((p.N,Nt)),
                        'w':np.zeros((p.N,Nt)),
                        'Z':np.zeros((p.N,Nt)),
                        'PSI_CL':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'PSI_CD':np.zeros((p.N,p.S,Nt))[...,1:,:],
                        'PSI_MNP':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'PSI_MPND':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
                        'PSI_MPL':np.zeros((p.N,p.N,p.S,Nt))[...,1:,:],
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
        
        idx_end += self.PSI_CL[...,1:,:].size
        self.guess_PSI_CL(x_old[idx:idx_end].reshape(self.PSI_CL[...,1:,:].shape)
                          ,only_patenting_sectors=True)
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
        
        idx_end += self.PSI_MPL[...,1:,:].size
        self.guess_PSI_MPL(x_old[idx:idx_end].reshape(self.PSI_MPL[...,1:,:].shape)
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
            
    def guess_PSI_CL(self,PSI_CL_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_CL_init.shape)
            shape[2] = shape[2]+1
            self.PSI_CL = np.zeros(shape)
            self.PSI_CL[...,1:,:] = PSI_CL_init
        else:
            self.PSI_CL = PSI_CL_init
        
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
        
    def guess_PSI_MPL(self,PSI_MPL_init,only_patenting_sectors=False):
        if only_patenting_sectors:
            shape = list(PSI_MPL_init.shape)
            shape[2] = shape[2]+1
            self.PSI_MPL = np.zeros(shape)
            self.PSI_MPL[...,1:,:] = PSI_MPL_init
        else:
            self.PSI_MPL = PSI_MPL_init
        
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
        self.guess_PSI_CL(dic_of_guesses['PSI_CL'],only_patenting_sectors=True)
        self.guess_PSI_CD(dic_of_guesses['PSI_CD'],only_patenting_sectors=True)
        self.guess_PSI_MNP(dic_of_guesses['PSI_MNP'],only_patenting_sectors=True)
        self.guess_PSI_MPND( dic_of_guesses['PSI_MPND'],only_patenting_sectors=True)
        self.guess_PSI_MPL( dic_of_guesses['PSI_MPL'],only_patenting_sectors=True)
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
        self.PSI_CL_0 = var.PSI_CL
        self.PSI_CD_0 = var.PSI_CD
        self.PSI_MNP_0 = var.PSI_MNP
        self.PSI_MPND_0 = var.PSI_MPND
        self.PSI_MPL_0 = var.PSI_MPL
        self.PSI_MPD_0 = var.PSI_MPD
        self.PSI_M_0 = self.PSI_MNP_0+self.PSI_MPND_0+self.PSI_MPL_0+self.PSI_MPD_0
    
    def compute_phi(self, p):
        self.phi = np.einsum('is,nis,nis,ist,ist->nist',
                p.T**(1/p.theta[None,:]),
                1/p.tau,
                1/(1+p.tariff),
                self.w[:,None,:]**(-p.alpha[None,:,None]),
                self.price_indices[:,None,:]**(p.alpha[None,:,None]-1))
        
    def compute_PSI_M(self,p):
        self.PSI_M = self.PSI_MNP + self.PSI_MPND + self.PSI_MPD + self.PSI_MPL
    
    def compute_sectoral_prices(self, p):
        power = p.sigma-1

        A = ((p.sigma/(p.sigma-1))**(1-p.sigma))[None, 1:, None] \
            * ((self.PSI_M[...,1:,:]+self.PSI_M_0[...,1:,None])*self.phi[...,1:,:]**power[None, None, 1:,None]).sum(axis=1)

        B = (self.PSI_CD[...,1:,:]+self.PSI_CD_0[...,1:,None])*(
            self.phi[...,1:,:]**p.theta[None,None,1:,None]).sum(axis=1)**(power/p.theta)[None, 1:,None]
        
        C = ((self.PSI_CL[...,1:,:]+self.PSI_CL_0[...,1:,None])*self.phi[...,1:,:]**power[None, None, 1:,None]).sum(axis=1)

        self.P_M = np.full((p.N, p.S, self.Nt),np.inf)
        self.P_M[:,1:,:] = (A/(A+B+C))**(1/(1-p.sigma))[None, 1:,None]
        
        self.P_CD = np.ones((p.N, p.S, self.Nt))
        self.P_CD[:,1:,:] = (B/(A+B+C))**(1/(1-p.sigma))[None, 1:,None]
        
        self.P_CL = np.ones((p.N, p.S, self.Nt))
        self.P_CL[:,1:,:] = (C/(A+B+C))**(1/(1-p.sigma))[None, 1:,None]
        
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
                # self.plot_numerical_derivatives()
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
        B = self.psi_o_star[:,1:,:]**-p.k[None,1:,None]*p.fo[None,1:,None]*p.r_hjort[:,None,None]*self.w[:,None,:]
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
            temp = ((self.PSI_CL+self.PSI_CL_0[...,None])[...,1:,:]*self.phi[...,1:,:]**(p.sigma-1)[None, None, 1:,None]).sum(axis=1)
            self.X_CL = np.zeros((p.N, p.N, p.S, self.Nt))
            self.X_CL[...,1:,:] = np.einsum('nist,nist,nst,nst,s,nt->nist',
                                    self.phi[..., 1:,:]**(p.sigma-1)[None, None, 1:,None],
                                    (self.PSI_CL+self.PSI_CL_0[...,None])[...,1:,:],
                                    1/temp,
                                    self.P_CL[...,1:,:]**(1-p.sigma[None,1:,None]),
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

            self.X = self.X_M+self.X_CD+self.X_CL
            
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
        # A = np.einsum('nist->it', self.X)
        # B = np.einsum('it,nist->it', self.w, self.l_Ae)
        # C = np.einsum('i,kt->it',p.deficit_share_world_output,self.Z)
        # D = np.einsum('nt,inst->it', self.w, self.l_Ae)
        # Z = (A+B-(C+D))
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
        C = ((self.PSI_CL + self.PSI_CL_0[...,None]) * self.phi**power[None, None, :, None]).sum(axis=1)

        temp = (gamma((p.theta+1-p.sigma)/p.theta)[None,:,None]*(A+B+C))
        one_over_price_indices_no_pow_no_prod =  np.divide(1, temp, out=np.full_like(temp,np.inf), where=temp > 0)
        price_indices = (one_over_price_indices_no_pow_no_prod**(p.beta[None, :, None]/(p.sigma[None, :, None]-1)) ).prod(axis=1)
        return price_indices
    
    def compute_PSI_CL(self,p):
        self.PSI_CL_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_CL[...,1:,:])/self.t_inf
        PSI_CL = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nist',
            p.nu[1:],
            self.PSI_MNP[...,1:,:]+self.PSI_MNP_0[...,1:,None],
            )
        numB = np.einsum('nis,nist->nist',
            p.delta_eff[:,:,1:],
            self.PSI_MPL[...,1:,:]+self.PSI_MPL_0[...,1:,None],
            )
        numC = self.PSI_CL_dot
        PSI_CL[...,1:,:] = np.einsum('nist,st->nist',
                           numA+numB-numC,
                           1/(self.g_s[1:,:]+p.zeta[1:,None]+p.nu_tilde[1:,None])
                           )-self.PSI_CL_0[...,1:,None]
        
        PSI_CL[...,-1] = 0
        
        return PSI_CL
    
    def compute_PSI_CD(self,p):
        self.PSI_CD_dot = 2*np.einsum('tu,nsu->nst',self.D_neuman,self.PSI_CD[...,1:,:])/self.t_inf
        PSI_CD = np.zeros((p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nst',
            p.nu_tilde[1:],
            self.PSI_MNP[...,1:,:]+self.PSI_MNP_0[...,1:,None]+self.PSI_CL[...,1:,:]+self.PSI_CL_0[...,1:,None],
            )
        numB = np.einsum('nis,nist->nst',
            p.delta_eff[:,:,1:],
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
        numB = np.einsum('nis,nist->nist',
            p.delta_eff[:,:,1:],
            self.PSI_MPND[...,1:,:]+self.PSI_MPND_0[...,1:,None],
            )
        numC = self.PSI_MNP_dot 
        PSI_MNP[...,1:,:] = np.einsum('nist,st->nist',
                           numA+numB-numC,
                           1/(self.g_s[1:,:]+p.zeta[1:,None]+p.nu[1:,None]+p.nu_tilde[1:,None])
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
        PSI_MPND[...,1:,:] = np.einsum('nist,nist->nist',
                           numA-numB,
                           1/(self.g_s[None,None,1:,:]+p.zeta[None,None,1:,None]\
                              +p.nu[None,None,1:,None]+p.nu_tilde[None,None,1:,None]+p.delta_eff[:,:,1:,None])
                           )-self.PSI_MPND_0[...,1:,None]
        
        PSI_MPND[...,-1] = 0
        
        return PSI_MPND
    
    def compute_PSI_MPL(self,p):
        self.PSI_MPL_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_MPL[...,1:,:])/self.t_inf
        PSI_MPL = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nist',
            p.nu[1:],
            self.PSI_MPND[...,1:,:]+self.PSI_MPND_0[...,1:,None],
            )
        numB = self.PSI_MPL_dot
        PSI_MPL[...,1:,:] = np.einsum('nist,nist->nist',
                           numA-numB,
                           1/(self.g_s[None,None,1:,:]+p.nu_tilde[None,None,1:,None]+p.zeta[None,None,1:,None]+p.delta_eff[:,:,1:,None])
                           )-self.PSI_MPL_0[...,1:,None]
        
        PSI_MPL[...,-1] = 0
        
        return PSI_MPL
        
    def compute_PSI_MPD(self,p):
        self.PSI_MPD_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.PSI_MPD[...,1:,:])/self.t_inf
        PSI_MPD = np.zeros((p.N,p.N,p.S,self.Nt))
        numA = np.einsum('s,nist->nist',
            p.nu_tilde[1:],
            self.PSI_MPND[...,1:,:]+self.PSI_MPND_0[...,1:,None]+self.PSI_MPL[...,1:,:]+self.PSI_MPL_0[...,1:,None],
            )
        numB = self.PSI_MPD_dot
        PSI_MPD[...,1:,:] = np.einsum('nist,nist->nist',
                           numA-numB,
                           1/(self.g_s[None,None,1:,:]+p.zeta[None,None,1:,None]+p.delta_eff[:,:,1:,None])
                           )-self.PSI_MPD_0[...,1:,None]
        
        PSI_MPD[...,-1] = 0
        
        return PSI_MPD
        
    def compute_V_PD(self,p):
        self.V_PD_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_PD[...,1:,:])/self.t_inf
        V_PD = np.zeros((p.N,p.N,p.S,self.Nt))
        V_PD[...,1:,:] = np.einsum('nist,nist->nist',
                                   self.profit[...,1:,:]+self.V_PD_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.delta_eff[:,:,1:,None]\
                                      +self.g_s[None,None,1:,:]-self.g[None,None,None,:])
                                   )
        return V_PD
        
    def compute_V_NP(self,p):
        self.V_NP_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_NP[...,1:,:])/self.t_inf
        V_NP = np.zeros((p.N,p.N,p.S,self.Nt))
        V_NP[...,1:,:] = np.einsum('nist,ist->nist',
                                   self.profit[...,1:,:]+self.V_NP_dot,
                                   1/(self.r[:,None,:]+p.zeta[None,1:,None]+p.nu[None,1:,None]+p.nu_tilde[None,1:,None]\
                                      +self.g_s[None,1:,:]-self.g[None,None,:])
                                   )
        return V_NP
        
    def compute_V_P(self,p):
        #Not used, replaced by DELTA_V = V_P - V_NP
        self.V_P_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.V_P[...,1:,:])/self.t_inf
        V_P = np.zeros((p.N,p.N,p.S,self.Nt))
        V_P[...,1:,:] = np.einsum('nist,nist->nist',
                                   self.profit[...,1:,:]+p.delta_eff[:,:,1:,None]*self.V_NP[...,1:,:]\
                                       +(p.nu[None,None,1:,None]+p.nu_tilde[None,None,1:,None])*self.V_PD[...,1:,:]+self.V_P_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.nu[None,None,1:,None]+p.nu_tilde[None,None,1:,None]+
                                      p.delta_eff[:,:,1:,None]+self.g_s[None,None,1:,:]-self.g[None,None,None,:])
                                   )
        return V_P
    
    def compute_DELTA_V(self,p):
        self.DELTA_V_dot = 2*np.einsum('tu,nisu->nist',self.D_neuman,self.DELTA_V[...,1:,:])/self.t_inf
        DELTA_V = np.zeros((p.N,p.N,p.S,self.Nt))
        DELTA_V[...,1:,:] = np.einsum('nist,nist->nist',
                                   (p.nu[None,None,1:,None]+p.nu_tilde[None,None,1:,None])*self.V_PD[...,1:,:]+self.DELTA_V_dot,
                                   1/(self.r[None,:,None,:]+p.zeta[None,None,1:,None]+p.nu[None,None,1:,None]+p.nu_tilde[None,None,1:,None]+
                                      p.delta_eff[:,:,1:,None]+self.g_s[None,None,1:,:]-self.g[None,None,None,:])
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
            self.list_of_moments = ['GPDIFF', 'GROWTH', 'OUT', 'KM','KM_DD_DD','KMCHEM','KMPHARMA','KMPHARMACHEM',
                                    'KMPATENT',
                                    'KM_GDP', 
                                    'RD','RDPHARMA','RDCHEM','RDPHARMACHEM','RD_US','RD_RUS', 'RP',
                               'SRDUS', 'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US','SDOMTFLOW','STFLOW',
                               'STFLOWSDOM','SPFLOWDOM_RUS', 'SPFLOW_RUS','SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST',
                               'UUPCOST','UUPCOSTS','PCOST','PCOSTINTER','PCOSTNOAGG','PCOSTINTERNOAGG',
                               'JUPCOSTRD','SINNOVPATUS','TO','TO_DD_DD','TOCHEM','TOPHARMA','TOPHARMACHEM','TOPATENT',
                               'TE','TECHEM','TEPHARMA','TEPHARMACHEM',
                               'DOMPATRATUSEU','DOMPATUS','DOMPATEU','AGGAVMARKUP','AVMARKUPPHARCHEM',
                               'DOMPATINUS','DOMPATINCHN','DOMPATINEU','SPATORIG','SPATDEST','TWSPFLOW','TWSPFLOWDOM','ERDUS',
                               'PROBINNOVENT','SHAREEXPMON','SGDP','RGDPPC','SDFLOW','FDI_FLOW_N','FDI_FLOW','FDI_ELAST']
        else:
            self.list_of_moments = list_of_moments
        self.weights_dict = {'GPDIFF': 1,
                             'GROWTH': 5,
                             'KM': 1,
                             'KM_DD_DD': 1,
                             'KMCHEM': 1,
                             'KMPATENT': 1,
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
                             'TOPATENT': 5,
                             'TECHEM': 5,
                             'TEPHARMA': 5,
                             'TEPHARMACHEM': 5,
                             'TE': 5,
                             'DOMPATRATUSEU': 2,
                             'DOMPATUS': 1,
                             'DOMPATEU': 1,
                             'DOMPATINUS': 1,
                             'DOMPATINCHN': 1,
                             'DOMPATINEU': 1,
                             'SPATORIG': 2,
                             'SPATDEST': 2,
                             'TWSPFLOW': 1,
                             'TWSPFLOWDOM': 1,
                             'ERDUS': 3,
                             'PROBINNOVENT': 5,
                             'FDI_FLOW_N': 1,
                             # FDI_FLOW is bilateral (N*N entries) — per-entry
                             # weight 1 makes its aggregate contribution
                             # dominate other moments by ~N. Keep per-entry
                             # weight at 1; tune externally via list_of_moments
                             # or by overriding weights_dict after init.
                             'FDI_FLOW': 1,
                             'FDI_ELAST': 5,
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
        return ['GPDIFF', 'GROWTH', 'KM','KM_DD_DD','KMCHEM','KMPHARMA','KMPHARMACHEM','KMPATENT','KM_GDP', 'OUT', 'RD',
                'RDPHARMA','RDCHEM','RDPHARMACHEM','RD_US','RD_RUS', 'RP', 
                'SPFLOWDOM', 'SPFLOW','SPFLOWDOM_US', 'SPFLOW_US','SDOMTFLOW','STFLOW','STFLOWSDOM',
                'SPFLOWDOM_RUS', 'SPFLOW_RUS','DOMPATUS','DOMPATEU','DOMPATINUS','DOMPATINCHN','DOMPATINEU',
                'SRDUS', 'SRGDP','SRGDP_US','SRGDP_RUS', 'JUPCOST','UUPCOST','UUPCOSTS','PCOST','PCOSTINTER',
                'PCOSTNOAGG','PCOSTINTERNOAGG','JUPCOSTRD', 'TP', 'Z','inter_TP', 
                'SINNOVPATEU','SINNOVPATUS','TO','TO_DD_DD','TOCHEM','TOPHARMA','TOPHARMACHEM','TOPATENT',
                'TE','TECHEM','TEPHARMA','TEPHARMACHEM','NUR','DOMPATRATUSEU','AGGAVMARKUP','AVMARKUPPHARCHEM',
                'SPATDEST','SPATORIG','TWSPFLOW','TWSPFLOWDOM','ERDUS','PROBINNOVENT',
                'SHAREEXPMON','SGDP','RGDPPC','SDFLOW','FDI_FLOW_N','FDI_FLOW','FDI_ELAST']
    
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
            self.KMPHARMACHEM_target = self.moments.loc['KMPHARMACHEM'].value
            self.KMPATENT_target = self.moments.loc['KMPATENT'].value
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
        try:
            self.PROBINNOVENT_target = self.moments.loc['PROBINNOVENT'].value 
        except:
            pass
        try:
            self.SHAREEXPMON_target = self.moments.loc['SHAREEXPMON'].value 
        except:
            pass
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
            self.TOPHARMACHEM_target = self.moments.loc['TOPHARMACHEM'].value
            self.TOPATENT_target = self.moments.loc['TOPATENT'].value
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
        self.DOMPATINCHN_target = self.cc_moments.loc[(3,3),'patent flows']/self.cc_moments.xs(3,level=0)['patent flows'].sum()
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
                
        try:
            # FDI affiliate-sales matrix.  CSV uses 1-based integer country
            # codes (File_ccode = origin, Rep_ccode = destination)
            # positionally matching self.countries.  Diagonal zeroed (model
            # X^{M,F}_{njs} is foreign affiliates only, n != j).
            _fdi = pd.read_csv('data/fdi_longformat_2015_AAMNE.csv')
            _piv = (_fdi.pivot_table(index='Rep_ccode', columns='File_ccode',
                                     values='FileToRep_Flow', aggfunc='sum')
                       .reindex(index=range(1, N+1),
                                columns=range(1, N+1))
                       .fillna(0.0))
            fdi_matrix = _piv.values / self.unit       # [dest, origin]
            np.fill_diagonal(fdi_matrix, 0.0)
            self.fdi_matrix = fdi_matrix
            # FDI_FLOW_N target (sector 1):
            #   sum_i X^{M,F}_{ni} / (X_{nn} - sum_i X^{M,F}_{ni})
            trade_flows_mat = self.ccs_moments.trade.values.reshape(N, N, S)
            X_nn = np.einsum('nns->n', trade_flows_mat[:, :, 1:2]).squeeze()
            fdi_sum_n = fdi_matrix.sum(axis=1)
            denom = X_nn / self.unit - fdi_sum_n
            self.FDI_FLOW_N_target = np.maximum(
                np.where(denom > 0, fdi_sum_n / denom, 1e-6), 1e-6)
            # FDI_FLOW target (bilateral, sector 1):
            #   X^{M,F}_{ni} / (X_{nn} - sum_j X^{M,F}_{nj})
            # Same denominator as FDI_FLOW_N (net domestic absorption);
            # bilateral numerator. Diagonal forced to 1.0 (matches model-side
            # convention in compute_FDI_FLOW) so that log(1/1)=0 contributes
            # zero to the residual without NaNs.
            safe_denom = np.where(denom > 0, denom, 1.0)
            fdi_flow_target = np.where(
                denom[:, None] > 0,
                fdi_matrix / safe_denom[:, None],
                1e-6)
            fdi_flow_target = np.maximum(fdi_flow_target, 1e-6)
            np.fill_diagonal(fdi_flow_target, 1.0)
            self.FDI_FLOW_target = fdi_flow_target
            # ── FDI_FLOW_mask  (N, N) bool ─────────────────────────────────
            # True  -> include this cell in the calibration residual
            # False -> exclude (set both model and target to 1.0 at
            #          compute_FDI_FLOW time, so log(1/1)=0 contributes
            #          nothing).
            # Two natural reasons to exclude:
            #   (a) the cell is on the diagonal (n == i, no domestic FDI by
            #       construction);
            #   (b) the data is "effectively zero": fdi_matrix[n, i] is at
            #       the noise floor (raw value below FDI_FLOW_zero_threshold
            #       times its row sum, or the denominator was non-positive
            #       so we floored the target to 1e-6).
            # The threshold (set by set_fdi_flow_zero_threshold below) is 0
            # by default => only the diagonal is masked, preserving prior
            # behaviour. Setting a positive threshold drops near-zero data
            # cells from the residual; the corresponding bilateral-a entries
            # should also be dropped from calibration via
            # parameters.mask_a_from_fdi_flow_mask(m).
            self.FDI_FLOW_mask = np.ones((N, N), bool)
            np.fill_diagonal(self.FDI_FLOW_mask, False)
            # Store the raw data-implied flow ratio (pre-floor, pre-diag
            # rewrite) so the data-zero detection can use it later. This is
            # what would have gone into FDI_FLOW_target before flooring.
            self._FDI_FLOW_raw_data = np.where(
                denom[:, None] > 0,
                fdi_matrix / safe_denom[:, None], 0.0)
            np.fill_diagonal(self._FDI_FLOW_raw_data, 0.0)
            # FDI_ELAST target: Blonigen (2002), semi-elasticity = 0.08
            self.FDI_ELAST_target = np.array([0.08])
        except:
            pass
        
            
        self.idx = {'GPDIFF':pd.Index(['scalar']), 
                    'GROWTH':pd.Index(['scalar']), 
                    'KM':pd.Index(['scalar']), 
                    'KM_DD_DD':pd.Index(['scalar']), 
                    'KMCHEM':pd.Index(['scalar']), 
                    'KMPHARMA':pd.Index(['scalar']), 
                    'KMPHARMACHEM':pd.Index(['scalar']), 
                    'KMPATENT':pd.Index(['scalar']), 
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
                    'objective':pd.Index(['scalar']),
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
                    'FDI_FLOW_N': pd.Index(self.countries, name='country'),
                    'FDI_FLOW': pd.MultiIndex.from_product(
                        [self.countries, self.countries],
                        names=['destination', 'origin']),
                    'FDI_ELAST': pd.Index(['scalar']),
                    'TO':pd.Index(['scalar']),
                    'TO_DD_DD':pd.Index(['scalar']),
                    'TE':pd.Index(['scalar']),
                    'TOPATENT':pd.Index(['scalar']),
                    'TOPHARMACHEM':pd.Index(['scalar']),
                    'TEPHARMACHEM':pd.Index(['scalar']),
                    'TOPHARMA':pd.Index(['scalar']),
                    'TEPHARMA':pd.Index(['scalar']),
                    'TOCHEM':pd.Index(['scalar']),
                    'TECHEM':pd.Index(['scalar']),
                    'DOMPATUS':pd.Index(['scalar']),
                    'DOMPATEU':pd.Index(['scalar']),
                    'DOMPATINUS':pd.Index(['scalar']),
                    'DOMPATINCHN':pd.Index(['scalar']),
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
            self.idx['DOMPATINCHN'] = pd.Index(self.sectors[1:], name='sector')
        
        self.shapes = {'SPFLOW':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM':(len(self.countries),len(self.countries)),
                       'SPFLOW_RUS':(len(self.countries),len(self.countries)-1),
                       'SPFLOWDOM_RUS':(len(self.countries),len(self.countries)),
                       'SDOMTFLOW':(len(self.countries),len(self.sectors)),
                       'STFLOWSDOM':(len(self.countries),len(self.countries),len(self.sectors)),
                       'TWSPFLOW':(len(self.countries),len(self.countries)-1),
                       'TWSPFLOWDOM':(len(self.countries),len(self.countries)),
                       # Bilateral FDI flow moment: (destination, origin). Used
                       # by moments.load_run when re-loading a saved checkpoint;
                       # without this entry the saved target stays as a flat
                       # (N*N,) vector and the broadcast in
                       # compute_moments_deviations against the (N, N) model
                       # value crashes.
                       'FDI_FLOW':(len(self.countries),len(self.countries)),
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
        
        if p.S==3:
            bracket = 1/(var.G[None,1:]+p.delta[:,1:]-p.nu[None,1:]) - 1/(var.G[None,1:]+p.delta[:,1:])
            KM = np.einsum('s,is,is,nis,nis,ns,is->nis',
                p.k[1:]/(p.k[1:]-1),
                p.eta[:,1:],
                var.l_R[:,1:]**(1-p.kappa),
                var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                var.profit[:,:,1:],
                bracket,
                1/(var.l_R[:,1:]+var.l_Ao[:,1:]+np.einsum('n,ins,i->is',var.w,var.l_Ae[:,:,1:],1/var.w))
                )
            self.KMPATENT = KM[0,0,0]
            self.KMPHARMACHEM = KM[0,0,1]
            
            # if self.aggregate_moments:
            self.KM = np.einsum('s,is,is,nis,nis,ns,i->ni',
                p.k[1:]/(p.k[1:]-1),
                p.eta[:,1:],
                var.l_R[:,1:]**(1-p.kappa),
                var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                var.profit[:,:,1:],
                bracket,
                1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                )[0,0]
        
        if p.S==4:
            bracket = 1/(var.G[None,1:]+p.delta[:,1:]-p.nu[None,1:]) - 1/(var.G[None,1:]+p.delta[:,1:])
            KM = np.einsum('s,is,is,nis,nis,ns,i->nis',
                p.k[1:]/(p.k[1:]-1),
                p.eta[:,1:],
                var.l_R[:,1:]**(1-p.kappa),
                var.psi_m_star[:,:,1:]**(1-p.k[None,None,1:]),
                var.profit[:,:,1:],
                bracket,
                1/(var.l_R[:,1:].sum(axis=1)+var.l_Ao[:,1:].sum(axis=1)+(var.w[:,None]*var.l_Ae[:,:,1:].sum(axis=2)/var.w[None,:]).sum(axis=0))
                )
            self.KMPATENT = KM[0,0,0]
            self.KMPHARMA = KM[0,0,1]
            self.KMCHEM = KM[0,0,2]
            
            # if self.aggregate_moments:
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
        # num_brack_E = var.PSI_MPD*eps(p.nu*delt)[None,None,:]
        num_brack_E = var.PSI_MPD*eps(p.delta*delt)[:,None,:]
        ##!!!!!!!!!!!!
        
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
        weights = (np.exp(-p.zeta*delt) * gamma((p.theta+1-p.sigma)/p.theta))[None,:]*var.sectoral_cons
        self.TO = self.turnover[0,1]
        if p.S==4:
            self.TOPHARMA = self.turnover[0,2]
            self.TOCHEM = self.turnover[0,3]
            self.TOPHARMACHEM = np.nan
        elif p.S==3:
            self.TOPHARMA = np.nan
            self.TOCHEM = np.nan
            self.TOPHARMACHEM = self.turnover[0,2]
            self.TOPATENT = self.turnover[0,1]
            self.TO = ((weights[:,1:]*num[:,1:]).sum(axis=1)/(weights[:,1:]*denom[:,1:]).sum(axis=1))[0]
        else:
            self.TOPHARMA = np.nan
            self.TOCHEM = np.nan
            self.TOPHARMACHEM = np.nan
        
        self.num_TO = num          
        self.num_TO_B = num_brack_B                             
        self.num_TO_C = num_brack_C                                                            
        self.num_TO_E = num_brack_E                                                     
        self.denom_TO = denom 
        
    # def compute_TO_DD_DD(self,var,p):
    #     delt = 5
    #     self.delta_t = delt
    #     PHI = var.phi**p.theta[None,None,:]
        
    #     num_brack_B = var.PSI_MNP*eps(p.nu_tilde*delt)[None,None,:]
    #     num_brack_C = var.PSI_MPND*(eps(p.delta_eff*delt)*eps(p.nu_tilde*delt)[None,None,:])
    #     num_brack_B = var.PSI_MNP*eps(p.nu_tilde*delt)[None,None,:]
    #     num_brack_C = var.PSI_MPND*(eps(p.delta_eff*delt)*eps(p.nu_tilde*delt)[None,None,:])
    #     num_brack_D = var.PSI_MPD*eps(p.delta_eff*delt)
    #     num_brack_E = var.PSI_MPL*(eps(p.delta_eff*delt)*eps(p.nu_tilde*delt)[None,None,:])
    #     num_brack_F = var.PSI_CL*eps(p.nu_tilde*delt)[None,None,:]
        
    #     num_brack = (num_brack_B + num_brack_C + num_brack_D + num_brack_E + num_brack_F)
        
    #     num_sum = np.einsum('nis,njs->ns',
    #                         num_brack,
    #                         PHI
    #                         ) - \
    #               np.einsum('ns,njs->ns',
    #                         np.diagonal(num_brack).transpose(),
    #                         PHI
    #                         ) - \
    #               np.einsum('nis,ns->ns',
    #                         num_brack,
    #                         np.diagonal(PHI).transpose()
    #                         ) + \
    #               np.einsum('ns,ns->ns',
    #                         np.diagonal(num_brack).transpose(),
    #                         np.diagonal(PHI).transpose()
    #                         )

    #     num = np.einsum('ns,ns->ns',
    #                     PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:],
    #                     num_sum
    #                     )
        
    #     denom_A = np.einsum('nis,ns,ns->nis',
    #                               PHI,
    #                               var.PSI_CD,
    #                               PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:]
    #                               )
        
    #     denom_B_a = var.PSI_MNP*np.exp(-delt*(p.nu+p.nu_tilde))[None,None,:]
    #     denom_B_b = var.PSI_MPND*(np.exp(-delt*p.delta_eff)
    #                               +np.exp(-(p.nu+p.nu_tilde)*delt)[None,None,:]*eps(delt*p.delta_eff))
    #     denom_B_c = (var.PSI_MPD+var.PSI_MPL)*np.exp(-delt*p.delta_eff)
    #     denom_B = np.einsum('nis,nis,s->nis',
    #                         denom_B_a + denom_B_b + denom_B_c,
    #                         var.phi**(p.sigma-1)[None,None,:],
    #                         (p.sigma/(p.sigma-1))**(1-p.sigma)
    #                         )
        
    #     denom_C_a = var.PSI_MNP*eps(p.nu)[None,None,:]
    #     denom_C_b = var.PSI_MPND*(eps(p.delta_eff*delt)*eps(p.nu*delt)[None,None,:])
    #     denom_C_c = var.PSI_MPL*eps(p.delta_eff*delt)
    #     denom_C = np.einsum('s,nis,nis,s->nis',
    #                         np.exp(-delt*p.nu_tilde),
    #                         denom_C_a + denom_C_b + denom_C_c,
    #                         var.phi**(p.sigma-1)[None,None,:],
    #                         (p.sigma/(p.sigma-1))**(1-p.sigma)
    #                         )
        
    #     denom_D_sum = np.einsum('nis,njs->nis',
    #                             num_brack,
    #                             PHI
    #                             ) - \
    #                   np.einsum('nis,ns->nis',
    #                             num_brack,
    #                             np.diagonal(PHI).transpose()
    #                             )
        
    #     denom_D = np.einsum('ns,nis->nis',
    #                     PHI.sum(axis=1)**((p.sigma-1)/p.theta-1)[None,:],
    #                     denom_D_sum
    #                     )
        
    #     denom = np.einsum('nis->ns',
    #                       denom_A + denom_B + denom_C + denom_D
    #                       ) - np.einsum('nns->ns',
    #                                         denom_A + denom_B + denom_C + denom_D
    #                                         ) 
        
    #     self.turnover_DD_DD = num/denom
    #     self.TO_DD_DD = self.turnover_DD_DD[0,1]
    
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
        
        self.num_TO_DD_DD = num                                
        self.num_TO_DD_DD_B = num_brack_B                             
        self.num_TO_DD_DD_C = num_brack_C                                
        self.num_TO_DD_DD_D = num_brack_D                                
        self.num_TO_DD_DD_E = num_brack_E                                
        self.num_TO_DD_DD_F = num_brack_F                                
        self.denom_TO_DD_DD = denom                            
                                        
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
            # !!! just changed it, it was commented
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
        self.DOMPATINCHN = var.pflow[3,3]/var.pflow[3,:].sum()
        
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
                res = ( p.k[1]*psi**(-p.k[1]-1)*np.min( inside_min[1:,0] )**(-p.d) )
                return res
            
            self.PROBINNOVENT = integrate.quad(integrand_US,1,np.inf)[0]
            
            def integrand_JAP(psi):
                inside_min = (aleph_P_star(psi) * (psi >= var.psi_m_star[...,1])) + (aleph_NP_star(psi) * (psi <= var.psi_m_star[...,1]))
                mask = np.ones(p.N)
                mask = (mask == 1)
                mask[2]=False
                res = ( p.k[1]*psi**(-p.k[1]-1)*np.min( inside_min[mask,2] )**(-p.d) )
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
        
    def compute_FDI_FLOW_N(self, var, p):
        """
        TeX (lines 5070-5087): destination-aggregated affiliate-sales ratio
            FDI_FLOW_N_n = sum_i X^{M,F}_{ni,s=1} / (X_{nn,s=1} - sum_i X^{M,F}_{ni,s=1})
        X_nn includes sum_i X^{M,F}_{ni} on its diagonal per eq 47, so the
        subtraction gives pure (X^{M,O}_{nn} + X^{CD}_{nn}) absorption.
        Floor at 1e-6 keeps log(.) finite in the calibration loss.
        """
        X_M_F_sum_n = var.X_M_F[:, :, 1:].sum(axis=(1, 2))
        X_nn = np.einsum('nns->n', var.X[:, :, 1:2]).squeeze()
        denom = X_nn - X_M_F_sum_n
        ratio = np.where(denom > 0, X_M_F_sum_n / denom, 1e-6)
        self.FDI_FLOW_N = np.maximum(ratio, 1e-6)

    def compute_FDI_FLOW(self, var, p):
        """
        Bilateral affiliate-sales ratio (sector 1):
            FDI_FLOW_{ni} = X^{M,F}_{ni,s=1}
                            / (X_{nn,s=1} - sum_j X^{M,F}_{nj,s=1})
        Same denominator as FDI_FLOW_N (net domestic absorption), bilateral
        numerator.

        Diagonal entries (n=i, no domestic FDI) are filled with 1.0 to match
        the FDI_FLOW_target diagonal convention (also 1.0); this prevents
        log(0/0) NaNs in the deviation builder while contributing exactly
        zero to the residual (since deviation = |log(1/1)| = 0).

        Off-diagonal cells masked out by self.FDI_FLOW_mask (e.g. data-zero
        cells set via set_fdi_flow_zero_threshold) are likewise rewritten on
        BOTH sides to 1.0 so they contribute zero to the residual without
        injecting NaNs.  The mask is True where the cell is INCLUDED.

        Sums to FDI_FLOW_N along origin axis (excluding the diagonal floor),
        providing a more granular target for bilateral FDI cost calibration.

        Output shape: (N, N) — matches FDI_FLOW_target loaded in load_data.
        """
        X_M_F_s1 = var.X_M_F[:, :, 1]                              # (N, N)
        X_M_F_sum_n = X_M_F_s1.sum(axis=1)                         # (N,)
        X_nn = np.einsum('nns->n', var.X[:, :, 1:2]).squeeze()     # (N,)
        denom = X_nn - X_M_F_sum_n                                 # (N,)
        # Safe-divide; floor at 1e-6 for entries with non-zero numerator
        safe_denom = np.where(denom > 0, denom, 1.0)
        ratio = np.where(denom[:, None] > 0,
                         X_M_F_s1 / safe_denom[:, None], 1e-6)     # (N, N)
        # Floor off-diagonal at 1e-6 (avoid log(0)); set diagonal to 1
        # to match the FDI_FLOW_target diagonal convention.
        ratio = np.maximum(ratio, 1e-6)
        np.fill_diagonal(ratio, 1.0)
        # Mask out excluded cells: rewrite to 1 on BOTH model and target so
        # the deviation log(1/1) is zero. We touch the target here (not at
        # load_data time) so that toggling the mask is non-destructive.
        if hasattr(self, 'FDI_FLOW_mask') and self.FDI_FLOW_mask is not None:
            ratio = np.where(self.FDI_FLOW_mask, ratio, 1.0)
            # Apply the same rewrite to the saved target.  Re-derive the
            # masked target on each call to keep it in sync with the mask
            # (in case the threshold was changed without re-loading data).
            if hasattr(self, '_FDI_FLOW_target_original'):
                self.FDI_FLOW_target = np.where(self.FDI_FLOW_mask,
                                                self._FDI_FLOW_target_original,
                                                1.0)
        self.FDI_FLOW = ratio

    def set_fdi_flow_zero_threshold(self, threshold):
        """
        Exclude bilateral FDI flow cells whose DATA value is below
        `threshold` from the FDI_FLOW calibration residual.

        Why: the model structurally cannot drive FDI_FLOW_{ni} down to the
        1e-6 floor for many pairs (the Pareto-tail leverage of the bilateral
        cost a^s_{ni} is bounded; see compute_entry_costs). Including the
        ~30 "data-zero" cells then amplifies noise and pulls the
        bilateral-a optimization away from the corridors where FDI is
        actually present.

        After this call:
          - self.FDI_FLOW_mask[n, i] = False on cells where the raw data
            FDI flow ratio is below `threshold` (or on the diagonal).
          - self._FDI_FLOW_target_original holds the un-masked target;
            compute_FDI_FLOW rewrites both model and target to 1.0 on the
            masked cells at call time.
          - Cells where the raw flow was at the safe-divide fallback
            (denom <= 0) are also masked (they were floored arbitrarily).

        To also drop the corresponding bilateral-a entries from
        calibration, call parameters.mask_a_from_fdi_flow_mask(self) on
        the matching parameters instance.

        Parameters
        ----------
        threshold : float
            Cells with raw data FDI ratio below this value are masked.
            Use 0 to keep only the diagonal masked (default).  A reasonable
            non-zero value is ~1e-4 (one fortieth the median target).
        """
        # Stash the pristine target the first time we touch it, so repeated
        # threshold changes are non-destructive.
        if not hasattr(self, '_FDI_FLOW_target_original'):
            self._FDI_FLOW_target_original = self.FDI_FLOW_target.copy()
        raw = self._FDI_FLOW_raw_data
        keep = (raw >= threshold)
        np.fill_diagonal(keep, False)
        self.FDI_FLOW_mask = keep
        # Refresh the target so it matches what compute_FDI_FLOW will use.
        self.FDI_FLOW_target = np.where(keep,
                                        self._FDI_FLOW_target_original, 1.0)
        n_kept = int(keep.sum())
        n_total = keep.size - self.N  # exclude diagonal from "total"
        print(f"[FDI_FLOW] threshold={threshold:.2e}: keeping {n_kept} of "
              f"{n_total} off-diagonal cells "
              f"({n_total - n_kept} masked as data-zero)")
        return keep


    def _vartheta_from_thresholds(self, p, c2, psi_mO, psi_mF, a_npf, a_npo,
                                   a_po, psi_bar, a_nis):
        """
        Probability of FDI by origin i in destination n, given thresholds.

        Implements eq (\\ref{eq:varthetaABs}) of algorithm_corrected.tex
        (7 terms A1, A2, A3, B1, B2, B3, B4 with beta-shape (d+1, k-d) and
        d-k exponents — matches RomerEKnewversionwithFDI3.tex lines 5298-5334).

        Inputs are arrays of shape (N, N, S-1). Pairs not in Case 2 return 0.
        This function is used twice: once with BASELINE thresholds for theta_n,
        and once with PERTURBED thresholds (eqs \\ref{eq:adag1}-\\ref{eq:adag2})
        for theta_n^dagger in the partial-equilibrium FDI elasticity.
        """
        k = p.k[1]; d = p.d
        h_fe = np.einsum('n,s->ns', p.r_hjort, p.fe[1:])[:, None, :]
        a_safe = np.where(a_nis > 0, a_nis, 1.0)
        psi_bar_safe = np.where((psi_bar > 0) & np.isfinite(psi_bar), psi_bar, 1.0)

        # NP-O -> NP-F integrals
        A1 = np.where(
            c2 & (a_npf >= psi_mF),
            k * a_npf**(-d) / (k - d) * (1 - psi_mF**(d - k)),
            0.0)
        A2 = np.where(
            c2 & (a_npf > 1) & (a_npf < psi_mF),
            k * a_npf**(-d) / (k - d) * (1 - a_npf**(d - k))
            + a_npf**(-k) - psi_mF**(-k),
            0.0)
        A3 = np.where(
            c2 & (a_npf <= 1),
            1 - psi_mF**(-k),
            0.0)

        # NP-O -> P-F (transition strip)
        B1 = np.where(
            c2 & (a_npo < psi_mO),
            np.maximum(a_npo, psi_mF)**(-k) - psi_mO**(-k),
            0.0)

        # NP-O -> P-F beta term
        ind_B2 = c2 & (a_npo >= psi_mF)
        upper_min = np.minimum(a_npo, psi_mO)
        t1_B2 = np.where(ind_B2, 1 - psi_bar_safe / psi_mF, 0.0)
        t2_B2 = np.where(ind_B2,
                         1 - psi_bar_safe / np.where(upper_min > 0, upper_min, 1.0),
                         0.0)
        B_B2 = np.where(ind_B2, _betainc_vec(d + 1, k - d, t1_B2, t2_B2), 0.0)
        B2 = np.where(
            ind_B2 & (a_nis > 0),
            (h_fe / a_safe)**d * k / psi_bar_safe**k * B_B2,
            0.0)

        # P-O -> P-F region
        max_top = np.maximum(psi_mO, a_po)
        B3 = np.where(
            c2,
            k / (k - d) * (psi_mO**(d - k) - max_top**(d - k)) * a_po**(-d),
            0.0)

        # Tail (a^{PF,PO} above max)
        B4 = np.where(c2, max_top**(-k), 0.0)

        vartheta = A1 + A2 + A3 + B1 + B2 + B3 + B4
        vartheta = np.where(c2, vartheta, 0.0)
        vartheta = np.where(np.isfinite(vartheta), vartheta, 0.0)
        return np.clip(vartheta, 0.0, 1.0)

    def _vartheta_ni(self, var, p):
        """Baseline FDI probability — uses var's pre-computed thresholds."""
        return self._vartheta_from_thresholds(
            p,
            c2     = var.case2[..., 1:],
            psi_mO = var.psi_m_star_O[..., 1:],
            psi_mF = var.psi_m_star_F[..., 1:],
            a_npf  = var.a_NPF_NPO[..., 1:],
            a_npo  = var.a_PF_NPO[..., 1:],
            a_po   = var.a_PF_PO[..., 1:],
            psi_bar= var.psi_bar_NPO_PF[..., 1:],
            a_nis  = var.a[..., 1:])

    def _vartheta_ni_perturbed(self, var, p, n_idx, delta_b):
        """
        Partial-equilibrium perturbed FDI probability.

        Implements section "FDI tariff semi-elasticity --- partial equilibrium"
        of algorithm_corrected.tex (eqs \\ref{eq:lamdag}-\\ref{eq:varthetaABs}).

        Tariff b_{n,i,s=1} -> b_{n,i,s=1} + delta_b for destination n=n_idx,
        for all origins i, s=1. All baseline aggregates (wages, expenditures,
        qualities, prices, phi, pi_F) are held fixed at var. Only the export-
        side per-quality profit pi^{w,dag}_{nis} = lambda^dag_{ni} * pi^w_{nis}
        with lambda^dag_{ni} = ((1+b+delta_b)/(1+b))^{-sigma_s} changes.

        That rescales V^{NP,dag}, V^{P,dag}, V^{P,D,dag} (export-side V's),
        the Case 2 indicator, and the four auxiliary thresholds. Affiliate-
        side V^{P,F}, V^{NP,F}, V^{P,D,F} are unchanged.

        Returns vartheta^dag of shape (N, N, S-1) — only column n_idx is
        meaningful; other destinations are NOT perturbed and return baseline.
        """
        sigma_s = p.sigma[1:]                    # (S-1,)
        k = p.k[1]; d = p.d

        # ---- Tariff scaling lambda^dag ----
        # Apply ONLY to destination n_idx, all origins, sector 1.
        # Build full (N, N, S-1) lambda where most entries are 1 (no perturbation),
        # and entries with n == n_idx get the scaling factor.
        b_old = p.tariff[..., 1:]                              # (N, N, S-1)
        # Only destination n_idx, all origins i, gets the perturbation
        b_new = b_old.copy()
        b_new[n_idx, :, :] = b_old[n_idx, :, :] + delta_b
        # F1 (ChatGPT report): the perturbation target is origins i != n_idx;
        # leave the own-origin diagonal entry unperturbed so it doesn't
        # contaminate the recomputed psi^{o*,dag} via the diagonal pair's
        # Case-1 LHS contribution.
        b_new[n_idx, n_idx, :] = b_old[n_idx, n_idx, :]
        lam_dag = ((1 + b_new) / (1 + b_old))**(-sigma_s[None, None, :])  # (N, N, S-1)

        # ---- Perturbed export-side profit: pi^{w,dag}_{nis} = lambda^dag * pi^w_{nis} ----
        # var.profit shape (N, N, S); slice sector 1.
        profit_dag = lam_dag * var.profit[..., 1:]            # (N, N, S-1)

        # ---- Perturbed V's (export side only; FDI side unchanged) ----
        # Match var_with_fdi.compute_V: V_NP = profit * w_i / G; V_P_F similar with w_n.
        # G shape (S-1,) for sector >= 1
        G = var.G[1:]                                          # (S-1,)
        # G_post_P_D = G + delta - nu (denominator (G - nu + delta) in compute_V)
        # Use the same formula as compute_V:
        #   V_NP = profit * w_i / G
        #   V_P  = profit * w_i * (1/(G-nu+delta) - 1/(G+delta) + 1/G)
        # The perturbed V's just rescale these by lambda^dag.
        w_i = var.w[None, :, None]                             # (1, N, 1) — broadcast over s
        V_NP_dag = profit_dag * w_i / G[None, None, :]
        # V_P_dag uses the same triple-term coefficient as baseline V_P
        coef_P = (1.0 / (G[None, None, :] - p.nu[1:][None, None, :]
                          + p.delta[:, 1:][:, None, :])
                  - 1.0 / (G[None, None, :] + p.delta[:, 1:][:, None, :])
                  + 1.0 / G[None, None, :])
        V_P_dag = profit_dag * w_i * coef_P

        # Baseline FDI-side V's (UNCHANGED in partial equilibrium)
        V_NP_F = var.V_NP_F[..., 1:]
        V_P_F  = var.V_P_F[..., 1:]

        # ---- Perturbed Case 2 indicator ----
        # Case 2: tilde_w_n * Pi^{w,F}_{nis} > tilde_w_i * pi^{w,dag}_{nis}
        # var.pi_F is Pi^{w,F}_{nis}; var.profit is pi^w_{nis}; profit_dag is the scaled version.
        off_diag = np.ones((p.N, p.N, p.S-1), dtype=bool)
        for n in range(p.N): off_diag[n, n, :] = False
        c2_dag = (
            (var.w[:, None, None] * var.pi_F[..., 1:]
             > var.w[None, :, None] * profit_dag)
            & off_diag)

        # ---- Perturbed auxiliary thresholds (eqs adag1-adag2) ----
        # a_NPF_NPO_dag = w_n * a_nis / (V_NP_F - V_NP_dag)
        # a_PF_PO_dag   = w_n * a_nis / (V_P_F  - V_P_dag)
        # a_PF_NPO_dag  = w_n * (a_nis + h_n fe_s) / (V_P_F - V_NP_dag)
        # psi_bar_dag   = w_n * h_n fe_s         / (V_P_F - V_NP_dag)
        a_nis = var.a[..., 1:]
        w_n = var.w[:, None, None]
        h_fe = np.einsum('n,s->ns', p.r_hjort * var.w, p.fe[1:])[:, None, :]
        # ^ matches compute_auxiliary_thresholds convention: w_n * h_n * fe_s

        def _thr(num, den):
            return np.where(np.abs(den) > 0,
                            num / np.where(np.abs(den) > 0, den, 1.0),
                            np.inf)

        a_NPF_NPO_dag = _thr(w_n * a_nis, V_NP_F - V_NP_dag)
        a_PF_PO_dag   = _thr(w_n * a_nis, V_P_F  - V_P_dag)
        a_PF_NPO_dag  = _thr(w_n * a_nis + h_fe, V_P_F - V_NP_dag)
        psi_bar_dag   = _thr(h_fe, V_P_F - V_NP_dag)

        # ---- Perturbed Case-1/Case-2 patenting cutoffs psi^{*,O,dag}, psi^{*,F,dag} ----
        # For destination n != US: V's are unchanged, so these equal baseline.
        # For destination n = US: V_NP, V_P change to V_NP_dag, V_P_dag.
        # FDI-side V's (V_P_F, V_NP_F) are unchanged at all destinations.
        w_fe_h_nis = np.einsum('n,n,s->ns', var.w, p.r_hjort,
                                p.fe[1:])[:, None, :]   # (N, 1, S-1)
        psi_star_O_dag = var.psi_star_O[..., 1:].copy()
        psi_star_F_dag = var.psi_star_F[..., 1:].copy()
        # Update only the n=US row using daggered V's.
        c2_us = c2_dag[n_idx, :, :]                     # (N, S-1)

        # F1 fix (ChatGPT report): for US-row pairs that stay in Case 1 after
        # the perturbation, the algorithm/TeX requires the Case-1 analytically-
        # continued cutoff (eq:psiC1) evaluated with pi^{w,dag}, NOT inf.
        # Setting it to inf drops the Case-1 export-patenting contribution
        # to the perturbed psi^{o*,dag} LHS root-solve, which propagates to
        # vartheta_dag at OTHER destinations for those origins.
        #
        # Case-1 perturbed cutoff (mirrors compute_patenting_thresholds line 1949):
        #   psi^{C,O,dag}_{n_idx, i, s} = w_n h_n f^e_s
        #     / (pi^{w,dag}_{n_idx,i,s} * w_i
        #        * (1/(G_s + delta_{n_idx,s} - nu_s) - 1/(G_s + delta_{n_idx,s})))
        delta_us = p.delta[n_idx, 1:]                      # (S-1,)
        bracket_us = (1.0 / (G - p.nu[1:] + delta_us)
                      - 1.0 / (G + delta_us))               # (S-1,)
        # profit_dag[n_idx, :, :] is (N, S-1) over origins; var.w is (N,)
        psi_C1_O_us_denom = (profit_dag[n_idx, :, :]        # (N, S-1)
                             * var.w[:, None]               # (N, 1) = w_i
                             * bracket_us[None, :])         # (1, S-1)
        psi_C1_O_us = np.where(
            np.abs(psi_C1_O_us_denom) > 0,
            w_fe_h_nis[n_idx, 0, :][None, :]
              / np.where(np.abs(psi_C1_O_us_denom) > 0, psi_C1_O_us_denom, 1.0),
            np.inf)

        # Case-2 perturbed cutoff (unchanged formula)
        psi_C2_O_us = np.where(
            c2_us,
            w_fe_h_nis[n_idx, 0, :][None, :]
              / (V_P_dag[n_idx, :, :] - V_NP_dag[n_idx, :, :] + 1e-30),
            np.inf)

        # Mixed assignment per eq:psib: max(psi^{C,O}, 1) with the case-
        # appropriate analytical continuation
        psi_star_O_us = np.where(c2_us, psi_C2_O_us, psi_C1_O_us)
        psi_star_O_us = np.maximum(psi_star_O_us, 1.0)
        psi_star_O_dag[n_idx, :, :] = psi_star_O_us

        # FIX 2.2: per algorithm eq (12), the FDI patenting cutoff compares
        # V^{P,F} vs V^{NP,F} — BOTH unchanged by the export-tariff shock
        # (the perturbation touches export profits only, not FDI profits).
        # Case 1 has no psi^{*,F} (no FDI), so leave it inf (default from baseline
        # where Case-1 pairs already have inf via compute_patenting_thresholds).
        psi_C2_F_us = np.where(
            c2_us,
            w_fe_h_nis[n_idx, 0, :][None, :]
              / (V_P_F[n_idx, :, :] - V_NP_F[n_idx, :, :] + 1e-30),
            np.inf)
        psi_C2_F_us = np.maximum(psi_C2_F_us, 1.0)
        psi_star_F_dag[n_idx, :, :] = psi_C2_F_us

        # ---- Perturbed psi^{o*,dag}_{is}: root-find with only n=US daggered ----
        # Mixed (N, N, S-1) arrays: baseline for n != US, daggered for n == US.
        def _mix_us(base_arr, dag_arr):
            """Replace n=US slice of base_arr with dag_arr's n=US slice."""
            out = base_arr.copy()
            out[n_idx, :, :] = dag_arr[n_idx, :, :]
            return out

        V_NP_mix    = _mix_us(var.V_NP[..., 1:],      V_NP_dag)
        V_P_mix     = _mix_us(var.V_P[..., 1:],       V_P_dag)
        # FDI-side V's are baseline at every n (no dagger):
        V_NP_F_mix  = var.V_NP_F[..., 1:]
        V_P_F_mix   = var.V_P_F[..., 1:]
        # Thresholds (a's and psi_bar): baseline for n != US, daggered for n == US
        c2_mix         = _mix_us(var.case2[..., 1:],            c2_dag)
        psi_star_O_mix = _mix_us(var.psi_star_O[..., 1:],       psi_star_O_dag)
        psi_star_F_mix = _mix_us(var.psi_star_F[..., 1:],       psi_star_F_dag)
        a_NPF_NPO_mix  = _mix_us(var.a_NPF_NPO[..., 1:],        a_NPF_NPO_dag)
        a_PF_NPO_mix   = _mix_us(var.a_PF_NPO[..., 1:],         a_PF_NPO_dag)
        a_PF_PO_mix    = _mix_us(var.a_PF_PO[..., 1:],          a_PF_PO_dag)
        psi_bar_mix    = _mix_us(var.psi_bar_NPO_PF[..., 1:],   psi_bar_dag)

        psi_o_star_dag = self._solve_psi_o_star_perturbed(
            var, p, c2_mix, psi_star_O_mix, psi_star_F_mix,
            a_NPF_NPO_mix, a_PF_NPO_mix, a_PF_PO_mix,
            psi_bar_mix, V_NP_mix, V_NP_F_mix, V_P_mix, V_P_F_mix)

        # ---- psi^{m*,O/F,dag}_{nis} = max(psi^{*,O/F,dag}, psi^{o*,dag}_{is}) ----
        # psi_o_star_dag shape: (N, S-1)
        psi_mO = np.maximum(psi_star_O_mix, psi_o_star_dag[None, :, :])
        psi_mF = np.maximum(psi_star_F_mix, psi_o_star_dag[None, :, :])

        return self._vartheta_from_thresholds(
            p, c2=c2_dag,
            psi_mO=psi_mO, psi_mF=psi_mF,
            a_npf=a_NPF_NPO_dag, a_npo=a_PF_NPO_dag, a_po=a_PF_PO_dag,
            psi_bar=psi_bar_dag, a_nis=a_nis)

    def _solve_psi_o_star_perturbed(
            self, var, p, c2_b, psi_O_b, psi_F_b,
            a_npf_b, a_pf_npo_b, a_pf_po_b, psi_bar_b,
            V_NP_b, V_NP_F_b, V_P_b, V_P_F_b):
        """
        Root-solve for psi^{o*,dag}_{is} given MIXED baseline/perturbed
        per-(n, i, s) constants. Mirrors compute_patenting_thresholds's
        lhs_minus_rhs_vec but with arrays passed in (not self.*).
        """
        from scipy.optimize import root
        d = p.d; k = p.k[1]

        w_a_b      = (var.w[:, None, None] * var.a[..., 1:])
        w_n_fe_h_b = np.einsum('n,n,s->ns', var.w, p.r_hjort,
                                p.fe[1:])[:, None, :]

        def _eps_NPO_NPF_v(psi_b):
            mask = np.isfinite(a_npf_b)
            return np.where(mask & (psi_b > 0),
                            np.maximum(1.0, a_npf_b
                                       / np.where(psi_b > 0, psi_b, 1.0)),
                            1.0)

        def _eps_NPO_PF_v(psi_b):
            denom = psi_b - psi_bar_b
            num = a_pf_npo_b - psi_bar_b
            ok = np.isfinite(a_pf_npo_b) & np.isfinite(psi_bar_b) & (denom > 0)
            ratio = np.where(ok, num / np.where(denom > 0, denom, 1.0), 1.0)
            return np.where(ok, np.maximum(1.0, ratio), 1.0)

        def _eps_PO_PF_v(psi_b):
            ok = np.isfinite(a_pf_po_b)
            return np.where(ok & (psi_b > 0),
                            np.maximum(1.0, a_pf_po_b
                                       / np.where(psi_b > 0, psi_b, 1.0)),
                            1.0)

        def _ValmidPs_v(psi_b):
            e = _eps_NPO_PF_v(psi_b)
            return (psi_b * V_NP_b * (1 - e**(-d))
                    + (psi_b * V_P_F_b - w_n_fe_h_b) * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def _ValhighPs_v(psi_b):
            e = _eps_PO_PF_v(psi_b)
            return ((psi_b * V_P_b - w_n_fe_h_b) * (1 - e**(-d))
                    + (psi_b * V_P_F_b - w_n_fe_h_b) * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def _ValnoorigPs_v(psi_b):
            e = _eps_NPO_NPF_v(psi_b)
            return (psi_b * V_NP_b * (1 - e**(-d))
                    + psi_b * V_NP_F_b * e**(-d)
                    - d/(d+1) * w_a_b * e**(-d-1))

        def lhs_minus_rhs_vec(psi_arr):
            psi_b = psi_arr[None, :, :]
            case1_active = (~c2_b) & (psi_b >= psi_O_b)
            case1_term = case1_active * (psi_b * (V_P_b - V_NP_b) - w_n_fe_h_b)

            c2_mid_active   = c2_b & (psi_b >= psi_F_b) & (psi_b < psi_O_b)
            c2_high_active  = c2_b & (psi_b >= psi_O_b)
            c2_noorig_active = c2_b & (psi_b >= psi_F_b)

            psi_b_safe = np.where(psi_b > 0, psi_b, 1e-30)
            psi_b3 = np.broadcast_to(psi_b_safe, (p.N, p.N, p.S-1))

            mid_v    = _ValmidPs_v(psi_b3)
            high_v   = _ValhighPs_v(psi_b3)
            noorig_v = _ValnoorigPs_v(psi_b3)

            terms = (case1_term
                     + c2_mid_active   * mid_v
                     + c2_high_active  * high_v
                     - c2_noorig_active * noorig_v)
            terms = np.where(np.isfinite(terms), terms, 0.0)
            lhs = terms.sum(axis=0)

            rhs = (var.w * p.r_hjort)[:, None] * p.fo[None, 1:]
            return lhs - rhs

        # Start from baseline psi_o_star
        psi_o_star = np.full((p.N, p.S - 1), 1.0)
        psi_o_baseline = var.psi_o_star[:, 1:]

        # Step 1: check whether psi^{o*}=1 at this perturbed system
        eq1 = lhs_minus_rhs_vec(np.ones((p.N, p.S - 1)))
        need_solve = (eq1 < 0)

        if need_solve.any():
            x0 = np.where(np.isfinite(psi_o_baseline) & (psi_o_baseline > 1.0),
                          psi_o_baseline, 1.0).ravel()

            def func_to_solve(psi_flat):
                psi_arr = psi_flat.reshape(p.N, p.S - 1).copy()
                psi_arr = np.where(need_solve, psi_arr, 1.0)
                psi_arr = np.maximum(psi_arr, 1.0)
                res = lhs_minus_rhs_vec(psi_arr)
                res = np.where(need_solve, res, 0.0)
                return res.ravel()
            try:
                sol = root(func_to_solve, x0=x0, tol=1e-8)
                psi_sol = sol.x.reshape(p.N, p.S - 1)
                psi_sol = np.where(need_solve, np.maximum(psi_sol, 1.0), 1.0)
            except Exception:
                psi_sol = np.where(need_solve,
                                   x0.reshape(p.N, p.S - 1), 1.0)
            psi_o_star = psi_sol

        return psi_o_star

    def _theta_n(self, var, p, n_idx, vartheta=None):
        """
        Aggregated FDI propensity for destination n (sector 1, the FDI sector).

        Implements eq (\\ref{eq:thetabar}) of algorithm_corrected.tex:
            bar_vartheta_n = sum_{i!=n} eta_i (L^R_i)^{1-kappa} vartheta_{ni}
                              / sum_{i!=n} eta_i (L^R_i)^{1-kappa}

        Weights are baseline R&D shares. Setting vartheta=None uses baseline
        thresholds via _vartheta_ni; pass a precomputed vartheta (e.g. the
        perturbed one) to evaluate at non-baseline thresholds.
        """
        if vartheta is None:
            vartheta = self._vartheta_ni(var, p)
        eta_lR   = p.eta[..., 1:] * var.l_R[..., 1:]**(1 - p.kappa)
        w = np.broadcast_to(eta_lR[None, :, :], vartheta.shape).copy()
        w[n_idx, n_idx, :] = 0.0   # exclude self
        num = (vartheta[n_idx, :, :] * w[n_idx, :, :]).sum(axis=0)
        den = w[n_idx, :, :].sum(axis=0)
        theta = np.where(den > 0, num / den, 0.0)
        return float(theta[0])

    def compute_FDI_ELAST(self, var, p):
        """
        PARTIAL-EQUILIBRIUM FDI tariff semi-elasticity.

        Implements section "FDI tariff semi-elasticity --- partial equilibrium"
        of algorithm_corrected.tex, eq (\\ref{eq:FDIELAST}):

            FDI_ELAST_n = (bar_vartheta^dagger_n - bar_vartheta_n) / Delta_b

        with n = US, Delta_b = 0.1, sector s = 1, applied to all origins i.

        Procedure:
        1. baseline vartheta_{ni} from var (no perturbation).
        2. Apply tariff perturbation b_{n,i,1} -> b_{n,i,1} + Delta_b
           ONLY to destination n = US, ALL origins, sector 1.
        3. Recompute the export-side V's (V^NP, V^P, V^{P,D}) by scaling
           pi^w with lambda^dag = ((1+b+dag)/(1+b))^{-sigma}. The FDI-side
           V's (V^{NP,F}, V^{P,F}, V^{P,D,F}) are UNCHANGED.
        4. Recompute Case 2 indicator and the four auxiliary thresholds
           (a_NPF_NPO, a_PF_PO, a_PF_NPO, psi_bar) with the perturbed V's.
        5. Recompute Case-2 patenting cutoffs (psi^{*,O}, psi^{*,F}) for
           the n=US row from the perturbed thresholds, and re-solve the
           original-patent cutoff psi^{o*}_{is} with ONLY the n=US destination
           term daggered (partial-equilibrium recomputation per algorithm
           p. 9 just below eq 77).  The full entry-cost cutoffs
           psi^{m*,O,dag} = max(psi^{*,O,dag}, psi^{o*,dag}) and similarly
           for psi^{m*,F,dag} are then computed.  Evaluate vartheta^dag at
           these perturbed thresholds.
        6. Aggregate both vartheta and vartheta^dag with baseline R&D weights
           and take the level difference / Delta_b.

        NO full GE re-solve — just a threshold rebuild from the same var.
        """
        us_idx = 0
        delta_b = 0.1

        # Baseline aggregated FDI propensity
        vartheta_base = self._vartheta_ni(var, p)
        theta_base = self._theta_n(var, p, us_idx, vartheta=vartheta_base)

        # Perturbed aggregated FDI propensity (column n=US only is affected)
        vartheta_pert = self._vartheta_ni_perturbed(var, p, us_idx, delta_b)
        theta_pert = self._theta_n(var, p, us_idx, vartheta=vartheta_pert)

        if np.isfinite(theta_pert) and np.isfinite(theta_base):
            self.FDI_ELAST = np.array([(theta_pert - theta_base) / delta_b])
        else:
            self.FDI_ELAST = np.array([0.0])


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
            try:
                self.compute_TO_DD_DD(var,p)
            except:
                pass
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
            try:
                self.compute_FDI_FLOW_N(var, p)
            except:
                pass
            try:
                self.compute_FDI_FLOW(var, p)
            except:
                pass
            try:
                self.compute_FDI_ELAST(var, p)
            except:
                pass
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
                if mom != 'GPDIFF' and mom != 'TO' and mom != 'TE' and mom != 'GROWTH' and mom != 'OUT' and mom != 'SPFLOW' and mom != 'SPFLOWDOM' and mom != 'FDI_FLOW':
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


                elif mom == 'SPFLOW' or mom == 'SPFLOWDOM' or mom == 'FDI_FLOW':
                    # Safe denominator for the distortion factor: at masked /
                    # diagonal cells the target is rewritten to 1.0, so
                    # log(target) == 0 and the naive (1 + c/|log(target)|)
                    # produces inf, then 0 * inf = NaN in the deviation.
                    # Replace the |log(target)| under the bar with 1.0 wherever
                    # it is < 1e-12, so the distortion factor becomes (1 + c)
                    # there and the deviation stays a clean zero (since the
                    # |log(model/target)| factor is already zero on those cells).
                    _log_tgt = np.log(getattr(self, mom + '_target'))
                    _safe_log_tgt = np.where(np.abs(_log_tgt) > 1e-12,
                                              np.abs(_log_tgt), 1.0)
                    if self.loss == 'log':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    *(1+distort_for_large_pflows_fac/_safe_log_tgt)
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(np.log(getattr(self,mom)/getattr(self,mom+'_target')))
                                    *(1+distort_for_large_pflows_fac/_safe_log_tgt)
                                    /getattr(self,mom+'_target')
                                    )
                    if self.loss == 'ratio':
                        if self.dim_weight == 'lin':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))
                                    *(1+distort_for_large_pflows_fac/_safe_log_tgt)
                                    /getattr(self,mom+'_target')
                                    /getattr(self,mom+'_target').size**(1/2)
                                    )
                        if self.dim_weight == 'sqr':
                            setattr(self,
                                    mom+'_deviation',
                                    self.weights_dict[mom]*np.abs(getattr(self,mom)-getattr(self,mom+'_target'))
                                    *(1+distort_for_large_pflows_fac/_safe_log_tgt)
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

