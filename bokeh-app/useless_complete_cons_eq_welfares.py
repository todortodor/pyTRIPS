#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:42:43 2023

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
from tqdm import tqdm
import matplotlib.pylab as pylab


baseline_dics = [
    {'baseline':'501',
                      'variation':'2.0.0'},
    {'baseline':'501',
                      'variation':'2.0.3'},
    {'baseline':'501',
                      'variation':'2.0.4'},
    {'baseline':'501',
                      'variation':'2.0.5'},
    {'baseline':'501',
                      'variation':'2.0.6'},
    {'baseline':'501',
                      'variation':'2.0.7'},
    {'baseline':'501',
                      'variation':'2.0.8'},
    {'baseline':'501',
                      'variation':'2.0.9'},
    {'baseline':'501',
                      'variation':'2.0.10'},
    {'baseline':'501',
                      'variation':'2.0.11'},
    {'baseline':'501',
                      'variation':'2.0.12'},
    {'baseline':'501',
                      'variation':'2.0.13'},
    {'baseline':'501',
                      'variation':'2.0.14'},
    # {'baseline':'501',
    #                   'variation':'2.0.15'},
    # {'baseline':'501',
    #                   'variation':'2.0.16'},
    # {'baseline':'501',
    #                   'variation':'2.0.17'},
    # {'baseline':'501',
    #                   'variation':'2.0.18'},
    # {'baseline':'501',
    #                   'variation':'2.0.19'},
    # {'baseline':'501',
    #                   'variation':'2.0.20'},
    ]

nash_deltas = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline','variation'],keep='last')
nash_deltas['variation'] = nash_deltas['variation'].astype('str')
nash_deltas['baseline'] = nash_deltas['baseline'].astype('str')
coop_negishi_deltas = (pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
                       .drop_duplicates(['baseline','variation','aggregation_method'],keep='last'))
coop_negishi_deltas = coop_negishi_deltas.loc[coop_negishi_deltas.aggregation_method == 'negishi']
coop_negishi_deltas['variation'] = coop_negishi_deltas['variation'].astype('str')
coop_negishi_deltas['baseline'] = coop_negishi_deltas['baseline'].astype('str')
coop_equal_deltas = (pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
                       .drop_duplicates(['baseline','variation','aggregation_method'],keep='last'))
coop_equal_deltas = coop_equal_deltas.loc[coop_equal_deltas.aggregation_method == 'pop_weighted']
coop_equal_deltas['variation'] = coop_equal_deltas['variation'].astype('str')
coop_equal_deltas['baseline'] = coop_equal_deltas['baseline'].astype('str')

# for equilibrium in ['nash_eq','coop_negishi_eq','coop_equal_eq']:
for equilibrium in ['nash_eq']:
    
    recaps_path = f'counterfactual_recaps/around_dyn_{equilibrium}/'
    
    try:
        os.mkdir(recaps_path)
    except:
        pass
    
    for baseline_dic in baseline_dics:
        
        if baseline_dic['variation'] == 'baseline':
            baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
        else:
            baseline_path = \
                f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        print(baseline_path)
                
        p_baseline = parameters(n=7,s=2)
        p_baseline.load_data(baseline_path)
        
        if equilibrium == 'nash_eq':
            deltas_of_equilibrium = nash_deltas.loc[(nash_deltas.baseline == baseline_dic["baseline"])
                                                    &(nash_deltas.variation == baseline_dic["variation"])
                                                    ][p_baseline.countries].values.squeeze()
        if equilibrium == 'coop_negishi_eq':
            deltas_of_equilibrium = coop_negishi_deltas.loc[(coop_negishi_deltas.baseline == baseline_dic["baseline"])
                                                            &(coop_negishi_deltas.variation == baseline_dic["variation"])
                                                            ][p_baseline.countries].values.squeeze()
        if equilibrium == 'coop_equal_eq':
            deltas_of_equilibrium = coop_equal_deltas.loc[(coop_equal_deltas.baseline == baseline_dic["baseline"])
                                                            &(coop_equal_deltas.variation == baseline_dic["variation"])
                                                            ][p_baseline.countries].values.squeeze()
        
        
        sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
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
                                )
        sol_baseline.scale_P(p_baseline)
        sol_baseline.compute_non_solver_quantities(p_baseline)
        
        p = p_baseline.copy()
        p.delta[:,1] = deltas_of_equilibrium
        
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline,Nt=23,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=True,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=True,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        
        if equilibrium == 'nash_eq':
            rec_path = 'nash_eq_recaps/'
            method = 'fixed_point'
            method_name = 'method'
        if equilibrium == 'coop_negishi_eq':
            rec_path = 'coop_eq_recaps/'
            method = 'negishi'
            method_name = 'aggregation_method'
        if equilibrium == 'coop_equal_eq':
            rec_path = 'coop_eq_recaps/'
            method = 'pop_weighted'
            method_name = 'aggregation_method'
            
        if not os.path.exists(rec_path+'dyn_cons_eq_welfares.csv'):
            cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                             method_name] + p_baseline.countries + ['Equal','Negishi'])
            cons_eq_welfares.to_csv(rec_path+'dyn_cons_eq_welfares.csv')
        cons_eq_welfares = pd.read_csv(rec_path+'dyn_cons_eq_welfares.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+dyn_sol_c.cons_eq_welfare.tolist()+[dyn_sol_c.cons_eq_pop_average_welfare_change,
                                                           dyn_sol_c.cons_eq_negishi_welfare_change], 
                        index = cons_eq_welfares.columns).T
        cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
        cons_eq_welfares.to_csv(rec_path+'dyn_cons_eq_welfares.csv')
    
    