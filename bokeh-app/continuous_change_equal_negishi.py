#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:37:46 2023

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_coop_eq
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
    {'baseline':'501',
                      'variation':'2.0'}
    ]


lb_delta = 0.01
ub_delta = 12

for baseline_dic in baseline_dics:    
# for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        # baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'_'+baseline_dic['variation']+'/'
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)

    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                           context = 'counterfactual',
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
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)   
    
    # p.delta[:,1] = np.array([0.01,0.01,0.01,12,12,12,12])
    
    aggregation_method = 'custom_weights'
    
    for lamb in np.linspace(0,1,1001): 
        print(lamb)

        weights = sol_baseline.cons**lamb*p_baseline.labor**(1-lamb)
    
        p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method=aggregation_method,
                         lb_delta=0.01,ub_delta=12,dynamics=False,
                         solver_options=None,tol=1e-15,
                         static_eq_deltas = None,custom_weights=weights,
                         custom_x0 = np.array([0.01,0.01,0.01,12,12,12,12]))
        
        write = True
        if write:
            if not os.path.exists('coop_eq_recaps/deltas_from_equal_to_negishi.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/deltas_from_equal_to_negishi.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/deltas_from_equal_to_negishi.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+p_opti.delta[...,1].tolist(), 
                            index = deltas_df.columns).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/deltas_from_equal_to_negishi.csv')
            
            if not os.path.exists('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
                cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi.csv')
            cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
                                                               sol_opti.cons_eq_negishi_welfare_change], 
                            index = cons_eq_welfares.columns).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi.csv')
