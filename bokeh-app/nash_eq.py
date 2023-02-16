#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:27:47 2022

@author: slepot
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from solver_funcs import find_nash_eq
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab
# from scipy.signal import argrelmin, argrelmax

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# baseline_dics = [
#                 # {'baseline':'311',
#                 #   'variation': '4.0'},
#                 # {'baseline':'311',
#                 #   'variation': '5.0'},
#                 # {'baseline':'311',
#                 #   'variation': '6.1'},
#                 # {'baseline':'311',
#                 #   'variation': '6.2'},
#                 # {'baseline':'311',
#                 #   'variation': '6.3'},
#                 # {'baseline':'311',
#                 #   'variation': '7.0'},
#                 {'baseline':'311',
#                   'variation': '8.0'}
#                  ]

# baseline_dics = [
#     # {'baseline':'312',
#     #                   'variation': 'baseline'},
#     # {'baseline':'312',
#     #                   'variation': '1.0'},
#     # {'baseline':'312',
#     #                   'variation': '2.0'},
#     # {'baseline':'312',
#     #                   'variation': '3.0'},
#     # {'baseline':'312',
#     #                   'variation': '4.0'},
#     # {'baseline':'312',
#     #                   'variation': '5.0'},
#     # {'baseline':'312',
#     #                   'variation': '6.0'},
#     # {'baseline':'312',
#     #                   'variation': '7.0'},
#     # {'baseline':'312',
#     #                   'variation': '8.0'},
#     # {'baseline':'312',
#     #                   'variation': '9.0'},
#     # {'baseline':'312',
#     #                   'variation': '10.0'},
#     # {'baseline':'312',
#     #                   'variation': '11.0'},
#     {'baseline':'312',
#                       'variation': '11.1'},
#     # {'baseline':'312',
#     #                   'variation': '11.2'},
#     # {'baseline':'312',
#     #                   'variation': '11.3'},
#     ]

baseline_dics = [
    # {'baseline':'401',
    #                   'variation': 'baseline'},
    # {'baseline':'401',
    #                   'variation': '1.0'},
    # {'baseline':'401',
    #                   'variation': '2.0'},
    # {'baseline':'401',
    #                   'variation': '3.0'},
    # {'baseline':'401',
    #                   'variation': '4.0'},
    # {'baseline':'401',
    #                   'variation': '5.0'},
    # {'baseline':'401',
    #                   'variation': '6.0'},
    # {'baseline':'401',
    #                   'variation': '7.0'},
    # {'baseline':'401',
    #                   'variation': '8.0'},
    # {'baseline':'401',
    #                   'variation': '9.0'},
    # {'baseline':'401',
    #                   'variation': '10.0'},
    # {'baseline':'401',
    #                   'variation': '11.0'},
    # {'baseline':'401',
    #                   'variation': '11.1'},
    {'baseline':'401',
                      'variation': '11.2'},
    {'baseline':'401',
                      'variation': '11.3'},
    ]

for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
    
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
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_baseline.scale_P(p_baseline)
    # sol_baseline.compute_price_indices(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)   
    
    write = True
    
    method = 'fixed_point'
    
    deltas,welfares = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=12,method='fixed_point',
                     plot_convergence = True,solver_options=None,tol=1e-4)
    
    p = p_baseline.copy()
    p.delta[...,1] = deltas[...,-1]
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
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
    # sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    
    if write:
        if not os.path.exists('nash_eq_recaps/deltas.csv'):
            deltas_df = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries)
            deltas_df.to_csv('nash_eq_recaps/deltas.csv')
        deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+deltas[...,-1].tolist(), 
                        index = deltas_df.columns).T
        deltas_df = pd.concat([deltas_df, run],ignore_index=True)
        deltas_df.to_csv('nash_eq_recaps/deltas.csv')
        
        if not os.path.exists('nash_eq_recaps/cons_eq_welfares.csv'):
            cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries + ['Equal','Negishi'])
            cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        cons_eq_welfares = pd.read_csv('nash_eq_recaps/cons_eq_welfares.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+welfares[...,-1].tolist()+[sol_c.cons_eq_pop_average_welfare_change,
                                                           sol_c.cons_eq_negishi_welfare_change], 
                        index = cons_eq_welfares.columns).T
        cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
        cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
