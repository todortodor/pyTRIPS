#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:27:47 2022

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab
# from scipy.signal import argrelmin, argrelmax

def is_oscillating(deltas,column,window):
    a = deltas[column].values[-window:]
    a = a-a.mean()
    sign_changes = np.where(np.sign(a[:-1]) != np.sign(a[1:]))
    # return ((window_values-window_values.mean())>=0).sum() >= window/2 and ((window_values-window_values.mean())<=0).sum() >= window/2
    # if a.shape == sign_changes.shape:
    #     return np.allclose(np.where(np.sign(a[:-1]) != np.sign(a[1:])) , np.arange(window-1))
    return (sign_changes[0].shape[0] >= window-2) or (np.all(a == 0))

def minus_welfare_of_delta(delta,p,c,sol_it_baseline):
    # print('solving')
    back_up_delta_value = p.delta[p.countries.index(c),1]
    p.delta[p.countries.index(c),1] = delta
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
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
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    p.delta[p.countries.index(c),1] = back_up_delta_value
    p.guess = sol.x
    
    return -sol_c.cons_eq_welfare[p.countries.index(c)]
    

def compute_new_deltas_Newton(p, sol_it_baseline, small_change, lb_delta, ub_delta):
    
    new_deltas = pd.Series(index = p.countries, dtype = float)

    for c in p.countries:
        back_up_delta_value = p.delta[p.countries.index(c),1]
        p.delta[p.countries.index(c),1] = p.delta[p.countries.index(c),1] * (1+small_change)
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
        # print(sol.status)
    
        sol_c = var.var_from_vector(sol.x, p)    
        # sol_c.scale_tau(p)
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
        
        # print(sol_c.cons_eq_welfare)
        
        if sol_c.cons_eq_welfare[p.countries.index(c)] > 1:
            new_deltas.loc[c] = min(back_up_delta_value * (1+small_change),ub_delta)
            # if back_up_delta_value * (1+small_change)< ub_delta:
            #     new_deltas.loc[c] = back_up_delta_value * (1+small_change)
            # else:
            #     new_deltas.loc[c] = ub_delta
        
        elif sol_c.cons_eq_welfare[p.countries.index(c)] < 1:
            new_deltas.loc[c] = max(back_up_delta_value * (1-small_change),lb_delta)
            # if back_up_delta_value * (1-small_change) > lb_delta:
            #     new_deltas.loc[c] = back_up_delta_value * (1-small_change)
            # else:
            #     new_deltas.loc[c] = lb_delta
        
        else:
            new_deltas.loc[c] = back_up_delta_value
        
        p.delta[p.countries.index(c),1] = back_up_delta_value
    
    return new_deltas
    
def compute_new_deltas_fixed_point(p, sol_it_baseline, lb_delta, ub_delta):
    new_deltas = pd.Series(index = p.countries, dtype = float)

    for c in p.countries:
        # print(c)
        delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
                                             method='bounded',
                                             bounds=(lb_delta, ub_delta),
                                             args = (p,c,sol_it_baseline))
        # id_it = track_how_many_identical_iterations[p.countries.index(c)]
        # interval_size = (ub_delta-lb_delta)/(id_it+1)**2
        # max_welfare_delta = lb_delta
        # max_welfare = 0
        # back_up_delta_value = p.delta[p.countries.index(c),1]
        # local_lb_delta =max(lb_delta,back_up_delta_value-interval_size)
        # local_ub_delta =min(ub_delta,back_up_delta_value+interval_size)
        # print(c,local_lb_delta,local_ub_delta)
        # for delta in tqdm(np.linspace(local_lb_delta,local_ub_delta,10)):
        #     p.delta[p.countries.index(c),1] = delta
        #     sol, sol_c = fixed_point_solver(p,x0=p.guess,
        #                             cobweb_anim=False,tol =1e-15,
        #                             accelerate=False,
        #                             accelerate_when_stable=True,
        #                             cobweb_qty='phi',
        #                             plot_convergence=False,
        #                             plot_cobweb=False,
        #                             safe_convergence=0.001,
        #                             disp_summary=False,
        #                             damping = 10,
        #                             max_count = 1e4,
        #                             accel_memory = 50, 
        #                             accel_type1=True, 
        #                             accel_regularization=1e-10,
        #                             accel_relaxation=0.5, 
        #                             accel_safeguard_factor=1, 
        #                             accel_max_weight_norm=1e6,
        #                             damping_post_acceleration=5
        #                             # damping=10
        #                               # apply_bound_psi_star=True
        #                             )
        #     # print(sol.status)
        
        #     sol_c = var.var_from_vector(sol.x, p)    
        #     # sol_c.scale_tau(p)
        #     sol_c.scale_P(p)
        #     sol_c.compute_price_indices(p)
        #     sol_c.compute_non_solver_quantities(p)
        #     sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
        
        # # print(sol_c.cons_eq_welfare)
        
        #     if sol_c.cons_eq_welfare[p.countries.index(c)] > max_welfare:
        #         max_welfare = sol_c.cons_eq_welfare[p.countries.index(c)]
        #         max_welfare_delta = delta
        # print(delta_min.x)
        new_deltas.loc[c] = delta_min.x
        
        # p.delta[p.countries.index(c),1] = back_up_delta_value
    
    return new_deltas

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = []

for baseline_number in ['101','102','104']:
    baseline_dics.append({'baseline':baseline_number,
                      'variation':'baseline'})
    
    files_in_dir = next(os.walk('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        baseline_dics.append({'baseline':baseline_number,
                          'variation':run})

for baseline_dic in baseline_dics[57:]:    
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
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_price_indices(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    
    
    # p_it_baseline.delta[0,1] = p_it_baseline.delta[0,1]*10
    # p_it_baseline.eta[0,1] = p_it_baseline.eta[0,1]*2
    
    for method in ['fixed_point']:
    
        new_run = True
        plot_convergence = True
        condition = True
        lb_delta = 0.01
        ub_delta = 100
        
        if new_run:
            deltas = pd.DataFrame(index = p_baseline.countries).T
            welfares = pd.DataFrame(index = p_baseline.countries).T
            p_it_baseline = p_baseline.copy()
            sol_it_baseline = sol_baseline.copy()
            all_oscillating = False
            buffer = 0
            condition = True
            it = 0
            small_change = 0.5
        
        window = 4       
        
        while condition:
            print(it)
            if method == 'newton':
                deltas.loc[it] = compute_new_deltas_Newton(p_it_baseline, sol_it_baseline, 
                                                           small_change, lb_delta, ub_delta)
            if method == 'fixed_point':
                deltas.loc[it] = compute_new_deltas_fixed_point(p_it_baseline, 
                                                                sol_it_baseline, 
                                                                lb_delta, 
                                                                ub_delta)
            
            p_it_baseline.delta[...,1] = deltas.loc[it].values
            
            sol, sol_it_baseline = fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
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
            sol_it_baseline = var.var_from_vector(sol.x, p_it_baseline)    
            # sol_c.scale_tau(p)
            sol_it_baseline.scale_P(p_it_baseline)
            sol_it_baseline.compute_price_indices(p_it_baseline)
            sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
            sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
            
            welfares.loc[it] = sol_it_baseline.cons_eq_welfare
            
            p_it_baseline.guess = sol.x
            
            if it>window:
                all_oscillating = np.array([is_oscillating(deltas,c,window) for c in deltas.columns]).prod()
            
            if all_oscillating:
                # if small_change<0.01:
                #     print('checking')
                #     condition = np.linalg.norm(deltas.loc[it].values-deltas.loc[it-1].values)\
                #         /np.linalg.norm(deltas.loc[it].values) < 1e-4
                # else:
                buffer += 1
                if buffer >= window:
                    small_change = small_change/2
                    buffer = 0
                    
                # print(small_change)
            # condition = np.linalg.norm(deltas.loc[it].values-deltas.loc[it-1].values)\
            #     /np.linalg.norm(deltas.loc[it].values) > 1e-3
            if it != 0:
                condition = np.linalg.norm((deltas.loc[it].values-deltas.loc[it-1].values)/
                                           deltas.loc[it].values)> 5e-3
            it +=1
            
            if plot_convergence:
                fig,ax = plt.subplots()
                
                ax2 = ax.twinx()
                
                deltas.plot(logy=True,ax=ax, xlabel = 'Iteration', 
                              ylabel = 'Delta', 
                              title = 'Convergence to Nash equilibrium')
                welfares.plot(ax=ax2, ls = '--', ylabel = 'Consumption eq. welfare')
                
                plt.show()
            
        write = True
        if write:
            if not os.path.exists('nash_eq_recaps/deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'method'] + p_baseline.countries)
                deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            method]+deltas.iloc[-1].to_list(), 
                            index = deltas_df.columns).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            
            if not os.path.exists('nash_eq_recaps/cons_eq_welfares.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'method'] + p_baseline.countries)
                cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
            cons_eq_welfares = pd.read_csv('nash_eq_recaps/cons_eq_welfares.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            method]+welfares.iloc[-1].to_list(), 
                            index = cons_eq_welfares.columns).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')

    
# #%%    

# fig,ax = plt.subplots()

# ax2 = ax.twinx()

# deltas.plot(logy=True,ax=ax, xlabel = 'Code iterations', 
#               ylabel = 'Delta', 
#               title = 'Convergence to Nash equilibrium')
# welfares.plot(ax=ax2, ls = '--', ylabel = 'Consumption eq. welfare')

# ax.legend(loc=(-0.15,0.1))
# ax2.legend(loc=(1.05,0.1))

# plt.show()
