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
from solver_funcs import fixed_point_solver
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def minus_welfare_of_delta_custom_weighted(deltas,p,sol_baseline,weights):
    p.delta[...,1] = deltas
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 5,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    # sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, weights)
    # print(p.delta[...,1])
    
    
    return -sol_c.cons_eq_custom_weights_welfare_change

baseline_dics = [
    # {'baseline':'501',
    #                   'variation':'baseline'}
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
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_baseline.scale_P(p_baseline)
    # sol_baseline.compute_price_indices(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)   
    
    p = p_baseline.copy()
    p.delta[:,1] = np.array([0.01,0.01,0.01,12,12,12,12])
    
    for lamb in np.linspace(0,1,1001): 
        print(lamb)
        
        bounds = [(lb_delta,ub_delta)]*len(p.countries)
        weights = sol_baseline.cons**lamb*p_baseline.labor**(1-lamb)
    
        sol = optimize.minimize(fun = minus_welfare_of_delta_custom_weighted,
                                x0 = p.delta[...,1],
                                tol = 1e-15,
                                args=(p,sol_baseline,weights),
                                # options = {'disp':True},
                                bounds=bounds,
            )
        
        p.delta[...,1] = sol.x
        solution_welfare = -sol.fun
        
        #make a 'corner check'
        corner_corrected_deltas = p.delta[...,1].copy()
        for i,c in enumerate(p_baseline.countries):
            p_corner = p.copy()
            p_corner.delta[i,1] = ub_delta
            sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
                                            context = 'counterfactual',
                                            cobweb_anim=False,tol =1e-15,
                                            accelerate=False,
                                            accelerate_when_stable=True,
                                            cobweb_qty='profit',
                                            plot_convergence=False,
                                            plot_cobweb=False,
                                            safe_convergence=0.001,
                                            disp_summary=False,
                                            # apply_bound_psi_star = False,
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
            sol_corner.compute_non_solver_quantities(p_corner)
            sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
            sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, weights)
            if sol_corner.cons_eq_custom_weights_welfare_change > solution_welfare:
                print('corner was better for ',c)
                corner_corrected_deltas[i] = ub_delta
                
        p.delta[...,1] = corner_corrected_deltas
        
        
        # solve here opt_deltas
        
        
        
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                        context = 'counterfactual',
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
        # sol_c = var.var_from_vector(sol.x, p)    
        # sol_c.scale_tau(p)
        sol_c.scale_P(p)
        # sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
        # sol_c.compute_world_welfare_changes(p,sol_baseline)
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, weights)
        
        # welfares = sol_c.cons_eq_welfare
            
        write = True
        if write:
            if not os.path.exists('coop_eq_recaps/deltas_from_equal_to_negishi_3.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'lambda'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/deltas_from_equal_to_negishi_3.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/deltas_from_equal_to_negishi_3.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            lamb]+p.delta[...,1].tolist(), 
                            index = deltas_df.columns).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/deltas_from_equal_to_negishi_3.csv')
            
            if not os.path.exists('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi_3.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'lambda'] + p_baseline.countries + ['World'])
                cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi_3.csv')
            cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi_3.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            lamb]+sol_c.cons_eq_welfare.tolist()+[sol_c.cons_eq_custom_weights_welfare_change], 
                            index = cons_eq_welfares.columns).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares_from_equal_to_negishi_3.csv')
        deltas_df.plot()
        plt.show()