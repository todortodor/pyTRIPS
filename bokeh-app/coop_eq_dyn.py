#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:40:41 2023

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

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def minus_welfare_of_delta_pop_weighted(deltas,p,sol_baseline):
    p.delta[...,1] = deltas
    print(p.delta[...,1])
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
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    
    sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                          t_inf=500,
                                          # x0 = p.dyn_guess,#!!!
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=False,
                            damping = 60,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    p.dyn_guess = dyn_sol_c.vector_from_var()
    p.guess = dyn_sol_c.sol_fin.vector_from_var()
    print(sol.status)
    dyn_sol_c.compute_non_solver_quantities(p)
    print(dyn_sol_c.cons_eq_pop_average_welfare_change)
    
    return -dyn_sol_c.cons_eq_pop_average_welfare_change

def minus_welfare_of_delta_negishi_weighted(deltas,p,sol_baseline):
    p.delta[...,1] = deltas
    print(p.delta[...,1])
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
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)

    sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                          t_inf=500,
                                          # x0 = p.dyn_guess,#!!!
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=False,
                            damping = 60,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    p.dyn_guess = dyn_sol_c.vector_from_var()
    p.guess = dyn_sol_c.sol_fin.vector_from_var()
    print(sol.status)
    dyn_sol_c.compute_non_solver_quantities(p)
    print(dyn_sol_c.cons_eq_negishi_welfare_change)
    
    return -dyn_sol_c.cons_eq_negishi_welfare_change
        
baseline_dics = [
    {'baseline':'501',
                      'variation':'1.0'},
    {'baseline':'501',
                      'variation':'2.0'}
    ]

lb_delta = 0.01
ub_delta = 12

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
    
    for aggregation_method in ['negishi','pop_weighted']:
        print(aggregation_method)
        p = p_baseline.copy()
        static_eq_deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0).drop_duplicates(
            ['baseline','variation','aggregation_method'],keep='last')
        static_eq_deltas = static_eq_deltas.loc[
            (static_eq_deltas.baseline.astype('str') == baseline_dic['baseline'])
            & (static_eq_deltas.variation.astype('str') == baseline_dic['variation'])
            & (static_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()
        bounds = [(lb_delta,ub_delta)]*len(p.countries)
        if aggregation_method == 'pop_weighted':
            sol = optimize.minimize(fun = minus_welfare_of_delta_pop_weighted,
                                    x0 = static_eq_deltas,
                                    tol = 1e-8,
                                    args=(p,sol_baseline),
                                    options = {'disp':3},
                                    bounds=bounds,
                )
        if aggregation_method == 'negishi':
            sol = optimize.minimize(fun = minus_welfare_of_delta_negishi_weighted,
                                    x0 = static_eq_deltas,
                                    tol = 1e-8,
                                    args=(p,sol_baseline),
                                    options = {'disp':3},
                                    bounds=bounds
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
                                            damping = 10,
                                            max_count = 1e4,
                                            accel_memory = 50, 
                                            accel_type1=True, 
                                            accel_regularization=1e-10,
                                            accel_relaxation=0.5, 
                                            accel_safeguard_factor=1, 
                                            accel_max_weight_norm=1e6,
                                            damping_post_acceleration=5
                                            )
            sol_corner.compute_non_solver_quantities(p_corner)
            sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
            sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
            sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, Nt=23,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )

            dyn_sol_corner.compute_non_solver_quantities(p)

            if aggregation_method == 'negishi':
                if dyn_sol_corner.cons_eq_negishi_welfare_change > solution_welfare:
                    print('corner was better for ',c)
                    corner_corrected_deltas[i] = ub_delta
            if aggregation_method == 'pop_weighted':
                if dyn_sol_corner.cons_eq_pop_average_welfare_change > solution_welfare:
                    print('corner was better for ',c)
                    corner_corrected_deltas[i] = ub_delta
                
        p.delta[...,1] = corner_corrected_deltas
        
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
        # sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
        # sol_c.compute_world_welfare_changes(p,sol_baseline)
        
        # welfares = sol_c.cons_eq_welfare
            
        write = True
        if write:
            if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+p.delta[...,1].tolist(), 
                            index = deltas_df.columns).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
            # if not os.path.exists('coop_eq_recaps/cons_eq_welfares.csv'):
            #     cons_eq_welfares = pd.DataFrame(columns = ['baseline',
            #                                     'variation',
            #                                     'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
            #     cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
            # cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares.csv',index_col=0)
            # run = pd.DataFrame(data = [baseline_dic['baseline'],
            #                 baseline_dic['variation'],
            #                 aggregation_method]+sol_c.cons_eq_welfare.tolist()+[sol_c.cons_eq_pop_average_welfare_change,
            #                                                     sol_c.cons_eq_negishi_welfare_change], 
            #                 index = cons_eq_welfares.columns).T
            # cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            # cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
