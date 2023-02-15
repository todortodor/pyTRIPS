#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:21:54 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
import time

baseline_dics = [
    # {'baseline':'311',
    #   'variation': None},
    # {'baseline':'311',
    #   'variation': '1.0'},
    # {'baseline':'311',
    #   'variation': '1.1'},
    # {'baseline':'311',
    #   'variation': '1.2'},
    # {'baseline':'311',
    #   'variation': '1.3'},
    # {'baseline':'311',
    #   'variation': '1.4'},
    # {'baseline':'311',
    #   'variation': '1.5'},
    # {'baseline':'311',
    #   'variation': '1.6'},
    # {'baseline':'311',
    #   'variation': '1.7'},
    # {'baseline':'311',
    #   'variation': '1.8'},
    # {'baseline':'311',
    #   'variation': '1.9'},
    # {'baseline':'311',
    #   'variation': '1.10'},
    # {'baseline':'311',
    #   'variation': '1.11'},
    # {'baseline':'311',
    #   'variation': '2.0'},
    # {'baseline':'311',
    #   'variation': '2.1'},
    # {'baseline':'311',
    #   'variation': '2.1.1'},
    # {'baseline':'311',
    #   'variation': '2.1.2'},
    # {'baseline':'311',
    #   'variation': '2.1.3'},
    # {'baseline':'311',
    #   'variation': '2.1.4'},
    # {'baseline':'311',
    #   'variation': '2.1.5'},
    # {'baseline':'311',
    #   'variation': '2.1.6'},
    # {'baseline':'311',
    #   'variation': '2.1.7'},
    # {'baseline':'311',
    #   'variation': '2.1.8'},
    # {'baseline':'311',
    #   'variation': '2.1.9'},
    # {'baseline':'311',
    #   'variation': '2.1.9.2'},
    # {'baseline':'311',
    #   'variation': '2.1.10'},
    # {'baseline':'311',
    #   'variation': '2.1.11'},
    # {'baseline':'311',
    #   'variation': '2.2'},
    # {'baseline':'311',
    #   'variation': '2.3'},
    # {'baseline':'311',
    #   'variation': '3.0'},
    # {'baseline':'311',
    #   'variation': '4.0'},
    # {'baseline':'311',
    #   'variation': '5.0'},
    # {'baseline':'311',
    #   'variation': '6.1'},
    # {'baseline':'311',
    #   'variation': '6.2'},
    # {'baseline':'311',
    #   'variation': '6.3'},
    # {'baseline':'311',
    #   'variation': '7.0'},
    # {'baseline':'311',
    #   'variation': '8.0'},
    {'baseline':'312',
                      'variation': None},
    # {'baseline':'312',
    #                   'variation': '1.0'},
    # {'baseline':'312',
    #                   'variation': '1.1'},
    # {'baseline':'312',
    #                   'variation': '1.2'},
    # {'baseline':'312',
    #                   'variation': '1.3'},
    # {'baseline':'312',
    #                   'variation': '2.0'},
    # {'baseline':'312',
    #                   'variation': '2.1'},
    # {'baseline':'312',
    #                   'variation': '2.2'},
    # {'baseline':'312',
    #                   'variation': '2.3'},
    # {'baseline':'312',
    #                   'variation': '3.0'},
    # {'baseline':'312',
    #                   'variation': '3.1'},
    # {'baseline':'312',
    #                   'variation': '3.2'},
    # {'baseline':'312',
    #                   'variation': '3.3'},
    # {'baseline':'312',
    #                   'variation': '4.0'},
    # {'baseline':'312',
    #                   'variation': '4.1'},
    # {'baseline':'312',
    #                   'variation': '4.2'},
    # {'baseline':'312',
    #                   'variation': '4.3'},
    # {'baseline':'312',
    #                   'variation': '5.0'},
    # {'baseline':'312',
    #                   'variation': '5.1'},
    # {'baseline':'312',
    #                   'variation': '5.2'},
    # {'baseline':'312',
    #                   'variation': '5.3'},
    # {'baseline':'312',
    #                   'variation': '6.0'},
    # {'baseline':'312',
    #                   'variation': '6.1'},
    # {'baseline':'312',
    #                   'variation': '6.2'},
    # {'baseline':'312',
    #                   'variation': '6.3'},
    # {'baseline':'312',
    #                   'variation': '7.0'},
    # {'baseline':'312',
    #                   'variation': '7.1'},
    # {'baseline':'312',
    #                   'variation': '7.2'},
    # {'baseline':'312',
    #                   'variation': '7.3'},
    # {'baseline':'312',
    #                   'variation': '8.0'},
    # {'baseline':'312',
    #                   'variation': '8.1'},
    # {'baseline':'312',
    #                   'variation': '8.2'},
    # {'baseline':'312',
    #                   'variation': '8.3'},
    # {'baseline':'312',
    #                   'variation': '9.0'},
    # {'baseline':'312',
    #                   'variation': '9.1'},
    # {'baseline':'312',
    #                   'variation': '9.2'},
    # {'baseline':'312',
    #                   'variation': '9.3'},
    # {'baseline':'312',
    #                   'variation': '10.0'},
    # {'baseline':'312',
    #                   'variation': '10.1'},
    # {'baseline':'312',
    #                   'variation': '10.2'},
    # {'baseline':'312',
    #                   'variation': '10.3'},
    # {'baseline':'312',
    #                   'variation': '11.0'},
    # {'baseline':'312',
    #                   'variation': '11.1'},
    # {'baseline':'312',
    #                   'variation': '11.2'},
    # {'baseline':'312',
    #                   'variation': '11.3'},
    ]

for baseline_dic in baseline_dics:
    if baseline_dic['variation'] is None:
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        # baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'_'+baseline_dic['variation']+'/'
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    print(baseline_dic['baseline'],baseline_dic['variation'])
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
    # p_baseline.beta = np.array([0.62485507, 0.37514493])
    # print(p_baseline.tau.shape)
    sol_baseline = var.var_from_vector(p_baseline.guess,p_baseline,context='calibration',compute=True)
    sol, sol_calibration = fixed_point_solver(p_baseline,
                            context = 'calibration', x0=p_baseline.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=True,
                            safe_convergence=1,
                            disp_summary=True,
                            damping = 10,
                            max_count = 5000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    # sol_calibration = var.var_from_vector(sol.x, p_baseline, context='calibration',compute=True)
    
    # sol_calibration.scale_tau(p_baseline)
    # sol_calibration.scale_P(p_baseline)
    sol_calibration.compute_non_solver_quantities(p_baseline)
    
    # p_baseline.tau = sol_calibration.tau.copy()
    # p_baseline.guess = sol_calibration.vector_from_var()
    # print(p_baseline.tau)
    
    sol, sol_counterfactual = fixed_point_solver(p_baseline,
                            context = 'counterfactual',x0=p_baseline.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='profit',
                            plot_convergence=True,
                            plot_cobweb=False,
                            safe_convergence=1,
                            disp_summary=True,
                            damping = 10,
                            max_count = 5000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    # print((sol_c.vector_from_var()/p_baseline.guess).min())
    # print((sol_c.vector_from_var()/p_baseline.guess).max())
    # print((sol_baseline.vector_from_var()/sol_c.vector_from_var()).max())
    # time.sleep(5)
    # w = sol_baseline.w
    # l_R = sol_baseline.l_R[...,1:].ravel()
    # profit = sol_baseline.profit[...,1:].ravel()
    # Z = sol_baseline.Z
    # phi = sol_baseline.phi.ravel()
    # vec = np.concatenate((w,Z,l_R,profit,phi), axis=0)
    # print(vec.shape)
    # p_baseline.guess = vec
    # p_baseline.write_params(baseline_path)
    
    sol_counterfactual.scale_P(p_baseline)
    sol_counterfactual.compute_non_solver_quantities(p_baseline)
    # sol_c.compute_price_indices(p)
    # sol_c.compute_non_solver_quantities(p_baseline) 
    # p_baseline.tau = sol_c.tau
    # p_baseline.guess = sol_c.vector_from_var()
    # p_baseline.write_params(baseline_path)
# compare_two_sol(sol_calibration,sol_counterfactual)
# print(sol_calibration.phi/sol_counterfactual.phi)
#%%

def compare_two_sol(sol1,sol2):
    for key, item in sorted(sol1.__dict__.items()):
        print(key)
        try:
            if np.allclose(getattr(sol1,key),getattr(sol2,key)):
                print(key,'same')
            else:
                print(key,'different')
                print(np.nanmin(getattr(sol1,key)/getattr(sol2,key)),np.nanmax(getattr(sol1,key)/getattr(sol2,key)))
        except:
            pass
            
#%%
p = p_baseline.copy()

init_calibration = var.var_from_vector(p.guess, p, context = 'calibration',compute=False)
init_counterfactual = var.var_from_vector(p.guess, p, context = 'counterfactual',compute=False)

init_calibration.compute_growth(p)
init_counterfactual.compute_growth(p)

init_calibration.compute_patenting_thresholds(p)
init_counterfactual.compute_patenting_thresholds(p)

init_calibration.compute_aggregate_qualities(p)
init_counterfactual.compute_aggregate_qualities(p)

init_calibration.compute_sectoral_prices(p)
init_counterfactual.compute_sectoral_prices(p)

init_calibration.compute_labor_allocations(p)
init_counterfactual.compute_labor_allocations(p)

init_calibration.compute_trade_flows_and_shares(p)
init_counterfactual.compute_trade_flows_and_shares(p)

init_calibration.compute_price_indices(p)
init_counterfactual.compute_price_indices(p)

compare_two_sol(init_calibration,init_counterfactual)

#%%

sol_calibration.context = 'calibration'
X_M, X_CD, X = sol_calibration.compute_trade_flows_and_shares(p_baseline,assign=False)
phi = sol_calibration.compute_phi(p_baseline)
sol_calibration.context = 'counterfactual'
X_M_cf, X_CD_cf, X_cf = sol_calibration.compute_trade_flows_and_shares(p_baseline,assign=False)
phi_cf = sol_calibration.compute_phi(p_baseline)
# print('X_M',X_M/X_M_cf)
# print('X_CD',X_CD/X_CD_cf)
# print('X',X/X_cf)
print('X',X/np.diagonal(X).transpose()[:,None,:]/(X_cf/np.diagonal(X_cf).transpose()[:,None,:]))
# print('phi',phi/phi_cf)
