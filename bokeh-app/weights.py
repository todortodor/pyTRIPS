#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:09:24 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import time
import os
import seaborn as sns
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver, compute_deriv_welfare_to_patent_protec_US, compute_deriv_growth_to_patent_protec_US
from data_funcs import write_calibration_results, compare_params
from functools import reduce
from tqdm import tqdm

#%% define baseline and conditions of weights analysis

baseline = '101'
variation = '11.7'
if variation is None:
    baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
else:
    baseline_path = 'calibration_results_matched_economy/baseline_'+baseline+'_variations/'+variation+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)
sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 5e4,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )

sol_baseline = var.var_from_vector(sol.x, p_baseline)    
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

m_baseline.compute_moments(sol_baseline, p_baseline)

m_baseline.drop_CHN_IND_BRA_ROW_from_RD = True

weights_to_change = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRDUS',
 'SRGDP', 'JUPCOST', 'SINNOVPATUS', 'TO', 'SPFLOW', 'DOMPATEU', 'DOMPATUS']
if variation is None:
    parent_weights_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_weights_variations/'
else:
    parent_weights_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_'+variation+'_weights_variations/'

if variation is None:
    weights_tables_path = 'calibration_results_matched_economy/baseline_'+baseline+'_weights_tables/'
else:
    weights_tables_path = 'calibration_results_matched_economy/baseline_'+baseline+'_'+variation+'_weights_tables/'
    
def make_dirs(list_of_paths):
    for path in list_of_paths:
        try:
            os.mkdir(path)
        except:
            pass
        
#%%

make_dirs([parent_weights_result_path])

#%%

dic_runs = dict([(mom, np.linspace(m_baseline.weights_dict[mom]*0.5,m_baseline.weights_dict[mom]*1.5,11))
                 for mom in weights_to_change])

for k, v in dic_runs.items():
    print(k)
    print(v)
    weight_to_change = k
    weight_list = v
    result_path = parent_weights_result_path+weight_to_change+'/'
    make_dirs([result_path])
    for i,weight in enumerate(weight_list):
        m = m_baseline.copy()
        p = p_baseline.copy()
        m.weights_dict[weight_to_change] = weight
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        test_ls = optimize.least_squares(fun = calibration_func,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time), 
                                bounds = bounds,
                                # method= 'dogbox',
                                # loss='arctan',
                                # jac='3-point',
                                max_nfev=1e8,
                                # ftol=1e-14, 
                                xtol=1e-16, 
                                # gtol=1e-14,
                                # f_scale=scale,
                                verbose = 2)
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=True,
                                plot_cobweb=True,
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
    
        # sol_c = var.var_from_vector(sol.x, p_sol)    
        # sol_c.scale_tau(p_sol)
        sol_c.scale_P(p_sol)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p_sol) 
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
#%% Gather all weight changes

dic_of_variation_dics = {}

for weight_to_change in weights_to_change:
    variation_dic = {}
    
    print(weight_to_change)
    result_path = parent_weights_result_path+weight_to_change+'/'
    baseline_weight = m_baseline.weights_dic[weight_to_change]
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_values = {}
    dic_change = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        # if run != '99':
        print(run)
        run_path = result_path+run+'/'
        p = parameters(n=7,s=2)
        p.load_data(run_path)
        m = moments()
        m.load_data()
        m.load_run(run_path)
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 5e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        sol_c = var.var_from_vector(sol.x, p)    
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p) 
        m.compute_moments(sol_c,p)
        m.compute_moments_deviations()
        
        run_name = run+': '+weight_to_change+str(m.weights_dic[weight_to_change])
        runs.append(run_name)
        dic_m[run_name] = m
        dic_p[run_name] = p
        dic_sol[run_name] = sol_c
        dic_values[run_name] = float(m.weights_dic[weight_to_change])
        dic_change[run_name] = float((m.weights_dic[weight_to_change]-baseline_weight)*100/baseline_weight)
        
    variation_dic['changing_quantity'] = weight_to_change+'_target'
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_weight
    
    dic_of_variation_dics[weight_to_change] = variation_dic
    
#%% write recaps

make_dirs([weights_tables_path])