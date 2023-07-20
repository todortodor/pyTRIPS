#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:06:50 2022

@author: simonl
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
from data_funcs import write_calibration_results
from functools import reduce
from tqdm import tqdm

#%% define baseline and conditions of sensitivity analysis

baseline = '1010'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters()
p_baseline.load_run(baseline_path)
m_baseline = moments()
m_baseline.load_run(baseline_path)
sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                       context='calibration',
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

sol_baseline = var.var_from_vector(sol.x, p_baseline,context = 'calibration')    
sol_baseline.scale_P(p_baseline)
# sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

m_baseline.compute_moments(sol_baseline, p_baseline)

moments_to_change = ['KM','UUPCOST','SINNOVPATUS','TO','GROWTH',
                      'DOMPATINUS','DOMPATINEU','TE','OUT']
# moments_to_change = ['TE','OUT']
parameters_to_change = ['gamma','rho']

weights_to_change = m_baseline.list_of_moments

dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'

parent_moment_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_targets_variations/'
parent_moment_dropbox_path = dropbox_path+'baseline_'+baseline+'_targets_variation/'

parent_param_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_parameters_variations/'
parent_param_dropbox_path = dropbox_path+'baseline_'+baseline+'_parameters_variation/'

parent_weight_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_weights_variations/'
parent_weight_dropbox_path = dropbox_path+'baseline_'+baseline+'_weights_variation/'

sensitivity_path = dropbox_path+'baseline_'+baseline+'_sensitivities/'
sensitivity_tables_path = 'calibration_results_matched_economy/baseline_'+baseline+'_sensitivity_tables/'

sensitivity_weights_path = dropbox_path+'baseline_'+baseline+'_sensitivities_weights/'
sensitivity_weights_tables_path = 'calibration_results_matched_economy/baseline_'+baseline+'_sensitivity_weights_tables/'

def make_dirs(list_of_paths):
    for path in list_of_paths:
        try:
            os.mkdir(path)
        except:
            pass

def GetSpacedElements(array, numElems = 13):
    idx = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    out = array[idx]
    return out, idx

#%% make dirs

make_dirs([parent_moment_result_path,
           parent_moment_dropbox_path,
           parent_param_result_path,
           parent_param_dropbox_path,
           parent_weight_result_path,
           parent_weight_dropbox_path,
           sensitivity_path])

#%% make calibration runs for different moment(s) target

dic_runs = dict([(mom, np.linspace(getattr(m_baseline,mom+'_target')*0.5,getattr(m_baseline,mom+'_target')*1.5,11))
                 for mom in moments_to_change])

for k, v in dic_runs.items():
    print(k)
    print(v)
    moment_to_change = k
    target_list = v
    result_path = parent_moment_result_path+moment_to_change+'/'
    dropbox_path = parent_moment_dropbox_path+moment_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,target in enumerate(target_list):
        print(k)
        print(v)
        print(target)
        # m = moments()
        # m.load_run(baseline_path)
        # m.drop_CHN_IND_BRA_ROW_from_RD = True
        m = m_baseline.copy()
        # p = parameters()
        # p.load_run(baseline_path)
        p = p_baseline.copy()
        # if moment_to_change == 'ERDUS' and 'ERDUS' not in m.list_of_moments:
        #     m.list_of_moments.append('ERDUS')
        #     if 'kappa' not in p.calib_parameters:
        #         p.calib_parameters.append('kappa')
        setattr(m,moment_to_change+'_target',target)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        cond = True
        iterations = 0
        max_iter = 5
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        # method= 'dogbox',
                                        # loss='arctan',
                                        # jac='3-point',
                                        max_nfev=1e8,
                                        # ftol=1e-14, 
                                        xtol=1e-10, 
                                        # gtol=1e-14,
                                        # f_scale=scale,
                                        verbose = 2)
            else:
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
            cond = iterations < max_iter
            iterations += 1
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                        context = 'calibration',
                                cobweb_anim=False,tol =1e-14,
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
    
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.guess = sol.x
        p_sol.tau = sol_c.tau
        
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        
#%% make calibration runs for different parameters

dic_runs = dict([(par, np.linspace(getattr(p_baseline,par)*0.5,getattr(p_baseline,par)*1.5,11))
                 for par in parameters_to_change])
# if 'kappa' in parameters_to_change:
#     dic_runs['kappa'] = np.linspace(getattr(p_baseline,'kappa')*0.8,getattr(p_baseline,'kappa')*1.2,21)
if 'zeta' in parameters_to_change:
    dic_runs['zeta'] = [np.array([p_baseline.zeta[0], i]) for i in np.linspace(0,0.1,21)]
    # p_baseline.calib_parameters.remove('zeta')

for k, v in dic_runs.items():
    print(k)
    print(v)
    par_to_change = k
    par_list = v
    result_path = parent_param_result_path+par_to_change+'/'
    dropbox_path = parent_param_dropbox_path+par_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,par in enumerate(par_list):
        print(k)
        print(v)
        print(par)
        # m = moments()
        # m.load_run(baseline_path)
        # m.drop_CHN_IND_BRA_ROW_from_RD = True
        m = m_baseline.copy()
        # p = parameters()
        # p.load_run(baseline_path)
        p = p_baseline.copy()
        if par_to_change == 'zeta':
            if 'zeta' in p.calib_parameters:
                p.calib_parameters.remove('zeta')
        setattr(p,par_to_change,par)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        cond = True
        iterations = 0
        max_iter = 5
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        # method= 'dogbox',
                                        # loss='arctan',
                                        # jac='3-point',
                                        max_nfev=1e8,
                                        # ftol=1e-14, 
                                        xtol=1e-10, 
                                        # gtol=1e-14,
                                        # f_scale=scale,
                                        verbose = 2)
            else:
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
            cond = iterations < max_iter
            iterations += 1
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,
                                        context = 'calibration',
                                        x0=p_sol.guess,
                                cobweb_anim=False,tol =1e-14,
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
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.guess = sol.x
        p_sol.tau = sol_c.tau
        
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        # m.plot_moments(m.list_of_moments, save_plot = dropbox_path+str(i))
        
        
#%% make calibration runs for different weights

dic_runs = dict([(mom, np.linspace(m_baseline.weights_dict[mom]*0.1,m_baseline.weights_dict[mom]*1.9,19))
                 for mom in weights_to_change])

for k, v in dic_runs.items(): 
    print(k)
    print(v)
    mom_weight_to_change = k
    weight_list = v 
    result_path = parent_weight_result_path+mom_weight_to_change+'/'
    dropbox_path = parent_weight_dropbox_path+mom_weight_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,weight in enumerate(weight_list):
        print(k)
        print(v)
        print(weight)
        # m = moments()
        # m.load_run(baseline_path)
        # m.drop_CHN_IND_BRA_ROW_from_RD = True
        m = m_baseline.copy()
        # p = parameters()
        # p.load_run(baseline_path)
        p = p_baseline.copy()
        m.weights_dict[mom_weight_to_change] = weight
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        cond = True
        iterations = 0
        max_iter = 8
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        # method= 'dogbox',
                                        # loss='arctan',
                                        # jac='3-point',
                                        max_nfev=1e8,
                                        # ftol=1e-14, 
                                        xtol=1e-10, 
                                        # gtol=1e-14,
                                        # f_scale=scale,
                                        verbose = 2)
            else:
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
            cond = iterations < max_iter
            iterations += 1
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,
                                        context = 'calibration',
                                        x0=p_sol.guess,
                                cobweb_anim=False,tol =1e-14,
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
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.guess = sol.x
        p_sol.tau = sol_c.tau
        
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        # m.plot_moments(m.list_of_moments, save_plot = dropbox_path+str(i))

#%% Gather all parameters and moments variations

dic_of_variation_dics = {}

for moment_to_change in moments_to_change:
    variation_dic = {}
    print(moment_to_change)
    result_path = parent_moment_result_path+moment_to_change+'/'
    
    baseline_moment = getattr(m_baseline, moment_to_change+'_target')
    if moment_to_change == 'ERDUS':
        baseline_moment = getattr(m_baseline, moment_to_change)
    
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
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters()
            p.load_run(run_path)
            m = moments()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,
                                            context = 'calibration',
                                            x0=p.guess,
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
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            run_name = run+': '+moment_to_change+str(getattr(m,moment_to_change+'_target'))
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(getattr(m,moment_to_change+'_target'))
            dic_change[run_name] = float((getattr(m,moment_to_change+'_target')-baseline_moment)*100/baseline_moment)
        
    variation_dic['changing_quantity'] = moment_to_change+'_target'
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_moment
    
    dic_of_variation_dics[moment_to_change+'_target'] = variation_dic
    
for parameter_to_change in parameters_to_change:
    variation_dic = {}
    print(parameter_to_change)
    result_path = parent_param_result_path+parameter_to_change+'/'
    
    try:
        baseline_param = getattr(p_baseline, parameter_to_change)[1]
    except:
        baseline_param = getattr(p_baseline, parameter_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_change = {}
    dic_values = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters()
            p.load_run(run_path)
            m = moments()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'calibration',
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
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            
            try:
                current_param = getattr(p,parameter_to_change)[1]
            except:
                current_param = getattr(p,parameter_to_change)
            
            run_name = run+': '+parameter_to_change+str(current_param)
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(current_param)
            dic_change[run_name] = float((current_param-baseline_param)*100/baseline_param)
        
    variation_dic['changing_quantity'] = parameter_to_change
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_param
    
    dic_of_variation_dics[parameter_to_change] = variation_dic

#%% Gather weight change runs

dic_of_weights_variation_dic = {}

for mom_weight_to_change in m_baseline.list_of_moments:
    variation_dic = {}
    print(mom_weight_to_change)
    result_path = parent_weight_result_path+mom_weight_to_change+'/'
    print(result_path)
    
    baseline_weight = m_baseline.weights_dict[mom_weight_to_change]
    
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
        print(run)
        run_path = result_path+run+'/'
        p = parameters()
        p.load_run(run_path)
        m = moments()
        # m.load_data()
        m.load_run(run_path)
        sol, sol_c = fixed_point_solver(p,
                                        context = 'calibration',
                                        x0=p.guess,
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
        sol_c = var.var_from_vector(sol.x, p,context='calibration')    
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p) 
        m.compute_moments(sol_c,p)
        m.compute_moments_deviations()

        run_name = run+': '+mom_weight_to_change+str(m.weights_dict[mom_weight_to_change])
        runs.append(run_name)
        dic_m[run_name] = m
        dic_p[run_name] = p
        dic_sol[run_name] = sol_c
        dic_values[run_name] = m.weights_dict[mom_weight_to_change]
        dic_change[run_name] = (m.weights_dict[mom_weight_to_change]-baseline_weight)*100/baseline_weight
    
    variation_dic['changing_quantity'] = mom_weight_to_change+'_weight'
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_weight
    
    dic_of_weights_variation_dic[mom_weight_to_change+'_weight'] = variation_dic
            
#%% write the tables to be used by bokeh

make_dirs([sensitivity_tables_path])

df_dic = {}

for s_spec_par in ['theta','fe','fo','zeta','nu']:
    list_of_dfs = []
    for qty,variation_dic in dic_of_variation_dics.items():
        df = pd.DataFrame()
        df['Change'] = [round(change) for change in variation_dic['change'].values()]
        df[qty] = [getattr(p,s_spec_par)[1] for p in variation_dic['p'].values()]
        list_of_dfs.append(df)
    big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
    df_dic[s_spec_par] = big_df
        
for scal_par in ['g_0','k','kappa']:
    list_of_dfs = []
    for qty,variation_dic in dic_of_variation_dics.items():
        df = pd.DataFrame()
        df['Change'] = [round(change) for change in variation_dic['change'].values()]
        df[qty] = [getattr(p,scal_par) for p in variation_dic['p'].values()]
        list_of_dfs.append(df)
    big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
    df_dic[scal_par] = big_df
    
        
for c_spec_par in ['delta']:
    for i,c in enumerate(['US','EUR']):
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic['delta '+c] = big_df
        
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1]/p.nu[1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic[c_spec_par+' '+c+' '+'over nu'] = big_df
              
for c_spec_par in ['eta']:
    for i,c in enumerate(['US']):
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic['eta_US'] = big_df

list_of_dfs = []
for qty,variation_dic in dic_of_variation_dics.items():
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [getattr(m,'RD_US') for m in variation_dic['m'].values()]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
df_dic['RD_US'] = big_df
        
# list_of_dfs = []
# for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
#     df = pd.DataFrame()
#     df['Change'] = [round(change) for change in variation_dic['change'].values()]
#     df[qty] = [compute_deriv_welfare_to_patent_protec_US(variation_dic['sol'][r],variation_dic['p'][r],v0=None) for r in variation_dic['p'].keys()]
#     list_of_dfs.append(df)
# big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
# df_dic['d_W_US_d_delta_US'] = big_df 

# list_of_dfs = []
# for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
#     df = pd.DataFrame()
#     df['Change'] = [round(change) for change in variation_dic['change'].values()]
#     df[qty] = [compute_deriv_growth_to_patent_protec_US(variation_dic['sol'][r],variation_dic['p'][r],v0=None) for r in variation_dic['p'].keys()]
#     list_of_dfs.append(df)
# big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
# df_dic['d_g_d_delta_US'] = big_df 

list_of_dfs = []
for qty,variation_dic in dic_of_variation_dics.items():
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [m.objective_function() for m in variation_dic['m'].values()]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
df_dic['objective'] = big_df 

# list_of_dfs = []
# for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
#     df = pd.DataFrame()
#     df['Change'] = [round(change) for change in variation_dic['change'].values()]
#     df[qty] = [m.ERDUS for m in variation_dic['m'].values()]
#     list_of_dfs.append(df)
# big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
# df_dic['ERDUS'] = big_df 
        
for k,df in df_dic.items():
    df.to_csv(sensitivity_tables_path+k+'.csv')

#%% tables of weight changes to be used by bokeh

make_dirs([sensitivity_weights_tables_path])

df_dic = {}

list_of_dfs = []
for qty,variation_dic in dic_of_weights_variation_dic.items():
    print(qty)
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [m.objective_function() for m in variation_dic['m'].values()]
    # df = df.loc[df['Change'] <= 65].loc[df['Change'] >= -65]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
# big_df.set_index('Change').plot(title='Objective function')
# plt.show()
df_dic['objective'] = big_df

for mom in m.list_of_moments:
    list_of_dfs = []
    for qty,variation_dic in dic_of_weights_variation_dic.items():
        if qty != 'OUT_weight':
            # list_of_dfs = []
            # print(qty)
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            # for mom in m.list_of_moments:
            df[qty] = [(getattr(m,mom+'_deviation')**2).sum()/
                       m.weights_dict[mom]**2/
                       sum([m.weights_dict[momm] for momm in m.list_of_moments])
                       for m in variation_dic['m'].values()]
            # df['objective'] = [m.objective_function() for m in variation_dic['m'].values()]
            # df = df.loc[df['Change'] <= 50].loc[df['Change'] >= -50]
            list_of_dfs.append(df)
    big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
    # print(mom,big_df)
    # fig,ax=plt.subplots()
    # big_df.set_index('Change').plot.bar(stacked=True,title=qty)
    # big_df.set_index('Change').plot(title=mom)
    # ax.plot(big_df.index,[m.objective_function() for m in variation_dic['m'].values()])
    # plt.plot(big_df.index,[m.objective_function() for m in variation_dic['m'].values()])
    # plt.show()
    df_dic[mom] = big_df

for k,df in df_dic.items():
    df.to_csv(sensitivity_weights_tables_path+k+'.csv')

#

# big_df.to_csv(sensitivity_tables_path+'d_W_US_d_delta_US'+'.csv')   
# big_df.to_csv(sensitivity_tables_path+'d_g_d_delta_US'+'.csv')   
# big_df.to_csv(sensitivity_tables_path+'objective'+'.csv')   
        