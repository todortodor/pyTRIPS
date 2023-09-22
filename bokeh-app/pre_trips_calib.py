#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:19:12 2023

@author: slepot
"""

from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver, dyn_fixed_point_solver
from data_funcs import write_calibration_results
import os
import numpy as np


baseline_number = '1030'

p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

# _, sol_baseline = fixed_point_solver(p_baseline,context = 'calibration',x0=p_baseline.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=True,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_baseline.scale_P(p_baseline)
# sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline = moments()
m_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

#%%

runs_params = [
    # {
    #   'number': 3.0,
    #   'calib_params':p_baseline.calib_parameters,
    #   'list_of_moments':m_baseline.list_of_moments,
    #   'year':2015
    #   },
    # {
    #   'number': 3.1,
    #   'calib_params':p_baseline.calib_parameters,
    #   'list_of_moments':m_baseline.list_of_moments,
    #   'year':1992
    #   },
    # {
    #   'number': 9.0,
    #   'calib_params':['delta','T','eta'],
    #   'list_of_moments':['SPFLOW','DOMPATINUS','OUT','RD','RP','SRGDP','UUPCOST'],
    #   'year':2015
    #   },
    # {
    #   'number': 9.1,
    #   'calib_params':['delta','T','eta'],
    #   'list_of_moments':['SPFLOW','DOMPATINUS','OUT','RD','RP','SRGDP','UUPCOST'],
    #   'year':1992
    #   },
    {
      'number': 9.2,
      'calib_params':['delta','T','eta'],
      'list_of_moments':['SPFLOW','DOMPATINUS','OUT','RD','RP','SRGDP','UUPCOST'],
      'year':1992
      },
    ]

for run_params in runs_params:
    print(run_params)
    baseline_dic = {'baseline':baseline_number,
                    'variation':str(run_params['number'])}
    year = run_params['year']
    
    p = p_baseline.copy()
    if run_params['number'] == 9.2:
        p.load_data(f'data_smooth_3_years/data_11_countries_{run_params["year"]}/',
                    keep_already_calib_params=True)
    else:
        p.load_data(f'data/data_11_countries_{run_params["year"]}/',
                    keep_already_calib_params=True)
    p.calib_parameters = run_params['calib_params']
    
    m = m_baseline.copy()
    m.load_data(f'data/data_11_countries_{run_params["year"]}/')
    print(m.data_path)
    m.list_of_moments = run_params['list_of_moments']
    
    hist = history(*tuple(m.list_of_moments+['objective']))
    bounds = p.make_parameters_bounds()
    start_time = time.perf_counter()
    cond = True
    iterations = 0
    max_iter = 5
    
    while cond:
        if iterations < max_iter-2:
            test_ls = optimize.least_squares(fun = calibration_func,    
                                    x0 = p.make_p_vector(), 
                                    args = (p,m,p.guess,hist,start_time), 
                                    bounds = bounds,
                                    max_nfev=1e8,
                                    xtol=1e-10, 
                                    verbose = 2)
        else:
            test_ls = optimize.least_squares(fun = calibration_func,    
                                    x0 = p.make_p_vector(), 
                                    args = (p,m,p.guess,hist,start_time), 
                                    bounds = bounds,
                                    max_nfev=1e8,
                                    xtol=1e-16, 
                                    verbose = 2)
        cond = iterations < max_iter
        iterations += 1
        p.update_parameters(test_ls.x)
    
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
    p_sol.guess = sol.x 
    sol_c.scale_P(p_sol)
    sol_c.compute_non_solver_quantities(p_sol) 
    p_sol.tau = sol_c.tau
    m.compute_moments(sol_c,p_sol)
    m.compute_moments_deviations()
    m.plot_moments(m.list_of_moments)
    
    ##%% writing results as excel and locally
    commentary = ''
    dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
    local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
    run_number = baseline_dic['variation']
    path = dropbox_path+'baseline_'+baseline_number+'_variations/'
        
    try:
        os.mkdir(path)
    except:
        pass
    
    write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
    
    try:
        os.mkdir(local_path)
    except:
        pass
    p_sol.write_params(local_path+str(run_number)+'/')
    m.write_moments(local_path+str(run_number)+'/')

#%% save a version of calibration with doubled trade costs in patenting sector

baseline_number = 1030

p_alt_trade_costs = p_baseline.copy()
p_alt_trade_costs.tau[...,1] = p_baseline.tau[...,1]*2
for j,_ in enumerate(p_baseline.countries):
    p_alt_trade_costs.tau[j,j,:] = 1

_, sol_alt_trade_costs = fixed_point_solver(p_alt_trade_costs,context = 'calibration',
                        x0=p_alt_trade_costs.guess,
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
sol_alt_trade_costs.scale_P(p_alt_trade_costs)
sol_alt_trade_costs.compute_non_solver_quantities(p_alt_trade_costs)

p_alt_trade_costs.guess = sol_alt_trade_costs.vector_from_var()

m_alt_trade_costs = moments()
m_alt_trade_costs.load_run('calibration_results_matched_economy/'+str(baseline_number)+'/')
m_alt_trade_costs.compute_moments(sol_alt_trade_costs,p_alt_trade_costs)
m_alt_trade_costs.compute_moments_deviations()

local_path = 'calibration_results_matched_economy/baseline_'+str(baseline_number)+'_variations/'
run_number = 10.2

p_alt_trade_costs.write_params(local_path+str(run_number)+'/')
m_alt_trade_costs.write_moments(local_path+str(run_number)+'/')

#%% save a version of calibration with 1992 trade costs

# baseline_number = 1020
# pre_trips_variation = 9.2

# p_pre = parameters()
# p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{pre_trips_variation}/')

# p_alt_trade_costs = p_baseline.copy()
# p_alt_trade_costs.tau = p_pre.tau.copy()

# _, sol_alt_trade_costs = fixed_point_solver(p_alt_trade_costs,context = 'calibration',
#                         x0=p_alt_trade_costs.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=True,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_alt_trade_costs.scale_P(p_alt_trade_costs)
# sol_alt_trade_costs.compute_non_solver_quantities(p_alt_trade_costs)

# p_alt_trade_costs.guess = sol_alt_trade_costs.vector_from_var()

# m_alt_trade_costs = moments()
# m_alt_trade_costs.load_run('calibration_results_matched_economy/'+str(baseline_number)+'/')
# m_alt_trade_costs.compute_moments(sol_alt_trade_costs,p_alt_trade_costs)
# m_alt_trade_costs.compute_moments_deviations()

# local_path = 'calibration_results_matched_economy/baseline_'+str(baseline_number)+'_variations/'
# run_number = 10.1

# p_alt_trade_costs.write_params(local_path+str(run_number)+'/')
# m_alt_trade_costs.write_moments(local_path+str(run_number)+'/')
