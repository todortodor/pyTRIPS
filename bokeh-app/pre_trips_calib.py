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


baseline_number = '618'

p_baseline = parameters()
p_baseline.correct_eur_patent_cost = True
p_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

_, sol_baseline = fixed_point_solver(p_baseline,context = 'calibration',x0=p_baseline.guess,
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

m_baseline = moments()
m_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')
m_baseline.compute_moments(sol_baseline,p_baseline)

inter_TP_target_baseline = m_baseline.inter_TP.copy()
inter_TP_data_baseline = m_baseline.inter_TP_data.copy()

TP_target_baseline = m_baseline.TP.copy()
TP_data_baseline = m_baseline.TP_data.copy()

runs_params = [
    # {
    #  'number': 1.0,
    #  'calib_params':p_baseline.calib_parameters,
    #  'list_of_moments':m_baseline.list_of_moments,
    #  'year':2005
    #  },
    # {
    #  'number': 1.1,
    #  'calib_params':p_baseline.calib_parameters,
    #  'list_of_moments':m_baseline.list_of_moments,
    #  'year':1992
    #  },
    # {
    #   'number': 2.0,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':2005
    #   },
    # {
    #   'number': 2.1,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':1992
    #   },
    # {
    #   'number': 3.0,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':2005
    #   },
    # {
    #   'number': 3.1,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':1992
    #   },
    # {
    #   'number': 4.0,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':2005
    #   },
    # {
    #   'number': 4.1,
    #   'calib_params':['eta','T','delta','fe','fo'],
    #   'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #     'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #   'year':1992
    #   },
    # {
    #  'number': 5.0,
    #  'calib_params':['eta','T','delta',],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    # {
    #  'number': 5.1,
    #  'calib_params':['eta','T','delta',],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':1992
    #  },
    # {
    #  'number': 6.0,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    # {
    #  'number': 6.1,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':1992
    #  },
    # {
    #  'number': 7.0,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    # {
    #  'number': 7.1,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #    'UUPCOST','SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':1992
    #  },
    # {
    #  'number': 8.0,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
    #    'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    {
     'number': 8.1,
     'calib_params':['eta','T','delta'],
     'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW',
       'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
     'year':1992
     },
    # {
    #  'number': 9.0,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
    #    'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    {
     'number': 9.1,
     'calib_params':['eta','T','delta'],
     'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','TP',
       'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
     'year':1992
     },
    # {
    #  'number': 10.0,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #    'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':2005
    #  },
    # {
    #  'number': 10.1,
    #  'calib_params':['eta','T','delta'],
    #  'list_of_moments':['OUT','RD','RP','SRGDP','SINNOVPATUS','SPFLOW','inter_TP',
    #    'SRDUS','SINNOVPATEU','DOMPATINUS','DOMPATINEU'],
    #  'year':1992
    #  },
    ]



for run_params in runs_params:
    print(run_params)
    baseline_dic = {'baseline':baseline_number,
                    'variation':str(run_params['number'])}
    year = run_params['year']
    
    p = p_baseline.copy()
    p.load_data(f'data/data_7_countries_{run_params["year"]}/',
                keep_already_calib_params=True)
    p.calib_parameters = run_params['calib_params']
    
    m = m_baseline.copy()
    m.load_data(f'data/data_7_countries_{run_params["year"]}/')
    m.TP_target = m_baseline.TP*m.TP_data/m_baseline.TP_data
    m.inter_TP_target = m_baseline.inter_TP*m.inter_TP_data/m_baseline.inter_TP_data
    m.list_of_moments = run_params['list_of_moments']
    # m.weights_dict['UUPCOST'] = 3
    
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
    
#%%
baseline_number = '618'

p_baseline = parameters()
p_baseline.correct_eur_patent_cost = True
p_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

_, sol_baseline = fixed_point_solver(p_baseline,context = 'calibration',x0=p_baseline.guess,
                        cobweb_anim=False,tol =1e-15,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
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

m_baseline = moments()
m_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')
m_baseline.compute_moments(sol_baseline,p_baseline)

import pandas as pd

recap = pd.DataFrame(
    index = pd.MultiIndex.from_product([[i for i in range(1,11)], p_baseline.countries],
                                       names = ['variation','country'])
    )

for i in [1,5,6,7,8,9,10]:
    print(i)
    p = parameters()
    p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{i}.0/')
    _, sol = fixed_point_solver(p,context = 'calibration',x0=p.guess,
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
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
    sol.scale_P(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{i}.0/')
    m.compute_moments(sol,p)
    
    p_pre = parameters()
    p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{i}.1/')
    _, sol_pre = fixed_point_solver(p_pre,context = 'calibration',x0=p_pre.guess,
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
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
    sol_pre.scale_P(p_pre)
    sol_pre.compute_non_solver_quantities(p_pre)
    m_pre = moments()
    m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{i}.1/')
    m_pre.compute_moments(sol_pre,p_pre)
    
    p_pre_cf = p.copy()
    p_pre_cf.delta[...,1] = p_pre.delta[...,1]
    
    _, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
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
    sol_pre_cf.scale_P(p_pre_cf)
    sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
    sol_pre_cf.compute_consumption_equivalent_welfare(p_pre_cf,sol_baseline)
    
    _, dyn_sol_pre_cf = dyn_fixed_point_solver(p_pre_cf, sol_baseline,sol_fin=sol_pre_cf,
                            Nt=25,t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=True,
                            damping = 50,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            )
    dyn_sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
    
    if 'fe' in p.calib_parameters:
        recap.loc[i,'fixed fe / fo'] = 'N'
    else:
        recap.loc[i,'fixed fe / fo'] = 'Y'
    if 'UUPCOST' in m.list_of_moments:
        recap.loc[i,'target UUPCOST'] = 'Y'
    else:
        recap.loc[i,'target UUPCOST'] = 'N'
    if 'TP' in m.list_of_moments:
        recap.loc[i,'target number of patents'] = 'All'
    elif 'inter_TP' in m.list_of_moments:
        recap.loc[i,'target number of patents'] = 'Inter. pats'
    else:
        recap.loc[i,'target number of patents'] = 'No'
    if i == 1: 
        recap.loc[i,'target number of patents'] = 'full calib'
        recap.loc[i,'fixed fe / fo'] = 'full calib'
        recap.loc[i,'target UUPCOST'] = 'full calib'
        
    recap.loc[i,'delta baseline'] = p_baseline.delta[...,1]
    recap.loc[i,'delta 2005'] = p.delta[...,1]
    recap.loc[i,'delta 2005/baseline'] = p.delta[...,1]/p_baseline.delta[...,1]
    recap.loc[i,'delta 1992'] = p_pre.delta[...,1]
    # recap.loc[i,'(std/mean)(1992)/(std/mean)(2005)'] = (p_pre.delta[...,1].std()/p_pre.delta[...,1].mean())/(p_baseline.delta[...,1].std()/p_baseline.delta[...,1].mean())
    recap.loc[i,'static welfare change'] = sol_pre_cf.cons_eq_welfare
    recap.loc[i,'dynamic welfare change'] = dyn_sol_pre_cf.cons_eq_welfare
    