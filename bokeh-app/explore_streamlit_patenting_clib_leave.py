#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:26:10 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq, find_nash_eq
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
# from bokeh.palettes import Category10, Dark2
import time
# Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

baseline = '1300'
variation = 'baseline'

baseline_pre_trips_full_variation = baseline

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

m_baseline = moments()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, 
                                   p_baseline, 
                                   compute=True, 
                                   context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

tau_factor_dic = {'baseline':1,'high':2}
eta_factor_dic = {'low':0.001,'baseline':1}

path = 'countries_leave_pat_club/baseline_1300/'

def try_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
try_mkdir(path)

for country in p_baseline.countries:
    if country in ['CAN','KOR','MEX']:
        #'USA','EUR','JAP',
        country_path = path+country+'/'
        idx_country = p_baseline.countries.index(country)
        try_mkdir(country_path)
        print('country', country)
        for tau_factor in tau_factor_dic:
            tau_path = country_path+tau_factor+'_tau/'
            try_mkdir(tau_path)
            print('tau factor', tau_factor)
            for eta_factor in eta_factor_dic:
                final_path = tau_path+eta_factor+'_eta/'
                try_mkdir(final_path)
                print('eta factor', eta_factor)
                
                p = p_baseline.copy()
                
                p.eta[idx_country,1] = eta_factor_dic[eta_factor] * p_baseline.eta[idx_country,1]
                p.tau[idx_country,:,1] = tau_factor_dic[tau_factor]*p_baseline.tau[idx_country,:,1]
                p.tau[idx_country,idx_country,:] = 1
                
                print('solving base')
                sol, sol_c = fixed_point_solver(p,
                                        x0=p.guess,
                                        context = 'counterfactual',
                                        cobweb_anim=False,tol =0.5e-12,
                                        accelerate=False,
                                        accelerate_when_stable=False,
                                        cobweb_qty='phi',
                                        plot_convergence=True,
                                        plot_cobweb=False,
                                        # plot_live=True,
                                        safe_convergence=0.001,
                                        disp_summary=False,
                                        damping = 50,
                                        max_count = 1e5,
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
                # assert sol.status = 'success'
                p.guess = sol_c.vector_from_var()
                
                p.write_params(final_path+'baseline/')
                
                p_protec = p.copy()
                p_protec.delta[:,1] = np.array([1.00e-02, 1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01,
                       1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01, 1.20e+01])
                p_protec.delta[idx_country,1] = 0.01
                print('solving protec')
                sol, sol_protec = fixed_point_solver(p_protec,
                                                x0=p_protec.guess,
                                                context = 'counterfactual',
                                        cobweb_anim=False,tol =1e-14,
                                        accelerate=False,
                                        accelerate_when_stable=True,
                                        cobweb_qty='phi',
                                        plot_convergence=True,
                                        plot_cobweb=False,
                                        # plot_live=True,
                                        safe_convergence=0.001,
                                        disp_summary=False,
                                        damping = 20,
                                        max_count = 1e5,
                                        accel_memory = 50, 
                                        accel_type1=True, 
                                        accel_regularization=1e-10,
                                        accel_relaxation=0.5, 
                                        accel_safeguard_factor=1, 
                                        accel_max_weight_norm=1e6,
                                        damping_post_acceleration=2
                                        )
                sol_protec.scale_P(p_protec)
                sol_protec.compute_non_solver_quantities(p_protec)
                p_protec.guess = sol_protec.vector_from_var()
                
                p_protec.write_params(final_path+'protec/')
                
                p_no_protec = p.copy()
                p_no_protec.delta[:,1] = np.array([0.01, 1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01,
                       1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01, 1.20e+01])
                p_no_protec.delta[idx_country,1] = 12
                print('solving no protec')
                sol, sol_no_protec = fixed_point_solver(p_no_protec,
                                                x0=p_no_protec.guess,
                                                context = 'counterfactual',
                                        cobweb_anim=False,tol =1e-14,
                                        accelerate=False,
                                        accelerate_when_stable=True,
                                        cobweb_qty='phi',
                                        plot_convergence=True,
                                        plot_cobweb=False,
                                        # plot_live=True,
                                        safe_convergence=0.001,
                                        disp_summary=False,
                                        damping = 20,
                                        max_count = 1e5,
                                        accel_memory = 50, 
                                        accel_type1=True, 
                                        accel_regularization=1e-10,
                                        accel_relaxation=0.5, 
                                        accel_safeguard_factor=1, 
                                        accel_max_weight_norm=1e6,
                                        damping_post_acceleration=2
                                        )
                sol_no_protec.scale_P(p_no_protec)
                sol_no_protec.compute_non_solver_quantities(p_no_protec)
                p_no_protec.guess = sol_no_protec.vector_from_var()
                
                p_no_protec.write_params(final_path+'no_protec/')