#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:13:37 2024

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

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

list_of_countries_to_run = ['USA']
# list_of_countries_to_run = ['KOR','MEX']
dynamics = True

p = p_baseline.copy()
for i,country in enumerate(p_baseline.countries):
        
    if country in list_of_countries_to_run:
        print(country)

        lb = p_baseline.eta[i,1]
        ub = p_baseline.eta[i,1]*2
        it = 0

        df = pd.DataFrame()
    
        while (ub-lb)/lb>1e-2:
            it = it+1
            x = (ub+lb)/2
            p = p_baseline.copy()
            p.eta[i,1] = x
            p.delta[:,1] = 12
            sol, dyn_sol = dyn_fixed_point_solver(p, 
                                                  sol_init=sol_baseline,
                                                  Nt=25,
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
            dyn_sol.compute_non_solver_quantities(p)
            dyn_sol.compute_consumption_equivalent_welfare(p)
            
            p.delta[i,1] = 0.01
            sol, dyn_sol_2 = dyn_fixed_point_solver(p, 
                                                    sol_init=sol_baseline,
                                                    Nt=25,
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
            dyn_sol_2.compute_non_solver_quantities(p)
            dyn_sol_2.compute_consumption_equivalent_welfare(p)
            
            if dyn_sol_2.cons_eq_welfare[i]>dyn_sol.cons_eq_welfare[i]:
                ub = x
                df.loc[it,f'eta_{country}'] = x
                for j,c in enumerate(p_baseline.countries):
                    df.loc[it,'delta_opti_'+c] = 12
                df.loc[it,'delta_opti_'+country] = 0.01
            else:
                lb = x
                df.loc[it,f'eta_{country}'] = x
                for j,c in enumerate(p_baseline.countries):
                    df.loc[it,'delta_opti_'+c] = 12
                df.loc[it,'delta_opti_'+country] = 12
            print(x,lb,ub)
            print(df)
            try:
                os.mkdir(f'solve_to_join_nash_club/eta/baseline_{baseline}/')
            except:
                pass
            df.to_csv(f'solve_to_join_nash_club/eta/baseline_{baseline}/nash_{country}.csv')