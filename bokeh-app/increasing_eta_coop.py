#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:07:59 2023

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
from bokeh.palettes import Category10, Dark2
import time
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

baseline = '1300'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'

baseline_pre_trips_full_variation = baseline

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'

save_formats = ['eps','png','pdf']

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

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


#%%
dynamics = True

if __name__ == '__main__':
    
    #%% ETA
    # p = p_baseline.copy()
    # for coop in ['pop_weighted']:
    #     for i,country in enumerate(p_baseline.countries):
            
    #         if country in ['CHN','IND','RUS']:
    #             print(country)

    #             lb = p_baseline.eta[i,1]
    #             ub = p_baseline.eta[:,1].max()*6
    #             it = 0
                
    #             lb_delta = 0.01
    #             ub_delta = 12
                
    #             df = pd.DataFrame()
            
    #             while (ub-lb)/lb>1e-2:
    #                 it = it+1
    #                 x = (ub+lb)/2
    #                 p = p_baseline.copy()
    #                 p.eta[i,1] = x
    #                 sol, sol_c = fixed_point_solver(p,x0=p.guess,
    #                                                 context = 'counterfactual',
    #                                         cobweb_anim=False,tol =1e-14,
    #                                         accelerate=False,
    #                                         accelerate_when_stable=True,
    #                                         cobweb_qty='phi',
    #                                         plot_convergence=False,
    #                                         plot_cobweb=False,
    #                                         # plot_live=True,
    #                                         safe_convergence=0.001,
    #                                         disp_summary=False,
    #                                         damping = 500,
    #                                         max_count = 1e4,
    #                                         accel_memory = 50, 
    #                                         accel_type1=True, 
    #                                         accel_regularization=1e-10,
    #                                         accel_relaxation=0.5, 
    #                                         accel_safeguard_factor=1, 
    #                                         accel_max_weight_norm=1e6,
    #                                         damping_post_acceleration=5
    #                                         ) 
    #                 sol_c.scale_P(p)
    #                 sol_c.compute_non_solver_quantities(p)
    #                 print(lb,ub,x)
    #                 p.guess = sol.x 
    #                 p_opti, sol_opti = find_coop_eq(p,coop,
    #                                   lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
    #                                     # solver_options=None,
    #                                   tol=1e-6,
    #                                   custom_dyn_sol_options = None,
    #                                     solver_options = dict(cobweb_anim=False,tol =1e-14,
    #                                                             accelerate=False,
    #                                                             accelerate_when_stable=True,
    #                                                             cobweb_qty='phi',
    #                                                             plot_convergence=False,
    #                                                             plot_cobweb=False,
    #                                                             safe_convergence=0.001,
    #                                                             disp_summary=False,
    #                                                             damping = 50,
    #                                                             max_count = 1e4,
    #                                                             accel_memory = 50, 
    #                                                             accel_type1=True, 
    #                                                             accel_regularization=1e-10,
    #                                                             accel_relaxation=0.5, 
    #                                                             accel_safeguard_factor=1, 
    #                                                             accel_max_weight_norm=1e6,
    #                                                             damping_post_acceleration=20
    #                                                             ),
    #                                   custom_weights=None,
    #                                   max_workers=12,parallel=False)
    #                 if dynamics:
    #                     p_opti, sol_opti = find_coop_eq(p,coop,
    #                                      lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
    #                                      tol=1e-6,
    #                                         static_eq_deltas = p_opti.delta[...,1],
    #                                         #   custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-12,
    #                                         #       accelerate=False,
    #                                         #       accelerate_when_stable=False,
    #                                         #       cobweb_qty='l_R',
    #                                         #       plot_convergence=False,
    #                                         #       plot_cobweb=False,
    #                                         #       plot_live = False,
    #                                         #       safe_convergence=1e-8,
    #                                         #       disp_summary=False,
    #                                         #       damping = 500,
    #                                         #       max_count = 1000000,
    #                                         #       accel_memory =5, 
    #                                         #       accel_type1=True, 
    #                                         #       accel_regularization=1e-10,
    #                                         #       accel_relaxation=1, 
    #                                         #       accel_safeguard_factor=1, 
    #                                         #       accel_max_weight_norm=1e6,
    #                                         #       damping_post_acceleration=10),
    #                                         # solver_options = dict(cobweb_anim=False,tol =1e-12,
    #                                         #                         accelerate=False,
    #                                         #                         accelerate_when_stable=True,
    #                                         #                         cobweb_qty='phi',
    #                                         #                         plot_convergence=False,
    #                                         #                         plot_cobweb=False,
    #                                         #                         safe_convergence=0.001,
    #                                         #                         disp_summary=False,
    #                                         #                         damping = 50,
    #                                         #                         max_count = 1e6,
    #                                         #                         accel_memory = 50, 
    #                                         #                         accel_type1=True, 
    #                                         #                         accel_regularization=1e-10,
    #                                         #                         accel_relaxation=0.5, 
    #                                         #                         accel_safeguard_factor=1, 
    #                                         #                         accel_max_weight_norm=1e6,
    #                                         #                         damping_post_acceleration=20
    #                                         #                         ),
    #                                         custom_dyn_sol_options = None,
    #                                         solver_options=None,
    #                                      custom_weights=None,max_workers=12,displays=True,
    #                                      parallel=False)
    #                 if p_opti.delta[i,1]<p_baseline.delta[i,1]:
    #                     ub = x
    #                     df.loc[it,f'eta_{country}'] = x
    #                     for j,c in enumerate(p_baseline.countries):
    #                         df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
    #                 else:
    #                     lb = x
    #                     df.loc[it,f'eta_{country}'] = x
    #                     for j,c in enumerate(p_baseline.countries):
    #                         df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
    #                 print(x,lb,ub)
    #                 print(df)
    #                 try:
    #                     os.mkdir(f'solve_to_join_pat_club/eta/baseline_{baseline}/')
    #                 except:
    #                     pass
    #                 df.to_csv(f'solve_to_join_pat_club/eta/baseline_{baseline}/{coop}_{country}.csv')
    
    #%% T pat
    p = p_baseline.copy()
    for coop in ['pop_weighted']:
        for i,country in enumerate(p_baseline.countries):
            
            if country in ['CHN','IND','RUS']:
                print(country)
                
                lb = p_baseline.T[i,1]
                ub = p_baseline.T[:,1].max()*1e6
                it = 0
                
                lb_delta = 0.01
                ub_delta = 12
                
                df = pd.DataFrame()
            
                while (ub-lb)/lb>1:
                    it = it+1
                    x = np.sqrt(ub*lb)
                    p = p_baseline.copy()
                    p.T[i,1] = x
                    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                                    context = 'counterfactual',
                                            cobweb_anim=False,tol =1e-14,
                                            accelerate=False,
                                            accelerate_when_stable=True,
                                            cobweb_qty='phi',
                                            plot_convergence=False,
                                            plot_cobweb=False,
                                            # plot_live=True,
                                            safe_convergence=0.001,
                                            disp_summary=False,
                                            damping = 500,
                                            max_count = 1e4,
                                            accel_memory = 50, 
                                            accel_type1=True, 
                                            accel_regularization=1e-10,
                                            accel_relaxation=0.5, 
                                            accel_safeguard_factor=1, 
                                            accel_max_weight_norm=1e6,
                                            damping_post_acceleration=5
                                            ) 
                    sol_c.scale_P(p)
                    sol_c.compute_non_solver_quantities(p)
                    print(lb,ub,x)
                    p.guess = sol.x 
                    p_opti, sol_opti = find_coop_eq(p,coop,
                                      lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                                        # solver_options=None,
                                      tol=1e-6,
                                      custom_dyn_sol_options = None,
                                        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                                                accelerate=False,
                                                                accelerate_when_stable=True,
                                                                cobweb_qty='phi',
                                                                plot_convergence=False,
                                                                plot_cobweb=False,
                                                                safe_convergence=0.001,
                                                                disp_summary=False,
                                                                damping = 50,
                                                                max_count = 1e4,
                                                                accel_memory = 50, 
                                                                accel_type1=True, 
                                                                accel_regularization=1e-10,
                                                                accel_relaxation=0.5, 
                                                                accel_safeguard_factor=1, 
                                                                accel_max_weight_norm=1e6,
                                                                damping_post_acceleration=20
                                                                ),
                                      custom_weights=None,
                                      max_workers=12,parallel=False)
                    if dynamics:
                        p_opti, sol_opti = find_coop_eq(p,coop,
                                         lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
                                         tol=1e-6,
                                            static_eq_deltas = p_opti.delta[...,1],
                                            custom_dyn_sol_options = None,
                                            solver_options=None,
                                         custom_weights=None,max_workers=12,displays=True,
                                         parallel=False)
                    if p_opti.delta[i,1]<p_baseline.delta[i,1]:
                        ub = x
                    else:
                        lb = x
                    df.loc[it,f'T_pat_{country}'] = x
                    for j,c in enumerate(p_baseline.countries):
                        df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                    print(x,lb,ub)
                    print(df)
                    try:
                        os.mkdir(f'solve_to_join_pat_club/T_pat/baseline_{baseline}/')
                    except:
                        pass
                    df.to_csv(f'solve_to_join_pat_club/T_pat/baseline_{baseline}/{coop}_{country}.csv')
                    
    #%% tau in
    p = p_baseline.copy()
    for coop in ['pop_weighted']:
        for i,country in enumerate(p_baseline.countries):
            
            if country in ['CHN','IND','RUS']:
                print(country)
                
                lb = 0.1
                ub = 2
                it = 0
                
                lb_delta = 0.01
                ub_delta = 12
                
                df = pd.DataFrame()
            
                while (ub-lb)/lb>1e-3:
                    it = it+1
                    x = (ub+lb)/2
                    p = p_baseline.copy()
                    p.tau[i,:,1] = p_baseline.tau[i,:,1]*x
                    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                                    context = 'counterfactual',
                                            cobweb_anim=False,tol =1e-14,
                                            accelerate=False,
                                            accelerate_when_stable=True,
                                            cobweb_qty='phi',
                                            plot_convergence=False,
                                            plot_cobweb=False,
                                            # plot_live=True,
                                            safe_convergence=0.001,
                                            disp_summary=False,
                                            damping = 500,
                                            max_count = 1e4,
                                            accel_memory = 50, 
                                            accel_type1=True, 
                                            accel_regularization=1e-10,
                                            accel_relaxation=0.5, 
                                            accel_safeguard_factor=1, 
                                            accel_max_weight_norm=1e6,
                                            damping_post_acceleration=5
                                            ) 
                    sol_c.scale_P(p)
                    sol_c.compute_non_solver_quantities(p)
                    print(lb,ub,x)
                    p.guess = sol.x 
                    p_opti, sol_opti = find_coop_eq(p,coop,
                                      lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                                        # solver_options=None,
                                      tol=1e-6,
                                      custom_dyn_sol_options = None,
                                        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                                                accelerate=False,
                                                                accelerate_when_stable=True,
                                                                cobweb_qty='phi',
                                                                plot_convergence=False,
                                                                plot_cobweb=False,
                                                                safe_convergence=0.001,
                                                                disp_summary=False,
                                                                damping = 50,
                                                                max_count = 1e4,
                                                                accel_memory = 50, 
                                                                accel_type1=True, 
                                                                accel_regularization=1e-10,
                                                                accel_relaxation=0.5, 
                                                                accel_safeguard_factor=1, 
                                                                accel_max_weight_norm=1e6,
                                                                damping_post_acceleration=20
                                                                ),
                                      custom_weights=None,
                                      max_workers=12,parallel=False)
                    if dynamics:
                        p_opti, sol_opti = find_coop_eq(p,coop,
                                         lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
                                         tol=1e-6,
                                            static_eq_deltas = p_opti.delta[...,1],
                                            custom_dyn_sol_options = None,
                                            solver_options=None,
                                         custom_weights=None,max_workers=12,displays=True,
                                         parallel=False)
                    if p_opti.delta[i,1]<p_baseline.delta[i,1]:
                        ub = x
                        df.loc[it,f'tau_in_factor_{country}'] = x
                        for j,c in enumerate(p_baseline.countries):
                            df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                    else:
                        lb = x
                        df.loc[it,f'tau_in_factor_{country}'] = x
                        for j,c in enumerate(p_baseline.countries):
                            df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                    print(x,lb,ub)
                    print(df)
                    try:
                        os.mkdir(f'solve_to_join_pat_club/iceberg_trade_cost_in/baseline_{baseline}/')
                    except:
                        pass
                    df.to_csv(f'solve_to_join_pat_club/iceberg_trade_cost_in/baseline_{baseline}/{coop}_{country}.csv')
                    
    #%% tau out
    p = p_baseline.copy()
    for coop in ['pop_weighted']:
        for i,country in enumerate(p_baseline.countries):
            
            if country in ['CHN','IND','RUS']:
                print(country)
                
                lb = 0.1
                ub = 2
                it = 0
                
                lb_delta = 0.01
                ub_delta = 12
                
                df = pd.DataFrame()
            
                while (ub-lb)/lb>1e-3:
                    it = it+1
                    x = (ub+lb)/2
                    p = p_baseline.copy()
                    p.tau[:,i,1] = p_baseline.tau[:,i,1]*x
                    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                                    context = 'counterfactual',
                                            cobweb_anim=False,tol =1e-14,
                                            accelerate=False,
                                            accelerate_when_stable=True,
                                            cobweb_qty='phi',
                                            plot_convergence=False,
                                            plot_cobweb=False,
                                            # plot_live=True,
                                            safe_convergence=0.001,
                                            disp_summary=False,
                                            damping = 500,
                                            max_count = 1e4,
                                            accel_memory = 50, 
                                            accel_type1=True, 
                                            accel_regularization=1e-10,
                                            accel_relaxation=0.5, 
                                            accel_safeguard_factor=1, 
                                            accel_max_weight_norm=1e6,
                                            damping_post_acceleration=5
                                            ) 
                    sol_c.scale_P(p)
                    sol_c.compute_non_solver_quantities(p)
                    print(lb,ub,x)
                    p.guess = sol.x 
                    p_opti, sol_opti = find_coop_eq(p,coop,
                                      lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                                        # solver_options=None,
                                      tol=1e-6,
                                      custom_dyn_sol_options = None,
                                        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                                                accelerate=False,
                                                                accelerate_when_stable=True,
                                                                cobweb_qty='phi',
                                                                plot_convergence=False,
                                                                plot_cobweb=False,
                                                                safe_convergence=0.001,
                                                                disp_summary=False,
                                                                damping = 50,
                                                                max_count = 1e4,
                                                                accel_memory = 50, 
                                                                accel_type1=True, 
                                                                accel_regularization=1e-10,
                                                                accel_relaxation=0.5, 
                                                                accel_safeguard_factor=1, 
                                                                accel_max_weight_norm=1e6,
                                                                damping_post_acceleration=20
                                                                ),
                                      custom_weights=None,
                                      max_workers=12,parallel=False)
                    if dynamics:
                        p_opti, sol_opti = find_coop_eq(p,coop,
                                         lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
                                         tol=1e-6,
                                            static_eq_deltas = p_opti.delta[...,1],
                                            custom_dyn_sol_options = None,
                                            solver_options=None,
                                         custom_weights=None,max_workers=12,displays=True,
                                         parallel=False)
                    if p_opti.delta[i,1]<p_baseline.delta[i,1]:
                        ub = x
                        df.loc[it,f'tau_out_factor_{country}'] = x
                        for j,c in enumerate(p_baseline.countries):
                            df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                    else:
                        lb = x
                        df.loc[it,f'tau_out_factor_{country}'] = x
                        for j,c in enumerate(p_baseline.countries):
                            df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                    print(x,lb,ub)
                    print(df)
                    try:
                        os.mkdir(f'solve_to_join_pat_club/iceberg_trade_cost_out/baseline_{baseline}/')
                    except:
                        pass
                    df.to_csv(f'solve_to_join_pat_club/iceberg_trade_cost_out/baseline_{baseline}/{coop}_{country}.csv')
