#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:20:43 2024

@author: slepot
"""

#%% One delta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
import sys

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'

baseline_dics = [
    # {'baseline':'2000','variation': 'baseline'},
    {'baseline':'2000','variation': '14.0'},
    # {'baseline':'2000','variation': '99.0'},
    # {'baseline':'2000','variation': '99.1'},
    # {'baseline':'2000','variation': '99.2'},
    # {'baseline':'2000','variation': '99.3'},
    # {'baseline':'2000','variation': '99.4'},
    # {'baseline':'2000','variation': '99.5'},
    # {'baseline':'2000','variation': '99.6'},
    # {'baseline':'2000','variation': '99.7'},
    # {'baseline':'2000','variation': '99.8'},
    # {'baseline':'2000','variation': '99.9'},
    # {'baseline':'2000','variation': '99.10'},
    # {'baseline':'2000','variation': '99.11'},
    # {'baseline':'2000','variation': '99.12'},
    # {'baseline':'2000','variation': '99.13'},
    # {'baseline':'2000','variation': '99.14'},
    # {'baseline':'2000','variation': '99.15'},
    ]

lb_delta = 0.01
ub_delta = 12

for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    assert os.path.exists(baseline_path), 'run doesnt exist'
    
    print(baseline_path)
    p_baseline = parameters()
    p_baseline.load_run(baseline_path)
    
    sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)
    
    for aggregation_method in ['negishi','pop_weighted']:
    # for aggregation_method in ['pop_weighted']:

        # deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(
        #     ['baseline','variation','aggregation_method'],keep='last')
        # deltas = deltas.loc[
        #     (deltas.baseline.astype('str') == baseline_dic['baseline'])
        #     & (deltas.variation.astype('str') == baseline_dic['variation'])
        #     & (deltas.aggregation_method == 'pop_weighted')][p_baseline.countries].values.squeeze()
        # deltas[deltas>0.9] = ub_delta

        # p_opti = p_baseline.copy()
        # p_opti.delta[...,1] = deltas
        
        # p_opti.dyn_guess = None
        
        direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
        
        p_opti = parameters()
        p_opti.load_run(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')
        # p_opti.load_run(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')
        
        solver_options = dict(cobweb_anim=False,tol =1e-14,
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
        custom_sol_options = solver_options
        
        dyn_solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=True,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 80,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10)
        custom_dyn_sol_options = dyn_solver_options
        
        # sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_opti, sol_baseline, Nt=25,
        #                                       t_inf=500,
        #                         cobweb_anim=False,tol =1e-14,
        #                         accelerate=False,
        #                         accelerate_when_stable=False,
        #                         cobweb_qty='l_R',
        #                         plot_convergence=True,
        #                         plot_cobweb=False,
        #                         plot_live = False,
        #                         safe_convergence=1e-8,
        #                         disp_summary=True,
        #                         damping = 60,
        #                         max_count = 50000,
        #                         accel_memory =5, 
        #                         accel_type1=True, 
        #                         accel_regularization=1e-10,
        #                         accel_relaxation=1, 
        #                         accel_safeguard_factor=1, 
        #                         accel_max_weight_norm=1e6,
        #                         damping_post_acceleration=5
        #                         )
        sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_opti, sol_init=sol_baseline,Nt=25,
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
        dyn_sol_coop_equal.compute_non_solver_quantities(p_opti)
        
        #%%
        
        sol_opti = dyn_sol_coop_equal.copy()
        p = p_opti.copy()
        
        if aggregation_method == 'negishi':
            solution_welfare = sol_opti.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            solution_welfare = sol_opti.cons_eq_pop_average_welfare_change
        
        corner_corrected_deltas = p.delta[...,1:].copy()
        for sector in range(1,p.S):
            for i,c in enumerate(p_baseline.countries):
                # if p.delta[i,sector] < 2*lb_delta or c=='MEX':
                # if p.delta[i,sector] < 0.07 or c=='MEX':
                if True:
                    print(
                        pd.DataFrame(index=p.countries,
                                     columns=p.sectors[1:],
                                     data=p.delta[:,1:])
                                     )
                    print('checking on ',c)
                    p_corner = p.copy()
                    p_corner.delta[i,sector] = lb_delta
                    
                    sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
                                                    context = 'counterfactual',
                                                    **solver_options
                                                    )
                    sol_corner.compute_non_solver_quantities(p_corner)
                    sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
                    sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
                    sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                                 sol_fin=sol_corner,
                                                                 Nt=25,
                                                          t_inf=500,
                                            **custom_dyn_sol_options
                                            )
            
                    dyn_sol_corner.compute_non_solver_quantities(p)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
                    print(corner_welfare,solution_welfare)
                    if corner_welfare > solution_welfare:
                        print('lower corner was better for ',c)
                        corner_corrected_deltas[i,sector-1] = lb_delta
    
        p.delta[...,1:] = corner_corrected_deltas
        
        
        sol, sol_c = fixed_point_solver(p_corner,x0=p_corner.guess,
                                        context = 'counterfactual',
                                        **solver_options
                                        )
        sol_c.compute_non_solver_quantities(p_corner)
        sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
        sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
        
        if aggregation_method == 'negishi':
            solution_welfare = sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            solution_welfare = sol_c.cons_eq_pop_average_welfare_change
        
        sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
                                                     Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
    
        dyn_sol.compute_non_solver_quantities(p)
        
        if aggregation_method == 'negishi':
            solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
        
        # corner_corrected_deltas = p.delta[...,1].copy()
        for sector in range(1,p.S):
            print(sector)
            for i,c in enumerate(p_baseline.countries):
                # if p.delta[i,sector] > 0.1 or c=='MEX':
                if True:
                    p_corner = p.copy()
                    print(
                        pd.DataFrame(index=p.countries,
                                     columns=p.sectors[1:],
                                     data=p.delta[:,1:])
                                     )
                    print('checking on ',c)
                    p_corner.delta[i,sector] = ub_delta
                    
                    sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
                                                    context = 'counterfactual',
                                                    **solver_options
                                                    )
                    sol_corner.compute_non_solver_quantities(p_corner)
                    sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
                    sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
                    sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                                 sol_fin=sol_corner,
                                                                 Nt=23,
                                                          t_inf=500,
                                            **custom_dyn_sol_options
                                            )
            
                    dyn_sol_corner.compute_non_solver_quantities(p)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
                    print(corner_welfare,solution_welfare)
                    if corner_welfare > solution_welfare:
                        print('upper corner was better for ',c)
                        corner_corrected_deltas[i,sector-1] = ub_delta
                
        p.delta[...,1:] = corner_corrected_deltas
        
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                        context = 'counterfactual',
                                **solver_options
                                )
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
        sol_c.compute_world_welfare_changes(p,sol_baseline)
        
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        
        p_opti = p.copy()
        sol_opti = dyn_sol_c.copy()
        
        write = False
        if write:
            if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+p_opti.delta[...,1].tolist(), 
                            # index = deltas_df.columns).T
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
            if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
                cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
            cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
                                                               sol_opti.cons_eq_negishi_welfare_change], 
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
        
        save_directly = True
        if save_directly:
            direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
            p_opti.write_params(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')
          
#%% Delta int
          
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# from classes import moments, parameters, var_double_diff_double_delta, dynamic_var_double_diff_double_delta
# from solver_funcs import fixed_point_solver_double_diff_double_delta, dyn_fixed_point_solver_double_diff_double_delta
# import matplotlib.pylab as pylab
# from data_funcs import write_calibration_results
# import seaborn as sns

# data_path = 'data/'
# results_path = 'calibration_results_matched_economy/'

# baseline_dics = [
#     {'baseline':'2000','variation': 'baseline'},
#     # {'baseline':'1312','variation': '1.07'},
#     # {'baseline':'1312','variation': '2.02'},
#     # {'baseline':'1312','variation': '2.07'},
#     # {'baseline':'1300','variation': '2.0'},
#     # {'baseline':'1300','variation': '10.2'},
#     # {'baseline':'1300','variation': '10.3'},
#     # {'baseline':'1300','variation': '10.4'},
#     # {'baseline':'1300','variation': '10.5'},
#     # {'baseline':'1300','variation': '12.0'},
#     # {'baseline':'1300','variation': '13.0'},
#     # {'baseline':'1300','variation': '99.0'},
#     # {'baseline':'1300','variation': '99.1'},
#     # {'baseline':'1300','variation': '99.2'},
#     # {'baseline':'1300','variation': '99.3'},
#     # {'baseline':'1300','variation': '99.4'},
#     # {'baseline':'1300','variation': '99.5'},
#     # {'baseline':'1300','variation': '99.6'},
#     # {'baseline':'1300','variation': '99.7'},
#     # {'baseline':'1300','variation': '99.8'},
#     # {'baseline':'1300','variation': '99.9'},
#     # {'baseline':'1300','variation': '99.10'},
#     # {'baseline':'1300','variation': '99.11'},
#     # {'baseline':'1300','variation': '99.12'},
#     # {'baseline':'1300','variation': '99.13'},
#     # {'baseline':'1300','variation': '99.14'},
#     # {'baseline':'1300','variation': '99.15'},
#     # {'baseline':'4003','variation': 'baseline'},
#     # {'baseline':'4004','variation': 'baseline'},
#     # {'baseline':'6001','variation': '4.02'},
#     ]

# lb_delta = 0.01
# ub_delta = 12

# for baseline_dic in baseline_dics:    
#     if baseline_dic['variation'] == 'baseline':
#         baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
#     else:
#         baseline_path = \
#             f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
#     assert os.path.exists(baseline_path), 'run doesnt exist'
    
#     print(baseline_path)
#     p_baseline = parameters()
#     p_baseline.load_run(baseline_path)
    
#     sol_baseline = var_double_diff_double_delta.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
#     sol_baseline.scale_P(p_baseline)
#     sol_baseline.compute_non_solver_quantities(p_baseline)
    
#     for aggregation_method in ['negishi','pop_weighted']:
#     # for aggregation_method in ['pop_weighted']:

#         # deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(
#         #     ['baseline','variation','aggregation_method'],keep='last')
#         # deltas = deltas.loc[
#         #     (deltas.baseline.astype('str') == baseline_dic['baseline'])
#         #     & (deltas.variation.astype('str') == baseline_dic['variation'])
#         #     & (deltas.aggregation_method == 'pop_weighted')][p_baseline.countries].values.squeeze()
#         # deltas[deltas>0.9] = ub_delta

#         # p_opti = p_baseline.copy()
#         # p_opti.delta[...,1] = deltas
        
#         direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
        
#         p_opti = parameters()
#         p_opti.load_run(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')
#         # p_opti.load_run(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')
        
#         solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=True,
#                                 cobweb_qty='phi',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 safe_convergence=0.001,
#                                 disp_summary=False,
#                                 damping = 5,
#                                 max_count = 1e4,
#                                 accel_memory = 50, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=0.5, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=2
#                                 )
#         custom_sol_options = solver_options
        
#         dyn_solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10)
#         custom_dyn_sol_options = dyn_solver_options
        
#         sol, dyn_sol_coop_equal = dyn_fixed_point_solver_double_diff_double_delta(p_opti, sol_init=sol_baseline,Nt=25,
#                                               t_inf=500,
#                                 cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10
#                                 )
#         dyn_sol_coop_equal.compute_non_solver_quantities(p_opti)
        
#         sol_opti = dyn_sol_coop_equal.copy()
#         p = p_opti.copy()
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_opti.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_opti.cons_eq_pop_average_welfare_change
        
#         corner_corrected_deltas = p.delta_int[...,1:].copy()
#         for sector in range(1,p.S):
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] < 2*lb_delta or c=='MEX':
#                 # if p.delta[i,sector] < 0.07 or c=='MEX':
#                 if True:
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_int[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner = p.copy()
#                     p_corner.delta_int[i,sector] = lb_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('lower corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = lb_delta
    
#         p.delta_int[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                         context = 'counterfactual',
#                                         **solver_options
#                                         )
#         sol_c.compute_non_solver_quantities(p_corner)
#         sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#         sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_c.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_c.cons_eq_pop_average_welfare_change
        
#         sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, 
#                                                      Nt=23,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
    
#         dyn_sol.compute_non_solver_quantities(p)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
        
#         # corner_corrected_deltas = p.delta[...,1].copy()
#         for sector in range(1,p.S):
#             print(sector)
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] > 0.1 or c=='MEX':
#                 if True:
#                     p_corner = p.copy()
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_int[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner.delta_int[i,sector] = ub_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('upper corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = ub_delta
                
#         p.delta_int[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
#                                         context = 'counterfactual',
#                                 **solver_options
#                                 )
#         sol_c.scale_P(p)
#         sol_c.compute_non_solver_quantities(p)
#         sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
#         sol_c.compute_world_welfare_changes(p,sol_baseline)
        
#         sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p,  sol_baseline, sol_fin=sol_c, Nt=25,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
#         dyn_sol_c.compute_non_solver_quantities(p)
        
#         p_opti = p.copy()
#         sol_opti = dyn_sol_c.copy()
        
#         write = True
#         if write:
#             if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
#                 deltas_df = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries)
#                 deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
#             deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+p_opti.delta[...,1].tolist(), 
#                             # index = deltas_df.columns).T
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries).T
#             deltas_df = pd.concat([deltas_df, run],ignore_index=True)
#             deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
#             if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
#                 cons_eq_welfares = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
#                 cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
#             cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
#                                                                sol_opti.cons_eq_negishi_welfare_change], 
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
#             cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
#             cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
        
#         save_directly = True
#         if save_directly:
#             direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
#             p_opti.write_params(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')

# #%% Delta dom
          
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# from classes import moments, parameters, var_double_diff_double_delta, dynamic_var_double_diff_double_delta
# from solver_funcs import fixed_point_solver_double_diff_double_delta, dyn_fixed_point_solver_double_diff_double_delta
# import matplotlib.pylab as pylab
# from data_funcs import write_calibration_results
# import seaborn as sns

# data_path = 'data/'
# results_path = 'calibration_results_matched_economy/'

# baseline_dics = [
#     # {'baseline':'1312','variation': 'baseline'},
#     {'baseline':'1312','variation': '1.07'},
#     {'baseline':'1312','variation': '2.02'},
#     {'baseline':'1312','variation': '2.07'},
#     # {'baseline':'1300','variation': '2.0'},
#     # {'baseline':'1300','variation': '10.2'},
#     # {'baseline':'1300','variation': '10.3'},
#     # {'baseline':'1300','variation': '10.4'},
#     # {'baseline':'1300','variation': '10.5'},
#     # {'baseline':'1300','variation': '12.0'},
#     # {'baseline':'1300','variation': '13.0'},
#     # {'baseline':'1300','variation': '99.0'},
#     # {'baseline':'1300','variation': '99.1'},
#     # {'baseline':'1300','variation': '99.2'},
#     # {'baseline':'1300','variation': '99.3'},
#     # {'baseline':'1300','variation': '99.4'},
#     # {'baseline':'1300','variation': '99.5'},
#     # {'baseline':'1300','variation': '99.6'},
#     # {'baseline':'1300','variation': '99.7'},
#     # {'baseline':'1300','variation': '99.8'},
#     # {'baseline':'1300','variation': '99.9'},
#     # {'baseline':'1300','variation': '99.10'},
#     # {'baseline':'1300','variation': '99.11'},
#     # {'baseline':'1300','variation': '99.12'},
#     # {'baseline':'1300','variation': '99.13'},
#     # {'baseline':'1300','variation': '99.14'},
#     # {'baseline':'1300','variation': '99.15'},
#     # {'baseline':'4003','variation': 'baseline'},
#     # {'baseline':'4004','variation': 'baseline'},
#     # {'baseline':'6001','variation': '4.02'},
#     ]

# lb_delta = 0.01
# ub_delta = 12

# for baseline_dic in baseline_dics:    
#     if baseline_dic['variation'] == 'baseline':
#         baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
#     else:
#         baseline_path = \
#             f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
#     assert os.path.exists(baseline_path), 'run doesnt exist'
    
#     print(baseline_path)
#     p_baseline = parameters()
#     p_baseline.load_run(baseline_path)
    
#     sol_baseline = var_double_diff_double_delta.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
#     sol_baseline.scale_P(p_baseline)
#     sol_baseline.compute_non_solver_quantities(p_baseline)
    
#     for aggregation_method in ['negishi','pop_weighted']:
#     # for aggregation_method in ['pop_weighted']:

#         # deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(
#         #     ['baseline','variation','aggregation_method'],keep='last')
#         # deltas = deltas.loc[
#         #     (deltas.baseline.astype('str') == baseline_dic['baseline'])
#         #     & (deltas.variation.astype('str') == baseline_dic['variation'])
#         #     & (deltas.aggregation_method == 'pop_weighted')][p_baseline.countries].values.squeeze()
#         # deltas[deltas>0.9] = ub_delta

#         # p_opti = p_baseline.copy()
#         # p_opti.delta[...,1] = deltas
        
#         direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
        
#         p_opti = parameters()
#         p_opti.load_run(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')
#         # p_opti.load_run(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')
        
#         solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=True,
#                                 cobweb_qty='phi',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 safe_convergence=0.001,
#                                 disp_summary=False,
#                                 damping = 5,
#                                 max_count = 1e4,
#                                 accel_memory = 50, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=0.5, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=2
#                                 )
#         custom_sol_options = solver_options
        
#         dyn_solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10)
#         custom_dyn_sol_options = dyn_solver_options
        
#         sol, dyn_sol_coop_equal = dyn_fixed_point_solver_double_diff_double_delta(p_opti, sol_init=sol_baseline,Nt=25,
#                                               t_inf=500,
#                                 cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10
#                                 )
#         dyn_sol_coop_equal.compute_non_solver_quantities(p_opti)
        
#         sol_opti = dyn_sol_coop_equal.copy()
#         p = p_opti.copy()
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_opti.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_opti.cons_eq_pop_average_welfare_change
        
#         corner_corrected_deltas = p.delta_dom[...,1:].copy()
#         for sector in range(1,p.S):
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] < 2*lb_delta or c=='MEX':
#                 # if p.delta[i,sector] < 0.07 or c=='MEX':
#                 if True:
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_dom[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner = p.copy()
#                     p_corner.delta_dom[i,sector] = lb_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('lower corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = lb_delta
    
#         p.delta_dom[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                         context = 'counterfactual',
#                                         **solver_options
#                                         )
#         sol_c.compute_non_solver_quantities(p_corner)
#         sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#         sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_c.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_c.cons_eq_pop_average_welfare_change
        
#         sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, 
#                                                      Nt=23,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
    
#         dyn_sol.compute_non_solver_quantities(p)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
        
#         # corner_corrected_deltas = p.delta[...,1].copy()
#         for sector in range(1,p.S):
#             print(sector)
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] > 0.1 or c=='MEX':
#                 if True:
#                     p_corner = p.copy()
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_dom[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner.delta_dom[i,sector] = ub_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('upper corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = ub_delta
                
#         p.delta_dom[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
#                                         context = 'counterfactual',
#                                 **solver_options
#                                 )
#         sol_c.scale_P(p)
#         sol_c.compute_non_solver_quantities(p)
#         sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
#         sol_c.compute_world_welfare_changes(p,sol_baseline)
        
#         sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p,  sol_baseline, sol_fin=sol_c, Nt=25,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
#         dyn_sol_c.compute_non_solver_quantities(p)
        
#         p_opti = p.copy()
#         sol_opti = dyn_sol_c.copy()
        
#         write = False
#         if write:
#             if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
#                 deltas_df = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries)
#                 deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
#             deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+p_opti.delta[...,1].tolist(), 
#                             # index = deltas_df.columns).T
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries).T
#             deltas_df = pd.concat([deltas_df, run],ignore_index=True)
#             deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
#             if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
#                 cons_eq_welfares = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
#                 cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
#             cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
#                                                                sol_opti.cons_eq_negishi_welfare_change], 
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
#             cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
#             cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
        
#         save_directly = True
#         if save_directly:
#             direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
#             p_opti.write_params(f'coop_eq_direct_saves/dyn_{direct_save_path}_{aggregation_method}/')


# #%% Both deltas
          
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# from classes import moments, parameters, var_double_diff_double_delta, dynamic_var_double_diff_double_delta
# from solver_funcs import fixed_point_solver_double_diff_double_delta, dyn_fixed_point_solver_double_diff_double_delta
# import matplotlib.pylab as pylab
# from data_funcs import write_calibration_results
# import seaborn as sns

# data_path = 'data/'
# results_path = 'calibration_results_matched_economy/'

# baseline_dics = [
#     {'baseline':'1312','variation': 'baseline'},
#     # {'baseline':'1312','variation': '1.07'},
#     # {'baseline':'1312','variation': '2.02'},
#     # {'baseline':'1312','variation': '2.07'},
#     # {'baseline':'1300','variation': '2.0'},
#     # {'baseline':'1300','variation': '10.2'},
#     # {'baseline':'1300','variation': '10.3'},
#     # {'baseline':'1300','variation': '10.4'},
#     # {'baseline':'1300','variation': '10.5'},
#     # {'baseline':'1300','variation': '12.0'},
#     # {'baseline':'1300','variation': '13.0'},
#     # {'baseline':'1300','variation': '99.0'},
#     # {'baseline':'1300','variation': '99.1'},
#     # {'baseline':'1300','variation': '99.2'},
#     # {'baseline':'1300','variation': '99.3'},
#     # {'baseline':'1300','variation': '99.4'},
#     # {'baseline':'1300','variation': '99.5'},
#     # {'baseline':'1300','variation': '99.6'},
#     # {'baseline':'1300','variation': '99.7'},
#     # {'baseline':'1300','variation': '99.8'},
#     # {'baseline':'1300','variation': '99.9'},
#     # {'baseline':'1300','variation': '99.10'},
#     # {'baseline':'1300','variation': '99.11'},
#     # {'baseline':'1300','variation': '99.12'},
#     # {'baseline':'1300','variation': '99.13'},
#     # {'baseline':'1300','variation': '99.14'},
#     # {'baseline':'1300','variation': '99.15'},
#     # {'baseline':'4003','variation': 'baseline'},
#     # {'baseline':'4004','variation': 'baseline'},
#     # {'baseline':'6001','variation': '4.02'},
#     ]

# lb_delta = 0.01
# ub_delta = 12

# for baseline_dic in baseline_dics:    
#     if baseline_dic['variation'] == 'baseline':
#         baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
#     else:
#         baseline_path = \
#             f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
#     assert os.path.exists(baseline_path), 'run doesnt exist'
    
#     print(baseline_path)
#     p_baseline = parameters()
#     p_baseline.load_run(baseline_path)
    
#     sol_baseline = var_double_diff_double_delta.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
#     sol_baseline.scale_P(p_baseline)
#     sol_baseline.compute_non_solver_quantities(p_baseline)
    
#     for aggregation_method in ['negishi','pop_weighted']:
#     # for aggregation_method in ['pop_weighted']:

#         # deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(
#         #     ['baseline','variation','aggregation_method'],keep='last')
#         # deltas = deltas.loc[
#         #     (deltas.baseline.astype('str') == baseline_dic['baseline'])
#         #     & (deltas.variation.astype('str') == baseline_dic['variation'])
#         #     & (deltas.aggregation_method == 'pop_weighted')][p_baseline.countries].values.squeeze()
#         # deltas[deltas>0.9] = ub_delta

#         # p_opti = p_baseline.copy()
#         # p_opti.delta[...,1] = deltas
        
#         direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
        
#         p_opti = parameters()
#         p_opti.load_run(f'coop_eq_direct_saves/dyn_double_diff_{direct_save_path}_{aggregation_method}/')
#         # p_opti.load_run(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')
        
#         solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=True,
#                                 cobweb_qty='phi',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 safe_convergence=0.001,
#                                 disp_summary=False,
#                                 damping = 5,
#                                 max_count = 1e4,
#                                 accel_memory = 50, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=0.5, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=2
#                                 )
#         custom_sol_options = solver_options
        
#         dyn_solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10)
#         custom_dyn_sol_options = dyn_solver_options
        
#         sol, dyn_sol_coop_equal = dyn_fixed_point_solver_double_diff_double_delta(p_opti, sol_init=sol_baseline,Nt=25,
#                                               t_inf=500,
#                                 cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=False,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 plot_live = False,
#                                 safe_convergence=1e-8,
#                                 disp_summary=False,
#                                 damping = 60,
#                                 max_count = 50000,
#                                 accel_memory =5, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=1, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10
#                                 )
#         dyn_sol_coop_equal.compute_non_solver_quantities(p_opti)
        
#         sol_opti = dyn_sol_coop_equal.copy()
#         p = p_opti.copy()
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_opti.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_opti.cons_eq_pop_average_welfare_change
        
#         corner_corrected_deltas = p.delta_int[...,1:].copy()
#         for sector in range(1,p.S):
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] < 2*lb_delta or c=='MEX':
#                 # if p.delta[i,sector] < 0.07 or c=='MEX':
#                 if True:
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_int[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner = p.copy()
#                     p_corner.delta_int[i,sector] = lb_delta
#                     p_corner.delta_dom[i,sector] = lb_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('lower corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = lb_delta
    
#         p.delta_int[...,1:] = corner_corrected_deltas
#         p.delta_dom[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                         context = 'counterfactual',
#                                         **solver_options
#                                         )
#         sol_c.compute_non_solver_quantities(p_corner)
#         sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#         sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = sol_c.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = sol_c.cons_eq_pop_average_welfare_change
        
#         sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, 
#                                                      Nt=23,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
    
#         dyn_sol.compute_non_solver_quantities(p)
        
#         if aggregation_method == 'negishi':
#             solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
#         if aggregation_method == 'pop_weighted':
#             solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
        
#         # corner_corrected_deltas = p.delta[...,1].copy()
#         for sector in range(1,p.S):
#             print(sector)
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,sector] > 0.1 or c=='MEX':
#                 if True:
#                     p_corner = p.copy()
#                     print(
#                         pd.DataFrame(index=p.countries,
#                                      columns=p.sectors[1:],
#                                      data=p.delta_int[:,1:])
#                                      )
#                     print('checking on ',c)
#                     p_corner.delta_int[i,sector] = ub_delta
#                     p_corner.delta_dom[i,sector] = ub_delta
#                     p_corner.update_delta_eff()
                    
#                     sol, sol_corner = fixed_point_solver_double_diff_double_delta(p_corner,x0=p_corner.guess,
#                                                     context = 'counterfactual',
#                                                     **solver_options
#                                                     )
#                     sol_corner.compute_non_solver_quantities(p_corner)
#                     sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
#                     sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver_double_diff_double_delta(p_corner, sol_init=sol_baseline, 
#                                                                  sol_fin=sol_corner,
#                                                                  Nt=23,
#                                                           t_inf=500,
#                                             **custom_dyn_sol_options
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     print(corner_welfare,solution_welfare)
#                     if corner_welfare > solution_welfare:
#                         print('upper corner was better for ',c)
#                         corner_corrected_deltas[i,sector-1] = ub_delta
                
#         p.delta_int[...,1:] = corner_corrected_deltas
#         p.delta_dom[...,1:] = corner_corrected_deltas
#         p.update_delta_eff()
        
#         sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
#                                         context = 'counterfactual',
#                                 **solver_options
#                                 )
#         sol_c.scale_P(p)
#         sol_c.compute_non_solver_quantities(p)
#         sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
#         sol_c.compute_world_welfare_changes(p,sol_baseline)
        
#         sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p,  sol_baseline, sol_fin=sol_c, Nt=25,
#                                               t_inf=500,
#                                 **custom_dyn_sol_options
#                                 )
#         dyn_sol_c.compute_non_solver_quantities(p)
        
#         p_opti = p.copy()
#         sol_opti = dyn_sol_c.copy()
        
#         write = False
#         if write:
#             if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
#                 deltas_df = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries)
#                 deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
#             deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+p_opti.delta[...,1].tolist(), 
#                             # index = deltas_df.columns).T
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries).T
#             deltas_df = pd.concat([deltas_df, run],ignore_index=True)
#             deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
#             if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
#                 cons_eq_welfares = pd.DataFrame(columns = ['baseline',
#                                                 'variation',
#                                                 'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
#                 cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
#             cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
#             run = pd.DataFrame(data = [baseline_dic['baseline'],
#                             baseline_dic['variation'],
#                             aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
#                                                                sol_opti.cons_eq_negishi_welfare_change], 
#                             index = ['baseline',
#                                      'variation',
#                                      'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
#             cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
#             cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
        
#         save_directly = True
#         if save_directly:
#             direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
#             p_opti.write_params(f'coop_eq_direct_saves/dyn_double_diff_{direct_save_path}_{aggregation_method}/')


#%%

# p_cf = p_coop_equal.copy()
# # p_cf.delta[p_baseline.countries.index('MEX'),1] = p_cf.delta[p_baseline.countries.index('MEX'),1] * 0.9
# p_cf.delta[p_baseline.countries.index('MEX'),1] = 0.01

# sol, dyn_sol_cf = dyn_fixed_point_solver(p_cf, sol_init=sol_baseline,Nt=25,
#                                       t_inf=500,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=False,
#                         cobweb_qty='l_R',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         plot_live = False,
#                         safe_convergence=1e-8,
#                         disp_summary=False,
#                         damping = 60,
#                         max_count = 50000,
#                         accel_memory =5, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=1, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         )
# dyn_sol_cf.compute_non_solver_quantities(p_cf)
# dyn_sol_cf.sol_fin.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
# dyn_sol_cf.sol_fin.compute_world_welfare_changes(p_cf,sol_baseline)
