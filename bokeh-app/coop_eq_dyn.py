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
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
    {'baseline':'1210','variation': 'baseline'},
    {'baseline':'1210','variation': '2.0'},
    {'baseline':'1210','variation': '10.2'},
    {'baseline':'1210','variation': '10.3'},
    {'baseline':'1210','variation': '10.4'},
    {'baseline':'1210','variation': '10.5'},
    {'baseline':'1210','variation': '99.0'},
    {'baseline':'1210','variation': '99.1'},
    {'baseline':'1210','variation': '99.2'},
    {'baseline':'1210','variation': '99.3'},
    {'baseline':'1210','variation': '99.4'},
    {'baseline':'1210','variation': '99.5'},
    {'baseline':'1210','variation': '99.6'},
    {'baseline':'1210','variation': '99.7'},
    {'baseline':'1210','variation': '99.8'},
    {'baseline':'1210','variation': '99.9'},
    {'baseline':'1210','variation': '99.10'},
    {'baseline':'1210','variation': '99.11'},
    {'baseline':'1210','variation': '99.12'},
    {'baseline':'1210','variation': '99.13'},
    {'baseline':'1210','variation': '99.14'},
    {'baseline':'1210','variation': '99.15'},
    ]
# baseline_dics = [
#   {'baseline': '1030', 'variation': '2.0'},
#   ]

lb_delta = 0.01
ub_delta = 12
# ub_delta = 1
if __name__ == '__main__':
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
        
        for aggregation_method in ['negishi','pop_weighted']:
        # for aggregation_method in ['pop_weighted']:
            print(aggregation_method)
            static_eq_deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0).drop_duplicates(
                ['baseline','variation','aggregation_method'],keep='last')
            static_eq_deltas = static_eq_deltas.loc[
                (static_eq_deltas.baseline.astype('str') == baseline_dic['baseline'])
                & (static_eq_deltas.variation.astype('str') == baseline_dic['variation'])
                & (static_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()
            
            static_eq_deltas[static_eq_deltas>0.9] = ub_delta
            # static_eq_deltas[static_eq_deltas>0.9] = 12
            print(static_eq_deltas)
            
            p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method,
                             lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
                             solver_options=None,tol=1e-6,
                               static_eq_deltas = static_eq_deltas,
                              # static_eq_deltas = np.array([0.01,0.01,0.01,12.0,12.0,12.0,0.01,0.01,12.0,0.01,12.0]),
                              # custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
                              #     accelerate=False,
                              #     accelerate_when_stable=False,
                              #     cobweb_qty='l_R',
                              #     plot_convergence=False,
                              #     plot_cobweb=False,
                              #     plot_live = False,
                              #     safe_convergence=1e-8,
                              #     disp_summary=False,
                              #     damping = 500,
                              #     max_count = 100000,
                              #     accel_memory =5, 
                              #     accel_type1=True, 
                              #     accel_regularization=1e-10,
                              #     accel_relaxation=1, 
                              #     accel_safeguard_factor=1, 
                              #     accel_max_weight_norm=1e6,
                              #     damping_post_acceleration=10),
                               custom_dyn_sol_options = None,
                             custom_weights=None,max_workers=22)
            
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

#%%

# from scipy import optimize
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# from classes import moments, parameters, var
# from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq
# from tqdm import tqdm
# import matplotlib.pylab as pylab

# baseline_dics = [
#   {'baseline': '1030', 'variation': 'baseline'},
#   {'baseline': '1030', 'variation': '20.0'},
#   # {'baseline': '1030', 'variation': '20.1'},
#   # {'baseline': '1030', 'variation': '20.2'},
#   # {'baseline': '1030', 'variation': '20.3'},
#   # {'baseline': '1030', 'variation': '20.4'},
#   # {'baseline': '1030', 'variation': '20.5'},
#   # {'baseline': '1030', 'variation': '20.6'},
#   # {'baseline': '1030', 'variation': '20.7'},
#   # {'baseline': '1030', 'variation': '20.8'},
#   # {'baseline': '1030', 'variation': '20.9'},
#   # {'baseline': '1030', 'variation': '20.10'},
#   # {'baseline': '1030', 'variation': '20.11'},
#   # {'baseline': '1030', 'variation': '20.12'},
#   # {'baseline': '1030', 'variation': '20.13'},
#   # {'baseline': '1030', 'variation': '20.14'},
#   # {'baseline': '1030', 'variation': '20.15'},
#   # {'baseline': '1030', 'variation': '20.16'},
#   # {'baseline': '1030', 'variation': '20.17'},
#   # {'baseline': '1030', 'variation': '20.18'},
#   # {'baseline': '1030', 'variation': '20.19'},
#   # {'baseline': '1030', 'variation': '20.20'},
#   # {'baseline': '1030', 'variation': '20.21'},
#   # {'baseline': '1030', 'variation': '20.22'},
#   # {'baseline': '1030', 'variation': '20.23'},
#   # {'baseline': '1030', 'variation': '20.24'}
#   ]

# lb_delta = 0.01
# ub_delta = 12
# # ub_delta = 1
# if __name__ == '__main__':
#     for baseline_dic in baseline_dics:    
#         if baseline_dic['variation'] == 'baseline':
#             baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
#         else:
#             baseline_path = \
#                 f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        
#         assert os.path.exists(baseline_path), 'run doesnt exist'
        
#         print(baseline_path)
#         p_baseline = parameters()
#         p_baseline.load_run(baseline_path)
        
#         solver_options = dict(cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=True,
#                                 cobweb_qty='phi',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 safe_convergence=0.001,
#                                 disp_summary=False,
#                                 damping = 10,
#                                 max_count = 3e3,
#                                 accel_memory = 50, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=0.5, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=5)
        
#         sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
#                                         context = 'counterfactual',
#                                 **solver_options
#                                 )
#         sol_baseline.scale_P(p_baseline)
#         sol_baseline.compute_non_solver_quantities(p_baseline)
        
#         for aggregation_method in ['negishi','pop_weighted']:
#         # for aggregation_method in ['pop_weighted']:
#             print(aggregation_method)
#             dyn_eq_deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(
#                 ['baseline','variation','aggregation_method'],keep='last')
#             dyn_eq_deltas = dyn_eq_deltas.loc[
#                 (dyn_eq_deltas.baseline.astype('str') == baseline_dic['baseline'])
#                 & (dyn_eq_deltas.variation.astype('str') == baseline_dic['variation'])
#                 & (dyn_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()
            
#             print(dyn_eq_deltas)
            
#             p = p_baseline.copy()
#             p.delta[...,1] = dyn_eq_deltas
            
#             sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
#                                                           Nt=23,
#                                                   t_inf=500,
#                                     cobweb_anim=False,tol =1e-14,
#                                     accelerate=False,
#                                     accelerate_when_stable=False,
#                                     cobweb_qty='l_R',
#                                     plot_convergence=False,
#                                     plot_cobweb=False,
#                                     plot_live = False,
#                                     safe_convergence=1e-8,
#                                     disp_summary=False,
#                                     damping = 60,
#                                     max_count = 50000,
#                                     accel_memory =5, 
#                                     accel_type1=True, 
#                                     accel_regularization=1e-10,
#                                     accel_relaxation=1, 
#                                     accel_safeguard_factor=1, 
#                                     accel_max_weight_norm=1e6,
#                                     damping_post_acceleration=10
#                                     )
    
#             dyn_sol.compute_non_solver_quantities(p)
            
#             if aggregation_method == 'negishi':
#                 solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
#             if aggregation_method == 'pop_weighted':
#                 solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
            
#             corner_corrected_deltas = p.delta[...,1].copy()
#             for i,c in enumerate(p_baseline.countries):
#                 if c=='CHN':
#                     p_corner = p.copy()
#                     p_corner.delta[i,1] = ub_delta
                    
#                     sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
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
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
#                                                                   sol_fin=sol_corner,
#                                                                   Nt=23,
#                                                           t_inf=500,
#                                             cobweb_anim=False,tol =1e-14,
#                                             accelerate=False,
#                                             accelerate_when_stable=False,
#                                             cobweb_qty='l_R',
#                                             plot_convergence=False,
#                                             plot_cobweb=False,
#                                             plot_live = False,
#                                             safe_convergence=1e-8,
#                                             disp_summary=False,
#                                             damping = 60,
#                                             max_count = 50000,
#                                             accel_memory =5, 
#                                             accel_type1=True, 
#                                             accel_regularization=1e-10,
#                                             accel_relaxation=1, 
#                                             accel_safeguard_factor=1, 
#                                             accel_max_weight_norm=1e6,
#                                             damping_post_acceleration=10
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                        
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    
#                     if corner_welfare > solution_welfare:
#                         print('upper corner was better for ',c)
#                         corner_corrected_deltas[i] = ub_delta
#                     print(c,solution_welfare,corner_welfare,'upper corner')
            
#             p.delta[...,1] = corner_corrected_deltas
            
#             sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
#                                                           Nt=23,
#                                                   t_inf=500,
#                                     cobweb_anim=False,tol =1e-14,
#                                     accelerate=False,
#                                     accelerate_when_stable=False,
#                                     cobweb_qty='l_R',
#                                     plot_convergence=False,
#                                     plot_cobweb=False,
#                                     plot_live = False,
#                                     safe_convergence=1e-8,
#                                     disp_summary=False,
#                                     damping = 60,
#                                     max_count = 50000,
#                                     accel_memory =5, 
#                                     accel_type1=True, 
#                                     accel_regularization=1e-10,
#                                     accel_relaxation=1, 
#                                     accel_safeguard_factor=1, 
#                                     accel_max_weight_norm=1e6,
#                                     damping_post_acceleration=10
#                                     )
    
#             dyn_sol.compute_non_solver_quantities(p)
            
#             if aggregation_method == 'negishi':
#                 solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
#             if aggregation_method == 'pop_weighted':
#                 solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
            
#             # corner_corrected_deltas = p.delta[...,1].copy()
#             for i,c in enumerate(p_baseline.countries):
#                 # if p.delta[i,1] < 2*lb_delta or c=='MEX':
#                 if c=='CHN':
#                     p_corner = p.copy()
#                     p_corner.delta[i,1] = lb_delta
                    
#                     sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
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
                    
#                     sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
#                                                                   sol_fin=sol_corner,
#                                                                   Nt=23,
#                                                           t_inf=500,
#                                             cobweb_anim=False,tol =1e-14,
#                                             accelerate=False,
#                                             accelerate_when_stable=False,
#                                             cobweb_qty='l_R',
#                                             plot_convergence=False,
#                                             plot_cobweb=False,
#                                             plot_live = False,
#                                             safe_convergence=1e-8,
#                                             disp_summary=False,
#                                             damping = 60,
#                                             max_count = 50000,
#                                             accel_memory =5, 
#                                             accel_type1=True, 
#                                             accel_regularization=1e-10,
#                                             accel_relaxation=1, 
#                                             accel_safeguard_factor=1, 
#                                             accel_max_weight_norm=1e6,
#                                             damping_post_acceleration=10
#                                             )
            
#                     dyn_sol_corner.compute_non_solver_quantities(p)
                    
#                     if aggregation_method == 'negishi':
#                         corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
#                     if aggregation_method == 'pop_weighted':
#                         corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
#                         # if aggregation_method == 'custom_weights':
#                         #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
                    
#                     if corner_welfare > solution_welfare:
#                         print('lower corner was better for ',c)
#                         corner_corrected_deltas[i] = lb_delta
                        
#                     print(c,solution_welfare,corner_welfare,'lower corner')
                    
#             p.delta[...,1] = corner_corrected_deltas
            
#             sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
#                                                           Nt=23,
#                                                   t_inf=500,
#                                     cobweb_anim=False,tol =1e-14,
#                                     accelerate=False,
#                                     accelerate_when_stable=False,
#                                     cobweb_qty='l_R',
#                                     plot_convergence=False,
#                                     plot_cobweb=False,
#                                     plot_live = False,
#                                     safe_convergence=1e-8,
#                                     disp_summary=False,
#                                     damping = 60,
#                                     max_count = 50000,
#                                     accel_memory =5, 
#                                     accel_type1=True, 
#                                     accel_regularization=1e-10,
#                                     accel_relaxation=1, 
#                                     accel_safeguard_factor=1, 
#                                     accel_max_weight_norm=1e6,
#                                     damping_post_acceleration=10
#                                     )
    
#             dyn_sol.compute_non_solver_quantities(p)
            
#             sol_opti = dyn_sol
#             p_opti = p
            
#             write = False
#             if write:
#                 if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
#                     deltas_df = pd.DataFrame(columns = ['baseline',
#                                                     'variation',
#                                                     'aggregation_method'] + p_baseline.countries)
#                     deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
#                 deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
#                 run = pd.DataFrame(data = [baseline_dic['baseline'],
#                                 baseline_dic['variation'],
#                                 aggregation_method]+p_opti.delta[...,1].tolist(), 
#                                 # index = deltas_df.columns).T
#                                 index = ['baseline',
#                                           'variation',
#                                           'aggregation_method'] + p_baseline.countries).T
#                 deltas_df = pd.concat([deltas_df, run],ignore_index=True)
#                 deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
                
#                 if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
#                     cons_eq_welfares = pd.DataFrame(columns = ['baseline',
#                                                     'variation',
#                                                     'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
#                     cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
#                 cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
#                 run = pd.DataFrame(data = [baseline_dic['baseline'],
#                                 baseline_dic['variation'],
#                                 aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
#                                                                     sol_opti.cons_eq_negishi_welfare_change], 
#                                 index = ['baseline',
#                                           'variation',
#                                           'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
#                 cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
#                 cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
                
#%%

