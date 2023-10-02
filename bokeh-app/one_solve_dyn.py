#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:42:42 2023

@author: slepot
"""

from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore', np.RankWarning)

p_init = parameters()
# p_init.load_run('calibration_results_matched_economy/1030/')
p_init.load_run('calibration_results_matched_economy/baseline_1030_variations/99.7/')

sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
sol_init.scale_P(p_init)
sol_init.compute_non_solver_quantities(p_init) 

p = p_init.copy()
# p.delta[:,1] = np.array([0.01,0.01,0.01,12,12.0,12.0,12.0,0.01,0.01,12.0,12.0])
p.delta[:,1] = np.array([0.01,0.01,0.01,12.0,12.0,12.0,0.01,0.01,12.0,0.01,12.0])
# p.delta[:,1] = np.array([0.01,0.01,0.01,12.0,12.0,12.0,0.01,0.01,0.1,0.01,12.0])
# p.delta[-3,1] = 12

# p.load_run('calibration_results_matched_economy/baseline_1020_variations/20.0/')
# p.delta[0,1] = 0.05
# p.delta[1,1] = 0.1
# p.delta[2,1] = 0.2
# p.delta[3,1] = 0.3
# p.delta[4,1] = 0.01
# p.delta[:,1] = 12
# p.delta[0,1] = 1e-2


sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=21,
                                      t_inf=500,
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
                        damping_post_acceleration=10
                        )
dyn_sol.compute_non_solver_quantities(p)
print(dyn_sol.cons_eq_pop_average_welfare_change)
# print(dyn_sol.cons_eq_negishi_welfare_change)
# def make_time_evolution_df(dyn_sol):
#     qties = ['w','l_R','l_Ae','l_Ao','price_indices','Z','g','r','profit']
#     df = pd.DataFrame(index = qties, columns = ['Initial jump','Typical time of evolution'])
#     for qty in qties:
#         df.loc[qty,'Initial jump'] = dyn_sol.get_jump(qty)
#         df.loc[qty,'Typical time of evolution'] = dyn_sol.get_typical_time_evolution(qty)
#     return df
# print(make_time_evolution_df(dyn_sol))