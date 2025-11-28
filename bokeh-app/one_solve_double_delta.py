#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:57:06 2022

@author: simonl
"""


# from classes import moments, parameters, var, var_double_diff_double_delta
# from solver_funcs import fixed_point_solver, fixed_point_solver_double_diff_double_delta
# import numpy as np

# p = parameters()
# p.load_run('calibration_results_matched_economy/1311/')

# sol, sol_init_dd = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
#                                 # context = 'counterfactual',
#                                 context = 'calibration',
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='l_R',
#                         plot_convergence=True,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=True,
#                         damping = 10,
#                         max_count = 10000,
#                         accel_memory =50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=2
#                         # damping=10
#                           # apply_bound_psi_star=True
#                         )
# sol_init_dd.scale_P(p)
# sol_init_dd.compute_non_solver_quantities(p)


# p_cf = p.copy()
# p_cf.delta_dom[0,1] = p_cf.delta_dom[0,1]/10
# p_cf.delta_int[0,1] = p_cf.delta_int[0,1]/10
# p_cf.update_delta_eff()

# sol, sol_cf = fixed_point_solver_double_diff_double_delta(p_cf,x0=p_cf.guess,
#                                 # context = 'counterfactual',
#                                 context = 'counterfactual',
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='l_R',
#                         plot_convergence=True,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=True,
#                         damping = 10,
#                         max_count = 10000,
#                         accel_memory =50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=2
#                         # damping=10
#                           # apply_bound_psi_star=True
#                         )
# sol_cf.scale_P(p_cf)
# sol_cf.compute_non_solver_quantities(p_cf)
# sol_cf.compute_consumption_equivalent_welfare(p_cf, sol_init_dd)


# print(sol_cf.cons_eq_welfare)


#%%

from solver_funcs import fixed_point_solver_double_diff_double_delta, dyn_fixed_point_solver_double_diff_double_delta
from classes import moments, parameters, var_double_diff_double_delta, dynamic_var_double_diff_double_delta

p_init = parameters()

# p_init.load_run('coop_eq_direct_saves/4003_baseline_nash/')
# p_init.load_run('calibration_results_matched_economy/baseline_1312_variations/1.07/')
p_init.load_run('calibration_results_matched_economy/1312/')
# p_init.nu[1] = 0.05

sol, sol_init = fixed_point_solver_double_diff_double_delta(p_init,x0=p_init.guess,
                                # context = 'calibration',
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
# p_init.tau = sol_init.tau
p_init.guess = sol_init.vector_from_var()

p = p_init.copy()
# p.load_run('counterfactual_results/double_delta/baseline_1312_1.07/both/USA/5/')
# p.load_run('calibration_results_matched_economy/baseline_1312_variations/1.07/')

#%%
# p.delta_dom[:,1] = 12.0
# p.delta_dom[1,1] = 0.01
# p.delta_int[5,1] = 12.0
# p.update_delta_eff()
sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-13,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
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
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p) 
sol_c.compute_consumption_equivalent_welfare(p, sol_init)
# p.guess = sol_c.vector_from_var()
#%%
sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p_init, sol_init, Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-4,
                        disp_summary=True,
                        damping = 60,
                        max_count = 5000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p_init)