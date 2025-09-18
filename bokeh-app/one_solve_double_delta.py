#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:57:06 2022

@author: simonl
"""


from classes import moments, parameters, var, var_double_diff_double_delta
from solver_funcs import fixed_point_solver, fixed_point_solver_double_diff_double_delta
import numpy as np

p = parameters()
p.load_run('calibration_results_matched_economy/1311/')

sol, sol_init_dd = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 10000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )
sol_init_dd.scale_P(p)
sol_init_dd.compute_non_solver_quantities(p)


p_cf = p.copy()
p_cf.delta_dom[0,1] = p_cf.delta_dom[0,1]/10
p_cf.delta_int[0,1] = p_cf.delta_int[0,1]/10
p_cf.update_delta_eff()

sol, sol_cf = fixed_point_solver_double_diff_double_delta(p_cf,x0=p_cf.guess,
                                # context = 'counterfactual',
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 10000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )
sol_cf.scale_P(p_cf)
sol_cf.compute_non_solver_quantities(p_cf)
sol_cf.compute_consumption_equivalent_welfare(p_cf, sol_init_dd)


print(sol_cf.cons_eq_welfare)