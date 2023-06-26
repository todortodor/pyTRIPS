#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:57:06 2022

@author: simonl
"""

from classes import moments, parameters, var
from solver_funcs import fixed_point_solver

p = parameters()
p.load_run('calibration_results_matched_economy/baseline_803_variations/1.0/')
for c in p.countries:
    p_US = p.make_one_country_parameters(c)
    # p.load_run('calibration_results_matched_economy/405/')
    # m = moments()
    # m.load_data()
    # m.load_run('calibration_results_matched_economy/baseline_312_variations/2.0/')
    # m.compute_moments_deviations()
    # test = compute_rough_jacobian(p, m, 'delta', (0,1),change_by = 0.1)
    # plt.barh(m.get_signature_list(),test)
    # p.fo = p.fe
    # p.delta[0,1] = 0.1*p.delta[0,1]
    # p.d_np = np.ones((p.N,p.N))
    # np.fill_diagonal(p.d_np,p.d)
    # p = p_it_baseline
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 2,
                            max_count = 1000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    # print(c,sol_c.cons[p.countries.index(c)])
    # print(c,sol_c.g)
    
    sol, sol_US = fixed_point_solver(p_US,x0=p_US.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 2,
                            max_count = 1000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_US.scale_P(p_US)
    sol_US.compute_non_solver_quantities(p_US) 
    sol_US.compute_one_country_welfare_change(p_US,sol_c.cons[p.countries.index(c)],sol_c.g)
    print(c,sol_US.cons_eq_welfare)
    # list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP',
    #                     'SRDUS', 'SPFLOWDOM', 'SRGDP', 'JUPCOST',
    #                     'SINNOVPATUS','TO']
    # m = moments(list_of_moments)
    # m.load_data()
    # m.compute_moments(sol_c,p)
    # m.compute_moments_deviations()
    # m.plot_moments(m.list_of_moments)

