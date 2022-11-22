#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:57:06 2022

@author: simonl
"""

from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
import numpy as np

p = parameters(n=7,s=2)
p.load_data('calibration_results_matched_economy/101/')
p.delta[0,1] = 0.1*p.delta[0,1]
# p.d_np = np.ones((p.N,p.N))
# np.fill_diagonal(p.d_np,p.d)
sol, sol_c = fixed_point_solver(p,x0=p.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=True,
                        safe_convergence=0.01,
                        disp_summary=True,
                        damping = 20,
                        max_count = 50000,
                        accel_memory = 10, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )

sol_c = var.var_from_vector(sol.x, p)    
sol_c.scale_P(p)
sol_c.compute_price_indices(p)
sol_c.compute_non_solver_quantities(p) 
list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP',
                    'SRDUS', 'SPFLOWDOM', 'SRGDP', 'JUPCOST',
                    'SINNOVPATUS','TO']
m = moments(list_of_moments)
m.load_data()
m.compute_moments(sol_c,p)
m.compute_moments_deviations()
# m.plot_moments(m.list_of_moments)