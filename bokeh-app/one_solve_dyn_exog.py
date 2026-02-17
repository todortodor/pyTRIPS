#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:42:42 2023

@author: slepot
"""

from solver_funcs import fixed_point_solver_exog_lr, dyn_fixed_point_solver_exog_lr
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# warnings.simplefilter('ignore', np.RankWarning)

df = pd.DataFrame()
p_init = parameters()

p_init.load_run('coop_eq_direct_saves/dyn_2000_14.0_negishi/')

sol, sol_init = fixed_point_solver_exog_lr(p_init,p_init,x0=p_init.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
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
sol_init.compute_export_price_index(p_init)
# p_init.guess = sol_init.vector_from_var()


#%%

p = parameters()
p.load_run('coop_eq_direct_saves/2000_baseline_pop_weighted/')

sol, sol_c = fixed_point_solver_exog_lr(p,p_init,x0=p_init.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-13,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
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
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p) 
sol_c.compute_consumption_equivalent_welfare(p, sol_init)
p.guess = sol_c.vector_from_var()

sol, dyn_sol = dyn_fixed_point_solver_exog_lr(p, sol_init, Nt=25,
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
                        damping = 60,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p)