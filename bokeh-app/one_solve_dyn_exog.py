#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:42:42 2023

@author: slepot
"""

from solver_funcs import fixed_point_solver_exog_lr, dyn_fixed_point_solver_exog_lr, fixed_point_solver\
    , dyn_fixed_point_solver, fixed_point_solver_exog_patent_thresholds, dyn_fixed_point_solver_exog_patent_thresholds,\
    fixed_point_solver_exog_lr_and_patent_thresholds, dyn_fixed_point_solver_exog_lr_and_patent_thresholds
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# warnings.simplefilter('ignore', np.RankWarning)

df = pd.DataFrame()
p_init = parameters()

# p_init.load_run('coop_eq_direct_saves/dyn_2000_14.0_negishi/')
p_init.load_run('calibration_results_matched_economy/2000/')

sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
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
sol_init.compute_non_solver_quantities(p_init)

p = parameters()
p.load_run('coop_eq_direct_saves/2000_baseline_nash/')

sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=25,
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


#%% solve steady state for exog lR

sol, sol_c_exog_lr = fixed_point_solver_exog_lr(p,p_init,x0=p_init.guess,
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
sol_c_exog_lr.scale_P(p)
sol_c_exog_lr.compute_non_solver_quantities(p) 
sol_c_exog_lr.compute_consumption_equivalent_welfare(p, sol_init)


#%% solve dyn for exog lR

sol, dyn_sol_exog_lr = dyn_fixed_point_solver_exog_lr(p,p_init, sol_init, Nt=25,
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
dyn_sol_exog_lr.compute_non_solver_quantities(p)

#%% solve steady state for exog patent thresholds

sol, sol_c_exog_patent_thresholds = fixed_point_solver_exog_patent_thresholds(p,p_init,x0=p_init.guess,
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
sol_c_exog_patent_thresholds.scale_P(p)
sol_c_exog_patent_thresholds.compute_non_solver_quantities(p) 
sol_c_exog_patent_thresholds.compute_consumption_equivalent_welfare(p, sol_init)


#%% solve dyn for exog patent thresholds

sol, dyn_sol_exog_patent = dyn_fixed_point_solver_exog_patent_thresholds(p,p_init, sol_init, Nt=25,
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
dyn_sol_exog_patent.compute_non_solver_quantities(p)

#%% solve steady state for exog patent thresholds and lr

sol, sol_c_exog_lr_and_patent_thresholds = fixed_point_solver_exog_lr_and_patent_thresholds(p,p_init,x0=p_init.guess,
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
sol_c_exog_lr_and_patent_thresholds.scale_P(p)
sol_c_exog_lr_and_patent_thresholds.compute_non_solver_quantities(p) 
sol_c_exog_lr_and_patent_thresholds.compute_consumption_equivalent_welfare(p, sol_init)

#%% solve dyn for exog lr and patenting thresholds

sol, dyn_sol_exog_lr_and_patent = dyn_fixed_point_solver_exog_lr_and_patent_thresholds(p,p_init, sol_init, Nt=25,
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
dyn_sol_exog_lr_and_patent.compute_non_solver_quantities(p)

#%% gather all in one dataframe

df = pd.DataFrame(index=p.countries+['Equal'])

df['full solve'] = dyn_sol.cons_eq_welfare.tolist() + [dyn_sol.cons_eq_pop_average_welfare_change]
df['exog l_R'] = dyn_sol_exog_lr.cons_eq_welfare.tolist() + [dyn_sol_exog_lr.cons_eq_pop_average_welfare_change]
df['exog patent thresh'] = dyn_sol_exog_patent.cons_eq_welfare.tolist() + [dyn_sol_exog_patent.cons_eq_pop_average_welfare_change]
df['exog l_R and patent thresh'] = dyn_sol_exog_lr_and_patent.cons_eq_welfare.tolist() + [dyn_sol_exog_lr_and_patent.cons_eq_pop_average_welfare_change]

df.to_csv('../misc/exog_solvers_nash.csv')