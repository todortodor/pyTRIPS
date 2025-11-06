#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 08:44:45 2025

@author: slepot
"""


from classes import moments, parameters, var, var_double_diff_double_delta
from solver_funcs import fixed_point_solver, fixed_point_solver_double_diff_double_delta, dyn_fixed_point_solver_double_diff_double_delta
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

baseline_number = 1312
# variation = '2.07'
variation = 'baseline'
# pre_trips_number = 4096
path = f'double_delta/1312_{variation}/'
try:
    os.mkdir(path)
except:
    pass

p = parameters()
# p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation}/')
p.load_run(f'calibration_results_matched_economy/{baseline_number}/')

sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=True,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 100,
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
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
# plt.show()
# m = moments()
# m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation}/')
# # m.load_run(f'calibration_results_matched_economy/{baseline_number}/')
# m.compute_moments(sol_c,p)
# m.compute_moments_deviations()


#%%

p_coop_negishi = parameters()
p_coop_negishi.load_run(f'coop_eq_direct_saves/dyn_double_diff_{baseline_number}_{variation}_negishi/')

recap_negishi = pd.DataFrame(index=p.countries) 
recap_negishi['delta_dom'] = p_coop_negishi.delta_dom[:,1]
recap_negishi['delta_int'] = p_coop_negishi.delta_int[:,1]


sol, dyn_sol = fixed_point_solver_double_diff_double_delta(p_coop_negishi,x0=p_coop_negishi.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=True,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 100,
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
dyn_sol.compute_non_solver_quantities(p_coop_negishi)
dyn_sol.compute_consumption_equivalent_welfare(p_coop_negishi,sol_c)
dyn_sol.compute_world_welfare_changes(p_coop_negishi,sol_c)

sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p_coop_negishi, sol_c, 
                                                               Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-12,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-4,
                        disp_summary=True,
                        damping = 50,
                        max_count = 10000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p)

recap_negishi['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_negishi.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_negishi.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

recap_negishi.to_csv(path+'double_diff_negishi.csv')

#%%


p_coop_equal = parameters()
p_coop_equal.load_run(f'coop_eq_direct_saves/dyn_double_diff_{baseline_number}_{variation}_pop_weighted/')


p_coop_equal.delta_dom[8,1] = 12.0
p_coop_equal.delta_int[8,1] = 12.0
p_coop_equal.update_delta_eff()

recap_equal = pd.DataFrame(index=p.countries) 
recap_equal['delta_dom'] = p_coop_equal.delta_dom[:,1]
recap_equal['delta_int'] = p_coop_equal.delta_int[:,1]


sol, dyn_sol = fixed_point_solver_double_diff_double_delta(p_coop_equal,x0=p_coop_equal.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=True,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 100,
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
dyn_sol.compute_non_solver_quantities(p_coop_equal)
dyn_sol.compute_consumption_equivalent_welfare(p_coop_equal,sol_c)
dyn_sol.compute_world_welfare_changes(p_coop_equal,sol_c)

sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p_coop_equal, sol_c, 
                                                               Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-12,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-4,
                        disp_summary=True,
                        damping = 50,
                        max_count = 10000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p)

recap_equal['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_equal.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_equal.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

# recap_equal.to_csv(path+'double_diff_equal.csv')

#%%

p_nash = parameters()
p_nash.load_run(f'coop_eq_direct_saves/dyn_{baseline_number}_{variation}_nash/')


recap_nash = pd.DataFrame(index=p.countries) 
recap_nash['delta_dom'] = p_nash.delta_dom[:,1]
recap_nash['delta_int'] = p_nash.delta_int[:,1]


sol, dyn_sol = fixed_point_solver_double_diff_double_delta(p_nash,x0=p_nash.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=True,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 100,
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
dyn_sol.compute_non_solver_quantities(p_nash)
dyn_sol.compute_consumption_equivalent_welfare(p_nash,sol_c)
dyn_sol.compute_world_welfare_changes(p_nash,sol_c)

sol, dyn_sol = dyn_fixed_point_solver_double_diff_double_delta(p_nash, sol_c, 
                                                               Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-12,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-4,
                        disp_summary=True,
                        damping = 50,
                        max_count = 10000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p)

recap_nash['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_nash.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_nash.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

recap_nash.to_csv(path+'double_diff_nash.csv')

#%%

# recap_pre_trips = pd.DataFrame(index=p.countries,
#                                columns=p.sectors[1:],
#                                data=p_pre_trips.delta[:,1:]) 

# for sector in recap_pre_trips.columns:
#     recap_pre_trips[sector+'_2015'] = baseline_deltas.loc[p.countries][sector]
#     recap_pre_trips[sector+'_change_percent'] = (recap_pre_trips[sector] - recap_pre_trips[sector+'_2015'])*100 / recap_pre_trips[sector+'_2015']

# sol, dyn_sol = dyn_fixed_point_solver(p_pre_trips, sol_c, Nt=25,
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
# dyn_sol.compute_non_solver_quantities(p)

# recap_pre_trips['welfare'] = dyn_sol.cons_eq_welfare*100 - 100

# recap_pre_trips.reindex(sorted(recap_pre_trips.columns), axis=1).to_csv(path+'pre_trips.csv')

