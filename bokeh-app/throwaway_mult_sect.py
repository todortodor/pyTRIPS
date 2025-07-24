#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 08:44:45 2025

@author: slepot
"""


from classes import moments, parameters, var, var_with_entry_costs
from solver_funcs import fixed_point_solver, fixed_point_solver_with_entry_costs
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

baseline_number = 6002
variation = 'baseline'
# pre_trips_number = 4096
path = 'mult_sector_calib/merge_pharma_chem/'
try:
    os.mkdir(path)
except:
    pass

p = parameters()
# p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation}/')
p.load_run(f'calibration_results_matched_economy/{baseline_number}/')

sol, sol_c = fixed_point_solver(p,x0=p.guess,
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
m = moments()
# m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation}/')
m.load_run(f'calibration_results_matched_economy/{baseline_number}/')
m.compute_moments(sol_c,p)
m.compute_moments_deviations()

# p_pre_trips_full = parameters()
# p_pre_trips_full.load_run(f'calibration_results_matched_economy/{pre_trips_number}/')
# delta_pre_trips = p_pre_trips_full.delta.copy()
# p_pre_trips = p.copy()
# p_pre_trips.delta = delta_pre_trips.copy()


# m.plot_moments(list_of_moments = m.list_of_moments)

# plt.scatter(m.SPFLOW_target.ravel(),m.SPFLOW.ravel())
# texts = [plt.text(m.SPFLOW_target.ravel()[i],m.SPFLOW.ravel()[i],idx) 
#          for i,idx in enumerate(m.idx['SPFLOW'])] 
# plt.xscale('log')
# plt.yscale('log')
# plt.plot()
# plt.show()

#%%

baseline_nus = pd.DataFrame(index=p.sectors[1:],
                               data=p.nu[1:])

baseline_thetas = pd.DataFrame(index=p.sectors[1:],
                               data=p.theta[1:])

baseline_zetas = pd.DataFrame(index=p.sectors[1:],
                               data=p.zeta[1:])

baseline_fo = pd.DataFrame(index=p.sectors[1:],
                               data=p.fo[1:])

baseline_fe = pd.DataFrame(index=p.sectors[1:],
                               data=p.fe[1:])

recap_sectors = pd.DataFrame(index=p.sectors[1:])

recap_sectors['nu'] = p.nu[1:]
recap_sectors['theta'] = p.theta[1:]
recap_sectors['zeta'] = p.zeta[1:]
recap_sectors['fe'] = p.fe[1:]
recap_sectors['fo'] = p.fo[1:]

recap_sectors.to_csv(path+'recap_sectors.csv')

baseline_deltas = pd.DataFrame(index=p.countries,
                               columns=p.sectors[1:],
                               data=p.delta[:,1:]) 
baseline_deltas = pd.concat((baseline_deltas,baseline_deltas.describe()))

baseline_deltas.to_csv(path+'baseline_deltas.csv')


baseline_etas = pd.DataFrame(index=p.countries,
                               columns=p.sectors[1:],
                               data=p.eta[:,1:]) 
baseline_etas = pd.concat((baseline_etas,baseline_etas.describe()))

baseline_etas.to_csv(path+'baseline_etas.csv')

baseline_T = pd.DataFrame(index=p.countries,
                               columns=p.sectors,
                               data=p.T) 
baseline_T = pd.concat((baseline_T,baseline_T.describe()))

baseline_T.to_csv(path+'baseline_T.csv')

#%%

p_coop_negishi = parameters()
p_coop_negishi.load_run(f'coop_eq_direct_saves/dyn_{baseline_number}_{variation}_negishi/')

recap_negishi = pd.DataFrame(index=p.countries,
                               columns=p.sectors[1:],
                               data=p_coop_negishi.delta[:,1:]) 


sol, dyn_sol = dyn_fixed_point_solver(p_coop_negishi, sol_c, Nt=25,
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

recap_negishi['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_negishi.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_negishi.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

recap_negishi.to_csv(path+'negishi.csv')

#%%


p_coop_equal = parameters()
p_coop_equal.load_run(f'coop_eq_direct_saves/dyn_{baseline_number}_{variation}_pop_weighted/')


recap_equal = pd.DataFrame(index=p.countries,
                               columns=p.sectors[1:],
                               data=p_coop_equal.delta[:,1:]) 

p_coop_equal.delta[8,1] = 12

sol, dyn_sol = dyn_fixed_point_solver(p_coop_equal, sol_c, Nt=25,
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

recap_equal['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_equal.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_equal.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

recap_equal.to_csv(path+'equal.csv')

#%%

p_nash = parameters()
p_nash.load_run(f'coop_eq_direct_saves/dyn_{baseline_number}_{variation}_nash/')


recap_nash = pd.DataFrame(index=p.countries,
                               columns=p.sectors[1:],
                               data=p_nash.delta[:,1:]) 

sol, dyn_sol = dyn_fixed_point_solver(p_nash, sol_c, Nt=25,
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

recap_nash['welfare'] = dyn_sol.cons_eq_welfare*100 - 100
recap_nash.loc['Negishi','welfare'] = dyn_sol.cons_eq_negishi_welfare_change*100 - 100
recap_nash.loc['Equal','welfare'] = dyn_sol.cons_eq_pop_average_welfare_change*100 - 100

recap_nash.to_csv(path+'nash.csv')

# #%%

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

