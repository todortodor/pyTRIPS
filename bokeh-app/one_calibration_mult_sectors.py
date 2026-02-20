#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:00:03 2022

@author: simonl
"""

from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver
from data_funcs import write_calibration_results
import os
import numpy as np

new_run = True
baseline_number = '2002'
# baseline_number = '2000'
# baseline_number = '1300'
variation_to_load = '4.0'
# baseline_number = '6001'
# variation_to_load = '4.02'
# n = 4
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    # p.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    # p.load_data('data/data_12_countries_4_sectors_1992/',keep_already_calib_params=True,nbr_sectors=4)
    # p.load_data('data/data_12_countries_3_sectors_2015/',keep_already_calib_params=True,nbr_sectors=3)
    start_time = time.perf_counter()

    m = moments()
    # m.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    m.aggregate_moments = True
    
    sol, sol_init = fixed_point_solver(p,x0=p.guess,
                                    # context = 'counterfactual',
                                    context = 'calibration',
                            cobweb_anim=False,tol =1e-10,
                            accelerate=True,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=True,
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
    sol_init.scale_P(p)
    sol_init.compute_non_solver_quantities(p)
    
    m.compute_moments(sol_init, p)

    # m.load_data('data/data_12_countries_3_sectors_2015/')
    # m.load_data('data/data_12_countries_4_sectors_1992/')

# m.list_of_moments = ['GPDIFF',
#  'GROWTH',
#  # 'KM',
#  'KMPATENT',
#  'OUT',
#  'RD',
#  'RP',
#  'SRGDP',
#  'SINNOVPATUS',
#  'TO',
#  # 'TOPATENT',
#  'SPFLOW',
#  'UUPCOSTS',
#  'DOMPATINUS',
#  'TE',
#  'TEPHARMACHEM',
#  'TOPHARMACHEM',
#  'RDPHARMACHEM',
#  'KMPHARMACHEM',
#  # 'AGGAVMARKUP',
#  'AVMARKUPPHARCHEM'
#  ]

# p.sigma[1] = 2.9

# p.calib_parameters = ['eta', 'k', 'fe', 'T', 'zeta', 'g_0', 'delta', 'nu', 'fo', 'theta']

# print('variation',variation_to_load)
# print('KMPHARMA',m.KMPHARMA_target,'KMCHEM',m.KMCHEM_target)
# print('TOPHARMA',m.TOPHARMA_target,'TOCHEM',m.TOCHEM_target)

# m.KMPHARMACHEM_deviation = 1
#
# m.weights_dict['KMPHARMACHEM'] = 5
# m.weights_dict['RDPHARMACHEM'] = 10
# m.weights_dict['RP'] = 10
# m.weights_dict['RD'] = 10
# m.weights_dict['RDPHARMACHEM'] = 2
# m.weights_dict['SPFLOW'] = 1

# p.ub_dict['k'] = 1.15
# p.lb_dict['sigma'] = 2
# p.beta[2] = p.beta[2]*2
# p.beta = p.beta/p.beta.sum()

# p.delta = np.maximum(p.delta,0.01)
# m.list_of_moments.remove('AGGAVMARKUP')
# p.calib_parameters.remove('sigma')
# p.sigma[2] = p.sigma[2]*2

# m.weights_dict['KM'] = 5
# m.weights_dict['KMPATENT'] = 5
# m.weights_dict['AGGAVMARKUP'] = 20

# m.AGGAVMARKUP_target = np.float64(1.0629487478533735)

#%%

m.drop_CHN_IND_BRA_ROW_from_RD = True

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 3
# max_iter = 6

while cond:
    # if iterations < max_iter - 4:
    if iterations < max_iter - 2:
        test_ls = optimize.least_squares(fun = calibration_func,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time), 
                                bounds = bounds,
                                max_nfev=1e8,
                                xtol=1e-10, 
                                verbose = 2)
    else:
        test_ls = optimize.least_squares(fun = calibration_func,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time), 
                                bounds = bounds,
                                max_nfev=1e8,
                                xtol=1e-14, 
                                verbose = 2)
    cond = iterations < max_iter
    iterations += 1
    p.update_parameters(test_ls.x)
        
    cost = test_ls.cost
finish_time = time.perf_counter()
print('minimizing time',finish_time-start_time)

p_sol = p.copy()
p_sol.update_parameters(test_ls.x)

sol, sol_c = fixed_point_solver(p_sol,context = 'calibration',x0=p_sol.guess,
                        cobweb_anim=False,tol =1e-13,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )

sol_c.scale_P(p_sol)
p_sol.guess = sol.x 

sol_c.compute_non_solver_quantities(p_sol) 
p_sol.tau = sol_c.tau
m.compute_moments(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

# print(sol_c.semi_elast_patenting_delta[0,1]/12)

#%% writing results as excel and locally

# run_number = '4013'
# local_path = 'calibration_results_matched_economy/'

# try:
#     os.mkdir(local_path)
# except:
#     pass
# p_sol.write_params(local_path+str(run_number)+'/')
# m.write_moments(local_path+str(run_number)+'/')
# # p_sol.write_params(local_path+run_str+'/')
# # m.write_moments(local_path+run_str+'/')

baseline_number = '2002'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 4.0

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')
