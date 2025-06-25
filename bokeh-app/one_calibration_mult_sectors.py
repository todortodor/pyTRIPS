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
baseline_number = '5001'
variation_to_load = '3.0'
# n = 4
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    p.fix_fe_across_sectors = True
    # p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    p.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    # p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    # p.load_data('data/data_12_countries_4_sectors_1992/',keep_already_calib_params=True,nbr_sectors=4)
    start_time = time.perf_counter()

    m = moments()
    # m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    m.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    # m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    # m.load_data('data/data_12_countries_4_sectors_2015/')
    # m.load_data('data/data_12_countries_4_sectors_1992/')
    # m.list_of_moments = ['GPDIFF',
    #  'GROWTH',
    #  'KM',
    #  'OUT',
    #  'RD',
    #  'RP',
    #  'SRGDP',
    #  'SINNOVPATUS',
    #  'TO',
    #  'TOCHEM',
    #  'TOPHARMA',
    #  'SPFLOW',
    #  'SDFLOW',
    #  'UUPCOSTS',
    #  'DOMPATINUS',
    #  'TE']

# m.list_of_moments.append('TEPHARMA')
# m.list_of_moments.append('TECHEM')
# m.list_of_moments.append('TOPHARMA')
# m.list_of_moments.append('TOCHEM')
# m.list_of_moments.append('RDPHARMA')
# m.list_of_moments.append('RDCHEM')
# m.list_of_moments.append('KMPHARMA')
# m.list_of_moments.append('KMCHEM')
m.list_of_moments.append('UUPCOST')
m.list_of_moments.remove('UUPCOSTS')
# m.list_of_moments.remove('GPDIFF')

# m.weights_dict['RDPHARMA'] = 7.5
# m.weights_dict['RDCHEM'] = m.weights_dict['RDPHARMA']


# p.delta[p.delta<0.02] = 0.02
# p.eta[p.eta<1e-4] = 1e-4
# p.eta[:,2:] = p.eta[:,2:]*2
# p.fix_fe_across_sectors = True

m.drop_CHN_IND_BRA_ROW_from_RD = True

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 4
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
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
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
# m.plot_moments(m.list_of_moments)

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

baseline_number = '5001'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 21.0

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')
