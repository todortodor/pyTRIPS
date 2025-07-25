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
baseline_number = '6001'
variation_to_load = '1.03'
# n = 4
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    # p.fix_fe_across_sectors = True
    # p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # p.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    # p.load_data('data/data_12_countries_4_sectors_1992/',keep_already_calib_params=True,nbr_sectors=4)
    # p.load_data('data/data_12_countries_3_sectors_2015/',keep_already_calib_params=True,nbr_sectors=3)
    start_time = time.perf_counter()

    m = moments()
    # m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # m.load_run(f'calibration_results_matched_economy/{baseline_number}/')
    m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_to_load}/')
    m.aggregate_moments = True
    # m.load_data('data/data_12_countries_3_sectors_2015/')
    # m.load_data('data/data_12_countries_4_sectors_1992/')
    


# p.calib_parameters = ['eta', 'k', 'fe', 'T', 'zeta', 'g_0', 'delta', 'nu', 'fo', 'theta']

# print('variation',variation_to_load)
# print('KMPHARMA',m.KMPHARMA_target,'KMCHEM',m.KMCHEM_target)
# print('TOPHARMA',m.TOPHARMA_target,'TOCHEM',m.TOCHEM_target)

# m.KMPHARMACHEM_deviation = 1

# m.weights_dict['KMPHARMACHEM'] = 5
# m.weights_dict['RDPHARMACHEM'] = 10
# m.weights_dict['RP'] = 10
# m.weights_dict['RD'] = 10
# m.weights_dict['RDPHARMACHEM'] = 20

# p.ub_dict['k'] = 1.15
# p.lb_dict['sigma'] = 2
# p.beta[2] = p.beta[2]*2
# p.beta = p.beta/p.beta.sum()

# p.delta = np.maximum(p.delta,0.05)
# for mm in m.weights_dict:
#     m.weights_dict[mm] = np.minimum(m.weights_dict[mm],2.0)
# p.ub_dict['nu'] = 50
# p.nu[1] = 0.1
# m.list_of_moments.append('TEPHARMA')
# m.list_of_moments.append('TECHEM')
# m.list_of_moments.append('TOPHARMA')
# m.list_of_moments.append('TOCHEM')
# m.list_of_moments.append('RDPHARMACHEM')
# m.list_of_moments.append('RDPHARMA')
# m.list_of_moments.append('RDCHEM')
# m.list_of_moments.append('KMPHARMACHEM')
# m.list_of_moments.append('KMPHARMA')
# m.list_of_moments.append('KMCHEM')
# # m.list_of_moments.append('UUPCOST')
# # m.list_of_moments.remove('UUPCOSTS')
# m.list_of_moments.remove('SDFLOW')
# m.list_of_moments.remove('KMPHARMACHEM')

#%%
# m.TOPHARMA_target = np.float64(0.24055165)
# m.KMPHARMA_target = np.float64(0.156020318914251)
# m.TOCHEM_target = np.float64(0.05290285)
# m.KMCHEM_target = np.float64(0.108714651159773)
# # m.KMCHEM_target = 0.156020318914251
# m.KMPHARMACHEM_target = (m.KMPHARMA_target+m.KMCHEM_target)/2
# m.TOPHARMACHEM_target = (m.TOPHARMA_target+m.TOCHEM_target)/2
# m.TOPHARMACHEM_target = np.float64(0.162464114570443)
# m.KMPHARMACHEM_target = np.float64(0.183202572)
# average 1995-2007
# m.KMPHARMACHEM_target = 0.084591245
# 2007 value
# m.weights_dict['KM'] = 5.0
m.weights_dict['RD'] = 20.0
# m.weights_dict['RDPHARMA'] = 1.0
# m.weights_dict['RDCHEM'] = m.weights_dict['RDPHARMA']
# m.weights_dict['KMPHARMA'] = 4.0
# m.weights_dict['KMCHEM'] = m.weights_dict['KMPHARMA']

# p.sigma[2] = 2.5
# p.sigma[3] = 2.5
# p.calib_parameters.append('sigma')
# p.calib_parameters.remove('sigma')
# p.delta[p.delta<0.02] = 0.02
# p.eta[p.eta<1e-4] = 1e-4
# p.eta[:,2:] = p.eta[:,2:]*2
# p.fix_fe_across_sectors = True
# p.calib_parameters.remove('sigma')
# p.sigma[:] = 2.9

# p.sigma[1] = p.sigma[1]*0.8
# p.sigma[2] = p.sigma[2]*1.2
# p.calib_parameters.remove('sigma')

# p.nu[2] = 10
# p.calib_parameters.remove('nu')
# m.list_of_moments.remove('TOPHARMACHEM')
# m.list_of_moments.remove('KMPHARMACHEM')

#%%

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

baseline_number = '6001'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 1.031

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')
