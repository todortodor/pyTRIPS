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
baseline_number = '1300'
# n = 4
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # p.load_run('calibration_results_matched_economy/baseline_1020_all_targets_variations_20/RD_CHN/')
    # p.load_run('calibration_results_matched_economy/baseline_1220_variations/1.0/')
    # p.load_data('data/data_12_countries_2015/',keep_already_calib_params=True)
    start_time = time.perf_counter()

    m = moments()
    m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/3.0/')
    # m.load_run('calibration_results_matched_economy/baseline_1220_variations/1.0/')
    # m.load_data('data/data_12_countries_2015/')


# p.kappa = 0.4
# m.TO_target = np.float64(0.055589733)
# m.list_of_moments.remove('SRGDP')
# m.list_of_moments.remove('RP')
# m.list_of_moments.append('SGDP')
# m.list_of_moments.append('RGDPPC')
# # p.calib_parameters.remove('zeta')
# # p.zeta[:] = 0
# m.drop_CHN_IND_BRA_ROW_from_RD = True
# p.sigma = np.array([2.7, 2.9])
# m.weights_dict['RD'] = 10
# p.calib_parameters = ['eta', 'k', 'fe', 'T', 'zeta', 'g_0', 'delta', 'nu', 'fo', 'theta']
# m.list_of_moments = ['GPDIFF',
#  'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRGDP', 'SINNOVPATUS', 'TO', 'SPFLOW', 'UUPCOST', 'DOMPATINUS',
#  'TE']
# m.load_data('data/data_11_countries_1992/')
# m.weights_dict['TO'] = 5
# m.weights_dict['TE'] = 10
# m.weights_dict['GROWTH'] = 10
# m.weights_dict['TE'] = 5
# m.weights_dict['GROWTH'] = 5
# m.weights_dict = {'GPDIFF': 1,
#  'GROWTH': 5,
#  'KM': 1,
#  'KM_GDP': 5,
#  'OUT': 5,
#  'RD': 10,
#  'RD_US': 3,
#  'RD_RUS': 3,
#  'RP': 1,
#  'SPFLOW': 1,
#  'SPFLOW_US': 1,
#  'SPFLOW_RUS': 1,
#  'SPFLOWDOM': 1,
#  'SPFLOWDOM_US': 1,
#  'SPFLOWDOM_RUS': 1,
#  'SRDUS': 1,
#  'SRGDP': 1,
#  'SRGDP_US': 1,
#  'SRGDP_RUS': 1,
#  'STFLOW': 1,
#  'SDOMTFLOW': 1,
#  'JUPCOST': 1,
#  'UUPCOST': 1,
#  'PCOSTNOAGG': 1,
#  'PCOSTINTERNOAGG': 1,
#  'PCOST': 1,
#  'PCOSTINTER': 1,
#  'JUPCOSTRD': 1,
#  'TP': 1,
#  'inter_TP': 3,
#  'Z': 1,
#  'STFLOWSDOM': 1,
#  'SINNOVPATEU': 1,
#  'SINNOVPATUS': 1,
#  'NUR': 1,
#  'TO': 5,
#  'TE': 5,
#  'DOMPATRATUSEU': 2,
#  'DOMPATUS': 1,
#  'DOMPATEU': 1,
#  'DOMPATINUS': 1,
#  'DOMPATINEU': 1,
#  'SPATORIG': 2,
#  'SPATDEST': 2,
#  'TWSPFLOW': 1,
#  'TWSPFLOWDOM': 1,
#  'ERDUS': 3}
# p.kappa = 0.1
# p.sigma[1] = 3.375

# p.tariff[:] = 1.0
# np.einsum('iis->is',p.tariff)[:] = 0

# m.RP_target[3] = m.RP_target[3]*1.2
# m.SRGDP_target[1] = m.SRGDP_target[1]*1.2
# m.SRGDP_target = m.SRGDP_target/m.SRGDP_target.sum()

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 6

while cond:
    if iterations < max_iter - 4:
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
p_sol.guess = sol.x 
sol_c.scale_P(p_sol)

sol_c.compute_non_solver_quantities(p_sol) 
p_sol.tau = sol_c.tau
m.compute_moments(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

# print(sol_c.semi_elast_patenting_delta[0,1]/12)

#%% writing results as excel and locally

commentary = ''
baseline_number = '1300'
dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 13.0
# run_number = f'{n}.1'
# run_str = '4.'
path = dropbox_path+'baseline_'+baseline_number+'_variations/'

# p_sol.nu[1] = p_sol.nu[1]*2
# run_number = 2.0

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'
    path = dropbox_path
    
try:
    os.mkdir(path)
except:
    pass

# write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
# m.plot_moments(m.list_of_moments, save_plot = path+str(run_number))
# write_calibration_results(path+run_str,p_sol,m,sol_c,commentary = commentary)
# m.plot_moments(m.list_of_moments, save_plot = path+run_str)

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')
# p_sol.write_params(local_path+run_str+'/')
# m.write_moments(local_path+run_str+'/')

