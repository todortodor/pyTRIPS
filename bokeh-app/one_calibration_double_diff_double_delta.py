#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:10:17 2024

@author: slepot
"""

from scipy import optimize
import time
from classes import moments, parameters,  var_double_diff_double_delta, history
from solver_funcs import calibration_func_double_diff_double_delta, fixed_point_solver_double_diff_double_delta
from data_funcs import write_calibration_results
import os
import numpy as np

new_run = True
baseline_number = '1310'

if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    # p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # print(p.k)
    # p.mask['k'] = np.array([ True,  True])
    p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/3.0/')
    start_time = time.perf_counter()

    m = moments()
    # m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/3.0/')
# m.load_data('data_smooth_3_years/data_12_countries_1992/')
# p.load_data('data_smooth_3_years/data_12_countries_1992/',keep_already_calib_params=True)
# sol = var_double_diff_double_delta.var_from_vector(p.guess, p, compute=True, context = 'calibration')
# sol.scale_P(p)
# sol.compute_non_solver_quantities(p)
# m.compute_moments(sol, p)
# print(m.SPFLOWDOM[8,8])
# m.compute_moments_deviations()
#%%
# p.delta_dom[p.delta_dom<0.05] = 0.05
# p.delta_int[p.delta_int<0.05] = 0.05
# p.update_delta_eff()
m.drop_CHN_IND_BRA_ROW_from_RD = True
# m.weights_dict['SPFLOWDOM'] = 5

# p.d = 1.1
# p.d = 0.191473
# p.a=0.1


p.nu[:] = 1e-6
p.calib_parameters = ['eta', 'k', 'fe', 'T', 
                      'nu_tilde',
                      'zeta', 'g_0', 'delta_int', 'delta_dom', 'fo', 'theta']
m.list_of_moments=['GPDIFF',
 'GROWTH',
 'KM_DD_DD',
 'OUT',
 'RD',
 'TO_DD_DD',
 'RP',
 'SRGDP',
 'SINNOVPATUS',
 'SPFLOWDOM',
 'UUPCOST',
 'DOMPATINUS',
 'TE']
# p.guess = np.concatenate((p.guess,np.ones(p.N)),axis=0)
# p.calib_parameters.append('a')
# p.calib_parameters.append('d')
# m.list_of_moments.append('PROBINNOVENT')


sol, sol_c = fixed_point_solver_double_diff_double_delta(p,
                        context = 'calibration',x0=p.guess,
                        cobweb_anim=False,tol =1e-14,
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
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
m.compute_moments(sol_c, p)
m.compute_moments_deviations()

# sol_c.compute_non_solver_quantities(p)
# p.guess = sol_c.vector_from_var()

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 6

while cond:
    if iterations < max_iter - 4:
        test_ls = optimize.least_squares(fun = calibration_func_double_diff_double_delta,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time), 
                                bounds = bounds,
                                max_nfev=1e8,
                                xtol=1e-10, 
                                verbose = 2)
    else:
        test_ls = optimize.least_squares(fun = calibration_func_double_diff_double_delta,    
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

sol, sol_c = fixed_point_solver_double_diff_double_delta(p_sol,
                        context = 'calibration',x0=p_sol.guess,
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

#%% writing results as excel and locally

commentary = ''
baseline_number = '1310'
dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 4.0
path = dropbox_path+'baseline_'+baseline_number+'_variations/'

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'
    path = dropbox_path
    
try:
    os.mkdir(path)
except:
    pass

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')
