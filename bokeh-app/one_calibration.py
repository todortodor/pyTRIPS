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
baseline_number = '618'
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # p.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/2.0/')
    # p_back_up = p.copy()
    # p.load_data('data/data_7_countries_2005/',keep_already_calib_params=True)
    start_time = time.perf_counter()

    m = moments()
    # m.load_data()
    m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/10.1/')
    # m_back_up = m.copy()
    # m.load_data('data/data_7_countries_2005/')

p.calib_parameters.remove('nu')
p.nu[1] = 0.1
m.list_of_moments.remove('TO') 

# sol = var.var_from_vector(p.guess,p,context='calibration')
# sol.scale_P(p)
# sol.compute_non_solver_quantities(p)
# m.compute_moments(sol,p)
# m.inter_TP_target = m.inter_TP
# m.list_of_moments.append('inter_TP')

# p.calib_parameters = ['eta','T','delta','fe','fo']

# p.update_khi_and_r_hjort(0.16)

# m.weights_dict['RP'] = 4
# m.weights_dict['SRGDP'] = 3
# m.weights_dict['GROWTH'] = 4
# m.weights_dict['TE'] = 5

# m.TO_target = np.float64(0.015629)

# p.sigma = np.array([2.7,2.7])

# if 'theta' in p.calib_parameters:
#     p.update_sigma_with_SRDUS_target(m)

# if 'fo' not in p.calib_parameters:
#     p.calib_parameters.append('fo')
# if 'd' in p.calib_parameters:
#     p.calib_parameters.remove('d')
# # if 'r_hjort' not in p.calib_parameters:
# #     p.calib_parameters.append('r_hjort')
# # if 'khi' not in p.calib_parameters:
# #     p.calib_parameters.append('khi')
# if 'DOMPATEU' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATEU')
# if 'DOMPATUS' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATUS')
# if 'DOMPATINEU' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATINEU')
# if 'DOMPATINUS' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATINUS')
# if 'SINNOVPATEU' not in m.list_of_moments:
#     m.list_of_moments.append('SINNOVPATEU')
# if 'SRDUS' not in m.list_of_moments:
#     m.list_of_moments.append('SRDUS')
# if 'SRDUS' in m.list_of_moments:
#     m.list_of_moments.remove('SRDUS')
# if 'SINNOVPATEU' in m.list_of_moments:
#     m.list_of_moments.remove('SINNOVPATEU')
# if 'DOMPATINEU' in m.list_of_moments:
#     m.list_of_moments.remove('DOMPATINEU')
# if 'DOMPATINUS' in m.list_of_moments:
#     m.list_of_moments.remove('DOMPATINUS')

# if 'DOMPATEU' in m.list_of_moments:
#     m.list_of_moments.remove('DOMPATEU')
# if 'DOMPATUS' in m.list_of_moments:
#     m.list_of_moments.remove('DOMPATUS')
# if 'DOMPATINUS' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATINUS')
# if 'DOMPATINEU' not in m.list_of_moments:
#     m.list_of_moments.append('DOMPATINEU')

# # replacing UUPCOST with PCOST(INTER)
# if not any(mom in m.list_of_moments for mom in ['DOMPATEU','DOMPATUS','DOMPATINEU','DOMPATINEU']):
#     if 'UUPCOST' in m.list_of_moments:
#         m.list_of_moments.remove('UUPCOST')
#     if 'PCOST' in m.list_of_moments:
#         m.list_of_moments.remove('PCOST')
#     if 'PCOSTINTER' not in m.list_of_moments:
#         m.list_of_moments.append('PCOSTINTER')
# elif any(mom in m.list_of_moments for mom in ['DOMPATEU','DOMPATUS','DOMPATINEU','DOMPATINEU']):
#     if 'UUPCOST' in m.list_of_moments:
#         m.list_of_moments.remove('UUPCOST')
#     if 'PCOSTINTER' in m.list_of_moments:
#         m.list_of_moments.remove('PCOSTINTER')
#     if 'PCOST' not in m.list_of_moments:
#         m.list_of_moments.append('PCOST')
        
# # replacing UUPCOST with PCOST(INTER)NOAGG
# if not any(mom in m.list_of_moments for mom in ['DOMPATEU','DOMPATUS','DOMPATINEU','DOMPATINEU']):
#     if 'UUPCOST' in m.list_of_moments:
#         m.list_of_moments.remove('UUPCOST')
#     if 'PCOSTNOAGG' in m.list_of_moments:
#         m.list_of_moments.remove('PCOSTNOAGG')
#     if 'PCOSTINTERNOAGG' not in m.list_of_moments:
#         m.list_of_moments.append('PCOSTINTERNOAGG')
# elif any(mom in m.list_of_moments for mom in ['DOMPATEU','DOMPATUS','DOMPATINEU','DOMPATINEU']):
#     if 'UUPCOST' in m.list_of_moments:
#         m.list_of_moments.remove('UUPCOST')
#     if 'PCOSTINTERNOAGG' in m.list_of_moments:
#         m.list_of_moments.remove('PCOSTINTERNOAGG')
#     if 'PCOSTNOAGG' not in m.list_of_moments:
#         m.list_of_moments.append('PCOSTNOAGG')


m.drop_CHN_IND_BRA_ROW_from_RD = True

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 4

while cond:
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
                                xtol=1e-16, 
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
                        cobweb_anim=False,tol =1e-15,
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
# baseline_number = '501'
dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
run_number = 15.0
# run_str = '4.'
path = dropbox_path+'baseline_'+baseline_number+'_variations/'

new_baseline = False
if new_baseline:
    local_path = 'calibration_results_matched_economy/'
    path = dropbox_path
    
try:
    os.mkdir(path)
except:
    pass

write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
m.plot_moments(m.list_of_moments, save_plot = path+str(run_number))
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
