#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:00:03 2022

@author: simonl
"""

from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver, compute_deriv_welfare_to_patent_protec_US
from data_funcs import write_calibration_results, compare_params
import os
import numpy as np
from solver_funcs import find_nash_eq, minus_welfare_of_delta

new_run = True
baseline_number = '404'
if new_run:
    p = parameters(n=7,s=2)
    p.load_data('calibration_results_matched_economy/'+baseline_number+'/')
    # p.load_data('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/2.1.3/')
    # p.calib_parameters = ['eta', 'k', 'fe', 'T', 'zeta', 'g_0', 'delta', 'nu', 'fo']
    # p.calib_parameters = ['eta', 'k', 'fe', 'T', 'zeta', 'g_0', 'delta', 'nu', 'd']
    start_time = time.perf_counter()
    
# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOWDOM','SPFLOWDOM_US','SPFLOWDOM_RUS', 'SRGDP',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TE','TO']

    m = moments()
    m.load_data()
    m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    # m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/2.1.3/')
    
    # m_back_up = m.copy()
    # p_back_up = m.copy()
    
# m.SINNOVPATEU_target = np.float64(0.3475)

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

avoid_bad_nash = False
 

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
    bad_nash_weight = 1e2
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
max_iter = 5
# if avoid_bad_nash:
#     x0 = np.concatenate([p.make_p_vector()

while cond:
    if iterations < max_iter - 2:
        test_ls = optimize.least_squares(fun = calibration_func,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time,avoid_bad_nash,bad_nash_weight), 
                                bounds = bounds,
                                # method= 'dogbox',
                                # loss='arctan',
                                # jac='3-point',
                                max_nfev=1e8,
                                # ftol=1e-14, 
                                xtol=1e-10, 
                                # gtol=1e-14,
                                # f_scale=scale,
                                verbose = 2)
    else:
        test_ls = optimize.least_squares(fun = calibration_func,    
                                x0 = p.make_p_vector(), 
                                args = (p,m,p.guess,hist,start_time,avoid_bad_nash,bad_nash_weight), 
                                bounds = bounds,
                                # method= 'dogbox',
                                # loss='arctan',
                                # jac='3-point',
                                max_nfev=1e8,
                                # ftol=1e-14, 
                                xtol=1e-16, 
                                # gtol=1e-14,
                                # f_scale=scale,
                                verbose = 2)
        # cond = test_ls.nfev>15
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
                        # damping=10
                          # apply_bound_psi_star=True
                        )
p_sol.guess = sol.x
# sol_c = var.var_from_vector(sol.x, p_sol)    
# sol_c = var.var_from_vector(p_sol.guess, p_sol)    
sol_c.scale_P(p_sol)

# sol_c.compute_price_indices(p_sol)
sol_c.compute_non_solver_quantities(p_sol) 
p_sol.tau = sol_c.tau
m.compute_moments(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

#%% writing results as excel and locally

# commentary = 'With DOMPATINUS/EU and SINNOVPATEU'
# commentary = 'With PCOSTNOAGG and no DOMPAT'
commentary = 'New baseline 404'
# commentary = ''
baseline_number = '404'
dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
# local_path = 'calibration_results_matched_economy/'
run_number = 1.0
# run_str = '4.'
# run_number = baseline_number
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

#%%
import matplotlib.pyplot as plt
# jac = test_ls.jac
# IND_idx = -4
m_sign_list = ['GPDIFF',
  'GROWTH',
  'KM',
  'OUT',
  'RD',
  'RD',
  'RD',
  'RP',
  'RP',
  'RP',
  'RP',
  'RP',
  'RP',
  'RP',
  'SRDUS',
  'SRGDP',
  'SRGDP',
  'SRGDP',
  'SRGDP',
  'SRGDP',
  'SRGDP',
  'SRGDP',
  'JUPCOST',
  'SINNOVPATUS',
  'TO',
  'SPFLOW USA_EUR', 'SPFLOW USA_JAP', 'SPFLOW USA_CHN', 'SPFLOW USA_BRA', 'SPFLOW USA_IND', 'SPFLOW USA_ROW', 'SPFLOW EUR_USA', 'SPFLOW EUR_JAP', 'SPFLOW EUR_CHN', 'SPFLOW EUR_BRA', 'SPFLOW EUR_IND', 'SPFLOW EUR_ROW', 'SPFLOW JAP_USA', 'SPFLOW JAP_EUR', 'SPFLOW JAP_CHN', 'SPFLOW JAP_BRA', 'SPFLOW JAP_IND', 'SPFLOW JAP_ROW', 'SPFLOW CHN_USA', 'SPFLOW CHN_EUR', 'SPFLOW CHN_JAP', 'SPFLOW CHN_BRA', 'SPFLOW CHN_IND', 'SPFLOW CHN_ROW', 'SPFLOW BRA_USA', 'SPFLOW BRA_EUR', 'SPFLOW BRA_JAP', 'SPFLOW BRA_CHN', 'SPFLOW BRA_IND', 'SPFLOW BRA_ROW', 'SPFLOW IND_USA', 'SPFLOW IND_EUR', 'SPFLOW IND_JAP', 'SPFLOW IND_CHN', 'SPFLOW IND_BRA', 'SPFLOW IND_ROW', 'SPFLOW ROW_USA', 'SPFLOW ROW_EUR', 'SPFLOW ROW_JAP', 'SPFLOW ROW_CHN', 'SPFLOW ROW_BRA', 'SPFLOW ROW_IND',
  'DOMPATEU',
  'DOMPATUS']

# for i in range(3,6):
# fig,ax=plt.subplots(figsize = (20,10))
# ax.plot(jac[:,-4])
# ax.set_xticks(np.arange(len(jac[:,-4])))
# ax.set_xticklabels(m_sign_list,rotation = 45)
# plt.show()
fig,ax=plt.subplots(figsize = (15,20))
ax.plot(m.deviation_vector()-m_back_up.deviation_vector(),np.arange(len(m.deviation_vector())))
ax.set_yticks(np.arange(len(m.deviation_vector())))
ax.set_yticklabels(m_sign_list)
plt.grid()

plt.show()

# SPFLOW_sign = []
# for c1 in p.countries:
#     for c2 in p.countries:
#         if c1 != c2:
#             SPFLOW_sign.append('SPFLOW '+c1+'_'+c2)
# print(SPFLOW_sign)
