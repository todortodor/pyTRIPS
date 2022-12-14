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
baseline_number = '101'
if new_run:
    p = parameters(n=7,s=2)
    # p.load_data('calibration_results_matched_economy/'+baseline_number+'/')
    p.load_data('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/11.7/')
    # p.calib_parameters = ['eta','k','fe','T','zeta','theta','g_0',
    #                       'delta','nu','nu_tilde']
    start_time = time.perf_counter()

# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOWDOM','SPFLOWDOM_US','SPFLOWDOM_RUS', 'SRGDP',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TE','TO']

m = moments()
m.load_data()
m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/11.7/')
if 'theta' in p.calib_parameters:
    p.update_sigma_with_SRDUS_target(m)
# m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOW_US','SPFLOW_RUS',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TO']

# uncomment following for run 11.7
if 'd' not in p.calib_parameters:
    p.calib_parameters.append('d')
if 'DOMPATEU' not in m.list_of_moments:
    m.list_of_moments.append('DOMPATEU')
if 'DOMPATUS' not in m.list_of_moments:
    m.list_of_moments.append('DOMPATUS')
m.drop_CHN_IND_BRA_ROW_from_RD = True

# if 'kappa' not in p.calib_parameters:
#     p.calib_parameters.append('kappa')
    
# if 'ERDUS' not in m.list_of_moments:
#     m.list_of_moments.append('ERDUS')
#     m.weights_dict['ERDUS'] = 5

# p.update_khi_and_r_hjort(0.16)
p.r_hjort[3] = 17.33029162

avoid_bad_nash = False
# p.kappa = np.array(0.75)
# m.list_of_moments.remove('SPFLOW')
# m.list_of_moments.remove('SPFLOW_RUS')
# m.list_of_moments.append('SPFLOWDOM')
# m.list_of_moments.append('SPFLOWDOM_RUS')
# m.list_of_moments.remove('KM')
# m.list_of_moments.append('KM_GDP')
# m.weights_dict['SPFLOW'] = 10
# m.weights_dict['SPFLOW'] = 3
# m.weights_dict['SPFLOW_US'] = 3
# m.weights_dict['SPFLOW_RUS'] = 3
# m.TO_target = np.array(0.0465)
# m.KM_target = np.array(0.1322)
# m.GROWTH_target = np.array(0.02)
# m.GROWTH_target = np.array(0.03)
# m.add_domestic_US_to_SPFLOW = True
# m.add_domestic_EU_to_SPFLOW = True
p.calib_parameters.remove('eta')

if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
    bad_nash_weight = 1e2
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
# if avoid_bad_nash:
#     x0 = np.concatenate([p.make_p_vector()

while cond:
    if iterations < 14:
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
                                method= 'dogbox',
                                # loss='arctan',
                                # jac='3-point',
                                max_nfev=200,
                                # ftol=1e-14, 
                                xtol=1e-16, 
                                # gtol=1e-14,
                                # f_scale=scale,
                                verbose = 2)
    if avoid_bad_nash:
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)

        sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=True,
                                plot_cobweb=True,
                                safe_convergence=0.001,
                                disp_summary=False,
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
        sol_c = var.var_from_vector(sol.x, p_sol)    
        sol_c.scale_P(p_sol)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p_sol) 
        US_deriv_w_to_d = np.array(compute_deriv_welfare_to_patent_protec_US(sol_c,p,p.guess))
        cond = US_deriv_w_to_d<-1e-8
        bad_nash_weight = bad_nash_weight*5
    else:
        # cond = test_ls.nfev>15
        cond = iterations < 15
        iterations += 1
        p.update_parameters(test_ls.x)
        
    cost = test_ls.cost
finish_time = time.perf_counter()
print('minimizing time',finish_time-start_time)

p_sol = p.copy()
p_sol.update_parameters(test_ls.x)

sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                        cobweb_anim=False,tol =1e-15,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=True,
                        plot_cobweb=True,
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
sol_c.compute_price_indices(p_sol)
sol_c.compute_non_solver_quantities(p_sol) 
m.compute_moments(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

#%%

# p_sol.eta[0,1] = 0.0005
# c = 'USA'

# deltas, welfares = find_nash_eq(p_sol,lb_delta=0.01,ub_delta=100,method='fixed_point',
#                   plot_convergence = True,solver_options=None,tol=5e-3,window=4,
#                   initial_small_change_newton=0.5)

# delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
#                                      method='bounded',
#                                      bounds=(0.01, 100),
#                                      args = (p_sol,c,sol_c),
#                                      options = {'disp':3})

#%% writing results as excel and locally

commentary = '11.7, keep eta fixed add hjort factor only for China'
# commentary = ''
dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
# local_path = 'calibration_results_matched_economy/'
# baseline_number = '102'
run_number = 20.2
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

try:
    os.mkdir(local_path)
except:
    pass
p_sol.write_params(local_path+str(run_number)+'/')
m.write_moments(local_path+str(run_number)+'/')

#%%
# import matplotlib.pyplot as plt
# jac = test_ls.jac
# IND_idx = -4
# m_sign_list = ['GPDIFF',
#  'GROWTH',
#  'KM',
#  'OUT',
#  'RD',
#  'RD',
#  'RD',
#  'RP',
#  'RP',
#  'RP',
#  'RP',
#  'RP',
#  'RP',
#  'RP',
#  'SRDUS',
#  'SRGDP',
#  'SRGDP',
#  'SRGDP',
#  'SRGDP',
#  'SRGDP',
#  'SRGDP',
#  'SRGDP',
#  'JUPCOST',
#  'SINNOVPATUS',
#  'TO',
#  'SPFLOW USA_EUR', 'SPFLOW USA_JAP', 'SPFLOW USA_CHN', 'SPFLOW USA_BRA', 'SPFLOW USA_IND', 'SPFLOW USA_ROW', 'SPFLOW EUR_USA', 'SPFLOW EUR_JAP', 'SPFLOW EUR_CHN', 'SPFLOW EUR_BRA', 'SPFLOW EUR_IND', 'SPFLOW EUR_ROW', 'SPFLOW JAP_USA', 'SPFLOW JAP_EUR', 'SPFLOW JAP_CHN', 'SPFLOW JAP_BRA', 'SPFLOW JAP_IND', 'SPFLOW JAP_ROW', 'SPFLOW CHN_USA', 'SPFLOW CHN_EUR', 'SPFLOW CHN_JAP', 'SPFLOW CHN_BRA', 'SPFLOW CHN_IND', 'SPFLOW CHN_ROW', 'SPFLOW BRA_USA', 'SPFLOW BRA_EUR', 'SPFLOW BRA_JAP', 'SPFLOW BRA_CHN', 'SPFLOW BRA_IND', 'SPFLOW BRA_ROW', 'SPFLOW IND_USA', 'SPFLOW IND_EUR', 'SPFLOW IND_JAP', 'SPFLOW IND_CHN', 'SPFLOW IND_BRA', 'SPFLOW IND_ROW', 'SPFLOW ROW_USA', 'SPFLOW ROW_EUR', 'SPFLOW ROW_JAP', 'SPFLOW ROW_CHN', 'SPFLOW ROW_BRA', 'SPFLOW ROW_IND',
#  'DOMPATEU',
#  'DOMPATUS']

# # for i in range(3,6):
# # fig,ax=plt.subplots(figsize = (20,10))
# # ax.plot(jac[:,-4])
# # ax.set_xticks(np.arange(len(jac[:,-4])))
# # ax.set_xticklabels(m_sign_list,rotation = 45)
# # plt.show()
# fig,ax=plt.subplots(figsize = (15,20))
# ax.plot(after-before,np.arange(len(jac[:,-4])))
# ax.set_yticks(np.arange(len(jac[:,-4])))
# ax.set_yticklabels(m_sign_list)
# plt.show()

# SPFLOW_sign = []
# for c1 in p.countries:
#     for c2 in p.countries:
#         if c1 != c2:
#             SPFLOW_sign.append('SPFLOW '+c1+'_'+c2)
# print(SPFLOW_sign)
