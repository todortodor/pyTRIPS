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
from data_funcs import write_calibration_results, compare_params
import os

new_run = True
baseline_number = '104'
if new_run:
    p = parameters(n=7,s=2)
    p.load_data('calibration_results_matched_economy/'+baseline_number+'/')
    # p.calib_parameters = ['eta','k','fe','T','zeta','theta','g_0',
    #                       'delta','nu','nu_tilde']
    start_time = time.perf_counter()

# list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOWDOM','SPFLOWDOM_US','SPFLOWDOM_RUS', 'SRGDP',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TE','TO']

m = moments()
m.load_data()
m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
if 'theta' in p.calib_parameters:
    p.update_sigma_with_SRDUS_target(m)
# m.list_of_moments = ['GPDIFF','GROWTH', 'KM', 'OUT', 'RD_US','RD_RUS', 'RP',
#                     'SRDUS','SPFLOW_US','SPFLOW_RUS',
#                     'SRGDP_US','SRGDP_RUS', 'JUPCOST',
#                     'SINNOVPATUS','TO']

# p.calib_parameters.append('fo')
# p.calib_parameters.remove('nu_tilde')
m.list_of_moments.append('DOMPATUS')
m.list_of_moments.append('DOMPATEU')
# m.list_of_moments.remove('JUPCOST')
# m.list_of_moments.append('JUPCOSTRD')
# m.weights_dict['SPFLOW'] = 10
m.weights_dict['SPFLOW'] = 3
m.weights_dict['SPFLOW_US'] = 3
m.weights_dict['SPFLOW_RUS'] = 3
# m.TO_target = np.array(0.02)
# m.KM_target = np.array(0.2)
m.drop_CHN_IND_BRA_ROW_from_RD = True
# m.add_domestic_US_to_SPFLOW = True
# m.add_domestic_EU_to_SPFLOW = True
if new_run:
    hist = history(*tuple(m.list_of_moments+['objective']))
bounds = p.make_parameters_bounds()
cond = True
iterations = 0
test_ls = optimize.least_squares(fun = calibration_func,    
                        x0 = p.make_p_vector(), 
                        args = (p,m,p.guess,hist,start_time), 
                        bounds = bounds,
                        # method= 'dogbox',
                        # loss='arctan',
                        # jac='3-point',
                        max_nfev=1e8,
                        # ftol=1e-14, 
                        xtol=1e-11, 
                        # gtol=1e-14,
                        # f_scale=scale,
                        verbose = 2)
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
sol_c = var.var_from_vector(sol.x, p_sol)    
sol_c.scale_P(p_sol)
sol_c.compute_price_indices(p)
sol_c.compute_non_solver_quantities(p_sol) 
m.compute_moments(sol_c,p_sol)
m.compute_moments_deviations()
m.plot_moments(m.list_of_moments)

#%% writing results as excel and locally

commentary = 'drop South, DOMPAT moments, weight3 SPFLOW'
# commentary = ''
dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
# local_path = 'calibration_results_matched_economy/'
# baseline_number = '102'
run_number = 8.2
# run_number = baseline_number
path = dropbox_path+'baseline_'+baseline_number+'_variations/'
# path = dropbox_path
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