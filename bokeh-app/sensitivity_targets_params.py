#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:06:50 2022

@author: simonl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import time
import os
import seaborn as sns
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver, compute_deriv_welfare_to_patent_protec_US, compute_deriv_growth_to_patent_protec_US
from data_funcs import write_calibration_results, compare_params
from functools import reduce
from tqdm import tqdm

#%% define baseline and conditions of sensitivity analysis

baseline = '403'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)
sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                       context='calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 5e4,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )

sol_baseline = var.var_from_vector(sol.x, p_baseline,context = 'calibration')    
sol_baseline.scale_P(p_baseline)
# sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

m_baseline.compute_moments(sol_baseline, p_baseline)

moments_to_change = ['KM','UUPCOST','SINNOVPATUS','TO','GROWTH','SRDUS',
                     'SINNOVPATEU','DOMPATINUS','DOMPATINEU','TE']
parameters_to_change = ['kappa','gamma','rho','zeta']

dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'

parent_moment_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_targets_variations/'
parent_moment_dropbox_path = dropbox_path+'baseline_'+baseline+'_targets_variation/'

parent_param_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_parameters_variations/'
parent_param_dropbox_path = dropbox_path+'baseline_'+baseline+'_parameters_variation/'

sensitivity_path = dropbox_path+'baseline_'+baseline+'_sensitivities/'
sensitivity_tables_path = 'calibration_results_matched_economy/baseline_'+baseline+'_sensitivity_tables/'

def make_dirs(list_of_paths):
    for path in list_of_paths:
        try:
            os.mkdir(path)
        except:
            pass

def GetSpacedElements(array, numElems = 13):
    idx = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    out = array[idx]
    return out, idx

#%% make dirs

make_dirs([parent_moment_result_path,
           parent_moment_dropbox_path,
           parent_param_result_path,
           parent_param_dropbox_path,
           sensitivity_path])

#%% make calibration runs for different moment(s) target

dic_runs = dict([(mom, np.linspace(getattr(m_baseline,mom+'_target')*0.5,getattr(m_baseline,mom+'_target')*1.5,11))
                 for mom in moments_to_change])

for k, v in dic_runs.items():
    print(k)
    print(v)
    moment_to_change = k
    target_list = v
    result_path = parent_moment_result_path+moment_to_change+'/'
    dropbox_path = parent_moment_dropbox_path+moment_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,target in enumerate(target_list):
        print(k)
        print(v)
        print(target)
        m = moments()
        m.load_data()
        m.load_run(baseline_path)
        m.drop_CHN_IND_BRA_ROW_from_RD = True
        p = parameters(n=7,s=2)
        p.load_data(baseline_path)
        # if moment_to_change == 'ERDUS' and 'ERDUS' not in m.list_of_moments:
        #     m.list_of_moments.append('ERDUS')
        #     if 'kappa' not in p.calib_parameters:
        #         p.calib_parameters.append('kappa')
        setattr(m,moment_to_change+'_target',target)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        cond = True
        iterations = 0
        max_iter = 5
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
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
                                        args = (p,m,p.guess,hist,start_time), 
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
            cond = iterations < max_iter
            iterations += 1
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                        context = 'calibration',
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
    
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.guess = sol.x
        p_sol.tau = sol_c.tau
        
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        
#%% make calibration runs for different parameters

dic_runs = dict([(par, np.linspace(getattr(p_baseline,par)*0.5,getattr(p_baseline,par)*1.5,11))
                 for par in parameters_to_change])
# if 'kappa' in parameters_to_change:
#     dic_runs['kappa'] = np.linspace(getattr(p_baseline,'kappa')*0.8,getattr(p_baseline,'kappa')*1.2,21)
if 'zeta' in parameters_to_change:
    dic_runs['zeta'] = [np.array([p_baseline.zeta[0], i]) for i in np.linspace(0,0.1,21)]
    # p_baseline.calib_parameters.remove('zeta')

for k, v in dic_runs.items():
    print(k)
    print(v)
    par_to_change = k
    par_list = v
    result_path = parent_param_result_path+par_to_change+'/'
    dropbox_path = parent_param_dropbox_path+par_to_change+'/'
    
    try:
        os.mkdir(result_path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass
    
    for i,par in enumerate(par_list):
        print(k)
        print(v)
        print(par)
        m = moments()
        m.load_data()
        m.load_run(baseline_path)
        # m.drop_CHN_IND_BRA_ROW_from_RD = True
        p = parameters(n=7,s=2)
        p.load_data(baseline_path)
        if par_to_change == 'zeta':
            if 'zeta' in p.calib_parameters:
                p.calib_parameters.remove('zeta')
        setattr(p,par_to_change,par)
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        hist = history(*tuple(m.list_of_moments+['objective']))
        cond = True
        iterations = 0
        max_iter = 5
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
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
                                        args = (p,m,p.guess,hist,start_time), 
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
            cond = iterations < max_iter
            iterations += 1
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        sol, sol_c = fixed_point_solver(p_sol,
                                        context = 'calibration',
                                        x0=p_sol.guess,
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
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.guess = sol.x
        p_sol.tau = sol_c.tau
        
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        
        p_sol.write_params(result_path+str(i)+'/')
        m.write_moments(result_path+str(i)+'/')
        
        write_calibration_results(dropbox_path+str(i),p_sol,m,sol_c,commentary = '')
        # m.plot_moments(m.list_of_moments, save_plot = dropbox_path+str(i))

#%% write summaries calibration runs for moment target change

with_dropped = False
save = True

for moment_to_change in moments_to_change:
    print(moment_to_change)
    result_path = parent_moment_result_path+moment_to_change+'/'
    dropbox_path = parent_moment_dropbox_path+moment_to_change+'/'
    dropbox_summary_path = dropbox_path+'summary/'
    if with_dropped:
        dropbox_summary_path = dropbox_path+'summary_with_dropped_moment/'
    
    try:
        os.mkdir(result_path+'summary/')
    except:
        pass
    try:
        os.mkdir(dropbox_summary_path)
    except:
        pass
    
    if moment_to_change == 'sales_mark_up_US':
        m_baseline.get_sales_mark_up_US_from_sigma(p_baseline)
    
    baseline_moment = getattr(m_baseline, moment_to_change+'_target')
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            print(p.sigma)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,
                                            context = 'calibration',x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            run_name = run+': '+moment_to_change+str(getattr(m,moment_to_change+'_target'))
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
        
        elif run == '99' and with_dropped:
            run_list.remove('99')
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,
                                            context = 'calibration',x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            dropped_m = m
            dropped_p = p
            dropped_sol = sol_c
            
        
    targets = np.array([getattr(m,moment_to_change+'_target') for m in dic_m.values()]).squeeze()
    
    # moments.compare_moments(dic_m, contin_cmap=True)
    # compare_params(dic_p)
    
    # moments.compare_moments(dic_m, contin_cmap=True, save_path = dropbox_summary_path)
    # compare_params(dic_p, save = True, save_path = dropbox_summary_path)
    
    # fig,ax = plt.subplots(figsize=(12,8))
    # ax2 = ax.twinx()
    
    # nus = [p.nu[1] for p in dic_p.values()]
    # deltas_US = [p.delta[0,1] for p in dic_p.values()]
    # deltas_mean = [p.delta[:,1].mean() for p in dic_p.values()]
    
    # ax.plot(targets,nus,color=sns.color_palette()[0],label='nu')
    # ax2.plot(targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    # ax2.plot(targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    # ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    # ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    # ax.set_ylabel('Nu',fontsize = 15)
    # ax2.set_ylabel('Delta',fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    
    up_ten_percent_moment_index = np.abs(baseline_moment*1.1-targets).argmin()
    down_ten_percent_moment_index = np.abs(baseline_moment*0.9-targets).argmin()
    
    up_ten_percent_run = runs[up_ten_percent_moment_index]
    down_ten_percent_run = runs[down_ten_percent_moment_index]
    
    down_change_str = '% change for a '+str(((targets[down_ten_percent_moment_index]-baseline_moment)*100/baseline_moment).round(2))+'% change in moment'
    up_change_str = '% change for a '+str(((targets[up_ten_percent_moment_index]-baseline_moment)*100/baseline_moment).round(2))+'% change in moment'
    
    table_10 = pd.DataFrame(columns = [down_change_str,up_change_str,'baseline_value'])
    
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['delta '+c] = [(dic_p[down_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    (dic_p[up_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    p_baseline.delta[i,1]
                                    ]
        
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['eta '+c] = [(dic_p[down_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    (dic_p[up_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    p_baseline.eta[i,1]
                                    ]
    
    table_10.loc['nu'] = [(dic_p[down_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                (dic_p[up_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                p_baseline.nu[1]
                                ]
    table_10.loc['fe'] = [(dic_p[down_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                (dic_p[up_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                p_baseline.fe[1]
                                ]
    table_10.loc['k'] = [(dic_p[down_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                (dic_p[up_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                p_baseline.k
                                ]
    table_10.loc['zeta'] = [(dic_p[down_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                (dic_p[up_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                p_baseline.zeta
                                ]
    table_10.loc['nu_tilde'] = [(dic_p[down_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                (dic_p[up_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                p_baseline.nu_tilde[1]
                                ]
    table_10.loc['theta'] = [(dic_p[down_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                (dic_p[up_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                p_baseline.theta[1]
                                ]
    table_10.loc['sigma'] = [(dic_p[down_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                (dic_p[up_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                p_baseline.sigma[1]
                                ]
    if save:
        table_10.to_csv(dropbox_summary_path +'plus_minus_ten_percent_change_table.csv')
    
    less_targets, less_idx = GetSpacedElements(targets,len(targets))
    less_runs = [runs[i] for i in less_idx]
    less_dic_m = {}
    less_dic_p = {}
    less_dic_sol = {}
    for r in less_runs:
        less_dic_m[r] = dic_m[r]
        less_dic_p[r] = dic_p[r]
        less_dic_sol[r] = dic_sol[r]
        
    temp_m = less_dic_m.copy()    
    try:
        temp_m['99: '+moment_to_change+'dropped'] = dropped_m
    except:
        pass
    temp_p = less_dic_p.copy()   
    try:
        temp_p['99: '+moment_to_change+'dropped'] = dropped_p
    except:
        pass
        
    if save:    
        moments.compare_moments(temp_m, contin_cmap=True, save_path = dropbox_summary_path)
        compare_params(temp_p, save = True, save_path = dropbox_summary_path)
    else:
        moments.compare_moments(temp_m, contin_cmap=True)
        compare_params(temp_p, save = False)    
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    
    nus = [p.nu[1] for p in less_dic_p.values()]
    nus_tilde = [p.nu_tilde[1] for p in less_dic_p.values()]
    thetas = [p.theta[1] for p in less_dic_p.values()]
    deltas_US = [p.delta[0,1] for p in less_dic_p.values()]
    deltas_EU = [p.delta[1,1] for p in less_dic_p.values()]
    deltas_JAP = [p.delta[2,1] for p in less_dic_p.values()]
    deltas_mean = [p.delta[:,1].mean() for p in less_dic_p.values()]
    
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax2.plot(less_targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    ax2.plot(less_targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    try:
        ax2.plot([],[],lw=0,label=f"\n Moment dropped :\n nu :{dropped_p.nu[1]:.3}\n delta_US:{dropped_p.delta[0,1]:.3}\n delta_average:{dropped_p.delta[:,1].mean():.3}")
    except:
        pass
    
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Nu',fontsize = 15)
    ax2.set_ylabel('Delta',fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    
    ax.plot(less_targets,deltas_US,label='US')
    ax.plot(less_targets,deltas_EU,label='EUR')
    ax.plot(less_targets,deltas_JAP,label='JAP')
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Delta',fontsize = 15)
    try:
        ax.plot([],[],lw=0,label=f"\n Moment dropped :\n delta_US:{dropped_p.delta[0,1]:.3}\n delta_EUR:{dropped_p.delta[1,1]:.3}\n delta_JAP:{dropped_p.delta[2,1]:.3}")
    except:
        pass
    ax.legend()
    if save:
        plt.savefig(dropbox_summary_path +'delta_US_EUR_JAP.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax.plot(less_targets,nus_tilde,color=sns.color_palette()[1],label='nu_tilde')
    ax2.plot(less_targets,thetas,color=sns.color_palette()[2],label='theta')    
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    ax.set_ylabel('Nu and nu_tilde',fontsize = 15)
    ax2.set_ylabel('Theta',fontsize = 15)
    # if not with_dropped:
    try:
        ax2.plot([],[],lw=0,label=f"\n Moment dropped :\n nu :{dropped_p.nu[1]:.3}\n nu_tilde :{dropped_p.nu_tilde[1]:.3}\n theta :{dropped_p.theta[1]:.3}")
    except:
        pass
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_nu_tilde_theta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ratio_delta_US_to_diffusion = np.array(deltas_US)/np.array(nus)
    ax.plot(less_targets,ratio_delta_US_to_diffusion)
    ax.set_ylabel('Delta_US / Nu',fontsize = 15)
    ax.set_xlabel(moment_to_change+'_target',fontsize = 15)
    # if with_dropped:
    try:
        ax.plot([],[],lw=0,label=f" Moment dropped :\n delta_US/nu :{dropped_p.delta[0,1]/dropped_p.nu[1]:.3}")
    except:
        pass
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'patent_protection_US_to_diffusion_ratio.png')
    plt.show()

#%% write summaries calibration runs for parameter change

save = True

# parameter_to_change = 'SRDUS'
for parameter_to_change in parameters_to_change:
# for parameter_to_change in ['zeta']:
    print(parameter_to_change)
    result_path = parent_param_result_path+parameter_to_change+'/'
    dropbox_path = parent_param_dropbox_path+parameter_to_change+'/'
    dropbox_summary_path = dropbox_path+'summary/'
    
    try:
        os.mkdir(result_path+'summary/')
    except:
        pass
    try:
        os.mkdir(dropbox_summary_path)
    except:
        pass
    
    try:
        baseline_param = getattr(p_baseline, parameter_to_change)[1]
    except:
        baseline_param = getattr(p_baseline, parameter_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        print(run)
        run_path = result_path+run+'/'
        p = parameters(n=7,s=2)
        p.load_data(run_path)
        m = moments()
        m.load_data()
        m.load_run(run_path)
        sol, sol_c = fixed_point_solver(p,
                                        context = 'calibration',
                                        x0=p.guess,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 5e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        sol_c = var.var_from_vector(sol.x, p,context='calibration')    
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p) 
        m.compute_moments(sol_c,p)
        m.compute_moments_deviations()
        
        try:
            run_name = run+': '+parameter_to_change+str(getattr(p,parameter_to_change)[1])
        except:
            run_name = run+': '+parameter_to_change+str(getattr(p,parameter_to_change))
        runs.append(run_name)
        dic_m[run_name] = m
        dic_p[run_name] = p
        dic_sol[run_name] = sol_c
    
    try:
        targets = np.array([getattr(p,parameter_to_change)[1] for p in dic_p.values()]).squeeze()
    except:
        targets = np.array([getattr(p,parameter_to_change) for p in dic_p.values()]).squeeze()
    
    # moments.compare_moments(dic_m, contin_cmap=True)
    # compare_params(dic_p)
    
    # moments.compare_moments(dic_m, contin_cmap=True, save_path = dropbox_summary_path)
    # compare_params(dic_p, save = True, save_path = dropbox_summary_path)
    
    # fig,ax = plt.subplots(figsize=(12,8))
    # ax2 = ax.twinx()
    
    # nus = [p.nu[1] for p in dic_p.values()]
    # deltas_US = [p.delta[0,1] for p in dic_p.values()]
    # deltas_mean = [p.delta[:,1].mean() for p in dic_p.values()]
    
    # ax.plot(targets,nus,color=sns.color_palette()[0],label='nu')
    # ax2.plot(targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    # ax2.plot(targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    # ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    # ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_xlabel(parameter_to_change+'_target',fontsize = 15)
    # ax.set_ylabel('Nu',fontsize = 15)
    # ax2.set_ylabel('Delta',fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    
    up_ten_percent_param_index = np.abs(baseline_param*1.1-targets).argmin()
    down_ten_percent_param_index = np.abs(baseline_param*0.9-targets).argmin()
    
    up_ten_percent_run = runs[up_ten_percent_param_index]
    down_ten_percent_run = runs[down_ten_percent_param_index]
    
    down_change_str = '% change for a '+str(((targets[down_ten_percent_param_index]-baseline_param)*100/baseline_param).round(2))+'% change in parameter'
    up_change_str = '% change for a '+str(((targets[up_ten_percent_param_index]-baseline_param)*100/baseline_param).round(2))+'% change in parameter'
    
    table_10 = pd.DataFrame(columns = [down_change_str,up_change_str,'baseline_value'])
    
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['delta '+c] = [(dic_p[down_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    (dic_p[up_ten_percent_run].delta[i,1]-p_baseline.delta[i,1])*100/p_baseline.delta[i,1],
                                    p_baseline.delta[i,1]
                                    ]
        
    for i,c in enumerate(p_baseline.countries):
        table_10.loc['eta '+c] = [(dic_p[down_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    (dic_p[up_ten_percent_run].eta[i,1]-p_baseline.eta[i,1])*100/p_baseline.eta[i,1],
                                    p_baseline.eta[i,1]
                                    ]
    
    table_10.loc['nu'] = [(dic_p[down_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                (dic_p[up_ten_percent_run].nu[1]-p_baseline.nu[1])*100/p_baseline.nu[1],
                                p_baseline.nu[1]
                                ]
    table_10.loc['fe'] = [(dic_p[down_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                (dic_p[up_ten_percent_run].fe[1]-p_baseline.fe[1])*100/p_baseline.fe[1],
                                p_baseline.fe[1]
                                ]
    table_10.loc['k'] = [(dic_p[down_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                (dic_p[up_ten_percent_run].k-p_baseline.k)*100/p_baseline.k,
                                p_baseline.k
                                ]
    table_10.loc['zeta'] = [(dic_p[down_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                (dic_p[up_ten_percent_run].zeta-p_baseline.zeta)*100/p_baseline.zeta,
                                p_baseline.zeta
                                ]
    table_10.loc['nu_tilde'] = [(dic_p[down_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                (dic_p[up_ten_percent_run].nu_tilde[1]-p_baseline.nu_tilde[1])*100/p_baseline.nu_tilde[1],
                                p_baseline.nu_tilde[1]
                                ]
    table_10.loc['theta'] = [(dic_p[down_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                (dic_p[up_ten_percent_run].theta[1]-p_baseline.theta[1])*100/p_baseline.theta[1],
                                p_baseline.theta[1]
                                ]
    table_10.loc['sigma'] = [(dic_p[down_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                (dic_p[up_ten_percent_run].sigma[1]-p_baseline.sigma[1])*100/p_baseline.sigma[1],
                                p_baseline.sigma[1]
                                ]
    if save:
        table_10.to_csv(dropbox_summary_path +'plus_minus_ten_percent_change_table.csv')
    
    less_targets, less_idx = GetSpacedElements(targets,len(targets))
    less_runs = [runs[i] for i in less_idx]
    less_dic_m = {}
    less_dic_p = {}
    less_dic_sol = {}
    for r in less_runs:
        less_dic_m[r] = dic_m[r]
        less_dic_p[r] = dic_p[r]
        less_dic_sol[r] = dic_sol[r]
        
    if save:    
        moments.compare_moments(less_dic_m, contin_cmap=True, save_path = dropbox_summary_path)
        compare_params(less_dic_p, save = True, save_path = dropbox_summary_path)
    else:
        moments.compare_moments(less_dic_m, contin_cmap=True)
        compare_params(less_dic_p, save = False)
        
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    
    nus = [p.nu[1] for p in less_dic_p.values()]
    nus_tilde = [p.nu_tilde[1] for p in less_dic_p.values()]
    thetas = [p.theta[1] for p in less_dic_p.values()]
    deltas_US = [p.delta[0,1] for p in less_dic_p.values()]
    deltas_EU = [p.delta[1,1] for p in less_dic_p.values()]
    deltas_JAP = [p.delta[2,1] for p in less_dic_p.values()]
    deltas_mean = [p.delta[:,1].mean() for p in less_dic_p.values()]
    
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax2.plot(less_targets,deltas_US,color=sns.color_palette()[1],label='delta_US')
    ax2.plot(less_targets,deltas_mean,color=sns.color_palette()[2],label='delta_average')
    
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Nu',fontsize = 15)
    ax2.set_ylabel('Delta',fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_and_delta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    
    ax.plot(less_targets,deltas_US,label='US')
    ax.plot(less_targets,deltas_EU,label='EUR')
    ax.plot(less_targets,deltas_JAP,label='JAP')
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Delta',fontsize = 15)
    ax.legend()
    if save:
        plt.savefig(dropbox_summary_path +'delta_US_EUR_JAP.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    ax.plot(less_targets,nus,color=sns.color_palette()[0],label='nu')
    ax.plot(less_targets,nus_tilde,color=sns.color_palette()[1],label='nu_tilde')
    ax2.plot(less_targets,thetas,color=sns.color_palette()[2],label='theta')    
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    ax.set_ylabel('Nu and nu_tilde',fontsize = 15)
    ax2.set_ylabel('Theta',fontsize = 15)
    ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'nu_nu_tilde_theta.png')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    ratio_delta_US_to_diffusion = np.array(deltas_US)/np.array(nus)
    ax.plot(less_targets,ratio_delta_US_to_diffusion)
    ax.set_ylabel('Delta_US / Nu',fontsize = 15)
    ax.set_xlabel(parameter_to_change,fontsize = 15)
    plt.tight_layout()
    if save:
        plt.savefig(dropbox_summary_path +'patent_protection_US_to_diffusion_ratio.png')
    plt.show()

#%% Gather all parameters and moments variations

dic_of_variation_dics = {}

for moment_to_change in moments_to_change:
    variation_dic = {}
    print(moment_to_change)
    result_path = parent_moment_result_path+moment_to_change+'/'
    
    baseline_moment = getattr(m_baseline, moment_to_change+'_target')
    if moment_to_change == 'ERDUS':
        baseline_moment = getattr(m_baseline, moment_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_values = {}
    dic_change = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,
                                            context = 'calibration',
                                            x0=p.guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            if moment_to_change == 'sales_mark_up_US':
                m.get_sales_mark_up_US_from_sigma(p)
            
            run_name = run+': '+moment_to_change+str(getattr(m,moment_to_change+'_target'))
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(getattr(m,moment_to_change+'_target'))
            dic_change[run_name] = float((getattr(m,moment_to_change+'_target')-baseline_moment)*100/baseline_moment)
        
    variation_dic['changing_quantity'] = moment_to_change+'_target'
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_moment
    
    dic_of_variation_dics[moment_to_change+'_target'] = variation_dic
    
for parameter_to_change in parameters_to_change:
    variation_dic = {}
    print(parameter_to_change)
    result_path = parent_param_result_path+parameter_to_change+'/'
    
    try:
        baseline_param = getattr(p_baseline, parameter_to_change)[1]
    except:
        baseline_param = getattr(p_baseline, parameter_to_change)
    
    dic_p = {}
    dic_m = {}
    dic_sol = {}
    dic_change = {}
    dic_values = {}
    runs = []
    
    files_in_dir = next(os.walk(result_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    try:
        run_list.remove('99')
    except:
        pass
    
    for run in run_list:
        if run != '99':
            print(run)
            run_path = result_path+run+'/'
            p = parameters(n=7,s=2)
            p.load_data(run_path)
            m = moments()
            m.load_data()
            m.load_run(run_path)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'calibration',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 5e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c = var.var_from_vector(sol.x, p,context='calibration')    
            sol_c.scale_P(p)
            sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p) 
            m.compute_moments(sol_c,p)
            m.compute_moments_deviations()
            
            try:
                current_param = getattr(p,parameter_to_change)[1]
            except:
                current_param = getattr(p,parameter_to_change)
            
            run_name = run+': '+parameter_to_change+str(current_param)
            runs.append(run_name)
            dic_m[run_name] = m
            dic_p[run_name] = p
            dic_sol[run_name] = sol_c
            dic_values[run_name] = float(current_param)
            dic_change[run_name] = float((current_param-baseline_param)*100/baseline_param)
        
    variation_dic['changing_quantity'] = parameter_to_change
    variation_dic['run_names'] = runs
    variation_dic['values'] = dic_values
    variation_dic['change'] = dic_change
    variation_dic['m'] = dic_m
    variation_dic['p'] = dic_p
    variation_dic['sol'] = dic_sol
    variation_dic['baseline'] = baseline_param
    
    dic_of_variation_dics[parameter_to_change] = variation_dic
    
#%% build a big table and write as excel

df = pd.DataFrame(columns = ['quantity','value','change to baseline']).set_index(['quantity','value','change to baseline'])

for variation_dic in dic_of_variation_dics.values():
    print(variation_dic['changing_quantity'])
    for run in variation_dic['run_names']:
        print(variation_dic['values'][run])
        # value = f'{float(f"{variation_dic["values"][run]:.1g}"):g}'
        value = '{:g}'.format(float('{:.3g}'.format(variation_dic["values"][run])))
        change = '{:g}'.format(float('{:.2g}'.format(variation_dic["change"][run])))
        for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
            if s_spec_par in variation_dic['p'][run].calib_parameters:
                df.loc[(variation_dic['changing_quantity'],value,change),s_spec_par] = getattr(variation_dic['p'][run],s_spec_par)[1]
        for scal_par in ['g_0','k',]:
            df.loc[(variation_dic['changing_quantity'],value,change),scal_par] = getattr(variation_dic['p'][run],scal_par)
        for c_spec_par in ['delta','eta','T']:
            for i,c in enumerate(variation_dic['p'][run].countries):
                df.loc[(variation_dic['changing_quantity'],value,change),c_spec_par+'_'+c] = getattr(variation_dic['p'][run],c_spec_par)[i,1]

dropbox_variation_table_path = dropbox_path+baseline+'_all_variations_table'

writer = pd.ExcelWriter(dropbox_variation_table_path+'.xlsx', engine='xlsxwriter')
workbook = writer.book
# worksheet = workbook.add_worksheet('all_quantities')
df.to_excel(writer,sheet_name='all_quantities',startrow=0 , startcol=0)   

writer.save()

#%% build a sensivity table

for percent_change in [-10,10]:

    df = pd.DataFrame()
    
    for variation_dic in dic_of_variation_dics.values():
        print(variation_dic['changing_quantity'])
        ten_percent_idx = np.abs([i-percent_change for i in variation_dic['change'].values()]).argmin()
        zero_percent_idx = np.abs([i for i in variation_dic['change'].values()]).argmin()
        run = variation_dic['run_names'][ten_percent_idx]
        run_baseline = variation_dic['run_names'][zero_percent_idx]
        for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
            if s_spec_par in variation_dic['p'][run].calib_parameters:
                df.loc[variation_dic['changing_quantity'],s_spec_par] = \
                    (getattr(variation_dic['p'][run],s_spec_par)[1] - getattr(p_baseline,s_spec_par)[1])*100/getattr(p_baseline,s_spec_par)[1]
        for scal_par in ['g_0','k',]:
            df.loc[variation_dic['changing_quantity'],scal_par] = \
                (getattr(variation_dic['p'][run],scal_par) - getattr(p_baseline,scal_par))*100/getattr(p_baseline,scal_par)
        for c_spec_par in ['delta','eta','T']:
            for i,c in enumerate(variation_dic['p'][run].countries):
                df.loc[variation_dic['changing_quantity'],c_spec_par+'_'+c] = \
                    (getattr(variation_dic['p'][run],c_spec_par)[i,1] - getattr(p_baseline,c_spec_par)[i,1])*100/getattr(p_baseline,c_spec_par)[i,1]
                 
    df = df.T
    df = df.round(2)
    
    dropbox_sensitivity_table = sensitivity_path+baseline+'_senstivity_'+str(percent_change)+'percent_change'
    
    writer = pd.ExcelWriter(dropbox_sensitivity_table+'.xlsx', engine='xlsxwriter')
    workbook = writer.book
    # worksheet = workbook.add_worksheet('all_quantities')
    df.to_excel(writer,sheet_name='all_quantities',startrow=0 , startcol=0)   
    
    writer.save()

#%% sensitivity graphs

undisplayed_list = ['zeta']

fig, ax = plt.subplots(figsize = (12,8))

for variation_dic in dic_of_variation_dics.values():
    if variation_dic['changing_quantity'] not in undisplayed_list:
        ax.plot([change for change in variation_dic['change'].values()],[p.delta[0,1]/p.nu[1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
        
ax.set_xlabel('Change in respective moment or parameter')    
ax.set_ylabel('Delta_US / nu',fontsize = 20)    
plt.title('Delta_US / nu',fontsize = 20) 
ax.legend()
plt.savefig(sensitivity_path+baseline+'_patent_protection_to_diffusion_ratio')
plt.show()    

for s_spec_par in ['theta','sigma','fe','zeta','nu','nu_tilde']:
    if s_spec_par in p_baseline.calib_parameters:
        fig, ax = plt.subplots(figsize = (12,8))
        
        for variation_dic in dic_of_variation_dics.values():
            if variation_dic['changing_quantity'] not in undisplayed_list:
                ax.plot([change for change in variation_dic['change'].values()],[getattr(p,s_spec_par)[1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                
        ax.set_xlabel('Change in respective moment or parameter')    
        ax.set_ylabel(s_spec_par,fontsize = 20)
        plt.title(s_spec_par,fontsize = 20)
            
        ax.legend()
        plt.savefig(sensitivity_path+baseline+'_'+s_spec_par)
        plt.show()   
        
for scal_par in ['g_0','k',]:
    if scal_par in p_baseline.calib_parameters:
        fig, ax = plt.subplots(figsize = (12,8))
        
        for variation_dic in dic_of_variation_dics.values():
            if variation_dic['changing_quantity'] not in undisplayed_list:
                ax.plot([change for change in variation_dic['change'].values()],[getattr(p,scal_par) for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                
        ax.set_xlabel('Change in respective moment or parameter')    
        ax.set_ylabel(scal_par,fontsize = 20)
        plt.title(scal_par,fontsize = 20)
            
        ax.legend()
        plt.savefig(sensitivity_path+baseline+'_'+scal_par)
        plt.show()    
        
for c_spec_par in ['delta']:
    for i,c in enumerate(['US']):
        if scal_par in p_baseline.calib_parameters:
            fig, ax = plt.subplots(figsize = (12,8))
            
            for variation_dic in dic_of_variation_dics.values():
                if variation_dic['changing_quantity'] not in undisplayed_list:
                    ax.plot([change for change in variation_dic['change'].values()],[getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()],  label = variation_dic['changing_quantity'])
                    
            ax.set_xlabel('Change in respective moment or parameter')    
            ax.set_ylabel(c_spec_par+'_'+c,fontsize = 20)
            plt.title(c_spec_par+'_'+c,fontsize = 20)
                
            ax.legend()
            plt.savefig(sensitivity_path+baseline+'_'+c_spec_par+'_'+c)
            plt.show()
            
#%% write the tables to be used by bokeh

make_dirs([sensitivity_tables_path])

df_dic = {}

for s_spec_par in ['theta','fe','fo','zeta','nu']:
    list_of_dfs = []
    for qty,variation_dic in dic_of_variation_dics.items():
        df = pd.DataFrame()
        df['Change'] = [round(change) for change in variation_dic['change'].values()]
        df[qty] = [getattr(p,s_spec_par)[1] for p in variation_dic['p'].values()]
        list_of_dfs.append(df)
    big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
    df_dic[s_spec_par] = big_df
        
for scal_par in ['g_0','k','kappa']:
    list_of_dfs = []
    for qty,variation_dic in dic_of_variation_dics.items():
        df = pd.DataFrame()
        df['Change'] = [round(change) for change in variation_dic['change'].values()]
        df[qty] = [getattr(p,scal_par) for p in variation_dic['p'].values()]
        list_of_dfs.append(df)
    big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
    df_dic[scal_par] = big_df
    
        
for c_spec_par in ['delta']:
    for i,c in enumerate(['US','EUR']):
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic['delta '+c] = big_df
        
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1]/p.nu[1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic[c_spec_par+' '+c+' '+'over nu'] = big_df
              
for c_spec_par in ['eta']:
    for i,c in enumerate(['US']):
        list_of_dfs = []
        for qty,variation_dic in dic_of_variation_dics.items():
            df = pd.DataFrame()
            df['Change'] = [round(change) for change in variation_dic['change'].values()]
            df[qty] = [getattr(p,c_spec_par)[0,1] for p in variation_dic['p'].values()]
            list_of_dfs.append(df)
        big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
        df_dic['eta_US'] = big_df

list_of_dfs = []
for qty,variation_dic in dic_of_variation_dics.items():
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [getattr(m,'RD_US') for m in variation_dic['m'].values()]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs)
df_dic['RD_US'] = big_df
        
list_of_dfs = []
for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [compute_deriv_welfare_to_patent_protec_US(variation_dic['sol'][r],variation_dic['p'][r],v0=None) for r in variation_dic['p'].keys()]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
df_dic['d_W_US_d_delta_US'] = big_df 

list_of_dfs = []
for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
    df = pd.DataFrame()
    df['Change'] = [round(change) for change in variation_dic['change'].values()]
    df[qty] = [compute_deriv_growth_to_patent_protec_US(variation_dic['sol'][r],variation_dic['p'][r],v0=None) for r in variation_dic['p'].keys()]
    list_of_dfs.append(df)
big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
df_dic['d_g_d_delta_US'] = big_df 

# list_of_dfs = []
# for qty,variation_dic in tqdm(dic_of_variation_dics.items()):
#     df = pd.DataFrame()
#     df['Change'] = [round(change) for change in variation_dic['change'].values()]
#     df[qty] = [m.ERDUS for m in variation_dic['m'].values()]
#     list_of_dfs.append(df)
# big_df = reduce(lambda  left,right: pd.merge(left,right,on='Change',how='outer'), list_of_dfs) 
# df_dic['ERDUS'] = big_df 
        
for k,df in df_dic.items():
    df.to_csv(sensitivity_tables_path+k+'.csv')
    
#%%

# big_df.to_csv(sensitivity_tables_path+'d_W_US_d_delta_US'+'.csv')   
# big_df.to_csv(sensitivity_tables_path+'d_g_d_delta_US'+'.csv')   
# big_df.to_csv(sensitivity_tables_path+'RD_US'+'.csv')   
        