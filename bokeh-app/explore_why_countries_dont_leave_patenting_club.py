#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:33:05 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq, find_nash_eq
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
# from bokeh.palettes import Category10, Dark2
import time
# Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

baseline = '1300'
variation = 'baseline'

baseline_pre_trips_full_variation = baseline

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

m_baseline = moments()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

welfare_protec = []
welfare_no_protec = []
sols_protec = []
dyn_sols_protec = []
dyn_welfare_protec = []
dyn_welfare_no_protec = []
sols_no_protec = []
dyn_sols_no_protec = []
etas = []
p = p_baseline.copy()
# p.guess = None

country = 'KOR'
idx_country = p_baseline.countries.index(country)

# tau_fact = np.linspace(1,10,3)
# p.tau[idx_country,:,:] = 10*p_baseline.tau[idx_country,:,:]
p.tau[idx_country,:,:] = 1*p_baseline.tau[idx_country,:,:]
p.tau[idx_country,idx_country,:] = 1

for i,eta in enumerate([0.0280461769031755]+np.logspace(-2,-4,2).tolist()):
    print(eta)
    p.eta[idx_country,1] = eta
    # p.tau[0,:,:] = tau_fact[i]*p_baseline.tau[0,:,:]
    # p.tau[0,0,:] = 1
    sol, sol_c = fixed_point_solver(p,
                            x0=p.guess,
                            context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e5,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    p.guess = sol_c.vector_from_var()
    
    p_protec = p.copy()
    p_protec.delta[:,1] = np.array([1.00e-02, 1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01,
           1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01, 1.20e+01])
    sol, sol_protec = fixed_point_solver(p_protec,
                                    x0=p_protec.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 20,
                            max_count = 1e5,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            )
    sol_protec.scale_P(p_protec)
    sol_protec.compute_non_solver_quantities(p_protec)
    sol_protec.compute_world_welfare_changes(p_protec,sol_c)
    sol_protec.compute_consumption_equivalent_welfare(p_protec,sol_c)
    
    welfare_protec.append(sol_protec.cons_eq_pop_average_welfare_change)
    sols_protec.append(sol_protec)
    
    sol, dyn_sol_protec = dyn_fixed_point_solver(p_protec, sol_init=sol_c,
                                             sol_fin=sol_protec,Nt=25,
                            t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=False,
                            damping = 60,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    dyn_sol_protec.compute_non_solver_quantities(p_protec)
    dyn_sol_protec.compute_consumption_equivalent_welfare(p_protec)
    
    dyn_sols_protec.append(dyn_sol_protec)
    dyn_welfare_protec.append(dyn_sol_protec.cons_eq_pop_average_welfare_change)
    
    p_no_protec = p.copy()
    p_no_protec.delta[:,1] = np.array([0.01, 1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01,
           1.00e-02, 1.00e-02, 1.20e+01, 1.20e+01, 1.20e+01, 1.20e+01])
    p_no_protec.delta[idx_country,1] = 12
    sol, sol_no_protec = fixed_point_solver(p_no_protec,
                                    x0=p_no_protec.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 20,
                            max_count = 1e5,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            )
    sol_no_protec.scale_P(p_no_protec)
    sol_no_protec.compute_non_solver_quantities(p_no_protec)
    sol_no_protec.compute_consumption_equivalent_welfare(p_no_protec,sol_c)
    sol_no_protec.compute_world_welfare_changes(p_no_protec,sol_c)
    
    sols_no_protec.append(sol_no_protec)
    welfare_no_protec.append(sol_no_protec.cons_eq_pop_average_welfare_change)
    
    sol, dyn_sol_no_protec = dyn_fixed_point_solver(p_no_protec, sol_init=sol_c,
                                             sol_fin=sol_no_protec,Nt=25,
                            t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=False,
                            damping = 60,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    dyn_sol_no_protec.compute_non_solver_quantities(p_no_protec)
    dyn_sol_no_protec.compute_consumption_equivalent_welfare(p_no_protec)
    
    dyn_sols_no_protec.append(dyn_sol_no_protec)
    dyn_welfare_no_protec.append(dyn_sol_no_protec.cons_eq_pop_average_welfare_change)
    
    etas.append(eta)
    
    plt.plot(etas,welfare_protec,label=f'{country} full protection')
    plt.plot(etas,dyn_welfare_protec,label=f'{country} full protection with dynamics')
    plt.plot(etas,welfare_no_protec,label=f'{country} no protection')
    plt.plot(etas,dyn_welfare_no_protec,label=f'{country} no protection with dynamics')
    plt.xscale('log')
    plt.xlabel(f'Eta {country}')
    plt.ylabel('Equal welfare world')
    plt.legend()
    plt.show()

    #%%
    
    fig,ax = plt.subplots(2,1,figsize=[6,6])
    ax[0].plot(etas,[sol.cons_eq_welfare for sol in dyn_sols_protec])
    # ax[0].legend(p_baseline.countries)
    ax[0].set_xscale('log')
    ax[0].set_title(f'{country} full protection')
    ax[0].set_ylabel('Welfare change')
    
    ax[1].plot(etas,[sol.cons_eq_welfare for sol in dyn_sols_no_protec])
    ax[1].legend(p_baseline.countries,loc=[1.1,0.5])
    ax[1].set_title(f'{country} no protection')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(f'Eta {country}')
    ax[1].set_ylabel('Welfare change')
    plt.show()
    
    #%%
    
    fig,ax = plt.subplots(2,1,figsize=[6,6])
    ax[0].plot(etas,[sol.sol_fin.l_R[:,1]/dyn_sols_protec[0].sol_fin.l_R[:,1] for sol in dyn_sols_protec])
    # ax[0].legend(p_baseline.countries)
    ax[0].set_xscale('log')
    ax[0].set_title(f'{country} full protection')
    ax[0].set_ylabel('Research labor indexed \n by value at baseline eta')
    
    ax[1].plot(etas,[sol.sol_fin.l_R[:,1]/dyn_sols_no_protec[0].sol_fin.l_R[:,1] for sol in dyn_sols_no_protec])
    ax[1].legend(p_baseline.countries,loc=[1.1,0.5])
    ax[1].set_title(f'{country} no protection')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(f'Eta {country}')
    ax[1].set_ylabel('Research labor indexed \n by value at baseline eta')
    plt.show()
    
    #%%
    
    fig,ax = plt.subplots(2,1,figsize=[6,6])
    ax[0].plot(etas,[sol.sol_fin.cons/dyn_sols_protec[0].sol_fin.cons for sol in dyn_sols_protec])
    # ax[0].legend(p_baseline.countries)
    ax[0].set_xscale('log')
    ax[0].set_title(f'{country} full protection')
    ax[0].set_ylabel('Real consumption')
    
    ax[1].plot(etas,[sol.sol_fin.cons/dyn_sols_no_protec[0].sol_fin.cons for sol in dyn_sols_no_protec])
    ax[1].legend(p_baseline.countries,loc=[1.1,0.5])
    ax[1].set_title(f'{country} no protection')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(f'Eta {country}')
    ax[1].set_ylabel('Real consumption')
    plt.show()
    
    #%%
    
    fig,ax = plt.subplots(2,1,figsize=[6,6])
    ax[0].plot(etas,[sol.sol_fin.price_indices/dyn_sols_protec[0].sol_fin.price_indices for sol in dyn_sols_protec])
    # ax[0].legend(p_baseline.countries)
    ax[0].set_xscale('log')
    ax[0].set_title(f'{country} full protection')
    ax[0].set_ylabel('Price indices')
    
    ax[1].plot(etas,[sol.sol_fin.price_indices/dyn_sols_no_protec[0].sol_fin.price_indices for sol in dyn_sols_no_protec])
    ax[1].legend(p_baseline.countries,loc=[1.1,0.5])
    ax[1].set_title(f'{country} no protection')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Eta US')
    ax[1].set_ylabel('Price indices')
    plt.show()
    
    #%%
    
    fig,ax = plt.subplots(2,1,figsize=[6,6])
    ax[0].plot(etas,[sol.sol_fin.g for sol in dyn_sols_protec])
    # ax[0].legend(p_baseline.countries)
    ax[0].set_xscale('log')
    ax[0].set_title(f'{country} full protection')
    ax[0].set_ylabel('Growth rate')
    
    ax[1].plot(etas,[sol.sol_fin.g for sol in dyn_sols_no_protec])
    # ax[1].legend(p_baseline.countries,loc=[1.1,0.5])
    ax[1].set_title(f'{country} no protection')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(f'Eta {country}')
    ax[1].set_ylabel('Growth rate')
    plt.show()
