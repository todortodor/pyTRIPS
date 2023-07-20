#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:38:31 2023

@author: slepot
"""

from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from bokeh.palettes import Category10, Dark2
import seaborn as sns
import matplotlib.pylab as pylab
import os
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
save_path_no_norm = 'additional_material/1010_baseline_additional_material/time_series_no_norm/'
save_path_norm_start = 'additional_material/1010_baseline_additional_material/time_series_norm_start/'
save_path_norm_all = 'additional_material/1010_baseline_additional_material/time_series_norm_all/'

#%% make dirs to save

save_paths = [save_path_no_norm,save_path_norm_start,save_path_norm_all]

for save_path in save_paths:
    try:
        os.mkdir(save_path)
    except:
        pass


#%% Loading baseline and computing counterfactuals

p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/1010/')

sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

p = p_baseline.copy()
p.delta[0,1] = 0.05
# p.delta[0,1] = 1e-2


sol, dyn_sol = dyn_fixed_point_solver(p, sol_baseline, Nt=21,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
dyn_sol.compute_non_solver_quantities(p)

#%% Different normaizations

conditions_of_norm = [
    {
     'save_path':save_path_no_norm,
     'normalize_start':False,
     'normalize_end':False,
     },
    {
     'save_path':save_path_norm_start,
     'normalize_start':True,
     'normalize_end':False,
     },
    {
     'save_path':save_path_norm_all,
     'normalize_start':True,
     'normalize_end':True,
     },
    ]

#%% Functions for fit and plotting


time = np.linspace(0,dyn_sol.t_inf,10001)
time_truncated = time[:2001]

def fit_and_eval(vec,dyn_sol,time,time_truncated,
                 normalization_start,normalization_end,
                 normalize_start=False,normalize_end=False):
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                vec,
                dyn_sol.Nt),time)
    res = fit
    if normalize_start:
        res = fit/normalization_start
    if normalize_start and normalize_end:
        res = (fit-normalization_start)/np.abs(normalization_end-normalization_start)
    return res[:time_truncated.shape[0]]

def add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,normalize_start,
               normalize_end,label=None,color=sns.color_palette()[0]):
    ax.plot(time_truncated,fit_and_eval(qty,
                                        dyn_sol,
                                        time,time_truncated,
                      normalization_start = norm_start,
                      normalization_end = norm_end,
                      normalize_start=normalize_start,
                      normalize_end=normalize_end)
            ,label=label,
            color=color)
    if not normalize_start and not normalize_end:
        ax.scatter(x=[0,100],
                    y=[norm_start,norm_end],
                    color=color)
    if normalize_start and not normalize_end:
        ax.scatter(x=[0,100],
                    y=[1,norm_end/np.abs(norm_start)],
                    color=color)
    if normalize_start and normalize_end:
        ax.scatter(x=[0,100],
                    y=[0,np.sign(norm_end-norm_start)*norm_end/np.abs(norm_end)],
                    color='k')
        
    ax.set_xlabel('Time (years)')
    
save = True
    
#%% Growth rate

for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    qty = dyn_sol.g
    norm_start = dyn_sol.sol_init.g
    norm_end = dyn_sol.sol_fin.g
    name = 'growth_rate'
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Growth rate',
                   color=Category18[0])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Growth rate')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Real final consumption

name = 'real_consumption'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty = dyn_sol.nominal_final_consumption[i,:]/dyn_sol.price_indices[i,:]
        norm_start = dyn_sol.sol_init.nominal_final_consumption[i]/dyn_sol.sol_init.price_indices[i]
        norm_end = dyn_sol.sol_fin.nominal_final_consumption[i]/dyn_sol.sol_fin.price_indices[i]
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Real final consumption')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Real profit

name = 'real_profit'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty = (dyn_sol.profit[:,i,1,:]/dyn_sol.price_indices[i,:]).sum(axis=0)
        norm_start = (dyn_sol.sol_init.profit[:,i,1]*dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
                      ).sum()
        norm_end = (dyn_sol.sol_fin.profit[:,i,1]*dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
                    ).sum()
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Real profit')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Research Labor

name = 'research_labor'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty =dyn_sol.l_R[i,1,:]
        norm_start = dyn_sol.sol_init.l_R[i,1]
        norm_end = dyn_sol.sol_fin.l_R[i,1]
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Labor allocated to research')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Real wage

name = 'real_wage'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty = dyn_sol.w[i,:]/dyn_sol.price_indices[i,:]
        norm_start = dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
        norm_end = dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Real wage')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% PSI CD

name = 'psi_cd'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty = dyn_sol.PSI_CD[i,1,:]+dyn_sol.PSI_CD_0[i,1,None]
        norm_start = dyn_sol.sol_init.PSI_CD[i,1]
        norm_end = dyn_sol.sol_fin.PSI_CD[i,1]
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel(r'$\Psi^{CD}_n$')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Interest rate

name = 'interest_rate'
for cond_of_norm in conditions_of_norm:

    fig,ax = plt.subplots()
    for i,country in enumerate(p_baseline.countries):
        qty = dyn_sol.r[i,:]
        norm_start = dyn_sol.sol_init.r
        norm_end = dyn_sol.sol_fin.r
        add_graph(dyn_sol,qty,norm_start,norm_end,
                       ax,time,time_truncated,
                       normalize_start=cond_of_norm['normalize_start'],
                       normalize_end=cond_of_norm['normalize_end'],
                       label=country,
                       color=Category18[i])
    plt.axvline(x=20,ls = '--',color='grey')
    ax.set_ylabel('Interest rate')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
    
#%% Different US quantities

name = 'US_quantities'
for cond_of_norm in conditions_of_norm:
    
    fig,ax = plt.subplots()
    
    i = 0
    
    qty = dyn_sol.g
    norm_start = dyn_sol.sol_init.g
    norm_end = dyn_sol.sol_fin.g
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Growth rate',
                   color=Category18[0])
    
    qty = dyn_sol.nominal_final_consumption[i,:]/dyn_sol.price_indices[i,:]
    norm_start = dyn_sol.sol_init.nominal_final_consumption[i]/dyn_sol.sol_init.price_indices[i]
    norm_end = dyn_sol.sol_fin.nominal_final_consumption[i]/dyn_sol.sol_fin.price_indices[i]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Real final consumption',
                   color=Category18[1])
    
    qty = (dyn_sol.profit[:,i,1,:]/dyn_sol.price_indices[i,:]).sum(axis=0)
    norm_start = (dyn_sol.sol_init.profit[:,i,1]*dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
                  ).sum()
    norm_end = (dyn_sol.sol_fin.profit[:,i,1]*dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
                ).sum()
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Real profit',
                   color=Category18[2])
    
    qty =dyn_sol.l_R[i,1,:]
    norm_start = dyn_sol.sol_init.l_R[i,1]
    norm_end = dyn_sol.sol_fin.l_R[i,1]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Research labor',
                   color=Category18[3])
    
    qty = dyn_sol.w[i,:]/dyn_sol.price_indices[i,:]
    norm_start = dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
    norm_end = dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Real wage',
                   color=Category18[4])
    
    qty = dyn_sol.PSI_CD[i,1,:]+dyn_sol.PSI_CD_0[i,1,None]
    norm_start = dyn_sol.sol_init.PSI_CD[i,1]
    norm_end = dyn_sol.sol_fin.PSI_CD[i,1]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label=r'$\Psi^{CD}_n$',
                   color=Category18[5])

    qty = dyn_sol.r[i,:]
    norm_start = dyn_sol.sol_init.r
    norm_end = dyn_sol.sol_fin.r
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=cond_of_norm['normalize_start'],
                   normalize_end=cond_of_norm['normalize_end'],
                   label='Interest rate',
                   color=Category18[6])
    
    plt.axvline(x=20,ls = '--',color='grey')
    
    ax.set_ylabel('Time evolution of US quantities')
    plt.legend()
    if save:
        plt.savefig(cond_of_norm['save_path']+name)
    plt.show()
