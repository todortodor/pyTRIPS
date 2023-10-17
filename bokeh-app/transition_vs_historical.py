#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:06:42 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
import scienceplots
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

# plt.style.use(['science','nature','no-latex'])
# plt.style.use(['science','no-latex'])

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

baseline = '1030'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'
partial_variation = '9.0'
variation_with_doubled_tau_in_pat_sect = '10.2'
variation_with_doubled_nu = '2.0'

baseline_pre_trips_full_variation = baseline
pre_trips_full_variation = '3.1'

variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low TO',
    '99.1':'High TO',
    '99.2':'Low TE',
    '99.3':'High TE',
    '99.4':'Low KM',
    '99.5':'High KM',
    '99.6':'Low Sigma',
    '99.7':'High Sigma',
    '99.8':'Low Kappa',
    '99.9':'High Kappa',
    '99.10':'Low Growth',
    '99.11':'High Growth',
    '99.12':'Low rho',
    '99.13':'High rho',
    '99.14':'Low UUPCOST',
    '99.15':'High UUPCOST',
    }

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'
table_path = 'calibration_results_matched_economy/'

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

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

years = [y for y in range(1990,2019)]
nb_countries = p_baseline.N

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
deltas = p_pre.delta.copy()
p_pre = p_baseline.copy()
p_pre.delta = deltas

_, sol_pre = fixed_point_solver(p_pre,context = 'counterfactual',x0=p_pre.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
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
                        )
sol_pre.scale_P(p_pre)
sol_pre.compute_non_solver_quantities(p_pre)

_, dyn_sol_fwd = dyn_fixed_point_solver(p_baseline, sol_pre,sol_fin=sol_baseline,
                        Nt=25,t_inf=500,
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
                        damping_post_acceleration=5
                        )
dyn_sol_fwd.compute_non_solver_quantities(p_baseline)

#%%

def fit_and_eval(vec,dyn_sol,time,time_truncated,
                 # normalization_start,normalization_end,
                 normalize_start=False,normalize_end=False):
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                vec,
                dyn_sol.Nt),time)
    res = fit
    # if normalize_start:
    #     res = fit/normalization_start
    # if normalize_start and normalize_end:
    #     # res = (fit-normalization_start)/np.abs(normalization_end-normalization_start)
    #     res = np.sign(normalization_end-normalization_start)*(fit-normalization_start)/np.abs(normalization_end-normalization_start)
    return res[:time_truncated.shape[0]]


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (14, 11),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

data_path = 'data_smooth_3_years/'

pflows = pd.concat(
    [pd.read_csv(data_path+f'data_{nb_countries}_countries_{y}/country_country_moments.csv',
                 index_col=[0,1])
     for y in years],
    axis=0,
    keys=years,
    names=['year','origin_code','destination_code'],
    # ignore_index=True
    )

c_map = {i+1:p_baseline.countries[i] for i in range(nb_countries)}

ori = pflows.query('origin_code!=destination_code').groupby(['destination_code','year']).sum().reset_index()
ori['destination_code'] = ori['destination_code'].map(c_map)
# ori['destination_code'] = ori['destination_code'].map(countries_names)
ori = ori.pivot(
    columns = 'destination_code',
    index = 'year',
    values = 'patent flows'
    )

time = np.linspace(0,dyn_sol_fwd.t_inf,10001)
time_truncated = time[:461]

fig,ax = plt.subplots(figsize=(14,14))
# ax2 = ax.twinx()
for i,country in enumerate(p_baseline.countries[:-1]):

    # ax.plot(ori.index,(ori[country]-ori[country].loc[1992])/(ori[country].loc[2015]-ori[country].loc[1992]),
    ax.plot(ori.index,ori[country],
            color=Category18[i],
            ls='--'
            )
    
    normalization_start = dyn_sol_fwd.sol_init.pflow[i,:].sum()
    normalization_end = sol_baseline.pflow[i,:].sum()
    # ax2.plot(time_truncated+1992,fit_and_eval(dyn_sol_fwd.pflow[i,...].sum(axis=0),
    #                                           dyn_sol_fwd,time,time_truncated,
    #                   normalization_start,normalization_end,
    #                   normalize_start=False,normalize_end=False))
    qty = fit_and_eval(dyn_sol_fwd.pflow[i,...].sum(axis=0),
                                              dyn_sol_fwd,time,time_truncated,
                      # normalization_start,normalization_end,
                     normalize_start=False,normalize_end=False)
    # ax.plot(time_truncated+1992,(qty-qty[0])/(qty[-1]-qty[0]))
    ax.plot(time_truncated+1992,((ori[country].loc[2015]-ori[country].loc[1992])*qty+ori[country].loc[1992]*qty[-1]-ori[country].loc[2015]*qty[0])
            /(qty[-1]-qty[0]),label=countries_names[country],color=Category18[i])
    
    plt.yscale('log')
ax.plot([],[],ls='--',color='grey',label='Historical smoothed out 3 years')
ax.plot([],[],color='grey',label='Simulated transitional dynamics')
ax.legend(loc=[1.02,0.02])
ax.set_ylabel('International patent families by destination')
ax.set_xlim([1992,2015])
ax.set_ylim([1e3,2e5])

plt.show()

#%%

def fit_and_eval(vec,dyn_sol,time,time_truncated,
                  normalization_start,normalization_end,
                 normalize_start=False,normalize_end=False):
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                vec,
                dyn_sol.Nt),time)
    res = fit
    if normalize_start:
        res = fit*normalization_start/fit[0]
    if normalize_end:
        res = fit*normalization_end/fit[-1]
    if normalize_start and normalize_end:
        # res = (fit-normalization_start)/np.abs(normalization_end-normalization_start)
        res = np.sign(normalization_end-normalization_start)*(fit-normalization_start)/np.abs(normalization_end-normalization_start)
    return res[:time_truncated.shape[0]]

def add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,normalize_start,
               normalize_end,label=None,color=sns.color_palette()[0],
               return_data = False):
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
        ax.scatter(x=[0,60],
                    y=[norm_start,norm_end],
                    color=color)
    if normalize_start and not normalize_end:
        ax.scatter(x=[0,60],
                    # y=[1,norm_end/np.abs(norm_start)],
                    y=[1,np.abs(norm_end)/np.abs(norm_start)],
                    color=color)
    if normalize_start and normalize_end:
        ax.scatter(x=[0,60],
                    # y=[0,np.sign(norm_end-norm_start)*norm_end/np.abs(norm_end)],
                    y=[0,1],
                    color='k')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (14, 11),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

data_path = 'data_smooth_3_years/'

pflows = pd.concat(
    [pd.read_csv(data_path+f'data_{nb_countries}_countries_{y}/country_country_moments.csv',
                 index_col=[0,1])
     for y in years],
    axis=0,
    keys=years,
    names=['year','origin_code','destination_code'],
    # ignore_index=True
    )

c_map = {i+1:p_baseline.countries[i] for i in range(nb_countries)}

ori = pflows.query('origin_code!=destination_code').groupby(['destination_code','year']).sum().reset_index()
ori['destination_code'] = ori['destination_code'].map(c_map)
# ori['destination_code'] = ori['destination_code'].map(countries_names)
ori = ori.pivot(
    columns = 'destination_code',
    index = 'year',
    values = 'patent flows'
    )

time = np.linspace(0,dyn_sol_fwd.t_inf,10001)
time_truncated = time[:461]

fig,ax = plt.subplots(figsize=(14,14))
# ax2 = ax.twinx()
for i,country in enumerate(p_baseline.countries[:-1]):

    # ax.plot(ori.index,(ori[country]-ori[country].loc[1992])/(ori[country].loc[2015]-ori[country].loc[1992]),
    # ax.plot(ori.index,ori[country],
    #         color=Category18[i],
    #         ls='--'
    #         )
    
    # norm_start = ori[country].loc[1992]
    # norm_end = ori[country].loc[2015]
    norm_start = dyn_sol_fwd.sol_init.pflow[i,...].sum(axis=0)
    norm_end = dyn_sol_fwd.sol_fin.pflow[i,...].sum(axis=0)
    # ax2.plot(time_truncated+1992,fit_and_eval(dyn_sol_fwd.pflow[i,...].sum(axis=0),
    #                                           dyn_sol_fwd,time,time_truncated,
    #                   normalization_start,normalization_end,
    #                   normalize_start=False,normalize_end=False))
    # qty = fit_and_eval(dyn_sol_fwd.pflow[i,...].sum(axis=0),
    #                    dyn_sol_fwd,time,time_truncated,
    #                    normalization_start,normalization_end,
    #                  # normalize_start=True,normalize_end=False)
    #                  normalize_start=True,normalize_end=True)
    test = add_graph(dyn_sol_fwd,dyn_sol_fwd.pflow[i,...].sum(axis=0),norm_start,norm_end,
                   ax,time,time_truncated,normalize_start=True,
                   normalize_end=True,label=None,color=sns.color_palette()[0],
                   return_data = True)
    # ax.plot(time_truncated+1992,(qty-qty[0])/(qty[-1]-qty[0]))
    # ax.plot(time_truncated+1992,qty,label=countries_names[country],color=Category18[i])
    
    plt.yscale('log')
ax.plot([],[],ls='--',color='grey',label='Historical smoothed out 3 years')
ax.plot([],[],color='grey',label='Simulated transitional dynamics')
ax.legend(loc=[1.02,0.02])
ax.set_ylabel('International patent families by destination')
# ax.set_xlim([1992,2015])
# ax.set_ylim([1e3,2e5])

plt.show()

