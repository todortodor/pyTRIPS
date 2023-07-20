#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:04:46 2023

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
from bokeh.palettes import Category10, Dark2
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

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

#%% Choose a run, load parameters, moments, solution

baseline = '1011'
variation = 'baseline'

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


#%%
years = [y for y in range(1990,2019)]
df = pd.DataFrame()

# for i,year in enumerate(years):
#     print(year)
year = 1992
p_year = p_baseline.copy()
# p_year.load_run(f'calibration_results_matched_economy/baseline_{baseline}_variations/1.{i}/')
baseline = 1010
p_year.load_run(f'calibration_results_matched_economy/baseline_{baseline}_variations/9.2/')

p_year_cf_trade_cost = p_baseline.copy()
p_year_cf_trade_cost.tau = p_year.tau.copy()

sol, sol_year_cf_trade_cost = fixed_point_solver(p_year_cf_trade_cost,x0=p_year_cf_trade_cost.guess,
                        context = 'counterfactual',
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
sol_year_cf_trade_cost.scale_P(p_year_cf_trade_cost)
sol_year_cf_trade_cost.compute_non_solver_quantities(p_year_cf_trade_cost)
sol_year_cf_trade_cost.compute_consumption_equivalent_welfare(p_year_cf_trade_cost,sol_baseline)

sol, dyn_sol_year_cf_trade_cost = dyn_fixed_point_solver(p_year_cf_trade_cost, sol_init=sol_baseline,Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
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
dyn_sol_year_cf_trade_cost.compute_non_solver_quantities(p_year_cf_trade_cost)
dyn_sol_year_cf_trade_cost.sol_fin.compute_consumption_equivalent_welfare(p_year_cf_trade_cost,sol_baseline)

p_year_cf_trade_cost_and_delta = p_baseline.copy()
p_year_cf_trade_cost_and_delta.tau = p_year.tau.copy()
p_year_cf_trade_cost_and_delta.delta[...,1] = p_year.delta[...,1]

sol, sol_year_cf_trade_cost_and_delta = fixed_point_solver(p_year_cf_trade_cost_and_delta,x0=p_year_cf_trade_cost_and_delta.guess,
                        context = 'counterfactual',
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
sol_year_cf_trade_cost_and_delta.scale_P(p_year_cf_trade_cost_and_delta)
sol_year_cf_trade_cost_and_delta.compute_non_solver_quantities(p_year_cf_trade_cost_and_delta)
sol_year_cf_trade_cost_and_delta.compute_consumption_equivalent_welfare(p_year_cf_trade_cost_and_delta,sol_baseline)

sol, dyn_sol_year_cf_trade_cost_and_delta = dyn_fixed_point_solver(p_year_cf_trade_cost_and_delta, sol_init=sol_baseline,Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
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
dyn_sol_year_cf_trade_cost_and_delta.compute_non_solver_quantities(p_year_cf_trade_cost_and_delta)
dyn_sol_year_cf_trade_cost_and_delta.sol_fin.compute_consumption_equivalent_welfare(p_year_cf_trade_cost_and_delta,sol_baseline)

for j,country in enumerate(p_baseline.countries):
    df.loc['welfare_tau_cf',country] = sol_year_cf_trade_cost.cons_eq_welfare[j]
    df.loc['dyn_welfare_tau_cf',country] = dyn_sol_year_cf_trade_cost.cons_eq_welfare[j]
    df.loc['welfare_tau_and_delta_cf',country] = sol_year_cf_trade_cost_and_delta.cons_eq_welfare[j]
    df.loc['dyn_welfare_tau_and_delta_cf',country] = dyn_sol_year_cf_trade_cost_and_delta.cons_eq_welfare[j]
    df.loc['median_tau_inward',country] = np.median(p_year.tau[j,:,:])
    df.loc['median_tau_outward',country] = np.median(p_year.tau[:,j,:])
    df.loc['delta',country] = p_year.delta[j,1]

df_bu = df.copy()

#%%

fig,ax = plt.subplots()


for k,country in enumerate(p_baseline.countries):
    ax.scatter(df.loc['welfare_tau_cf',country],df.loc['welfare_tau_and_delta_cf',country],
               label=country,color=Category18[k])

ax.set_xlabel('Welfare change from changing tau back to 1992')
ax.set_ylabel('Welfare change from changing tau and delta back to 1992')
x = [df.loc['welfare_tau_and_delta_cf',country] for country in p_baseline.countries]
ax.plot(np.sort(x),np.sort(x),
        ls = '--',color='grey')
plt.legend()
plt.show()

#%%

fig,ax = plt.subplots()


for k,country in enumerate(p_baseline.countries):
    ax.scatter(df.loc['dyn_welfare_tau_cf',country],df.loc['dyn_welfare_tau_and_delta_cf',country],
               label=country,color=Category18[k])

ax.set_xlabel('Welfare change from changing tau back to 1992')
ax.set_ylabel('Welfare change from changing tau and delta back to 1992')
x = [df.loc['dyn_welfare_tau_and_delta_cf',country] for country in p_baseline.countries]
ax.plot(np.sort(x),np.sort(x),
        ls = '--',color='grey')
plt.legend()
plt.show()
