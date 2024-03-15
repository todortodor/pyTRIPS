#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:39:35 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_coop_eq, find_nash_eq, dyn_fixed_point_solver
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
from tqdm import tqdm
from IPython.display import display, HTML,Markdown, Latex
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 10),
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

baseline = '1200'
variation = 'baseline'

pre_trips_variation = '9.2'

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

l_sol = []
l_p = []
tariffs = np.linspace(-0.1,0.1,21)
for i,tariff in tqdm(enumerate(tariffs)):
    p = p_baseline.copy()
    p.tariff[...,1] = (p_baseline.tariff[...,1]+tariff)
    np.einsum('iis->is',p.tariff)[...] = 0
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='profit',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                # apply_bound_psi_star = False,
                                damping = 10,
                                max_count = 1e4,
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
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if sol.status != 'successful':
        print('failed',tariff)
        break
    p.guess = sol_c.vector_from_var()
    l_sol.append(sol_c)
    l_p.append(p)
    
#%%

derivatives = pd.DataFrame(columns = ['tariff']+p_baseline.countries)
for i,sol_c in tqdm(enumerate(l_sol)):
    p = l_p[i]
    derivatives.loc[i,'tariff'] = tariffs[i]
    for j,country in enumerate(p_baseline.countries):
        p_cf = p.copy()
        p_cf.delta[j,1] = p_cf.delta[j,1]*1.1
        sol, sol_c = fixed_point_solver(p_cf,x0=p_cf.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='profit',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    # apply_bound_psi_star = False,
                                    damping = 10,
                                    max_count = 1e4,
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
        sol_c.compute_non_solver_quantities(p_cf)
        sol_c.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
        sol_c.compute_world_welfare_changes(p_cf,sol_baseline)
        
        derivatives.loc[i,country] = (sol_c.cons_eq_welfare[j]-l_sol[i].cons_eq_welfare[j])/(0.1*p.delta[j,1])

derivatives['tariff'] = derivatives['tariff']*100

#%%

derivatives[['tariff']+p_baseline.countries[:-1]].plot(x='tariff',y=p_baseline.countries[:-1],
                                                       figsize=(12,8))
plt.ylabel(r'$\frac{\partial CEW_n}{\partial \delta_n}$')
plt.xlabel('Change in all tariffs of the patenting sector (pp)')

plt.show()

#%%

l_sol = {}
l_p = {}
tariffs = np.linspace(-0.5,0.5,101)

derivatives = pd.DataFrame(columns = ['tariff']+p_baseline.countries)

for j,country in enumerate(p_baseline.countries):
    p = p_baseline.copy()
    l_p[country] = []
    l_sol[country] = []
    print(country)
    for i,tariff in tqdm(enumerate(tariffs)):
        p.tariff[j,:,1] = (p_baseline.tariff[j,:,1]+tariff)
        np.einsum('jjs->js',p.tariff)[...] = 0
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='profit',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    # apply_bound_psi_star = False,
                                    damping = 10,
                                    max_count = 1e4,
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
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        sol_c.compute_world_welfare_changes(p,sol_baseline)
        if sol.status != 'successful':
            print('failed',tariff)
            break
        p.guess = sol_c.vector_from_var()
        l_sol[country].append(sol_c)
        l_p[country].append(p)
        
        p_cf = p.copy()
        p_cf.delta[j,1] = p_cf.delta[j,1]*1.1
        sol, sol_d = fixed_point_solver(p_cf,x0=p_cf.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='profit',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    # apply_bound_psi_star = False,
                                    damping = 10,
                                    max_count = 1e4,
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
        sol_d.compute_non_solver_quantities(p_cf)
        sol_d.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
        sol_d.compute_world_welfare_changes(p_cf,sol_baseline)
        
        derivatives.loc[i,country] = (sol_d.cons_eq_welfare[j]-sol_c.cons_eq_welfare[j])/(0.1*p.delta[j,1])
        derivatives.loc[i,'tariff'] = tariff

derivatives['tariff'] = derivatives['tariff']*100


#%%

derivatives[['tariff']+p_baseline.countries[:-1]].plot(x='tariff',y=p_baseline.countries[:-1],
                                                       figsize=(12,8))
plt.ylabel(r'$\frac{\partial CEW_n}{\partial \delta_n}$')
plt.xlabel(r'Change in import tariffs of the patenting sector of country $n$ (pp)')

plt.show()