#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:03:15 2024

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

plt.style.use(['science','nature','no-latex'])
plt.style.use(['science','no-latex'])
import matplotlib.pyplot as plt 
plt.rcParams.update({"axes.grid" : True, 
                     "grid.color": "grey", 
                     'axes.axisbelow':True,
                     "grid.linewidth": 0.1, 
                     'legend.framealpha':1,
                     'legend.frameon':1,
                     'legend.edgecolor':'white',
                     'figure.dpi':288,
                     })

#%% setup path and stuff

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
table_path = 'calibration_results_matched_economy/'

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

baseline = '1050'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_variation = '9.2'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
# p_baseline.load_data(run_path)
p_baseline.load_run(run_path)

m_baseline = moments()
# m_baseline.load_data()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

#%% pre-TRIPS calibration and counterfactual

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
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

#%%

df = pd.DataFrame(columns = p_baseline.countries)
x = np.linspace(0,0.25,101)

for i,country in enumerate(p_baseline.countries):
    p_pre_cf = p_baseline.copy()
    p_pre_cf.delta = p_pre.delta.copy()
    p_pre_cf.tariff[:,:,1] = p_pre.tariff[:,:,1]
    for country_idx in [0,1,2,6,7,10]:
        p_pre_cf.delta[country_idx,1] = p_baseline.delta[country_idx,1]
    for tariff_reduction in x:
        p_pre_cf.tariff[:,i,1] = p_pre.tariff[:,i,1]-tariff_reduction
        p_pre_cf.tariff[i,i,1] = 0
        _, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 5,
                                max_count = 3e3,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
        sol_pre_cf.scale_P(p_pre_cf)
        sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
        sol_pre_cf.compute_consumption_equivalent_welfare(p_pre_cf,sol_baseline)
        sol_pre_cf.compute_world_welfare_changes(p_pre_cf,sol_baseline)
        
        p_pre_cf.guess = sol_pre_cf.vector_from_var()
        
        df.loc[tariff_reduction,country] = sol_pre_cf.cons_eq_welfare[i]
        print(df)

df = df.astype(float)

#%%

df_max = pd.DataFrame((df-1).abs().idxmin()*100)
df_max.columns = ['additional tariff reduction needed']