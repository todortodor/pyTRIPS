#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:13:11 2024

@author: slepot
"""

import pandas as pd
from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver, dyn_fixed_point_solver
from data_funcs import write_calibration_results
import os
import numpy as np


baseline_number = '1060'

p_b = parameters()
p_b.load_run('calibration_results_matched_economy/1060/')
sol_b = var.var_from_vector(p_b.guess, p_b, compute=True, context = 'counterfactual')
sol_b.scale_P(p_b)
sol_b.compute_non_solver_quantities(p_b)

p_baseline = p_b.copy()
p_baseline.tau[...] = 1
p_baseline.tariff[...] = 0
# p_baseline = parameters()
# p_baseline.load_run('calibration_results_matched_economy/baseline_1060_variations/10.3/')
p_baseline.delta[...,1] = np.array([0.01      , 0.01      , 0.35296464, 0.01      , 0.07932502,
       0.13668579, 0.01      , 0.01      , 0.08936727, 0.1661441 ,
       1.51579331])

sol, sol_baseline = fixed_point_solver(p_baseline,context = 'counterfactual',x0=p_baseline.guess,
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
                        damping_post_acceleration=2
                        )
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)
sol_baseline.compute_consumption_equivalent_welfare(p_baseline,sol_b)

m_baseline = moments()
m_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares.csv',index_col=0)

deltas = deltas.loc[(deltas.baseline == 1060) & (deltas.variation.isin(['baseline','10.3']))]
welfares = welfares.loc[(welfares.baseline == 1060) & (welfares.variation.isin(['baseline','10.3']))]

deltas = deltas[['variation', 'aggregation_method']+p_baseline.countries]
welfares = welfares[['variation', 'aggregation_method']+p_baseline.countries+['Equal', 'Negishi']]

deltas['variation'] = deltas['variation'].str.replace('10.3','No trade costs')
welfares['variation'] = welfares['variation'].str.replace('10.3','No trade costs')

deltas = deltas.set_index(['variation', 'aggregation_method']).T
deltas[deltas>5] = 12
welfares = welfares.set_index(['variation', 'aggregation_method']).T



#%%

p_baseline_temp = p_baseline.copy()
sol_baseline_temp = sol_baseline.copy()

p_opt = p_baseline_temp.copy()
l_w = []
for i,delta in enumerate(np.logspace(-2,0,21)):
    print(i)
    p_opt.delta[3,1] = delta
    sol, sol_c = fixed_point_solver(p_opt,context = 'counterfactual',x0=p_opt.guess,
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
                            damping_post_acceleration=2
                            )
    p_opt.guess = sol_c.vector_from_var()
    sol_c.scale_P(p_opt)
    sol_c.compute_non_solver_quantities(p_opt) 
    sol_c.compute_world_welfare_changes(p_opt,sol_baseline_temp)
    sol_c.compute_consumption_equivalent_welfare(p_opt,sol_baseline_temp)
    l_w.append(sol_c.cons_eq_welfare.tolist()
        +[sol_c.cons_eq_pop_average_welfare_change]
        +[sol_c.cons_eq_negishi_welfare_change])

#%%

import matplotlib.pyplot as plt

plt.plot(np.logspace(-2,0,21),l_w)
plt.xlabel('delta China')
plt.ylabel('Welfare')
# plt.ylim(0.98,1.02)
plt.xscale('log')
plt.legend(p_baseline.countries+['Equal', 'Negishi'],loc=(1.05,0))
plt.show()

plt.plot(np.logspace(-2,0,21),[l[-1] for l in l_w])
plt.xlabel('delta China')
plt.ylabel('Welfare Negishi')
plt.xscale('log')
plt.legend(['Negishi'],loc=(1.05,0))
plt.show()
