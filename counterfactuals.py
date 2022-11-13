#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:38:05 2022

@author: simonl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver

baseline = '85'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters(n=7,s=2)
p_baseline.load_data(baseline_path)
m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(baseline_path)

save = True
dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/counterfactuals/unilaterat_patent_protection_'+baseline+'/'

try:
    os.mkdir(dropbox_path)
except:
    pass

#%% counterfactual


sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                        cobweb_anim=False,tol =1e-15,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
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

sol_baseline.scale_P(p_baseline)
sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

counterfactuals_by_country = {}

for c in p_baseline.countries:
    print(c)
    p = p_baseline.copy()
    sols_c = []
    deltas = np.logspace(-1,1,111)
    for delt in deltas:
        print(delt)
        p.delta[p.countries.index(c),1] = p_baseline.delta[p.countries.index(c),1] * delt
        # print(p.guess)
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
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
        
    
        sol_c = var.var_from_vector(sol.x, p)    
        # sol_c.scale_tau(p)
        sol_c.scale_P(p)
        sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p)
        # sol_c.compute_welfare(p)
        sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        if sol.status == 'success':
            p.guess = sol_c.vector_from_var()
        else:
            p.guess = None
        
        sols_c.append(sol_c)
    counterfactuals_by_country[c] = sols_c

#%% plot counterfactual

for c in p.countries:
    consumption = {}
    consumption_eq_welfare = {}
    for j,c2 in enumerate(p.countries):
        consumption[c2] = [(sol.nominal_final_consumption[j]/sol.price_indices[j])
                           /(sol_baseline.nominal_final_consumption[j]/sol_baseline.price_indices[j])
                           for sol in counterfactuals_by_country[c]]
        consumption_eq_welfare[c2] = [sol.cons_eq_welfare[j]
                           for sol in counterfactuals_by_country[c]] 
    growth_rate = [sol.g for sol in counterfactuals_by_country[c]]
    fig,ax = plt.subplots(2,1,figsize = (10,14),constrained_layout=True)
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax2.plot(deltas,growth_rate, color='k', ls = '--', label = 'Growth rate')
    # ax2.set_ylim(0.016,0.026)
    ax3.plot(deltas,growth_rate, color='k', ls = '--', label = 'Growth rate')
    # ax3.set_ylim(0.016,0.026)
    ax2.set_ylabel('Growth rate')
    ax3.legend(loc = (1,1.05))
    ax[0].set_ylabel('Normalized consumption')
    ax3.set_ylabel('Growth rate')
    ax[1].set_ylabel('Consumption equivalent welfare')
    for j,c2 in enumerate(p.countries):
        ax[0].plot(deltas,consumption[c2],label = c2,color = sns.color_palette()[j])
        ax[1].plot(deltas,consumption_eq_welfare[c2],label = c2,color = sns.color_palette()[j])
    ax[0].legend(loc = 'center right')
    ax[0].scatter([deltas[np.argmax(consumption[c3])] for c3 in p.countries],
                  [max(consumption[c3]) for c3 in p.countries],
                  color=sns.color_palette()[:len(p.countries)])
    ax[1].scatter([deltas[np.argmax(consumption_eq_welfare[c3])] for c3 in p.countries],
                  [max(consumption_eq_welfare[c3]) for c3 in p.countries],
                  color=sns.color_palette()[:len(p.countries)])
    # ax[1].legend()
    plt.title('Counterfactual - patent protection '+c,pad = 20)
    if save:
        plt.savefig(dropbox_path+c)
    plt.show()
