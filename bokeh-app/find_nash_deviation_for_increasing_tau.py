#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:07:07 2023

@author: slepot
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var, history_nash
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, minus_welfare_of_delta
from scipy import optimize

baseline = '1020'
variation = 'baseline'
results_path = 'calibration_results_matched_economy/'

recap_path = 'counterfactual_recaps/optimal_delta_for_increasing_tau_in_pat_sector/'

try:
    os.mkdir(recap_path)
except:
    pass

if variation == 'baseline':
    run_path = results_path+baseline+'/'
    recap_path = recap_path+'baseline_'+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'
    recap_path = recap_path+f'baseline_{baseline}_variations/{variation}/'

try:
    os.mkdir(recap_path)
except:
    pass

p_baseline = parameters()
p_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

lb_delta = 0.01
ub_delta = 12

country = 'USA'

#%%

dynamics = True
lb = 1
ub = 5
it = -1

df = pd.DataFrame()

while ub-lb>1e-5:
    it = it+1
    x = (ub+lb)/2
    p = p_baseline.copy()
    p.tau[...,1] = p_baseline.tau[...,1]*x
    for j,_ in enumerate(p_baseline.countries):
        p.tau[j,j,:] = 1
    # hist_nash = history_nash()
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
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
                            ) 
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    p.delta[...,1] = 12
    if not dynamics:
        bounds=(lb_delta, ub_delta)
        delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
                                              method='bounded',
                                              bounds=bounds,
                                              args = (p,country,sol_c, None, dynamics),
                                                options={'disp':3},
                                              tol=1e-15
                                              )
        df.loc[it,'tau_factor_in_patenting_sector'] = x
        df.loc[it,'optimal_delta_'+country] = delta_min.x
        if delta_min.x<11.9:
            ub = x
        else:
            lb = x
        print(delta_min.x,lb,ub)
    if dynamics:
        # welf_of_min = minus_welfare_of_delta(lb_delta,p,country,sol_baseline, None, dynamics)
        # welf_of_max = minus_welfare_of_delta(ub_delta,p,country,sol_baseline, None, dynamics)
        welf_of_min = minus_welfare_of_delta(lb_delta,p,country,sol_c, None, dynamics)
        welf_of_max = minus_welfare_of_delta(ub_delta,p,country,sol_c, None, dynamics)
        if welf_of_min<welf_of_max:
            ub = x
            df.loc[it,'tau_factor_in_patenting_sector'] = x
            df.loc[it,'optimal_delta_'+country] = lb_delta
        else:
            lb = x
            df.loc[it,'tau_factor_in_patenting_sector'] = x
            df.loc[it,'optimal_delta_'+country] = ub_delta
        print(x,lb,ub)
    plt.scatter(df['tau_factor_in_patenting_sector'],df['optimal_delta_'+country])
    plt.scatter(df['tau_factor_in_patenting_sector'].iloc[-1],df['optimal_delta_'+country].iloc[-1],
                color='red')
    plt.show()
    

# it = df.loc[df['optimal_delta'] != 12].idxmax()

# p.delta[...,1] = 12
p = p_baseline.copy()
p.tau[...,1] = p_baseline.tau[...,1]*x
for j,_ in enumerate(p_baseline.countries):
    p.tau[j,j,:] = 1
sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        # plot_live=True,
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
                        ) 
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
p.delta[...,1] = 12
for country_to_validate in p_baseline.countries:
    if country_to_validate != country:
        # delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
        #                                       method='bounded',
        #                                       bounds=bounds,
        #                                       args = (p,country_to_validate,sol_c, None, dynamics),
        #                                         options={'disp':3},
        #                                       tol=1e-15
        #                                       )
        # df.loc[country,'tau_factor_in_patenting_sector'] = x
        df.loc[it,'optimal_delta_'+country_to_validate] = 12
    
# if dynamics:
#     df.to_csv(recap_path+f'dyn_{country}.csv')
# else:
#     df.to_csv(recap_path+f'{country}.csv')
    
#%%
import numpy as np

dynamics = False

test = {}
deltas = np.logspace(np.log10(0.01),np.log10(12),31)

for x in np.linspace(1,3,5):
    test[x] = {}
    test[x]['delta'] = []
    test[x]['welf'] = []
    # hist_nash = history_nash()
    p = p_baseline.copy()
    p.tau[...,1] = p_baseline.tau[...,1]*x
    for j,_ in enumerate(p_baseline.countries):
        p.tau[j,j,:] = 1
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
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
                            ) 
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    p.delta[...,1] = 12
    # bounds=(lb_delta, ub_delta)
    for delta in deltas:
        print(x,delta)
        test[x]['delta'].append(delta)
        test[x]['welf'].append(-minus_welfare_of_delta(delta,p,country,sol_c, None, dynamics))
        
#%%
from mycolorpy import colorlist as mcp
cmap = mcp.gen_color('brg',len(test[x]['delta']))

fig,ax = plt.subplots(figsize = (12,8))
for i,x in enumerate(test):
    ax.plot(test[x]['delta'],test[x]['welf'],label='tau factor ='+str(round(x,1)),
            c = cmap[i])
# fig.colorbar(cmap)
ax.legend()
ax.set_xscale('log')
plt.show()

df = pd.DataFrame(index = test[x]['delta'])
for x in test:
    df[x] = test[x]['welf']
    
# if dynamics:
#     df.to_csv(f'../misc/dyn_counterfactuals_of_optimal_delta_{country}_for_diff_taus_in_pat_sector_when_other_count_go_to_full_protec.csv')
# else:
#     df.to_csv(f'../misc/counterfactuals_of_optimal_delta_{country}_for_diff_taus_in_pat_sector_when_other_count_go_to_full_protec.csv')
