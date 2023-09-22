#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:04:02 2023

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

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

baseline = '1020'
variation = 'baseline'

baseline_pre_trips_variation = '1020'
pre_trips_cf = True
pre_trips_variation = '9.2'

baseline_pre_trips_full_variation = '1020'
pre_trips_full_variation = '3.1'

output_path = 'output/'
output_name = 'draft_NBER'

save_path = output_path+output_name+'/'+baseline+'_'+variation+'/'

try:
    os.mkdir(save_path)
except:
    pass

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

save_formats = ['eps','png','pdf']

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

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

#%% going backward TRIPS

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
_, sol_pre = fixed_point_solver(p_pre,context = 'calibration',x0=p_pre.guess,
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

p_pre_cf = p_baseline.copy()
p_pre_cf.delta[...,1] = p_pre.delta[...,1]

_, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
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
sol_pre_cf.scale_P(p_pre_cf)
sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
sol_pre_cf.compute_consumption_equivalent_welfare(p_pre_cf,sol_baseline)
sol_pre_cf.compute_world_welfare_changes(p_pre_cf,sol_baseline)

_, dyn_sol_pre_cf = dyn_fixed_point_solver(p_pre_cf, sol_baseline,sol_fin=sol_pre_cf,
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
dyn_sol_pre_cf.compute_non_solver_quantities(p_pre_cf)

p_pre_cf_fix_north = p_baseline.copy()
p_pre_cf_fix_north.delta[...,1] = p_pre.delta[...,1]
for country_idx in [0,1,2,6,7]:
    p_pre_cf_fix_north.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north = fixed_point_solver(p_pre_cf_fix_north,context = 'counterfactual',x0=p_pre_cf_fix_north.guess,
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
sol_pre_cf_fix_north.scale_P(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_consumption_equivalent_welfare(p_pre_cf_fix_north,sol_baseline)
sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf,sol_baseline)

_, dyn_sol_pre_cf_fix_north = dyn_fixed_point_solver(p_pre_cf_fix_north, sol_baseline,sol_fin=sol_pre_cf_fix_north,
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
dyn_sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)

pre_trips_path = save_path+'pre-trips/'
try:
    os.mkdir(pre_trips_path)
except:
    pass

modified_countries_names = {'USA': 'USA',
 'EUR': 'Europe',
 'JAP': 'Japan',
 'KOR':'Korea',
 'CAN':'Canada',
 'MEX':'Mexico',
 'RUS':'Russia',
 'CHN': 'China',
 'BRA': 'Brazil',
 'IND': 'India',
 'ROW': 'Rest of\nthe world'}

df = pd.DataFrame(
    index = pd.Index([modified_countries_names[c] for c in p_baseline.countries]+['World\nNegishi','World\nEqual'],
                                       name = 'country')
    )

df['delta_baseline'] = p_baseline.delta[...,1].tolist()+[np.nan,np.nan]
df['delta_1992'] = p_pre.delta[...,1].tolist()+[np.nan,np.nan]
df['static_welfare_change'] = sol_pre_cf.cons_eq_welfare.tolist()+[
    sol_pre_cf.cons_eq_negishi_welfare_change,sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change'] = dyn_sol_pre_cf.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf.cons_eq_negishi_welfare_change,dyn_sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare change_fixed_delta_north'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
grey_rgb = (105/256,105/256,105/256)
# grey_rgb = (0,0,0)

for col in ['static_welfare_change','dynamic_welfare_change',
            'static_welfare_change_fixed_delta_north','dynamic_welfare change_fixed_delta_north']:

    fig,ax = plt.subplots()
    # ax.bar(df.index, df['static welfare change']*100-100)
    ax.barh(df.index, df[col]*100-100, 
            color = Category18[:len(p_baseline.countries)]+[grey_rgb,grey_rgb],
           # color = Category18[:len(p_baseline.countries)+2],
           # hatch = ['']*len(p_baseline.countries)+['/','x']
           )
    ax.invert_yaxis()
    ax.set_xlabel('Welfare change (%)')
    
    plt.show()

df.loc['growth_rate','delta_baseline'] = sol_baseline.g
df.loc['growth_rate','static_welfare_change'] = sol_pre_cf.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.g

#%% going forward for TRIPS

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
_, sol_pre = fixed_point_solver(p_pre,context = 'calibration',x0=p_pre.guess,
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

p_pre_cf = p_pre.copy()
p_pre_cf.delta[...,1] = p_baseline.delta[...,1]

_, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
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
sol_pre_cf.scale_P(p_pre_cf)
sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
sol_pre_cf.compute_consumption_equivalent_welfare(p_pre_cf,sol_pre)
sol_pre_cf.compute_world_welfare_changes(p_pre_cf,sol_pre)

_, dyn_sol_pre_cf = dyn_fixed_point_solver(p_pre_cf, sol_pre,sol_fin=sol_pre_cf,
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
dyn_sol_pre_cf.compute_non_solver_quantities(p_pre_cf)

p_pre_cf_fix_north = p_pre.copy()
p_pre_cf_fix_north.delta[...,1] = p_baseline.delta[...,1]
for country_idx in [0,1,2,6,7]:
    p_pre_cf_fix_north.delta[country_idx,1] = p_pre.delta[country_idx,1]

_, sol_pre_cf_fix_north = fixed_point_solver(p_pre_cf_fix_north,context = 'counterfactual',x0=p_pre_cf_fix_north.guess,
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
sol_pre_cf_fix_north.scale_P(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_consumption_equivalent_welfare(p_pre_cf_fix_north,sol_pre)
sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf,sol_pre)

_, dyn_sol_pre_cf_fix_north = dyn_fixed_point_solver(p_pre_cf_fix_north, sol_pre,sol_fin=sol_pre_cf_fix_north,
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
dyn_sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)

#%%

def compute_welfare_for_trade_cost(factor,country,p_baseline,
                                   sol_baseline,p_pre_trips):
    index_country = p_baseline.countries.index(country)
    p_cf = p_baseline.copy()
    p_cf.tau[:,3,1:] = 1
    _, sol_cf = fixed_point_solver(p_cf,
                            context = 'counterfactual',
                            x0=p_cf.guess,
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
    
    sol_cf.scale_P(p_cf)
    sol_cf.compute_non_solver_quantities(p_cf)
    sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
    
    # print(sol_cf.cons_eq_welfare)
    
    return sol_cf.cons_eq_welfare[index_country] - 1

# for country in p_baseline.countries:
    
# print(compute_welfare_for_trade_cost(0.99,'CHN',p_baseline,

#                                    sol_baseline,p_pre
#                                    ))

df = pd.DataFrame()


for x in range(50,201,50):
    print(x/100)
    for country in ['CHN', 'BRA', 'IND', 'RUS']:
    # for country in ['MEX']:
        df.loc[x/100,country] = compute_welfare_for_trade_cost(x/100,
                                            country,p_baseline,
                                            sol_baseline,p_pre
                                            )

df.plot()

# fig,ax = plt.subplots()

# for country in ['CHN', 'BRA', 'IND', 'RUS']:
#     print(country)
#     ax.plot([x/10 for x in range(5,20)],[compute_welfare_for_trade_cost(x/10,
#                                     country,p_baseline,
#                                     sol_baseline,p_pre
#                                     ) for x in range(5,20)],
#             label = country)

# plt.legend()
# plt.show()

#%%

def compute_welfare_for_trade_cost(factor,country,p_baseline,
                                   sol_baseline,p_pre_trips):
    index_country = p_baseline.countries.index(country)
    p_cf = p_baseline.copy()
    p_cf.delta[...,1:] = p_pre_trips.delta[...,1:]
    for rich_country_index in [0,1,2,6,7]:
        # p_cf.delta[rich_country_index,1] = p_baseline.delta[rich_country_index,1]
        p_cf.tau[rich_country_index,index_country,:][
            p_cf.tau[rich_country_index,index_country,:] > 1
            ] = (p_cf.tau[rich_country_index,index_country,:][
                p_cf.tau[rich_country_index,index_country,:] > 1
                ] - 1)*factor+1
        # print((p_cf.tau[:,index_country,:]-1)/(p_baseline.tau[:,index_country,:]-1))
    _, sol_cf = fixed_point_solver(p_cf,
                            context = 'counterfactual',
                            x0=p_cf.guess,
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
    
    sol_cf.scale_P(p_cf)
    sol_cf.compute_non_solver_quantities(p_cf)
    sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
    
    # print(sol_cf.cons_eq_welfare)
    
    return sol_cf.cons_eq_welfare[index_country] - 1



# for country in p_baseline.countries:
    
# print(compute_welfare_for_trade_cost(0.99,'CHN',p_baseline,

#                                    sol_baseline,p_pre
#                                    ))

df2 = pd.DataFrame()


for x in range(50,200):
    print(x/100)
    for country in ['CHN', 'BRA', 'IND', 'RUS','MEX']:
    # for country in ['MEX']:
        df2.loc[x/100,country] = compute_welfare_for_trade_cost(x/100,
                                            country,p_baseline,
                                            sol_baseline,p_pre
                                            )
    
# fig,ax = plt.subplots()

# for country in ['CHN', 'BRA', 'IND', 'RUS']:
#     print(country)
#     ax.plot([x/10 for x in range(5,20)],[compute_welfare_for_trade_cost(x/10,
#                                     country,p_baseline,
#                                     sol_baseline,p_pre
#                                     ) for x in range(5,20)],
#             label = country)

# plt.legend()
# plt.show()

#%%

df3 = pd.DataFrame()

for country in ['CHN', 'BRA', 'IND', 'RUS']:
    df3.loc[country,'value'] = df.index[df[country].abs().argmin()]
    
#%%

def compute_welfare_for_trade_cost(factor,country,p_baseline,
                                   sol_baseline,p_pre_trips):
    index_country = p_baseline.countries.index(country)
    p_cf = p_baseline.copy()
    p_cf.delta[...,1:] = p_pre_trips.delta[...,1:]
    for rich_country_index in [0,1,2,6,7]:
        # p_cf.delta[rich_country_index,1] = p_baseline.delta[rich_country_index,1]
        p_cf.tau[index_country,rich_country_index,:][
            p_cf.tau[index_country,rich_country_index,:] > 1
            ] = (p_cf.tau[index_country,rich_country_index,:][
                p_cf.tau[index_country,rich_country_index,:] > 1
                ] - 1)*factor+1
        # print((p_cf.tau[:,index_country,:]-1)/(p_baseline.tau[:,index_country,:]-1))
    _, sol_cf = fixed_point_solver(p_cf,
                            context = 'counterfactual',
                            x0=p_cf.guess,
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
    
    sol_cf.scale_P(p_cf)
    sol_cf.compute_non_solver_quantities(p_cf)
    sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
    
    # print(sol_cf.cons_eq_welfare)
    
    return sol_cf.cons_eq_welfare[index_country] - 1



# for country in p_baseline.countries:
    
# print(compute_welfare_for_trade_cost(0.99,'CHN',p_baseline,

#                                    sol_baseline,p_pre
#                                    ))

df4 = pd.DataFrame()


for x in range(50,200):
    print(x/100)
    for country in ['CHN', 'BRA', 'IND', 'RUS']:
    # for country in ['MEX']:
        df4.loc[x/100,country] = compute_welfare_for_trade_cost(x/100,
                                            country,p_baseline,
                                            sol_baseline,p_pre
                                            )