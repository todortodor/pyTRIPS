#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:53:03 2023

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
import math

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

baseline = '1030'
variation = 'baseline'

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

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
countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

temp_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/misc/discussion_material_02_10_meeting/robustness_checks/'

#%% Unilateral patent protections
variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low Turnover target',
    '99.1':'High Turnover target',
    '99.2':'Low Trade elasticity target',
    '99.3':'High Trade elasticity target',
    '99.4':'Low Kogan moment target',
    '99.5':'High Kogan moment target',
    '99.6':r'Low $\sigma^1$',
    '99.7':r'High $\sigma^1$',
    '99.8':r'Low $\kappa$',
    '99.9':r'High $\kappa$',
    '99.10':'Low Growth target',
    '99.11':'High Growth target',
    '99.12':r'Low $\rho$',
    '99.13':r'High $\rho$',
    '99.14':'Low Aggregate pat. cost in US',
    '99.15':'High Aggregate pat. cost in US',
    }
for country in p_baseline.countries:
# for country in ['USA']:
    fig,ax = plt.subplots()
    for i,rob_check in enumerate(variations_of_robust_checks):
        variation = rob_check
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
        df_welfare = pd.read_csv(local_path+'dyn_'+country+'.csv')
        if rob_check == 'baseline':
            ax.plot(df_welfare['delt'],df_welfare[country],color='k',
                    label=variations_of_robust_checks[rob_check],lw=5)
        else:
            if variations_of_robust_checks[rob_check].startswith('High'):
                ls = '-'
            if variations_of_robust_checks[rob_check].startswith('Low'):
                ls = '--'
            ax.plot(df_welfare['delt'],df_welfare[country],
                    label=variations_of_robust_checks[rob_check],
                    color = sns.color_palette()[math.floor((i-1)/2)],
                    ls = ls)
    ax.set_ylabel('Welfare change')
    ax.set_xlabel(r'Proportional change of $\delta$')
    ax.set_xscale('log')
    plt.legend()
    plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
    plt.savefig(temp_path+'unilateral_patent_protection_'+country+'.png')
    plt.show()

#%% Graph by check

# variations_of_robust_checks = {
#     'baseline':'Baseline',
#     '99.0':'Low TO',
#     '99.1':'High TO',
#     '99.2':'Low TE',
#     '99.3':'High TE',
#     '99.4':'Low KM',
#     '99.5':'High KM',
#     '99.6':'Low Sigma',
#     '99.7':'High Sigma',
#     '99.8':'Low Kappa',
#     '99.9':'High Kappa',
#     '99.10':'Low Growth',
#     '99.11':'High Growth',
#     '99.12':'Low rho',
#     '99.13':'High rho',
#     '99.14':'Low UUPCOST',
#     '99.15':'High UUPCOST',
#     }
# # for country in p_baseline.countries:

# dfs = {}
    
# for country in ['USA']:
#     fig,ax = plt.subplots()
#     for i,rob_check in enumerate(variations_of_robust_checks):
#         variation = rob_check
#         if variation == 'baseline':
#             local_path = cf_path+'baseline_'+baseline+'/'
#         else:
#             local_path = \
#                 cf_path+f'baseline_{baseline}_{variation}/'
                
#         df_welfare = pd.read_csv(local_path+country+'.csv')
#         if rob_check == 'baseline':
#             df_baseline = df_welfare
#             for j,c in enumerate(p_baseline.countries):
#                 ax.plot(df_welfare['delt'],df_welfare[c],
#                 color = Category18[j],
#                         label=c,lw=5)
#         else:
#             if variations_of_robust_checks[rob_check].startswith('High'):
#                 ls = ':'
#             if variations_of_robust_checks[rob_check].startswith('Low'):
#                 ls = '--'
#             for j,c in enumerate(p_baseline.countries):
#                 ax.plot(df_welfare['delt'],df_welfare[c],
#                         # label=variations_of_robust_checks[rob_check],
#                         color = Category18[j],
#                         ls = ls)
        
#         dfs[variations_of_robust_checks[rob_check]] = df_welfare
#     ax.set_ylabel('Welfare change')
#     ax.set_xlabel(r'Proportional change of $\delta$')
#     ax.set_xscale('log')
#     plt.legend()
#     plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
#     # plt.savefig(temp_path+'unilateral_patent_protection_'+country+'.png')
#     plt.show()

#%% Coop eq

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
coop_deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

df = coop_deltas.loc[(coop_deltas.baseline == int(baseline))
                        & (coop_deltas.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)

coop_w = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

dfw = coop_w.loc[(coop_w.baseline == int(baseline))
                        & (coop_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1).set_index(['variation','aggregation_method'])

for i,rob_check in enumerate(variations_of_robust_checks):

    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
    for coop in ['negishi','pop_weighted']:
        
        p.delta[...,1] = df.set_index(['variation','aggregation_method']).loc[(variation,coop)][p_baseline.countries].values.squeeze()
        
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
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
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p) 
        
        dfw.loc[(variation,coop),'growth rate'] = sol_c.g

#%%

dfw = dfw.reset_index()

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries].sort_index().T.round(3) 
# df.to_csv(temp_path+'coop_equilibria.csv')

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi', 'growth rate']].sort_index()
dfw[p_baseline.countries+['Equal', 'Negishi']
    ] = (dfw[p_baseline.countries+['Equal', 'Negishi']]*100-100).round(2)

dfw['growth rate'] = (dfw['growth rate']*100).round(2)

df.to_csv(temp_path+'coop_equilibria.csv')
dfw.T.to_csv(temp_path+'coop_equilibria_welfare.csv')

#%%  Nash eq

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
nash_deltas = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

df = nash_deltas.loc[(nash_deltas.baseline == int(baseline))
                        & (nash_deltas.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)


nash_w = pd.read_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')
dfw = nash_w.loc[(nash_w.baseline == int(baseline))
                        & (nash_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1).set_index('variation').drop('method',axis=1)

for i,rob_check in enumerate(variations_of_robust_checks):

    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
        
    p.delta[...,1] = df.set_index(
        ['variation']).loc[variation][p_baseline.countries].values.squeeze()
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
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
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    
    dfw.loc[variation,'growth rate'] = sol_c.g

dfw = dfw.reset_index()

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['Quantity','Change'])[p_baseline.countries].sort_index().T.round(3) 
# df.to_csv(temp_path+'coop_equilibria.csv')

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi', 'growth rate']].sort_index()
dfw[p_baseline.countries+['Equal', 'Negishi']
    ] = (dfw[p_baseline.countries+['Equal', 'Negishi']]*100-100).round(2)

dfw['growth rate'] = (dfw['growth rate']*100).round(2)

df.to_csv(temp_path+'nash_equilibrium.csv')
dfw.T.to_csv(temp_path+'nash_equilibrium_welfare.csv')

#%%

results_path = 'calibration_results_matched_economy/'

# df = pd.DataFrame()
# df_pre = pd.DataFrame()
df_welf = pd.DataFrame(index = p_baseline.countries+['Growth rate','North','South'])

for i,rob_check in enumerate(variations_of_robust_checks):
    print(rob_check)
    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
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
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    
    p_pre = parameters()
    p_pre.load_run(local_path_pre_trips)
    
    # df[variations_of_robust_checks[rob_check]] = p.delta[...,1]
    # df_pre[variations_of_robust_checks[rob_check]] = p_pre.delta[...,1]
    
    p_pre_cf_fix_north = p.copy()
    p_pre_cf_fix_north.delta[...,1] = p_pre.delta[...,1]
    for country_idx in [0,1,2,6,7,10]:
        p_pre_cf_fix_north.delta[country_idx,1] = p.delta[country_idx,1]

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
    sol_pre_cf_fix_north.compute_consumption_equivalent_welfare(p_pre_cf_fix_north,sol_c)
    sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf_fix_north,sol_c)

    _, dyn_sol_pre_cf_fix_north = dyn_fixed_point_solver(p_pre_cf_fix_north, sol_c,sol_fin=sol_pre_cf_fix_north,
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
    
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 2015 delta'] = p.delta[:,1]
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 2015 delta'] = sol_c.g*100
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 1992 delta'] = p_pre.delta[:,1]
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 1992 delta'] = sol_pre_cf_fix_north.g*100
    
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare*100-100
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.sol_fin.g*100
    
    rich_count_indices = [0,1,2,6,7]
    poor_count_indices = [3,4,5,8,9]
    
    df_welf.loc['North',variations_of_robust_checks[rob_check]+' welfare change'] = sum([dyn_sol_pre_cf_fix_north.cons_eq_welfare[i]*p.labor[i] for i in rich_count_indices]
                                                                      )/sum([p.labor[i] for i in rich_count_indices])*100-100
    df_welf.loc['South',variations_of_robust_checks[rob_check]+' welfare change'] = sum([dyn_sol_pre_cf_fix_north.cons_eq_welfare[i]*p.labor[i] for i in poor_count_indices]
                                                                      )/sum([p.labor[i] for i in poor_count_indices])*100-100
    
    print(df_welf)

    df_welf.loc['Diff South North'] = df_welf.loc['South'] - df_welf.loc['North']


# df.to_csv(temp_path+'deltas.csv')
# df_pre.to_csv(temp_path+'pre_trips_deltas.csv')
    df_welf.round(3).to_csv(temp_path+'pre_trips_welfares.csv')
    
#     df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 2015 delta'] = p.delta[:,1]
#     df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 2015 delta'] = sol_c.g*100
#     df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 1992 delta'] = p_pre.delta[:,1]
#     df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 1992 delta'] = sol_pre_cf_fix_north.g*100
    
#     df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' welfare change'] = sol_pre_cf_fix_north.cons_eq_welfare*100-100
#     df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' welfare change'] = sol_pre_cf_fix_north.g*100
    
#     rich_count_indices = [0,1,2,6,7]
#     poor_count_indices = [3,4,5,8,9]
    
#     df_welf.loc['North',variations_of_robust_checks[rob_check]+' welfare change'] = sum([sol_pre_cf_fix_north.cons_eq_welfare[i]*p.labor[i] for i in rich_count_indices]
#                                                                       )/sum([p.labor[i] for i in rich_count_indices])*100-100
#     df_welf.loc['South',variations_of_robust_checks[rob_check]+' welfare change'] = sum([sol_pre_cf_fix_north.cons_eq_welfare[i]*p.labor[i] for i in poor_count_indices]
#                                                                       )/sum([p.labor[i] for i in poor_count_indices])*100-100
    
#     print(df_welf)

#   df_welf['Diff South North'] = df_welf['South'] - df_welf['North']

# # df.to_csv(temp_path+'deltas.csv')
# # df_pre.to_csv(temp_path+'pre_trips_deltas.csv')
    # df_welf.round(3).to_csv(temp_path+'stat_pre_trips_welfares.csv')

#%%

fig,ax = plt.subplots()    

for i,c in enumerate(p_baseline.countries):
    ax.scatter(df.loc[c],df_pre.loc[c],color = Category18[i],label=c)

ax.plot([df.min().min(),df.max().max()],[df.min().min(),df.max().max()],ls='--',color='grey')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('2015 delta')
ax.set_ylabel('1992 delta')

# plt.savefig(temp_path+'pre_TRIPS_deltas_fig.png')

plt.legend()
plt.show()

#%%
    
sns.violinplot(data=df.T)
plt.yscale('log')

# df_welf.plot.scatter()