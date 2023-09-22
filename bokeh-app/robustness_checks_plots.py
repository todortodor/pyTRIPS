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

temp_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/misc/discussion_material_18_09_meeting/robustness_checks/'

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
        df_welfare = pd.read_csv(local_path+country+'.csv')
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
    # plt.savefig(temp_path+'unilateral_patent_protection_'+country+'.png')
    plt.show()

#%% Graph by check

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
# for country in p_baseline.countries:

dfs = {}
    
for country in ['USA']:
    fig,ax = plt.subplots()
    for i,rob_check in enumerate(variations_of_robust_checks):
        variation = rob_check
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
                
        df_welfare = pd.read_csv(local_path+country+'.csv')
        if rob_check == 'baseline':
            df_baseline = df_welfare
            for j,c in enumerate(p_baseline.countries):
                ax.plot(df_welfare['delt'],df_welfare[c],
                color = Category18[j],
                        label=c,lw=5)
        else:
            if variations_of_robust_checks[rob_check].startswith('High'):
                ls = ':'
            if variations_of_robust_checks[rob_check].startswith('Low'):
                ls = '--'
            for j,c in enumerate(p_baseline.countries):
                ax.plot(df_welfare['delt'],df_welfare[c],
                        # label=variations_of_robust_checks[rob_check],
                        color = Category18[j],
                        ls = ls)
        
        dfs[variations_of_robust_checks[rob_check]] = df_welfare
    ax.set_ylabel('Welfare change')
    ax.set_xlabel(r'Proportional change of $\delta$')
    ax.set_xscale('log')
    plt.legend()
    plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
    # plt.savefig(temp_path+'unilateral_patent_protection_'+country+'.png')
    plt.show()

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

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries].sort_index().T.round(3) 
coopT = df.T
# df.to_csv(temp_path+'coop_equilibria.csv')

coop_w = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

dfw = coop_w.loc[(coop_w.baseline == int(baseline))
                        & (coop_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi']].sort_index().T 
coopwT = (dfw.T*100-100).round(2)
# df.to_csv(temp_path+'coop_equilibria.csv')

#%%  Nash eq

variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low TO',
    '99.1':'High TO',
    '99.2':'Low TE',
    '99.3':'High TE',
    # '99.6':'Low Sigma',
    # '99.7':'High Sigma',
    '99.8':'Low Kappa',
    '99.9':'High Kappa',
    '99.10':'Low Growth',
    '99.11':'High Growth',
    '99.4':'Low KM',
    '99.5':'High KM',
    '99.12':'Low rho',
    '99.13':'High rho',
    }
nash_deltas = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

df = nash_deltas.loc[(nash_deltas.baseline == int(baseline))
                        & (nash_deltas.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['Quantity','Change'])[p_baseline.countries].sort_index().T.round(2) 
nashT = df.T

nash_w = pd.read_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')
dfw = nash_w.loc[(nash_w.baseline == int(baseline))
                        & (nash_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi']].sort_index().T
nashwT = (dfw.T*100-100).round(2)
# df.to_csv(temp_path+'nash_equilibrium.csv')
