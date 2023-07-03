#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:40:41 2023

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
    # {'baseline':'1003','variation': 'baseline'},
    {'baseline':'1003','variation': '0.4'},
    ]

lb_delta = 0.01
ub_delta = 12

for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    assert os.path.exists(baseline_path), 'run doesnt exist'
    
    print(baseline_path)
    p_baseline = parameters()
    p_baseline.load_run(baseline_path)
    
    for aggregation_method in ['negishi','pop_weighted']:
        print(aggregation_method)
        static_eq_deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0).drop_duplicates(
            ['baseline','variation','aggregation_method'],keep='last')
        static_eq_deltas = static_eq_deltas.loc[
            (static_eq_deltas.baseline.astype('str') == baseline_dic['baseline'])
            & (static_eq_deltas.variation.astype('str') == baseline_dic['variation'])
            & (static_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()
        
        p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method,
                         lb_delta=0.01,ub_delta=12,dynamics=True,
                         solver_options=None,tol=1e-15,
                         static_eq_deltas = static_eq_deltas,
                         custom_weights=None)
        
        write = True
        if write:
            if not os.path.exists('coop_eq_recaps/dyn_deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+p_opti.delta[...,1].tolist(), 
                            # index = deltas_df.columns).T
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/dyn_deltas.csv')
            
            if not os.path.exists('coop_eq_recaps/dyn_cons_eq_welfares.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
                cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')
            cons_eq_welfares = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
                                                               sol_opti.cons_eq_negishi_welfare_change], 
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv')

#%%