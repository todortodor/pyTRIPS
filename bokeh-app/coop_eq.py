#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:59:20 2022

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_coop_eq
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
    {'baseline':'802','variation': 'baseline'},
    {'baseline':'802','variation': '1.0'},
    {'baseline':'802','variation': '2.0'},
    {'baseline':'802','variation': '3.0'},
    # {'baseline':'608','variation': 'baseline'},
    # {'baseline':'609','variation': 'baseline'},
    # {'baseline':'610','variation': 'baseline'},
    # {'baseline':'618','variation': 'baseline'},
    # {'baseline':'618','variation': '15.0'},
    # {'baseline':'601','variation': 'baseline'},
    # {'baseline':'601','variation': '1.0'},
    # {'baseline':'601','variation': '1.1'},
    # {'baseline':'601','variation': '1.2'},
    # {'baseline':'601','variation': '1.3'},
    # {'baseline':'601','variation': '1.4'},
    # {'baseline':'601','variation': '1.5'},
    # {'baseline':'601','variation': '1.6'},
    # {'baseline':'601','variation': '1.7'},
    # {'baseline':'601','variation': '1.8'},
    # {'baseline':'601','variation': '1.9'},
    # {'baseline':'601','variation': '1.10'},
    # {'baseline':'601','variation': '1.11'},
    # {'baseline':'601','variation': '1.12'},
    # {'baseline':'601','variation': '1.13'},
    # {'baseline':'601','variation': '1.14'},
    # {'baseline':'601','variation': '1.15'},
    # {'baseline':'601','variation': '1.16'},
    # {'baseline':'601','variation': '1.17'},
    # {'baseline':'601','variation': '1.18'},
    # {'baseline':'601','variation': '1.19'},
    # {'baseline':'601','variation': '1.20'},
    # {'baseline':'601','variation': '1.21'},
    # {'baseline':'601','variation': '1.22'},
    # {'baseline':'601','variation': '1.23'},
    # {'baseline':'601','variation': '1.24'},
    # {'baseline':'601','variation': '1.25'},
    # {'baseline':'601','variation': '1.26'},
    # {'baseline':'601','variation': '1.27'},
    # {'baseline':'601','variation': '1.28'}
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
    
    for aggregation_method in ['pop_weighted','negishi']:
        print(aggregation_method)
        
        p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method,
                         lb_delta=0.01,ub_delta=12,dynamics=False,
                         solver_options=None,tol=1e-15,
                         static_eq_deltas = None,custom_weights=None)
        
        write = True
        if write:
            if not os.path.exists('coop_eq_recaps/deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries)
                deltas_df.to_csv('coop_eq_recaps/deltas.csv')
            deltas_df = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+p_opti.delta[...,1].tolist(), 
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('coop_eq_recaps/deltas.csv')
            
            if not os.path.exists('coop_eq_recaps/cons_eq_welfares.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
                cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
            cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
                                                               sol_opti.cons_eq_negishi_welfare_change], 
                            index = ['baseline',
                                     'variation',
                                     'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
