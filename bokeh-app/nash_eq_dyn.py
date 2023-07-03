#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:27:29 2022

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from solver_funcs import find_nash_eq
# import seaborn as sns|
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_nash_eq
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab

baseline_dics = [
    # {'baseline':'1003','variation': 'baseline'},
    {'baseline':'1003','variation': '0.4'},
    ]

for baseline_dic in baseline_dics:
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    assert os.path.exists(baseline_path), 'run doesnt exist'
    
    method = 'fixed_point'
    
    p_baseline = parameters()
    p_baseline.load_run(baseline_path)
    
    p_nash, sol_nash = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=12,method=method,
                     plot_convergence = True,solver_options=None,tol=5e-3,plot_history=False,
                     dynamics=True)
    
    write = True
    if write:
        if not os.path.exists('nash_eq_recaps/dyn_deltas.csv'):
            deltas_df = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries)
            deltas_df.to_csv('nash_eq_recaps/dyn_deltas.csv')
        deltas_df = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        'fixed_point']+p_nash.delta[...,1].tolist(), 
                        index = ['baseline',
                                 'variation',
                                 'aggregation_method'] + p_baseline.countries).T
        deltas_df = pd.concat([deltas_df, run],ignore_index=True)
        deltas_df.to_csv('nash_eq_recaps/dyn_deltas.csv')
        
        if not os.path.exists('nash_eq_recaps/dyn_cons_eq_welfares.csv'):
            cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries + ['Equal','Negishi'])
            cons_eq_welfares.to_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv')
        cons_eq_welfares = pd.read_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+sol_nash.cons_eq_welfare.tolist()+[sol_nash.cons_eq_pop_average_welfare_change,
                                                           sol_nash.cons_eq_negishi_welfare_change], 
                        index = ['baseline',
                                 'variation',
                                 'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
        cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
        cons_eq_welfares.to_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv')

#%%