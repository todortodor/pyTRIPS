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
    # {'baseline':'501',
    #                   'variation':'1.0'},
    # {'baseline':'501',
    #                   'variation':'2.0.0'},
    # {'baseline':'501',
    #                   'variation':'2.0.1'},
    # {'baseline':'501',
    #                   'variation':'2.0.2'},
    # {'baseline':'501',
    #                   'variation':'2.0.3'},
    # {'baseline':'501',
    #                   'variation':'2.0.4'},
    # {'baseline':'501',
    #                   'variation':'2.0.5'},
    # {'baseline':'501',
    #                   'variation':'2.0.6'},
    # {'baseline':'501',
    #                   'variation':'2.0.7'},
    # {'baseline':'501',
    #                   'variation':'2.0.8'},
    # {'baseline':'501',
    #                   'variation':'2.0.9'},
    # {'baseline':'501',
    #                   'variation':'2.0.10'},
    {'baseline':'501',
                      'variation':'2.0.11'},
    {'baseline':'501',
                      'variation':'2.0.12'},
    {'baseline':'501',
                      'variation':'2.0.13'},
    {'baseline':'501',
                      'variation':'2.0.14'},
    {'baseline':'501',
                      'variation':'2.0.15'},
    {'baseline':'501',
                      'variation':'2.0.16'},
    {'baseline':'501',
                      'variation':'2.0.17'},
    {'baseline':'501',
                      'variation':'2.0.18'},
    {'baseline':'501',
                      'variation':'2.0.19'},
    # {'baseline':'501',
    #                   'variation':'2.0.20'},
    ]
# baseline_dic = {'baseline':'501',
#                       'variation':'1.0'}
for baseline_dic in baseline_dics:
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    
    p_baseline = parameters(n=7,s=2)
    # p_baseline.load_data('calibration_results_matched_economy/baseline_402_variations/17.1.1/')
    p_baseline.load_data(baseline_path)
    
    deltas, welfares = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=1,method='fixed_point',
                     plot_convergence = True,solver_options=None,tol=5e-3,window=4,plot_history=False,
                     reverse_search=False,dynamics=True)
    
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
                        'fixed_point']+deltas[:,-1].tolist(), 
                        index = deltas_df.columns).T
        deltas_df = pd.concat([deltas_df, run],ignore_index=True)
        deltas_df.to_csv('nash_eq_recaps/dyn_deltas.csv')