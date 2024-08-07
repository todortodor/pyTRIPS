#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:27:47 2022

@author: slepot
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from solver_funcs import find_nash_eq_tariff
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab
import time
# from scipy.signal import argrelmin, argrelmax

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
    {'baseline':'1210','variation': 'baseline'},
    # {'baseline':'1030','variation': '99.0'},
    # {'baseline':'1030','variation': '99.1'},
    # {'baseline':'1030','variation': '99.2'},
    # {'baseline':'1030','variation': '99.3'},
    # {'baseline':'1030','variation': '99.4'},
    # {'baseline':'1030','variation': '99.5'},
    # {'baseline':'1030','variation': '99.6'},
    # {'baseline':'1030','variation': '99.7'},
    # {'baseline':'1030','variation': '99.8'},
    # {'baseline':'1030','variation': '99.9'},
    # {'baseline':'1030','variation': '99.10'},
    # {'baseline':'1030','variation': '99.11'},
    # {'baseline':'1030','variation': '99.12'},
    # {'baseline':'1030','variation': '99.13'},
    # {'baseline':'1030','variation': '99.14'},
    # {'baseline':'1030','variation': '99.15'},
    ]


lb_tariff=0
ub_tariff=1

if __name__ == '__main__':
    for baseline_dic in baseline_dics:    
        if baseline_dic['variation'] == 'baseline':
            baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
        else:
            baseline_path = \
                f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        
        assert os.path.exists(baseline_path), 'run doesnt exist'
        
        print(baseline_path)
        
        method = 'fixed_point'
        
        p_baseline = parameters()
        p_baseline.load_run(baseline_path)
        
        p_nash, sol_nash = find_nash_eq_tariff(p_baseline,lb_tariff=lb_tariff,ub_tariff=ub_tariff,method='fixed_point',
                         solver_options=None,tol=1e-4,
                         max_workers=12,parallel=True
                         )
        
        write = True
        if write:
        #     if not os.path.exists('nash_eq_recaps/deltas.csv'):
        #         deltas_df = pd.DataFrame(columns = ['baseline',
        #                                         'variation',
        #                                         'method'] + p_baseline.countries)
        #         deltas_df.to_csv('nash_eq_recaps/deltas.csv')
        #     deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
        #     run = pd.DataFrame(data = [baseline_dic['baseline'],
        #                     baseline_dic['variation'],
        #                     method]+p_nash.delta[:,1].tolist(), 
        #                     index = ['baseline',
        #                              'variation',
        #                              'aggregation_method'] + p_baseline.countries).T
        #     deltas_df = pd.concat([deltas_df, run],ignore_index=True)
        #     deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            
        #     if not os.path.exists('nash_eq_recaps/cons_eq_welfares.csv'):
        #         cons_eq_welfares = pd.DataFrame(columns = ['baseline',
        #                                         'variation',
        #                                         'method'] + p_baseline.countries + ['Equal','Negishi'])
        #         cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        #     cons_eq_welfares = pd.read_csv('nash_eq_recaps/cons_eq_welfares.csv',index_col=0)
        #     run = pd.DataFrame(data = [baseline_dic['baseline'],
        #                     baseline_dic['variation'],
        #                     method]+sol_nash.cons_eq_welfare.tolist()+[sol_nash.cons_eq_pop_average_welfare_change,
        #                                                        sol_nash.cons_eq_negishi_welfare_change], 
        #                     index = ['baseline',
        #                              'variation',
        #                              'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
        #     cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
        #     cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        
            baseline = baseline_dic['baseline']
            
            try:
                os.mkdir(f'opt_tariff_delta/{baseline}/')
            except:
                pass
    
            try:
                os.mkdir(f'opt_tariff_delta/{baseline}/scenario_0')
            except:
                pass
            p_nash.write_params(f'opt_tariff_delta/{baseline}/scenario_0/')
