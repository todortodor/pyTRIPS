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
    # {'baseline':'1300','variation': 'baseline'},
    # {'baseline':'1300','variation': '2.0'},
    # {'baseline':'1300','variation': '10.2'},
    # {'baseline':'1300','variation': '10.3'},
    # {'baseline':'1300','variation': '10.4'},
    # {'baseline':'1300','variation': '10.5'},
    # {'baseline':'1300','variation': '12.0'},
    # {'baseline':'1300','variation': '13.0'},
    # {'baseline':'1300','variation': '99.0'},
    # {'baseline':'1300','variation': '99.1'},
    # {'baseline':'1300','variation': '99.2'},
    # {'baseline':'1300','variation': '99.3'},
    # {'baseline':'1300','variation': '99.4'},
    # {'baseline':'1300','variation': '99.5'},
    # {'baseline':'1300','variation': '99.6'},
    # {'baseline':'1300','variation': '99.7'},
    # {'baseline':'1300','variation': '99.8'},
    # {'baseline':'1300','variation': '99.9'},
    # {'baseline':'1300','variation': '99.10'},
    # {'baseline':'1300','variation': '99.11'},
    # {'baseline':'1300','variation': '99.12'},
    # {'baseline':'1300','variation': '99.13'},
    # {'baseline':'1300','variation': '99.14'},
    # {'baseline':'1300','variation': '99.15'},
    # {'baseline':'4003','variation': 'baseline'},
    # {'baseline':'4004','variation': 'baseline'},
    {'baseline':'4013','variation': 'baseline'},
    ]


lb_delta = 0.01
ub_delta = 12
# ub_delta = 1

import time

if __name__ == '__main__':
    for baseline_dic in baseline_dics:    
        if baseline_dic['variation'] == 'baseline':
            baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
        else:
            baseline_path = \
                f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        
        assert os.path.exists(baseline_path), 'run doesnt exist'
        
        # baseline_path = 'opt_tariff_delta/1040/scenario_3/'
        
        print(baseline_path)
        p_baseline = parameters()
        p_baseline.load_run(baseline_path)  
        
        for aggregation_method in ['pop_weighted','negishi']:
        # for aggregation_method in ['negishi']:
        # for aggregation_method in ['pop_weighted']:
            print(aggregation_method)
            
            start = time.perf_counter()
            
            p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method,
                             lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                             solver_options=None,tol=1e-12,
                             static_eq_deltas = None,custom_weights=None,
                             # custom_x0 = np.ones(p_baseline.N)*12,
                             parallel=False,
                             custom_x0 = None,
                             max_workers=12)
            
            print(time.perf_counter() - start)
            
            write = False
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

            # p_opti.write_params(f'opt_tariff_delta/{baseline_dic["baseline"]}/scenario_15/')
            
            save_directly = True
            if save_directly:
                direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
                p_opti.write_params(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')