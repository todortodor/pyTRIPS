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
from solver_funcs import find_nash_eq
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab
# from scipy.signal import argrelmin, argrelmax

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
                # {'baseline':'101',
                #   'variation':'13.1'},
                {'baseline':'104',
                  'variation':'11.7'}
                 ]

# for baseline_number in ['101','102','104']:
#     baseline_dics.append({'baseline':baseline_number,
#                       'variation':'baseline'})
    
#     files_in_dir = next(os.walk('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list.sort(key=float)
    
#     for run in run_list:
#         baseline_dics.append({'baseline':baseline_number,
#                           'variation':run})

for baseline_dic in baseline_dics:    
# for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        # baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'_'+baseline_dic['variation']+'/'
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
        
    write = True
    
    method = 'fixed_point'
    
    deltas,welfares = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',
                     plot_convergence = True,solver_options=None,tol=5e-3)
    
    if write:
        if not os.path.exists('nash_eq_recaps/deltas.csv'):
            deltas_df = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries)
            deltas_df.to_csv('nash_eq_recaps/deltas.csv')
        deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+deltas[...,-1].tolist(), 
                        index = deltas_df.columns).T
        deltas_df = pd.concat([deltas_df, run],ignore_index=True)
        deltas_df.to_csv('nash_eq_recaps/deltas.csv')
        
        if not os.path.exists('nash_eq_recaps/cons_eq_welfares.csv'):
            cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                            'variation',
                                            'method'] + p_baseline.countries)
            cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        cons_eq_welfares = pd.read_csv('nash_eq_recaps/cons_eq_welfares.csv',index_col=0)
        run = pd.DataFrame(data = [baseline_dic['baseline'],
                        baseline_dic['variation'],
                        method]+welfares[...,-1].tolist(), 
                        index = cons_eq_welfares.columns).T
        cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
        cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')


# #%%    

# fig,ax = plt.subplots()

# ax2 = ax.twinx()

# deltas.plot(logy=True,ax=ax, xlabel = 'Code iterations', 
#               ylabel = 'Delta', 
#               title = 'Convergence to Nash equilibrium')
# welfares.plot(ax=ax2, ls = '--', ylabel = 'Consumption eq. welfare')

# ax.legend(loc=(-0.15,0.1))
# ax2.legend(loc=(1.05,0.1))

# plt.show()
