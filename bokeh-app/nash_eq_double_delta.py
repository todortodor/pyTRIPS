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
from solver_funcs import find_nash_eq_double_delta
# import seaborn as sns
from classes import moments, parameters
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
    {'baseline':'1312','variation': 'baseline'},
    {'baseline':'1312','variation': '1.07'},
    {'baseline':'1312','variation': '2.02'},
    {'baseline':'1312','variation': '2.07'},
    # {'baseline':'1312','variation': '4.0'},
    ]


lb_delta=0.01
ub_delta=12

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
        
        p_nash, sol_nash = find_nash_eq_double_delta(p_baseline,lb_delta=lb_delta,ub_delta=ub_delta,method='fixed_point',
                         plot_convergence = False,solver_options=None,tol=1e-4,
                            # delta_init=np.ones(p_baseline.N)*ub_delta,
                            max_workers=12,parallel=False
                         )
        
        save_directly = True
        if save_directly:
            direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
            p_nash.write_params(f'coop_eq_direct_saves/{direct_save_path}_nash/')
