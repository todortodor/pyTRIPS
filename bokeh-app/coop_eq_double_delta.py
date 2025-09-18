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
from classes import moments, parameters, var_double_diff_double_delta
from solver_funcs import fixed_point_solver_double_diff_double_delta, find_coop_eq_double_delta
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
    {'baseline':'1311','variation': 'baseline'},
    # {'baseline':'1311','variation': '2.0'},
    # {'baseline':'1311','variation': '3.0'},
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
        print(baseline_path)
        p_baseline = parameters()
        p_baseline.load_run(baseline_path)  
        
        for aggregation_method in ['pop_weighted','negishi']:
        # for aggregation_method in ['pop_weighted']:
            print(aggregation_method)
            
            start = time.perf_counter()
            
            p_opti, sol_opti = find_coop_eq_double_delta(p_baseline,aggregation_method,
                             lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                             solver_options=None,tol=1e-12,
                             static_eq_deltas = None,custom_weights=None,
                             custom_x0 = None,parallel=False,
                             max_workers=12)
            
            print(time.perf_counter() - start)
            
            write = True
            if write:
                save_directly = True
                if save_directly:
                    direct_save_path = baseline_dic["baseline"] + '_' + baseline_dic['variation']
                    p_opti.write_params(f'coop_eq_direct_saves/{direct_save_path}_{aggregation_method}/')
