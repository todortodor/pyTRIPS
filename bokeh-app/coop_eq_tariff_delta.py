#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:01:14 2024

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_coop_eq_tariff_delta
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
    {'baseline':'1050','variation': 'baseline'},
    ]

lb_tariff = -0.1
ub_tariff = 1
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
            
            # p_opti, sol_opti = find_opt_tariff_delta(p_baseline,scenario,
            #                  lb_tariff=lb_tariff,ub_tariff=ub_tariff,dynamics=False,
            #                  solver_options=None,tol=1e-8,
            #                  static_eq_tariff = None,custom_weights=None,
            #                  # custom_x0 = np.ones(p_baseline.N)*12,
            #                  custom_x0 = None,
            #                  max_workers=12)
            
            p_opti, sol_opti = find_coop_eq_tariff_delta(p_baseline,aggregation_method,
                             lb_delta=0.01,ub_delta=12,
                             lb_tariff=lb_tariff,ub_tariff=ub_tariff,dynamics=False,
                             solver_options=None,tol=1e-8,
                             static_eq_tariff = None,custom_weights=None,
                             custom_x0 = None,max_workers=12,
                             custom_dyn_sol_options=None, displays = False,
                             parallel=True)
            
            print(time.perf_counter() - start)
            
            baseline = baseline_dic['baseline']
            
            try:
                os.mkdir(f'opt_tariff_delta/{baseline}/')
            except:
                pass
            
            if aggregation_method == 'pop_weighted':
                try:
                    os.mkdir(f'opt_tariff_delta/{baseline}/scenario_7')
                except:
                    pass
                p_opti.write_params(f'opt_tariff_delta/{baseline}/scenario_7/')
            if aggregation_method == 'negishi':
                try:
                    os.mkdir(f'opt_tariff_delta/{baseline}/scenario_11')
                except:
                    pass
                p_opti.write_params(f'opt_tariff_delta/{baseline}/scenario_11/')
            
            