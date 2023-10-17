#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:38:45 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

baseline = '1020'
variation = 'baseline'

baseline_pre_trips_variation = '1020'
pre_trips_cf = True
pre_trips_variation = '9.2'
partial_variation = '9.0'

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

p_pre = parameters()
p_pre.load_run('calibration_results_matched_economy/baseline_1030_variations/9.2/')

p_pre_cf = p_baseline.copy()
p_pre_cf.delta[...,1] = p_pre.delta[...,1]
for country_idx in [0,1,2,6,7,10]:
    p_pre_cf.delta[country_idx,1] = p_baseline.delta[country_idx,1]

p_low_to = parameters()
p_low_to.load_run('calibration_results_matched_economy/baseline_1030_variations/99.0/')

p_low_to_pre = parameters()
p_low_to_pre.load_run('calibration_results_matched_economy/baseline_1030_variations/9.2/')

p_low_to_pre_cf = p_low_to.copy()
p_low_to_pre_cf.delta[...,1] = p_low_to_pre.delta[...,1]
for country_idx in [0,1,2,6,7,10]:
    p_low_to_pre_cf.delta[country_idx,1] = p_low_to.delta[country_idx,1]
