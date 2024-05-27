#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:37:24 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classes import moments, parameters, var, var_with_entry_costs, dynamic_var

baseline = '1300'
variation = 'baseline'

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

#%%
from classes import var_with_entry_costs

p = p_baseline.copy()
p.a = 0.5
p.d = 1.5

vec = sol_baseline.vector_from_var()
vec = np.concatenate((vec,sol_baseline.price_indices))

sol = var_with_entry_costs.var_from_vector(vec, p, context='counterfactual',compute=False)

sol.compute_growth(p)
sol.compute_entry_costs(p)
sol.compute_V(p)
sol.compute_patenting_thresholds(p)
