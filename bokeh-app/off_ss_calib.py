#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:53:01 2023

@author: slepot
"""

from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver
from data_funcs import write_calibration_results
import os
import numpy as np

new_run = True
baseline_number = '618'
if new_run:
    p = parameters()
    p.correct_eur_patent_cost = True
    # p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
    p.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/1.1/')
    start_time = time.perf_counter()

    m = moments()
    m.load_run('calibration_results_matched_economy/'+baseline_number+'/')