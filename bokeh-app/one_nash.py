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
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_nash_eq
# from random import random
from tqdm import tqdm
import matplotlib.pylab as pylab

p_baseline = parameters(n=7,s=2)
p_baseline.load_data('calibration_results_matched_economy/baseline_402_variations/17.1.1/')

deltas, welfares = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',
                 plot_convergence = False,solver_options=None,tol=5e-3,window=4,plot_history=True)