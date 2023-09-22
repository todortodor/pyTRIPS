#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:36:11 2023

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
# p_baseline.load_data(run_path)
p_baseline.load_run(run_path)

m_baseline = moments()
# m_baseline.load_data()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

#%%

data_path = 'data/'
df_kog = pd.read_csv(data_path+'koga_updated.csv')

km_series = df_kog['KM_article'].dropna().values

import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
import matplotlib.pyplot as plt

data = km_series

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, m_baseline.KM_target, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.axvline(x=m_baseline.KM_target)

# plt.show()

data_path = 'data/'
df_to_data = pd.read_csv(data_path+'turnover_imports_weighted_11_countries.csv'
                         )[['year','HS_digits','A3']].pivot(
                                columns= 'HS_digits',
                                index = 'year',
                                values = 'A3'
                            )[[6,8,10]]

import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm
import matplotlib.pyplot as plt

data = df_to_data[10].values

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, m_baseline.TO_target, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.axvline(x=m_baseline.TO_target)

plt.show()

#%%

data_path = 'data/'
df_kog = pd.read_csv(data_path+'koga_updated.csv')
km_series = df_kog['KM_article'].dropna().values

to_series = pd.read_csv(data_path+'turnover_imports_weighted_11_countries.csv'
                         )[['year','HS_digits','A3']].pivot(
                                columns= 'HS_digits',
                                index = 'year',
                                values = 'A3'
                            )[10].values
                             
km_fit = norm.fit(km_series)
km_target = m_baseline.KM_target
to_fit = norm.fit(to_series)
to_target = m_baseline.TO_target

fig,ax = plt.subplots()

ax.hist(km_series/km_target, bins=25, density=True, label='Historical KM')
ax.hist(to_series/to_target, bins=15, density=True, label='Historical TO')

x = np.linspace(0, km_series.max(), 100)
p = norm.pdf(x, *km_fit)
ax.plot(x/km_target, p/p.max(), 'k', linewidth=5, 
         label = 'KM fit')

x = np.linspace(0, to_series.max(), 100)
p = norm.pdf(x, *to_fit)
ax.plot(x/to_target, p/p.max(), 'grey', linewidth=5,  ls = '--',
         label = 'TO fit')

ax.axvline(x=km_target, color = 'r')
# ax.axvline(x=m_baseline.TO_target)

plt.legend()

plt.show()
