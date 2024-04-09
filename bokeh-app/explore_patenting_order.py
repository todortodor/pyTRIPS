#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:41:28 2024

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
import scienceplots
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])
# import matplotlib.pyplot as plt
# plt.rcParams.update(plt.rcParamsDefault)

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (14, 11),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
# sns.set()
# sns.set_context('talk')
# # sns.set_style('whitegrid')
# sns.set(style="whitegrid",
#         font_scale=2,
#         rc={
#     "lines.markersize": 10,
#     "lines.linewidth": 3,
#     }
#     )
plt.style.use(['science', 'nature', 'no-latex'])
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({"axes.grid": True,
                     "grid.color": "grey",
                     'axes.axisbelow': True,
                     "grid.linewidth": 0.1,
                     'legend.framealpha': 1,
                     'legend.frameon': 1,
                     'legend.edgecolor': 'white',
                     'figure.dpi': 288,
                     })
# mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})

save_to_tex_options = dict(position_float='centering',
                           clines='all;index',
                           hrules=True)

# %% load baseline

baseline = '1210'
variation = 'baseline'

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'

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

sol_baseline = var.var_from_vector(
    p_baseline.guess, p_baseline, compute=True, context='counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline, p_baseline)
m_baseline.compute_moments_deviations()

# PC_model = np.maximum(1,
#                       np.einsum('ci,ni->nic',
#                                 sol_baseline.psi_m_star[...,1],
#                                 sol_baseline.psi_m_star[...,1]
#                                 )
#                       )**(-p_baseline.k)

A = np.zeros((p_baseline.N, p_baseline.N, p_baseline.N))
PC_model = np.zeros((p_baseline.N, p_baseline.N, p_baseline.N))

for n, destination in enumerate(p_baseline.countries):
    for i, origin in enumerate(p_baseline.countries):
        for c, origin in enumerate(p_baseline.countries):
                A[n, i, c] = np.maximum(
                    sol_baseline.psi_m_star[n,i,1],
                    sol_baseline.psi_m_star[c,i,1]
                )
                PC_model[n, i, c] = (A[n, i, c]/sol_baseline.psi_m_star[n,i,1])**(-p_baseline.k)
                # PC_model[n, i, c] = (A[n, i, c])**(-p_baseline.k)

# PC_model = np.einsum('nic,ci->nic',
#                      A,
#                      1/sol_baseline.psi_m_star[..., 1]
#                      )**(-p_baseline.k)

PC_data = pd.read_csv(
    f'/Users/slepot/Documents/taff/datas/PATSTAT/patenting_order/{p_baseline.N}_countries/year_2015.csv',
    index_col = ['destination_code', 'origin_code', 'condition_code']
)[['probability']]

df = pd.DataFrame(
    index=pd.MultiIndex.from_product([np.arange(1,13),np.arange(1,13),np.arange(1,13)],
                                     names = ['destination_code', 'origin_code', 'condition_code']),
    columns=['probability model'],
    data=PC_model.ravel()
    )

df = df.join(PC_data)

df.columns = ['probability model', 'probability data']
df.index = pd.MultiIndex.from_product([p_baseline.countries,p_baseline.countries,p_baseline.countries],
                                 names = ['destination_code', 'origin_code', 'condition_code'])
# df = df.fillna(0)
# test = df.loc[:,:,'USA']

#%%

for condition_country in p_baseline.countries:
    plt.scatter(df.loc[:,:,condition_country]['probability model'],df.loc[:,:,condition_country]['probability data'])
    plt.title('Conditional country : '+condition_country)
    plt.xlabel('model')
    plt.ylabel('data')
    plt.show()
    
# test = df.groupby('condition_code').mean()