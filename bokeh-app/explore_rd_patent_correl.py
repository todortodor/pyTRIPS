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

baseline = '1300'
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

#%% Plot number of patents per RD of research

df = pd.DataFrame(index=p_baseline.countries)

df['RD'] = (p_baseline.data.rnd_gdp * p_baseline.data.gdp).values/1e6/p_baseline.data.labor.values
df['patent flows'] = m_baseline.cc_moments.query('destination_code!=origin_code'
                                    ).groupby('origin_code').sum()['patent flows'
                                    ].values/p_baseline.data.labor.values
north_countries = ['USA', 'EUR', 'JAP', 'CAN', 'KOR']
south_countries = ['CHN', 'BRA', 'IND', 'RUS', 'MEX', 'ZAF' #,'ROW'
                   ]

north = df.loc[north_countries]
south = df.loc[south_countries]

print('World average for number of patent families per million dollar of RD is :',df.sum()['patent flows']/df.sum()['RD'])
print('North countries :',north.sum()['patent flows']/north.sum()['RD'])
print('South countries :',south.sum()['patent flows']/south.sum()['RD'])

print('Correlation between number of patent families per million dollar of RD is :',np.corrcoef(df['patent flows'],df['RD'])[1,0])
print('North countries :',np.corrcoef(north['patent flows'],north['RD'])[1,0])
print('South countries :',np.corrcoef(south['patent flows'],south['RD'])[1,0])

print('Correlation between log of number of patent families per million dollar of RD is :',np.corrcoef(np.log(df['patent flows']),np.log(df['RD']))[1,0])
print('North countries :',np.corrcoef(np.log(north['patent flows']),np.log(north['RD']))[1,0])
print('South countries :',np.corrcoef(np.log(south['patent flows']),np.log(south['RD']))[1,0])

fig,ax = plt.subplots()
plt.scatter(north['RD'],north['patent flows'],color='b',label='North countries')
texts = [ax.annotate(label,
                     xy=(north['RD'].iloc[i],north['patent flows'].iloc[i]),
                    xytext=(1,1),
                    textcoords='offset points',
                      # fontsize = 2,
                      color='b'
                    )
         for i,label in enumerate(north_countries)]
plt.scatter(south['RD'],south['patent flows'],color='r',label='South countries')
texts = [ax.annotate(label,
                     xy=(south['RD'].iloc[i],south['patent flows'].iloc[i]),
                    xytext=(1,1),
                    textcoords='offset points',
                      # fontsize = 2,
                      color='r'
                    )
         for i,label in enumerate(south_countries)]
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel('RD (Mio. $)')
ax.set_ylabel('# of international patent families as origin')
plt.legend()
plt.show()
