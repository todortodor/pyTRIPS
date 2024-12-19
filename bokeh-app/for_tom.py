#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:29:34 2024

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
Category18 = list(Category10[10])+['#0e6655','#e8ba02']+list(Dark2[8])

plt.style.use(['science','nature','no-latex'])
plt.style.use(['science','no-latex'])
import matplotlib.pyplot as plt 
plt.rcParams.update({"axes.grid" : True, 
                     "grid.color": "grey", 
                     'axes.axisbelow':True,
                     "grid.linewidth": 0.1, 
                     'legend.framealpha':1,
                     'legend.frameon':1,
                     'legend.edgecolor':'white',
                     'figure.dpi':288,
                     })
# mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

#%% setup path and stuff

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'
table_path = 'calibration_results_matched_economy/'

# save_formats = ['eps','png','pdf']
save_formats = ['pdf']

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India',
                  'ZAF': 'S. Africa','ROW':'Rest of the world'}

rich_countries = ['USA','JAP','EUR','CAN','KOR','ROW']

parameters_description = {
    'delta':'Patent protection',
    'g_0':'Growth rate of no patenting sector',
    'f_o':'Patent preparation cost',
    'f_e':'Patent application cost',
    'k':'Shape parameter of Pareto quality distribution',
    'nu':'technology diffusion rate',
    'theta':'Shape parameter of Frechet productivity distribution',
    'zeta':'Product obsolescence rate',
    'T':'Productivty',
    'eta':'RD efficiency'
    }

#%% create output folder

output_path = 'output/'
output_name = 'draft_v12'

presentation_version_path = output_path+output_name+'/'
try:
    os.mkdir(presentation_version_path)
except:
    pass

baseline = '1300'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'
# variation_with_doubled_tau_in_pat_sect = '10.2'
variation_with_zero_trade_costs = '10.3'
# variation_with_zero_tariffs = '10.4'
# variation_with_ten_times_tariffs = '10.5'
variation_with_doubled_nu = '2.0'

variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low TO',
    '99.1':'High TO',
    '99.2':'Low TE',
    '99.3':'High TE',
    '99.4':'Low KM',
    '99.5':'High KM',
    '99.6':'Low Sigma',
    '99.7':'High Sigma',
    '99.8':'Low Kappa',
    '99.9':'High Kappa',
    '99.10':'Low Growth',
    '99.11':'High Growth',
    '99.12':'Low rho',
    '99.13':'High rho',
    '99.14':'Low UUPCOST',
    '99.15':'High UUPCOST',
    }

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

#%% Comparing trade flows with patent flows

fig,ax = plt.subplots()

tflow_shares = m_baseline.ccs_moments.query('destination_code!=origin_code'
                                    ).xs(1,level=2)
tflow_shares = tflow_shares/tflow_shares.sum()
pflow_shares = m_baseline.cc_moments.query('destination_code!=origin_code'
                                    )
pflow_shares = pflow_shares/pflow_shares.sum()

# ax.scatter(pflow_shares.values.ravel(),
#             m_baseline.SPFLOW.ravel(),
#             label = 'International patent shares: model')
ax.scatter(pflow_shares.values.ravel(),
            tflow_shares.values.ravel(),
            label = 'International trade shares',marker='^')

ax.set_xlabel('International patent shares')
ax.set_ylabel('International trade shares')

ax.set_xscale('log')
ax.set_yscale('log')

# plt.legend()
    
ax.plot([pflow_shares.values.ravel().min(),pflow_shares.values.ravel().max()],
        [pflow_shares.values.ravel().min(),pflow_shares.values.ravel().max()],
        ls='--',color='grey')

labels = m_baseline.idx['SPFLOW'].to_series().transform(','.join).astype('str')

y = tflow_shares.values.ravel()
x = pflow_shares.values.ravel()

texts = [ax.annotate(label,
                         xy=(x[i],y[i]),
                        xytext=(1,1),
                        textcoords='offset points',
                          fontsize = 3
                        )
             for i,label in enumerate(labels)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))

plt.savefig('../misc/trade_vs_patent_flows_for_tom_with_labels.pdf',format='pdf')

plt.show()
#%% Comparing trade flows with patent flows

# fig,ax = plt.subplots()

# tflow_shares = m_baseline.ccs_moments.query('destination_code!=origin_code'
#                                     ).xs(1,level=2)
# tflow_shares = tflow_shares/tflow_shares.sum()
# pflow_shares = m_baseline.cc_moments.query('destination_code!=origin_code'
#                                     )
# pflow_shares = pflow_shares/pflow_shares.sum()

# # ax.scatter(pflow_shares.values.ravel(),
# #             m_baseline.SPFLOW.ravel(),
# #             label = 'International patent shares: model')
# ax.scatter(tflow_shares.values.ravel(),
#            pflow_shares.values.ravel(), 
#             label = 'International patent shares',marker='^')

# ax.set_ylabel('International patent shares')
# ax.set_xlabel('International trade shares')

# ax.set_xscale('log')
# ax.set_yscale('log')

# # plt.legend()
    
# ax.plot([tflow_shares.values.ravel().min(),tflow_shares.values.ravel().max()],
#         [tflow_shares.values.ravel().min(),tflow_shares.values.ravel().max()],
#         ls='--',color='grey')



# labels = m_baseline.idx['SPFLOW'].to_series().transform(','.join).astype('str')

# x = tflow_shares.values.ravel()
# y = pflow_shares.values.ravel()

# texts = [ax.annotate(label,
#                          xy=(x[i],y[i]),
#                         xytext=(1,1),
#                         textcoords='offset points',
#                           fontsize = 3
#                         )
#              for i,label in enumerate(labels)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

# plt.savefig('../misc/patent_vs_trade_flows_for_tom_with_labels.pdf',format='pdf')

# plt.show()
