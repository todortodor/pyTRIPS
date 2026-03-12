#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:56:39 2026

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var, var_with_entry_costs
from solver_funcs import fixed_point_solver, fixed_point_solver_with_entry_costs, dyn_fixed_point_solver
from solver_funcs import dyn_fixed_point_solver_exog_lr, dyn_fixed_point_solver_exog_patent_thresholds, dyn_fixed_point_solver_exog_lr_and_patent_thresholds
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
import scienceplots
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#0e6655','#e8ba02']+list(Dark2[8])
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
gravity_patents_path = 'Gravity_patents/'

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
output_name = 'draft_v14'

presentation_version_path = output_path+output_name+'/'
try:
    os.mkdir(presentation_version_path)
except:
    pass

#%% Choose a run, load parameters, moments, solution

baseline = '2000'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'

variation_with_doubled_tau_in_pat_sect = '10.2'
variation_with_zero_trade_costs = '10.3'

variation_with_entry_costs = '11.02'
pre_trips_variation_with_entry_costs = '11.92'
# variation_with_zero_tariffs = '10.4'
# variation_with_ten_times_tariffs = '10.5'
variation_with_doubled_nu = '2.0'
variation_with_no_obsolescence = '12.0'

multi_sector_variation = '14.0'

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

#%% Create different folders to save plots

save_path = output_path+output_name+'/'+baseline+'_'+variation+'/'

try:
    os.mkdir(save_path)
except:
    pass

data_fact_path = save_path+'/data_facts/'
try:
    os.mkdir(data_fact_path)
except:
    pass

calibration_path = save_path+'/baseline_calibration/'
try:
    os.mkdir(calibration_path)
except:
    pass

sensitivity_path = save_path+'/sensitivity/'
try:
    os.mkdir(sensitivity_path)
except:
    pass

counterfactual_plots_path = save_path+'/counterfactuals/'
try:
    os.mkdir(counterfactual_plots_path)
except:
    pass

pre_TRIPS_plots_path = save_path+'/pre_TRIPS/'
try:
    os.mkdir(pre_TRIPS_plots_path)
except:
    pass

nash_coop_path = save_path+'/nash_and_coop_eq/'
try:
    os.mkdir(nash_coop_path)
except:
    pass

partial_equilibria_path = save_path+'/partial_equilibria/'
try:
    os.mkdir(partial_equilibria_path)
except:
    pass

post_trips_path = save_path+'trips-implementation/'
try:
    os.mkdir(post_trips_path)
except:
    pass

# doubled_trade_costs_path = save_path+'doubled-trade-costs-optima/'
# try:
#     os.mkdir(doubled_trade_costs_path)
# except:
#     pass

no_trade_costs_path = save_path+'no-trade-costs-nor-tariffs-optima/'
try:
    os.mkdir(no_trade_costs_path)
except:
    pass

no_obsolescence_path = save_path+'no-obsolescence/'
try:
    os.mkdir(no_obsolescence_path)
except:
    pass

doubled_nu_path = save_path+'doubled-nu/'
try:
    os.mkdir(doubled_nu_path)
except:
    pass

counterfactuals_doubled_nu_tau_path = save_path+'counterfactuals_with_doubled_nu_or_no_tau/'
try:
    os.mkdir(counterfactuals_doubled_nu_tau_path)
except:
    pass

dyn_save_path = save_path+'dynamics/'
try:
    os.mkdir(dyn_save_path)
except:
    pass

solve_to_join_pat_club_save_path = save_path+'solve_to_join_pat_club/'
try:
    os.mkdir(solve_to_join_pat_club_save_path)
except:
    pass

robustness_checks_path = save_path+'robustness_checks/'
try:
    os.mkdir(robustness_checks_path)
except:
    pass

with_entry_costs_path = save_path+'entry_costs/'
try:
    os.mkdir(with_entry_costs_path)
except:
    pass

gravity_patents_output_path = save_path+'gravity_patents/'
try:
    os.mkdir(gravity_patents_output_path)
except:
    pass

multi_sector_path = save_path+'multi_sector/'
try:
    os.mkdir(multi_sector_path)
except:
    pass


#%% Check that the US deviates in Nash for doubled trade costs

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')
_, sol_pre = fixed_point_solver(p_pre,context = 'counterfactual',x0=p_pre.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_pre.scale_P(p_pre)
sol_pre.compute_non_solver_quantities(p_pre)

p_nash = p_pre.copy()
p_nash.delta[:,1] = 12.0

sol, dyn_sol_nash = dyn_fixed_point_solver(p_nash, sol_init=sol_pre,Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=False,
                        damping = 60,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
dyn_sol_nash.compute_non_solver_quantities(p_nash)
dyn_sol_nash.sol_fin.compute_consumption_equivalent_welfare(p_nash,sol_pre)
dyn_sol_nash.sol_fin.compute_world_welfare_changes(p_nash,sol_pre)

p_nash_dev = p_pre.copy()
p_nash_dev.delta[:,1] = 12.0
p_nash_dev.delta[0,1] = 0.01

sol, dyn_sol_nash_dev = dyn_fixed_point_solver(p_nash_dev, sol_init=sol_pre,Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=False,
                        damping = 60,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
dyn_sol_nash_dev.compute_non_solver_quantities(p_nash_dev)
dyn_sol_nash_dev.sol_fin.compute_consumption_equivalent_welfare(p_nash_dev,sol_pre)
dyn_sol_nash_dev.sol_fin.compute_world_welfare_changes(p_nash_dev,sol_pre)

df = pd.DataFrame(columns = ['welfare_US'])
df.loc['no protection','welfare_US'] = dyn_sol_nash.cons_eq_welfare[0]
df.loc['full protection','welfare_US'] = dyn_sol_nash_dev.cons_eq_welfare[0]

df.to_csv(counterfactuals_doubled_nu_tau_path+'check_US_dev_Nash_doubled_tau_pat_sect.csv',float_format='%.5f')