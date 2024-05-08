#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:21:07 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import time
import os
import seaborn as sns
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver, compute_deriv_welfare_to_patent_protec_US, compute_deriv_growth_to_patent_protec_US
from data_funcs import write_calibration_results
from functools import reduce
from tqdm import tqdm

#%% define baseline and conditions of sensitivity analysis

baseline = '1300'
baseline_path = 'calibration_results_matched_economy/'+baseline+'/'
p_baseline = parameters()
p_baseline.load_run(baseline_path)
m_baseline = moments()
m_baseline.load_run(baseline_path)
sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                        context='calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 5e4,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )

sol_baseline = var.var_from_vector(sol.x, p_baseline,context = 'calibration')    
sol_baseline.scale_P(p_baseline)
# sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

m_baseline.compute_moments(sol_baseline, p_baseline)

moments_to_change = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRGDP', 'SINNOVPATUS',
  'TO', 'SPFLOW', 'UUPCOST', 'DOMPATINUS', 'TE']
# moments_to_change = ['SPFLOW', 'UUPCOST', 'DOMPATINUS', 'TE']
parameters_to_change = ['rho','kappa','sigma','theta','gamma']
# parameters_to_change = ['rho','kappa','sigma']
# parameters_to_change = ['theta','gamma']

weights_to_change = m_baseline.list_of_moments

dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'

parent_moment_result_path = 'calibration_results_matched_economy/baseline_'+baseline+'_all_targets_variations_20/'

def make_dirs(list_of_paths):
    for path in list_of_paths:
        try:
            os.mkdir(path)
        except:
            pass

def GetSpacedElements(array, numElems = 13):
    idx = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    out = array[idx]
    return out, idx

#%% make dirs

make_dirs([parent_moment_result_path,
            ])

#%% run alternative calibrations for different targets

def make_alternative_calib(m_baseline,p_baseline,qty_to_change,target,result_path,mom_or_par='mom'):
    m = m_baseline.copy()
    p = p_baseline.copy()
    if mom_or_par=='mom':
        setattr(m,qty_to_change+'_target',target)
    if mom_or_par=='par':
        setattr(p,qty_to_change,target)
    bounds = p.make_parameters_bounds()
    start_time = time.perf_counter()
    hist = history(*tuple(m.list_of_moments+['objective']))
    cond = True
    iterations = 0
    max_iter = 8
    while cond:
        if iterations < max_iter-3:
            test_ls = optimize.least_squares(fun = calibration_func,    
                                    x0 = p.make_p_vector(), 
                                    args = (p,m,p.guess,hist,start_time), 
                                    bounds = bounds,
                                    # method= 'dogbox',
                                    # loss='arctan',
                                    # jac='3-point',
                                    max_nfev=1e8,
                                    # ftol=1e-14, 
                                    xtol=1e-10, 
                                    # gtol=1e-14,
                                    # f_scale=scale,
                                    verbose = 2)
        else:
            test_ls = optimize.least_squares(fun = calibration_func,    
                                    x0 = p.make_p_vector(), 
                                    args = (p,m,p.guess,hist,start_time), 
                                    bounds = bounds,
                                    # method= 'dogbox',
                                    # loss='arctan',
                                    # jac='3-point',
                                    max_nfev=1e8,
                                    # ftol=1e-14, 
                                    xtol=1e-16, 
                                    # gtol=1e-14,
                                    # f_scale=scale,
                                    verbose = 2)
        cond = iterations < max_iter
        iterations += 1
    p_sol = p.copy()
    p_sol.update_parameters(test_ls.x)
    sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                    context = 'calibration',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=True,
                            plot_cobweb=True,
                            safe_convergence=0.001,
                            disp_summary=True,
                            damping = 10,
                            max_count = 3e3,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_c.scale_P(p_sol)
    sol_c.compute_non_solver_quantities(p_sol) 
    p_sol.guess = sol.x
    p_sol.tau = sol_c.tau
    
    m.compute_moments(sol_c,p_sol)
    m.compute_moments_deviations()
    
    p_sol.write_params(result_path)
    m.write_moments(result_path)

#%%

for par in parameters_to_change:
    if par in ['kappa','rho','gamma']:
        target = getattr(p_baseline,par)*1.20
        print(par)
        result_path = parent_moment_result_path+par+'/'
        make_alternative_calib(m_baseline,p_baseline,par,target,result_path,mom_or_par='par')
    if par in ['sigma']:
        target = getattr(p_baseline,par).copy()
        target[1] = target[1]*1.2
        print(par)
        result_path = parent_moment_result_path+par+'/'
        make_alternative_calib(m_baseline,p_baseline,par,target,result_path,mom_or_par='par')
    if par in ['theta']:
        target = getattr(p_baseline,par).copy()
        target[0] = target[0]*1.2
        print(par)
        result_path = parent_moment_result_path+par+'/'
        make_alternative_calib(m_baseline,p_baseline,par,target,result_path,mom_or_par='par')

for mom in moments_to_change:
    if mom in ['KM','UUPCOST','SINNOVPATUS','TO','GROWTH',
                          'DOMPATINUS','TE','OUT','GPDIFF']:
        target = getattr(m_baseline,mom+'_target')*1.20
        print(mom)
        result_path = parent_moment_result_path+mom+'/'
        make_alternative_calib(m_baseline,p_baseline,mom,target,result_path)
    if mom in ['RD','RP','SRGDP']:
        for c,country in enumerate(p_baseline.countries):
            target = getattr(m_baseline,mom+'_target').copy()
            target[c] = target[c]*1.20
            if mom == 'RP':
                target = target/target[0]
            if mom == 'SRGDP':
                target = target/target.sum()
            print(mom)
            result_path = parent_moment_result_path+mom+'_'+country+'/'
            make_alternative_calib(m_baseline,p_baseline,mom,target,result_path)
    if mom in ['SPFLOW']:
        for d,destination in enumerate(p_baseline.countries):
            target = m_baseline.cc_moments.copy()
            # target.loc[d+1,'patent flows'] = target.loc[d+1,'patent flows']*1.05
            for o,origin in enumerate(p_baseline.countries):
                target.loc[(d+1,o+1),'patent flows'] = target.loc[(d+1,o+1),'patent flows']*1.20
            target = target.query("destination_code != origin_code")['patent flows'].values
            target = target.reshape((p_baseline.N,p_baseline.N-1))
            target = target/target.sum()
            print(mom)
            result_path = parent_moment_result_path+mom+'_destination_'+destination+'/'
            make_alternative_calib(m_baseline,p_baseline,mom,target,result_path)
        for o,origin in enumerate(p_baseline.countries):
            target = m_baseline.cc_moments.copy()
            for d,destination in enumerate(p_baseline.countries):
                target.loc[(d+1,o+1),'patent flows'] = target.loc[(d+1,o+1),'patent flows']*1.20
            target = target.query("destination_code != origin_code")['patent flows'].values
            target = target.reshape((p_baseline.N,p_baseline.N-1))
            target = target/target.sum()
            print(mom)
            result_path = parent_moment_result_path+mom+'_origin_'+origin+'/'
            make_alternative_calib(m_baseline,p_baseline,mom,target,result_path)
    

#%% Gather changes in one dataframe

df = pd.DataFrame()

moments_to_change = ['theta','gamma','kappa','rho','sigma','KM','TE','TO','UUPCOST','OUT','GROWTH','GPDIFF','SINNOVPATUS', 
   'DOMPATINUS', 'SRGDP', 'RD', 'RP', 'SPFLOW']
# moments_to_change = ['GPDIFF', 'GROWTH', 'KM', 'SINNOVPATUS',
#   'UUPCOST', 'DOMPATINUS', 'SRGDP', 'RD', 'RP', 'SPFLOW']
# moments_to_change = ['RD']


def fill_row(df,parent_moment_result_path,mom_idx,p_baseline):
    p = parameters()
    p.load_run(parent_moment_result_path+mom_idx+'/')
    df.loc[mom_idx,'k'] = (p.k/p_baseline.k-1)/0.20
    df.loc[mom_idx,'g_0'] = (p.g_0/p_baseline.g_0-1)/0.20
    for par in ['fe','fo','zeta','nu','theta']:
        df.loc[mom_idx,par] = (getattr(p,par)[1]/getattr(p_baseline,par)[1]-1)/0.20
    for par in ['delta','eta']:
        for c,country in enumerate(p_baseline.countries):
            df.loc[mom_idx,par+' '+country] = (getattr(p,par)[c,1]/getattr(p_baseline,par)[c,1]-1)/0.20
    for par in ['T']:
        for s,sector in enumerate(p_baseline.sectors):
            for c,country in enumerate(p_baseline.countries):
                df.loc[mom_idx,par+' '+sector+' '+country
                       ] = (getattr(p,par)[c,s]**(1/p.theta[s])
                            /getattr(p_baseline,par)[c,s]**(1/p_baseline.theta[s])-1)/0.20

for mom in moments_to_change:
    print(mom)
    if mom in ['KM','UUPCOST','SINNOVPATUS','TO','GROWTH',
                          'DOMPATINUS','TE','OUT','GPDIFF','kappa','rho','sigma','theta','gamma']:
        mom_idx = mom
        fill_row(df,parent_moment_result_path,mom_idx,p_baseline)
    if mom in ['RD','RP','SRGDP']:
        for c,country in enumerate(p_baseline.countries):
            mom_idx = mom+'_'+country
            fill_row(df,parent_moment_result_path,mom_idx,p_baseline)
    if mom in ['RD']:
        for c,country in enumerate(p_baseline.countries):
            mom_idx = mom+'_'+country
            fill_row(df,parent_moment_result_path,mom_idx,p_baseline)
    if mom in ['SPFLOW']:
        for d,destination in enumerate(p_baseline.countries):
            mom_idx = mom+'_destination_'+destination
            fill_row(df,parent_moment_result_path,mom_idx,p_baseline)
        for o,origin in enumerate(p_baseline.countries):
            mom_idx = mom+'_origin_'+origin
            fill_row(df,parent_moment_result_path,mom_idx,p_baseline)

table_path = 'calibration_results_matched_economy/baseline_1300_sensitivity_tables/'
try:
    os.mkdir(table_path)
except:
    pass
df.to_csv(table_path+'all_sensitivity_table_20.csv')

#%% try for graphs

import pandas as pd
import numpy as np
from sankeyflow import Sankey
from bokeh.palettes import Category10, Dark2
Category20 = Category10[10]+Dark2[8]

nodes_mom = []
nodes_params = []
flows = []
# df = df.sort_index()
df_temp = df.copy()
df_temp = df_temp[[c for c in df_temp.columns if any([c.startswith(p) for p in ['eta','delta','T']])]]
df_temp = df_temp.loc[[c for c in df_temp.index if any([c.startswith(m) for m in ['RD','RP','SPFLOW']])]]

# colors = sns.color_palette("hls", 20).as_hex()

# country_colors = {country:colors[i] for i,country in enumerate(p_baseline.countries)}
# countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
#                    'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
#                   'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}
# no_country_colors = [c for c in colors if c not in country_colors.values()]

# mapping_mom_dictionary = {}

# for country in p_baseline.countries:
#     for mom in ['RD','RP','SRGDP']:
#         if mom == 'RD':
#             mom_dic = {'name':,
#                        'color':}
#         mom

plt.figure(figsize=(15, 30), dpi=144)

for count,mom in enumerate(df_temp.index):
    # if mom not in ['TE','TO','GROWTH','OUT']:
    # if any([mom.startswith(m) for m in ['RD','RP','SPFLOW']]):
    # if mom not in []:
        # nodes_mom.append([mom,df.loc[mom].sum()])
        for param in df_temp.columns:
            # if any([param.startswith(p) for p in ['eta','delta','T']]):
            # if count == 0:
            #     nodes_params.append([param,df[param].sum()])
            # if np.abs(df.loc[mom,param]) == np.abs(df.loc[mom]).max() or np.abs(df.loc[mom,param])==np.abs(df[param]).max():
            if np.abs(df_temp.loc[mom,param]) == np.abs(df_temp.loc[mom]).max():
            #     flows.append([mom,param,1])
            # if np.abs(df_temp.loc[mom,param])==np.abs(df_temp[param]).max():
                # flows.append([mom,param,1])
                # flows.append([mom,param,np.abs(df.loc[mom,param])**4])
                # flows.append([mom,param,np.abs(df.loc[mom,param])**(1/2)])
                # flows.append([mom,param,np.abs(df.loc[mom,param])])
                # flows.append([mom,param,1])
                # flows.append([mom,param,np.abs(df.loc[mom,param])/df[param].abs().sum()])
                flows.append([mom,param,(np.abs(df_temp.loc[mom,param])/df_temp.loc[mom].abs().sum())**2])
            # else:
            #     flows.append([mom,param,0])

nodes = [nodes_mom, nodes_params]
        
# nodes = [
# [('A', 10)],
# [('B1', 4), ('B2', 5)],
# [('C', 3)]
# ]
        
# flows = [
#     ('A', 'B1', 4),
#     ('A', 'B2', 5),
#     ('B1', 'C', 1),
#     ('B2', 'C', 2, {'color': 'red'}),
# ] 

s = Sankey(
    flows=flows,
    flow_color_mode='source',
    # flow_color_mode='dest',
    # nodes=nodes,
    nodes=None,
    # label_format='{label},{value:,.0f}',
    # cmap=plt.cm.Pastel1,
    # flow_opts=dict(curvature=1),
)
# s._layout_tree(max_level=1)
s.draw()
plt.show()



#%% build max depndency dataframes (not really useful)

df_max_col = pd.DataFrame()
df_min_col = pd.DataFrame()

for col in df.columns:
    df_max_col.loc[col,'max_sensitivity'] = df[col].idxmax()
    df_min_col.loc[col,'max_sensitivity'] = df[col].idxmin()
    # df_max_col.loc[col,'max_or_min_sensitivity'] = df[col].abs().idxmax()

df_max_row = pd.DataFrame()

for idx in df.index:
    df_max_row.loc[idx,'max_sensitivity'] = df.T[idx].abs().idxmax()

#%%

table_path = 'calibration_results_matched_economy/baseline_1300_sensitivity_tables/'

try:
    os.mkdir(table_path)
except:
    pass

df.to_csv(table_path+'all_sensitivity_table_20.csv')
df_max_col.to_csv(table_path+'max_col_all_sensitivity_table_20_no_TE_TO_OUT.csv')
df.to_csv(table_path+'all_sensitivity_table_20_no_TE_TO_OUT.csv')
df_max_col.to_csv(table_path+'max_col_all_sensitivity_table_20_no_TE_TO_OUT.csv')
