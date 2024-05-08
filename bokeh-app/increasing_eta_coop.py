#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:07:59 2023

@author: slepot
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq, find_nash_eq
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
from bokeh.palettes import Category10, Dark2
import time
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

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

baseline = '1300'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'

baseline_pre_trips_full_variation = baseline
# pre_trips_full_variation = '3.1'

output_path = 'output/'
output_name = 'draft_NBER'

save_path = output_path+output_name+'/'+baseline+'_'+variation+'/'

try:
    os.mkdir(save_path)
except:
    pass

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

save_formats = ['eps','png','pdf']

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

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
if __name__ == '__main__':
    
    p = p_baseline.copy()
    # coop = 'pop_weighted'
    # for coop in ['pop_weighted','negishi']:
    for coop in ['pop_weighted']:
        for i,country in enumerate(p_baseline.countries):
            
            if country in ['CHN','IND','ZAF','RUS']:
                print(country)
                
                dynamics = True
                # lb = p_baseline.eta[:,1].min()/10
                lb = p_baseline.eta[i,1]
                ub = p_baseline.eta[:,1].max()*6
                # 0.005403032289568117 0.005246461012418196 0.005403032289568117 values for negishi MEX to restart
                # lb = p_baseline.eta[i,1]/10
                # ub = p_baseline.eta[i,1]/2
                
                # lb = p_baseline.eta[i,1]
                # ub = p_baseline.eta[i,1]*200
                it = 0
                
                lb_delta = 0.01
                ub_delta = 12
                
                df = pd.DataFrame()
                # df = pd.read_csv('/Users/slepot/Documents/taff/pyTRIPS/bokeh-app/solve_for_eta_to_join_pat_club/baseline_1030/pop_weighted_MEX.csv')
            
                while (ub-lb)/lb>1e-2:
                    it = it+1
                    x = (ub+lb)/2
                    p = p_baseline.copy()
                    p.eta[i,1] = x
                    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                                    context = 'counterfactual',
                                            cobweb_anim=False,tol =1e-14,
                                            accelerate=False,
                                            accelerate_when_stable=True,
                                            cobweb_qty='phi',
                                            plot_convergence=False,
                                            plot_cobweb=False,
                                            # plot_live=True,
                                            safe_convergence=0.001,
                                            disp_summary=False,
                                            damping = 500,
                                            max_count = 1e4,
                                            accel_memory = 50, 
                                            accel_type1=True, 
                                            accel_regularization=1e-10,
                                            accel_relaxation=0.5, 
                                            accel_safeguard_factor=1, 
                                            accel_max_weight_norm=1e6,
                                            damping_post_acceleration=5
                                            ) 
                    sol_c.scale_P(p)
                    sol_c.compute_non_solver_quantities(p)
                    print(lb,ub,x)
                    p.guess = sol.x 
                    p_opti, sol_opti = find_coop_eq(p,coop,
                                      lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                                        # solver_options=None,
                                      tol=1e-6,
                                      custom_dyn_sol_options = None,
                                        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                                                accelerate=False,
                                                                accelerate_when_stable=True,
                                                                cobweb_qty='phi',
                                                                plot_convergence=False,
                                                                plot_cobweb=False,
                                                                safe_convergence=0.001,
                                                                disp_summary=False,
                                                                damping = 50,
                                                                max_count = 1e4,
                                                                accel_memory = 50, 
                                                                accel_type1=True, 
                                                                accel_regularization=1e-10,
                                                                accel_relaxation=0.5, 
                                                                accel_safeguard_factor=1, 
                                                                accel_max_weight_norm=1e6,
                                                                damping_post_acceleration=20
                                                                ),
                                      custom_weights=None,
                                      max_workers=12,parallel=False)
                    if dynamics:
                        p_opti, sol_opti = find_coop_eq(p,coop,
                                         lb_delta=lb_delta,ub_delta=ub_delta,dynamics=True,
                                         tol=1e-6,
                                            static_eq_deltas = p_opti.delta[...,1],
                                            #   custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-12,
                                            #       accelerate=False,
                                            #       accelerate_when_stable=False,
                                            #       cobweb_qty='l_R',
                                            #       plot_convergence=False,
                                            #       plot_cobweb=False,
                                            #       plot_live = False,
                                            #       safe_convergence=1e-8,
                                            #       disp_summary=False,
                                            #       damping = 500,
                                            #       max_count = 1000000,
                                            #       accel_memory =5, 
                                            #       accel_type1=True, 
                                            #       accel_regularization=1e-10,
                                            #       accel_relaxation=1, 
                                            #       accel_safeguard_factor=1, 
                                            #       accel_max_weight_norm=1e6,
                                            #       damping_post_acceleration=10),
                                            # solver_options = dict(cobweb_anim=False,tol =1e-12,
                                            #                         accelerate=False,
                                            #                         accelerate_when_stable=True,
                                            #                         cobweb_qty='phi',
                                            #                         plot_convergence=False,
                                            #                         plot_cobweb=False,
                                            #                         safe_convergence=0.001,
                                            #                         disp_summary=False,
                                            #                         damping = 50,
                                            #                         max_count = 1e6,
                                            #                         accel_memory = 50, 
                                            #                         accel_type1=True, 
                                            #                         accel_regularization=1e-10,
                                            #                         accel_relaxation=0.5, 
                                            #                         accel_safeguard_factor=1, 
                                            #                         accel_max_weight_norm=1e6,
                                            #                         damping_post_acceleration=20
                                            #                         ),
                                            custom_dyn_sol_options = None,
                                            solver_options=None,
                                         custom_weights=None,max_workers=12,displays=True,
                                         parallel=False)
                        if p_opti.delta[i,1]<p_baseline.delta[i,1]:
                            ub = x
                            # df.loc[it,'eta_china'] = x
                            df.loc[it,f'eta_{country}'] = x
                            for j,c in enumerate(p_baseline.countries):
                                df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                        else:
                            lb = x
                            # df.loc[it,'eta_china'] = x
                            df.loc[it,f'eta_{country}'] = x
                            for j,c in enumerate(p_baseline.countries):
                                df.loc[it,'delta_opti_'+c] = p_opti.delta[j,1]
                        print(x,lb,ub)
                        print(df)
                    # plt.scatter(df['eta_china'],df['delta_opti_CHN'])
                    # plt.scatter(df['eta_china'].iloc[-1],df['delta_opti_CHN'].iloc[-1],
                    #             color='red')
                    df.to_csv(f'solve_for_eta_to_join_pat_club/baseline_{baseline}/{coop}_{country}.csv')
                    # plt.scatter(df[f'eta_{country}'],df[f'delta_opti_{country}'])
                    # plt.scatter(df[f'eta_{country}'].iloc[-1],df[f'delta_opti_{country}'].iloc[-1],
                    #             color='red')
                    # plt.show()

#%%

# df_chn = pd.read_csv('solve_for_eta_to_join_pat_club/baseline_1030/pop_weighted_CHN.csv')
# df_ind = pd.read_csv('solve_for_eta_to_join_pat_club/baseline_1030/pop_weighted_stat_IND.csv')
# # df_chn = pd.read_csv('solve_for_eta_to_join_pat_club/baseline_1030/negishi_CHN.csv')
# # df_ind = pd.read_csv('solve_for_eta_to_join_pat_club/baseline_1030/negishi_IND.csv')
# # df_bra = pd.read_csv('solve_for_eta_to_join_pat_club/baseline_1030/stat_BRA.csv')

# df = pd.DataFrame(index = p_baseline.countries)
# df['baseline eta'] = p_baseline.eta[:,1]

# df.loc['CHN','eta for which delta_opti = delta_baseline'] = df_chn['eta_CHN'].iloc[-1]
# df.loc['IND','eta for which delta_opti = delta_baseline'] = df_ind['eta_IND'].iloc[-1]
# # df.loc['BRA','eta for which delta_opti = delta_baseline'] = df_bra['eta_BRA'].iloc[-1]

# df['as ratio to baseline'] = df['eta for which delta_opti = delta_baseline']/df['baseline eta']
# df['as ratio to baseline US'] = df['eta for which delta_opti = delta_baseline']/df.loc['USA','baseline eta']

## df.to_csv('/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/simon_version/code/misc/discussion_material_02_10_meeting/china_india_ex.csv')

#%%

# markers = {'pop_weighted':'o',
#       'negishi':'^'}
# label_coop = {'pop_weighted':'Equal',
#       'negishi':'Negishi'}

# run_countries = []

# fig,ax = plt.subplots()

# for i, country in enumerate(p_baseline.countries):
    
#     for j,coop in enumerate(['pop_weighted','negishi']):
#         try:
#             df = pd.read_csv(f'solve_for_eta_to_join_pat_club/baseline_1030/{coop}_{country}.csv')
#             # ax.scatter([p_baseline.T[i,1].mean()],[df[f'eta_{country}'].iloc[-1]],
#             ax.scatter([country],[df[f'eta_{country}'].iloc[-1]],
#                         # label = f'{country} {label_coop[coop]}',
#                         marker = markers[coop],
#                         color = Category18[i])
#             ax.errorbar([country],[df[f'eta_{country}'].iloc[-1]], yerr = [np.abs(df[f'eta_{country}'].iloc[-1] - df[f'eta_{country}'].iloc[-2])])
#             run_countries.append(country)
#             print(coop,country,df[f'eta_{country}'].iloc[-1])
#         except:
#             pass
#     if country in run_countries:
#         print(run_countries)
#         ax.scatter([country],[p_baseline.eta[i,1]],
#                     # label = f'{country} baseline',
#                     marker = '*',
#                     color = Category18[i])
# ax.scatter([],[],marker = 'o', label = 'Equal')
# ax.scatter([],[],marker = '^', label = 'Negishi')
# ax.scatter([],[],marker = '*', label = 'Baseline')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# plt.axhline(y=p_baseline.eta[0,1],color='grey',label='Baseline USA')
# ax.set_ylabel('Eta')
# plt.legend()
# plt.show()

#%%

# country = 'IND'

# # x_0 = [0.01,12.00,0.01,12.00,12.00, 0.01,0.01,0.01,12.00,12.00,12.00]
# x_0 = [0.01,12.00,0.01,12.00,12.00,12.00,0.01,0.01,12.00,12.00,12.00]

# p_cf = p_baseline.copy()
# p_cf.eta[p_baseline.countries.index(country),1
#          ] = p_cf.eta[0,1]#/10
# t1 = time.perf_counter()
# custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 10,
#                         max_count = 1e4,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=2
#                         )
# p_coop, sol_coop = find_coop_eq(p_cf,aggregation_method = 'pop_weighted',
#                  lb_delta=0.01,ub_delta=12,dynamics=False,
#                   solver_options=custom_sol_options,tol=1e-15,
#                  # solver_options=None,tol=1e-15,
#                  static_eq_deltas = None,custom_weights=None,
#                   custom_x0 = x_0)
#                  # custom_x0 = None)
# print(time.perf_counter() - t1)
# print(pd.DataFrame(index=p_baseline.countries,
#                    data = p_coop.delta[...,1]))
# print(sol_coop.cons_eq_negishi_welfare_change)

# #%%
# from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq, find_nash_eq

# p_opti,sol_opti = find_coop_eq(p_baseline,aggregation_method = 'negishi',
#                  lb_delta=0.01,ub_delta=12,dynamics=False,
#                  solver_options=None,tol=1e-15,
#                  static_eq_deltas = None,custom_weights=None,
#                  custom_x0 = None)
# print(pd.DataFrame(index=p_baseline.countries,
#                    data = p_opti.delta[...,1]))
# print(sol_opti.cons_eq_negishi_welfare_change)

# #%%
# fig,ax = plt.subplots()
# ax2 = ax.twinx()

# years_time = [y for y in range(1990,2019)]
# # eta_time = np.concatenate([pd.read_csv(f'calibration_results_matched_economy/baseline_{baseline}_variations/1.{i}/eta.csv'
# #                                   ,index_col=0,header=None
# #                                   ).values.squeeze().reshape(p_baseline.N,p_baseline.S)[:,1][:,None]
# #                       for i in range(29)],axis=1)
# eta_time_partial = np.concatenate([pd.read_csv(f'calibration_results_matched_economy/baseline_{baseline}_variations/11.{i}/eta.csv'
#                                   ,index_col=0,header=None
#                                   ).values.squeeze().reshape(p_baseline.N,p_baseline.S)[:,1][:,None]
#                       for i in range(29)],axis=1)   
# delta_coop = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
#                                                      'variation',
#                                                      'aggregation_method'],keep='last')

# def sort_key(series):
#     res = series.str.split('.',expand=True)[1]
#     res = res.astype(int)
#     return res

# eta_time_partial[3,3] = eta_time_partial[3,2]
# eta_time_partial[3,4] = eta_time_partial[3,2]

# delta_coop_time = delta_coop.loc[(delta_coop.baseline == int(baseline))
#                                   & (delta_coop.variation.str.startswith('20'))
#                                   ].sort_values('variation',key=sort_key)[['aggregation_method','CHN']]

# # # delta_coop_time['variation'] = delta_coop_time['variation'].astype(float)

# # delta_coop_time[delta_coop_time>2] = 12


# # plt.plot(years_time,eta_time[3,:],label='Calibrated eta of China')
# ax.plot(years_time,eta_time_partial[3,:],label='Eta calibrated')
# ax2.plot([y for y in range(1990,1990+50)][-25:],delta_coop_time.loc[delta_coop_time.aggregation_method == 'pop_weighted']['CHN'],
#           label='Delta in\nEqual eq',color=sns.color_palette()[1])
# ax2.plot([y for y in range(1990,1990+50)][-25:],delta_coop_time.loc[delta_coop_time.aggregation_method == 'negishi']['CHN'],
#           label='Delta in\nNegishi eq',color=sns.color_palette()[2])
# ax2.set_yscale('log')
# # plt.plot(np.arange(0,29),eta_time[3,:])
# # fit = np.polyfit(x=np.arange(0,29),y=eta_time_partial[3,:][4:],deg=2)
# fit = np.polyfit(x=np.arange(0,29),y=eta_time_partial[3,:],deg=2)
# ax.plot([y for y in range(1990,1990+50)],np.polyval(fit,np.arange(0,50)),color='red'
#          ,label='Eta Extrapolated')
# # plt.axhline(y=0.00567866431997453,color='orange',ls='--',label='Equal weights cooperative breakout eta')
# # plt.axhline(y=0.009737144083241169,color='k',ls='--',label='Negishi weights cooperative breakout eta')
# ax.legend(loc='upper left')
# ax2.legend(loc='upper right')
# ax.set_ylabel('Eta')
# ax2.grid(None)
# ax.set_xticks([y for y in range(1990,1990+50)][::2])
# ax.tick_params(axis='x', rotation=45,pad=-10,labelsize=20)
# plt.title('Patent protection in China in the Equal cooperative equilibrium for increasing eta')

# temp_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/misc/discussion_material_18_09_meeting/'
# # plt.savefig(temp_path+'increasing_china_eta_static.png')

# plt.show()

# #%%

# from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq, find_nash_eq

# for i,eta in enumerate(
#         np.polyval(np.polyfit(x=np.arange(0,29),y=eta_time_partial[3,:],deg=2),np.arange(0,50))[25:]):
#     p = p_baseline.copy()
#     p.eta[3,1] = eta
#     _, sol_c = fixed_point_solver(p,context = 'counterfactual',
#                             x0=p.guess,
#                             cobweb_anim=False,tol =1e-14,
#                             accelerate=False,
#                             accelerate_when_stable=True,
#                             cobweb_qty='phi',
#                             plot_convergence=False,
#                             plot_cobweb=False,
#                             safe_convergence=0.001,
#                             disp_summary=True,
#                             damping = 10,
#                             max_count = 3e3,
#                             accel_memory = 50, 
#                             accel_type1=True, 
#                             accel_regularization=1e-10,
#                             accel_relaxation=0.5, 
#                             accel_safeguard_factor=1, 
#                             accel_max_weight_norm=1e6,
#                             damping_post_acceleration=5
#                             )
#     sol_c.scale_P(p)
#     sol_c.compute_non_solver_quantities(p)

#     p.guess = sol_c.vector_from_var()
    
#     local_path = 'calibration_results_matched_economy/baseline_'+str(baseline)+'_variations/'
#     run_number = 20
    
#     p.write_params(local_path+f'{run_number}.{i}/')
    

# #%%

# years_time = [y for y in range(1990,2019)]
# eta_time = np.concatenate([pd.read_csv(f'calibration_results_matched_economy/baseline_{baseline}_variations/11.{i}/eta.csv'
#                                   ,index_col=0,header=None
#                                   ).values.squeeze().reshape(p_baseline.N,p_baseline.S)[:,1][:,None]
#                       for i in range(29)],axis=1)
# plt.plot(years_time,eta_time[5,:],label='Calibrated eta of India')
# # plt.plot(np.arange(0,29),eta_time[3,:])
# fit = np.polyfit(x=np.arange(0,29),y=eta_time[5,:],deg=3)
# plt.plot([y for y in range(1990,1990+29)],np.polyval(fit,np.arange(0,29)),color='red'
#          ,label='Extrapolation of calibrated eta of India')
# # plt.axhline(y=0.00567866431997453,color='orange',ls='--',label='Equal weights cooperative breakout eta')
# # plt.axhline(y=0.009737144083241169,color='k',ls='--',label='Negishi weights cooperative breakout eta')
# plt.legend()
# plt.ylabel('Eta')

# # print([y for y in range(1990,1990+50)][np.argmin(np.abs(np.polyval(fit,np.arange(0,50)) - 0.00567866431997453))])
# # print([y for y in range(1990,1990+50)][np.argmin(np.abs(np.polyval(fit,np.arange(0,50)) - 0.009737144083241169))])

