#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:45:08 2023

@author: slepot
"""

import numpy as np
import pandas as pd
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver,dyn_fixed_point_solver,compute_deriv_welfare_to_patent_protec

baseline = '1030'
variation = 'baseline'

run_path = 'calibration_results_matched_economy/'+baseline+'/'

p_baseline = parameters()
p_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, 
                                   compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

dfs = {}
for country in p_baseline.countries:
    dfs[country] = pd.DataFrame(columns = ['tau_factor']+p_baseline.countries
                                          +['pop_weighted','negishi'])

for t,tau_factor in enumerate(np.linspace(0.5,2,76)):
    print(tau_factor)
    p = p_baseline.copy()
    p.tau[...,1] = p.tau[...,1]*tau_factor
    for j,_ in enumerate(p.countries):
        p.tau[j,j,:] = 1
    sol, sol_c = fixed_point_solver(p,x0=p.guess,tol=1e-14,
                                    context = 'counterfactual',
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=2e3,
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
    p.guess = sol_c.vector_from_var()
    
    for country in p.countries:
        deriv_welfare = compute_deriv_welfare_to_patent_protec(sol_c,p,
                                                country=country,
                                                dynamics=True)
        dfs[country].loc[t] = [tau_factor]+deriv_welfare
        print(country,dfs[country].T)
        dfs[country].to_csv(
            f'deriv_welfare_to_patent_protec_cf/tau_pat_sector/baseline_{baseline}/{country}')
        
for n,nu_factor in enumerate(np.linspace(0.5,2,76)):
    print(nu_factor)
    p = p_baseline.copy()
    p.nu[1] = p.nu[1]*nu_factor
    sol, sol_c = fixed_point_solver(p,x0=p.guess,tol=1e-14,
                                    context = 'counterfactual',
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=2e3,
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
    p.guess = sol_c.vector_from_var()
    
    for country in p.countries:
        deriv_welfare = compute_deriv_welfare_to_patent_protec(sol_c,p,
                                                country=country,
                                                dynamics=True)
        dfs[country].loc[n] = [nu_factor]+deriv_welfare
        print(country,dfs[country].T)
        dfs[country].to_csv(
            f'deriv_welfare_to_patent_protec_cf/nu/baseline_{baseline}/{country}')
        
#%%
import matplotlib.pyplot as plt
import pandas as pd

# country = 'USA'
# country2 = 'EUR'
qty = 'nu'

fig,axes = plt.subplots(4,3,dpi=288,figsize = (8,8))

for j,country in enumerate(p_baseline.countries):
    ax = axes.flat[j]
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_1030/{country}',
                     index_col=0)
    
    for country2 in p_baseline.countries:
        ax.plot(df[df.columns[0]],df[country2],label=country2)
    ax.plot(df[df.columns[0]],df['pop_weighted'],label='Equal',color='grey',ls='--')
    ax.plot(df[df.columns[0]],df['negishi'],label='Negishi',color='grey',ls=':')
    ax.set_title(country)
    # ax.legend()
    #.rolling(2).mean()
    ax.set_xscale('log')
    ax.set_xlabel(r'Proportional change in $\nu$')
    ax.set_ylabel(fr'$dW/d\delta$')
    
handles, labels = ax.get_legend_handles_labels()
    
axes[3,2].legend(handles, labels, loc='upper left',ncol=2,fontsize=8)
axes[3,2].set_axis_off()
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

# country = 'USA'
# country2 = 'EUR'
qty = 'tau_pat_sector'

fig,ax = plt.subplots(dpi=288,figsize = (12,8))

for j,country in enumerate(p_baseline.countries):
    # ax = axes.flat[j]
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_1030/{country}',
                     index_col=0)
    
    ax.plot(df[df.columns[0]],df[country],label=country,color=Category18[j])
    
    # for country2 in p_baseline.countries:
    #     ax.plot(df[df.columns[0]],df[country2],label=country2)
    # ax.plot(df[df.columns[0]],df['pop_weighted'],label='Equal',color='grey',ls='--')
    # ax.plot(df[df.columns[0]],df['negishi'],label='Negishi',color='grey',ls=':')
    # ax.set_title(country)
    # ax.legend()
    #.rolling(2).mean()
    
ax.set_xscale('log')
ax.set_xlabel(r'Proportional change in $\tau$')
ax.set_ylabel(fr'$dW/d\delta$')
ax.legend()
# handles, labels = ax.get_legend_handles_labels()
    
# axes[3,2].legend(handles, labels, loc='upper left',ncol=2,fontsize=8)
# axes[3,2].set_axis_off()
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

# country = 'USA'
# country2 = 'EUR'
qty = 'tau_pat_sector'

fig,axes = plt.subplots(4,3,dpi=288,figsize = (8,8))

for j,country in enumerate(p_baseline.countries):
    ax = axes.flat[j]
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_1030/{country}',
                     index_col=0)
    
    # ax.plot(df[df.columns[0]],df[country],label=country,color=Category18[j])
    
    for country2 in p_baseline.countries:
        ax.plot(df[df.columns[0]],df[country2],label=country2)
    ax.plot(df[df.columns[0]],df['pop_weighted'],label='Equal',color='grey',ls='--')
    ax.plot(df[df.columns[0]],df['negishi'],label='Negishi',color='grey',ls=':')
    ax.set_title(country)
    # ax.legend()
    #.rolling(2).mean()
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Proportional change in $\tau$')
    ax.set_ylabel(fr'$dW/d\delta$')
    # ax.legend()
handles, labels = ax.get_legend_handles_labels()
    
axes[3,2].legend(handles, labels, loc='upper left',ncol=2,fontsize=8)
axes[3,2].set_axis_off()
plt.tight_layout()
plt.show()

#%%

# variations_of_robust_checks = {
#     'baseline':'Baseline',
#     # variation_with_doubled_nu:r'Doubled $\nu$',
#     #  variation_with_doubled_tau_in_pat_sect:r'Doubled $\tau_{n,i}^{1}$',
#     }

# import math

# fig,ax = plt.subplots()

# for c,country in enumerate(p_baseline.countries):
#     # ax = axes.flat[c]
#     for i,rob_check in enumerate(variations_of_robust_checks):
#         variation = rob_check
#         if variation == 'baseline':
#             local_path = cf_path+'baseline_'+baseline+'/'
#         else:
#             local_path = \
#                 cf_path+f'baseline_{baseline}_{variation}/'
#         df_welfare = pd.read_csv(local_path+'dyn_'+country+'.csv')
#         # if rob_check == 'baseline':
#         #     ax.plot(df_welfare['delt'],df_welfare[country],color='k',
#         #             label=variations_of_robust_checks[rob_check]
#         #             # ,lw=5
#         #             )
#         # else:
#         #     if variations_of_robust_checks[rob_check].startswith('High'):
#         #         ls = '-'
#         #         zorder = 0
#         #     if variations_of_robust_checks[rob_check].startswith('Low'):
#         #         ls = '--'
#         #         zorder = 0
#         #     if variations_of_robust_checks[rob_check].startswith('baseline'):
#         #         zorder = 99
#         ax.plot(df_welfare['delt'],df_welfare[country],
#                 label=country,
#                 color = Category18[c],
#                 # ls = ls,
#                 # zorder = zorder
#                 )
#     ax.set_ylabel('Welfare change')
#     name = countries_names[country]
#     # ax.set_xlabel(fr'Proportional change of $\delta$ {name}')
#     ax.set_xlabel(fr'Proportional change of $\delta$')
#     # if country == 'USA':
#     #     ax.set_xlabel(r'Proportional change of $\delta_{US}$')
#     # elif country == 'EUR':
#     #     ax.set_xlabel(r'Proportional change of $\delta_{Europe}$')
#     # else:
#         # ax.set_xlabel(r'Proportional change of $\delta$')
#     # ax.set_xscale('log')
#     # handles, labels = ax.get_legend_handles_labels()
#     # ax.set_title(country)
# # axes[3,2].legend(handles, labels, loc='center left',fontsize=12)
# # axes[3,2].set_axis_off()
#     legend = ax.legend(ncol=2)
#     # def export_legend(legend, filename="legend.pdf"):
#     #     fig  = legend.figure
#     #     fig.canvas.draw()
#     #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         # fig.savefig(filename, dpi="figure", bbox_inches=bbox)

#     # export_legend(legend,filename = robustness_checks_path+'legend.pdf')
#     # plt.legend(False)
#     # legend.remove()
#     # ax.legend(ncol = 2,loc=[1.02,0.02])
#     # plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
#     # plt.savefig(robustness_checks_path+'unilateral_patent_protection_'+country+'.pdf')
# thresh = 1e-3
# thresh2 = 1e-5
# ax.set_xlim(1-thresh,1+thresh)
# ax.set_ylim(1-thresh2,1+thresh2)
# # plt.tight_layout()
# plt.show()
    