#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:59:20 2022

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, find_coop_eq_tariff
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

p_baseline = parameters()
# p_baseline.load_run('opt_tariff_delta/1040/scenario_0/')
# p_baseline.load_data('data/data_11_countries_2018/')
p_baseline.load_run('calibration_results_matched_economy/1050/')

welfs = []

exporter='CAN'
importer='KOR'

exporter_index = p_baseline.countries.index(exporter)
importer_index = p_baseline.countries.index(importer) 

sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 2,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=1
                        # damping=10
                          # apply_bound_psi_star=True
                        )

sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

#%%

p = p_baseline.copy()

x = np.linspace(0,100,101)
l_sol = []

for i,tariff in enumerate(x):
# for importer in p_baseline.countries:
#     for exporter in p_baseline.countries:
        # importer = 'RUS'
        # exporter = 'BRA'

        exporter_index = p_baseline.countries.index(exporter)
        importer_index = p_baseline.countries.index(importer) 
        
        # print(importer,exporter)
        print(i)
        p = p_baseline.copy()
        p.tariff[importer_index,exporter_index,1] = tariff
        # p.delta[-2,1] = tariff
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1000,
                                accel_memory =50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
    
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p) 
        sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        sol_c.compute_world_welfare_changes(p,sol_baseline)
        
        # welfs.append(sol_c.cons_eq_welfare.tolist()+[sol_c.cons_eq_negishi_welfare_change,
        #                                               sol_c.cons_eq_pop_average_welfare_change])
        
        p.guess = sol_c.vector_from_var()
        l_sol.append(sol_c)

# welfs = np.array(welfs)

# # print(welfs[:,0])

# plt.plot(x,welfs[:,importer_index])
# plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()
cycler = plt.cycler(linestyle=['-']*11+['--','--'],
                    color=sns.color_palette()+sns.color_palette()[:3])
# plt.plot(welfs[:,-2])
ax.set_prop_cycle(cycler)
# ax.set_xlabel(f'{importer}, {exporter} tariff')
ax.set_ylabel('Welfare')
plt.plot(x,welfs,lw=2)
plt.legend(p.countries)
# plt.xscale('symlog',linthresh=1)
# plt.xscale('log')
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()

for i,c in enumerate(p.countries):
    ax.plot(x,[so.price_indices[i]/sol_baseline.price_indices[i]
               for so in l_sol])
ax.set_xlabel(f'{importer}, {exporter} tariff')
ax.set_ylabel('Price indices')
plt.legend(p.countries)
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()

# for i,c in enumerate(p.countries):
ax.plot(x,[so.X[importer_index,exporter_index,1]*x[j]/(1+x[j])#/sol_baseline.price_indices[i]
           for j,so in enumerate(l_sol)],color='black')
ax.set_xlabel(f'{importer}, {exporter} tariff')
ax.set_ylabel('Tariff revenue')
# plt.legend(p.countries)
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()
ax2 = ax.twinx()

ax.plot(1+x,[so.X[importer_index,exporter_index,1]/(1+x[j])#/sol_baseline.price_indices[i]
           for j,so in enumerate(l_sol)],color='black',label='Trade flow')
ax.plot(1+x,[so.X[importer_index,exporter_index,1]*x[j]/(1+x[j])#/sol_baseline.price_indices[i]
           for j,so in enumerate(l_sol)],color='b',label='Tariff revenue')
# ax.plot(1+x,[so.profit[importer_index,:,1] #/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='growth rate')
# ax.plot(1+x,[so.profit[importer_index,:,1].sum() #/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='agg profit')
# ax.plot(1+x,[so.PSI_CD[importer_index,1] #/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='psi CD')
# ax.plot(1+x,[so.X_M[importer_index,exporter_index,1]/(
#     so.X_M[importer_index,exporter_index,1]+so.X_CD[importer_index,exporter_index,1]) #/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='monopolistic to competitive trade flow')
# ax.plot(1+x,[so.X_M[:,:,1].sum()/(
#     so.X_M[:,:,1].sum()+so.X_CD[:,:,1].sum()) #/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='black',label='monopolistic to competitive all trade')

# ax.plot(1+x,[so.price_indices[importer_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='Consumption')
# ax.plot(1+x,[so.price_indices[importer_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='green',label='Price')
# ax2.plot(1+x,[so.w[importer_index]/sol_baseline.w[importer_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='wage')
# ax2.plot(1+x,[so.cons[importer_index]/sol_baseline.cons[importer_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='black',label='consumption')
# ax2.plot(1+x,[so.Z[importer_index]/sol_baseline.Z[importer_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='Z')
# ax2.plot(1+x,[so.nominal_intermediate_input[importer_index,:].sum(
#     )/sol_baseline.nominal_intermediate_input[importer_index,:].sum()#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='orange',label='Intermediate inputs')

# ax.set_yscale('log')
ax.set_xlabel(f'{importer}, {exporter} tariff')
# ax.set_ylabel('Trade flow')

ax2.plot(1+x,[so.cons_eq_welfare[importer_index]#/sol_baseline.price_indices[i]
           for j,so in enumerate(l_sol)],color='red',label=f'welfare {importer}')
# ax2.plot(1+x,[so.cons_eq_welfare[exporter_index]#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='green',label=f'welfare {exporter}')
ax.legend(loc='center left')
ax2.legend(loc='center right')
plt.xscale('log')
# plt.yscale('log')
plt.show()