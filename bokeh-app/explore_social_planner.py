#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:36:40 2023

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)



# baseline_dics = []

# for baseline_number in ['101','102','104']:
#     baseline_dics.append({'baseline':baseline_number,
#                       'variation':'baseline'})
    
#     files_in_dir = next(os.walk('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list.sort(key=float)
    
#     for run in run_list:
#         baseline_dics.append({'baseline':baseline_number,
#                           'variation':run})
        
baseline_dics = [
                # {'baseline':'101',
                #   'variation':'16.1'},
                # {'baseline':'311',
                #   'variation': 'basline'},
                # {'baseline':'311',
                #   'variation': '1.0'},
                # {'baseline':'311',
                #   'variation': '1.1'},
                # {'baseline':'311',
                #   'variation': '1.2'},
                # {'baseline':'311',
                #   'variation': '1.3'},
                # {'baseline':'311',
                #   'variation': '1.4'},
                # {'baseline':'311',
                #   'variation': '1.5'},
                {'baseline':'311',
                  'variation': '1.6'},
                # {'baseline':'311',
                #   'variation': '1.7'},
                # {'baseline':'311',
                #   'variation': '1.8'}
                 ]
        
# lb_delta = 0.01
# ub_delta = 100

for baseline_dic in baseline_dics:    
# for baseline_dic in baseline_dics:    
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        # baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'_'+baseline_dic['variation']+'/'
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)

    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                            cobweb_anim=False,tol =1e-15,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
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
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_price_indices(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)

l_df = []    
d_sol = {}
d_p = {}
    
for delta in np.logspace(-1,1,111):
    country = 'USA'
    p = p_baseline.copy()
    p.delta[p.countries.index(country),1] = p_baseline.delta[p.countries.index(country),1]*delta
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 5,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    print(sol.status)
    if sol.status == 'failed':
        break
    # sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    d_p[delta] = p
    d_sol[delta] = sol_c
    
    l_df.append(pd.DataFrame(index=p.countries+['growth'],
                                          data=sol_c.cons_eq_welfare.tolist()+[sol_c.g/sol_baseline.g]))
    print('Delta '+country, delta)
    # print('Negishi weighted', sol_c.negishi_welfare_change)
    # print('Pop weighted', sol_c.pop_average_welfare_change)
    print('cons_eq_welfare', pd.DataFrame(index=p.countries+['growth'],
                                          data=sol_c.cons_eq_welfare.tolist()+[sol_c.g]))
    print('')
#%%
df = pd.concat(l_df,axis=1)
df = df.T
df.index = np.logspace(-1,1,21)
df.plot(logx=True)

#%%

x = np.logspace(-1,1,111)

for delta in d_p:
    print('delta',round(delta,1),'EU welfare',d_sol[delta].cons_eq_welfare[1])
    # print(np.argwhere(d_sol[delta].psi_m_star ==1 ))
    print(d_sol[delta].cons[1])
    # plt.plot(d_sol[delta].psi_m_star[...,1].ravel())
    # plt.show()

#%%

plt.semilogx(x
             ,[d_sol[delta].cons_eq_welfare[1]/d_sol[0.1].cons_eq_welfare[1]
               for delta in d_sol]
             ,label = 'Cons eq welfare EU')
plt.semilogx(x
             ,[d_sol[delta].cons[1]/d_sol[0.1].cons[1] for delta in d_sol]
             ,label = 'Nominal Cons EU')
plt.semilogx(x
             ,[d_sol[delta].nominal_intermediate_input[1,:].sum()/d_sol[0.1].nominal_intermediate_input[1,:].sum() 
               for delta in d_sol]
             ,label = 'PQ_sum_EU')
plt.semilogx(x
             ,[d_sol[delta].Z[1]/d_sol[0.1].Z[1]
               for delta in d_sol]
             ,label = 'Z_EU')
plt.semilogx(x
             ,[d_sol[delta].price_indices[1]/d_sol[0.1].price_indices[1] 
               for delta in d_sol]
             ,label = 'price indice EU')
plt.semilogx(x
             ,[d_sol[delta].price_indices[0]/d_sol[0.1].price_indices[0] 
               for delta in d_sol]
             ,label = 'price indice USA')
plt.semilogx(x
             ,[d_sol[delta].PSI_CD[1,1]/d_sol[0.1].PSI_CD[1,1]
               for delta in d_sol]
             ,label = 'PSI_CD_EU')
plt.semilogx(x
             ,[d_sol[delta].PSI_M[1,1,1]/d_sol[0.1].PSI_M[1,1,1]
               for delta in d_sol]
             ,label = 'PSI_M_EU_EU')
plt.semilogx(x
             ,[d_sol[delta].l_R[1,1]/d_sol[0.1].l_R[1,1]
               for delta in d_sol]
             ,label = 'l_R_EU')
plt.semilogx(x
             ,[d_sol[delta].profit[1,1,1]/d_sol[0.1].profit[1,1,1]
               for delta in d_sol]
             ,label = 'profit_EU_EU'
             ,ls = '--')
plt.semilogx(x
             ,[d_sol[delta].profit[1,0,1]/d_sol[0.1].profit[1,0,1]
               for delta in d_sol]
             ,label = 'profit_EU_USA'
             ,ls = '--')
plt.semilogx(x
             ,[d_sol[delta].X_M[1,1,1]/d_sol[0.1].X_M[1,1,1]
               for delta in d_sol]
             ,label = 'X_M_EU_EU'
             ,ls = '--')


plt.legend()
plt.title('USA patent protection unilateral counterfactual, Normalized quantities')
plt.show()
#%%
fig, ax = plt.subplots(3,2,figsize = (10,14))

ax[0,0].semilogx(x
             ,[d_sol[delta].psi_o_star[0,1]
               for delta in d_sol]
             ,label = 'psi_o_star_USA')
ax[0,0].set_title('psi_o_star_USA')
ax[0,1].semilogx(x
             ,[d_sol[delta].psi_o_star[1,1]
               for delta in d_sol]
             ,label = 'psi_o_star_EU')
ax[0,1].set_title('psi_o_star_EU')
ax[1,1].semilogx(x
             ,[d_sol[delta].psi_m_star[1,0,1]
               for delta in d_sol]
             ,label = 'psi_m_star_EU_USA')
ax[1,1].set_title('psi_m_star_EU_USA')
ax[2,1].semilogx(x
             ,[d_sol[delta].psi_m_star[1,1,1]
               for delta in d_sol]
             ,label = 'psi_m_star_EU_EU')
ax[2,1].set_title('psi_m_star_EU_EU')
ax[1,0].semilogx(x
             ,[d_sol[delta].psi_m_star[0,1,1]
               for delta in d_sol]
             ,label = 'psi_m_star_USA_EU')
ax[1,0].set_title('psi_m_star_USA_EU')
ax[2,0].semilogx(x
             ,[d_sol[delta].psi_m_star[0,0,1]
               for delta in d_sol]
             ,label = 'psi_m_star_USA_USA')
ax[2,0].set_title('psi_m_star_USA_USA')
# plt.legend()
plt.suptitle('USA patent protection unilateral counterfactual\nPatenting thresholds')
plt.show()

#%% psi_star

fig,ax = plt.subplots()

ax.semilogx(x
             ,[d_sol[delta].psi_m_star[1,0,1]
               for delta in d_sol]
             ,label = 'psi_m_star_EU_USA',
             lw=5,ls='--')
ax.semilogx(x
              ,[d_sol[delta].psi_star[1,0,1]
                for delta in d_sol]
              ,label = 'psi_star_EU_USA')
ax.semilogx(x
              ,[d_sol[delta].psi_o_star[0,1]
                for delta in d_sol]
              ,label = 'psi_o_star_USA')

plt.title('Decomposition of psi_m_star_EU_USA \n psi_m_star_EU_USA=max(psi_star_EU_USA,psi_o_star_USA)')
plt.legend()
plt.show()