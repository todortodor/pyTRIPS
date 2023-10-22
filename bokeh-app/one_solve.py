#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:57:06 2022

@author: simonl
"""

from classes import moments, parameters, var
from solver_funcs import fixed_point_solver

p = parameters()
# p.load_run('calibration_results_matched_economy/1020/')
p.load_run('calibration_results_matched_economy/1030/')
# for c in p.countries:
    # p_US = p.make_one_country_parameters(c)
    # p.load_run('calibration_results_matched_economy/405/')
    # m = moments()
    # m.load_data()
    # m.load_run('calibration_results_matched_economy/baseline_312_variations/2.0/')
    # m.compute_moments_deviations()
    # test = compute_rough_jacobian(p, m, 'delta', (0,1),change_by = 0.1)
    # plt.barh(m.get_signature_list(),test)
    # p.fo = p.fe
    # p.delta[0,1] = 0.1*p.delta[0,1]
    # p.d_np = np.ones((p.N,p.N))
    # np.fill_diagonal(p.d_np,p.d)
    # p = p_it_baseline

# for kappa in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#     p.kappa = kappa
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
                        damping = 100,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=100
                        # damping=10
                          # apply_bound_psi_star=True
                        )
# sol, sol_c = fixed_point_solver(p,x0=p.guess,
#                                 context = 'counterfactual',
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='l_R',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 2,
#                         max_count = 1000,
#                         accel_memory =50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         # damping=10
#                           # apply_bound_psi_star=True
#                         )

sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p) 
# sol_c.compute_quantities_with_prod_patents(p)
# print(kappa,round(sol_c.semi_elast_patenting_delta[0,1]*100/12,3))

sol_c.compute_quantities_with_prod_patents(p)


#
# np.abs(sol_c.compute_quantities_with_prod_patents(p,1e5)-sol_c.compute_quantities_with_prod_patents(p,1e50)).max()
# plt.plot()
# import time
# start = time.perf_counter()



#%%


import numpy as np
import pandas as pd

df1 = pd.DataFrame(index = p.countries)

dfs = {}

for cas in ['a','b','c']:
    dfs[cas] = pd.DataFrame(index = p.countries)
    dfs[cas]['percentage share of innovations patented without prod patent'] = getattr(sol_c,f'share_innov_patented_dom_without_prod_patent_{cas}')[...,1]*100
    dfs[cas]['percentage share of innovations patented with prod patent'] = getattr(sol_c,f'share_innov_patented_dom_with_prod_patent_{cas}')[...,1]*100
    dfs[cas]['percentage points share of innovations patented diff'] = dfs[cas]['percentage share of innovations patented with prod patent'
                                                                        ] - dfs[cas]['percentage share of innovations patented without prod patent']
    

cases = {'USA':'a', 
         'EUR':'c', 
         'JAP':'a', 
         'CHN':'a', 
         'BRA':'b', 
         'IND':'b', 
         'CAN':'b', 
         'KOR':'c', 
         'RUS':'b', 
         'MEX':'b', 
         'ROW':'b'}
    
for cas in cases:
    df1.loc[cas,'percentage share of innovations patented without prod patent'
            ] = dfs[cases[cas]].loc[cas,'percentage share of innovations patented without prod patent']
    df1.loc[cas,'percentage share of innovations patented with prod patent'
            ] = dfs[cases[cas]].loc[cas,'percentage share of innovations patented with prod patent']
    
df1['percentage points share of innovations patented diff'] = df1['percentage share of innovations patented with prod patent'
                                    ] - df1['percentage share of innovations patented without prod patent']

df1['Mult Val Pat'] = sol_c.mult_val_pat
df1['Mult Val All Innov'] = sol_c.mult_val_all_innov

df2 = pd.DataFrame(index = p.countries)

# df2['initial_pat_thresh'] = np.min(sol_c.psi_m_star[...,1], axis=1)
df2['min_n(psi_m_star_{n,i}) '] = np.min(sol_c.psi_m_star[...,1], axis=0)
df2['a_pat_thresh'] = sol_c.psi_o_star_without_prod_patent_a[...,1]
df2['a_with'] = sol_c.psi_o_star_with_prod_patent_a[...,1]
df2['b_pat_thresh'] = sol_c.psi_o_star_without_prod_patent_b[...,1]
df2['b_with'] = sol_c.psi_o_star_with_prod_patent_b[...,1]
df2['c_pat_thresh'] = sol_c.psi_o_star_without_prod_patent_c[...,1]
df2['c_with'] = sol_c.psi_o_star_with_prod_patent_c[...,1]

df4 = pd.DataFrame(index = p.countries)

# df2['initial_pat_thresh'] = np.min(sol_c.psi_m_star[...,1], axis=1)
# df4['min_n(psi_m_star_{n,i}) '] = np.min(sol_c.psi_m_star[...,1], axis=0)
df4['cc_pat_thresh'] = sol_c.psi_o_star_without_prod_patent_cc[...,1]
df4['cc1_with'] = sol_c.psi_o_star_with_prod_patent_cc1[...,1]
df4['cc2_with'] = sol_c.psi_o_star_with_prod_patent_cc2[...,1]

df5 = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))
 
df5['aa_pat_thresh'] = sol_c.psi_m_star_without_prod_patent_aa[...,1].ravel()
df5['aa_with'] = sol_c.psi_m_star_with_prod_patent_aa[...,1].ravel()

df3 = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))

df3['small pi normalized'] = sol_c.profit[...,1].ravel()
df3['large pi B normalized'] = sol_c.profit_with_prod_patent[...,1].ravel()

df1.to_csv('../misc/country_specific_terms.csv')
# for cas in ['a','b','c']:
#     dfs[cas].to_csv(f'../misc/case_{cas}.csv')
df2.to_csv('../misc/patenting_thresholds_cases_a_b_c.csv')
df4.to_csv('../misc/patenting_thresholds_cases_cc.csv')
df5.to_csv('../misc/patenting_thresholds_case_aa.csv')
df3.to_csv('../misc/profits.csv')


#%% checks

import numpy as np

check_b = (sol_c.psi_o_star_with_prod_patent_b[...,1] < np.min(sol_c.psi_m_star[...,1],axis=1))



#%%

import matplotlib.pyplot as plt

fig,axes = plt.subplots(3,4,figsize = (14,14), dpi = 288)
# fig,ax = plt.subplots(figsize = (14,14), dpi = 288)

for i,ax in enumerate(axes.flatten()):
    try:
        ax.bar(p.countries, sol_c.psi_m_star[:,i,1])
        ax.set_title('Origin : '+p.countries[i])
        ax.set_xticklabels(p.countries, rotation=90, ha='right')
        ax.set_yscale('log')
    except:
        pass

plt.suptitle(r'$\psi^{m,\ast}$ across destinations by origin')
plt.tight_layout()
plt.show()

#%%
import numpy as np

for i,origin in enumerate(p.countries):
    print(origin,'pat first in :', [destination for n,destination in 
                                    enumerate(p.countries) 
                                    if np.isclose(sol_c.psi_m_star[n,i,1],
                                                  np.min(sol_c.psi_m_star[:,i,1])
                                                  )])

#%%

import numpy as np

for i,origin in enumerate(p.countries):
    print(origin,'pat first in :', [destination for n,destination in 
                                    enumerate(p.countries) 
                                    if np.isclose(sol_c.psi_m_star[n,i,1],
                                                  np.min(sol_c.psi_m_star[:,i,1])
                                                  )])
    print(np.min(sol_c.psi_m_star[:,i,1]),sol_c.psi_m_star[i,i,1])
    # for n,destination in enumerate(p.countries):
    #     if np.isclose(sol_c.psi_m_star[n,i,1],np.min(sol_c.psi_m_star[:,i,1])):
    #         print(destination)