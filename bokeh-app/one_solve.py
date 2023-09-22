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

# print(kappa,round(sol_c.semi_elast_patenting_delta[0,1]*100/12,3))

#%%

#
# np.abs(sol_c.compute_quantities_with_prod_patents(p,1e5)-sol_c.compute_quantities_with_prod_patents(p,1e50)).max()
# plt.plot()
# import time
# start = time.perf_counter()

df_terms = sol_c.compute_quantities_with_prod_patents(p)

import pandas as pd

df1 = pd.DataFrame(index = p.countries)
df1['share of innovations patented without prod patent'] = sol_c.psi_o_star[...,1]**-p.k
df1['share of innovations patented with prod patent'] = sol_c.share_innov_patented_dom_with_prod_patent[...,1]
df1['Mult Val Pat'] = sol_c.mult_val_pat
df1['Mult Val All Innov'] = sol_c.mult_val_all_innov

df3 = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))

df3['small pi normalized'] = sol_c.profit[...,1].ravel()
df3['large pi B normalized'] = sol_c.profit_with_prod_patent[...,1].ravel()
df3['large pi D normalized'] = sol_c.profit_with_prod_patent_D[...,1].ravel()
df3['large pi D normalized bis'] = sol_c.profit_with_prod_patent_D_bis[...,1].ravel()
# df3['ratio'] = (sol_c.profit_with_prod_patent[...,1]/sol_c.profit[...,1]).ravel()
# df1.round(4).to_csv('../misc/country_specific_terms.csv')
# df3.round(4).to_csv('../misc/small_pi_large_pi_B_large_pi_D.csv')

# df_terms.to_csv('../misc/terms_details.csv')
# 
# print(time.perf_counter() - start)
    # print(c,sol_c.cons[p.countries.index(c)])
    # print(c,sol_c.g)
    
    # sol, sol_US = fixed_point_solver(p_US,x0=p_US.guess,
                                    # context = 'counterfactual',
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
    
    # sol_US.scale_P(p_US)
    # sol_US.compute_non_solver_quantities(p_US) 
    # sol_US.compute_one_country_welfare_change(p_US,sol_c.cons[p.countries.index(c)],sol_c.g)
    # print(c,sol_US.cons_eq_welfare)
    # list_of_moments = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP',
    #                     'SRDUS', 'SPFLOWDOM', 'SRGDP', 'JUPCOST',
    #                     'SINNOVPATUS','TO']
    # m = moments(list_of_moments)
    # m.load_data()
    # m.compute_moments(sol_c,p)
    # m.compute_moments_deviations()
    # m.plot_moments(m.list_of_moments)

