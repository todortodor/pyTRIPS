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
p.load_run('calibration_results_matched_economy/1060/')
p_bu = p.copy()
# p.load_data(f'data_smooth_3_years/data_11_countries_1992/',
# # p.load_data(f'data_smooth_3_years/data_12_countries_{run_params["year"]}/',
#             keep_already_calib_params=True)
p.tau = p_bu.tau.copy()
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
p.delta[0,1] = 0.05
sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                # context = 'counterfactual',
                                context = 'calibration',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 100,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )

#%%

importer = 'RUS'
exporter = 'BRA'


# p.delta[0,1] = 0.01
exporter_index = p.countries.index(exporter)
importer_index = p.countries.index(importer) 

print(importer,exporter)
# print(i)
# p.tariff[importer_index,exporter_index,1] = -0.5
p.tariff[:,:,1] = -0.5
sol, sol_c = fixed_point_solver(p,x0=p.guess,
                        #         context = 'counterfactual',
                        # cobweb_anim=False,tol =1e-14,
                        # accelerate=False,
                        # accelerate_when_stable=True,
                        # cobweb_qty='l_R',
                        # plot_convergence=True,
                        # plot_cobweb=False,
                        # safe_convergence=0.1,
                        # disp_summary=False,
                        # damping = 10,
                        # max_count = 10000,
                        # accel_memory =50, 
                        # accel_type1=True, 
                        # accel_regularization=1e-10,
                        # accel_relaxation=0.5, 
                        # accel_safeguard_factor=1, 
                        # accel_max_weight_norm=1e6,
                        # damping_post_acceleration=2
                        # # damping=10
                        #   # apply_bound_psi_star=True
                        # )
context = 'counterfactual',
cobweb_anim=False,tol =1e-14,
accelerate=False,
accelerate_when_stable=True,
cobweb_qty='l_R',
plot_convergence=True,
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
# sol_c.compute_quantities_with_prod_patents(p)
# print(kappa,round(sol_c.semi_elast_patenting_delta[0,1]*100/12,3))

#%%

import numpy as np
import pandas as pd

p_cf = p.copy()
# p_cf.tau[:,:,1] = 1e10*p_cf.tau[:,:,1]
p_cf.tau[:,:,:] = np.inf

for i in range(11):
    p_cf.tau[i,i,:] = 1
    
from solver_funcs import fixed_point_solver_with_exog_pat_and_rd 

sol, sol_cf = fixed_point_solver_with_exog_pat_and_rd(p_cf,p,x0=p.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.1,
                        disp_summary=False,
                        damping = 1,
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
sol_cf.scale_P(p_cf)
sol_cf.compute_growth(p)

sol_cf.psi_C = sol_c.psi_C.copy()
sol_cf.psi_star = sol_c.psi_star.copy()
sol_cf.psi_o_star = sol_c.psi_o_star.copy()
sol_cf.psi_m_star = sol_c.psi_m_star.copy()

sol_cf.PSI_M = sol_c.PSI_M.copy()
sol_cf.PSI_CD = sol_c.PSI_CD.copy()

sol_cf.compute_sectoral_prices(p)

sol_cf.l_Ae = sol_c.l_Ae.copy()
sol_cf.l_Ao = sol_c.l_Ao.copy()
sol_cf.l_P = sol_c.l_P.copy()

sol_cf.compute_trade_flows_and_shares(p)
sol_cf.compute_price_indices(p)

sol_cf.compute_non_solver_quantities(p_cf) 
sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_c)
sol_cf.compute_world_welfare_changes(p_cf,sol_c)

df = pd.DataFrame()

for i,country in enumerate(p.countries):
    df.loc[country,'welfare'] = sol_cf.cons_eq_welfare[i]*100-100
    df.loc[country,'change research labor (%)'] = sol_cf.l_R[i,1]/sol_c.l_R[i,1]*100-100
    df.loc[country,'change cons sector 0 (%)'] = sol_cf.sectoral_cons[i,0]/sol_c.sectoral_cons[i,0]*100-100
    df.loc[country,'change cons sector 1 (%)'] = sol_cf.sectoral_cons[i,1]/sol_c.sectoral_cons[i,1]*100-100
df.loc['Equal','welfare'] = sol_cf.cons_eq_pop_average_welfare_change*100-100
df.loc['Negishi','welfare'] = sol_cf.cons_eq_negishi_welfare_change*100-100
df.loc['Growth rate','welfare'] = sol_cf.g*100

# df = df.round(2)

# df.round(2).to_csv('../misc/welfare.csv')

df_PSI_m = pd.DataFrame(index=p.countries)

df_PSI_m['baseline PSI CD'] =  sol_c.PSI_CD[...,1]
df_PSI_m['inf trade costs PSI CD'] =  sol_cf.PSI_CD[...,1]
df_PSI_m['change (%)'] = df_PSI_m['inf trade costs PSI CD']/df_PSI_m['baseline PSI CD']*100-100

# df_PSI_m.round(2).to_csv('../misc/psi_CD_inf_trade_costs.csv')

#%%

sol_c.compute_quantities_with_prod_patents(p)

#
# np.abs(sol_c.compute_quantities_with_prod_patents(p,1e5)-sol_c.compute_quantities_with_prod_patents(p,1e50)).max()
# plt.plot()
# import time
# start = time.perf_counter()
import pandas as pd
df = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))

df['psi_m_star_without'] = sol_c.psi_m_star[...,1].ravel()
df['psi_m_star_with'] = sol_c.psi_m_star_with_prod_patent[...,1].ravel()
df['change pat threshold'] = (df['psi_m_star_with']-df['psi_m_star_without'])*100/df['psi_m_star_without']
df['share_innov_pat_without'] = sol_c.psi_m_star[...,1].ravel()**-p.k
df['share_innov_pat_with'] = sol_c.psi_m_star_with_prod_patent[...,1].ravel()**-p.k
df['change share_innov_pat'] = (df['share_innov_pat_with']-df['share_innov_pat_without'])*100/df['share_innov_pat_without']
df['case origin'] = sol_c.case_marker.ravel()

df = df.sort_index(level=1)



#%%

df['change pat threshold'].describe().round(2)
df['change share_innov_pat'].describe().round(2)
df[df['change pat threshold'] < 0]['change pat threshold'].describe().round(2)
df[df['change pat threshold'] < 0]['change share_innov_pat'].describe().round(2)
df[df['change pat threshold'] == 0]['change pat threshold'].describe().round(2)
df[df['change pat threshold'] > 0]['change pat threshold'].describe().round(2)
df[df['change pat threshold'] > 0]['change share_innov_pat'].describe().round(2)

for country in p.countries:
    list_without = df.xs(country,level=1).sort_values('psi_m_star_without').index.get_level_values(0).tolist()
    list_with = df.xs(country,level=1).sort_values('psi_m_star_with').index.get_level_values(0).tolist()
    print(country,list_without==list_with)
    if list_without != list_with:
        print(country,list_without,list_with)
#%%

df_stats = pd.DataFrame()

df_stats['All change pat threshold'] = df['change pat threshold'].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)
df_stats['Inter change pat threshold'] = df.query('origin!=destination')['change pat threshold'].describe(
    percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)
df_stats['Domestic change pat threshold'] = df.query('origin==destination')['change pat threshold'].describe(
    percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)

df_stats['All change share innov pat'] = df['change share_innov_pat'].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)
df_stats['Inter change share innov pat'] = df.query('origin!=destination')['change share_innov_pat'].describe(
    percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)
df_stats['Domestic change share innov pat'] = df.query('origin==destination')['change share_innov_pat'].describe(
    percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2)



for country in p.countries:
    list_without = df.xs(country,level=1).sort_values('psi_m_star_without').index.get_level_values(0).tolist()
    list_with = df.xs(country,level=1).sort_values('psi_m_star_with').index.get_level_values(0).tolist()
    print(country,list_without==list_with)
    if list_without != list_with:
        print(country,list_without,list_with)

df_stats.to_csv('../misc/patenting_thresholds_with_production_patents_statistics.csv')

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

df4['cc_share_innov_pat'] = sol_c.psi_o_star_without_prod_patent_cc[...,1]**-p.k*100
df4['cc1_share_with'] = sol_c.psi_o_star_with_prod_patent_cc1[...,1]**-p.k*100
df4['cc2_share_with'] = sol_c.psi_o_star_with_prod_patent_cc2[...,1]**-p.k*100

df4['cc1_share_change'] = (df4['cc1_share_with']-df4['cc_share_innov_pat'])*100/df4['cc_share_innov_pat']
df4['cc2_share_change'] = (df4['cc2_share_with']-df4['cc_share_innov_pat'])*100/df4['cc_share_innov_pat']


df5 = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))
 
df5['aa_pat_thresh'] = sol_c.psi_m_star_without_prod_patent_aa[...,1].ravel()
df5['aa_with'] = sol_c.psi_m_star_with_prod_patent_aa[...,1].ravel()
 
df5['aa_share_innov_pat'] = sol_c.psi_m_star_without_prod_patent_aa[...,1].ravel()**-p.k*100
df5['aa_share_with'] = sol_c.psi_m_star_with_prod_patent_aa[...,1].ravel()**-p.k*100

df5['aa_share_change'] = (df5['aa_share_with']-df5['aa_share_innov_pat'])*100/df5['aa_share_innov_pat']

df3 = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p.countries,p.countries], names = ['destination','origin']
    ))

df3['small pi normalized'] = sol_c.profit[...,1].ravel()
df3['large pi B normalized'] = sol_c.profit_with_prod_patent[...,1].ravel()

# df1.to_csv('../misc/country_specific_terms.csv')
# # for cas in ['a','b','c']:
# #     dfs[cas].to_csv(f'../misc/case_{cas}.csv')
# df2.to_csv('../misc/patenting_thresholds_cases_a_b_c.csv')
# df4.to_csv('../misc/patenting_thresholds_cases_cc.csv')
# df5.to_csv('../misc/patenting_thresholds_case_aa.csv')
# df3.to_csv('../misc/profits.csv')


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