#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:20:31 2023

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

#%% Choose a run, load parameters, moments, solution

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

#%% Stylized fact about patent flows / trade flows

data = m_baseline.cc_moments.reset_index().pivot(index='destination_code',
                                                    columns='origin_code',
                                                    values='patent flows').values*1e6/\
         (m_baseline.ccs_moments.xs(1,level=2).reset_index().pivot(index='destination_code',
                                                             columns='origin_code',
                                                             values='trade').values)

data_without_diag = data.copy()
np.fill_diagonal(data_without_diag,np.nan)

df = pd.DataFrame(index = pd.Index(p_baseline.countries,name='Origin'), 
                  columns = p_baseline.countries,
                  data = data.transpose())
df = df.T
df.index.name = 'Destination'

df.loc['mean'] = df.mean(axis=0)
df['mean'] = df.mean(axis=1)

df_without_diag = pd.DataFrame(index = pd.Index(p_baseline.countries,name='Origin'), 
                  columns = p_baseline.countries,
                  data = data_without_diag.transpose())
df_without_diag = df_without_diag.T
df_without_diag.index.name = 'Destination'

df_without_diag.loc['mean'] = df_without_diag.mean(axis=0)
df_without_diag['mean'] = df_without_diag.mean(axis=1)

caption = 'Patent flows over trade flows (Number of patent per trillion US$)'

df.style.format(precision=2).to_latex(data_fact_path+'pat_over_trade_coeffs.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(data_fact_path+'pat_over_trade_coeffs.csv',float_format='%.2f')

df_without_diag.style.format(precision=2).to_latex(data_fact_path+'pat_over_trade_coeffs_without_diag.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df_without_diag.to_csv(data_fact_path+'pat_over_trade_coeffs_without_diag.csv',float_format='%.2f')

#%% Time series of number of international families by office 

years = [y for y in range(1990,2019)]
nb_countries = p_baseline.N

pflows = pd.concat(
    [pd.read_csv(data_path+f'data_{nb_countries}_countries_{y}/country_country_moments.csv',
                 index_col=[0,1])
     for y in years],
    axis=0,
    keys=years,
    names=['year','origin_code','destination_code'],
    # ignore_index=True
    )

c_map = {i+1:p_baseline.countries[i] for i in range(nb_countries)}

ori = pflows.query('origin_code!=destination_code').groupby(['destination_code','year']).sum().reset_index()
ori['destination_code'] = ori['destination_code'].map(c_map)
# ori['destination_code'] = ori['destination_code'].map(countries_names)
ori = ori.pivot(
    columns = 'destination_code',
    index = 'year',
    values = 'patent flows'
    )
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):

    ax.plot(ori.index,ori[country],
            color=Category18[i],
            label=countries_names[country]
            )
    
    
    
    plt.yscale('log')

ax.legend(loc=[1.02,0.02])
# ax.legend()
ax.set_ylabel('International patent families by destination')
ax.set_xlim([1990,2015])
# plt.grid()

for save_format in save_formats:
    plt.savefig(data_fact_path+'international_pat_families_by_office.'+save_format,format=save_format)
    
plt.show()

#%% Time series of number of international families by origin
years = [y for y in range(1990,2019)]
nb_countries = p_baseline.N

pflows = pd.concat(
    [pd.read_csv(data_path+f'data_{nb_countries}_countries_{y}/country_country_moments.csv',
                 index_col=[0,1])
     for y in years],
    axis=0,
    keys=years,
    names=['year','origin_code','destination_code'],
    # ignore_index=True
    )

c_map = {i+1:p_baseline.countries[i] for i in range(nb_countries)}

ori = pflows.query('origin_code!=destination_code').groupby(['origin_code','year']).sum().reset_index()
ori['origin_code'] = ori['origin_code'].map(c_map)
# ori['destination_code'] = ori['destination_code'].map(countries_names)
ori = ori.pivot(
    columns = 'origin_code',
    index = 'year',
    values = 'patent flows'
    )
# fig,ax = plt.subplots(figsize=(14,11))
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):

    ax.plot(ori.index,ori[country],color=Category18[i],label=countries_names[country])
    
    ax.legend(loc=[1.02,0.02])
    
    plt.yscale('log')
    
ax.set_ylabel('International patent families by origin')
ax.set_xlim([1990,2015])

for save_format in save_formats:
    plt.savefig(data_fact_path+'international_pat_families_by_origin.'+save_format,format=save_format)
    
plt.show()

#%% write excel spredsheet of the baseline

write_calibration_results(calibration_path+'baseline',p_baseline,m_baseline,sol_baseline,commentary = '')

#%% Compute patenting quantities with production patents

sol_baseline.compute_quantities_with_prod_patents(p_baseline)

df = pd.DataFrame(index = p_baseline.countries)

df['Mult Val Pat'] = sol_baseline.mult_val_pat
df['Mult Val All Innov'] = sol_baseline.mult_val_all_innov
df.round(4).to_csv(calibration_path+'quantities_with_production_patents.csv')

df_profit = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p_baseline.countries,p_baseline.countries], names = ['destination','origin']
    ))

df_profit['small pi normalized'] = sol_baseline.profit[...,1].ravel()
df_profit['large pi B normalized'] = sol_baseline.profit_with_prod_patent[...,1].ravel()

df_profit.to_csv(calibration_path+'profits_with_production_patents.csv')

df = pd.DataFrame( index = pd.MultiIndex.from_product(
    [p_baseline.countries,p_baseline.countries], names = ['destination','origin']
    ))

df['psi_m_star_without'] = sol_baseline.psi_m_star[...,1].ravel()
df['psi_m_star_with'] = sol_baseline.psi_m_star_with_prod_patent[...,1].ravel()
df['change pat threshold'] = (df['psi_m_star_with']-df['psi_m_star_without'])*100/df['psi_m_star_without']
df['share_innov_pat_without'] = sol_baseline.psi_m_star[...,1].ravel()**-p_baseline.k
df['share_innov_pat_with'] = sol_baseline.psi_m_star_with_prod_patent[...,1].ravel()**-p_baseline.k
df['change share_innov_pat'] = (df['share_innov_pat_with']-df['share_innov_pat_without'])*100/df['share_innov_pat_without']
df['case origin'] = sol_baseline.case_marker.ravel()

df = df.sort_index(level=1)

df.to_csv(calibration_path+'patenting_thresholds_with_production_patents.csv')


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

df_stats.to_csv(calibration_path+'patenting_thresholds_with_production_patents_statistics.csv')


#%% Gains from trade


p_cf = p_baseline.copy()
p_cf.tau[:,:,1] = 1e5

for i in range(11):
    p_cf.tau[i,i,1] = 1

# static gains only

from solver_funcs import fixed_point_solver_with_exog_pat_and_rd 

sol, sol_cf = fixed_point_solver_with_exog_pat_and_rd(p_cf,p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.1,
                        disp_summary=False,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        # damping=10
                          # apply_bound_psi_star=True
                        )
sol_cf.scale_P(p_cf)
sol_cf.compute_growth(p_cf)

sol_cf.psi_C = sol_baseline.psi_C.copy()
sol_cf.psi_star = sol_baseline.psi_star.copy()
sol_cf.psi_o_star = sol_baseline.psi_o_star.copy()
sol_cf.psi_m_star = sol_baseline.psi_m_star.copy()

sol_cf.PSI_M = sol_baseline.PSI_M.copy()
sol_cf.PSI_CD = sol_baseline.PSI_CD.copy()

sol_cf.compute_sectoral_prices(p_cf)

sol_cf.l_Ae = sol_baseline.l_Ae.copy()
sol_cf.l_Ao = sol_baseline.l_Ao.copy()
sol_cf.l_P = sol_baseline.l_P.copy()

sol_cf.compute_trade_flows_and_shares(p_cf)
sol_cf.compute_price_indices(p_cf)

sol_cf.compute_non_solver_quantities(p_cf) 
sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
sol_cf.compute_world_welfare_changes(p_cf,sol_baseline)

df = pd.DataFrame()

for i,country in enumerate(p_baseline.countries):
    df.loc[country,'welfare static gains'] = sol_cf.cons_eq_welfare[i]*100-100
df.loc['Equal','welfare static gains'] = sol_cf.cons_eq_pop_average_welfare_change*100-100
df.loc['Negishi','welfare static gains'] = sol_cf.cons_eq_negishi_welfare_change*100-100
df.loc['Growth rate','welfare static gains'] = sol_cf.g*100

# with dynamic gains

sol, sol_cf = fixed_point_solver(p_cf,x0=p_cf.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.1,
                        disp_summary=False,
                        damping = 10,
                        max_count = 50000,
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

sol_cf.scale_P(p_cf)
sol_cf.compute_non_solver_quantities(p_cf) 

sol_cf.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
sol_cf.compute_world_welfare_changes(p_cf,sol_baseline)

for i,country in enumerate(p_baseline.countries):
    df.loc[country,'welfare dynamic gains'] = sol_cf.cons_eq_welfare[i]*100-100
df.loc['Equal','welfare dynamic gains'] = sol_cf.cons_eq_pop_average_welfare_change*100-100
df.loc['Negishi','welfare dynamic gains'] = sol_cf.cons_eq_negishi_welfare_change*100-100
df.loc['Growth rate','welfare dynamic gains'] = sol_cf.g*100

sol, dyn_sol_cf = dyn_fixed_point_solver(p_cf, sol_init=sol_baseline,
                                         sol_fin=sol_cf,Nt=25,
                        t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
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
dyn_sol_cf.compute_non_solver_quantities(p_cf)

for i,country in enumerate(p_baseline.countries):
    df.loc[country,'with transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_welfare[i]*100-100
df.loc['Equal','with transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_pop_average_welfare_change*100-100
df.loc['Negishi','with transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_negishi_welfare_change*100-100
# df.loc['Growth rate','with transition welfare dynamic gains'] = sol_cf.g*100

sol, dyn_sol_cf = dyn_fixed_point_solver(p_baseline, sol_init=sol_cf,sol_fin=sol_baseline,Nt=25,
                        t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
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
dyn_sol_cf.compute_non_solver_quantities(p_baseline)

for i,country in enumerate(p_baseline.countries):
    df.loc[country,'with reverse transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_welfare[i]*100-100
df.loc['Equal','with reverse transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_pop_average_welfare_change*100-100
df.loc['Negishi','with reverse transition welfare dynamic gains'] = dyn_sol_cf.cons_eq_negishi_welfare_change*100-100

df.to_csv(calibration_path+'gains_from_trade.csv')

#%% Comparing trade flows with patent flows

fig,ax = plt.subplots()

tflow_shares = m_baseline.ccs_moments.query('destination_code!=origin_code'
                                    ).xs(1,level=2)
tflow_shares = tflow_shares/tflow_shares.sum()
pflow_shares = m_baseline.cc_moments.query('destination_code!=origin_code'
                                    )
pflow_shares = pflow_shares/pflow_shares.sum()

ax.scatter(pflow_shares.values.ravel(),
            m_baseline.SPFLOW.ravel(),
            label = 'International patent shares: model')
ax.scatter(pflow_shares.values.ravel(),
            tflow_shares.values.ravel(),
            label = 'International trade shares',marker='^')

ax.set_xlabel('International patent shares: data')
ax.set_ylabel('Model')

ax.set_xscale('log')
ax.set_yscale('log')

plt.legend()
    
ax.plot([m_baseline.SPFLOW_target.ravel().min(),m_baseline.SPFLOW_target.ravel().max()],
        [m_baseline.SPFLOW_target.ravel().min(),m_baseline.SPFLOW_target.ravel().max()],
        ls='--',color='grey')

for save_format in save_formats:
    plt.savefig(calibration_path+'trade_vs_patent_flows.'+save_format,format=save_format)

plt.show()

#%% plot matching of moments : SPFLOW

moment = 'SPFLOW'

annotate_with_labels = True
replace_country_codes_with_labels = False

labels = m_baseline.idx[moment].to_series().transform(','.join).astype('str')
if replace_country_codes_with_labels:
    for code, name in countries_names.items():
        labels = labels.str.replace(code,name)

x = getattr(m_baseline, moment+'_target').ravel()
y = getattr(m_baseline, moment).ravel()

fig, ax = plt.subplots()

ax.scatter(x,y,s=6,lw=0.6
            ,marker='+'
           )
ax.plot([x.min(),x.max()],[x.min(),x.max()],color='grey',ls='--',lw=0.5)
if annotate_with_labels:
    texts = [ax.annotate(label,
                         xy=(x[i],y[i]),
                        xytext=(1,1),
                        textcoords='offset points',
                          fontsize = 2
                        )
             for i,label in enumerate(labels)]
    
    
    
plt.xlabel('Data')
plt.ylabel('Model')

# plt.title(m_baseline.description.loc[moment,'description'])

plt.xscale('log')
plt.yscale('log')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))

for save_format in save_formats:
    plt.savefig(calibration_path+moment+'_cross_small.'+save_format,format=save_format)
    plt.savefig(calibration_path+moment+'.'+save_format,format=save_format)

plt.show()

df = pd.DataFrame(index = labels.index)
df['Target'] = x
df['Model'] = y

df.to_csv(calibration_path+moment+'.csv')

#%% output table for matching of moments : scalars

df = pd.DataFrame(index = pd.Index([],name='Moment'),
                                       columns = ['Target','Model'])

for moment in m_baseline.list_of_moments:
    if getattr(m_baseline,moment).size == 1:
        df.loc[m_baseline.description.loc[moment,'description'],'Model'] = getattr(m_baseline,moment)
        df.loc[m_baseline.description.loc[moment,'description'],'Target'] = getattr(m_baseline,moment+'_target')

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = 'Scalar moments targeted'

df.style.format(precision=6).to_latex(calibration_path+'scalar_moments_matching.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+'scalar_moments_matching.csv',float_format='%.6f')

#%% output table for matching of real GDP

moment = 'SRGDP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(calibration_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+moment+'.csv',float_format='%.6f')

#%% output table for matching of price_indices

moment = 'RP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(calibration_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+moment+'.csv',float_format='%.6f')

#%% output table for matching of RD expenditures

moment = 'RD'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(calibration_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+moment+'.csv',float_format='%.6f')

#%% output table for scalar parameters

df = pd.DataFrame(index = pd.Index([],name='Parameter'),
                                       # columns = ['Description','Value'])
                                       columns = ['Value'])

for param in ['k']:
    df.loc[param,'Value'] = getattr(p_baseline,param)
    
# for param in ['fe','fo','nu']:
#     df.loc[param,'Value'] = getattr(p_baseline,param)[1]

df.loc[r'$f_e$'] = p_baseline.fe[1]
df.loc[r'$f_o$'] = p_baseline.fo[1]
df.loc[r'$\nu$'] = p_baseline.nu[1]
df.loc[r'$g_0$'] = p_baseline.g_0

df['Value'] = df['Value'].astype(float)

caption = 'Scalar parameters'

df.style.format(precision=6).to_latex(calibration_path+'scalar_parameters.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+'scalar_parameters.csv',float_format='%.6f')

#%% plot parameter delta and output table

parameter = 'delta'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df = df.rename(columns = {'delta':r'$\delta$'})

df.style.format(precision=6).to_latex(calibration_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+parameter+'.csv',float_format='%.6f')

#%% plot parameter eta and output table

parameter = 'eta'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df = df.rename(columns = {'eta':r'$\eta$'})

df.style.format(precision=6).to_latex(calibration_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+parameter+'.csv',float_format='%.6f')

#%% plot parameter T in non patenting sector and output table

parameter = 'T'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,0]

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df.style.format(precision=6).to_latex(calibration_path+parameter+'_non_patenting.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+parameter+'_non_patenting.csv',float_format='%.6f')

#%% plot parameter T in patenting sector and output table

parameter = 'T'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df.style.format(precision=6).to_latex(calibration_path+parameter+'_patenting.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(calibration_path+parameter+'_patenting.csv',float_format='%.6f')

#%% Harmonization to delta US

p_cf = p_baseline.copy()
p_cf.delta[...,1] = np.minimum(p_cf.delta[...,1],p_cf.delta[0,1])

sol, dyn_sol_cf = dyn_fixed_point_solver(p_cf, sol_init=sol_baseline,Nt=25,
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
dyn_sol_cf.compute_non_solver_quantities(p_cf)
dyn_sol_cf.sol_fin.compute_consumption_equivalent_welfare(p_cf,sol_baseline)
dyn_sol_cf.sol_fin.compute_world_welfare_changes(p_cf,sol_baseline)

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World Negishi',
                                     'World Equal'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_cf.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_cf.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_cf.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_cf.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_cf.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_cf.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_cf.sol_fin.cons_eq_pop_average_welfare_change

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Uniformization to delta US'

df.style.format(precision=5).to_latex(counterfactual_plots_path+'uniformization_to_delta_us_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(counterfactual_plots_path+'uniformization_to_delta_us_table.csv',float_format='%.5f')

#%% Unilateral patent protections counterfactuals

# recap_growth_rate = pd.DataFrame(columns = ['delta_change']+p_baseline.countries+['World'])

# for c in p_baseline.countries+['World','Uniform_delta']:
#     recap = pd.DataFrame(columns = ['delta_change','growth','world_negishi','world_equal']+p_baseline.countries)
#     if variation == 'baseline':
#         local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline+'/'
#     else:
#         local_path = \
#             f'counterfactual_results/unilateral_patent_protection/baseline_{baseline}_{variation}/'
#     print(c)
#     if c in p_baseline.countries:
#         idx_country = p_baseline.countries.index(c)
#     country_path = local_path+c+'/'
#     files_in_dir = next(os.walk(country_path))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list.sort(key=float)
#     for i,run in enumerate(run_list):
#         p = parameters()
#         p.load_run(country_path+run+'/')
#         if p.guess is not None:
#             sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
#             sol_c.scale_P(p)
#             sol_c.compute_non_solver_quantities(p)
#             sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
#             sol_c.compute_world_welfare_changes(p,sol_baseline)
#             if c == 'World':
#                 recap_growth_rate.loc[run,'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]            
#             recap_growth_rate.loc[run,c] = sol_c.g
    
# #%% Unilateral patent protections counterfactuals
recap_growth_rate = pd.DataFrame(columns = ['delta_change']+p_baseline.countries+['World'])

for c in p_baseline.countries+['World','Uniform_delta']:
# for c in ['Uniform_delta']:
    recap = pd.DataFrame(columns = ['delta_change','growth','world_negishi','world_equal']+p_baseline.countries)
    if variation == 'baseline':
        local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline+'/'
    else:
        local_path = \
            f'counterfactual_results/unilateral_patent_protection/baseline_{baseline}_{variation}/'
    print(c)
    if c in p_baseline.countries:
        idx_country = p_baseline.countries.index(c)
    country_path = local_path+c+'/'
    files_in_dir = next(os.walk(country_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    for i,run in enumerate(run_list):
        p = parameters()
        # p.load_data(country_path+run+'/')
        p.load_run(country_path+run+'/')
        if p.guess is not None:
            sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
            sol_c.scale_P(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            sol_c.compute_world_welfare_changes(p,sol_baseline)
            if c in p_baseline.countries:
                recap.loc[run, 'delta_change'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
            if c == 'World':
                recap.loc[run, 'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]
                recap_growth_rate.loc[run,'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]
            if c == 'Uniform_delta':
                recap.loc[run, 'delta_change'] = p.delta[0,1]
            if c == 'Upper_uniform_delta':
                recap.loc[run,'delta_change'] = np.logspace(-2,0,len(run_list))[i]
            if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
                recap.loc[run, 'delta_change'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
            recap.loc[run, 'growth'] = sol_c.g
            recap.loc[run, 'world_negishi'] = sol_c.cons_eq_negishi_welfare_change
            recap.loc[run, 'world_equal'] = sol_c.cons_eq_pop_average_welfare_change
            recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
            recap_growth_rate.loc[run,c] = sol_c.g

    fig,ax = plt.subplots()
    plt.xscale('log')
    ax.set_ylabel('Welfare change (%)')
    if c in p_baseline.countries:
        ax.set_xlabel(r'Proportional change of $\delta$')
    if c == 'World':
        ax.set_xlabel(r'Proportional change of $\delta$ of all countries')
    if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
        ax.set_xlabel(r'Proportional change of $\tau$ of all countries in the patenting sector')
        ax.set_xlim(0.98,1.02)
        ax.set_ylim(0.98,1.02)
        plt.xscale('linear')
    if c == 'Uniform_delta' or c == 'Upper_uniform_delta':
        ax.set_xlabel(r'Harmonized $\delta$')
        plt.axvline(x=p_baseline.delta[0,1], lw = 1, color = 'k')
        xt = ax.get_xticks() 
        xt=np.append(xt,p_baseline.delta[0,1])
        xtl=xt.tolist()
        xtl[-1]=r'$\delta_{US}$'
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl,rotation=45)
        
    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap.delta_change,recap[country]*100-100,color=Category18[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi']*100-100,color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal']*100-100,color='k',ls=':',label='World Equal')
    ax.legend(loc=([1.02,0]))
    
    if c == 'Uniform_delta':
        for save_format in save_formats:
            plt.savefig(counterfactual_plots_path+c+'_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
    plt.show()
    
    if c == 'Uniform_delta':
        caption = 'Consumption equivalent welfares in the harmonized delta counterfactual change of all countries'
        recap = recap.rename(columns = {'delta_change':'delta'})
        
        recap.style.to_latex(counterfactual_plots_path+c+'_unilateral_patent_protection_counterfactual.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        recap.to_csv(counterfactual_plots_path+c+'_unilateral_patent_protection_counterfactual.csv')
    
        delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_baseline.delta[0,1]))].to_frame()
        delta_US_values.style.to_latex(counterfactual_plots_path+c+'_US_values.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        delta_US_values.to_csv(counterfactual_plots_path+c+'_US_values.csv')

#%% Counterfactual growth rates

for with_world in [True,False]:

    fig,ax = plt.subplots()
    
    ax.set_ylabel('Growth rate (%)')
    ax.set_xlabel(r'Proportional change of $\delta$')
    
    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap_growth_rate.delta_change,
                recap_growth_rate[country]*100,
                color=Category18[i],
                label=countries_names[country])
    if with_world:
        ax.plot(recap_growth_rate.delta_change,
                recap_growth_rate['World']*100,color='grey',
                label='All countries',ls='--')
    # ax.legend(loc=[1.02,0.02])
    ax.legend(fontsize = 5,ncol=2)
    plt.xscale('log')
    for save_format in save_formats:
        if with_world:
            save_name = counterfactual_plots_path+'growth_rate_unilateral_patent_protection_counterfactual_with_world.'
        else:
            save_name = counterfactual_plots_path+'growth_rate_unilateral_patent_protection_counterfactual.'
        plt.savefig(save_name+save_format,format=save_format)
    plt.show()
    
    caption = 'Counterfactual growth rates'
    
    
    recap_growth_rate.style.to_latex(save_name+'tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap_growth_rate.to_csv(save_name+'csv')
    
#%%  Counterfactuals with transitional dynamics

#%% Unilateral patent protections counterfactuals with dynamics

# for c in p_baseline.countries+['World','Uniform_delta']:
for c in p_baseline.countries+['World','Uniform_delta','trade_cost_eq_trips_all_countries_pat_sectors']:
# for c in ['trade_cost_eq_trips_all_countries_pat_sectors']:
    recap = pd.DataFrame(columns = ['delta_change','world_negishi','world_equal']+p_baseline.countries)
    if variation == 'baseline':
        local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline+'/'
    else:
        local_path = \
            f'counterfactual_results/unilateral_patent_protection/baseline_{baseline}_{variation}/'
    print(c)
    if c in p_baseline.countries:
        idx_country = p_baseline.countries.index(c)
    country_path = local_path+c+'/'
    files_in_dir = next(os.walk(country_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    for i,run in enumerate(run_list):
        p = parameters()
        p.load_run(country_path+run+'/')
        if p.guess is not None:
            sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
            sol_c.scale_P(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        if p.dyn_guess is not None:
            dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                    Nt=25,t_inf=500,
                                                    sol_init = sol_baseline,
                                                    sol_fin = sol_c)
            dyn_sol_c.compute_non_solver_quantities(p)
        if c in p_baseline.countries:
            recap.loc[run, 'delta_change'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
        if c == 'World':
            recap.loc[run, 'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]
        if c == 'Uniform_delta':
            recap.loc[run, 'delta_change'] = p.delta[0,1]
        if c == 'Upper_uniform_delta':
            recap.loc[run,'delta_change'] = np.logspace(-2,0,len(run_list))[i]
        if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
            recap.loc[run, 'delta_change'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
        recap.loc[run, 'world_negishi'] = dyn_sol_c.cons_eq_negishi_welfare_change
        recap.loc[run, 'world_equal'] = dyn_sol_c.cons_eq_pop_average_welfare_change
        recap.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare

    fig,ax = plt.subplots()
    # plt.grid(linestyle='-',linewidth = .1,color='grey')
    plt.xscale('log')
    
    ax.set_ylabel('Welfare change (%)')
    if c in p_baseline.countries:
        ax.set_xlabel(r'Proportional change of $\delta$')
    if c == 'World':
        ax.set_xlabel(r'Proportional change of $\delta$ of all countries')
    if c == 'Uniform_delta' or c == 'Upper_uniform_delta':
        ax.set_xlabel(r'Harmonized $\delta$')
        plt.axvline(x=p_baseline.delta[0,1], lw = 1, color = 'k')
        xt = ax.get_xticks() 
        xt=np.append(xt,p_baseline.delta[0,1])
        xtl=xt.tolist()
        xtl[-1]=r'$\delta_{US}$'
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)
    if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
        ax.set_xlabel(r'Proportional change of $\tau$ of all countries in the patenting sector')
        ax.set_xlim(0.98,1.02)
        ax.set_ylim(-2,2)
        plt.xscale('linear')

    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap.delta_change,recap[country]*100-100,color=Category18[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi']*100-100,color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal']*100-100,color='k',ls=':',label='World Equal')

    # ax.legend(loc=[1.02,0.02])
    if c == 'USA':
        plt.legend(fontsize = 5,ncol=2)
        # legend = plt.legend(frameon = 1)
        # frame = legend.get_frame()
        # frame.set_facecolor('white')
        # frame.set_edgecolor('white')
        # # legend = plt.legend(loc="upper right", edgecolor="black")
        # legend.get_frame().set_alpha(1)
        # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        # pass
    
    # if c in ['USA','EUR','JAP','CHN']:
    #     ax.set_xlabel('')
    # if c in ['EUR','CHN','IND']:
    #     ax.set_ylabel('')

    for save_format in save_formats:
        plt.savefig(counterfactual_plots_path+c+'_dyn_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
    plt.show()
    
    if c in p_baseline.countries:
        caption = 'Consumption equivalent welfares in the unilateral patent protection counterfactual of '+countries_names[c]
    if c == 'World':
        caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
    if c == 'Uniform_delta':
        caption = 'Consumption equivalent welfares in the harmonized delta counterfactual change of all countries'
        recap = recap.rename(columns = {'delta_change':'delta'})
    if c == 'Upper_uniform_delta':
        caption = 'Consumption equivalent welfares in the partially harmonized delta counterfactual change of all countries'
        recap = recap.rename(columns = {'delta_change':'delta'})
    if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
        caption = 'Consumption equivalent welfares in the counterfactual change of delta pre-TRIPS and trade costs of the patenting sectors'
        recap = recap.rename(columns = {'delta_change':'tau_change'})
    
    recap.style.to_latex(counterfactual_plots_path+c+'_dyn_unilateral_patent_protection_counterfactual.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap.to_csv(counterfactual_plots_path+c+'_dyn_unilateral_patent_protection_counterfactual.csv')
    
    if c == 'Uniform_delta':
        delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_baseline.delta[0,1]))].to_frame()
        delta_US_values.style.to_latex(counterfactual_plots_path+c+'_dyn_US_values.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        delta_US_values.to_csv(counterfactual_plots_path+c+'_dyn_US_values.csv')
        
    if c == 'Upper_uniform_delta':
        delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_baseline.delta[0,1]))].to_frame()
        delta_US_values.style.to_latex(counterfactual_plots_path+c+'_dyn_US_values.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        delta_US_values.to_csv(counterfactual_plots_path+c+'_dyn_US_values.csv')
        
#%% Nash table with transitional dynamics

all_nashes = pd.read_csv('nash_eq_recaps/dyn_deltas.csv')
all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) & (all_nashes.variation.astype(str) == variation)]

p_nash = p_baseline.copy()
p_nash.delta[:,1] = run_nash[p_baseline.countries].values.squeeze()

sol, dyn_sol_nash = dyn_fixed_point_solver(p_nash, sol_init=sol_baseline,Nt=25,
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
dyn_sol_nash.sol_fin.compute_consumption_equivalent_welfare(p_nash,sol_baseline)
dyn_sol_nash.sol_fin.compute_world_welfare_changes(p_nash,sol_baseline)

m_nash = m_baseline.copy()
m_nash.compute_moments(dyn_sol_nash.sol_fin,p_nash)
m_nash.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_nash.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Nash equilibrium'

df.style.format(precision=5).to_latex(nash_coop_path+'dyn_Nash_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(nash_coop_path+'dyn_Nash_table.csv',float_format='%.5f')

write_calibration_results(nash_coop_path+'dyn_Nash',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')

#%% Coop equal weights table with transitional dynamics

all_coop_equales = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
                                     & (all_coop_equales.variation == variation)
                                     & (all_coop_equales.aggregation_method == 'pop_weighted')]

p_coop_equal = p_baseline.copy()
p_coop_equal.delta[:,1] = run_coop_equal[p_baseline.countries].values.squeeze()

sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_coop_equal, sol_init=sol_baseline,Nt=25,
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
dyn_sol_coop_equal.compute_non_solver_quantities(p_coop_equal)
dyn_sol_coop_equal.sol_fin.compute_consumption_equivalent_welfare(p_coop_equal,sol_baseline)
dyn_sol_coop_equal.sol_fin.compute_world_welfare_changes(p_coop_equal,sol_baseline)

m_coop_equal = m_baseline.copy()
m_coop_equal.compute_moments(dyn_sol_coop_equal.sol_fin,p_coop_equal)
m_coop_equal.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_equal.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with population weights'

df.style.format(precision=5).to_latex(nash_coop_path+'dyn_Coop_population_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(nash_coop_path+'dyn_Coop_population_weights_table.csv',float_format='%.5f')

write_calibration_results(nash_coop_path+'dyn_Coop_population_weights',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

#%% Coop negishi weights table with transitional dynamics

all_coop_negishies = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
                                     & (all_coop_negishies.variation == variation)
                                     & (all_coop_negishies.aggregation_method == 'negishi')]

p_coop_negishi = p_baseline.copy()
p_coop_negishi.delta[:,1] = run_coop_negishi[p_baseline.countries].values.squeeze()

sol, dyn_sol_coop_negishi = dyn_fixed_point_solver(p_coop_negishi, sol_init=sol_baseline,Nt=25,
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
dyn_sol_coop_negishi.compute_non_solver_quantities(p_coop_negishi)
dyn_sol_coop_negishi.sol_fin.compute_consumption_equivalent_welfare(p_coop_negishi,sol_baseline)
dyn_sol_coop_negishi.sol_fin.compute_world_welfare_changes(p_coop_negishi,sol_baseline)

m_coop_negishi = m_baseline.copy()
m_coop_negishi.compute_moments(dyn_sol_coop_negishi.sol_fin,p_coop_negishi)
m_coop_negishi.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_negishi.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with Negishi weights'

df.style.format(precision=5).to_latex(nash_coop_path+'dyn_Coop_negishi_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(nash_coop_path+'dyn_Coop_negishi_weights_table.csv',float_format='%.5f')

write_calibration_results(nash_coop_path+'dyn_Coop_negishi_weights',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')

#%% Write excel spreadsheets for partial calibration pre-TRIPS

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline}_variations/{pre_trips_variation}/')
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
m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline}_variations/{pre_trips_variation}/')
m_pre.compute_moments(sol_pre,p_pre)
m_pre.compute_moments_deviations()

write_calibration_results(pre_TRIPS_plots_path+'pre_TRIPS_partial_calibration',p_pre,m_pre,sol_pre,commentary = '')

#%% pre-TRIPS calibration and counterfactual

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
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

p_pre_cf = p_baseline.copy()
p_pre_cf.delta[...,1] = p_pre.delta[...,1]

_, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
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
sol_pre_cf.scale_P(p_pre_cf)
sol_pre_cf.compute_non_solver_quantities(p_pre_cf)
sol_pre_cf.compute_consumption_equivalent_welfare(p_pre_cf,sol_baseline)
sol_pre_cf.compute_world_welfare_changes(p_pre_cf,sol_baseline)

_, dyn_sol_pre_cf = dyn_fixed_point_solver(p_pre_cf, sol_baseline,sol_fin=sol_pre_cf,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf.compute_non_solver_quantities(p_pre_cf)

p_pre_cf_fix_north = p_baseline.copy()
p_pre_cf_fix_north.delta[...,1] = p_pre.delta[...,1]
for country_idx in [[p_baseline.countries.index(c) for c in rich_countries]]:
    p_pre_cf_fix_north.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north = fixed_point_solver(p_pre_cf_fix_north,context = 'counterfactual',x0=p_pre_cf_fix_north.guess,
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
sol_pre_cf_fix_north.scale_P(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)
sol_pre_cf_fix_north.compute_consumption_equivalent_welfare(p_pre_cf_fix_north,sol_baseline)
sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf_fix_north,sol_baseline)

_, dyn_sol_pre_cf_fix_north = dyn_fixed_point_solver(p_pre_cf_fix_north, sol_baseline,sol_fin=sol_pre_cf_fix_north,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)

p_pre_trips_increase_only = p_baseline.copy()
p_pre_trips_increase_only.delta[...,1] = np.maximum(p_pre.delta[...,1],p_baseline.delta[...,1])

_, sol_pre_trips_increase_only = fixed_point_solver(p_pre_trips_increase_only,context = 'counterfactual',x0=p_pre_trips_increase_only.guess,
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
sol_pre_trips_increase_only.scale_P(p_pre_trips_increase_only)
sol_pre_trips_increase_only.compute_non_solver_quantities(p_pre_trips_increase_only)
sol_pre_trips_increase_only.compute_consumption_equivalent_welfare(p_pre_trips_increase_only,sol_baseline)
sol_pre_trips_increase_only.compute_world_welfare_changes(p_pre_trips_increase_only,sol_baseline)

_, dyn_sol_pre_trips_increase_only = dyn_fixed_point_solver(p_pre_trips_increase_only, 
                                                            sol_baseline,sol_fin=sol_pre_trips_increase_only,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_trips_increase_only.compute_non_solver_quantities(p_pre_trips_increase_only)

p_pre_cf_fix_north_and_tariff = p_baseline.copy()
p_pre_cf_fix_north_and_tariff.delta[...,1] = p_pre.delta[...,1]
p_pre_cf_fix_north_and_tariff.tariff = p_pre.tariff.copy()
for country_idx in [[p_baseline.countries.index(c) for c in rich_countries]]:
    p_pre_cf_fix_north_and_tariff.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north_and_tariff = fixed_point_solver(p_pre_cf_fix_north_and_tariff,
                                                     context = 'counterfactual',
                        x0=p_pre_cf_fix_north_and_tariff.guess,
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
sol_pre_cf_fix_north_and_tariff.scale_P(p_pre_cf_fix_north_and_tariff)
sol_pre_cf_fix_north_and_tariff.compute_non_solver_quantities(p_pre_cf_fix_north_and_tariff)
sol_pre_cf_fix_north_and_tariff.compute_consumption_equivalent_welfare(p_pre_cf_fix_north_and_tariff,
                                                                    sol_baseline)
sol_pre_cf_fix_north_and_tariff.compute_world_welfare_changes(p_pre_cf_fix_north_and_tariff,
                                                           sol_baseline)

_, dyn_sol_pre_cf_fix_north_and_tariff = dyn_fixed_point_solver(p_pre_cf_fix_north_and_tariff, 
                        sol_baseline,sol_fin=sol_pre_cf_fix_north_and_tariff,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf_fix_north_and_tariff.compute_non_solver_quantities(p_pre_cf_fix_north_and_tariff)

p_pre_cf_fix_north_and_tariff_in_pat_sect = p_baseline.copy()
p_pre_cf_fix_north_and_tariff_in_pat_sect.delta[...,1] = p_pre.delta[...,1]
p_pre_cf_fix_north_and_tariff_in_pat_sect.tariff[...,1] = p_pre.tariff[...,1]
for country_idx in [[p_baseline.countries.index(c) for c in rich_countries]]:
    p_pre_cf_fix_north_and_tariff_in_pat_sect.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north_and_tariff_in_pat_sect = fixed_point_solver(p_pre_cf_fix_north_and_tariff_in_pat_sect,
                                                     context = 'counterfactual',
                        x0=p_pre_cf_fix_north_and_tariff_in_pat_sect.guess,
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
sol_pre_cf_fix_north_and_tariff_in_pat_sect.scale_P(p_pre_cf_fix_north_and_tariff_in_pat_sect)
sol_pre_cf_fix_north_and_tariff_in_pat_sect.compute_non_solver_quantities(p_pre_cf_fix_north_and_tariff_in_pat_sect)
sol_pre_cf_fix_north_and_tariff_in_pat_sect.compute_consumption_equivalent_welfare(p_pre_cf_fix_north_and_tariff_in_pat_sect,sol_baseline)
sol_pre_cf_fix_north_and_tariff_in_pat_sect.compute_world_welfare_changes(p_pre_cf_fix_north_and_tariff_in_pat_sect,sol_baseline)

_, dyn_sol_pre_cf_fix_north_and_tariff_in_pat_sect = dyn_fixed_point_solver(p_pre_cf_fix_north_and_tariff_in_pat_sect, 
                        sol_baseline,sol_fin=sol_pre_cf_fix_north_and_tariff_in_pat_sect,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf_fix_north_and_tariff_in_pat_sect.compute_non_solver_quantities(
    p_pre_cf_fix_north_and_tariff_in_pat_sect)

p_pre_cf_tariff = p_baseline.copy()
p_pre_cf_tariff.tariff[...] = p_pre.tariff[...]

_, sol_pre_cf_tariff = fixed_point_solver(p_pre_cf_tariff,context = 'counterfactual',x0=p_pre_cf_tariff.guess,
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
sol_pre_cf_tariff.scale_P(p_pre_cf_tariff)
sol_pre_cf_tariff.compute_non_solver_quantities(p_pre_cf_tariff)
sol_pre_cf_tariff.compute_consumption_equivalent_welfare(p_pre_cf_tariff,sol_baseline)
sol_pre_cf_tariff.compute_world_welfare_changes(p_pre_cf_tariff,sol_baseline)

_, dyn_sol_pre_cf_tariff = dyn_fixed_point_solver(p_pre_cf_tariff, sol_baseline,sol_fin=sol_pre_cf_tariff,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf_tariff.compute_non_solver_quantities(p_pre_cf_tariff)

p_pre_cf_tariff_in_pat_sect = p_baseline.copy()
p_pre_cf_tariff_in_pat_sect.tariff[...,1] = p_pre.tariff[...,1]

_, sol_pre_cf_tariff_in_pat_sect = fixed_point_solver(p_pre_cf_tariff_in_pat_sect
                        ,context = 'counterfactual',x0=p_pre_cf_tariff_in_pat_sect.guess,
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
sol_pre_cf_tariff_in_pat_sect.scale_P(p_pre_cf_tariff_in_pat_sect)
sol_pre_cf_tariff_in_pat_sect.compute_non_solver_quantities(p_pre_cf_tariff_in_pat_sect)
sol_pre_cf_tariff_in_pat_sect.compute_consumption_equivalent_welfare(p_pre_cf_tariff_in_pat_sect,sol_baseline)
sol_pre_cf_tariff_in_pat_sect.compute_world_welfare_changes(p_pre_cf_tariff_in_pat_sect,sol_baseline)

_, dyn_sol_pre_cf_tariff_in_pat_sect = dyn_fixed_point_solver(p_pre_cf_tariff_in_pat_sect, 
                        sol_baseline,sol_fin=sol_pre_cf_tariff_in_pat_sect,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_pre_cf_tariff_in_pat_sect.compute_non_solver_quantities(p_pre_cf_tariff_in_pat_sect)

modified_countries_names = {'USA': 'USA',
 'EUR': 'Europe',
 'JAP': 'Japan',
 'KOR':'Korea',
 'CAN':'Canada',
 'MEX':'Mexico',
 'RUS':'Russia',
 'CHN': 'China',
 'BRA': 'Brazil',
 'IND': 'India',
 'ZAF': 'S. Africa',
 'ROW': 'Rest of\nthe world'}

df = pd.DataFrame(
    index = pd.Index([modified_countries_names[c] for c in p_baseline.countries]+['World\nNegishi','World\nEqual'],
                                       name = 'country')
    )

df['delta_baseline'] = p_baseline.delta[...,1].tolist()+[np.nan,np.nan]
df['delta_1992'] = p_pre.delta[...,1].tolist()+[np.nan,np.nan]
df['static_welfare_change'] = sol_pre_cf.cons_eq_welfare.tolist()+[
    sol_pre_cf.cons_eq_negishi_welfare_change,sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change'] = dyn_sol_pre_cf.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf.cons_eq_negishi_welfare_change,dyn_sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_cf_tariff'] = sol_pre_cf_tariff.cons_eq_welfare.tolist()+[
    sol_pre_cf_tariff.cons_eq_negishi_welfare_change,sol_pre_cf_tariff.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_cf_tariff'] = dyn_sol_pre_cf_tariff.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_tariff.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_tariff.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_cf_tariff_in_pat_sect'] = sol_pre_cf_tariff_in_pat_sect.cons_eq_welfare.tolist()+[
    sol_pre_cf_tariff_in_pat_sect.cons_eq_negishi_welfare_change,sol_pre_cf_tariff_in_pat_sect.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_cf_tariff_in_pat_sect'] = dyn_sol_pre_cf_tariff_in_pat_sect.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_tariff_in_pat_sect.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_tariff_in_pat_sect.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north_and_cf_tariff'] = sol_pre_cf_fix_north_and_tariff.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north_and_tariff.cons_eq_negishi_welfare_change,
    sol_pre_cf_fix_north_and_tariff.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north_and_cf_tariff'] = dyn_sol_pre_cf_fix_north_and_tariff.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north_and_tariff.cons_eq_negishi_welfare_change,
    dyn_sol_pre_cf_fix_north_and_tariff.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north_and_cf_tariff_in_pat_sect'] = sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_negishi_welfare_change,
    sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north_and_cf_tariff_in_pat_sect'] = dyn_sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_negishi_welfare_change,
    dyn_sol_pre_cf_fix_north_and_tariff_in_pat_sect.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_increase_only'] = sol_pre_trips_increase_only.cons_eq_welfare.tolist()+[
    sol_pre_trips_increase_only.cons_eq_negishi_welfare_change,sol_pre_trips_increase_only.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_increase_only'] = dyn_sol_pre_trips_increase_only.cons_eq_welfare.tolist()+[
    dyn_sol_pre_trips_increase_only.cons_eq_negishi_welfare_change,dyn_sol_pre_trips_increase_only.cons_eq_pop_average_welfare_change
    ]
grey_rgb = (105/256,105/256,105/256)
# grey_rgb = (0,0,0)

for col in ['static_welfare_change','dynamic_welfare_change',
            'static_welfare_change_fixed_delta_north','dynamic_welfare_change_fixed_delta_north',
            'static_welfare_change_fixed_delta_north_and_cf_tariff','dynamic_welfare_change_fixed_delta_north_and_cf_tariff',
            'static_welfare_change_fixed_delta_north_and_cf_tariff_in_pat_sect','dynamic_welfare_change_fixed_delta_north_and_cf_tariff_in_pat_sect',
            'static_welfare_change_increase_only','dynamic_welfare_change_increase_only']:

    fig,ax = plt.subplots()
    # ax.bar(df.index, df['static welfare change']*100-100)
    ax.barh(df.index, df[col]*100-100, 
            color = Category18[:len(p_baseline.countries)]+[grey_rgb,grey_rgb],
           # color = Category18[:len(p_baseline.countries)+2],
           # hatch = ['']*len(p_baseline.countries)+['/','x']
           )
    ax.invert_yaxis()
    ax.set_xlabel('Welfare change (%)')
    
    for save_format in save_formats:
        plt.savefig(pre_TRIPS_plots_path+col+'.'+save_format,format=save_format)
    
    plt.show()

df.loc['growth_rate','delta_baseline'] = sol_baseline.g
df.loc['growth_rate','static_welfare_change'] = sol_pre_cf.g
df.loc['growth_rate','static_welfare_change_cf_tariff'] = sol_pre_cf_tariff.g
df.loc['growth_rate','static_welfare_change_cf_tariff_in_pat_sect'] = sol_pre_cf_tariff_in_pat_sect.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north_and_cf_tariff'] = sol_pre_cf_fix_north_and_tariff.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north_and_cf_tariff_in_pat_sect'
       ] = sol_pre_cf_fix_north_and_tariff_in_pat_sect.g
df.loc['growth_rate','static_welfare_change_increase_only'] = sol_pre_trips_increase_only.g

caption = 'Pre TRIPS calibration and counterfacual'

df.style.format(precision=5).to_latex(pre_TRIPS_plots_path+'pre_trips.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(pre_TRIPS_plots_path+'pre_trips.csv',float_format='%.5f')

#%% implementation of TRIPS counterfactual

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
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

p_post_cf = p_pre.copy()
p_post_cf.delta[...,1] = p_baseline.delta[...,1]

_, sol_post_cf = fixed_point_solver(p_post_cf,
                        context = 'counterfactual',
                        x0=p_post_cf.guess,
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

sol_post_cf.scale_P(p_post_cf)
sol_post_cf.compute_non_solver_quantities(p_post_cf)
sol_post_cf.compute_consumption_equivalent_welfare(p_post_cf,sol_pre)
sol_post_cf.compute_world_welfare_changes(p_post_cf,sol_pre)

_, dyn_sol_post_cf = dyn_fixed_point_solver(p_post_cf, sol_pre,sol_fin=sol_post_cf,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_post_cf.compute_non_solver_quantities(p_post_cf)

p_post_cf_fix_north = p_pre.copy()
p_post_cf_fix_north.delta[...,1] = p_baseline.delta[...,1]
for country_idx in [[p_baseline.countries.index(c) for c in rich_countries]]:
    p_post_cf_fix_north.delta[country_idx,1] = p_pre.delta[country_idx,1]

_, sol_post_cf_fix_north = fixed_point_solver(p_post_cf_fix_north,
                        context = 'counterfactual',
                        x0=p_post_cf_fix_north.guess,
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
sol_post_cf_fix_north.scale_P(p_post_cf_fix_north)
sol_post_cf_fix_north.compute_non_solver_quantities(p_post_cf_fix_north)
sol_post_cf_fix_north.compute_consumption_equivalent_welfare(p_post_cf_fix_north,sol_pre)
sol_post_cf_fix_north.compute_world_welfare_changes(p_post_cf_fix_north,sol_pre)

_, dyn_sol_post_cf_fix_north = dyn_fixed_point_solver(p_post_cf_fix_north, sol_pre,
                        sol_fin=sol_post_cf_fix_north,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_post_cf_fix_north.compute_non_solver_quantities(p_post_cf_fix_north)

p_post_trips_decrease_only = p_pre.copy()
p_post_trips_decrease_only.delta[...,1] = np.minimum(p_pre.delta[...,1],p_baseline.delta[...,1])

_, sol_post_trips_decrease_only = fixed_point_solver(p_post_trips_decrease_only,
                        context = 'counterfactual',x0=p_post_trips_decrease_only.guess,
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
sol_post_trips_decrease_only.scale_P(p_post_trips_decrease_only)
sol_post_trips_decrease_only.compute_non_solver_quantities(p_post_trips_decrease_only)
sol_post_trips_decrease_only.compute_consumption_equivalent_welfare(p_post_trips_decrease_only,sol_pre)
sol_post_trips_decrease_only.compute_world_welfare_changes(p_post_trips_decrease_only,sol_pre)

_, dyn_sol_post_trips_decrease_only = dyn_fixed_point_solver(p_post_trips_decrease_only, 
                        sol_pre,sol_fin=sol_post_trips_decrease_only,
                        Nt=25,t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol_post_trips_decrease_only.compute_non_solver_quantities(p_post_trips_decrease_only)

modified_countries_names = {'USA': 'USA',
 'EUR': 'Europe',
 'JAP': 'Japan',
 'KOR':'Korea',
 'CAN':'Canada',
 'MEX':'Mexico',
 'RUS':'Russia',
 'CHN': 'China',
 'BRA': 'Brazil',
 'IND': 'India',
 'ZAF': 'S. Africa',
 'ROW': 'Rest of\nthe world'}

df = pd.DataFrame(
    index = pd.Index([modified_countries_names[c] for c in p_baseline.countries]+['World\nNegishi','World\nEqual'],
                                       name = 'country')
    )

df['delta_baseline'] = p_baseline.delta[...,1].tolist()+[np.nan,np.nan]
df['delta_1992'] = p_pre.delta[...,1].tolist()+[np.nan,np.nan]
df['static_welfare_change'] = sol_post_cf.cons_eq_welfare.tolist()+[
    sol_post_cf.cons_eq_negishi_welfare_change,sol_post_cf.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change'] = dyn_sol_post_cf.cons_eq_welfare.tolist()+[
    dyn_sol_post_cf.cons_eq_negishi_welfare_change,dyn_sol_post_cf.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north'] = sol_post_cf_fix_north.cons_eq_welfare.tolist()+[
    sol_post_cf_fix_north.cons_eq_negishi_welfare_change,sol_post_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north'] = dyn_sol_post_cf_fix_north.cons_eq_welfare.tolist()+[
    dyn_sol_post_cf_fix_north.cons_eq_negishi_welfare_change,dyn_sol_post_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_decrease_only'] = sol_post_trips_decrease_only.cons_eq_welfare.tolist()+[
    sol_post_trips_decrease_only.cons_eq_negishi_welfare_change,sol_post_trips_decrease_only.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_decrease_only'] = dyn_sol_post_trips_decrease_only.cons_eq_welfare.tolist()+[
    dyn_sol_post_trips_decrease_only.cons_eq_negishi_welfare_change,dyn_sol_post_trips_decrease_only.cons_eq_pop_average_welfare_change
    ]
grey_rgb = (105/256,105/256,105/256)
# grey_rgb = (0,0,0)

for col in ['static_welfare_change','dynamic_welfare_change',
            'static_welfare_change_fixed_delta_north','dynamic_welfare_change_fixed_delta_north',
            'static_welfare_change_decrease_only','dynamic_welfare_change_decrease_only']:

    fig,ax = plt.subplots()
    # ax.bar(df.index, df['static welfare change']*100-100)
    ax.barh(df.index, df[col]*100-100, 
            color = Category18[:len(p_baseline.countries)]+[grey_rgb,grey_rgb],
           # color = Category18[:len(p_baseline.countries)+2],
           # hatch = ['']*len(p_baseline.countries)+['/','x']
           )
    ax.invert_yaxis()
    ax.set_xlabel('Welfare change (%)')
    
    for save_format in save_formats:
        plt.savefig(post_trips_path+col+'.'+save_format,format=save_format)
    
    plt.show()

df.loc['growth_rate','delta_baseline'] = sol_baseline.g
df.loc['growth_rate','static_welfare_change'] = sol_post_cf.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north'] = sol_post_cf_fix_north.g
df.loc['growth_rate','static_welfare_change_increase_only'] = sol_post_trips_decrease_only.g

caption = 'TRIPS implementation counterfacual'

df.style.format(precision=5).to_latex(post_trips_path+'trips_implementation.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(post_trips_path+'trips_implementation.csv',float_format='%.5f')

#%% tariff equivalent of TRIPS for specific countries

cf_path = 'counterfactual_recaps/unilateral_patent_protection/'

def find_zeros(df_welfare,country):
    x1 = df_welfare.loc[df_welfare[country]>1].iloc[-1]['delt']
    x2 = df_welfare.loc[df_welfare[country]<1].iloc[0]['delt']
    y1 = df_welfare.loc[df_welfare[country]>1].iloc[-1][country]-1
    y2 = df_welfare.loc[df_welfare[country]<1].iloc[0][country]-1
    
    return (y2*x1-x2*y1)/(y2-y1)

countries = ['CHN','IND','RUS']
exercises = ['tariff_eq_trips_exp_pat_sect','dyn_tariff_eq_trips_exp_pat_sect']

df = pd.DataFrame(index = [countries_names[c] for c in countries])
data_for_plot = {}

for country in countries:
    for i,exercise in enumerate(exercises):
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
        if exercise == 'tariff_eq_trips_exp_pat_sect':
            df_welfare = pd.read_csv(local_path+country+'_tariff_eq_trips_exp_pat_sect'+'.csv')
        if exercise == 'dyn_tariff_eq_trips_exp_pat_sect':
            df_welfare = pd.read_csv(local_path+'dyn_'+country+'_tariff_eq_trips_exp_pat_sect'+'.csv')
        if i == 0:
            data_for_plot['delt'] = df_welfare['delt'] 
                
        df.loc[countries_names[country],exercise] = find_zeros(df_welfare,country)
        if exercise == 'dyn_tariff_eq_trips_exp_pat_sect':
            data_for_plot[country] = df_welfare[country]

caption = 'Tariff equivalent of TRIPS when changing country-specific trade costs (exports in patenting sector)'

df.style.format(precision=5).to_latex(pre_TRIPS_plots_path+'tariff_eq_trips_exp_pat_sect.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(pre_TRIPS_plots_path+'tariff_eq_trips_exp_pat_sect.csv',float_format='%.5f')

#%% Coop and Nash equilibria with doubled trade costs in patenting sector

#%% Nash table with transitional dynamics with doubled trade costs in patenting sector

# all_nashes = pd.read_csv('nash_eq_recaps/dyn_deltas.csv')
# all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

# run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) 
#                          & (all_nashes.variation.astype(str) == variation_with_doubled_tau_in_pat_sect)]

# p_pre = parameters()
# p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')
# _, sol_pre = fixed_point_solver(p_pre,context = 'counterfactual',x0=p_pre.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_pre.scale_P(p_pre)
# sol_pre.compute_non_solver_quantities(p_pre)

# m_pre = moments()
# m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')

# p_nash = p_pre.copy()
# p_nash.delta[:,1] = run_nash[p_baseline.countries].values.squeeze()

# sol, dyn_sol_nash = dyn_fixed_point_solver(p_nash, sol_init=sol_pre,Nt=25,
#                                       t_inf=500,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=False,
#                         cobweb_qty='l_R',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         plot_live = False,
#                         safe_convergence=1e-8,
#                         disp_summary=False,
#                         damping = 60,
#                         max_count = 50000,
#                         accel_memory =5, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=1, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         )
# dyn_sol_nash.compute_non_solver_quantities(p_nash)
# dyn_sol_nash.sol_fin.compute_consumption_equivalent_welfare(p_nash,sol_pre)
# dyn_sol_nash.sol_fin.compute_world_welfare_changes(p_nash,sol_pre)

# m_nash = m_pre.copy()
# m_nash.compute_moments(dyn_sol_nash.sol_fin,p_nash)
# m_nash.compute_moments_deviations()

# df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
#                                    +['World aggregate according to Negishi weights',
#                                      'World aggregate according to population weights',
#                                      'Growth rate (%)'
#                                      ],
#                                    name = 'Countries'),
#                   columns = [r'$\delta$','Welfare change with transition dynamics',
#                              'Welfare change, steady state only']
#                   )
    
# for i,c in enumerate(p_baseline.countries):
#     df.loc[countries_names[c],r'$\delta$'] = p_nash.delta[i,1]
#     df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_welfare[i]
#     df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_welfare[i]

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_pop_average_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_pop_average_welfare_change

# df.loc['Growth rate (%)',
#        'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.g*100

# for col in df.columns:
#     df[col] = df[col].astype(float)

# caption = 'Nash equilibrium with doubled trade costs in the patenting sector'

# df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Nash_table_with_doubled_trade_costs_in_pat_sect.tex',
#                   caption=caption,
#                   **save_to_tex_options
#                   )

# df.to_csv(doubled_trade_costs_path+'dyn_Nash_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

# write_calibration_results(doubled_trade_costs_path+'dyn_Nash_with_doubled_trade_costs_in_pat_sect',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')


#%% Coop equal weights table with transitional dynamics with doubled trade costs in patenting sector

# all_coop_equales = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
# all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
#                                                      'variation',
#                                                      'aggregation_method'],keep='last')

# run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
#                                      & (all_coop_equales.variation == variation_with_doubled_tau_in_pat_sect)
#                                      & (all_coop_equales.aggregation_method == 'pop_weighted')]

# p_pre = parameters()
# p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')
# # p_pre.tau = p_baseline.tau.copy()
# _, sol_pre = fixed_point_solver(p_pre,context = 'counterfactual',x0=p_pre.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_pre.scale_P(p_pre)
# sol_pre.compute_non_solver_quantities(p_pre)

# m_pre = moments()
# m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')

# p_coop_equal = p_pre.copy()
# p_coop_equal.delta[:,1] = run_coop_equal[p_baseline.countries].values.squeeze()

# sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_coop_equal, sol_init=sol_pre,Nt=25,
#                                       t_inf=500,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=False,
#                         cobweb_qty='l_R',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         plot_live = False,
#                         safe_convergence=1e-8,
#                         disp_summary=False,
#                         damping = 60,
#                         max_count = 50000,
#                         accel_memory =5, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=1, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         )
# dyn_sol_coop_equal.compute_non_solver_quantities(p_coop_equal)
# dyn_sol_coop_equal.sol_fin.compute_consumption_equivalent_welfare(p_coop_equal,sol_pre)
# dyn_sol_coop_equal.sol_fin.compute_world_welfare_changes(p_coop_equal,sol_pre)

# m_coop_equal = m_pre.copy()
# m_coop_equal.compute_moments(dyn_sol_coop_equal.sol_fin,p_coop_equal)
# m_coop_equal.compute_moments_deviations()

# df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
#                                    +['World aggregate according to Negishi weights',
#                                      'World aggregate according to population weights',
#                                      'Growth rate (%)'],
#                                    name = 'Countries'),
#                   columns = [r'$\delta$','Welfare change with transition dynamics',
#                              'Welfare change, steady state only']
#                   )
    
# for i,c in enumerate(p_baseline.countries):
#     df.loc[countries_names[c],r'$\delta$'] = p_coop_equal.delta[i,1]
#     df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_welfare[i]
#     df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_welfare[i]

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_pop_average_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_pop_average_welfare_change

# df.loc['Growth rate (%)',
#        'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.g*100

# for col in df.columns:
#     df[col] = df[col].astype(float)

# caption = 'Cooperative equilibrium with population weights'

# df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Coop_population_weights_table_with_doubled_trade_costs_in_pat_sect.tex',
#                   caption=caption,
#                   **save_to_tex_options
#                   )

# df.to_csv(doubled_trade_costs_path+'dyn_Coop_population_weights_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

# write_calibration_results(doubled_trade_costs_path+'dyn_Coop_population_weights_with_doubled_trade_costs_in_pat_sect',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

#%% Coop negishi weights table with transitional dynamics with doubled trade costs in patenting sector

# all_coop_negishies = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
# all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
#                                                      'variation',
#                                                      'aggregation_method'],keep='last')

# run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
#                                      & (all_coop_negishies.variation == variation)
#                                      & (all_coop_negishies.aggregation_method == 'negishi')]

# p_pre = parameters()
# p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')
# _, sol_pre = fixed_point_solver(p_pre,context = 'counterfactual',x0=p_pre.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_pre.scale_P(p_pre)
# sol_pre.compute_non_solver_quantities(p_pre)

# p_coop_negishi = p_pre.copy()
# p_coop_negishi.delta[:,1] = run_coop_negishi[p_baseline.countries].values.squeeze()

# sol, dyn_sol_coop_negishi = dyn_fixed_point_solver(p_coop_negishi, sol_init=sol_pre,Nt=25,
#                                       t_inf=500,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=False,
#                         cobweb_qty='l_R',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         plot_live = False,
#                         safe_convergence=1e-8,
#                         disp_summary=False,
#                         damping = 60,
#                         max_count = 50000,
#                         accel_memory =5, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=1, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         )
# dyn_sol_coop_negishi.compute_non_solver_quantities(p_coop_negishi)
# dyn_sol_coop_negishi.sol_fin.compute_consumption_equivalent_welfare(p_coop_negishi,sol_pre)
# dyn_sol_coop_negishi.sol_fin.compute_world_welfare_changes(p_coop_negishi,sol_pre)

# m_coop_negishi = m_pre.copy()
# m_coop_negishi.compute_moments(dyn_sol_coop_negishi.sol_fin,p_coop_negishi)
# m_coop_negishi.compute_moments_deviations()
    
# df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
#                                    +['World aggregate according to Negishi weights',
#                                      'World aggregate according to population weights',
#                                      'Growth rate (%)'],
#                                    name = 'Countries'),
#                   columns = [r'$\delta$','Welfare change with transition dynamics',
#                              'Welfare change, steady state only']
#                   )
    
# for i,c in enumerate(p_baseline.countries):
#     df.loc[countries_names[c],r'$\delta$'] = p_coop_negishi.delta[i,1]
#     df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_welfare[i]
#     df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_welfare[i]

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to Negishi weights',
#        'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_negishi_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_pop_average_welfare_change

# df.loc['World aggregate according to population weights',
#        'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_pop_average_welfare_change

# df.loc['Growth rate (%)',
#        'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.g*100

# for col in df.columns:
#     df[col] = df[col].astype(float)

# caption = 'Cooperative equilibrium with Negishi weights'

# df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Coop_negishi_weights_table_with_doubled_trade_costs_in_pat_sect.tex',
#                   caption=caption,
#                   **save_to_tex_options
#                   )

# df.to_csv(doubled_trade_costs_path+'dyn_Coop_negishi_weights_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

# write_calibration_results(doubled_trade_costs_path+'dyn_Coop_negishi_weights_with_doubled_trade_costs_in_pat_sect',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')

#%% Coop and Nash equilibria with no trade costs or tariffs (not simulated)

#%% Nash table with transitional dynamics with no trade costs or tariffs

all_nashes = pd.read_csv('nash_eq_recaps/dyn_deltas.csv')
all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) 
                         & (all_nashes.variation.astype(str) == variation_with_zero_trade_costs)]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_zero_trade_costs}/')
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

m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_zero_trade_costs}/')

p_nash = p_pre.copy()
p_nash.delta[:,1] = run_nash[p_baseline.countries].values.squeeze()

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

m_nash = m_pre.copy()
m_nash.compute_moments(dyn_sol_nash.sol_fin,p_nash)
m_nash.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'
                                     ],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_nash.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Nash equilibrium with doubled trade costs in the patenting sector'

df.style.format(precision=5).to_latex(no_trade_costs_path+'dyn_Nash_table_with_no_trade_costs.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(no_trade_costs_path+'dyn_Nash_table_with_no_trade_costs.csv',float_format='%.5f')

write_calibration_results(no_trade_costs_path+'dyn_Nash_with_no_trade_costs',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')


#%% Coop equal weights table with transitional dynamics with no trade costs or tariffs

all_coop_equales = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
                                     & (all_coop_equales.variation == variation_with_zero_trade_costs)
                                     & (all_coop_equales.aggregation_method == 'pop_weighted')]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_zero_trade_costs}/')
# p_pre.tau = p_baseline.tau.copy()
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

m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_zero_trade_costs}/')

p_coop_equal = p_pre.copy()
p_coop_equal.delta[:,1] = run_coop_equal[p_baseline.countries].values.squeeze()

sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_coop_equal, sol_init=sol_pre,Nt=25,
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
dyn_sol_coop_equal.compute_non_solver_quantities(p_coop_equal)
dyn_sol_coop_equal.sol_fin.compute_consumption_equivalent_welfare(p_coop_equal,sol_pre)
dyn_sol_coop_equal.sol_fin.compute_world_welfare_changes(p_coop_equal,sol_pre)

m_coop_equal = m_pre.copy()
m_coop_equal.compute_moments(dyn_sol_coop_equal.sol_fin,p_coop_equal)
m_coop_equal.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_equal.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with population weights'

df.style.format(precision=5).to_latex(no_trade_costs_path+'dyn_Coop_population_weights_table_with_no_trade_costs.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(no_trade_costs_path+'dyn_Coop_population_weights_table_with_no_trade_costs.csv',float_format='%.5f')

write_calibration_results(no_trade_costs_path+'dyn_Coop_population_weights_with_no_trade_costs',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

#%% Coop negishi weights table with transitional dynamics with no trade costs or tariffs

all_coop_negishies = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
                                     & (all_coop_negishies.variation == variation)
                                     & (all_coop_negishies.aggregation_method == 'negishi')]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_zero_trade_costs}/')
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

p_coop_negishi = p_pre.copy()
p_coop_negishi.delta[:,1] = run_coop_negishi[p_baseline.countries].values.squeeze()

sol, dyn_sol_coop_negishi = dyn_fixed_point_solver(p_coop_negishi, sol_init=sol_pre,Nt=25,
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
dyn_sol_coop_negishi.compute_non_solver_quantities(p_coop_negishi)
dyn_sol_coop_negishi.sol_fin.compute_consumption_equivalent_welfare(p_coop_negishi,sol_pre)
dyn_sol_coop_negishi.sol_fin.compute_world_welfare_changes(p_coop_negishi,sol_pre)

m_coop_negishi = m_pre.copy()
m_coop_negishi.compute_moments(dyn_sol_coop_negishi.sol_fin,p_coop_negishi)
m_coop_negishi.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_negishi.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with Negishi weights'

df.style.format(precision=5).to_latex(no_trade_costs_path+'dyn_Coop_negishi_weights_table_with_no_trade_costs.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(no_trade_costs_path+'dyn_Coop_negishi_weights_table_with_no_trade_costs.csv',float_format='%.5f')

write_calibration_results(no_trade_costs_path+'dyn_Coop_negishi_weights_with_no_trade_costs',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')


#%% Elasticities of patented innovations with respect to trade costs (to compare with Coelli)

df = pd.DataFrame()

for i,country in enumerate(p_baseline.countries):
# for i,country in enumerate(['USA']):
    # print(country)
    p = p_baseline.copy()
    # if (sol_baseline.psi_m_star[:,i,1]==np.min(sol_baseline.psi_m_star[:,i,1])).sum() == 1 and (
    #         sol_baseline.psi_m_star[:,i,1] == np.min(sol_baseline.psi_m_star[:,i,1]))[i]:
    #     mask = np.ones(p_baseline.N, dtype=bool)
    #     mask[i] = False
    #     x = (sol_baseline.psi_m_star[:,i,1][mask].min()/sol_baseline.psi_m_star[:,i,1].min())**p.k
        
    # else:
    #     x = 1
    
    mask = np.ones(p_baseline.N, dtype=bool)
    mask[i] = False
    x = sol_baseline.pflow[:,i].sum()/sol_baseline.pflow[:,i][mask].sum()
    # print(x)
    # p.tariff[:,i,1] = (1+p_baseline.tariff[:,i,1])*(1-0.01*x/1.032)-1
    p.tariff[:,i,1] = p_baseline.tariff[:,i,1]-0.01*x
    p.tariff[i,i,1] = 0

    try:
        print('for '+country+' following cf exports tariffs are negative :'
              +str([p_baseline.countries[index[0]]+str(round(p.tariff[index[0],i,1]*100,1)) for index in np.argwhere(p.tariff[:,i,1]<0)]))
    except:
        pass
    #%%

    sol, dyn_sol = dyn_fixed_point_solver(p, sol_baseline, Nt=25,
                                          t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=True,
                            damping = 50,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    dyn_sol.compute_non_solver_quantities(p)
    df.loc[country,'baseline number of patented innovations'] = sol_baseline.psi_o_star[i,1]**-p.k * sol_baseline.l_R[i,1]**(1-p.kappa)
    df.loc[country,'change in number of patented innovations'] = (dyn_sol.psi_o_star[i,1,-3]**-p.k * dyn_sol.l_R[i,1,-3]**(1-p.kappa)
                                                                  / (sol_baseline.psi_o_star[i,1]**-p.k * sol_baseline.l_R[i,1]**(1-p.kappa))
                                                                  )*100-100
    df.loc[country,'change in tariff in percentage'] = x
    print(df)
    
df.loc['average change', 'change in number of patented innovations'] = df.loc[p_baseline.countries,'change in number of patented innovations'].mean()
df.loc['weighted average change', 'change in number of patented innovations'
        ] = (df.loc[p_baseline.countries,'change in number of patented innovations']*df.loc[p_baseline.countries,'baseline number of patented innovations']
            ).sum()/df.loc[p_baseline.countries,'baseline number of patented innovations'].sum()

df.to_csv(calibration_path+'patented_innovations_elast_tariffs.csv',float_format='%.5f')

#%% Semi-elasticities to compare with Bertolotti

def compute_ge_semielasticity(p,sol_baseline,dynamics=False):
    p_cf = p.copy()
    p_cf.delta[0,1] = 1/(1/p_cf.delta[0,1]+1/12)
    
    if not dynamics:
        _, sol_cf = fixed_point_solver(p_cf,context = 'counterfactual',x0=p_cf.guess,
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
        sol_cf.scale_P(p_cf)
        sol_cf.compute_non_solver_quantities(p_cf)
        
        return sol_cf.pflow[0,0]/sol_baseline.pflow[0,0]-1
    
    if dynamics:
        _, dyn_sol_cf = dyn_fixed_point_solver(p_cf, sol_baseline, Nt=21,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=True,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=True,
                                damping = 50,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_cf.compute_non_solver_quantities(p_cf)
        
        return dyn_sol_cf.pflow[0,0,-1]/sol_baseline.pflow[0,0]-1
    

df = pd.DataFrame(columns = ['Semi-elasticity partial','GE effect, steady state','GE effect, with dynamics'], 
                  index = pd.Index([],name='Calibration'))

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
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

p_pre_cf = p_baseline.copy()
p_pre_cf.delta[...,1] = p_pre.delta[...,1]

_, sol_pre_cf = fixed_point_solver(p_pre_cf,context = 'counterfactual',x0=p_pre_cf.guess,
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
sol_pre_cf.scale_P(p_pre_cf)
sol_pre_cf.compute_non_solver_quantities(p_pre_cf)

'Semi-elasticity partial','GE effect, steady state','GE effect, with dynamics'

df.loc['2015 baseline calibration','Semi-elasticity partial'] = sol_baseline.semi_elast_patenting_delta[0,1]

df.loc['2015 baseline calibration','GE effect, steady state'
       ] = compute_ge_semielasticity(p_baseline,sol_baseline,dynamics=False)
df.loc['2015 baseline calibration','GE effect, with dynamics'
       ] = compute_ge_semielasticity(p_baseline,sol_baseline,dynamics=True)

df.loc['1992 partial calibration','Semi-elasticity partial'] = sol_pre.semi_elast_patenting_delta[0,1]

df.loc['1992 partial calibration','GE effect, steady state'
        ] = compute_ge_semielasticity(p_pre,sol_pre,dynamics=False)
df.loc['1992 partial calibration','GE effect, with dynamics'
        ] = compute_ge_semielasticity(p_pre,sol_pre,dynamics=True)

df.loc['2015 with 1992 partial calibration deltas','Semi-elasticity partial'] = sol_pre_cf.semi_elast_patenting_delta[0,1]

df.loc['2015 with 1992 partial calibration deltas','GE effect, steady state'
        ] = compute_ge_semielasticity(p_pre_cf,sol_pre_cf,dynamics=False)
df.loc['2015 with 1992 partial calibration deltas','GE effect, with dynamics'
        ] = compute_ge_semielasticity(p_pre_cf,sol_pre_cf,dynamics=True)

df['Semi-elasticity partial divided by 12'] = df['Semi-elasticity partial']/12

caption = 'Effect of US patent protection on US-US patenting flows'

df.style.format(precision=5).to_latex(calibration_path+'patenting_semi_elast.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(calibration_path+'patenting_semi_elast.csv',float_format='%.5f')

#%% A look at dynamics
from matplotlib.ticker import FormatStrFormatter

save_dynamics = True

p = p_baseline.copy()
p.delta[0,1] = 0.05

sol, dyn_sol = dyn_fixed_point_solver(p, sol_baseline, Nt=21,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 50,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
dyn_sol.compute_non_solver_quantities(p)


time = np.linspace(0,dyn_sol.t_inf,10001)
time_truncated = time[:1201]

def fit_and_eval(vec,dyn_sol,time,time_truncated,
                 normalization_start,normalization_end,
                 normalize_start=False,normalize_end=False):
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                vec,
                dyn_sol.Nt),time)
    res = fit
    if normalize_start:
        res = fit/normalization_start
    if normalize_start and normalize_end:
        # res = (fit-normalization_start)/np.abs(normalization_end-normalization_start)
        res = np.sign(normalization_end-normalization_start)*(fit-normalization_start)/np.abs(normalization_end-normalization_start)
    return res[:time_truncated.shape[0]]

def add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,normalize_start,
               normalize_end,label=None,color=sns.color_palette()[0],
               return_data = False):
    ax.plot(time_truncated,fit_and_eval(qty,
                                        dyn_sol,
                                        time,time_truncated,
                      normalization_start = norm_start,
                      normalization_end = norm_end,
                      normalize_start=normalize_start,
                      normalize_end=normalize_end)
            ,label=label,
            color=color)
    if not normalize_start and not normalize_end:
        ax.scatter(x=[0,60],
                    y=[norm_start,norm_end],
                    color=color)
    if normalize_start and not normalize_end:
        ax.scatter(x=[0,60],
                    # y=[1,norm_end/np.abs(norm_start)],
                    y=[1,np.abs(norm_end)/np.abs(norm_start)],
                    color=color)
    if normalize_start and normalize_end:
        ax.scatter(x=[0,60],
                    # y=[0,np.sign(norm_end-norm_start)*norm_end/np.abs(norm_end)],
                    y=[0,1],
                    color='k')
        
    ax.set_xlabel('Time (years)')
    
    if return_data:
        return time_truncated,fit_and_eval(qty,
                                            dyn_sol,
                                            time,time_truncated,
                          normalization_start = norm_start,
                          normalization_end = norm_end,
                          normalize_start=normalize_start,
                          normalize_end=normalize_end)

# Growth rate
fig,ax = plt.subplots()
qty = dyn_sol.g
norm_start = dyn_sol.sol_init.g
norm_end = dyn_sol.sol_fin.g
name = 'growth_rate'
add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,
               normalize_start=True,
               normalize_end=False,
               label='Growth rate',
               color=Category18[0])
# ax.set_ylabel('Growth rate')
# plt.legend()
if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'growth_rate.'+save_format,format=save_format)
plt.show()

# Real final consumption
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty = dyn_sol.nominal_final_consumption[i,:]/dyn_sol.price_indices[i,:]
    norm_start = dyn_sol.sol_init.nominal_final_consumption[i]/dyn_sol.sol_init.price_indices[i]
    norm_end = dyn_sol.sol_fin.nominal_final_consumption[i]/dyn_sol.sol_fin.price_indices[i]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=False,
                   label=country,
                   color=Category18[i])
# ax.set_ylabel('Real final consumption')
# plt.legend(loc=[1.02,0.02])
plt.legend(fontsize=4.8)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'real_final_consumption.'+save_format,format=save_format)
plt.show()

# Real profit
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty = (dyn_sol.profit[:,i,1,:]/dyn_sol.price_indices[i,:]).sum(axis=0)
    norm_start = (dyn_sol.sol_init.profit[:,i,1]*dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
                  ).sum()
    norm_end = (dyn_sol.sol_fin.profit[:,i,1]*dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
                ).sum()
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=False,
                   label=country,
                   color=Category18[i])
# ax.set_ylabel('Real profit')
# plt.legend(loc=[1.02,0.02])
plt.legend(fontsize=4.8)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'real_profit.'+save_format,format=save_format)
plt.show()

# Research Labor
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty =dyn_sol.l_R[i,1,:]
    norm_start = dyn_sol.sol_init.l_R[i,1]
    norm_end = dyn_sol.sol_fin.l_R[i,1]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,                   
                   normalize_start=True,
                   normalize_end=False,
                   label=country,
                   color=Category18[i])
# ax.set_ylabel('Labor allocated to research')
# plt.legend(loc=[1.02,0.02])
plt.legend(fontsize=4.7,ncol=4)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'research_labor.'+save_format,format=save_format)
plt.show()

# Real wage
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty = dyn_sol.w[i,:]/dyn_sol.price_indices[i,:]
    norm_start = dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
    norm_end = dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=False,
                   label=country,
                   color=Category18[i])
# ax.set_ylabel('Real wage')
plt.legend(fontsize=4.8)
if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'real_wage.'+save_format,format=save_format)
plt.show()

# PSI CD
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty = dyn_sol.PSI_CD[i,1,:]+dyn_sol.PSI_CD_0[i,1,None]
    norm_start = dyn_sol.sol_init.PSI_CD[i,1]
    norm_end = dyn_sol.sol_fin.PSI_CD[i,1]
    add_graph(dyn_sol,qty,norm_start,norm_end,
                   ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=False,
                   label=country,
                   color=Category18[i])
# ax.set_ylabel(r'$\Psi^{CD}_n$')
# plt.legend(loc=[1.02,0.02])
plt.legend(fontsize=4.8)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'psi_cd.'+save_format,format=save_format)
plt.show()

# Interest rate
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):
    qty = dyn_sol.r[i,:]
    norm_start = dyn_sol.sol_init.r
    norm_end = dyn_sol.sol_fin.r
    add_graph(dyn_sol,qty,norm_start,norm_end,
                    ax,time,time_truncated,
                    normalize_start=True,
                    normalize_end=False,
                    label=country,
                    color=Category18[i])
# ax.set_ylabel('Interest rate')
# plt.legend(loc=[1.02,0.02])
plt.legend(fontsize=4.8)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'interest_rate.'+save_format,format=save_format)
plt.show()

# Different US quantities
fig,ax = plt.subplots()
i = 0
qty = dyn_sol.g
norm_start = dyn_sol.sol_init.g
norm_end = dyn_sol.sol_fin.g
time, growth = add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=True,
               label='Growth',
               color=Category18[0],
               return_data=True)

qty = dyn_sol.nominal_final_consumption[i,:]/dyn_sol.price_indices[i,:]
norm_start = dyn_sol.sol_init.nominal_final_consumption[i]/dyn_sol.sol_init.price_indices[i]
norm_end = dyn_sol.sol_fin.nominal_final_consumption[i]/dyn_sol.sol_fin.price_indices[i]
time, us_cons = add_graph(dyn_sol,qty,norm_start,norm_end,
               ax,time,time_truncated,
                   normalize_start=True,
                   normalize_end=True,
               label='Normalized US consumption',
               color=Category18[1],
               return_data=True)

# qty = (dyn_sol.profit[:,i,1,:]/dyn_sol.price_indices[i,:]).sum(axis=0)
# norm_start = (dyn_sol.sol_init.profit[:,i,1]*dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
#               ).sum()
# norm_end = (dyn_sol.sol_fin.profit[:,i,1]*dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
#             ).sum()
# add_graph(dyn_sol,qty,norm_start,norm_end,
#                 ax,time,time_truncated,
#                     normalize_start=True,
#                     normalize_end=True,
#                 label='Real profit',
#                 color=Category18[2])

# qty =dyn_sol.l_R[i,1,:]
# norm_start = dyn_sol.sol_init.l_R[i,1]
# norm_end = dyn_sol.sol_fin.l_R[i,1]
# add_graph(dyn_sol,qty,norm_start,norm_end,
#                 ax,time,time_truncated,
#                     normalize_start=True,
#                     normalize_end=True,
#                 label='Research labor',
#                 color=Category18[3])

# qty = dyn_sol.w[i,:]/dyn_sol.price_indices[i,:]
# norm_start = dyn_sol.sol_init.w[i]/dyn_sol.sol_init.price_indices[i]
# norm_end = dyn_sol.sol_fin.w[i]/dyn_sol.sol_fin.price_indices[i]
# add_graph(dyn_sol,qty,norm_start,norm_end,
#                 ax,time,time_truncated,
#                     normalize_start=True,
#                     normalize_end=True,
#                 label='Real wage',
#                 color=Category18[4])

# qty = dyn_sol.PSI_CD[i,1,:]+dyn_sol.PSI_CD_0[i,1,None]
# norm_start = dyn_sol.sol_init.PSI_CD[i,1]
# norm_end = dyn_sol.sol_fin.PSI_CD[i,1]
# add_graph(dyn_sol,qty,norm_start,norm_end,
#                 ax,time,time_truncated,
#                     normalize_start=True,
#                     normalize_end=True,
#                 label=r'$\Psi^{CD}_n$',
#                 color=Category18[5])

# qty = dyn_sol.r[i,:]
# norm_start = dyn_sol.sol_init.r
# norm_end = dyn_sol.sol_fin.r
# add_graph(dyn_sol,qty,norm_start,norm_end,
#                 ax,time,time_truncated,
#                     normalize_start=True,
#                     normalize_end=True,
#                 label='Interest rate',
#                 color=Category18[6])

# ax.set_ylabel('Time evolution of normalized US quantities')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.legend()
if save_dynamics:
    for save_format in save_formats:
        plt.savefig(dyn_save_path+'normalized_us_quantities.'+save_format,format=save_format)
plt.show()



df = pd.DataFrame(index = time.round(5)
                  )
df['Growth'] = growth
df['Normalized US consumption'] = us_cons

caption = 'Normalized growth rate and US consumption time evolution'

if save_dynamics:
    df.style.format(precision=5).to_latex(dyn_save_path+'normalized_us_quantities.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    
    df.to_csv(dyn_save_path+'normalized_us_quantities.csv',float_format='%.5f')


# welfares changes table
dyn_sol.sol_fin.compute_consumption_equivalent_welfare(p,sol_baseline)
dyn_sol.sol_fin.compute_world_welfare_changes(p,sol_baseline)

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = ['Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol.sol_fin.cons_eq_pop_average_welfare_change

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Welfare changes illustrating transitional dynamics effects'

if save_dynamics:
    df.style.format(precision=5).to_latex(dyn_save_path+'welfare_table.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    
    df.to_csv(dyn_save_path+'welfare_table.csv',float_format='%.5f')

#%% Solving for where countries will join the patenting club

qties_dic = {
    'eta':{
        'column_name':'eta',
        'title':r'$\eta$'
        },
    'T_pat':{
        'column_name':'T_pat',
        'title':r'$T$'
        },
    'labor':{
        'column_name':'labor',
        'title':'Labor'
        },
    # 'iceberg_trade_cost_in':{
    #     'column_name':'tau_in_factor',
    #     },
    # 'iceberg_trade_cost_out':{
    #     'column_name':'tau_out_factor',
    #     },
    }

df = pd.DataFrame()

# for qty in ['eta','T_pat','labor','iceberg_trade_cost_in','iceberg_trade_cost_out']:
for qty in ['eta','T_pat','labor']:
    for c,country in enumerate(p_baseline.countries):
        if country in ['CHN','IND','RUS']:

            df_c = pd.read_csv(f'solve_to_join_pat_club/{qty}/baseline_{baseline}/pop_weighted_{country}.csv')
            df.loc[country,qty] = df_c[f'{qties_dic[qty]["column_name"]}_{country}'].iloc[-1]
            
            if np.allclose(df_c[f'{qties_dic[qty]["column_name"]}_{country}'],
                           df_c[f'{qties_dic[qty]["column_name"]}_{country}'].sort_values()):
                df.loc[country,qty] = np.nan
                
            if qty == 'eta':
                df.loc[country,qty+'_baseline'] = p_baseline.eta[c,1]
                df.loc[country,'eta_US_baseline'] = p_baseline.eta[0,1]
                df[qty+'_as_ratio_to_baseline'] = df[qty] / df[qty+'_baseline']
                df[qty+'_as_ratio_to_US_baseline'] = df[qty] / df[qty+'_US_baseline']
            if qty == 'T_pat':
                df.loc[country,qty+'_baseline'] = p_baseline.T[c,1]
                df.loc[country,'T_pat_US_baseline'] = p_baseline.T[0,1]
                df[qty+'_as_ratio_to_baseline'] = df[qty] / df[qty+'_baseline']
                df[qty+'_as_ratio_to_US_baseline'] = df[qty] / df[qty+'_US_baseline']
            if qty == 'labor':
                df.loc[country,qty+'_baseline'] = p_baseline.labor[c]
                df.loc[country,'labor_world_baseline'] = p_baseline.labor.sum()
                df[qty+'_as_ratio_to_baseline'] = df[qty] / df[qty+'_baseline']
                df[qty+'_as_ratio_to_world_baseline'] = df[qty] / df[qty+'_world_baseline']
                
df = df.T

for qty in ['eta','T_pat','labor']:
    if qty in ['eta','T_pat']:
        fig,ax = plt.subplots()
        
        plt.axhline(y=df.loc[qty+'_US_baseline'].iloc[0],
                    color='grey',
                    label='Baseline USA')
        
        ax.scatter(df.columns,df.loc[qty+'_baseline'],
                                    label = 'Baseline',
                                    marker = 'o',
                                    )
        
        ax.scatter(df.columns,df.loc[qty],
                                    label = 'Threshold to join patenting club',
                                    marker = 'x',
                                    )
        
    if qty in ['labor']:
        fig,ax = plt.subplots()
        
        plt.axhline(y=df.loc[qty+'_world_baseline'].iloc[0],
                    color='grey',
                    label='Baseline World')
        
        ax.scatter(df.columns,df.loc[qty+'_baseline'],
                                    label = 'Baseline',
                                    marker = 'o',
                                    )
        
        if not df.loc[qty].isna().all():
            ax.scatter(df.columns,df.loc[qty],
                                        label = 'Threshold to join patenting club',
                                        marker = 'x',
                                        )


    if qty == 'T_pat':
        ax.set_yscale('log')
    
    plt.title(qties_dic[qty]["title"])
    plt.legend()
    
    for save_format in save_formats:
        plt.savefig(solve_to_join_pat_club_save_path+qty+'.'+save_format,format=save_format)
    plt.show()

df.to_csv(solve_to_join_pat_club_save_path+'summary.csv')

#%% Sensitivity graphs of the calibration

moments_to_change = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRGDP', 'SINNOVPATUS',
  'TO', 'SPFLOW', 'UUPCOST', 'DOMPATINUS', 'TE']
parameters_to_change = ['rho','kappa','sigma','theta','gamma']

df = pd.read_csv(table_path+f'baseline_{baseline}_sensitivity_tables/all_sensitivity_table_20.csv',index_col=0)

df = df.drop(['RP_USA','RD_CHN','RD_BRA','RD_IND','RD_ROW','RD_MEX','RD_RUS','RD_ZAF'])

names = {
    'GPDIFF':'Price growth diff.',
    'GROWTH':'Agg. growth rate',
    'KM':'Pat. value rel. R&D in US',
    'OUT':'World output',
    'SINNOVPATUS':'Share innov. pat. US',
    'DOMPATINUS':'Share dom. pat. US',
    'UUPCOST':'Domestic pat. expenditure US',
    'TO':'Turnover US imports',
    'TE':'Trade elasticity patenting',
    'rho':r'$\rho$',
    'kappa':r'$\kappa$',
    'sigma':r'$\sigma^1$',
    'theta':r'$\theta^0$',
    'gamma':r'$\gamma$',
    'g_0':r'$g^0$',
    'fo':r'$f^o$',
    'fe':r'$f^e$',
    'k':r'$k$',
    'nu':r'$\nu$',
    'zeta':r'$\zeta$',
    }



for c in p_baseline.countries:
    names['RD_'+c] = 'R&D expend. rel. GDP '+c
    names['RP_'+c] = 'Relative price '+c
    names['SRGDP_'+c] = 'Share real GDP '+c
    names['SPFLOW_destination_'+c] = 'Patents in '+c
    names['SPFLOW_origin_'+c] = 'Patents from '+c
    names['eta '+c] = r'$\eta$ '+c
    names['delta '+c] = r'$\delta$ '+c
    names['T Patent '+c] = 'No pat. sector productivity '+c
    names['T Non patent '+c] = 'Pat. sector productivity '+c
    # names['T Patent '+c] = r'${(T^{1/\theta})}^1$ '+c
    # names['T Non patent '+c] = r'${(T^{1/\theta})}^0$ '+c

df.index = [names[i] for i in df.index]

names['theta'] = r'$\theta^1$'

df.columns = [names[i] for i in df.columns]

fig,ax = plt.subplots(figsize=(14,9),dpi=144)

sns.heatmap(df,ax=ax,
            cmap="vlag",
            center=0,
            robust = True,
            linewidths=0.01,
            linecolor='grey',
            cbar_kws={'label':'Elasticity',
                      'spacing':'proportional'
                      }
            )

ax.tick_params('x', top=True, labeltop=True)
ax.tick_params(axis=u'both', which=u'both',length=0)

# Rotate and align bottom ticklabels
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="top", rotation_mode="anchor")
# Rotate and align top ticklabels
plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="bottom",rotation_mode="anchor")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
ax.figure.axes[-1].yaxis.label.set_size(11)
for save_format in save_formats:
    plt.savefig(sensitivity_path+'full_sensitivities_saturated.'+save_format,format=save_format)
    # plt.savefig(sensitivity_path+'full_sensitivities_saturated_tex.'+save_format,format=save_format)

plt.show()
    
#%% Sensitivity graphs of the calibration for patenting moments

moments_to_change = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRGDP', 'SINNOVPATUS',
  'TO', 'SPFLOW', 'UUPCOST', 'DOMPATINUS', 'TE']
parameters_to_change = ['rho','kappa','sigma','theta','gamma']

df = pd.read_csv(table_path+f'baseline_{baseline}_sensitivity_tables/all_sensitivity_table_20.csv',index_col=0)

df = df.drop(['RP_USA','RD_CHN','RD_BRA','RD_IND','RD_ROW','RD_MEX','RD_RUS','RD_ZAF'])

df = df.loc[[x for x in df.index if x.startswith('RD')]
            +[x for x in df.index if x.startswith('SPFLOW')]
            ]
df = df[[x for x in df.columns if x.startswith('delta')]
        +[x for x in df.columns if x.startswith('eta')]
        ]

names = {
    'GPDIFF':'Price growth diff.',
    'GROWTH':'Agg. growth rate',
    'KM':'Pat. value rel. R&D in US',
    'OUT':'World output',
    'SINNOVPATUS':'Share innov. pat. US',
    'DOMPATINUS':'Share dom. pat. US',
    'UUPCOST':'Domestic pat. expenditure US',
    'TO':'Turnover US imports',
    'TE':'Trade elasticity patenting',
    'rho':r'$\rho$',
    'kappa':r'$\kappa$',
    'sigma':r'$\sigma^1$',
    'theta':r'$\theta^0$',
    'gamma':r'$\gamma$',
    'g_0':r'$g^0$',
    'fo':r'$f^o$',
    'fe':r'$f^e$',
    'k':r'$k$',
    'nu':r'$\nu$',
    'zeta':r'$\zeta$',
    }



for c in p_baseline.countries:
    names['RD_'+c] = 'R&D expend. rel. GDP '+c
    names['RP_'+c] = 'Relative price '+c
    names['SRGDP_'+c] = 'Share real GDP '+c
    names['SPFLOW_destination_'+c] = 'Patents in '+c
    names['SPFLOW_origin_'+c] = 'Patents from '+c
    names['eta '+c] = r'$\eta$ '+c
    names['delta '+c] = r'$\delta$ '+c
    # names['T Patent '+c] = 'No pat. sector productivity '+c
    # names['T Non patent '+c] = 'Pat. sector productivity '+c
    names['T Patent '+c] = r'${(T^{1/\theta})}^1$ '+c
    names['T Non patent '+c] = r'${(T^{1/\theta})}^0$ '+c

df.index = [names[i] for i in df.index]

names['theta'] = r'$\theta^1$'

df.columns = [names[i] for i in df.columns]


fig,ax = plt.subplots(figsize=(14,8),dpi=144)

sns.heatmap(df,ax=ax,
            cmap="vlag",
            center=0,
            robust = True,
            linewidths=0.01,
            linecolor='grey',
            # cbar_kws={'label':'Elasticity (Colormap range is computed with robust quantiles instead of extreme values)',
            cbar_kws={'label':'Elasticity',
                      'spacing':'proportional'
                      }
            )

ax.tick_params('x', top=True, labeltop=True,labelrotation=90)
ax.tick_params(axis=u'both', which=u'both',length=0)
# Rotate and align bottom ticklabels
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
# Rotate and align top ticklabels
plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="center",rotation_mode="anchor")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
ax.figure.axes[-1].yaxis.label.set_size(11)

for save_format in save_formats:
    plt.savefig(sensitivity_path+'reduced_sensitivities_saturated.'+save_format,format=save_format)

plt.show()

#%% Results with doubled diffusion speed

## %% Unilateral patent protections counterfactuals with dynamics for doubled nu

# variation = variation_with_doubled_nu
# run_path_with_doubled_nu = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation_with_doubled_nu}/'

# p_double_nu = parameters()
# p_double_nu.load_run(run_path_with_doubled_nu)

# _, sol_double_nu = fixed_point_solver(p_double_nu,context = 'counterfactual',x0=p_double_nu.guess,
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='phi',
#                         plot_convergence=False,
#                         plot_cobweb=False,
#                         safe_convergence=0.001,
#                         disp_summary=False,
#                         damping = 10,
#                         max_count = 3e3,
#                         accel_memory = 50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=5
#                         )
# sol_double_nu.scale_P(p_double_nu)
# sol_double_nu.compute_non_solver_quantities(p_double_nu)


# # for c in ['USA']:
# for c in p_double_nu.countries:
#     recap = pd.DataFrame(columns = ['delta_change','world_negishi','world_equal']+p_double_nu.countries)
#     if variation == 'baseline':
#         local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline+'/'
#     else:
#         local_path = \
#             f'counterfactual_results/unilateral_patent_protection/baseline_{baseline}_{variation}/'
#     print(c)
#     if c in p_double_nu.countries:
#         idx_country = p_double_nu.countries.index(c)
#     country_path = local_path+c+'/'
#     files_in_dir = next(os.walk(country_path))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list.sort(key=float)
#     for i,run in enumerate(run_list):
#         p = parameters()
#         p.load_run(country_path+run+'/')
#         if p.guess is not None:
#             sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
#             sol_c.scale_P(p)
#             sol_c.compute_non_solver_quantities(p)
#             sol_c.compute_consumption_equivalent_welfare(p,sol_double_nu)
#         if p.dyn_guess is not None:
#             dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
#                                                     Nt=25,t_inf=500,
#                                                     sol_init = sol_double_nu,
#                                                     sol_fin = sol_c)
#             dyn_sol_c.compute_non_solver_quantities(p)
#         if c in p_double_nu.countries:
#             recap.loc[run, 'delta_change'] = p.delta[idx_country,1]/p_double_nu.delta[idx_country,1]
#         if c == 'World':
#             recap.loc[run, 'delta_change'] = p.delta[0,1]/p_double_nu.delta[0,1]
#         if c == 'Uniform_delta':
#             recap.loc[run, 'delta_change'] = p.delta[0,1]
#         if c == 'Upper_uniform_delta':
#             recap.loc[run,'delta_change'] = np.logspace(-2,0,len(run_list))[i]
#         if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
#             recap.loc[run, 'delta_change'] = p.tau[0,1,1]/p_double_nu.tau[0,1,1]
#         recap.loc[run, 'world_negishi'] = dyn_sol_c.cons_eq_negishi_welfare_change
#         recap.loc[run, 'world_equal'] = dyn_sol_c.cons_eq_pop_average_welfare_change
#         recap.loc[run,p_double_nu.countries] = dyn_sol_c.cons_eq_welfare

#     fig,ax = plt.subplots()
    
#     plt.xscale('log')
    
#     ax.set_ylabel('Welfare change (%)')
#     if c in p_double_nu.countries:
#         ax.set_xlabel(r'Proportional change of $\delta$')
#     if c == 'World':
#         ax.set_xlabel(r'Proportional change of $\delta$ of all countries')
#     if c == 'Uniform_delta' or c == 'Upper_uniform_delta':
#         ax.set_xlabel(r'Harmonized $\delta$')
#         plt.axvline(x=p_double_nu.delta[0,1], lw = 1, color = 'k')
#         xt = ax.get_xticks() 
#         xt=np.append(xt,p_double_nu.delta[0,1])
#         xtl=xt.tolist()
#         xtl[-1]=r'$\delta_{US}$'
#         ax.set_xticks(xt)
#         ax.set_xticklabels(xtl)
#     if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
#         ax.set_xlabel(r'Proportional change of $\tau$ of all countries in the patenting sector')
#         ax.set_xlim(0.98,1.02)
#         ax.set_ylim(-2,2)
#         plt.xscale('linear')

#     for i,country in enumerate(p_double_nu.countries):
#         ax.plot(recap.delta_change,recap[country]*100-100,color=Category18[i],label=countries_names[country])
    
#     ax.plot(recap.delta_change,recap['world_negishi']*100-100,color='k',ls='--',label='World Negishi')
#     ax.plot(recap.delta_change,recap['world_equal']*100-100,color='k',ls=':',label='World Equal')

#     # ax.legend(loc=[1.02,0.02])
#     ax.legend(fontsize=4,ncol=2)

#     for save_format in save_formats:
#         plt.savefig(doubled_nu_path+c+'_dyn_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
#     plt.show()
    
#     if c in p_double_nu.countries:
#         caption = 'Consumption equivalent welfares in the unilateral patent protection counterfactual of '+countries_names[c]
#     if c == 'World':
#         caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
#     if c == 'Uniform_delta':
#         caption = 'Consumption equivalent welfares in the harmonized delta counterfactual change of all countries'
#         recap = recap.rename(columns = {'delta_change':'delta'})
#     if c == 'Upper_uniform_delta':
#         caption = 'Consumption equivalent welfares in the partially harmonized delta counterfactual change of all countries'
#         recap = recap.rename(columns = {'delta_change':'delta'})
#     if c == 'trade_cost_eq_trips_all_countries_pat_sectors':
#         caption = 'Consumption equivalent welfares in the counterfactual change of delta pre-TRIPS and trade costs of the patenting sectors'
#         recap = recap.rename(columns = {'delta_change':'tau_change'})
    
#     recap.style.to_latex(doubled_nu_path+c+'_dyn_unilateral_patent_protection_counterfactual.tex',
#                       caption=caption,
#                       **save_to_tex_options
#                       )
#     recap.to_csv(doubled_nu_path+c+'_dyn_unilateral_patent_protection_counterfactual.csv')
    
#     if c == 'Uniform_delta':
#         delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_double_nu.delta[0,1]))].to_frame()
#         delta_US_values.style.to_latex(doubled_nu_path+c+'_dyn_US_values.tex',
#                           caption=caption,
#                           **save_to_tex_options
#                           )
#         delta_US_values.to_csv(doubled_nu_path+c+'_dyn_US_values.csv')
        
#     if c == 'Upper_uniform_delta':
#         delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_double_nu.delta[0,1]))].to_frame()
#         delta_US_values.style.to_latex(doubled_nu_path+c+'_dyn_US_values.tex',
#                           caption=caption,
#                           **save_to_tex_options
#                           )
#         delta_US_values.to_csv(doubled_nu_path+c+'_dyn_US_values.csv')

#%% Nash table with transitional dynamics with doubled nu

variation = variation_with_doubled_nu
run_path_with_doubled_nu = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation_with_doubled_nu}/'

p_double_nu = parameters()
p_double_nu.load_run(run_path_with_doubled_nu)

all_nashes = pd.read_csv('nash_eq_recaps/dyn_deltas.csv')
all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) 
                         & (all_nashes.variation.astype(str) == variation_with_doubled_nu)]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_nu}/')
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

m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_nu}/')

p_nash = p_pre.copy()
p_nash.delta[:,1] = run_nash[p_double_nu.countries].values.squeeze()

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

m_nash = m_pre.copy()
m_nash.compute_moments(dyn_sol_nash.sol_fin,p_nash)
m_nash.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_double_nu.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'
                                     ],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_double_nu.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_nash.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_nash.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_nash.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Nash equilibrium with doubled nu'

df.style.format(precision=5).to_latex(doubled_nu_path+'dyn_Nash_table_with_doubled_nu.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_nu_path+'dyn_Nash_table_with_doubled_nu.csv',float_format='%.5f')

write_calibration_results(doubled_nu_path+'dyn_Nash_with_doubled_nu',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')


#%% Coop equal weights table with transitional dynamics with doubled nu

variation = variation_with_doubled_nu
run_path_with_doubled_nu = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation_with_doubled_nu}/'

p_double_nu = parameters()
p_double_nu.load_run(run_path_with_doubled_nu)

all_coop_equales = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
                                     & (all_coop_equales.variation == variation_with_doubled_nu)
                                     & (all_coop_equales.aggregation_method == 'pop_weighted')]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_nu}/')
# p_pre.tau = p_double_nu.tau.copy()
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

m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_nu}/')

p_coop_equal = p_pre.copy()
p_coop_equal.delta[:,1] = run_coop_equal[p_double_nu.countries].values.squeeze()

sol, dyn_sol_coop_equal = dyn_fixed_point_solver(p_coop_equal, sol_init=sol_pre,Nt=25,
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
dyn_sol_coop_equal.compute_non_solver_quantities(p_coop_equal)
dyn_sol_coop_equal.sol_fin.compute_consumption_equivalent_welfare(p_coop_equal,sol_pre)
dyn_sol_coop_equal.sol_fin.compute_world_welfare_changes(p_coop_equal,sol_pre)

m_coop_equal = m_pre.copy()
m_coop_equal.compute_moments(dyn_sol_coop_equal.sol_fin,p_coop_equal)
m_coop_equal.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_double_nu.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_double_nu.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_equal.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_equal.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_equal.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with population weights'

df.style.format(precision=5).to_latex(doubled_nu_path+'dyn_Coop_population_weights_table_with_doubled_nu.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_nu_path+'dyn_Coop_population_weights_table_with_doubled_nu.csv',float_format='%.5f')

write_calibration_results(doubled_nu_path+'dyn_Coop_population_weights_with_doubled_nu',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

#%% Coop negishi weights table with transitional dynamics with doubled nu

variation = variation_with_doubled_nu
run_path_with_doubled_nu = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation_with_doubled_nu}/'

p_double_nu = parameters()
p_double_nu.load_run(run_path_with_doubled_nu)

all_coop_negishies = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
                                     & (all_coop_negishies.variation == variation)
                                     & (all_coop_negishies.aggregation_method == 'negishi')]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_nu}/')
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

p_coop_negishi = p_pre.copy()
p_coop_negishi.delta[:,1] = run_coop_negishi[p_double_nu.countries].values.squeeze()

sol, dyn_sol_coop_negishi = dyn_fixed_point_solver(p_coop_negishi, sol_init=sol_pre,Nt=25,
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
dyn_sol_coop_negishi.compute_non_solver_quantities(p_coop_negishi)
dyn_sol_coop_negishi.sol_fin.compute_consumption_equivalent_welfare(p_coop_negishi,sol_pre)
dyn_sol_coop_negishi.sol_fin.compute_world_welfare_changes(p_coop_negishi,sol_pre)

m_coop_negishi = m_pre.copy()
m_coop_negishi.compute_moments(dyn_sol_coop_negishi.sol_fin,p_coop_negishi)
m_coop_negishi.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_double_nu.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights',
                                     'Growth rate (%)'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only']
                  )
    
for i,c in enumerate(p_double_nu.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_negishi.delta[i,1]
    df.loc[countries_names[c],'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_welfare[i]
    df.loc[countries_names[c],'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_negishi_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change with transition dynamics'] = dyn_sol_coop_negishi.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to population weights',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.cons_eq_pop_average_welfare_change

df.loc['Growth rate (%)',
       'Welfare change, steady state only'] = dyn_sol_coop_negishi.sol_fin.g*100

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with Negishi weights'

df.style.format(precision=5).to_latex(doubled_nu_path+'dyn_Coop_negishi_weights_table_with_doubled_nu.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_nu_path+'dyn_Coop_negishi_weights_table_with_doubled_nu.csv',float_format='%.5f')

write_calibration_results(doubled_nu_path+'dyn_Coop_negishi_weights_with_nu',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')

#%% Unilateral patent protection counterfactuals for doubled nu, doubled tau and no trade costs

variations_of_robust_checks = {
    variation_with_doubled_nu:r'Doubled $\nu$',
     # variation_with_doubled_tau_in_pat_sect:r'Doubled trade costs',
     variation_with_zero_trade_costs:r'Zero trade costs and tariffs',
     # variation_with_zero_tariffs:r'Zero tariffs',
     # variation_with_ten_times_tariffs:r'Ten times tariffs',
     'baseline':'Baseline',
    }

import math

for c,country in enumerate(p_baseline.countries):
# for c,country in enumerate(['USA','EUR','JAP','CHN','KOR']):
    fig,ax = plt.subplots()
    for i,rob_check in enumerate(variations_of_robust_checks):
        variation = rob_check
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
        df_welfare = pd.read_csv(local_path+'dyn_'+country+'.csv')
        if rob_check == 'baseline':
            ax.plot(df_welfare['delt'],df_welfare[country]*100-100,color='k',
                    label=variations_of_robust_checks[rob_check]
                    # ,lw=5
                    )
        else:
            ax.plot(df_welfare['delt'],df_welfare[country]*100-100,
                    label=variations_of_robust_checks[rob_check],
                    color = Category18[i],
                    )
    ax.set_ylabel('Welfare change (%)')
    name = countries_names[country]
    # ax.set_xlabel(fr'Proportional change of $\delta$ {name}')
    ax.set_xlabel(r'Proportional change of $\delta$')
    # if country == 'USA':
    #     ax.set_xlabel(r'Proportional change of $\delta_{US}$')
    # elif country == 'EUR':
    #     ax.set_xlabel(r'Proportional change of $\delta_{Europe}$')
    # else:
        # ax.set_xlabel(r'Proportional change of $\delta$')
    ax.set_xscale('log')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.set_title(country)

    legend = ax.legend()
    def export_legend(legend, filename="dbl_tau_nu_legend.pdf"):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend,filename = counterfactuals_doubled_nu_tau_path+'dbl_tau_nu_legend.pdf')
    # plt.legend(False)
    legend.remove()
    if country == 'USA':
        ax.legend()
    # plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
    for save_format in save_formats:
        plt.savefig(counterfactuals_doubled_nu_tau_path+'dbl_tau_nu_unilateral_patent_protection_'+country+'.'+save_format,format=save_format)
    
    plt.show()
    
# axes[2,1].legend(handles, labels, loc='center left',fontsize=12)
# axes[2,1].set_axis_off()

# plt.tight_layout()
# plt.show()


#%% Derivatives of welfares with respect to delta as a function of tau

qty = 'tau_pat_sector'

fig,ax = plt.subplots()

for j,country in enumerate(p_baseline.countries):
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_{baseline}/dyn_{country}.csv',
                     index_col=0)
    ax.plot(df[df.columns[0]],df[country],label=country,color=Category18[j])
    
# ax.set_xscale('log')
ax.set_xlabel('Proportional change in trade costs')
ax.set_ylabel('Derivative of welfare to own patent protection')
ax.legend(loc=[1.02,0.02])

plt.tight_layout()
for save_format in save_formats:
    plt.savefig(counterfactuals_doubled_nu_tau_path+'welfare_derivative_function_of_tau.'+save_format,format=save_format)
plt.show()

#%% Derivatives of welfares with respect to delta as a function of tariff

qty = 'tariff_pat_sector'

fig,ax = plt.subplots()

for j,country in enumerate(p_baseline.countries):
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_{baseline}/dyn_{country}.csv',
                     index_col=0)
    ax.plot(df[df.columns[0]],df[country],label=country,color=Category18[j])
    
# ax.set_xscale('log')
ax.set_xlabel('Proportional change in tariff')
ax.set_ylabel('Derivative of welfare to own patent protection')
ax.legend(loc=[1.02,0.02])

plt.tight_layout()
for save_format in save_formats:
    plt.savefig(counterfactuals_doubled_nu_tau_path+'welfare_derivative_function_of_tariff.'+save_format,format=save_format)
plt.show()

#%% Derivatives of welfares with respect to delta as a function of nu

qty = 'nu'

fig,ax = plt.subplots()

for j,country in enumerate(p_baseline.countries):
    df = pd.read_csv(f'deriv_welfare_to_patent_protec_cf/{qty}/baseline_{baseline}/dyn_{country}.csv',
                     index_col=0)
    ax.plot(df[df.columns[0]],df[country],label=country,color=Category18[j])
    
# ax.set_xscale('log')
ax.set_xlabel(r'Proportional change in $\nu$')
ax.set_ylabel('Derivative of welfare to own patent protection')
ax.legend(loc=[1.02,0.02])

plt.tight_layout()
for save_format in save_formats:
    plt.savefig(counterfactuals_doubled_nu_tau_path+'welfare_derivative_function_of_nu.'+save_format,format=save_format)
plt.show()

#%% Robustness checks

#%% Unilateral patent protection counterfactuals

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
names = {
    'GPDIFF':'Price growth diff.',
    'GROWTH':'Agg. growth rate',
    'KM':'Pat. value rel. RD in US',
    'OUT':'World output',
    'SINNOVPATUS':'Share innov. pat. US',
    'DOMPATINUS':'Share dom. pat. US',
    'UUPCOST':'Domestic pat. expenditure US',
    'TO':'Turnover US imports',
    'TE':'Trade elasticity patenting',
    'rho':r'$\rho$',
    'kappa':r'$\kappa$',
    'sigma':r'$\sigma^1$',
    'theta':r'$\theta^0$',
    'gamma':r'$\gamma$',
    'g_0':r'$g^0$',
    'fo':r'$f^o$',
    'fe':r'$f^o$',
    'k':r'$k$',
    'nu':r'$\nu$',
    'zeta':r'$\zeta$',
    }

variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low Turnover in US imports',
    '99.1':'High Turnover in US imports',
    '99.2':'Low Trade elasticity in patenting sector',
    '99.3':'High Trade elasticity in patenting sector',
    '99.4':'Low Value of patents relative to R&D expenditure in US',
    '99.5':'High Value of patents relative to R&D expenditure in US',
    '99.6':r'Low $\sigma^1$',
    '99.7':r'High $\sigma^1$',
    '99.8':r'Low $\kappa$',
    '99.9':r'High $\kappa$',
    '99.10':'Low Aggregate growth rate',
    '99.11':'High Aggregate growth rate',
    '99.12':r'Low $\rho$',
    '99.13':r'High $\rho$',
    '99.14':'Low Domestic patenting expenditure US',
    '99.15':'High Domestic patenting expenditure US',
    }

import math

for country in p_baseline.countries:
    fig,ax = plt.subplots()
    for i,rob_check in enumerate(variations_of_robust_checks):
        variation = rob_check
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
        df_welfare = pd.read_csv(local_path+'dyn_'+country+'.csv')
        if rob_check == 'baseline':
            ax.plot(df_welfare['delt'],df_welfare[country]*100-100,color='k',
                    label=variations_of_robust_checks[rob_check]
                    # ,lw=5
                    )
        else:
            if variations_of_robust_checks[rob_check].startswith('High'):
                ls = '-'
                zorder = 0
            if variations_of_robust_checks[rob_check].startswith('Low'):
                ls = '--'
                zorder = 0
            if variations_of_robust_checks[rob_check].startswith('baseline'):
                zorder = 99
            ax.plot(df_welfare['delt'],df_welfare[country]*100-100,
                    label=variations_of_robust_checks[rob_check],
                    color = Category18[math.floor((i-1)/2)],
                    ls = ls,
                    zorder = zorder)
    ax.set_ylabel('Welfare change (%)')
    if country == 'USA':
        ax.set_xlabel(r'Proportional change of $\delta_{US}$')
    elif country == 'EUR':
        ax.set_xlabel(r'Proportional change of $\delta_{Europe}$')
    else:
        ax.set_xlabel(r'Proportional change of $\delta$')
    ax.set_xscale('log')
    legend = ax.legend(ncol = 2,loc = [1.02,0])
    def export_legend(legend, filename="legend.pdf"):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend,filename = robustness_checks_path+'rob_check_legend.pdf')
    # plt.legend(False)
    legend.remove()
    # ax.legend(ncol = 2,loc=[1.02,0.02])
    # plt.title('Robustness check of unilateral patent policy counterfactual for '+countries_names[country])
    for save_format in save_formats:
        plt.savefig(robustness_checks_path+'rob_check_unilateral_patent_protection_'+country+'.'+save_format,format=save_format)
    plt.show()
    
    fig,ax = plt.subplots()
    for i,rob_check in enumerate(variations_of_robust_checks):
        variation = rob_check
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
                
        df_welfare = pd.read_csv(local_path+'dyn_'+country+'.csv')
        
        other_countries_welfare = pd.DataFrame(index = df_welfare.index,columns=['other_countries'])
        other_countries_welfare['delt'] = df_welfare['delt']
        other_countries_welfare['other_countries'] = 0
        
        for k,other_country in enumerate(p_baseline.countries):
            if other_country != country:
                other_countries_welfare['other_countries'] = other_countries_welfare['other_countries']+df_welfare[other_country]*p_baseline.labor[k]
        other_countries_welfare['other_countries'] = other_countries_welfare['other_countries']/sum([p_baseline.labor[l] for l,c in enumerate(p_baseline.countries) if c!=country])
        
        if rob_check == 'baseline':
            ax.plot(other_countries_welfare['delt'],other_countries_welfare['other_countries']*100-100,color='k',
                    label=variations_of_robust_checks[rob_check]
                    # ,lw=5
                    )
        else:
            if variations_of_robust_checks[rob_check].startswith('High'):
                ls = '-'
                zorder = 0
            if variations_of_robust_checks[rob_check].startswith('Low'):
                ls = '--'
                zorder = 0
            if variations_of_robust_checks[rob_check].startswith('baseline'):
                zorder = 99
            ax.plot(other_countries_welfare['delt'],other_countries_welfare['other_countries']*100-100,
                    label=variations_of_robust_checks[rob_check],
                    color = Category18[math.floor((i-1)/2)],zorder=zorder,
                    ls = ls)
    ax.set_ylabel('Welfare change (%)')
    if country == 'USA':
        ax.set_xlabel(r'Proportional change of $\delta_{US}$')
    elif country == 'EUR':
        ax.set_xlabel(r'Proportional change of $\delta_{Europe}$')
    else:
        ax.set_xlabel(r'Proportional change of $\delta$')
    ax.set_xscale('log')

    for save_format in save_formats:
        plt.savefig(robustness_checks_path+'rob_check_unilateral_patent_protection_'+country+'_other_countries.'+save_format,format=save_format)
    plt.show()

#%% Coop eq

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

coop_deltas = pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

df = coop_deltas.loc[(coop_deltas.baseline == int(baseline))
                        & (coop_deltas.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)

coop_w = pd.read_csv('coop_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

dfw = coop_w.loc[(coop_w.baseline == int(baseline))
                        & (coop_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1).set_index(['variation','aggregation_method'])

for i,rob_check in enumerate(variations_of_robust_checks):
    
    robustness_check_path = robustness_checks_path+variations_of_robust_checks[rob_check]+'/'
    
    try:
        os.mkdir(robustness_check_path)
    except:
        pass
    
    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
    
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
                            damping_post_acceleration=10
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    
    m = moments()
    m.load_run(local_path)
    m.compute_moments(sol_c,p)
    m.compute_moments_deviations()
    
    write_calibration_results(robustness_check_path+'new_baseline',p,m,sol_c,commentary = '')
    
    for coop in ['negishi','pop_weighted']:
        
        p.delta[...,1] = df.set_index(['variation','aggregation_method']).loc[(variation,coop)][p_baseline.countries].values.squeeze()
        
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
                                damping_post_acceleration=10
                                )
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p) 
        
        m.compute_moments(sol_c,p)
        m.compute_moments_deviations()
        write_calibration_results(robustness_check_path+'coop_'+coop,p,m,sol_c,commentary = '')
        
        dfw.loc[(variation,coop),'growth rate'] = sol_c.g

dfw = dfw.reset_index()

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries].sort_index().T.round(3) 
# df.to_csv(temp_path+'coop_equilibria.csv')

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['aggregation_method','Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi', 'growth rate']].sort_index()
dfw[p_baseline.countries+['Equal', 'Negishi']
    ] = (dfw[p_baseline.countries+['Equal', 'Negishi']]*100-100).round(2)

dfw['growth rate'] = (dfw['growth rate']*100).round(2)

df.to_csv(robustness_checks_path+'coop_equilibria.csv')
dfw.T.to_csv(robustness_checks_path+'coop_equilibria_welfare.csv')

#%%  Nash eq

nash_deltas = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

df = nash_deltas.loc[(nash_deltas.baseline == int(baseline))
                        & (nash_deltas.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1)


nash_w = pd.read_csv('nash_eq_recaps/dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')
dfw = nash_w.loc[(nash_w.baseline == int(baseline))
                        & (nash_w.variation.isin(list(variations_of_robust_checks.keys())))
                           ].drop('baseline',axis=1).set_index('variation').drop('method',axis=1)

for i,rob_check in enumerate(variations_of_robust_checks):
    
    robustness_check_path = robustness_checks_path+variations_of_robust_checks[rob_check]+'/'
    
    try:
        os.mkdir(robustness_check_path)
    except:
        pass
    
    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
        
    p.delta[...,1] = df.set_index(
        ['variation']).loc[variation][p_baseline.countries].values.squeeze()
    
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
                            damping_post_acceleration=10
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    
    m = moments()
    m.load_run(local_path)
    m.compute_moments(sol_c,p)
    m.compute_moments_deviations()
    
    write_calibration_results(robustness_check_path+'nash',p,m,sol_c,commentary = '')
    
    dfw.loc[variation,'growth rate'] = sol_c.g

dfw = dfw.reset_index()

df['variation'] = df['variation'].map(variations_of_robust_checks)
df[['Change','Quantity']] = df['variation'].str.split(' ',expand=True)
df = df.drop('variation',axis=1)
df = df.set_index(['Quantity','Change'])[p_baseline.countries].sort_index().T.round(3) 
# df.to_csv(temp_path+'coop_equilibria.csv')

dfw['variation'] = dfw['variation'].map(variations_of_robust_checks)
dfw[['Change','Quantity']] = dfw['variation'].str.split(' ',expand=True)
dfw = dfw.drop('variation',axis=1)
dfw = dfw.set_index(['Quantity','Change'])[p_baseline.countries+['Equal', 'Negishi', 'growth rate']].sort_index()
dfw[p_baseline.countries+['Equal', 'Negishi']
    ] = (dfw[p_baseline.countries+['Equal', 'Negishi']]*100-100).round(2)

dfw['growth rate'] = (dfw['growth rate']*100).round(2)

df.to_csv(robustness_checks_path+'nash_equilibrium.csv')
dfw.T.to_csv(robustness_checks_path+'nash_equilibrium_welfare.csv')

#%% pre-TRIPS Counterfactuals

df_welf = pd.DataFrame(index = p_baseline.countries+['Growth rate','North','South'])


for i,rob_check in enumerate(variations_of_robust_checks):
    print(rob_check)
    variation = rob_check
    variation_pre_trips = '1'+variation
    if variation == 'baseline':
        local_path = results_path+baseline+'/'
        local_path_pre_trips = results_path+f'baseline_{baseline}_variations/9.2/'
    else:
        local_path = \
            results_path+f'baseline_{baseline}_variations/{variation}/'
        local_path_pre_trips = \
            results_path+f'baseline_{baseline}_variations/{variation_pre_trips}/'
    p = parameters()
    p.load_run(local_path)
    
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
                            damping_post_acceleration=10
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p) 
    
    p_pre = parameters()
    p_pre.load_run(local_path_pre_trips)
    
    p_pre_cf_fix_north = p.copy()
    p_pre_cf_fix_north.delta[...,1] = p_pre.delta[...,1]
    for country_idx in [[p_baseline.countries.index(c) for c in rich_countries]]:
        p_pre_cf_fix_north.delta[country_idx,1] = p.delta[country_idx,1]

    _, sol_pre_cf_fix_north = fixed_point_solver(p_pre_cf_fix_north,context = 'counterfactual',x0=p_pre_cf_fix_north.guess,
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
    sol_pre_cf_fix_north.scale_P(p_pre_cf_fix_north)
    sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)
    sol_pre_cf_fix_north.compute_consumption_equivalent_welfare(p_pre_cf_fix_north,sol_c)
    sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf_fix_north,sol_c)

    _, dyn_sol_pre_cf_fix_north = dyn_fixed_point_solver(p_pre_cf_fix_north, sol_c,sol_fin=sol_pre_cf_fix_north,
                            Nt=25,t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=True,
                            damping = 50,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            )
    dyn_sol_pre_cf_fix_north.compute_non_solver_quantities(p_pre_cf_fix_north)
    
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 2015 delta'] = p.delta[:,1]
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 2015 delta'] = sol_c.g*100
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' 1992 delta'] = p_pre.delta[:,1]
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' 1992 delta'] = sol_pre_cf_fix_north.g*100
    
    df_welf.loc[p.countries,variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare*100-100
    df_welf.loc['Growth rate',variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.sol_fin.g*100
    
    # rich_count_indices = [0,1,2,6,7]
    # poor_count_indices = [3,4,5,8,9]
    
    df_welf.loc['World Equal',variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change*100-100
    df_welf.loc['North',variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.compute_consumption_equivalent_welfare_for_subset_of_countries(
        p_pre_cf_fix_north,['USA','EUR','JAP','CAN','KOR'])['pop_weighted']*100-100
    df_welf.loc['South',variations_of_robust_checks[rob_check]+' welfare change'] = dyn_sol_pre_cf_fix_north.compute_consumption_equivalent_welfare_for_subset_of_countries(
        p_pre_cf_fix_north,['CHN','BRA','IND','RUS','MEX'])['pop_weighted']*100-100
    
    print(df_welf)

    df_welf.loc['Diff South North'] = df_welf.loc['South'] - df_welf.loc['North']

df_welf.round(3).to_csv(robustness_checks_path+'pre_trips_welfares.csv')
