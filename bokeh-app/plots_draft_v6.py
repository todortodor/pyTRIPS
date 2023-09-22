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
from bokeh.palettes import Category10, Dark2
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

#%% setup path and stuff

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'
cf_path = 'counterfactual_recaps/unilateral_patent_protection/'
nash_eq_path = 'nash_eq_recaps/'
coop_eq_path = 'coop_eq_recaps/'

save_formats = ['eps','png','pdf']

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China','KOR':'Korea',
                   'CAN':'Canada','MEX':'Mexico','RUS':'Russia',
                  'BRA':'Brazil','IND':'India','ROW':'Rest of the world'}

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
output_name = 'draft_v6'

presentation_version_path = output_path+output_name+'/'
try:
    os.mkdir(presentation_version_path)
except:
    pass

#%% Choose a run, load parameters, moments, solution

baseline = '1020'
variation = 'baseline'

baseline_pre_trips_variation = baseline
pre_trips_cf = True
pre_trips_variation = '9.2'
partial_variation = '9.0'
variation_with_doubled_tau_in_pat_sect = '10.2'

baseline_pre_trips_full_variation = baseline
pre_trips_full_variation = '3.1'

output_path = 'output/'
output_name = 'draft_v6'

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

doubled_trade_costs_path = save_path+'doubled-trade-costs-optima/'
try:
    os.mkdir(doubled_trade_costs_path)
except:
    pass

dyn_save_path = save_path+'dynamics/'
try:
    os.mkdir(dyn_save_path)
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

    ax.plot(ori.index,ori[country],color=Category18[i],label=countries_names[country])
    
    ax.legend()
    
    plt.yscale('log')
    
ax.set_ylabel('International patent families by destination')
ax.set_xlim([1990,2015])

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
fig,ax = plt.subplots()
for i,country in enumerate(p_baseline.countries):

    ax.plot(ori.index,ori[country],color=Category18[i],label=countries_names[country])
    
    ax.legend()
    
    plt.yscale('log')
    
ax.set_ylabel('International patent families by origin')
ax.set_xlim([1990,2015])

for save_format in save_formats:
    plt.savefig(data_fact_path+'international_pat_families_by_origin.'+save_format,format=save_format)
    
plt.show()

#%% write excel spredsheet of the baseline

write_calibration_results(calibration_path+'baseline',p_baseline,m_baseline,sol_baseline,commentary = '')

#%% Comparing trade flows with patent flows

fig,ax = plt.subplots()

tflow_shares = m_baseline.ccs_moments.query('destination_code!=origin_code'
                                    ).xs(1,level=2)
tflow_shares = tflow_shares/tflow_shares.sum()
pflow_shares = m_baseline.cc_moments.query('destination_code!=origin_code'
                                    )
pflow_shares = pflow_shares/pflow_shares.sum()

ax.scatter(pflow_shares.values.ravel()[pflow_shares.values.ravel()>1e-6],
            m_baseline.SPFLOW.ravel()[pflow_shares.values.ravel()>1e-6],
            label = 'International patent shares: model')
ax.scatter(pflow_shares.values.ravel()[pflow_shares.values.ravel()>1e-6],
            tflow_shares.values.ravel()[pflow_shares.values.ravel()>1e-6],
            label = 'International trade shares',marker='^')

ax.set_xlabel('International patent shares: data')
ax.set_ylabel('Model')

ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(np.sort(m_baseline.SPFLOW_target.ravel()[m_baseline.SPFLOW_target.ravel()>1e-6]),
        np.sort(m_baseline.SPFLOW_target.ravel()[m_baseline.SPFLOW_target.ravel()>1e-6]),
        ls='--',color='grey')

plt.legend()

for save_format in save_formats:
    plt.savefig(calibration_path+'trade_vs_patent_flows.'+save_format,format=save_format)

plt.show()

#%% plot matching of moments : SPFLOW

moment = 'SPFLOW'

annotate_with_labels = True
replace_country_codes_with_labels = False

labels = m_baseline.idx[moment].to_series().agg(','.join).astype('str')
if replace_country_codes_with_labels:
    for code, name in countries_names.items():
        labels = labels.str.replace(code,name)

x = getattr(m_baseline, moment+'_target').ravel()
y = getattr(m_baseline, moment).ravel()

fig, ax = plt.subplots()

ax.scatter(x,y)
ax.plot(x,x,color='grey',ls='--',lw=1)
if annotate_with_labels:
    texts = [ax.annotate(label,
                         xy=(x[i],y[i]),
                        xytext=(2,2),
                        textcoords='offset points',
                         fontsize = 10
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
    plt.savefig(calibration_path+moment+'.'+save_format,format=save_format)

plt.show()

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

# for c in p_baseline.countries+['World','Uniform_delta']:
for c in ['Uniform_delta']:
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
    ax.set_ylabel('Welfare change')
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
        ax.plot(recap.delta_change,recap[country],color=Category18[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi'],color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal'],color='k',ls=':',label='World Equal')
    ax.legend()
    
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

for with_world in [False]:

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
    ax.legend()
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

for c in p_baseline.countries+['World','Uniform_delta','trade_cost_eq_trips_all_countries_pat_sectors']:
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
    
    plt.xscale('log')
    
    ax.set_ylabel('Welfare change')
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
        ax.set_ylim(0.98,1.02)
        plt.xscale('linear')

    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap.delta_change,recap[country],color=Category18[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi'],color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal'],color='k',ls=':',label='World Equal')

    ax.legend()

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
for country_idx in [0,1,2,6,7,10]:
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

p_pre_cf_fix_north_and_tau = p_baseline.copy()
p_pre_cf_fix_north_and_tau.delta[...,1] = p_pre.delta[...,1]
p_pre_cf_fix_north_and_tau.tau = p_pre.tau.copy()
for country_idx in [0,1,2,6,7,10]:
    p_pre_cf_fix_north_and_tau.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north_and_tau = fixed_point_solver(p_pre_cf_fix_north_and_tau,
                                                     context = 'counterfactual',
                        x0=p_pre_cf_fix_north_and_tau.guess,
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
sol_pre_cf_fix_north_and_tau.scale_P(p_pre_cf_fix_north_and_tau)
sol_pre_cf_fix_north_and_tau.compute_non_solver_quantities(p_pre_cf_fix_north_and_tau)
sol_pre_cf_fix_north_and_tau.compute_consumption_equivalent_welfare(p_pre_cf_fix_north_and_tau,
                                                                    sol_baseline)
sol_pre_cf_fix_north_and_tau.compute_world_welfare_changes(p_pre_cf_fix_north_and_tau,
                                                           sol_baseline)

_, dyn_sol_pre_cf_fix_north_and_tau = dyn_fixed_point_solver(p_pre_cf_fix_north_and_tau, 
                        sol_baseline,sol_fin=sol_pre_cf_fix_north_and_tau,
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
dyn_sol_pre_cf_fix_north_and_tau.compute_non_solver_quantities(p_pre_cf_fix_north_and_tau)

p_pre_cf_fix_north_and_tau_in_pat_sect = p_baseline.copy()
p_pre_cf_fix_north_and_tau_in_pat_sect.delta[...,1] = p_pre.delta[...,1]
p_pre_cf_fix_north_and_tau_in_pat_sect.tau[...,1] = p_pre.tau[...,1]
for country_idx in [0,1,2,6,7,10]:
    p_pre_cf_fix_north_and_tau_in_pat_sect.delta[country_idx,1] = p_baseline.delta[country_idx,1]

_, sol_pre_cf_fix_north_and_tau_in_pat_sect = fixed_point_solver(p_pre_cf_fix_north_and_tau_in_pat_sect,
                                                     context = 'counterfactual',
                        x0=p_pre_cf_fix_north_and_tau_in_pat_sect.guess,
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
sol_pre_cf_fix_north_and_tau_in_pat_sect.scale_P(p_pre_cf_fix_north_and_tau_in_pat_sect)
sol_pre_cf_fix_north_and_tau_in_pat_sect.compute_non_solver_quantities(p_pre_cf_fix_north_and_tau_in_pat_sect)
sol_pre_cf_fix_north_and_tau_in_pat_sect.compute_consumption_equivalent_welfare(p_pre_cf_fix_north_and_tau_in_pat_sect,sol_baseline)
sol_pre_cf_fix_north_and_tau_in_pat_sect.compute_world_welfare_changes(p_pre_cf_fix_north_and_tau_in_pat_sect,sol_baseline)

_, dyn_sol_pre_cf_fix_north_and_tau_in_pat_sect = dyn_fixed_point_solver(p_pre_cf_fix_north_and_tau_in_pat_sect, 
                        sol_baseline,sol_fin=sol_pre_cf_fix_north_and_tau_in_pat_sect,
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
dyn_sol_pre_cf_fix_north_and_tau_in_pat_sect.compute_non_solver_quantities(
    p_pre_cf_fix_north_and_tau_in_pat_sect)

p_partial_2015 = parameters()
p_partial_2015.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{partial_variation}/')

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
 'ROW': 'Rest of\nthe world'}

df = pd.DataFrame(
    index = pd.Index([modified_countries_names[c] for c in p_baseline.countries]+['World\nNegishi','World\nEqual'],
                                       name = 'country')
    )

df['delta_baseline'] = p_baseline.delta[...,1].tolist()+[np.nan,np.nan]
df['delta_1992'] = p_pre.delta[...,1].tolist()+[np.nan,np.nan]
df['delta_partial_2015'] = p_partial_2015.delta[...,1].tolist()+[np.nan,np.nan]
df['static_welfare_change'] = sol_pre_cf.cons_eq_welfare.tolist()+[
    sol_pre_cf.cons_eq_negishi_welfare_change,sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change'] = dyn_sol_pre_cf.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf.cons_eq_negishi_welfare_change,dyn_sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north_and_cf_tau'] = sol_pre_cf_fix_north_and_tau.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north_and_tau.cons_eq_negishi_welfare_change,
    sol_pre_cf_fix_north_and_tau.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north_and_cf_tau'] = dyn_sol_pre_cf_fix_north_and_tau.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north_and_tau.cons_eq_negishi_welfare_change,
    dyn_sol_pre_cf_fix_north_and_tau.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north_and_cf_tau_in_pat_sect'] = sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_negishi_welfare_change,
    sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change_fixed_delta_north_and_cf_tau_in_pat_sect'] = dyn_sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_negishi_welfare_change,
    dyn_sol_pre_cf_fix_north_and_tau_in_pat_sect.cons_eq_pop_average_welfare_change
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
            'static_welfare_change_fixed_delta_north_and_cf_tau','dynamic_welfare_change_fixed_delta_north_and_cf_tau',
            'static_welfare_change_fixed_delta_north_and_cf_tau_in_pat_sect','dynamic_welfare_change_fixed_delta_north_and_cf_tau_in_pat_sect',
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
df.loc['growth_rate','static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north_and_cf_tau'] = sol_pre_cf_fix_north_and_tau.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north_and_cf_tau_in_pat_sect'
       ] = sol_pre_cf_fix_north_and_tau_in_pat_sect.g
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
for country_idx in [0,1,2,6,7,10]:
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

#%% trade cost equivalent of TRIPS for specific countries

cf_path = 'counterfactual_recaps/unilateral_patent_protection/'

def find_zeros(df_welfare,country):
    x1 = df_welfare.loc[df_welfare[country]>1].iloc[-1]['delt']
    x2 = df_welfare.loc[df_welfare[country]<1].iloc[0]['delt']
    y1 = df_welfare.loc[df_welfare[country]>1].iloc[-1][country]-1
    y2 = df_welfare.loc[df_welfare[country]<1].iloc[0][country]-1
    
    return (y2*x1-x2*y1)/(y2-y1)

countries = ['CHN','IND','RUS']
exercises = ['trade_cost_eq_trips_exp_imp_pat_sect','dyn_trade_cost_eq_trips_exp_imp_pat_sect']

df = pd.DataFrame(index = [countries_names[c] for c in countries])
data_for_plot = {}

for country in countries:
    for i,exercise in enumerate(exercises):
        if variation == 'baseline':
            local_path = cf_path+'baseline_'+baseline+'/'
        else:
            local_path = \
                cf_path+f'baseline_{baseline}_{variation}/'
        if exercise == 'trade_cost_eq_trips_exp_imp_pat_sect':
            df_welfare = pd.read_csv(local_path+country+'_trade_cost_eq_trips_exp_imp_pat_sect'+'.csv')
        if exercise == 'dyn_trade_cost_eq_trips_exp_imp_pat_sect':
            df_welfare = pd.read_csv(local_path+'dyn_'+country+'_trade_cost_eq_trips_exp_imp_pat_sect'+'.csv')
        if i == 0:
            data_for_plot['delt'] = df_welfare['delt'] 
                
        df.loc[countries_names[country],exercise] = find_zeros(df_welfare,country)
        if exercise == 'dyn_trade_cost_eq_trips_exp_imp_pat_sect':
            data_for_plot[country] = df_welfare[country]

caption = 'Trade cost equivalent of TRIPS when changing country-specific trade costs (imports and exports in patenting sector)'

df.style.format(precision=5).to_latex(pre_TRIPS_plots_path+'trade_cost_eq_trips_exp_imp_pat_sect.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(pre_TRIPS_plots_path+'trade_cost_eq_trips_exp_imp_pat_sect.csv',float_format='%.5f')

#%% Coop and Nash equilibria with doubled trade costs in patenting sector

#%% Nash table with transitional dynamics with doubled trade costs in patenting sector

all_nashes = pd.read_csv('nash_eq_recaps/dyn_deltas.csv')
all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) 
                         & (all_nashes.variation.astype(str) == variation_with_doubled_tau_in_pat_sect)]

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

m_pre = moments()
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')

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

df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Nash_table_with_doubled_trade_costs_in_pat_sect.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_trade_costs_path+'dyn_Nash_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

write_calibration_results(doubled_trade_costs_path+'dyn_Nash_with_doubled_trade_costs_in_pat_sect',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')


#%% Coop equal weights table with transitional dynamics with doubled trade costs in patenting sector

all_coop_equales = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
                                     & (all_coop_equales.variation == variation_with_doubled_tau_in_pat_sect)
                                     & (all_coop_equales.aggregation_method == 'pop_weighted')]

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')
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
m_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{variation_with_doubled_tau_in_pat_sect}/')

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

df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Coop_population_weights_table_with_doubled_trade_costs_in_pat_sect.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_trade_costs_path+'dyn_Coop_population_weights_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

write_calibration_results(doubled_trade_costs_path+'dyn_Coop_population_weights_with_doubled_trade_costs_in_pat_sect',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

#%% Coop negishi weights table with transitional dynamics with doubled trade costs in patenting sector

all_coop_negishies = pd.read_csv('coop_eq_recaps/dyn_deltas.csv')
all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
                                     & (all_coop_negishies.variation == variation)
                                     & (all_coop_negishies.aggregation_method == 'negishi')]

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

df.style.format(precision=5).to_latex(doubled_trade_costs_path+'dyn_Coop_negishi_weights_table_with_doubled_trade_costs_in_pat_sect.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(doubled_trade_costs_path+'dyn_Coop_negishi_weights_table_with_doubled_trade_costs_in_pat_sect.csv',float_format='%.5f')

write_calibration_results(doubled_trade_costs_path+'dyn_Coop_negishi_weights_with_doubled_trade_costs_in_pat_sect',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')

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

p_pre_full = parameters()
p_pre_full.load_run(
    f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_full_variation}/'
    )

_, sol_pre_full = fixed_point_solver(p_pre_full,context = 'counterfactual',x0=p_pre_full.guess,
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
sol_pre_full.scale_P(p_pre_full)
sol_pre_full.compute_non_solver_quantities(p_pre_full)

'Semi-elasticity partial','GE effect, steady state','GE effect, with dynamics'

df.loc['2015 baseline calibration','Semi-elasticity partial'] = sol_baseline.semi_elast_patenting_delta[0,1]

df.loc['2015 baseline calibration','GE effect, steady state'
       ] = compute_ge_semielasticity(p_baseline,sol_baseline,dynamics=False)
df.loc['2015 baseline calibration','GE effect, with dynamics'
       ] = compute_ge_semielasticity(p_baseline,sol_baseline,dynamics=True)

df.loc['1992 full calibration','Semi-elasticity partial'] = sol_pre_full.semi_elast_patenting_delta[0,1]

df.loc['1992 full calibration','GE effect, steady state'
        ] = compute_ge_semielasticity(p_pre_full,sol_pre_full,dynamics=False)
df.loc['1992 full calibration','GE effect, with dynamics'
        ] = compute_ge_semielasticity(p_pre_full,sol_pre_full,dynamics=True)

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
plt.legend()
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
plt.legend()
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
plt.legend()
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
plt.legend()
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
plt.legend()
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
plt.legend()
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