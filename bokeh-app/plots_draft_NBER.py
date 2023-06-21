#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:19:43 2023

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

countries_names = {'USA':'USA','EUR':'Europe','JAP':'Japan','CHN':'China',
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
output_name = 'draft_NBER'

presentation_version_path = output_path+output_name+'/'
try:
    os.mkdir(presentation_version_path)
except:
    pass

#%% Choose a run, load parameters, moments, solution

baseline = '607'
variation = 'baseline'

baseline_pre_trips_variation = '618'
pre_trips_cf = True
pre_trips_variation = '5.1'

output_path = 'output/'
output_name = 'draft_NBER'

save_path = output_path+output_name+'/'+baseline+'_'+variation+'/'

try:
    os.mkdir(save_path)
except:
    pass

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

df.style.format(precision=2).to_latex(save_path+'pat_over_trade_coeffs.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'pat_over_trade_coeffs.csv',float_format='%.2f')

df_without_diag.style.format(precision=2).to_latex(save_path+'pat_over_trade_coeffs_without_diag.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df_without_diag.to_csv(save_path+'pat_over_trade_coeffs_without_diag.csv',float_format='%.2f')

#%% write excel spredsheet of the baseline

write_calibration_results(save_path+'baseline',p_baseline,m_baseline,sol_baseline,commentary = '')

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
    plt.savefig(save_path+moment+'.'+save_format,format=save_format)

plt.show()

#%% plot matching of moments : TWSPFLOW

moment = 'TWSPFLOW'

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
    plt.savefig(save_path+moment+'.'+save_format,format=save_format)

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

df.style.format(precision=6).to_latex(save_path+'scalar_moments_matching.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+'scalar_moments_matching.csv',float_format='%.6f')

#%% output table for matching of real GDP

moment = 'SRGDP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv',float_format='%.6f')

#%% output table for matching of price_indices

moment = 'RP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv',float_format='%.6f')

#%% output table for matching of RD expenditures

moment = 'RD'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

df['Model'] = df['Model'].astype(float)
df['Target'] = df['Target'].astype(float)

caption = m_baseline.description.loc[moment,'description']

df.style.format(precision=6).to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv',float_format='%.6f')

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

df.style.format(precision=6).to_latex(save_path+'scalar_parameters.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+'scalar_parameters.csv',float_format='%.6f')

#%% plot parameter delta and output table

parameter = 'delta'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

fig, ax = plt.subplots()

ax.bar(x,y)
    
# plt.xlabel('Country')
plt.ylabel(r'$\delta$',rotation=0,fontsize = 30)

# plt.title('Patent protection')
plt.title(parameters_description[parameter])

for save_format in save_formats:
    plt.savefig(save_path+parameter+'.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df = df.rename(columns = {'delta':r'$\delta$'})

df.style.format(precision=6).to_latex(save_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'.csv',float_format='%.6f')

#%% plot parameter eta and output table

parameter = 'eta'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

fig, ax = plt.subplots()

ax.bar(x,y)
    
# plt.xlabel('Country')
plt.ylabel(r'$\eta$',rotation=0,fontsize = 30)

plt.title(parameters_description[parameter])

for save_format in save_formats:
    plt.savefig(save_path+parameter+'.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df = df.rename(columns = {'eta':r'$\eta$'})

df.style.format(precision=6).to_latex(save_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'.csv',float_format='%.6f')

#%% plot parameter T in non patenting sector and output table

parameter = 'T'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,0]

fig, ax = plt.subplots()

ax.bar(x,y)
    
# plt.xlabel('Country')
plt.ylabel(parameter,rotation=0)

plt.title(parameters_description[parameter]+' in non patenting sector')

for save_format in save_formats:
    plt.savefig(save_path+parameter+'_non_patenting.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df.style.format(precision=6).to_latex(save_path+parameter+'_non_patenting.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'_non_patenting.csv',float_format='%.6f')

#%% plot parameter T in patenting sector and output table

parameter = 'T'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

fig, ax = plt.subplots()

ax.bar(x,y)
    
# plt.xlabel('Country')
plt.ylabel(parameter,rotation=0)

plt.title(parameters_description[parameter]+' in patenting sector')

for save_format in save_formats:
    plt.savefig(save_path+parameter+'_patenting.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = y

caption = parameters_description[parameter]

df.style.format(precision=6).to_latex(save_path+parameter+'_patenting.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'_patenting.csv',float_format='%.6f')


#%% Unilateral patent protections counterfactuals

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
    for run in run_list:
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
            if c == 'Uniform_delta':
                recap.loc[run, 'delta_change'] = p.delta[0,1]
            recap.loc[run, 'growth'] = sol_c.g
            recap.loc[run, 'world_negishi'] = sol_c.cons_eq_negishi_welfare_change
            recap.loc[run, 'world_equal'] = sol_c.cons_eq_pop_average_welfare_change
            recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
            
            recap_growth_rate.loc[run,'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]
            recap_growth_rate.loc[run,c] = sol_c.g

    fig,ax = plt.subplots()
    # ax2 = ax.twinx()
    plt.xscale('log')
    ax.set_ylabel('Welfare change')
    if c in p_baseline.countries:
        ax.set_xlabel(r'Proportional change of $\delta$')
    if c == 'World':
        ax.set_xlabel(r'Proportional change of $\delta$ of all countries')
    if c == 'Uniform_delta':
        ax.set_xlabel(r'Harmonized $\delta$')
        plt.axvline(x=p_baseline.delta[0,1], lw = 1, color = 'k')
        xt = ax.get_xticks() 
        # xt = np.linspace(0.01,0.1,10).round(2).tolist()+np.linspace(0.1,1,10).round(2).tolist()[1:]
        xt=np.append(xt,p_baseline.delta[0,1])
        xtl=xt.tolist()
        xtl[-1]=r'$\delta_{US}$'
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl,rotation=45)
    # ax2.set_ylabel('Growth rate change')

    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap.delta_change,recap[country],color=sns.color_palette()[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi'],color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal'],color='k',ls=':',label='World Equal')
    # ax.plot([],[],color='grey',ls='-.',label='Growth rate')
    
    # ax.plot(recap.delta_change,recap['growth'].values/(recap.loc[recap.delta_change == 1]['growth'].values),color='grey',ls='-.',label='Growth rate')
    # ax2.grid(False)
    
    # ax2.legend()
    ax.legend()
    # plt.title('Response to unilateral patent protection change')
    
    # plt.yscale('log')
    for save_format in save_formats:
        plt.savefig(save_path+c+'_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
    plt.show()
    
    if c in p_baseline.countries:
        caption = 'Consumption equivalent welfares in the unilateral patent protection counterfactual of '+countries_names[c]
    if c == 'World':
        caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
    if c == 'Uniform_delta':
        caption = 'Consumption equivalent welfares in the harmonized delta counterfactual change of all countries'
        recap = recap.rename(columns = {'delta_change':'delta'})
        
    recap.style.to_latex(save_path+c+'_unilateral_patent_protection_counterfactual.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap.to_csv(save_path+c+'_unilateral_patent_protection_counterfactual.csv')
    
    if c == 'Uniform_delta':
        delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_baseline.delta[0,1]))].to_frame()
        delta_US_values.style.to_latex(save_path+c+'_US_values.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        delta_US_values.to_csv(save_path+c+'_US_values.csv')

#%% Counterfactual growth rates

for with_world in [True,False]:

    fig,ax = plt.subplots()
    
    ax.set_ylabel('Growth rate (%)')
    ax.set_xlabel(r'Proportional change of $\delta$')
    
    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap_growth_rate.delta_change,
                recap_growth_rate[country]*100,
                color=sns.color_palette()[i],
                label=countries_names[country])
    if with_world:
        ax.plot(recap_growth_rate.delta_change,
                recap_growth_rate['World']*100,color='grey',
                label='All countries',ls='--')
    ax.legend()
    plt.xscale('log')
    for save_format in save_formats:
        if with_world:
            save_name = save_path+'growth_rate_unilateral_patent_protection_counterfactual_with_world.'
        else:
            save_name = save_path+'growth_rate_unilateral_patent_protection_counterfactual.'
        plt.savefig(save_name+save_format,format=save_format)
    plt.show()
    
    caption = 'Counterfactual growth rates'
    
    
    recap_growth_rate.style.to_latex(save_name+'tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap_growth_rate.to_csv(save_name+'csv')


#%% Nash table

all_nashes = pd.read_csv('nash_eq_recaps/deltas.csv')
all_nashes = all_nashes.drop_duplicates(['baseline','variation'],keep='last')

run_nash= all_nashes.loc[(all_nashes.baseline == int(baseline)) & (all_nashes.variation == variation)]

p_nash = p_baseline.copy()
p_nash.delta[:,1] = run_nash[p_baseline.countries].values.squeeze()

sol, sol_nash = fixed_point_solver(p_nash,x0=p_nash.guess,
                        context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
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
                        )
sol_nash.scale_P(p_nash)
sol_nash.compute_non_solver_quantities(p_nash)
sol_nash.compute_consumption_equivalent_welfare(p_nash,sol_baseline)
sol_nash.compute_world_welfare_changes(p_nash,sol_baseline)

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
                        disp_summary=True,
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
m_nash.compute_moments(sol_nash,p_nash)
m_nash.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only','Growth rate']
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

df.loc['World aggregate according to Negishi weights',
        'Growth rate'] = sol_nash.g

df.loc['World aggregate according to population weights',
        'Growth rate'] = sol_nash.g

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Nash equilibrium'

df.style.format(precision=5).to_latex(save_path+'Nash_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Nash_table.csv',float_format='%.5f')

write_calibration_results(save_path+'Nash',p_nash,m_nash,sol_nash,commentary = '')


#%% Coop equal weights table

all_coop_equales = pd.read_csv('coop_eq_recaps/deltas.csv')
all_coop_equales = all_coop_equales.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_equal= all_coop_equales.loc[(all_coop_equales.baseline == int(baseline))
                                     & (all_coop_equales.variation == variation)
                                     & (all_coop_equales.aggregation_method == 'pop_weighted')]

p_coop_equal = p_baseline.copy()
p_coop_equal.delta[:,1] = run_coop_equal[p_baseline.countries].values.squeeze()

sol, sol_coop_equal = fixed_point_solver(p_coop_equal,x0=p_coop_equal.guess,
                        context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
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
                        )
sol_coop_equal.scale_P(p_coop_equal)
sol_coop_equal.compute_non_solver_quantities(p_coop_equal)
sol_coop_equal.compute_consumption_equivalent_welfare(p_coop_equal,sol_baseline)
sol_coop_equal.compute_world_welfare_changes(p_coop_equal,sol_baseline)

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
                        disp_summary=True,
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
m_coop_equal.compute_moments(sol_coop_equal,p_coop_equal)
m_coop_equal.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only','Growth rate']
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

df.loc['World aggregate according to Negishi weights',
        'Growth rate'] = sol_coop_equal.g

df.loc['World aggregate according to population weights',
        'Growth rate'] = sol_coop_equal.g

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with population weights'

df.style.format(precision=5).to_latex(save_path+'Coop_population_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Coop_population_weights_table.csv',float_format='%.5f')

write_calibration_results(save_path+'Coop_population_weights',p_coop_equal,m_coop_equal,sol_coop_equal,commentary = '')


#%% Coop negishi weights table

all_coop_negishies = pd.read_csv('coop_eq_recaps/deltas.csv')
all_coop_negishies = all_coop_negishies.drop_duplicates(['baseline',
                                                     'variation',
                                                     'aggregation_method'],keep='last')

run_coop_negishi= all_coop_negishies.loc[(all_coop_negishies.baseline == int(baseline))
                                     & (all_coop_negishies.variation == variation)
                                     & (all_coop_negishies.aggregation_method == 'negishi')]

p_coop_negishi = p_baseline.copy()
p_coop_negishi.delta[:,1] = run_coop_negishi[p_baseline.countries].values.squeeze()

sol, sol_coop_negishi = fixed_point_solver(p_coop_negishi,x0=p_coop_negishi.guess,
                        context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
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
                        )
sol_coop_negishi.scale_P(p_coop_negishi)
sol_coop_negishi.compute_non_solver_quantities(p_coop_negishi)
sol_coop_negishi.compute_consumption_equivalent_welfare(p_coop_negishi,sol_baseline)
sol_coop_negishi.compute_world_welfare_changes(p_coop_negishi,sol_baseline)

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
                        disp_summary=True,
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
m_coop_negishi.compute_moments(sol_coop_negishi,p_coop_negishi)
m_coop_negishi.compute_moments_deviations()
    
df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Welfare change with transition dynamics',
                             'Welfare change, steady state only','Growth rate']
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

df.loc['World aggregate according to Negishi weights',
        'Growth rate'] = sol_coop_negishi.g

df.loc['World aggregate according to population weights',
        'Growth rate'] = sol_coop_negishi.g

for col in df.columns:
    df[col] = df[col].astype(float)
    
caption = 'Cooperative equilibrium with Negishi weights'

df.style.format(precision=5).to_latex(save_path+'Coop_negishi_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Coop_negishi_weights_table.csv',float_format='%.5f')

write_calibration_results(save_path+'Coop_negishi_weights',p_coop_negishi,m_coop_negishi,sol_coop_negishi,commentary = '')


#%%  Counterfactuals with transitional dynamics

#%% Unilateral patent protections counterfactuals with dynamics

for c in p_baseline.countries+['World','Uniform_delta']:
# for c in ['Uniform_delta']:
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
    for run in run_list:
        # p = parameters(n=7,s=2)
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
    if c == 'Uniform_delta':
        ax.set_xlabel(r'Harmonized $\delta$')
        plt.axvline(x=p_baseline.delta[0,1], lw = 1, color = 'k')
        xt = ax.get_xticks() 
        xt=np.append(xt,p_baseline.delta[0,1])
        xtl=xt.tolist()
        xtl[-1]=r'$\delta_{US}$'
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

    for i,country in enumerate(p_baseline.countries):
        ax.plot(recap.delta_change,recap[country],color=sns.color_palette()[i],label=countries_names[country])
    
    ax.plot(recap.delta_change,recap['world_negishi'],color='k',ls='--',label='World Negishi')
    ax.plot(recap.delta_change,recap['world_equal'],color='k',ls=':',label='World Equal')

    ax.legend()

    for save_format in save_formats:
        plt.savefig(save_path+c+'_dyn_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
    plt.show()
    
    if c in p_baseline.countries:
        caption = 'Consumption equivalent welfares in the unilateral patent protection counterfactual of '+countries_names[c]
    if c == 'World':
        caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
    if c == 'Uniform_delta':
        caption = 'Consumption equivalent welfares in the harmonized delta counterfactual change of all countries'
        recap = recap.rename(columns = {'delta_change':'delta'})
    
    recap.style.to_latex(save_path+c+'_dyn_unilateral_patent_protection_counterfactual.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap.to_csv(save_path+c+'_dyn_unilateral_patent_protection_counterfactual.csv')
    
    if c == 'Uniform_delta':
        delta_US_values = recap.iloc[np.argmin(np.abs(recap.delta-p_baseline.delta[0,1]))].to_frame()
        delta_US_values.style.to_latex(save_path+c+'_dyn_US_values.tex',
                          caption=caption,
                          **save_to_tex_options
                          )
        delta_US_values.to_csv(save_path+c+'_dyn_US_values.csv')

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
                                     'World aggregate according to population weights'],
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

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Nash equilibrium'

df.style.format(precision=5).to_latex(save_path+'dyn_Nash_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'dyn_Nash_table.csv',float_format='%.5f')

write_calibration_results(save_path+'dyn_Nash',p_nash,m_nash,dyn_sol_nash.sol_fin,commentary = '')

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
                                     'World aggregate according to population weights'],
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

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with population weights'

df.style.format(precision=5).to_latex(save_path+'dyn_Coop_population_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'dyn_Coop_population_weights_table.csv',float_format='%.5f')

write_calibration_results(save_path+'dyn_Coop_population_weights',p_coop_equal,m_coop_equal,dyn_sol_coop_equal.sol_fin,commentary = '')

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
                                     'World aggregate according to population weights'],
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

for col in df.columns:
    df[col] = df[col].astype(float)

caption = 'Cooperative equilibrium with Negishi weights'

df.style.format(precision=5).to_latex(save_path+'dyn_Coop_negishi_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'dyn_Coop_negishi_weights_table.csv',float_format='%.5f')

write_calibration_results(save_path+'dyn_Coop_negishi_weights',p_coop_negishi,m_coop_negishi,dyn_sol_coop_negishi.sol_fin,commentary = '')

#%% pre-TRIPS calibration and counterfactual

p_pre = parameters()
p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
_, sol_pre = fixed_point_solver(p_pre,context = 'calibration',x0=p_pre.guess,
                        cobweb_anim=False,tol =1e-15,
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
                        cobweb_anim=False,tol =1e-15,
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
p_pre_cf_fix_north.delta[0:3,1] = p_baseline.delta[0:3,1]

_, sol_pre_cf_fix_north = fixed_point_solver(p_pre_cf_fix_north,context = 'counterfactual',x0=p_pre_cf_fix_north.guess,
                        cobweb_anim=False,tol =1e-15,
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
sol_pre_cf_fix_north.compute_world_welfare_changes(p_pre_cf,sol_baseline)

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

pre_trips_path = save_path+'pre-trips/'
try:
    os.mkdir(pre_trips_path)
except:
    pass

modified_countries_names = {'USA': 'USA',
 'EUR': 'Europe',
 'JAP': 'Japan',
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
df['static_welfare_change'] = sol_pre_cf.cons_eq_welfare.tolist()+[
    sol_pre_cf.cons_eq_negishi_welfare_change,sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare_change'] = dyn_sol_pre_cf.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf.cons_eq_negishi_welfare_change,dyn_sol_pre_cf.cons_eq_pop_average_welfare_change
    ]
df['static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
df['dynamic_welfare change_fixed_delta_north'] = dyn_sol_pre_cf_fix_north.cons_eq_welfare.tolist()+[
    dyn_sol_pre_cf_fix_north.cons_eq_negishi_welfare_change,dyn_sol_pre_cf_fix_north.cons_eq_pop_average_welfare_change
    ]
grey_rgb = (105/256,105/256,105/256)
# grey_rgb = (0,0,0)

for col in ['static_welfare_change','dynamic_welfare_change',
            'static_welfare_change_fixed_delta_north','dynamic_welfare change_fixed_delta_north']:

    fig,ax = plt.subplots()
    # ax.bar(df.index, df['static welfare change']*100-100)
    ax.barh(df.index, df[col]*100-100, 
            color = sns.color_palette()[:len(p_baseline.countries)]+[grey_rgb,grey_rgb],
           # color = sns.color_palette()[:len(p_baseline.countries)+2],
           # hatch = ['']*len(p_baseline.countries)+['/','x']
           )
    ax.invert_yaxis()
    ax.set_xlabel('Welfare change (%)')
    
    for save_format in save_formats:
        plt.savefig(pre_trips_path+col+'.'+save_format,format=save_format)
    
    plt.show()

df.loc['growth_rate','delta_baseline'] = sol_baseline.g
df.loc['growth_rate','static_welfare_change'] = sol_pre_cf.g
df.loc['growth_rate','static_welfare_change_fixed_delta_north'] = sol_pre_cf_fix_north.g

caption = 'Pre TRIPS calibration and counterfacual'

df.style.format(precision=5).to_latex(pre_trips_path+'pre_trips.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(pre_trips_path+'pre_trips.csv',float_format='%.5f')