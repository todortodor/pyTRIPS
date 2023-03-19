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
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
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

baseline = '501'
variation = '1.0'

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

p_baseline = parameters(n=7,s=2)
p_baseline.load_data(run_path)

m_baseline = moments()
m_baseline.load_data()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

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
                                       columns = ['Target','Value'])

for moment in m_baseline.list_of_moments:
    if getattr(m_baseline,moment).size == 1:
        df.loc[m_baseline.description.loc[moment,'description'],'Value'] = getattr(m_baseline,moment)
        df.loc[m_baseline.description.loc[moment,'description'],'Target'] = getattr(m_baseline,moment+'_target')

caption = 'Scalar moments targeted'

df.style.to_latex(save_path+'scalar_moments_matching.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+'scalar_moments_matching.csv')

#%% output table for matching of real GDP

moment = 'SRGDP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

caption = m_baseline.description.loc[moment,'description']

df.style.to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv')

#%% output table for matching of price_indices

moment = 'RP'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

caption = m_baseline.description.loc[moment,'description']

df.style.to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv')

#%% output table for matching of RD expenditures

moment = 'RD'

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                       columns = ['Target','Model'])

df['Target'] = getattr(m_baseline,moment+'_target')
df['Model'] = getattr(m_baseline,moment)

caption = m_baseline.description.loc[moment,'description']

df.style.to_latex(save_path+moment+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+moment+'.csv')

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

caption = 'Scalar parameters'

df.style.to_latex(save_path+'scalar_parameters.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+'scalar_parameters.csv')

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
df[parameter] = p_baseline.delta[:,1]

caption = parameters_description[parameter]

df = df.rename(columns = {'delta':r'$\delta$'})

df.style.to_latex(save_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'.csv')

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
df[parameter] = p_baseline.delta[:,1]

caption = parameters_description[parameter]

df = df.rename(columns = {'eta':r'$\eta$'})

df.style.to_latex(save_path+parameter+'.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'.csv')

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
    plt.savefig(save_path+parameter+'.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = p_baseline.delta[:,1]

caption = parameters_description[parameter]

df.style.to_latex(save_path+parameter+'_non_patent_sector.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'_non_patent_sector.csv')

#%% plot parameter T in non patenting sector and output table

parameter = 'T'

x = list(countries_names.values())
y = getattr(p_baseline, parameter)[:,1]

fig, ax = plt.subplots()

ax.bar(x,y)
    
# plt.xlabel('Country')
plt.ylabel(parameter,rotation=0)

plt.title(parameters_description[parameter]+' in patenting sector')

for save_format in save_formats:
    plt.savefig(save_path+parameter+'.'+save_format,format=save_format)

plt.show()


df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries],name='Countries'),
                                        columns = [parameter])
df[parameter] = p_baseline.delta[:,1]

caption = parameters_description[parameter]

df.style.to_latex(save_path+parameter+'_patent_sector.tex',
                  caption=caption,
                  **save_to_tex_options
                  )
df.to_csv(save_path+parameter+'_patent_sector.csv')


#%% Unilateral patent protections counterfactuals

recap_growth_rate = pd.DataFrame(columns = ['delta_change']+p_baseline.countries+['World'])

for c in p_baseline.countries+['World']:
# for c in ['World']:
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
        p = parameters(n=7,s=2)
        p.load_data(country_path+run+'/')
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
            recap.loc[run, 'growth'] = sol_c.g
            recap.loc[run, 'world_negishi'] = sol_c.cons_eq_negishi_welfare_change
            recap.loc[run, 'world_equal'] = sol_c.cons_eq_pop_average_welfare_change
            recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
            
            recap_growth_rate.loc[run,'delta_change'] = p.delta[0,1]/p_baseline.delta[0,1]
            recap_growth_rate.loc[run,c] = sol_c.g

    fig,ax = plt.subplots()
    # ax2 = ax.twinx()
    
    ax.set_ylabel('Welfare change')
    if c in p_baseline.countries:
        ax.set_xlabel(r'Proportional change of $\delta$')
    if c == 'World':
        ax.set_xlabel(r'Proportional change of $\delta$ of all countries')
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
    plt.xscale('log')
    # plt.yscale('log')
    for save_format in save_formats:
        plt.savefig(save_path+c+'_unilateral_patent_protection_counterfactual.'+save_format,format=save_format)
    plt.show()
    
    if c in p_baseline.countries:
        caption = 'Consumption equivalent welfares in the unilateral patent protection counterfactual of '+countries_names[c]
    if c == 'World':
        caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
    
    recap.style.to_latex(save_path+c+'_unilateral_patent_protection_counterfactual.tex',
                      caption=caption,
                      **save_to_tex_options
                      )
    recap.to_csv(save_path+c+'_unilateral_patent_protection_counterfactual.csv')

#%%

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
    
    caption = 'Consumption equivalent welfares in the patent protection counterfactual change of all countries'
    
    
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

m_nash = m_baseline.copy()
m_nash.compute_moments(sol_nash,p_nash)
m_nash.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Consumption equivalent welfare change','Growth rate']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_nash.delta[i,1]
    df.loc[countries_names[c],'Consumption equivalent welfare change'] = sol_nash.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Consumption equivalent welfare change'] = sol_nash.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Consumption equivalent welfare change'] = sol_nash.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Growth rate'] = sol_nash.g

df.loc['World aggregate according to population weights',
       'Growth rate'] = sol_nash.g

caption = 'Nash equilibrium'

df.style.to_latex(save_path+'Nash_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Nash_table.csv')

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

m_coop_equal = m_baseline.copy()
m_coop_equal.compute_moments(sol_coop_equal,p_coop_equal)
m_coop_equal.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Consumption equivalent welfare change','Growth rate']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_equal.delta[i,1]
    df.loc[countries_names[c],'Consumption equivalent welfare change'] = sol_coop_equal.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Consumption equivalent welfare change'] = sol_coop_equal.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Consumption equivalent welfare change'] = sol_coop_equal.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Growth rate'] = sol_coop_equal.g

df.loc['World aggregate according to population weights',
       'Growth rate'] = sol_coop_equal.g

caption = 'Cooperative equilibrium with population weights'

df.style.to_latex(save_path+'Coop_population_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Coop_population_weights_table.csv')

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

m_coop_negishi = m_baseline.copy()
m_coop_negishi.compute_moments(sol_coop_negishi,p_coop_negishi)
m_coop_negishi.compute_moments_deviations()

df = pd.DataFrame(index = pd.Index([countries_names[c] for c in p_baseline.countries]\
                                   +['World aggregate according to Negishi weights',
                                     'World aggregate according to population weights'],
                                   name = 'Countries'),
                  columns = [r'$\delta$','Consumption equivalent welfare change','Growth rate']
                  )
    
for i,c in enumerate(p_baseline.countries):
    df.loc[countries_names[c],r'$\delta$'] = p_coop_negishi.delta[i,1]
    df.loc[countries_names[c],'Consumption equivalent welfare change'] = sol_coop_negishi.cons_eq_welfare[i]

df.loc['World aggregate according to Negishi weights',
       'Consumption equivalent welfare change'] = sol_coop_negishi.cons_eq_negishi_welfare_change

df.loc['World aggregate according to population weights',
       'Consumption equivalent welfare change'] = sol_coop_negishi.cons_eq_pop_average_welfare_change

df.loc['World aggregate according to Negishi weights',
       'Growth rate'] = sol_coop_negishi.g

df.loc['World aggregate according to population weights',
       'Growth rate'] = sol_coop_negishi.g

caption = 'Cooperative equilibrium with Negishi weights'

df.style.to_latex(save_path+'Coop_negishi_weights_table.tex',
                  caption=caption,
                  **save_to_tex_options
                  )

df.to_csv(save_path+'Coop_negishi_weights_table.csv')

write_calibration_results(save_path+'Coop_negishi_weights',p_coop_negishi,m_coop_negishi,sol_coop_negishi,commentary = '')



