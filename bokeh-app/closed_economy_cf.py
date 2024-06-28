#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:50:18 2023

@author: slepot
"""


from classes import moments, parameters, var
from solver_funcs import fixed_point_solver
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from bokeh.palettes import Category10, Dark2

Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

baseline_number = '1300'

try:
    os.mkdir('closed_economy')
except:
    pass
try:
    os.mkdir(f'closed_economy/baseline_{baseline_number}')
except:
    pass
save_path = f'closed_economy/baseline_{baseline_number}/'

p = parameters()
# p.load_run('calibration_results_matched_economy/baseline_803_variations/1.0/')
p.load_run(f'calibration_results_matched_economy/{baseline_number}/')
# df = pd.DataFrame(index = p.countries, columns = ['welfare','growth rate'])
deltas = np.logspace(-2,1,301)
# deltas = np.array([1])
df = pd.DataFrame(index=[i for i in range(deltas.shape[0])],columns = p.countries+['delta'])
df_l_r = pd.DataFrame(index=[i for i in range(deltas.shape[0])],columns = p.countries+['delta'])
df_g_1 = pd.DataFrame(index=[i for i in range(deltas.shape[0])],columns = p.countries+['delta'])

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
                        damping = 2,
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

for c in p.countries:
    print(c)
    p_one_country = p.make_one_country_parameters(c)
    for i,delta in tqdm(enumerate(deltas)):
        p_one_country.delta[0,1] = delta
        sol, sol_one_country = fixed_point_solver(p_one_country,x0=p_one_country.guess,
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
                                damping_post_acceleration=10
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        
        sol_one_country.scale_P(p_one_country)
        sol_one_country.compute_non_solver_quantities(p_one_country) 
        sol_one_country.compute_one_country_welfare_change(p_one_country,
                        sol_c.cons[p.countries.index(c)],sol_c.g)
        p_one_country.guess = sol.x
        df.loc[i,c] = sol_one_country.cons_eq_welfare[0]
        df.loc[i,'delta'] = delta
        df_l_r.loc[i,c] = sol_one_country.l_R[0,1]/p_one_country.labor[0]
        df_l_r.loc[i,'delta'] = delta
        df_g_1.loc[i,c] = sol_one_country.g_s[1]
        df_g_1.loc[i,'delta'] = delta
    
    print(df_l_r)
    print(df_g_1)

df_l_r.to_csv(save_path+'labor_research.csv')
df_g_1.to_csv(save_path+'growth_rate.csv')

#%%
df_closed = pd.DataFrame(index = p.countries, columns = ['welfare'])
for c in p.countries:
# for c in ['USA']:
    p_one_country = p.make_one_country_parameters(c)
    
    p_one_country = p.make_one_country_parameters(c)
    sol, sol_one_country = fixed_point_solver(p_one_country,x0=p_one_country.guess,
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
                            damping_post_acceleration=10
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    
    sol_one_country.scale_P(p_one_country)
    sol_one_country.compute_non_solver_quantities(p_one_country) 
    sol_one_country.compute_one_country_welfare_change(p_one_country,sol_c.cons[p.countries.index(c)],sol_c.g)
    # print(c,sol_one_country.cons_eq_welfare)
    df_closed.loc[c,'welfare'] = sol_one_country.cons_eq_welfare[0]
    df_closed['welfare'] = df_closed['welfare'].astype(float)

df_closed.to_csv(save_path+'closed.csv')

#%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pylab as pylab

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (14, 10),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
# sns.set()
# sns.set_context('talk')
# sns.set_style('whitegrid')

# fig,ax = plt.subplots()

# for i,c in enumerate(p.countries):
#     ax.plot(df['delta'],df[c],label=c,color=Category18[i])

# ax.scatter(p.delta[:,1],df_closed['welfare'],color = Category18[:p.N],marker='x',label = 'Calibrated deltas')
# opt_deltas = [df.loc[np.argmax(df[c]),'delta'] for c in p.countries]
# opt_welfares = [df[c].max() for c in p.countries]
# ax.scatter(opt_deltas,opt_welfares,color = Category18[:p.N],label = 'Optimal deltas')

# ax.set_xscale('log')
# ax.set_xlabel('Delta')
# ax.set_ylabel('Welfare')

# plt.legend(loc = 'center right')

# plt.show()

#%%
# df_temp = df_g_1.copy()
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pylab as pylab

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (14, 10),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
# sns.set()
# sns.set_context('talk')
# sns.set_style('whitegrid')

# fig,ax = plt.subplots()

# for c in p.countries:
#     ax.plot(df_temp['delta'],df_temp[c],label=c)

# # ax.scatter(p.delta[:,1],df_temp_closed['welfare'],color = sns.color_palette()[:p.N],marker='x',label = 'Calibrated deltas')
# # opt_deltas = [df_temp.loc[np.argmax(df_temp[c]),'delta'] for c in p.countries]
# # opt_welfares = [df_temp[c].max() for c in p.countries]
# # ax.scatter(opt_deltas,opt_welfares,color = sns.color_palette()[:p.N],label = 'Optimal deltas')

# ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.set_xlabel('Delta')
# # ax.set_ylabel('Welfare')

# plt.legend()

# plt.show()

#%%
opt_deltas = [df.loc[np.argmax(df[c]),'delta'] for c in p.countries]
opt_welfares = [df[c].max() for c in p.countries]

recap = pd.DataFrame(index = p.countries)
recap['Calibrated deltas'] = p.delta[:,1]
recap['Closed economy welfare change'] = df_closed['welfare']
recap['Optimal deltas'] = opt_deltas
recap['Optimal closed economy welfare change'] = opt_welfares


recap.to_csv(save_path+'recap_closed_economy.csv')
