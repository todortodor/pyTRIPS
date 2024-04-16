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
from solver_funcs import fixed_point_solver, find_coop_eq_tariff
from tqdm import tqdm
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

baseline_dics = [
    {'baseline':'1210','variation': 'baseline'},
    #{'baseline':'1030','variation': '99.0'},
    #{'baseline':'1030','variation': '99.1'},
    #{'baseline':'1030','variation': '99.2'},
    #{'baseline':'1030','variation': '99.3'},
    #{'baseline':'1030','variation': '99.4'},
    # {'baseline':'1030','variation': '99.5'},
    # {'baseline':'1030','variation': '99.6'},
    # {'baseline':'1030','variation': '99.7'},
    # {'baseline':'1030','variation': '99.8'},
    # {'baseline':'1030','variation': '99.9'},
    # {'baseline':'1030','variation': '99.10'},
    # {'baseline':'1030','variation': '99.11'},
    # {'baseline':'1030','variation': '99.12'},
    # {'baseline':'1030','variation': '99.13'},
    # {'baseline':'1030','variation': '99.14'},
    # {'baseline':'1030','variation': '99.15'},
    ]

lb_tariff = 0
ub_tariff = 1
# ub_delta = 1

import time

if __name__ == '__main__':
    for baseline_dic in baseline_dics:    
        if baseline_dic['variation'] == 'baseline':
            baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
        else:
            baseline_path = \
                f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        
        assert os.path.exists(baseline_path), 'run doesnt exist'
        
        print(baseline_path)
        p_baseline = parameters()
        p_baseline.load_run(baseline_path)  
        
        # for aggregation_method in ['pop_weighted','negishi']:
        for aggregation_method in ['pop_weighted']:
            print(aggregation_method)
            
            start = time.perf_counter()
            
            p_opti, sol_opti = find_coop_eq_tariff(p_baseline,aggregation_method,
                             lb_tariff=lb_tariff,ub_tariff=ub_tariff,dynamics=True,
                             solver_options=None,tol=1e-8,
                             static_eq_tariff = None,custom_weights=None,
                             # custom_x0 = np.ones(p_baseline.N)*12,
                             custom_x0 = None,
                             max_workers=15)
            
            print(time.perf_counter() - start)
            
            write = True
            if write:
            #     if not os.path.exists('coop_eq_recaps/deltas.csv'):
            #         deltas_df = pd.DataFrame(columns = ['baseline',
            #                                         'variation',
            #                                         'aggregation_method'] + p_baseline.countries)
            #         deltas_df.to_csv('coop_eq_recaps/deltas.csv')
            #     deltas_df = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
            #     run = pd.DataFrame(data = [baseline_dic['baseline'],
            #                     baseline_dic['variation'],
            #                     aggregation_method]+p_opti.delta[...,1].tolist(), 
            #                     index = ['baseline',
            #                              'variation',
            #                              'aggregation_method'] + p_baseline.countries).T
            #     deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            #     deltas_df.to_csv('coop_eq_recaps/deltas.csv')
                
            #     if not os.path.exists('coop_eq_recaps/cons_eq_welfares.csv'):
            #         cons_eq_welfares = pd.DataFrame(columns = ['baseline',
            #                                         'variation',
            #                                         'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
            #         cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
            #     cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares.csv',index_col=0)
            #     run = pd.DataFrame(data = [baseline_dic['baseline'],
            #                     baseline_dic['variation'],
            #                     aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
            #                                                        sol_opti.cons_eq_negishi_welfare_change], 
            #                     index = ['baseline',
            #                              'variation',
            #                              'aggregation_method'] + p_baseline.countries + ['Equal','Negishi']).T
            #     cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            #     cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
            
                baseline = baseline_dic['baseline']
                
                try:
                    os.mkdir(f'opt_tariff_delta/{baseline}/')
                except:
                    pass
                
                if aggregation_method == 'pop_weighted':
                    try:
                        os.mkdir(f'opt_tariff_delta/{baseline}/scenario_6')
                    except:
                        pass
                    p_opti.write_params(f'opt_tariff_delta/{baseline}/scenario_6/')
                if aggregation_method == 'negishi':
                    try:
                        os.mkdir(f'opt_tariff_delta/{baseline}/scenario_10')
                    except:
                        pass
                    p_opti.write_params(f'opt_tariff_delta/{baseline}/scenario_10/')
            
# %%
# import numpy as np

# df = pd.DataFrame(index=p_baseline.countries)

# df['welfare'] = sol_opti.cons_eq_welfare
# df['trade_weighted_tariff'] = np.einsum('nis,nis->i',
#                                         p_opti.tariff,
#                                         sol_opti.X
#                                         )*100/np.einsum('nis->i',
#                                                     sol_opti.X
#                                                     )


# #%%
# import numpy as np

# p_baseline = parameters()
# p_baseline.load_run('opt_tariff_delta/1040/scenario_3/')

# welfs = []

# # exporter='CHN'
# # importer='EUR'

# # exporter_index = p_baseline.countries.index(exporter)
# # importer_index = p_baseline.countries.index(importer) 

# sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
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
#                         damping_post_acceleration=1
#                         # damping=10
#                           # apply_bound_psi_star=True
#                         )

# sol_baseline.scale_P(p_baseline)
# sol_baseline.compute_non_solver_quantities(p_baseline) 

# p = p_baseline.copy()

# x = np.linspace(0.01,12,101)
# l_sol = []

# for i,tariff in enumerate(x):
#     print(i)
#     # p.tariff[importer_index,exporter_index,:] = tariff
#     p.delta[-2,1] = tariff
#     sol, sol_c = fixed_point_solver(p,x0=p.guess,
#                             context = 'counterfactual',
#                             cobweb_anim=False,tol =1e-14,
#                             accelerate=False,
#                             accelerate_when_stable=True,
#                             cobweb_qty='l_R',
#                             plot_convergence=False,
#                             plot_cobweb=False,
#                             safe_convergence=0.001,
#                             disp_summary=False,
#                             damping = 10,
#                             max_count = 1000,
#                             accel_memory =50, 
#                             accel_type1=True, 
#                             accel_regularization=1e-10,
#                             accel_relaxation=0.5, 
#                             accel_safeguard_factor=1, 
#                             accel_max_weight_norm=1e6,
#                             damping_post_acceleration=1
#                             # damping=10
#                               # apply_bound_psi_star=True
#                             )

#     sol_c.scale_P(p)
#     sol_c.compute_non_solver_quantities(p) 
#     sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
#     sol_c.compute_world_welfare_changes(p,sol_baseline)
    
#     welfs.append(sol_c.cons_eq_welfare.tolist()+[sol_c.cons_eq_negishi_welfare_change,
#                                                   sol_c.cons_eq_pop_average_welfare_change])
    
#     p.guess = sol_c.vector_from_var()
#     l_sol.append(sol_c)

# welfs = np.array(welfs)

# #%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig,ax=plt.subplots()
# cycler = plt.cycler(linestyle=['-']*11+['--','--'],
#                     color=sns.color_palette()+sns.color_palette()[:3])
# # plt.plot(welfs[:,-2])
# ax.set_prop_cycle(cycler)
# # ax.set_xlabel(f'{importer}, {exporter} tariff')
# ax.set_ylabel('Welfare')
# plt.plot(x,welfs,lw=2)
# plt.legend(p.countries+['negishi','equal'])
# # plt.xscale('symlog',linthresh=1)
# plt.xscale('log')
# plt.show()

# #%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig,ax=plt.subplots()

# for i,c in enumerate(p.countries):
#     ax.plot(x,[so.price_indices[i]/sol_baseline.price_indices[i]
#                for so in l_sol])
# ax.set_xlabel(f'{importer}, {exporter} tariff')
# ax.set_ylabel('Price indices')
# plt.legend(p.countries)
# plt.show()

# #%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig,ax=plt.subplots()

# # for i,c in enumerate(p.countries):
# ax.plot(x,[so.X[importer_index,exporter_index,1]*x[j]/(1+x[j])#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='black')
# ax.set_xlabel(f'{importer}, {exporter} tariff')
# ax.set_ylabel('Tariff revenue')
# # plt.legend(p.countries)
# plt.show()

# #%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig,ax=plt.subplots()

# # for i,c in enumerate(p.countries):
# ax.plot(1+x,[so.X[importer_index,exporter_index,1]/(1+x[j])#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='black',label='Trade flow')
# ax.plot(1+x,[so.X[importer_index,exporter_index,1]*x[j]/(1+x[j])#/sol_baseline.price_indices[i]
#            for j,so in enumerate(l_sol)],color='b',label='Tariff revenue')
# ax.set_xlabel(f'{importer}, {exporter} tariff')
# # ax.set_ylabel('Trade flow')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
