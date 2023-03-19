#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:34:25 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from solver_funcs import find_nash_eq
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_nash_eq, dyn_fixed_point_solver
# from random import random
from tqdm import tqdm
import seaborn as sns

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

baseline_dics = [
    {'baseline':'501','variation': '1.0'}
    ]

# baseline_dics = [
#     {'baseline':'501','variation': None},
#     {'baseline':'501','variation': '1.0'}
#     ]

nash_deltas = pd.read_csv('nash_eq_recaps/dyn_deltas.csv',index_col=0).drop_duplicates(['baseline','variation'],keep='last')
nash_deltas['variation'] = nash_deltas['variation'].astype('str')
nash_deltas['baseline'] = nash_deltas['baseline'].astype('str')
coop_negishi_deltas = (pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
                       .drop_duplicates(['baseline','variation','aggregation_method'],keep='last'))
coop_negishi_deltas = coop_negishi_deltas.loc[coop_negishi_deltas.aggregation_method == 'negishi']
coop_negishi_deltas['variation'] = coop_negishi_deltas['variation'].astype('str')
coop_negishi_deltas['baseline'] = coop_negishi_deltas['baseline'].astype('str')
coop_equal_deltas = (pd.read_csv('coop_eq_recaps/dyn_deltas.csv',index_col=0)
                       .drop_duplicates(['baseline','variation','aggregation_method'],keep='last'))
coop_equal_deltas = coop_equal_deltas.loc[coop_equal_deltas.aggregation_method == 'pop_weighted']
coop_equal_deltas['variation'] = coop_equal_deltas['variation'].astype('str')
coop_equal_deltas['baseline'] = coop_equal_deltas['baseline'].astype('str')

for baseline_dic in baseline_dics:
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
    
    for equilibrium in ['nash_eq','coop_negishi_eq','coop_equal_eq']:
    
        if baseline_dic['variation'] == 'baseline':
            local_path = \
                f'counterfactual_results/around_dyn_{equilibrium}/baseline_{baseline_dic["baseline"]}/'
        else:
            local_path = \
                f'counterfactual_results/around_dyn_{equilibrium}/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'
    
        try:
            os.mkdir(local_path)
        except:
            pass
        
        if equilibrium == 'nash_eq':
            deltas_of_equilibrium = nash_deltas.loc[(nash_deltas.baseline == baseline_dic["baseline"])
                                                    &(nash_deltas.variation == baseline_dic["variation"])
                                                    ][p_baseline.countries].values
        if equilibrium == 'coop_negishi_eq':
            deltas_of_equilibrium = coop_negishi_deltas.loc[(coop_negishi_deltas.baseline == baseline_dic["baseline"])
                                                            &(coop_negishi_deltas.variation == baseline_dic["variation"])
                                                            ][p_baseline.countries].values
        if equilibrium == 'coop_equal_eq':
            deltas_of_equilibrium = coop_equal_deltas.loc[(coop_equal_deltas.baseline == baseline_dic["baseline"])
                                                            &(coop_equal_deltas.variation == baseline_dic["variation"])
                                                            ][p_baseline.countries].values
    
        lb_delta=0.01
        ub_delta=12
        
        # p_baseline.delta[:,1] = ub_delta
        # p_baseline.delta[0,1] = lb_delta
        
        solver_options = dict(cobweb_anim=False,tol =1e-14,
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
                                damping_post_acceleration=5)
        
        sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                        context = 'counterfactual',
                                **solver_options
                                )
        
        sol_baseline.scale_P(p_baseline)
        sol_baseline.compute_price_indices(p_baseline)
        sol_baseline.compute_non_solver_quantities(p_baseline)    
        
        deltas = np.logspace(np.log(lb_delta)/np.log(10),np.log(ub_delta)/np.log(10),31)
        
        for c in p_baseline.countries:
            country_path = local_path+c+'/'
            try:
                os.mkdir(country_path)
            except:
                pass
            print(country_path)
            c_index = p_baseline.countries.index(c)
            for i,delt in enumerate(deltas):
                print(delt)
                p = p_baseline.copy()
                p.delta[:,1] = deltas_of_equilibrium
                p.delta[c_index,1] = delt
                sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline, Nt=23,
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
                dyn_sol_c.compute_non_solver_quantities(p)
                if sol.status == 'successful':
                    test_initial_guess = dyn_sol_c.vector_from_var()
                    p.dyn_guess = dyn_sol_c.vector_from_var()
                    p.guess = dyn_sol_c.sol_fin.vector_from_var()
                else:
                    p.dyn_guess = None
                    p.guess = None

                p.write_params(country_path+'/'+str(i)+'/') 
                
