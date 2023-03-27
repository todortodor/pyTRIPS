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
from classes import parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver,make_counterfactual
from data_funcs import make_counterfactual_recap
import seaborn as sns

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

baseline_dics = [
    {'baseline':'501','variation': '3.0'}
    ]

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
        
        recaps_path = f'counterfactual_recaps/around_dyn_{equilibrium}/'
        if baseline_dic['variation'] is None:
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'
        else:
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'        
        
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
        sol_baseline.compute_non_solver_quantities(p_baseline)    
        
        delta_factor_array = np.logspace(np.log(lb_delta)/np.log(10),np.log(ub_delta)/np.log(10),31)
        
        for c in p_baseline.countries:
            make_counterfactual(p_baseline,c,local_path,
                                    delta_factor_array=delta_factor_array,dynamics=True,
                                    sol_baseline=sol_baseline,
                                    Nt=25,t_inf=500)
            make_counterfactual_recap(p_baseline, sol_baseline, c,
                                          local_path,recap_path,
                                          dynamics=True,Nt=25,t_inf=500)
