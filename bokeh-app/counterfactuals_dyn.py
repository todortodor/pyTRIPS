#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:27:13 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, make_counterfactual
from data_funcs import make_counterfactual_recap

recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'

baseline_dics = [
    {'baseline':'618','variation': 'baseline'},
    {'baseline':'618','variation': '15.0'},
    # {'baseline':'611','variation': 'baseline'},
    # {'baseline':'611','variation': '1.0'},
    # {'baseline':'611','variation': '1.1'},
    # {'baseline':'611','variation': '1.2'},
    # {'baseline':'611','variation': '1.3'},
    # {'baseline':'611','variation': '1.4'},
    # {'baseline':'611','variation': '1.5'},
    # {'baseline':'611','variation': '1.6'},
    # {'baseline':'611','variation': '1.7'},
    # {'baseline':'611','variation': '1.8'},
    # {'baseline':'611','variation': '1.9'},
    # {'baseline':'611','variation': '1.10'},
    # {'baseline':'611','variation': '1.11'},
    # {'baseline':'611','variation': '1.12'},
    # {'baseline':'611','variation': '1.13'},
    # {'baseline':'611','variation': '1.14'},
    # {'baseline':'611','variation': '1.15'},
    # {'baseline':'611','variation': '1.16'},
    # {'baseline':'611','variation': '1.17'},
    # {'baseline':'611','variation': '1.18'},
    # {'baseline':'611','variation': '1.19'},
    # {'baseline':'611','variation': '1.20'},
    # {'baseline':'611','variation': '1.21'},
    # {'baseline':'611','variation': '1.22'},
    # {'baseline':'611','variation': '1.23'},
    # {'baseline':'611','variation': '1.24'},
    # {'baseline':'611','variation': '1.25'},
    # {'baseline':'611','variation': '1.26'},
    # {'baseline':'611','variation': '1.27'},
    # {'baseline':'611','variation': '1.28'},
    # {'baseline':'611','variation': '1.29'},
    # {'baseline':'611','variation': '1.30'},
    # {'baseline':'611','variation': '1.31'},
    # {'baseline':'611','variation': '1.32'},
    # {'baseline':'611','variation': '1.33'},
    # {'baseline':'611','variation': '1.34'},
    # {'baseline':'611','variation': '1.35'},
    # {'baseline':'611','variation': '1.36'},
    # {'baseline':'611','variation': '1.37'},
    # {'baseline':'611','variation': '1.38'},
    # {'baseline':'611','variation': '1.39'},
    # {'baseline':'611','variation': '1.40'},
    ]

for baseline_dic in baseline_dics:
    if baseline_dic['variation'] == 'baseline':
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    print(baseline_path)
    
    assert os.path.exists(baseline_path), 'run doesnt exist'
    
    p_baseline = parameters()
    p_baseline.load_run(baseline_path)
    if baseline_dic['variation'] == 'baseline':
        local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline_dic['baseline']+'/'
    else:
        local_path = \
            f'counterfactual_results/unilateral_patent_protection/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'

    try:
        os.mkdir(local_path)
    except:
        pass
    
    if baseline_dic['variation'] == 'baseline':
        recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'
    else:
        recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
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
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)
    
    lb_delta=0.01
    ub_delta=12
    
    # delta_factor_array = np.logspace(-1,1,31)
    
    for c in p_baseline.countries:
        make_counterfactual(p_baseline,c,local_path,
                            sol_baseline=sol_baseline,
                            # delta_factor_array=delta_factor_array,
                            dynamics=True)
        make_counterfactual_recap(p_baseline, sol_baseline, c,
                                      local_path,recap_path,
                                      dynamics=True,Nt=25,t_inf=500)
    
    make_counterfactual(p_baseline,'World',local_path,
                        # delta_factor_array=delta_factor_array,
                        sol_baseline=sol_baseline,dynamics=True)
    make_counterfactual_recap(p_baseline, sol_baseline, 'World',
                                  local_path,recap_path,
                                  dynamics=True,Nt=25,t_inf=500)
    
    # delta_factor_array = np.linspace(0,1,31)
    make_counterfactual(p_baseline,'Harmonizing',local_path,
                        # delta_factor_array=delta_factor_array,
                        sol_baseline=sol_baseline,dynamics=True)
    make_counterfactual_recap(p_baseline, sol_baseline, 'Harmonizing',
                                  local_path,recap_path,
                                  dynamics=True,Nt=25,t_inf=500)
    
    make_counterfactual(p_baseline,'Uniform_delta',local_path,
                        sol_baseline=sol_baseline,dynamics=True)
    make_counterfactual_recap(p_baseline, sol_baseline, 'Uniform_delta',
                                  local_path,recap_path,
                                  dynamics=True,Nt=25,t_inf=500)
    
#%%
for country in p_baseline.countries:
    recap_dyn = pd.read_csv(recap_path+'dyn_'+country+'.csv')
    recap = pd.read_csv(recap_path+country+'.csv')
    
    fig,ax = plt.subplots(2,1,figsize=(15,10),layout = "constrained")
    
    for i,c in enumerate(p_baseline.countries):
        ax[1].plot(recap['delt'],recap[c],color=sns.color_palette()[i],label = c)
        ax[0].plot(recap_dyn['delt'],recap_dyn[c],color=sns.color_palette()[i],label = c)
    
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].legend(title='With dynamics')
    ax[1].legend(title='Steady state only')
    plt.suptitle('Patent protection counterfactual for '+country)
    
    # plt.savefig('/Users/slepot/Dropbox/TRIPS/simon_version/code/misc/dynamics_counterfactuals/'+country)
    
    plt.show()
    
    
          