#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:38:05 2022

@author: simonl
"""

import numpy as np
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, make_counterfactual
from data_funcs import make_counterfactual_recap

recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'

baseline_dics = [
    {'baseline':'1010','variation': 'baseline'},
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
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)
    
    for c in p_baseline.countries:
        make_counterfactual(p_baseline,c,local_path,dynamics=False)
        make_counterfactual_recap(p_baseline, sol_baseline, c,
                                      local_path,recap_path)
    
    make_counterfactual(p_baseline,'World',local_path,dynamics=False)
    make_counterfactual_recap(p_baseline, sol_baseline, 'World',
                                  local_path,recap_path)
    
    make_counterfactual(p_baseline,'Harmonizing',local_path,dynamics=False)
    make_counterfactual_recap(p_baseline, sol_baseline, 'Harmonizing',
                                  local_path,recap_path)
    
    make_counterfactual(p_baseline,'Uniform_delta',local_path,dynamics=False)
    make_counterfactual_recap(p_baseline, sol_baseline, 'Uniform_delta',
                                  local_path,recap_path)
