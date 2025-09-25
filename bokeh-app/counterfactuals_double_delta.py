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
from solver_funcs import fixed_point_solver_double_diff_double_delta, make_counterfactual_double_delta
from data_funcs import make_counterfactual_recap_double_delta
from concurrent.futures import ProcessPoolExecutor
import time

recaps_path = 'counterfactual_recaps/double_delta/'

baseline_dics = [
    # {'baseline':'1312','variation': 'baseline'},
    # {'baseline':'1312','variation': '1.0'},
    # {'baseline':'1312','variation': '1.08'},
    # {'baseline':'1312','variation': '1.09'},
    # {'baseline':'1312','variation': '2.01'},
    {'baseline':'1312','variation': '2.02'},
    # {'baseline':'1312','variation': '2.03'},
    # {'baseline':'1312','variation': '2.04'},
    # {'baseline':'1312','variation': '2.05'},
    # {'baseline':'1312','variation': '2.06'},
    # {'baseline':'1312','variation': '2.07'},
    # {'baseline':'1312','variation': '2.08'},
    # {'baseline':'1312','variation': '2.09'},
    # {'baseline':'1312','variation': '3.01'},
    # {'baseline':'1312','variation': '3.02'},
    # {'baseline':'1312','variation': '3.03'},
    # {'baseline':'1312','variation': '3.04'},
    # {'baseline':'1312','variation': '3.05'},
    # {'baseline':'1312','variation': '3.06'},
    # {'baseline':'1312','variation': '4.0'},
    ]

for bas in baseline_dics:
    try:
        os.mkdir(f"counterfactual_recaps/double_delta/baseline_1312_{bas['variation']}")
    except:
        pass
#%%

def process_country(args):
    p, c, local_path, sol_baseline, recap_path = args
    print(c,local_path)
    make_counterfactual_double_delta(p, c, local_path, dynamics=False)
    make_counterfactual_recap_double_delta(p, sol_baseline, c, local_path, recap_path, with_entry_costs=True)
    return 'done'

parallel = False

if __name__ == '__main__':
    for baseline_dic in baseline_dics:
        for delta_to_change in ['dom','int','both']:
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
                local_path = 'counterfactual_results/double_delta/baseline_'+baseline_dic['baseline']+'/'
            else:
                local_path = \
                    f'counterfactual_results/double_delta/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'
            
            try:
                os.mkdir(local_path)
            except:
                pass
            
            if baseline_dic['variation'] == 'baseline':
                local_path = 'counterfactual_results/double_delta/baseline_'+baseline_dic['baseline']+'/'+delta_to_change+'/'
            else:
                local_path = \
                    f'counterfactual_results/double_delta/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'+delta_to_change+'/'
            
            try:
                os.mkdir(local_path)
            except:
                pass
            
            if baseline_dic['variation'] == 'baseline':
                recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'+delta_to_change+'/'
            else:
                recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'+delta_to_change+'/'
            
            sol, sol_baseline = fixed_point_solver_double_diff_double_delta(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-10,
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
            
            # print('working')
            if parallel:
                args_list = [(p_baseline, c, local_path, sol_baseline, recap_path) for c in p_baseline.countries+['World']]
                with ProcessPoolExecutor(max_workers=12) as executor:
                    results = list(executor.map(process_country, args_list))
            
            else:
                for c in p_baseline.countries:
                    make_counterfactual_double_delta(p_baseline,c,local_path,dynamics=False,
                                                     delta_to_change=delta_to_change,#can be 'dom','int',or 'both'
                                                     )
                    make_counterfactual_recap_double_delta(p_baseline, sol_baseline, c,
                                             local_path,recap_path)
                    
                
                
                