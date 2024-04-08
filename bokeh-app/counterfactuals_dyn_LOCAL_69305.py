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
from concurrent.futures import ProcessPoolExecutor
import time

recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'

baseline_dics = [
    {'baseline':'1210','variation': 'baseline'},
    {'baseline':'1210','variation': '2.0'},
    {'baseline':'1210','variation': '10.2'},
    {'baseline':'1210','variation': '10.3'},
    {'baseline':'1210','variation': '10.4'},
    {'baseline':'1210','variation': '10.5'},
    {'baseline':'1210','variation': '99.0'},
    {'baseline':'1210','variation': '99.1'},
    {'baseline':'1210','variation': '99.2'},
    {'baseline':'1210','variation': '99.3'},
    {'baseline':'1210','variation': '99.4'},
    {'baseline':'1210','variation': '99.5'},
    {'baseline':'1210','variation': '99.6'},
    {'baseline':'1210','variation': '99.7'},
    {'baseline':'1210','variation': '99.8'},
    {'baseline':'1210','variation': '99.9'},
    {'baseline':'1210','variation': '99.10'},
    {'baseline':'1210','variation': '99.11'},
    {'baseline':'1210','variation': '99.12'},
    {'baseline':'1210','variation': '99.13'},
    {'baseline':'1210','variation': '99.14'},
    {'baseline':'1210','variation': '99.15'},
    ]


def process_country(args):
    p, c, local_path, sol_baseline, recap_path, dynamics, Nt, t_inf = args
    print(c,local_path)
    make_counterfactual(p, c, local_path, sol_baseline=sol_baseline, dynamics=dynamics)
    make_counterfactual_recap(p, sol_baseline, c, local_path, recap_path, dynamics=dynamics, Nt=Nt, t_inf=t_inf)
    return 'done'

if __name__ == '__main__':
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
        
        # lb_delta=0.01
        # ub_delta=12
        
        # delta_factor_array = np.logspace(-1,1,31)
        
        # sequential processes
        for c in p_baseline.countries:
             make_counterfactual(p_baseline,c,local_path,
                                 sol_baseline=sol_baseline,
                                 # delta_factor_array=delta_factor_array,
                                 dynamics=True)
             make_counterfactual_recap(p_baseline, sol_baseline, c,
                                           local_path,recap_path,
                                           dynamics=True,Nt=25,t_inf=500)
    
        #args_list = [(p_baseline.copy(), c, local_path, sol_baseline.copy(), recap_path, True, 25, 500
        #              # ) for c in p_baseline.countries]
        #               ) for c in p_baseline.countries+['World']]
        
        ## Create a ProcessPoolExecutor
        #with ProcessPoolExecutor(max_workers=15) as executor:
        #    # returns = executor.map(lambda args: process_country(*args), args_list)
        #    results = list(executor.map(process_country, args_list))
        
        # make_counterfactual(p_baseline,'World',local_path,
        #                     # delta_factor_array=delta_factor_array,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'World',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # # delta_factor_array = np.linspace(0,1,31)
        # make_counterfactual(p_baseline,'Harmonizing',local_path,
        #                     # delta_factor_array=delta_factor_array,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'Harmonizing',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'Uniform_delta',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'Uniform_delta',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'Upper_harmonizing',local_path,
        #                     # delta_factor_array=delta_factor_array,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'Upper_harmonizing',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'Upper_uniform_delta',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'Upper_uniform_delta',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # p_pre = parameters()
        # p_pre.load_run(f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/9.2/')
        
        # alt_delta = p_pre.delta[...,1]
        # for country_idx in [0,1,2,6,7,10]:
        #     alt_delta[country_idx] = p_baseline.delta[country_idx,1]
        
        # make_counterfactual(p_baseline,'trade_cost_eq_trips_all_countries_all_sectors',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True,alt_delta=alt_delta)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'trade_cost_eq_trips_all_countries_all_sectors',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'trade_cost_eq_trips_all_countries_pat_sectors',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True,alt_delta=alt_delta)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'trade_cost_eq_trips_all_countries_pat_sectors',
        #                                local_path,recap_path,
        #                                dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'trade_cost_all_countries_all_sectors',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'trade_cost_all_countries_all_sectors',
                                      # local_path,recap_path,
                                      # dynamics=True,Nt=25,t_inf=500)
        
        # make_counterfactual(p_baseline,'trade_cost_all_countries_pat_sectors',local_path,
        #                     sol_baseline=sol_baseline,dynamics=True)
        # make_counterfactual_recap(p_baseline, sol_baseline, 'trade_cost_all_countries_pat_sectors',
        #                               local_path,recap_path,
        #                               dynamics=True,Nt=25,t_inf=500)
        
        # for c in ['CHN','IND','RUS']:
        #     make_counterfactual(p_baseline,c+'_trade_cost_eq_trips_exp_imp_pat_sect',local_path,
        #                         sol_baseline=sol_baseline,dynamics=True,alt_delta=alt_delta)
        #     make_counterfactual_recap(p_baseline, sol_baseline,c+'_trade_cost_eq_trips_exp_imp_pat_sect',
        #                                   local_path,recap_path,
        #                                   dynamics=True,Nt=25,t_inf=500)
        
        # for c in ['CHN','IND','RUS']:
        #     make_counterfactual(p_baseline,c+'_tariff_eq_trips_exp_pat_sect',local_path,
        #                         sol_baseline=sol_baseline,dynamics=True,alt_delta=alt_delta)
        #     make_counterfactual_recap(p_baseline, sol_baseline,c+'_tariff_eq_trips_exp_pat_sect',
        #                                   local_path,recap_path,
        #                                   dynamics=True,Nt=25,t_inf=500)
    
