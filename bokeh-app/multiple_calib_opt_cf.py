#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:45:11 2023

@author: slepot
"""

from scipy import optimize
import time
from classes import moments, parameters,  var, history
from solver_funcs import calibration_func, fixed_point_solver, compute_deriv_welfare_to_patent_protec_US
from solver_funcs import find_nash_eq, find_coop_eq, make_counterfactual
from data_funcs import write_calibration_results, make_counterfactual_recap
import os
import numpy as np
import pandas as pd

# runs_params = [
#     {"number":0,"TO_target":0.01},
#     {"number":1,"TO_target":0.0105},
#     {"number":2,"TO_target":0.011},
#     {"number":3,"TO_target":0.0115},
#     {"number":4,"TO_target":0.012},
#     {"number":5,"TO_target":0.0125},
#     {"number":6,"TO_target":0.013},
#     {"number":7,"TO_target":0.0135},
#     {"number":8,"TO_target":0.014},
#     {"number":9,"TO_target":0.0145},
#     {"number":10,"TO_target":0.015},
#     {"number":11,"TO_target":0.0155},
#     {"number":12,"TO_target":0.016},
#     {"number":13,"TO_target":0.0165},
#     {"number":14,"TO_target":0.017},
#     {"number":15,"TO_target":0.0175},
#     {"number":16,"TO_target":0.018},
#     {"number":17,"TO_target":0.0185},
#     {"number":18,"TO_target":0.019},
#     {"number":19,"TO_target":0.0195},
#     {"number":20,"TO_target":0.02},
#     {"number":21,"TO_target":0.0205},
#     {"number":22,"TO_target":0.021},
#     {"number":23,"TO_target":0.0215},
#     {"number":24,"TO_target":0.022},
#     {"number":25,"TO_target":0.0225},
#     {"number":26,"TO_target":0.023},
#     {"number":27,"TO_target":0.0235},
#     {"number":28,"TO_target":0.024},
#     {"number":29,"TO_target":0.0245},
#     {"number":30,"TO_target":0.025},
#     {"number":31,"TO_target":0.0255},
#     {"number":32,"TO_target":0.026},
#     {"number":33,"TO_target":0.0265},
#     {"number":34,"TO_target":0.027},
#     {"number":35,"TO_target":0.0275},
#     {"number":36,"TO_target":0.028},
#     {"number":37,"TO_target":0.0285},
#     {"number":38,"TO_target":0.029},
#     {"number":39,"TO_target":0.0295},
#     {"number":40,"TO_target":0.03}
#     ]
runs_params = [
{'number': 0, 'year': 1990},
{'number': 1, 'year': 1991},
{'number': 2, 'year': 1992},
{'number': 3, 'year': 1993},
{'number': 4, 'year': 1994},
{'number': 5, 'year': 1995},
{'number': 6, 'year': 1996},
{'number': 7, 'year': 1997},
{'number': 8, 'year': 1998},
{'number': 9, 'year': 1999},
{'number': 10, 'year': 2000},
{'number': 11, 'year': 2001},
{'number': 12, 'year': 2002},
{'number': 13, 'year': 2003},
{'number': 14, 'year': 2004},
{'number': 15, 'year': 2005},
{'number': 16, 'year': 2006},
{'number': 17, 'year': 2007},
{'number': 18, 'year': 2008},
{'number': 19, 'year': 2009},
{'number': 20, 'year': 2010},
{'number': 21, 'year': 2011},
{'number': 22, 'year': 2012},
{'number': 23, 'year': 2013},
{'number': 24, 'year': 2014},
{'number': 25, 'year': 2015},
{'number': 26, 'year': 2016},
{'number': 27, 'year': 2017},
{'number': 28, 'year': 2018}
]

# for drop_SRDUS in [False,True]:
#     for patenting_cost_moment in ['UUPCOST','PCOSTNOAGG']:
#         for loss_func in ['log','ratio']:
#             runs_params.append({'number':i,
#                 'drop_SRDUS':drop_SRDUS,
#              'patenting_cost_moment':patenting_cost_moment,
#              'loss_func':loss_func,
#              })
#             i += 1
            
# runs_params = [
#  #    {'number': 0,
#  #  'drop_SRDUS': False,
#  #  'patenting_cost_moment': 'UUPCOST',
#  #  'loss_func': 'log'},
#  # {'number': 1,
#  #  'drop_SRDUS': False,
#  #  'patenting_cost_moment': 'UUPCOST',
#  #  'loss_func': 'ratio'},
#  # {'number': 2,
#  #  'drop_SRDUS': False,
#  #  'patenting_cost_moment': 'PCOSTNOAGG',
#  #  'loss_func': 'log'},
#  # {'number': 3,
#  #  'drop_SRDUS': False,
#  #  'patenting_cost_moment': 'PCOSTNOAGG',
#  #  'loss_func': 'ratio'},
#  # {'number': 4,
#  #  'drop_SRDUS': True,
#  #  'patenting_cost_moment': 'UUPCOST',
#  #  'loss_func': 'log'},
#  # {'number': 5,
#  #  'drop_SRDUS': True,
#  #  'patenting_cost_moment': 'UUPCOST',
#  #  'loss_func': 'ratio'},
#  # {'number': 6,
#  #  'drop_SRDUS': True,
#  #  'patenting_cost_moment': 'PCOSTNOAGG',
#  #  'loss_func': 'log'},
#  # {'number': 7,
#  #  'drop_SRDUS': True,
#  #  'patenting_cost_moment': 'PCOSTNOAGG',
#  #  'loss_func': 'ratio'},
#  {'number': 8,
#   'drop_RD': True,
#   'patenting_cost_moment': 'UUPCOST',
#   'loss_func': 'log'},
#  {'number': 9,
#   'drop_RD': True,
#   'patenting_cost_moment': 'UUPCOST',
#   'loss_func': 'ratio'},
#  {'number': 10,
#   'drop_RD': True,
#   'patenting_cost_moment': 'PCOSTNOAGG',
#   'loss_func': 'log'},
#  {'number': 11,
#   'drop_RD': True,
#   'patenting_cost_moment': 'PCOSTNOAGG',
#   'loss_func': 'ratio'},
#  ]

write = True

baseline_number = '620'

for variation_number in [1]:
    
    for run_params in runs_params:
        print(run_params)
        baseline_dic = {'baseline':baseline_number,
                        'variation':str(variation_number)+'.'+str(run_params['number'])}
        # baseline_dic = {'baseline':baseline_number,
        #                 'variation':'1.'+str(run_params['number'])}
        year = run_params['year']
        p = parameters()
        p.load_run('calibration_results_matched_economy/'+baseline_number+'/')
        # p.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_number}.{run_params["number"]-1}/')
        # p.load_data('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'+str(variation_number)+'.0/')
        # p.load_data(f'data_smooth_3_years/data_7_countries_{year}/',keep_already_calib_params=True)
        p.load_data(f'data/data_7_countries_{year}/',keep_already_calib_params=True)
        p.calib_parameters = ['eta','T','delta']
        
        sol = var.var_from_vector(p.guess,p,context='calibration')
        sol.scale_P(p)
        sol.compute_non_solver_quantities(p) 

        m = moments()
        # m.load_data()
        m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
        m.compute_moments(sol,p)
        number_of_int_patents_model_baseline = m.inter_TP.copy()
        number_of_int_patents_data_baseline = m.inter_TP_data.copy()
        # m.load_run(f'calibration_results_matched_economy/baseline_{baseline_number}_variations/{variation_number}.{run_params["number"]-1}/')
        # m.load_data(f'data_smooth_3_years/data_7_countries_{year}/')
        m.load_data(f'data/data_7_countries_{year}/')
        # m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'+str(variation_number)+'.0/')
        # if 'theta' not in p.calib_parameters:
        #     p.calib_parameters.append('theta')
        #     if 'TE' not in m.list_of_moments:
        #         m.list_of_moments.append('TE')
        # if 'theta' in p.calib_parameters:
        # p.update_sigma_with_SRDUS_target(m)
        # if 'TE' not in m.list_of_moments:
        #     m.list_of_moments.append('TE')
        m.inter_TP_target = number_of_int_patents_model_baseline*m.inter_TP_data/number_of_int_patents_data_baseline
        m.weights_dict['inter_TP'] = 3
        m.list_of_moments = ['OUT',
          'RD',
          'RP',
          'SRGDP',
          'inter_TP',
          'SINNOVPATUS',
          'SPFLOW',
          'UUPCOST',
          'SRDUS',
          'SINNOVPATEU',
          'DOMPATINUS',
          'DOMPATINEU']
        
        m.drop_CHN_IND_BRA_ROW_from_RD = True
        
        # m.TO_target = np.float64(run_params['TO_target'])
        
        # if run_params['drop_SRDUS']:
        #     if 'SRDUS' in m.list_of_moments:
        #         m.list_of_moments.remove('SRDUS')
        # if run_params['drop_RD']:
        #     if 'RD' in m.list_of_moments:
        #         m.list_of_moments.remove('RD')
        # if run_params['patenting_cost_moment'] == 'PCOSTNOAGG':
        #     if 'UUPCOST' in m.list_of_moments:
        #         m.list_of_moments.remove('UUPCOST')
        #     if 'PCOSTNOAGG' not in m.list_of_moments:
        #         m.list_of_moments.append('PCOSTNOAGG')

        # m.loss = run_params['loss_func']
        # m.dim_weight = run_params['dim_weight']
        
        hist = history(*tuple(m.list_of_moments+['objective']))
        bounds = p.make_parameters_bounds()
        start_time = time.perf_counter()
        cond = True
        iterations = 0
        max_iter = 5
        
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        max_nfev=1e8,
                                        xtol=1e-10, 
                                        verbose = 2)
            else:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        max_nfev=1e8,
                                        xtol=1e-16, 
                                        verbose = 2)
            cond = iterations < max_iter
            iterations += 1
            p.update_parameters(test_ls.x)
        
        p_sol = p.copy()
        p_sol.update_parameters(test_ls.x)
        
        sol, sol_c = fixed_point_solver(p_sol,x0=p_sol.guess,
                                        context = 'calibration',
                                cobweb_anim=False,tol =1e-15,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=True,
                                plot_cobweb=True,
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        p_sol.guess = sol.x 
        sol_c.scale_P(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.tau = sol_c.tau
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        m.plot_moments(m.list_of_moments)
        
        ##%% writing results as excel and locally
        commentary = ''
        dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
        local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
        run_number = baseline_dic['variation']
        path = dropbox_path+'baseline_'+baseline_number+'_variations/'
            
        try:
            os.mkdir(path)
        except:
            pass
        
        write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
        
        try:
            os.mkdir(local_path)
        except:
            pass
        p_sol.write_params(local_path+str(run_number)+'/')
        m.write_moments(local_path+str(run_number)+'/')
        
        #%%
        
        p_baseline = p_sol.copy()
        sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                        context = 'counterfactual',
                                cobweb_anim=False,tol =1e-15,
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
        
        ##%% Nash eq
        method = 'fixed_point'
        p_nash, sol_nash = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=12,method='fixed_point',
                          plot_convergence = True,solver_options=None,tol=1e-4)
        
        if write:
            if not os.path.exists('nash_eq_recaps/deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'method'] + p_baseline.countries)
                deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            method]+p_nash.delta[:,1].tolist(), 
                            index = deltas_df.columns).T
            deltas_df = pd.concat([deltas_df, run],ignore_index=True)
            deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            
            if not os.path.exists('nash_eq_recaps/cons_eq_welfares.csv'):
                cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'method'] + p_baseline.countries + ['Equal','Negishi'])
                cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
            cons_eq_welfares = pd.read_csv('nash_eq_recaps/cons_eq_welfares.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            method]+sol_nash.cons_eq_welfare.tolist()+[sol_nash.cons_eq_pop_average_welfare_change,
                                                                sol_nash.cons_eq_negishi_welfare_change], 
                            index = cons_eq_welfares.columns).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        
        ##%% Coop eq
        for aggregation_method in ['pop_weighted','negishi']:
            p_opti, sol_opti = find_coop_eq(p_baseline,aggregation_method,
                              lb_delta=0.01,ub_delta=12,dynamics=False,
                              solver_options=None,tol=1e-15,
                              static_eq_deltas = None,custom_weights=None)
            
            if write:
                if not os.path.exists('coop_eq_recaps/deltas.csv'):
                    deltas_df = pd.DataFrame(columns = ['baseline',
                                                    'variation',
                                                    'aggregation_method'] + p_baseline.countries)
                    deltas_df.to_csv('coop_eq_recaps/deltas.csv')
                deltas_df = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
                run = pd.DataFrame(data = [baseline_dic['baseline'],
                                baseline_dic['variation'],
                                aggregation_method]+p_opti.delta[...,1].tolist(), 
                                index = deltas_df.columns).T
                deltas_df = pd.concat([deltas_df, run],ignore_index=True)
                deltas_df.to_csv('coop_eq_recaps/deltas.csv')
                
                if not os.path.exists('coop_eq_recaps/cons_eq_welfares.csv'):
                    cons_eq_welfares = pd.DataFrame(columns = ['baseline',
                                                    'variation',
                                                    'aggregation_method'] + p_baseline.countries + ['Equal','Negishi'])
                    cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
                cons_eq_welfares = pd.read_csv('coop_eq_recaps/cons_eq_welfares.csv',index_col=0)
                run = pd.DataFrame(data = [baseline_dic['baseline'],
                                baseline_dic['variation'],
                                aggregation_method]+sol_opti.cons_eq_welfare.tolist()+[sol_opti.cons_eq_pop_average_welfare_change,
                                                                    sol_opti.cons_eq_negishi_welfare_change], 
                                index = cons_eq_welfares.columns).T
                cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
                cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
        
        ##%% counterfactuals 
        if baseline_dic['variation'] == 'baseline':
            local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline_dic['baseline']+'/'
        else:
            local_path = \
                f'counterfactual_results/unilateral_patent_protection/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'
        
        try:
            os.mkdir(local_path)
        except:
            pass
        
        recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'
        
        if baseline_dic['variation'] == 'baseline':
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'
        else:
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'

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