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
from solver_funcs import find_nash_eq
from data_funcs import write_calibration_results
import os
import numpy as np
# from solver_funcs import find_nash_eq, minus_welfare_of_delta
import pandas as pd

# runs_params = [
#     {'number':0,
#       'KM_target':0.06,
#       'TO_target':0.05,
#       'kappa':0.5
#       },
#     {'number':1,
#       'KM_target':0.09277,
#       'TO_target':0.05,
#       'kappa':0.5
#       },
#     {'number':2,
#       'KM_target':0.1322,
#       'TO_target':0.05,
#       'kappa':0.5
#       },
#     {'number':3,
#       'KM_target':0.06,
#       'TO_target':0.036,
#       'kappa':0.5
#       },
#     {'number':4,
#       'KM_target':0.09277,
#       'TO_target':0.036,
#       'kappa':0.5
#       },
#     {'number':5,
#       'KM_target':0.1322,
#       'TO_target':0.036,
#       'kappa':0.5
#       },
#     {'number':6,
#       'KM_target':0.06,
#       'TO_target':0.0242 ,
#       'kappa':0.5
#       },
#     {'number':7,
#       'KM_target':0.09277,
#       'TO_target':0.0242 ,
#       'kappa':0.5
#       },
#     {'number':8,
#       'KM_target':0.1322,
#       'TO_target':0.0242 ,
#       'kappa':0.5
#       },
#     {'number':9,
#       'KM_target':0.06,
#       'TO_target':0.0242,
#       'kappa':0.5
#       },
#     {'number':10,
#       'KM_target':0.09277,
#       'TO_target':0.0242,
#       'kappa':0.5
#       },
#     {'number':11,
#       'KM_target':0.1322,
#       'TO_target':0.0242,
#       'kappa':0.5
#       },
    # {'number':9,
    #   'KM_target':0.06,
    #   'TO_target':0.05,
    #   'kappa':0.7474
    #   },
    # {'number':10,
    #   'KM_target':0.09277,
    #   'TO_target':0.05,
    #   'kappa':0.7474
    #   },
    # {'number':11,
    #   'KM_target':0.1322,
    #   'TO_target':0.05,
    #   'kappa':0.7474
    #   },
    # {'number':12,
    #   'KM_target':0.06,
    #   'TO_target':0.036,
    #   'kappa':0.7474
    #   },
    # {'number':13,
    #   'KM_target':0.09277,
    #   'TO_target':0.036,
    #   'kappa':0.7474
    #   },
    # {'number':14,
    #   'KM_target':0.1322,
    #   'TO_target':0.036,
    #   'kappa':0.7474
    #   },
    # {'number':15,
    #   'KM_target':0.06,
    #   'TO_target':0.0124,
    #   'kappa':0.7474
    #   },
    # {'number':16,
    #   'KM_target':0.09277,
    #   'TO_target':0.0124,
    #   'kappa':0.7474
    #   },
    # {'number':17,
    #   'KM_target':0.1322,
    #   'TO_target':0.0124,
    #   'kappa':0.7474
    #   }
    # ]

runs_params = [
    # {'number':1,
    #   'TO_target':0.0183,
    #   },
    {'number':2,
      'TO_target':0.0124,
      },
    # {'number':3,
    #   'TO_target':0.0124,
    #   }
    ]
baseline_number = '402'
# variation_number = 1
for variation_number in range(1,19):
    print(variation_number)
    for run_params in runs_params:
        
        baseline_dic = {'baseline':baseline_number,
                        'variation':str(variation_number)+'.'+str(run_params['number'])}
        new_run = True
        
        if new_run:
            p = parameters(n=7,s=2)
            # p.load_data('calibration_results_matched_economy/'+baseline_number+'/')
            p.load_data('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'+str(variation_number)+'.0/')
            start_time = time.perf_counter()
        
        # test = var.var_from_vector(p.guess, p)    
        # # print(baseline_dic['variation'],test.psi_star.min())
        # print(baseline_dic['variation'],(test.psi_star==1).sum())
        
        m = moments()
        m.load_data()
        # m.load_run('calibration_results_matched_economy/'+baseline_number+'/')
        m.load_run('calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'+str(variation_number)+'.0/')
        if 'theta' in p.calib_parameters:
            p.update_sigma_with_SRDUS_target(m)
        
        m.drop_CHN_IND_BRA_ROW_from_RD = True
        
        m.TO_target = np.float64(run_params['TO_target'])
        # m.TO_target = np.array(run_params['TO_target'])
        # m.KM_target = np.float64(run_params['KM_target'])
        # m.KM_target = np.array(run_params['KM_target'])
        # m.weights_dict['SINNOVPATUS'] = 2
        # p.kappa = np.array(run_params['kappa'])
        
        if new_run:
            hist = history(*tuple(m.list_of_moments+['objective']))
        bounds = p.make_parameters_bounds()
        cond = True
        iterations = 0
        max_iter = 5
        
        while cond:
            if iterations < max_iter-2:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        # method= 'dogbox',
                                        # loss='arctan',
                                        # jac='3-point',
                                        max_nfev=1e8,
                                        # ftol=1e-14, 
                                        xtol=1e-10, 
                                        # gtol=1e-14,
                                        # f_scale=scale,
                                        verbose = 2)
            else:
                test_ls = optimize.least_squares(fun = calibration_func,    
                                        x0 = p.make_p_vector(), 
                                        args = (p,m,p.guess,hist,start_time), 
                                        bounds = bounds,
                                        # method= 'dogbox',
                                        # loss='arctan',
                                        # jac='3-point',
                                        max_nfev=1e8,
                                        # ftol=1e-14, 
                                        xtol=1e-16, 
                                        # gtol=1e-14,
                                        # f_scale=scale,
                                        verbose = 2)
            cond = iterations < max_iter
            iterations += 1
            p.update_parameters(test_ls.x)
        
        finish_time = time.perf_counter()
        print('minimizing time',finish_time-start_time)
        
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
        # sol_c.compute_price_indices(p_sol)
        sol_c.compute_non_solver_quantities(p_sol) 
        p_sol.tau = sol_c.tau
        m.compute_moments(sol_c,p_sol)
        m.compute_moments_deviations()
        m.plot_moments(m.list_of_moments)
        
        ##%% writing results as excel and locally
        
        commentary = 'TO:'+str(run_params['TO_target'])
        # commentary = ''
        dropbox_path = '/Users/slepot/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/'
        local_path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
        # local_path = 'calibration_results_matched_economy/'
        # baseline_number = '102'
        run_number = baseline_dic['variation']
        # run_number = baseline_number
        path = dropbox_path+'baseline_'+baseline_number+'_variations/'
        
        # new_baseline = False
        # if new_baseline:
        #     local_path = 'calibration_results_matched_economy/'
        #     path = dropbox_path
            
        try:
            os.mkdir(path)
        except:
            pass
        
        write_calibration_results(path+str(run_number),p_sol,m,sol_c,commentary = commentary)
        # m.plot_moments(m.list_of_moments, save_plot = path+str(run_number))
        
        try:
            os.mkdir(local_path)
        except:
            pass
        p_sol.write_params(local_path+str(run_number)+'/')
        m.write_moments(local_path+str(run_number)+'/')
        
        ##%% Nash eq
        p_baseline = p_sol.copy()
        
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        
        sol_baseline.scale_P(p_baseline)
        # sol_baseline.compute_price_indices(p_baseline)
        sol_baseline.compute_non_solver_quantities(p_baseline)   
        
        write = True
        
        method = 'fixed_point'
        
        deltas,welfares = find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=12,method='fixed_point',
                          plot_convergence = True,solver_options=None,tol=1e-3)
        
        p = p_baseline.copy()
        p.delta[...,1] = deltas[...,-1]
        
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
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
                                max_count = 1e4,
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
        # sol_c = var.var_from_vector(sol.x, p)    
        # sol_c.scale_tau(p)
        sol_c.scale_P(p)
        # sol_c.compute_price_indices(p)
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
        sol_c.compute_world_welfare_changes(p,sol_baseline)
        
        if write:
            if not os.path.exists('nash_eq_recaps/deltas.csv'):
                deltas_df = pd.DataFrame(columns = ['baseline',
                                                'variation',
                                                'method'] + p_baseline.countries)
                deltas_df.to_csv('nash_eq_recaps/deltas.csv')
            deltas_df = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
            run = pd.DataFrame(data = [baseline_dic['baseline'],
                            baseline_dic['variation'],
                            method]+deltas[...,-1].tolist(), 
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
                            method]+welfares[...,-1].tolist()+[sol_c.cons_eq_pop_average_welfare_change,
                                                                sol_c.cons_eq_negishi_welfare_change], 
                            index = cons_eq_welfares.columns).T
            cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
            cons_eq_welfares.to_csv('nash_eq_recaps/cons_eq_welfares.csv')
        
        ##%% Coop eq
        lb_delta = 0.01
        ub_delta = 12
        p_baseline = p_sol.copy()
        
        def minus_welfare_of_delta_pop_weighted(deltas,p,sol_baseline):
            p.delta[...,1] = deltas
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-15,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 5,
                                    max_count = 1e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=2
                                    # damping=10
                                      # apply_bound_psi_star=True
                                    )
            # sol_c = var.var_from_vector(sol.x, p)    
            # sol_c.scale_tau(p)
            sol_c.scale_P(p)
            # sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            sol_c.compute_world_welfare_changes(p, sol_baseline)
            # print(-sol_c.pop_average_welfare_change)
            
            return -sol_c.cons_eq_pop_average_welfare_change
    
        def minus_welfare_of_delta_negishi_weighted(deltas,p,sol_baseline):
            p.delta[...,1] = deltas
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-15,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 5,
                                    max_count = 1e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=2
                                    # damping=10
                                      # apply_bound_psi_star=True
                                    )
            # sol_c = var.var_from_vector(sol.x, p)    
            # sol_c.scale_tau(p)
            sol_c.scale_P(p)
            # sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            sol_c.compute_world_welfare_changes(p, sol_baseline)
            
            return -sol_c.cons_eq_negishi_welfare_change
    
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        
        sol_baseline.scale_P(p_baseline)
        # sol_baseline.compute_price_indices(p_baseline)
        sol_baseline.compute_non_solver_quantities(p_baseline)   
        
        for aggregation_method in ['pop_weighted','negishi']:
            print(aggregation_method)
            p = p_baseline.copy()
            bounds = [(lb_delta,ub_delta)]*len(p.countries)
            if aggregation_method == 'pop_weighted':
                sol = optimize.minimize(fun = minus_welfare_of_delta_pop_weighted,
                                        x0 = p.delta[...,1],
                                        tol = 1e-8,
                                        args=(p,sol_baseline),
                                        # options = {'disp':True},
                                        bounds=bounds,
                    )
            if aggregation_method == 'negishi':
                sol = optimize.minimize(fun = minus_welfare_of_delta_negishi_weighted,
                                        x0 = p.delta[...,1],
                                        tol = 1e-8,
                                        args=(p,sol_baseline),
                                        # options = {'disp':True},
                                        bounds=bounds
                    )
            
            
            # solve here opt_deltas
            
            p.delta[...,1] = sol.x
            
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-15,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 1e4,
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
            # sol_c = var.var_from_vector(sol.x, p)    
            # sol_c.scale_tau(p)
            sol_c.scale_P(p)
            # sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
            sol_c.compute_world_welfare_changes(p,sol_baseline)
            
            # welfares = sol_c.cons_eq_welfare
                
            write = True
            if write:
                if not os.path.exists('coop_eq_recaps/deltas.csv'):
                    deltas_df = pd.DataFrame(columns = ['baseline',
                                                    'variation',
                                                    'aggregation_method'] + p_baseline.countries)
                    deltas_df.to_csv('coop_eq_recaps/deltas.csv')
                deltas_df = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
                run = pd.DataFrame(data = [baseline_dic['baseline'],
                                baseline_dic['variation'],
                                aggregation_method]+p.delta[...,1].tolist(), 
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
                                aggregation_method]+sol_c.cons_eq_welfare.tolist()+[sol_c.cons_eq_pop_average_welfare_change,
                                                                   sol_c.cons_eq_negishi_welfare_change], 
                                index = cons_eq_welfares.columns).T
                cons_eq_welfares = pd.concat([cons_eq_welfares, run],ignore_index=True)
                cons_eq_welfares.to_csv('coop_eq_recaps/cons_eq_welfares.csv')
            
        ## %% counterfactuals
        
        if baseline_dic['variation'] is None:
            local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline_dic['baseline']+'/'
        else:
            local_path = \
                f'counterfactual_results/unilateral_patent_protection/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'
        
        try:
            os.mkdir(local_path)
        except:
            pass
        
        for c in p_baseline.countries:
            country_path = local_path+c+'/'
            try:
                os.mkdir(country_path)
            except:
                pass
        
            print(c)
            p = p_baseline.copy()
            deltas = np.logspace(-1,1,111)
            idx_country = p_baseline.countries.index(c)
            for i,delt in enumerate(deltas):
                print(delt)
                p.delta[p.countries.index(c),1] = p_baseline.delta[p.countries.index(c),1] * delt
                sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                                context = 'counterfactual',
                                        cobweb_anim=False,tol =1e-15,
                                        accelerate=False,
                                        accelerate_when_stable=True,
                                        cobweb_qty='phi',
                                        plot_convergence=False,
                                        plot_cobweb=False,
                                        safe_convergence=0.001,
                                        disp_summary=False,
                                        damping = 10,
                                        max_count = 1e4,
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
    
                # sol_c = var.var_from_vector(sol.x, p)    
                # sol_c.scale_tau(p)
                sol_c.scale_P(p)
                # sol_c.compute_price_indices(p)
                sol_c.compute_non_solver_quantities(p)
                # sol_c.compute_welfare(p)
                # sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
                if sol.status == 'successful':
                    p.guess = sol_c.vector_from_var()
                else:
                    p.guess = None
                # print(p.guess)
                p.write_params(country_path+'/'+str(i)+'/') 
            
        c = 'World'    
        
        country_path = local_path+c+'/'
        try:
            os.mkdir(country_path)
        except:
            pass
    
        print(c)
        p = p_baseline.copy()
        # sols_c = []
        deltas = np.logspace(-1,1,111)
        # idx_country = p_baseline.countries.index(c)
        for i,delt in enumerate(deltas):
            print(delt)
            p.delta[:,1] = p_baseline.delta[:,1] * delt
            # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
            # print(p.guess)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-15,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    # apply_bound_psi_star = False,
                                    damping = 10,
                                    max_count = 1e4,
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
        
            # sol_c = var.var_from_vector(sol.x, p)    
            # sol_c.scale_tau(p)
            sol_c.scale_P(p)
            # sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p)
            # sol_c.compute_welfare(p)
            # sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            if sol.status == 'successful':
                p.guess = sol_c.vector_from_var()
            else:
                p.guess = None
            p.write_params(country_path+'/'+str(i)+'/')
            
        c = 'Harmonizing'    
        
        country_path = local_path+c+'/'
        try:
            os.mkdir(country_path)
        except:
            pass
    
        print(c)
        p = p_baseline.copy()
        # sols_c = []
        deltas = np.linspace(0,1,101)
        # idx_country = p_baseline.countries.index(c)
        for i,delt in enumerate(deltas):
            print(delt)
            p.delta[:,1] = p_baseline.delta[:,1]**(1-delt) * p_baseline.delta[p_baseline.countries.index('USA'),1]**delt
            # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
            # print(p.guess)
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-15,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    # apply_bound_psi_star = False,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 1e4,
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
            # print(sol.status)
        
            # sol_c = var.var_from_vector(sol.x, p)    
            # sol_c.scale_tau(p)
            sol_c.scale_P(p)
            # sol_c.compute_price_indices(p)
            sol_c.compute_non_solver_quantities(p)
            # sol_c.compute_welfare(p)
            # sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            if sol.status == 'successful':
                p.guess = sol_c.vector_from_var()
            else:
                p.guess = None
            # print(p.guess)
            p.write_params(country_path+'/'+str(i)+'/')
            
        recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'
        
        if baseline_dic['variation'] is None:
            baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
        else:
            # baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'_'+baseline_dic['variation']+'/'
            baseline_path = \
                f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
        print(baseline_path)
        if baseline_dic['variation'] is None:
            local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline_dic['baseline']+'/'
        else:
            local_path = \
                f'counterfactual_results/unilateral_patent_protection/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'
                
        p_baseline = parameters(n=7,s=2) #!!!
        p_baseline.load_data(baseline_path)
        # print(p_baseline.delta)
        # m_baseline = moments()
        # m_baseline.load_data()
        # m_baseline.load_run(baseline_path)
        # sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True)
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        sol_baseline.scale_P(p_baseline)
        # sol_baseline.compute_price_indices(p_baseline)
        sol_baseline.compute_non_solver_quantities(p_baseline)
        
        if baseline_dic['variation'] is None:
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'
        else:
            recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'
        
        try:
            os.mkdir(recap_path)
        except:
            pass
        
        for c in p_baseline.countries:
            recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
            print(c)
            idx_country = p_baseline.countries.index(c)
            country_path = local_path+c+'/'
            files_in_dir = next(os.walk(country_path))[1]
            run_list = [f for f in files_in_dir if f[0].isnumeric()]
            run_list.sort(key=float)
            for run in run_list:
                # print(run) 
                p = parameters(n=7,s=2)
                p.load_data(country_path+run+'/')
                # print(p.delta)
                # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
                # time.sleep(100)
                # print(p.guess)
                if p.guess is not None:
                    sol = var.var_from_vector(p.guess, p, compute=True,context = 'counterfactual')
                    # sol.compute_non_solver_aggregate_qualities(p)
                    # sol.compute_non_solver_quantities(p)
                    sol.scale_P(p)
                    # sol.compute_price_indices(p)
                    sol.compute_non_solver_quantities(p)
                    sol.compute_consumption_equivalent_welfare(p,sol_baseline)
                    recap.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                    recap.loc[run, 'growth'] = sol.g
                    recap.loc[run,p_baseline.countries] = sol.cons_eq_welfare
            recap.to_csv(recap_path+c+'.csv', index=False)
            
        for c in ['World']:
            recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
            print(c)
            idx_country = p_baseline.countries.index('USA')
            country_path = local_path+c+'/'
            files_in_dir = next(os.walk(country_path))[1]
            run_list = [f for f in files_in_dir if f[0].isnumeric()]
            run_list.sort(key=float)
            for run in run_list:
                # print(run) 
                p = parameters(n=7,s=2)
                p.load_data(country_path+run+'/')
                # print(p.delta)
                # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
                # time.sleep(100)
                # print(p.guess)
                if p.guess is not None:
                    sol = var.var_from_vector(p.guess, p, compute=True,context = 'counterfactual')
                    # sol.compute_non_solver_aggregate_qualities(p)
                    # sol.compute_non_solver_quantities(p)
                    sol.scale_P(p)  
                    # sol.compute_price_indices(p)
                    sol.compute_non_solver_quantities(p)
                    sol.compute_consumption_equivalent_welfare(p,sol_baseline)
                    recap.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                    recap.loc[run, 'growth'] = sol.g
                    recap.loc[run,p_baseline.countries] = sol.cons_eq_welfare
                    # recap.loc[run, 'psi_star_min'] = 1+np.log(1+np.log(sol.psi_star.min()))
            recap.to_csv(recap_path+c+'.csv', index=False)
                # print(sol.psi_star.min())
            # recap.plot()
            
        for c in ['Harmonizing']:
            recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
            print(c)
            idx_country = p_baseline.countries.index('EUR')
            country_path = local_path+c+'/'
            files_in_dir = next(os.walk(country_path))[1]
            run_list = [f for f in files_in_dir if f[0].isnumeric()]
            run_list.sort(key=float)
            for run in run_list:
                # print(run) 
                p = parameters(n=7,s=2)
                p.load_data(country_path+run+'/')
                # print(p.delta)
                # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
                # time.sleep(100)
                # print(p.guess)
                if p.guess is not None:
                    sol = var.var_from_vector(p.guess, p, compute=True,context = 'counterfactual')
                    # sol.compute_non_solver_aggregate_qualities(p)
                    # sol.compute_non_solver_quantities(p)
                    sol.scale_P(p)
                    # sol.compute_price_indices(p)
                    sol.compute_non_solver_quantities(p)
                    sol.compute_consumption_equivalent_welfare(p,sol_baseline)
                    recap.loc[run, 'delt'] = np.log(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])/np.log(p_baseline.delta[0,1]/p_baseline.delta[idx_country,1])
                    recap.loc[run, 'growth'] = sol.g
                    recap.loc[run,p_baseline.countries] = sol.cons_eq_welfare
            recap.to_csv(recap_path+c+'.csv', index=False)