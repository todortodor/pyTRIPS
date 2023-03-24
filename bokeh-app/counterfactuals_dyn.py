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
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import time

baseline_dics = [
    {'baseline':'501','variation': '2.0'}
    ]

for baseline_dic in baseline_dics:
    if baseline_dic['variation'] is None:
        baseline_path = 'calibration_results_matched_economy/'+baseline_dic['baseline']+'/'
    else:
        baseline_path = \
            f'calibration_results_matched_economy/baseline_{baseline_dic["baseline"]}_variations/{baseline_dic["variation"]}/'
    print(baseline_path)
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
    if baseline_dic['variation'] is None:
        local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline_dic['baseline']+'/'
    else:
        local_path = \
            f'counterfactual_results/unilateral_patent_protection/baseline_{baseline_dic["baseline"]}_{baseline_dic["variation"]}/'

    try:
        os.mkdir(local_path)
    except:
        pass
    
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
    
    for c in p_baseline.countries:

        country_path = local_path+c+'/'
        # try:
        #     os.mkdir(country_path)
        # except:
        #     pass
    
        print(c)
        p = p_baseline.copy()
        deltas = np.logspace(-1,1,111)
        # deltas = np.logspace(0,1,11)
        test_initial_guess = None

        idx_country = p_baseline.countries.index(c)
        for i,delt in enumerate(deltas):
            print(delt)
            p.delta[p.countries.index(c),1] = p_baseline.delta[p.countries.index(c),1] * delt
            # sol, sol_c = fixed_point_solver(p,x0=p.guess,
            #                         context = 'counterfactual',
            #                         cobweb_anim=False,tol =1e-15,
            #                         accelerate=False,
            #                         accelerate_when_stable=True,
            #                         cobweb_qty='phi',
            #                         plot_convergence=False,
            #                         plot_cobweb=False,
            #                         safe_convergence=0.001,
            #                         disp_summary=False,
            #                         damping = 10,
            #                         max_count = 1e4,
            #                         accel_memory = 50, 
            #                         accel_type1=True, 
            #                         accel_regularization=1e-10,
            #                         accel_relaxation=0.5, 
            #                         accel_safeguard_factor=1, 
            #                         accel_max_weight_norm=1e6,
            #                         damping_post_acceleration=5
            #                         )
            # sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
            # sol_c.compute_solver_quantities(p)
            # sol_c.scale_P(p)
            # sol_c.compute_non_solver_quantities(p)
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline,Nt=25,
                                                  t_inf=500,x0=test_initial_guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=True,
                                    damping = 20,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            # dyn_sol_c.plot_country(0)
            # sol_c.scale_P(p)
            # sol_c.compute_non_solver_quantities(p)
            
            if sol.status == 'successful':
                test_initial_guess = dyn_sol_c.vector_from_var()
                p.dyn_guess = dyn_sol_c.vector_from_var()
                p.guess = dyn_sol_c.sol_fin.vector_from_var()
            else:
                p.dyn_guess = None
                p.guess = None

            p.write_params(country_path+'/'+str(i)+'/') 
            
        test_initial_guess=None
            
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
    # test_initial_guess = None
    
    for i,delt in enumerate(deltas):
        print(delt)
        p.delta[:,1] = p_baseline.delta[:,1] * delt
        # print(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])
        # print(p.guess)
        # sol, sol_c = fixed_point_solver(p,x0=p.guess,
        #                         context = 'counterfactual',
        #                         cobweb_anim=False,tol =1e-15,
        #                         accelerate=False,
        #                         accelerate_when_stable=True,
        #                         cobweb_qty='phi',
        #                         plot_convergence=False,
        #                         plot_cobweb=False,
        #                         safe_convergence=0.001,
        #                         disp_summary=False,
        #                         # apply_bound_psi_star = False,
        #                         damping = 10,
        #                         max_count = 1e4,
        #                         accel_memory = 50, 
        #                         accel_type1=True, 
        #                         accel_regularization=1e-10,
        #                         accel_relaxation=0.5, 
        #                         accel_safeguard_factor=1, 
        #                         accel_max_weight_norm=1e6,
        #                         damping_post_acceleration=5
        #                         # damping=10
        #                           # apply_bound_psi_star=True
        #                         )
        # # print(sol.status)
    
        # # sol_c = var.var_from_vector(sol.x, p)    
        # # sol_c.scale_tau(p)
        # sol_c.scale_P(p)
        # # sol_c.compute_price_indices(p)
        # sol_c.compute_non_solver_quantities(p)
        # sol_c.compute_welfare(p)
        # sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        # if sol.status == 'successful':
        #     p.guess = sol_c.vector_from_var()
        # else:
        #     p.guess = None
        # print(p.guess)
        # p.write_params(country_path+'/'+str(i)+'/')
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline,Nt=25,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=True,
                                damping = 20,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        
        if sol.status == 'successful':
            p.dyn_guess = dyn_sol_c.vector_from_var()
            p.guess = dyn_sol_c.sol_fin.vector_from_var()
        else:
            p.dyn_guess = None
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
        # sol, sol_c = fixed_point_solver(p,x0=p.guess,
        #                         context = 'counterfactual',
        #                         cobweb_anim=False,tol =1e-15,
        #                         accelerate=False,
        #                         accelerate_when_stable=True,
        #                         # apply_bound_psi_star = False,
        #                         cobweb_qty='phi',
        #                         plot_convergence=False,
        #                         plot_cobweb=False,
        #                         safe_convergence=0.001,
        #                         disp_summary=False,
        #                         damping = 10,
        #                         max_count = 1e4,
        #                         accel_memory = 50, 
        #                         accel_type1=True, 
        #                         accel_regularization=1e-10,
        #                         accel_relaxation=0.5, 
        #                         accel_safeguard_factor=1, 
        #                         accel_max_weight_norm=1e6,
        #                         damping_post_acceleration=5
        #                         # damping=10
        #                           # apply_bound_psi_star=True
        #                         )
        # # print(sol.status)
    
        # # sol_c = var.var_from_vector(sol.x, p)    
        # # sol_c.scale_tau(p)
        # # sol_c.compute_solver_quantities()
        # sol_c.scale_P(p)
        # # sol_c.compute_price_indices(p)
        # sol_c.compute_non_solver_quantities(p)
        # sol_c.compute_welfare(p)
        # sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline,Nt=25,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=True,
                                damping = 20,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        
        if sol.status == 'successful':
            p.dyn_guess = dyn_sol_c.vector_from_var()
            p.guess = dyn_sol_c.sol_fin.vector_from_var()
        else:
            p.dyn_guess = None
            p.guess = None

        p.write_params(country_path+'/'+str(i)+'/') 
 
#%%

# baseline_dics = [{'baseline':'405','variation': '1.12'}]

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import seaborn as sns
# from classes import moments, parameters, var, dynamic_var
# from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
# import time

recaps_path = 'counterfactual_recaps/unilateral_patent_protection/'

try:
    os.mkdir(recaps_path)
except:
    pass

for baseline_dic in baseline_dics:
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
    p_baseline = parameters(n=7,s=2)
    p_baseline.load_data(baseline_path)
    # print(p_baseline.delta)
    # m_baseline = moments()
    # m_baseline.load_data()
    # m_baseline.load_run(baseline_path)
    # sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True)
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
    
    if baseline_dic['variation'] is None:
        recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'/'
    else:
        recap_path = recaps_path+'baseline_'+baseline_dic['baseline']+'_'+baseline_dic["variation"]+'/'
    
    try:
        os.mkdir(recap_path)
    except:
        pass
    
    for c in p_baseline.countries:
    # for c in ['USA']:
        # recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
        recap_dyn = pd.DataFrame(columns = ['delt']+p_baseline.countries)
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
                sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
                # print(sol_c.price_indices[0])
                sol_c.compute_solver_quantities(p)
                # sol.compute_non_solver_aggregate_qualities(p)
                # sol.compute_non_solver_quantities(p)
                sol_c.scale_P(p)
                # sol.compute_price_indices(p)
                sol_c.compute_non_solver_quantities(p)
                sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
                # recap.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                # recap.loc[run, 'growth'] = sol_c.g
                # recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
            if p.dyn_guess is not None:
                # print('working')
                dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                        Nt=25,t_inf=500,
                                                        sol_init = sol_baseline,
                                                        sol_fin = sol_c)
                dyn_sol_c.compute_non_solver_quantities(p)
                recap_dyn.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                # recap_dyn.loc[run, 'growth'] = dyn_sol_c.g
                recap_dyn.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare
        recap_dyn.to_csv(recap_path+'dyn_'+c+'.csv', index=False)
    
    for c in ['World']:
        # recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
        recap_dyn = pd.DataFrame(columns = ['delt']+p_baseline.countries)
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
                sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
                # sol_c.compute_solver_quantities(p)
                # sol.compute_non_solver_aggregate_qualities(p)
                # sol.compute_non_solver_quantities(p)
                sol_c.scale_P(p)  
                # sol.compute_price_indices(p)
                sol_c.compute_non_solver_quantities(p)
                sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
                # recap.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                # recap.loc[run, 'growth'] = sol_c.g
                # recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
                # recap.loc[run, 'psi_star_min'] = 1+np.log(1+np.log(sol.psi_star.min()))
            if p.dyn_guess is not None:
                # print('working')
                dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                        Nt=25,t_inf=500,
                                                        sol_init = sol_baseline,
                                                        sol_fin = sol_c)
                dyn_sol_c.compute_non_solver_quantities(p)
                recap_dyn.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                # recap_dyn.loc[run, 'growth'] = dyn_sol_c.g
                recap_dyn.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare
        recap_dyn.to_csv(recap_path+'dyn_'+c+'.csv', index=False)
            # print(sol.psi_star.min())
        # recap.plot()
        
    for c in ['Harmonizing']:
        recap_dyn = pd.DataFrame(columns = ['delt']+p_baseline.countries)
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
                sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
                # sol.compute_non_solver_aggregate_qualities(p)
                # sol.compute_non_solver_quantities(p)
                # sol_c.compute_solver_quantities(p)
                sol_c.scale_P(p)
                # sol.compute_price_indices(p)
                sol_c.compute_non_solver_quantities(p)
                sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
            if p.dyn_guess is not None:
                # print('working')
                dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                        Nt=25,t_inf=500,
                                                        sol_init = sol_baseline,
                                                        sol_fin = sol_c)
                dyn_sol_c.compute_non_solver_quantities(p)
                # recap_dyn.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
                recap_dyn.loc[run, 'delt'] = np.log(p.delta[idx_country,1]/p_baseline.delta[idx_country,1])/np.log(p_baseline.delta[0,1]/p_baseline.delta[idx_country,1])
                # recap_dyn.loc[run, 'growth'] = dyn_sol_c.g
                recap_dyn.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare
        recap_dyn.to_csv(recap_path+'dyn_'+c+'.csv', index=False)
    
#%%
for country in p_baseline.countries:
# country = 'USA'
    recap_dyn = pd.read_csv(recap_path+'dyn_'+country+'.csv')
    recap = pd.read_csv(recap_path+country+'.csv')
    
    fig,ax = plt.subplots(2,1,figsize=(15,10),layout = "constrained")
    
    for i,c in enumerate(p_baseline.countries):
    # for i,c in enumerate(['USA']):
        ax[1].plot(recap['delt'],recap[c],color=sns.color_palette()[i],label = c)
        ax[0].plot(recap_dyn['delt'],recap_dyn[c],color=sns.color_palette()[i],label = c)
    
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].legend(title='With dynamics')
    ax[1].legend(title='Steady state only')
    plt.suptitle('Patent protection counterfactual for '+country)
    
    plt.savefig('/Users/slepot/Dropbox/TRIPS/simon_version/code/misc/dynamics_counterfactuals/'+country)
    
    plt.show()
    
    
          