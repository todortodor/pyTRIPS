#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:42:42 2023

@author: slepot
"""

from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# warnings.simplefilter('ignore', np.RankWarning)

df = pd.DataFrame()
p_init = parameters()

# p_init.load_run('coop_eq_direct_saves/4003_baseline_nash/')
# p_init.load_run('calibration_results_matched_economy/1300/')
p_init.load_run('coop_eq_direct_saves/dyn_2000_14.0_negishi/')
# p_init.delta[1,2] = 12
# p_init.delta[:,1] = np.array([1.0e-02, 1.0e-02, 1.0e-02, 1.2e+01, 1.2e+01, 1.2e+01, 1.0e-02,
#        1.0e-02, 1.0e-02, 1.0e-02, 1.2e+01, 1.2e+01])
# p_init.tau[3,:,1] = np.array([1.06800713, 1.04042852, 1.06321504, 1.        , 1.12498533,
#        1.24024972, 1.0039714 , 1.0376111 , 1.21034042, 1.19033989,
#        1.20639476, 1.05774921])

sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
sol_init.scale_P(p_init)
sol_init.compute_export_price_index(p_init)

# sol_init.export_price_index[1,0,0] = 10

# df = pd.DataFrame(index=p_init.countries,
#                   columns=p_init.countries,
#                   data=sol_init.export_price_index[...,0])

# df.to_csv('export_price_index.csv')

#%%

# sol_init.compute_non_solver_quantities(p_init) 

# p = p_init.copy()
p = p_opti.copy()
p.delta[0,1] = 12.0
# p.delta[:,1] = np.array(
#     [1.0e-02, 1.0e-02, 1.0e-02, 
#      1.2e+01, 1.2e+01, 1.2e+01, 
#      1.0e-02,1.0e-02, 
#      1.2e+01, #RUS
#      # 1.0e-02, #RUS
#      1.0e-02, 
#      1.2e+01, 1.2e+01])
# p.delta[1,2] = 0.01
# p = parameters()
# p.load_run('coop_eq_direct_saves/dyn_6001_4.02_nash/')

# p.delta[0,2] = 1.0

sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-13,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=10
                        )
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p) 
sol_c.compute_consumption_equivalent_welfare(p, sol_init)
p.guess = sol_c.vector_from_var()

sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=25,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
                        damping = 60,
                        max_count = 50000,
                        accel_memory =5, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=1, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
dyn_sol.compute_non_solver_quantities(p)

#%%

df = pd.DataFrame()

for i,country in enumerate(p_init.countries):
# for i,country in enumerate(['USA']):
    print(country)
    p = p_init.copy()
    p.tau[:,i,1] = p_init.tau[:,i,1]/1.01
    p.tau[i,i,1] = 1

    sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=25,
                                          t_inf=500,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=False,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            plot_live = False,
                            safe_convergence=1e-8,
                            disp_summary=True,
                            damping = 50,
                            max_count = 50000,
                            accel_memory =5, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=1, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=10
                            )
    dyn_sol.compute_non_solver_quantities(p)
    df.loc[country,'baseline number of patented innovations'] = sol_init.psi_o_star[i,1]**-p.k * sol_init.l_R[i,1]**(1-p.kappa)
    df.loc[country,'change in number of patented innovations'] = (dyn_sol.psi_o_star[i,1,-3]**-p.k * dyn_sol.l_R[i,1,-3]**(1-p.kappa)
                                                                  / (sol_init.psi_o_star[i,1]**-p.k * sol_init.l_R[i,1]**(1-p.kappa))
                                                                  )*100-100
    
df.loc['average change', 'change in number of patented innovations'] = df.loc[p_init.countries,'change in number of patented innovations'].mean()
df.loc['weighted average change', 'change in number of patented innovations'
       ] = (df.loc[p_init.countries,'change in number of patented innovations']*df.loc[p_init.countries,'baseline number of patented innovations']
            ).sum()/df.loc[p_init.countries,'baseline number of patented innovations'].sum()

#%%


p_init = parameters()
for x in np.linspace(0.01-1e-6,0.01+1e-6,3):
        # p_init.load_run('calibration_results_matched_economy/1030/')
        p_init.load_run(
            f'calibration_results_matched_economy/baseline_1030_variations/2.0/')
        
        sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
                                        context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1000,
                                accel_memory =50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        sol_init.scale_P(p_init)
        sol_init.compute_non_solver_quantities(p_init) 
        
        p = p_init.copy()
        # p.delta[:,1] = np.array([0.01,0.01,0.01,chn_pat[chn],12.0,12.0,0.01,0.01,rus_pat[rus],0.01,12.0])
        p.delta[:,1] = np.ones(p.N)*12
        # p.delta[-2,1] = 6
        p.delta[0,1] = x
        
        sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=23,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=True,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=True,
                                damping = 50,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol.compute_non_solver_quantities(p)
        
        df.loc[x,'welfare_US'] = dyn_sol.cons_eq_welfare[0]

df.plot(y='welfare_US')

#%%

variations_of_robust_checks = {
    'baseline':'Baseline',
    # '99.10':'Low Growth',
    # '99.11':'High Growth',
    # '99.12':'Low rho',
    # '99.13':'High rho',
    # '99.14':'Low UUPCOST',
    # '99.15':'High UUPCOST',
    }
variations_of_robust_checks = {
    'baseline':'Baseline',
    '99.0':'Low TO',
    '99.1':'High TO',
    '99.2':'Low TE',
    '99.3':'High TE',
    '99.4':'Low KM',
    '99.5':'High KM',
    '99.6':'Low Sigma',
    '99.7':'High Sigma',
    '99.8':'Low Kappa',
    '99.9':'High Kappa',
    # '99.10':'Low Growth',
    # '99.11':'High Growth',
    # '99.12':'Low rho',
    # '99.13':'High rho',
    # '99.14':'Low UUPCOST',
    # '99.15':'High UUPCOST',
    }
rus_pat = {'RUS strong':0.01,
           'RUS weak':12}
chn_pat = {'CHN strong':0.01,
           'CHN weak':12}

df = pd.DataFrame(index = pd.MultiIndex.from_product(
    [list(variations_of_robust_checks.values()),
     ['RUS strong','RUS weak'],
     ['CHN strong','CHN weak']]),
    columns = ['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'CAN', 'KOR', 'RUS', 'MEX', 'ROW', 'Equal'])

for rob_check in variations_of_robust_checks:
    for rus in rus_pat:
        for chn in chn_pat:
            p_init = parameters()
            # p_init.load_run('calibration_results_matched_economy/1030/')
            if rob_check == 'baseline':
                p_init.load_run('calibration_results_matched_economy/1030/')
            else:
                p_init.load_run(f'calibration_results_matched_economy/baseline_1030_variations/{rob_check}/')
            
            sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
                                            context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 1000,
                                    accel_memory =50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            sol_init.scale_P(p_init)
            sol_init.compute_non_solver_quantities(p_init) 
            
            p = p_init.copy()
            p.delta[:,1] = np.array([0.01,0.01,0.01,chn_pat[chn],12.0,12.0,0.01,0.01,rus_pat[rus],0.01,12.0])
            
            sol, dyn_sol = dyn_fixed_point_solver(p, sol_init, Nt=40,
                                                  t_inf=1000,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=True,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=True,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol.compute_non_solver_quantities(p)
            
            df.loc[variations_of_robust_checks[rob_check],rus,chn
                    ] = dyn_sol.cons_eq_welfare.tolist() + [dyn_sol.cons_eq_pop_average_welfare_change]
                   # ] = dyn_sol.cons_eq_welfare.tolist() + [dyn_sol.cons_eq_negishi_welfare_change]
            print(df.T)
# print(dyn_sol.cons_eq_negishi_welfare_change)
# def make_time_evolution_df(dyn_sol):
#     qties = ['w','l_R','l_Ae','l_Ao','price_indices','Z','g','r','profit']
#     df = pd.DataFrame(index = qties, columns = ['Initial jump','Typical time of evolution'])
#     for qty in qties:
#         df.loc[qty,'Initial jump'] = dyn_sol.get_jump(qty)
#         df.loc[qty,'Typical time of evolution'] = dyn_sol.get_typical_time_evolution(qty)
#     return df
# print(make_time_evolution_df(dyn_sol))

#%%
df_change = df*100-100
for c in df_change.columns:
    df_change[c] = df_change[c].astype(float).round(3)
    
#%%
df_add_row = df_change.copy()
for rob_check in variations_of_robust_checks:
    for chn in chn_pat:
        df_add_row.loc[variations_of_robust_checks[rob_check],'diff RUS strong / weak',chn] = \
            (df_add_row.loc[variations_of_robust_checks[rob_check],'RUS strong',chn
                           ] - df_add_row.loc[variations_of_robust_checks[rob_check],'RUS weak',chn]).round(3)
