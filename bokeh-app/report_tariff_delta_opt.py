#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:01:51 2024

@author: slepot
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver

scenarios = []
names = {
    '0':'Nash tariff, baseline delta',
    # '1':'Nash delta and tariff',
    # '2':'Nash tariff, cooperative delta Equal',
    # '3':'Nash tariff, cooperative delta Negishi',
    '6':'Cooperative tariff Equal, baseline delta',
    '7':'Cooperative tariff Equal and cooperative delta Equal',
    '10':'Cooperative tariff Negishi, baseline delta',
    '11':'Cooperative tariff Negishi and cooperative delta Negishi',
    # '14':'Nash delta, baseline tariff',
    # '15':'Cooperative delta Equal, baseline tariff',
    # '16':'Cooperative delta Negishi, baseline tariff'
    }

p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/1040/')
sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        # plot_live=True,
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
                        ) 
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

path = 'opt_tariff_delta/1050/summary'

writer = pd.ExcelWriter(path+'.xlsx', engine='xlsxwriter')
workbook = writer.book


# for scenario in ['0','1','2','3','6','7','10','11','14','15','16']:
for scenario in names:
    p = parameters()
    p.load_run(f'opt_tariff_delta/1050/scenario_{scenario}/')
    p.delta[p.delta>2] = 12
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
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
                            ) 
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    
    p.guess = sol_c.vector_from_var()
    # p.write_params(f'opt_tariff_delta/1040/scenario_{scenario}/')
    
    scenarios.append({
        'name':names[scenario],
        'p':p,
        'sol':sol_c
        })
    
    worksheet = workbook.add_worksheet(f'scenario_{scenario}')
    writer.sheets[f'scenario_{scenario}'] = worksheet

    worksheet.write_string(0, 0, names[scenario])
    
    df = pd.DataFrame(index = p.countries + ['Negishi', 'Equal'], columns = ['welfare','delta'])
    df['welfare'] = (sol_c.cons_eq_welfare*100-100).tolist() + [sol_c.cons_eq_negishi_welfare_change*100-100,
                                                                 sol_c.cons_eq_pop_average_welfare_change*100-100]
    df['welfare'] = df['welfare'].round(2)
    df.loc[p.countries,'delta'] = p.delta[:,1].round(4)
    
    df.to_excel(writer,sheet_name=f'scenario_{scenario}',startrow = 1 , startcol=0)
    
    df_tariff = pd.DataFrame(index = p.countries, columns = p.countries, data = p.tariff[:,:,1]*100)
    df_tariff = df_tariff.round(2)
    
    df_tariff.to_excel(writer,sheet_name=f'scenario_{scenario}',startrow = 15 , startcol=0)
    
    print(names[scenario])
    print(p.tariff[:,:,1].mean())
    print('Negishi welfare',sol_c.cons_eq_negishi_welfare_change*100-100)
    print('Equal welfare',sol_c.cons_eq_pop_average_welfare_change*100-100)

writer.close()