#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:37:24 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classes import moments, parameters, var, var_with_entry_costs, dynamic_var
from solver_funcs import fixed_point_solver_with_entry_costs, fixed_point_solver_with_entry_costs_cf

baseline = '1300'
variation = '11.02'

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

p = p_baseline.copy()
p.delta[0,1] = np.logspace(-1,1,51)[2]*p.delta[0,1]

sol, sol_c = fixed_point_solver_with_entry_costs(p,
                                                 # x0=p.guess[:-12],
                                                 x0=p.guess,
                        context = 'counterfactual',
                        # context = 'calibration',
                        cobweb_anim=False,tol =1e-10,
                        accelerate=True,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
                        # context = 'counterfactual',
                        # cobweb_anim=False,tol =1e-10,
                        # accelerate=False,
                        # accelerate_when_stable=True,
                        # cobweb_qty='phi',
                        # plot_convergence=True,
                        # plot_cobweb=False,
                        # safe_convergence=0.01,
                        # disp_summary=False,
                        # damping = 10,
                        # max_count = 500,
                        # accel_memory = 50, 
                        # accel_type1=True, 
                        # accel_regularization=1e-10,
                        # accel_relaxation=0.5, 
                        # accel_safeguard_factor=1, 
                        # accel_max_weight_norm=1e6,
                        # damping_post_acceleration=4
                        # )
                        # context = 'counterfactual',
                        # cobweb_anim=False,tol =1e-10,
                        # accelerate=False,
                        # accelerate_when_stable=True,
                        # cobweb_qty='phi',
                        # plot_convergence=True,
                        # plot_cobweb=False,
                        # safe_convergence=0.001,
                        # disp_summary=False,
                        # damping = 10,
                        # max_count = 1e3,
                        # accel_memory = 50, 
                        # accel_type1=True, 
                        # accel_regularization=1e-10,
                        # accel_relaxation=0.5, 
                        # accel_safeguard_factor=1, 
                        # accel_max_weight_norm=1e6,
                        # damping_post_acceleration=4
                        # )

sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
sol_c.compute_average_ratio_entry_costs_exports(p)

# p.guess

#%%

from classes import moments, parameters, var, var_with_entry_costs, dynamic_var
from solver_funcs import fixed_point_solver_with_entry_costs

baseline = '1300'
variation = '11.0'

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

p = p_baseline.copy()

p.d = 0.2

damping = 5

x_old = p.guess

init = var_with_entry_costs.var_from_vector(x_old,p,context='counterfactual',compute=False)
init.compute_growth(p)
init.compute_entry_costs(p)
init.compute_V(p)
init.compute_patenting_thresholds(p)
init.compute_mass_innovations(p)
init.compute_aggregate_qualities(p)
init.compute_sectoral_prices(p)
init.compute_labor_allocations(p)
init.compute_trade_flows_and_shares(p)

w = init.compute_wage(p)#/init.price_indices[0]
Z = init.compute_expenditure(p)#/init.price_indices[0]
l_R = init.compute_labor_research(p)[...,1:].ravel()
profit = init.compute_profit(p)[...,1:].ravel()
phi = init.compute_phi(p).ravel()#*init.price_indices[0]
price_indices = init.compute_price_indices(p)

x_new = np.concatenate((w/price_indices[0],Z/price_indices[0],l_R,profit,phi*price_indices[0],price_indices/price_indices[0]), axis=0)

x_old = (x_new+(damping-1)*x_old)/damping

print(l_R)

#%%

from solver_funcs import fixed_point_solver_with_entry_costs

probinnov = []
shareexp = []
p = p_baseline.copy()
p.guess= np.concatenate((p.guess,np.ones(p.N)),axis=0)

for d in np.linspace(1.1,5,10):
    print(d)
    
    #%%
    p = p_baseline.copy()
    p.guess= np.concatenate((p.guess,np.ones(p.N)),axis=0)
    
    p.a = 0.1
    p.d = 1.1
    
    sol, sol_c = fixed_point_solver_with_entry_costs(p,
                                                     x0=p.guess,
    # sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
                            context = 'counterfactual',
                            # context = 'calibration',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=True,
                            damping = 10,
                            max_count = 1000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    
    p.guess = sol_c.vector_from_var()  
    
    #%%
    from classes import moments
    
    m = moments()
    m.load_run(run_path)
    
    m.compute_moments(sol_c,p)
    m.compute_PROBINNOVENT(sol_c, p)
    m.compute_SHAREEXPMON(sol_c,p)
    print('US:',m.PROBINNOVENT_US)
    print('JAP:',m.PROBINNOVENT_JAP)
        
    # probinnov.append(m.PROBINNOVENT)
    # shareexp.append(m.SHAREEXPMON)

#%%

# plt.plot(np.linspace(0,1,10),probinnov,label='probability innovator enter a market\n(with max)')
# plt.plot(np.linspace(0,1,10),shareexp,label='share of sales coming from exports\nfor firms doing R&D i')
plt.plot(np.linspace(1.1,5,10),probinnov,label='probability innovator enter a market\n(with max)')
plt.plot(np.linspace(1.1,5,10),shareexp,label='share of sales coming from exports\nfor firms doing R&D i')
plt.legend()
plt.xlabel('d')
plt.show()

#%%
from classes import var_with_entry_costs
import time

p = p_baseline.copy()

p.a = 1
p.d = 1.5

vec = sol_baseline.vector_from_var()
vec = np.concatenate((vec,sol_baseline.price_indices))

sol = var_with_entry_costs.var_from_vector(vec, p, context='counterfactual',compute=False)

# start = time.perf_counter()
sol.compute_growth(p)
sol.compute_entry_costs(p)
sol.compute_V(p)
# sol.compute_patenting_thresholds(p,thresholds_to_compare=sol_baseline.psi_o_star[:,1])
sol.compute_patenting_thresholds(p)
sol.compute_mass_innovations(p)
sol.compute_aggregate_qualities(p)
sol.compute_sectoral_prices(p)
sol.compute_labor_allocations(p)
sol.compute_trade_flows_and_shares(p)
# print(time.perf_counter() - start)

# l_R = sol.compute_labor_research(p)
# price_indices = sol.compute_price_indices(p)
# print(test[2])

sol.compute_non_solver_quantities(p)



#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classes import moments, parameters, var, var_with_entry_costs, dynamic_var
from solver_funcs import fixed_point_solver_with_entry_costs, fixed_point_solver


baseline = '1300'
variation = 'baseline'

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)
df = pd.DataFrame(index = p_baseline.countries+['World Equal'])

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

aggregation_method = 'pop_weighted'

static_eq_deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0).drop_duplicates(
    ['baseline','variation','aggregation_method'],keep='last')
static_eq_deltas = static_eq_deltas.loc[
    (static_eq_deltas.baseline.astype('str') == baseline)
    & (static_eq_deltas.variation.astype('str') == variation)
    & (static_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()

p = p_baseline.copy()
static_eq_deltas[static_eq_deltas<0.5] = 0.01
# static_eq_deltas[static_eq_deltas<0.05] = 0.05
static_eq_deltas[static_eq_deltas>2] = 12
p.delta[:,1] = static_eq_deltas

sol, sol_c = fixed_point_solver(p,
                        x0=p.guess,
                        context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
sol_c.compute_world_welfare_changes(p, sol_baseline)

df.loc[p.countries,'optimal delta without entry costs'] = p.delta[:,1]
df.loc[p.countries,'welfare without entry costs'] = sol_c.cons_eq_welfare*100-100
df.loc['World Equal','welfare without entry costs'] = sol_c.cons_eq_pop_average_welfare_change*100-100
# variation = '11.01'

static_eq_deltas = np.ones(p_baseline.N)*12

p = p_baseline.copy()
p.delta[:,1] = static_eq_deltas

sol, sol_c = fixed_point_solver(p,
                        x0=p.guess,
                        context = 'counterfactual',
                        cobweb_anim=False,tol =1e-9,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        safe_convergence=0.1,
                        disp_summary=True,
                        damping = 10,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=2
                        # damping=10
                          # apply_bound_psi_star=True
                        )
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
sol_c.compute_world_welfare_changes(p, sol_baseline)

sol_opti = sol_c.copy()
p_opti = p.copy()

df.loc[p.countries,'nash delta without entry costs'] = p.delta[:,1]
df.loc[p.countries,'nash welfare without entry costs'] = sol_c.cons_eq_welfare*100-100
df.loc['World Equal','nash welfare without entry costs'] = sol_c.cons_eq_pop_average_welfare_change*100-100

print(df)

# for variation in ['11.0','11.01']:
    # if variation == '11.0':
    #     name_prefix = 'd=1.5 '
    # if variation == '11.01':
    #     name_prefix = 'd=1.1 '
    
for variation in ['11.02']:
    name_prefix = ''

    results_path = 'calibration_results_matched_economy/'
    
    if variation == 'baseline':
        run_path = results_path+baseline+'/'
    else:
        run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'
    
    p_baseline = parameters()
    p_baseline.load_run(run_path)
    
    sol_baseline = var_with_entry_costs.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)
    
    aggregation_method = 'pop_weighted'
    
    static_eq_deltas = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0).drop_duplicates(
        ['baseline','variation','aggregation_method'],keep='last')
    static_eq_deltas = static_eq_deltas.loc[
        (static_eq_deltas.baseline.astype('str') == baseline)
        & (static_eq_deltas.variation.astype('str') == 'baseline')
        & (static_eq_deltas.aggregation_method == aggregation_method)][p_baseline.countries].values.squeeze()
    
    static_eq_deltas[static_eq_deltas<0.5] = 0.01
    static_eq_deltas[static_eq_deltas>1] = 12
    
    p = p_baseline.copy()
    p.delta[:,1] = static_eq_deltas
    
    sol, sol_c = fixed_point_solver_with_entry_costs(p,
                            x0=p.guess,
                            context = 'counterfactual',
                            cobweb_anim=False,tol =1e-9,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=True,
                            damping = 10,
                            max_count = 1000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    sol_opti = sol_c.copy()
    p_opti = p.copy()
    
    df.loc[p.countries,name_prefix+'coop delta with entry costs'] = p.delta[:,1]
    df.loc[p.countries,name_prefix+'coop welfare with entry costs'] = sol_c.cons_eq_welfare*100-100
    df.loc['World Equal',name_prefix+'coop welfare with entry costs'] = sol_c.cons_eq_pop_average_welfare_change*100-100
    
    p_baseline = parameters()
    p_baseline.load_run(run_path)
    
    sol_baseline = var_with_entry_costs.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)
    
    static_eq_deltas = np.ones(p_baseline.N)*12
    
    p = p_baseline.copy()
    p.delta[:,1] = static_eq_deltas
    
    sol, sol_c = fixed_point_solver_with_entry_costs(p,
                            x0=p.guess,
                            context = 'counterfactual',
                            cobweb_anim=False,tol =1e-9,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=True,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=True,
                            damping = 10,
                            max_count = 1000,
                            accel_memory =50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    sol_nash = sol_c.copy()
    p_nash = p.copy()
    
    df.loc[p.countries,name_prefix+'nash delta with entry costs'] = p.delta[:,1]
    df.loc[p.countries,name_prefix+'nash welfare with entry costs'] = sol_c.cons_eq_welfare*100-100
    df.loc['World Equal',name_prefix+'nash welfare with entry costs'] = sol_c.cons_eq_pop_average_welfare_change*100-100
    
    print(df)
    

#%%

df.round(3).to_csv('../misc/coop_with_entry_costs.csv')

#%%
sol_dic = {}

for i,country in enumerate(p_baseline.countries):
    if i>-1:
        print(country)
        sol_dic[country] = {}
        for delta_value in [0.01,12]:
            print(delta_value)
            p = p_baseline.copy()
            p.delta[:,1] = static_eq_deltas
            p.delta[i,1] = delta_value
            
            sol, sol_cf = fixed_point_solver_with_entry_costs(p,
                                    x0=p.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-5,
                                    accelerate=False,
                                    accelerate_when_stable=True,
                                    cobweb_qty='l_R',
                                    plot_convergence=True,
                                    plot_cobweb=False,
                                    safe_convergence=0.1,
                                    disp_summary=True,
                                    damping = 10,
                                    max_count = 1000,
                                    accel_memory =50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=2
                                    # damping=10
                                      # apply_bound_psi_star=True
                                    )
            sol_cf.scale_P(p)
            sol_cf.compute_non_solver_quantities(p)
            sol_cf.compute_consumption_equivalent_welfare(p, sol_baseline)
            sol_cf.compute_world_welfare_changes(p, sol_baseline)
            
            sol_dic[country][delta_value] = sol_cf
        
#%%

df = pd.DataFrame(index=p_baseline.countries)

for i,country in enumerate(p_baseline.countries):
    for delta_value in [0.01,12]:
        df.loc[country,delta_value] = sol_dic[country][delta_value].cons_eq_pop_average_welfare_change - sol_opti.cons_eq_pop_average_welfare_change
        
#%%

df = pd.DataFrame(index=p_baseline.countries)

for i,country in enumerate(p_baseline.countries):
    for delta_value in [0.01,12]:
        df.loc[country,delta_value] = sol_dic[country][delta_value].cons_eq_welfare[i] - sol_nash.cons_eq_welfare[i]
