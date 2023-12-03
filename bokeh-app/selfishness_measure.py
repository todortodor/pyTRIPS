#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:53:13 2023

@author: slepot
"""

import pandas as pd
import os
from scipy import interpolate
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver
import matplotlib.pyplot as plt

baseline = '1030'

p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/1030/')

sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 2,
                        max_count = 1000,
                        accel_memory =50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=1
                        # damping=10
                          # apply_bound_psi_star=True
                        )

sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

df = pd.DataFrame()

# def compute_dw_country_over_dw_world(welfares,country):
    
fits = []
fits_n = []
splines = []

for c in p_baseline.countries:
# for c in ['USA']:
    recap = pd.DataFrame(columns = ['delta_change','world_negishi','world_equal','rest_of_world_equal']+p_baseline.countries)
    local_path = 'counterfactual_results/unilateral_patent_protection/baseline_'+baseline+'/'
    print(c)
    if c in p_baseline.countries:
        idx_country = p_baseline.countries.index(c)
    country_path = local_path+c+'/'
    files_in_dir = next(os.walk(country_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    for i,run in enumerate(run_list):
        p = parameters()
        p.load_run(country_path+run+'/')
        if p.guess is not None:
            sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
            sol_c.scale_P(p)
            sol_c.compute_non_solver_quantities(p)
            sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        if p.dyn_guess is not None:
            dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                    Nt=25,t_inf=500,
                                                    sol_init = sol_baseline,
                                                    sol_fin = sol_c)
            dyn_sol_c.compute_non_solver_quantities(p)
        if c in p_baseline.countries:
            recap.loc[run, 'delta_change'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
        recap.loc[run, 'world_negishi'] = dyn_sol_c.cons_eq_negishi_welfare_change
        recap.loc[run, 'world_equal'] = dyn_sol_c.cons_eq_pop_average_welfare_change
        recap.loc[run, 'rest_of_world_equal'] = dyn_sol_c.compute_consumption_equivalent_welfare_for_subset_of_countries(
            p,[c2 for c2 in p.countries if c2 != c])['pop_weighted']
        recap.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare
    
    fit = interpolate.interp1d(recap[c],recap['rest_of_world_equal'])
    fits.append(fit)
    
    fit_n = interpolate.interp1d(recap[c],recap['world_negishi'])
    fits_n.append(fit_n)
    
    spl = interpolate.UnivariateSpline(recap.sort_values(c)[c], recap.sort_values(c)['rest_of_world_equal'])
    splines.append(spl)
    
    # plt.plot(recap[c],[fit(x) for x in recap[c].values],label=c)
    
    # df.loc['country']
# plt.show()
    
#%%

# fit = interpolate.interp1d(recap[c],recap['world_equal'])

df = pd.DataFrame()

for i,c in enumerate(p_baseline.countries):

    df.loc[c,'S'] = 1/(1-1/(splines[i].derivative()(1)) )

# plt.legend()
# plt.xlim(0.99,1.01)
# plt.ylim(0.99,1.01)
# plt.show()

# for i,c in enumerate(p_baseline.countries):

#     plt.plot(fits[i].x,fits[i].y,label=c)

# plt.legend()
# plt.xlim(0.99,1.01)
# plt.ylim(0.99,1.01)
# plt.show()

# import matplotlib.pyplot as plt

# for i,c in enumerate(p_baseline.countries):

#     plt.plot(fits_n[i].x,fits_n[i].y,label=c)

# plt.legend()
# plt.xlim(0.99,1.01)
# plt.ylim(0.99,1.01)
# plt.show()


#%%

import numpy as np
from optimparallel import minimize_parallel
from scipy import optimize

p_nash = p_baseline.copy()
p_nash.delta[:,1] = 12.0

sol, sol_nash = fixed_point_solver(p_nash,x0=p_baseline.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 4,
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

sol_nash.scale_P(p_nash)
sol_nash.compute_non_solver_quantities(p_nash) 

def compute_welfare_quantity(deltas,betas,sol_nash,p_nash,p_baseline):
    p = p_baseline.copy()
    p.delta[:,1] = deltas.copy()
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 4,
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
    sol_c.compute_consumption_equivalent_welfare(p,sol_nash)
    p_baseline.guess = sol_c.vector_from_var()
    
    res = (
        (sol_nash.cons * sol_c.cons_eq_welfare * p_baseline.labor.sum() / p_baseline.labor)**(betas * p_baseline.labor / p_baseline.labor.sum())
            ).prod()
    # res = (
    #     # (sol_nash.cons*sol_c.cons_eq_welfare)**(betas)
    #     (sol_c.cons_eq_welfare)**(betas)
    #        ).prod()
    print(res)
    return -res
    
def find_optimal_deltas(betas,sol_nash,p_nash,p_baseline):
    # delta_min = optimize.shgo(func=compute_welfare_quantity,
    #                             # sampling_method='halton',
    #                             bounds=[(0.01,12.0)]*len(p_baseline.countries),
    #                             args = (betas,sol_nash,p_nash,p_baseline),
    #                             options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
    #                             minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2},
    #                             # workers=-1
    #                             )
    displays = False
    delta_min = minimize_parallel(fun = compute_welfare_quantity,
                            x0 = p_baseline.delta[:,1],
                            tol = 1e-16,
                            args=(betas,sol_nash,p_nash,p_baseline),
                            # options = {'disp':True},
                            bounds=[(0.01,2)]*len(p_baseline.countries),
                            parallel={'max_workers':12,
                                      'loginfo': displays,
                                      'time':displays,
                                      'verbose':displays}
        )
    print(betas,constraint(betas))
    print(delta_min.x)
    return np.linalg.norm(np.abs(delta_min.x - p_baseline.delta[:,1]))

def constraint(betas):
    return np.sum(p_baseline.labor*betas/p_baseline.labor.sum())-1

def find_betas(sol_nash,p_nash,p_baseline):
    # delta_min = optimize.shgo(func=compute_welfare_quantity,
    #                             # sampling_method='halton',
    #                             bounds=[(0.01,12.0)]*len(p_baseline.countries),
    #                             args = (betas,sol_nash,p_nash,p_baseline),
    #                             options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
    #                             minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2},
    #                             # workers=-1
    #                             )
    # displays = True
    # betas = minimize_parallel(fun = find_optimal_deltas,
    #                         x0 = p_baseline.labor.sum()*np.ones(11)/p_baseline.labor/11,
    #                         tol = 1e-16,
    #                         args=(sol_nash,p_nash,p_baseline),
    #                         # options = {'disp':True},
    #                         bounds=[(0.0,1.0)]*len(p_baseline.countries),
    #                         parallel={'max_workers':12,
    #                                   'loginfo': displays,
    #                                   'time':displays,
    #                                   'verbose':displays}
    #     )
    # betas = optimize.shgo(func = find_optimal_deltas, 
    #                       sampling_method='sobol',
    #                       # x0=p_baseline.labor.sum()*np.ones(11)/p_baseline.labor/11,  
    #                       args = (sol_nash,p_nash,p_baseline),
    #                       bounds=[(0.0,p_baseline.labor.sum()/labor) for labor in p_baseline.labor],
    #                       constraints={'type':'eq', 'fun': constraint},
    #                       options={'disp':True})
    betas = optimize.minimize(fun = find_optimal_deltas, 
                          # sampling_method='sobol',
                           x0=p_baseline.labor.sum()*np.ones(11)/p_baseline.labor/11,  
                          args = (sol_nash,p_nash,p_baseline),
                          bounds=[(0.0,p_baseline.labor.sum()/labor) for labor in p_baseline.labor],
                          constraints={'type':'eq', 'fun': constraint},
                          options={'disp':True})
    print(betas.x)
    return betas.x



# test = compute_welfare_quantity(np.ones(11),np.ones(11)/12,sol_nash,p_nash,p_baseline)

if __name__ == '__main__':
    # test = find_optimal_deltas(p_baseline.labor.sum()*np.array(
    #     [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])/p_baseline.labor/11,sol_nash,p_nash,p_baseline)
    test = find_betas(sol_nash,p_nash,p_baseline)
