#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:03:41 2023

@author: slepot
"""

import pandas as pd
import os
from scipy import interpolate
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver
import matplotlib.pyplot as plt
import numpy as np

baseline = '1030'

p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/1030/')

df = pd.DataFrame(index=p_baseline.countries)

# for r,rho in enumerate(np.linspace(0.03,0.1,71)):
# for r,rho in enumerate(np.linspace(0.0297,0.0298,11)):
# p_baseline.rho = rho
# p_baseline.rho = 0.02979
# p_baseline.rho = 0.1

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
                        damping = 5,
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

sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

p_nash = p_baseline.copy()
p_nash.delta[:,1] = 12
sol, sol_nash = fixed_point_solver(p_nash,x0=p_nash.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='l_R',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=False,
                        damping = 5,
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

sol_baseline.compute_consumption_equivalent_welfare(p_baseline,sol_nash)

A = np.zeros((p_baseline.N,p_baseline.N))
epsilon = 0.01

for i,country in enumerate(p_baseline.countries):
    # print(country)
    p = p_baseline.copy()
    p.delta[i,1] = p_baseline.delta[i,1]*(1+epsilon)
    
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
                            damping = 5,
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
    
    A[i,:] = (sol_c.cons_eq_welfare - sol_baseline.cons_eq_welfare)/(epsilon*p_baseline.delta[i,1])/sol_baseline.cons_eq_welfare

# print(rho,np.linalg.eig(A)[0].min())
# print(np.linalg.eig(A)[0].min())

#%%

def leading_principal_minors(matrix):
    n = min(matrix.shape)
    minors = []

    for k in range(1, n + 1):
        submatrix = matrix[:k, :k]
        minor_value = np.linalg.det(submatrix)
        minors.append(minor_value)

    return minors

minors_A = leading_principal_minors(A)
print("Leading Principal Minors of A:", minors_A)


#%%
from scipy.linalg import null_space

# for rcond in np.logspace(-3,-15,13):
#     B = null_space(A,rcond = rcond)
#     print(rcond,B)
B = null_space(A,rcond = 1e-6)[:,0]


#%%

beta = (B/B.sum())*p_baseline.labor.sum()/p_baseline.labor

df['bargaining_power'] = beta
# df = df.sort_values('bargaining_power',ascending=False)

# df.loc[r] = [rho]+beta.tolist()

# print(df)

#%%

from scipy.optimize import minimize

def objective_function(X, A):
    # return np.dot(A, X)
    return np.linalg.norm(np.dot(A, X))

def constraint(X):
    return np.sum(X) - 1

# Initial guess for X
initial_guess = B/B.sum()

# Define the constraint
constraint_definition = {'type': 'eq', 'fun': constraint}

# Define the bounds for X (all elements strictly positive)
bounds = [(0, None)] * A.shape[1]

# # Your matrix A (replace it with your actual matrix)
# A = np.array([[1, 2], [3, 4]])

# Minimize the objective function under the given constraint
result = minimize(objective_function, initial_guess, args=(A,), constraints=constraint_definition, bounds=bounds)

# Extract the optimal vector X
optimal_X = result.x

print("Optimal X:", optimal_X)
print("Minimum value of AX:", result.fun)

beta_min = (optimal_X/optimal_X.sum())*p_baseline.labor.sum()/p_baseline.labor

df['approx bargaining_power'] = beta_min