#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:52:12 2023

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from solver_funcs import find_nash_eq
# import seaborn as sns
from classes import moments, parameters, var
from solver_funcs import fixed_point_solver, find_nash_eq, dyn_fixed_point_solver
# from random import random
from tqdm import tqdm
import seaborn as sns

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

p_baseline = parameters(n=7,s=2)
# p_baseline.load_data('calibration_results_matched_economy/baseline_402_variations/17.1.1/')
p_baseline.load_data('calibration_results_matched_economy/baseline_501_variations/1.0/')

lb_delta=0.01
ub_delta=10

# p_baseline.delta[:,1] = ub_delta
# p_baseline.delta[0,1] = lb_delta

solver_options = dict(cobweb_anim=False,tol =1e-14,
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
                        damping_post_acceleration=5)

sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                context = 'counterfactual',
                        **solver_options
                        )

sol_baseline.scale_P(p_baseline)
sol_baseline.compute_price_indices(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)    

deltas = np.logspace(np.log(lb_delta)/np.log(10),np.log(ub_delta)/np.log(10),31)

df = pd.DataFrame(index = deltas, 
                  columns=p_baseline.countries)

for c in p_baseline.countries:
# for c in ['CHN', 'BRA', 'IND', 'ROW']:
# dyn_sols = {}
    
# c = 'USA'
    c_index = p_baseline.countries.index(c)
    for delt in deltas:
        p = p_baseline.copy()
        p.delta[:,1] = ub_delta
        p.delta[0,1] = lb_delta
        p.delta[c_index,1] = delt
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline, Nt=23,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        df.loc[delt,c] = dyn_sol_c.cons_eq_welfare[c_index]
        print(df)
        # dyn_sols[delt] = dyn_sol_c

#%%
fig, ax = plt.subplots(figsize = (14,10))

x=deltas

for c in p_baseline.countries:
    ax.plot(x,df[c],label=c)
ax.legend()
plt.xscale('log')
plt.show()

#%%
from matplotlib import cm
# from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(15,10))

Xflat = np.array(list(dyn_sols.keys()))
# Yflat = dyn_sols[list(dyn_sols.keys())[0]].t_real
Yflat = np.linspace(0,dyn_sols[list(dyn_sols.keys())[0]].t_inf,1001)
X, Y = np.meshgrid(Xflat, Yflat)
Z = np.zeros((Yflat.size,Xflat.size))
for i,dyn_sol in enumerate(dyn_sols.values()):
    
    Z[:,i] = np.polyval(np.polyfit(dyn_sol.t_real,
                                   dyn_sol.ratios_of_consumption_levels_change_not_normalized[0,:]*np.exp(-dyn_sol.sol_init.g*dyn_sol.t_real),
                dyn_sol.Nt),np.linspace(0,dyn_sol.t_inf,1001))
# Xflat = np.arange(-5, 5, 0.25)
# Yflat = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(Xflat, Yflat)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# def 


# Plot the surface.
surf = ax.plot_surface(np.log10(X), Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# ax.set_xticks(np.log10(deltas))
# ax.set_xticklabels([str(round(delt,3)) for delt in deltas])
ax.set_xticks([])
# ax.set_xticklabels([str(round(delt,3)) for delt in deltas])
ax.set_xlabel('delta US')
ax.set_ylabel('Time')

plt.title('Ratio of consumptions change not normalized')

# ax.set_xticks(np.logspace(deltas))
# ax.set_xticklabels([str(round(delt,3)) for delt in deltas])
# ax.set_xlabel('delta_US')
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
