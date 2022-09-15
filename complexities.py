#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:58:51 2022

@author: simonl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

#%% Number of countries complexity analysis

# n_l = []
# times = []
# for n in range(2,101):
#     sol = fixed_point_solver(parameters(n,s=5),plot_cobweb=False
#                              ,plot_convergence=False)
#     n_l.append(n)
#     times.append(sol.time)

# df = pd.DataFrame([n_l,times]).T
# df.columns = ['number_of_countries', 'solving_time']
# df.to_csv('/Users/simonl/Dropbox/Mac/Documents/taff/pyTRIPS/misc/solving_times_countries.csv',index=False)

df = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/pyTRIPS/misc/solving_times_countries.csv')
n_l = df.number_of_countries.to_list()
times = df.solving_time.to_list()

fit = np.poly1d(np.polyfit(n_l, times, 2))

fig,ax = plt.subplots()    

ax.plot(n_l,times,label='Numerical solving time')
ax.plot(n_l,fit(n_l), label = '2nd deg fit')
ax.set_ylabel('Solving time (s)')
ax.set_xlabel('Number of countries')

plt.title('Linear complexity in N2')

plt.legend()
plt.show()

#%% Number of sectors complexity analysis

# s_l = []
# times = []
# for s in range(2,101):
#     sol = fixed_point_solver(parameters(n=10,s=s),plot_cobweb=False
#                               ,plot_convergence=False)
#     s_l.append(s)
#     times.append(sol.time)

# df = pd.DataFrame([s_l,times]).T
# df.columns = ['number_of_sectors', 'solving_time']
# df.to_csv('/Users/simonl/Dropbox/Mac/Documents/taff/pyTRIPS/misc/solving_times_sectors.csv',index=False)

df = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/pyTRIPS/misc/solving_times_sectors.csv')
s_l = df.number_of_sectors.to_list()
times = df.solving_time.to_list()

def n_ln_n(t,a,b,c):
    return a+b*t*np.log(c*t)
fit_coeffs = optimize.curve_fit(f = n_ln_n, xdata = s_l, ydata = times)

fig,ax = plt.subplots()    

ax.plot(s_l,times,label='Numerical solving time')
ax.plot(s_l,[n_ln_n(s,fit_coeffs[0][0],fit_coeffs[0][1],fit_coeffs[0][2]) for s in s_l], label = 'S*ln(S) fit',ls='--')
ax.set_ylabel('Solving time (s)')
ax.set_xlabel('Number of sectors')

plt.title('S logS complexity')

plt.legend()
plt.show()