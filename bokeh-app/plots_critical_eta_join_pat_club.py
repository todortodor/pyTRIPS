#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 00:30:46 2023

@author: slepot
"""

from classes import moments, parameters, var
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bokeh.palettes import Category10, Dark2
import matplotlib
import scienceplots

Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])

markers = {'pop_weighted':'o',
      'negishi':'^'}
label_coop = {'pop_weighted':'Equal',
      'negishi':'Negishi'}

run_countries = []

p_baseline = parameters()
# p.load_run('calibration_results_matched_economy/1020/')
p_baseline.load_run('calibration_results_matched_economy/1030/')

plt.style.use(['science','nature','no-latex'])
plt.style.use(['science','no-latex'])
import matplotlib.pyplot as plt 
plt.rcParams.update({"axes.grid" : True, 
                     "grid.color": "grey", 
                     'axes.axisbelow':True,
                     "grid.linewidth": 0.1, 
                     'legend.framealpha':1,
                     'legend.frameon':1,
                     'legend.edgecolor':'white',
                     'figure.dpi':288,
                     })

fig,ax = plt.subplots()

for i, country in enumerate(p_baseline.countries):
    # x = p_baseline.tau[:,i,1].mean()
    # ax.set_xlabel('Average trade cost outward')
    # ax.set_xscale('log')
    
    # x = p_baseline.tau[i,:,1].mean()
    # ax.set_xlabel('Average trade cost inward')
    # ax.set_xscale('log')
    
    # x = (p_baseline.tau[:,i,1]*p_baseline.trade_shares[:,i,1]).sum()/p_baseline.trade_shares[:,i,1].sum()
    # ax.set_xlabel(' Weighted average trade cost outward')
    # ax.set_xscale('log')
    
    # x = (p_baseline.tau[:,i,1]*p_baseline.trade_shares[i,:,1]).sum()/p_baseline.trade_shares[i,:,1].sum()
    # ax.set_xlabel(' Weighted average trade cost inward')
    # ax.set_xscale('log')
    
    # x = p_baseline.T[i,0]
    # ax.set_xlabel('T sector 0')
    # ax.set_xscale('log')
    
    # x = p_baseline.T[i,1]
    # ax.set_xlabel('T sector 1')
    # ax.set_xscale('log')
    
    # x = p_baseline.delta[i,1]
    # ax.set_xlabel('Delta')
    # ax.set_xscale('log')
    
    # x = p_baseline.eta[i,1]
    # ax.set_xlabel('Eta baseline')
    # ax.set_xscale('log')
    
    # x = p_baseline.r_hjort[i]
    # ax.set_xlabel('Hjort factor')
    # ax.set_xscale('log')
    
    
    x = country
    for j,coop in enumerate(['pop_weighted','negishi']):
        # if country != 'USA' or coop == 'negishi':
        try:
            df = pd.read_csv(f'solve_for_eta_to_join_pat_club/baseline_1030/{coop}_{country}.csv')
            # if j == 0:
            #     x = df[f'eta_{country}'].iloc[-1]
            ax.scatter([x],[df[f'eta_{country}'].iloc[-1]],
            # ax.scatter([country],[df[f'eta_{country}'].iloc[-1]],
                        # label = f'{country} {label_coop[coop]}',
                        marker = markers[coop],
                        color = Category18[i])
            ax.errorbar([x],[df[f'eta_{country}'].iloc[-1]], yerr = [np.abs(df[f'eta_{country}'].iloc[-1] - df[f'eta_{country}'].iloc[-2]) + 1e-5])
            run_countries.append(country)
            print(coop,country,df[f'eta_{country}'].iloc[-1])
        except:
            pass
    if country in run_countries:
        print(run_countries)
        ax.scatter([x],[p_baseline.eta[i,1]],
        # ax.scatter([country],[p_baseline.eta[i,1]],
                    label = f'{country}',
                    marker = '*',
                    color = Category18[i])
ax.scatter([],[],marker = 'o', label = 'Equal',color = Category18[0])
ax.scatter([],[],marker = '^', label = 'Negishi',color = Category18[0])
ax.scatter([],[],marker = '*', label = 'Baseline',color = Category18[0])

ax.set_yscale('log')
plt.axhline(y=p_baseline.eta[0,1],color='grey',label='Baseline USA')
ax.set_ylabel('Eta')
plt.legend(loc=[1.02,0])
plt.savefig('../misc/eta_join_pat_club.pdf')
plt.show()