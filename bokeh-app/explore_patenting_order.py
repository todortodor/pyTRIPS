#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:41:28 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
import matplotlib.pylab as pylab
from data_funcs import write_calibration_results
import seaborn as sns
from adjustText import adjust_text
import scienceplots
from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#e8ba02']+list(Dark2[8])
# import matplotlib.pyplot as plt
# plt.rcParams.update(plt.rcParamsDefault)

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (14, 11),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
# sns.set()
# sns.set_context('talk')
# # sns.set_style('whitegrid')
# sns.set(style="whitegrid",
#         font_scale=2,
#         rc={
#     "lines.markersize": 10,
#     "lines.linewidth": 3,
#     }
#     )
plt.style.use(['science', 'nature', 'no-latex'])
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({"axes.grid": True,
                     "grid.color": "grey",
                     'axes.axisbelow': True,
                     "grid.linewidth": 0.1,
                     'legend.framealpha': 1,
                     'legend.frameon': 1,
                     'legend.edgecolor': 'white',
                     'figure.dpi': 288,
                     })
# mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})

save_to_tex_options = dict(position_float='centering',
                           clines='all;index',
                           hrules=True)

# %% load baseline

baseline = '1210'
variation = 'baseline'

data_path = 'data/'
results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'


p_baseline = parameters()
# p_baseline.load_data(run_path)
p_baseline.load_run(run_path)

m_baseline = moments()
# m_baseline.load_data()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(
    p_baseline.guess, p_baseline, compute=True, context='counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline, p_baseline)
m_baseline.compute_moments_deviations()

# PC_model = np.maximum(1,
#                       np.einsum('ci,ni->nic',
#                                 sol_baseline.psi_m_star[...,1],
#                                 sol_baseline.psi_m_star[...,1]
#                                 )
#                       )**(-p_baseline.k)

# A = np.zeros((p_baseline.N, p_baseline.N, p_baseline.N))
PC_model = np.zeros((p_baseline.N, p_baseline.N, p_baseline.N))

c_of_n_i = np.zeros((p_baseline.N, p_baseline.N))

for n, destination in enumerate(p_baseline.countries):
    for i, origin in enumerate(p_baseline.countries):
        for c, origin in enumerate(p_baseline.countries):
                PC_model[n, i, c] = np.maximum(
                    (sol_baseline.psi_m_star[n,i,1]/sol_baseline.psi_m_star[c,i,1]),
                    1
                )**(-p_baseline.k)
                # PC_model[n, i, c] = (A[n, i, c]/sol_baseline.psi_m_star[n,i,1])**(-p_baseline.k)
                # PC_model[n, i, c] = (A[n, i, c])**(-p_baseline.k)

def find_min_strictly_larger_indices(A):
    result = np.ones_like(A)*(-1)
    for i,_ in enumerate(A):
        temp = A.copy()
        if np.min(temp*(1/(A>A[i]))) < np.inf:
            result[i] = np.argmin(temp*(1/(A>A[i])))
        
    return result.astype(int)

for i, origin in enumerate(p_baseline.countries):
    c_of_n_i[:,i] = find_min_strictly_larger_indices(sol_baseline.psi_m_star[:,i,1])
    # c_of_n_i[:,i]
    
# PC_model = np.einsum('nic,ci->nic',
#                      A,
#                      1/sol_baseline.psi_m_star[..., 1]
#                      )**(-p_baseline.k)

P_model = pd.DataFrame(index = pd.MultiIndex.from_product([p_baseline.countries,p_baseline.countries],
                                 names = ['destination', 'origin']),
                       columns = ['unconditional probability model'],
                       data = sol_baseline.psi_m_star[...,1].ravel()**(-p_baseline.k)
                       )

PC_data = pd.read_csv(
    f'/Users/slepot/Documents/taff/datas/PATSTAT/patenting_order/{p_baseline.N}_countries/year_2015.csv',
    index_col = ['destination_code', 'origin_code', 'condition_code']
)[['probability']]

df = pd.DataFrame(
    index=pd.MultiIndex.from_product([np.arange(1,13),np.arange(1,13),np.arange(1,13)],
                                     names = ['destination_code', 'origin_code', 'condition_code']),
    columns=['probability model'],
    data=PC_model.ravel()
    )

df = df.join(PC_data)

df.index = pd.MultiIndex.from_product([p_baseline.countries,p_baseline.countries,p_baseline.countries],
                                 names = ['destination', 'origin', 'condition'])

sorting_order = {country:i for i,country in enumerate(p_baseline.countries)}
def sort_func(series):
    return series.replace(sorting_order)

df = pd.merge(df.reset_index(),P_model.reset_index(),on=['destination', 'origin']
              ).sort_values(['condition','destination', 'origin'],key=sort_func,
                  ).set_index(['destination', 'origin', 'condition'])

df.columns = ['conditional probability model', 'conditional probability data','unconditional probability model']

data = df.reset_index().copy()
data = data.query('destination!=condition')
data = data.query('destination!="IND"')
data = data.query('origin!="IND"')
data = data.query('condition!="IND"')
data = data.loc[data["conditional probability model"] == 1]

print(data.mean())

# df = df.fillna(0)
# test = df.loc[:,:,'USA']

#%% Model order of patenting

fig,ax=plt.subplots(p_baseline.N,figsize = (10,12),constrained_layout=True)

for origin in p_baseline.countries:
    # origin = 'CHN'
    origin_index = p_baseline.countries.index(origin)
    
    df_temp = pd.DataFrame()
    
    df_temp['date'] = sol_baseline.psi_m_star[:,origin_index,1]
    df_temp['event'] = p_baseline.countries
    
    levels = np.tile(
        np.array([-5, 5, -3, 3, -1, 1])*0.1,
        int(np.ceil(len(df_temp)/6))
    )[:len(df_temp)]
    
    # fig, ax = plt.subplots(figsize=(12.8, 4), constrained_layout=True);
    ax[origin_index].set_title(origin,color='r',fontsize=10,loc='left')
    
    # ax[origin_index].set_ylabel(origin)
    
    ax[origin_index].vlines(df_temp['date'], 0, levels, color="tab:red");  # The vertical stems.
    ax[origin_index].plot(   # Baseline and markers on it.
        df_temp['date'],
        np.zeros_like(df_temp['date']),
        "-o",
        color="k",
        markerfacecolor="w"
    );
    
    # annotate lines
    for d, l, r in zip(df_temp['date'], levels, df_temp['event']):
        if r == origin:
            color = 'r'
        else:
            color='k'
        ax[origin_index].annotate(
            r,
            xy=(d, l),
            # xytext=(-3, np.sign(l)*1.5),
            xytext=(0, 0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom" if l > 0 else "top",
            color=color
        );
    
    # ax[origin_index].annotate(origin,xy=(1,0.5),color='r')
    
    # format xaxis with 4 month intervals
    
    
    # remove y axis and spines
    ax[origin_index].yaxis.set_visible(False);
    ax[origin_index].yaxis.set_visible(False);
    # ax.spines["left"].set_visible(False)
    ax[origin_index].spines["top"].set_visible(False)
    ax[origin_index].spines["right"].set_visible(False)    
    # ax[origin_index].margins(y=0.1);
    plt.setp(ax[origin_index].get_xticklabels(), rotation=30, ha="right");
    ax[origin_index].set_ylim((-1,1))
    ax[origin_index].set_xlim((1,1e3))
    ax[origin_index].set_xscale('log')
ax[origin_index].set_xlabel('Innovation quality',fontsize=12)
plt.suptitle('Where countries patent first',fontsize= 20)
plt.show()

#%% Model order of patenting

fig,ax=plt.subplots(p_baseline.N,figsize = (10,12),constrained_layout=True)

for origin in p_baseline.countries:
    # origin = 'CHN'
    origin_index = p_baseline.countries.index(origin)
    
    df_temp = pd.DataFrame()
    
    df_temp['date'] = sol_baseline.psi_m_star[:,origin_index,1]**(-p_baseline.k)
    df_temp['event'] = p_baseline.countries
    
    levels = np.tile(
        np.array([-5, 5, -3, 3, -1, 1])*0.1,
        int(np.ceil(len(df_temp)/6))
    )[:len(df_temp)]
    
    # fig, ax = plt.subplots(figsize=(12.8, 4), constrained_layout=True);
    ax[origin_index].set_title(origin,color='r',fontsize=10,loc='left')
    
    # ax[origin_index].set_ylabel(origin)
    
    ax[origin_index].vlines(df_temp['date'], 0, levels, color="tab:red");  # The vertical stems.
    ax[origin_index].plot(   # Baseline and markers on it.
        df_temp['date'],
        np.zeros_like(df_temp['date']),
        "-o",
        color="k",
        markerfacecolor="w"
    );
    
    # annotate lines
    for d, l, r in zip(df_temp['date'], levels, df_temp['event']):
        if r == origin:
            color = 'r'
        else:
            color='k'
        ax[origin_index].annotate(
            r,
            xy=(d, l),
            # xytext=(-3, np.sign(l)*1.5),
            xytext=(0, 0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom" if l > 0 else "top",
            color=color
        );
    
    # ax[origin_index].annotate(origin,xy=(1,0.5),color='r')
    
    # format xaxis with 4 month intervals
    
    
    # remove y axis and spines
    ax[origin_index].yaxis.set_visible(False);
    ax[origin_index].yaxis.set_visible(False);
    # ax.spines["left"].set_visible(False)
    ax[origin_index].spines["top"].set_visible(False)
    ax[origin_index].spines["right"].set_visible(False)    
    # ax[origin_index].margins(y=0.1);
    plt.setp(ax[origin_index].get_xticklabels(), rotation=30, ha="right");
    ax[origin_index].set_ylim((-1,1))
    ax[origin_index].set_xlim((1e-4,1))
    ax[origin_index].set_xscale('log')
ax[origin_index].set_xlabel('Patenting %',fontsize=12)
plt.suptitle('Probabilities of patenting',fontsize= 20)
plt.show()


#%% correlation model vs data

from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#0e6655','#e8ba02']+list(Dark2[8])

data = df.reset_index().copy()
data = data.query('destination!=condition')
data = data.query('origin==condition')

fig,ax = plt.subplots()

sns.scatterplot(data=data, x="conditional probability data", 
                y="conditional probability model",
                ax=ax,
                )


ax.set_xlabel('Conditional probability data')
ax.set_ylabel('Conditional probability model')

plt.show()

#%% Model and data with Hue condition

from bokeh.palettes import Category10, Dark2
Category18 = list(Category10[10])+['#0e6655','#e8ba02']+list(Dark2[8])

data = df.reset_index().copy()
data = data.query('destination!=condition')

fig,ax = plt.subplots(2,1,figsize=(4,6))

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')

sns.scatterplot(data=data, x="unconditional probability model", 
                y="conditional probability model", hue="condition",
                ax=ax[0],
                palette=Category18, 
                marker="X",
                # legend=False
                )

ax[0].plot([data["unconditional probability model"].min(),data["unconditional probability model"].max()],
           [data["unconditional probability model"].min(),data["unconditional probability model"].max()],
           ls='--',color='k',lw=0.5)

sns.scatterplot(data=data, x="unconditional probability model", 
                y="conditional probability data", hue="condition",
                ax=ax[1],
                palette=Category18, 
                marker="^",
                # legend=False
                )
ax[1].plot([data["unconditional probability model"].min(),data["unconditional probability model"].max()],
           [data["unconditional probability model"].min(),data["unconditional probability model"].max()],
           ls='--',color='k',lw=0.5)


sns.move_legend(ax[0],(1.02,0))
sns.move_legend(ax[1],(1.02,0))

ax[0].set_ylabel('Conditional probability model')
ax[1].set_ylabel('Conditional probability data')

ax2 = ax[0].twinx()
ax2.scatter([],[],color='k',marker='^',label='Data')
ax2.scatter([],[],color='k',marker='X',label='Model')
ax2.set_yticks([],[])
ax2.legend(loc='lower right')
# ax.set_xlim([1e-3,2])
# ax.set_ylim([1e-4,1])

# ax.set_xlim([0.5,1e3])
# ax.set_ylim([1e-4,100])

plt.show()

#%% Model order of patenting with data annotation

fig,ax=plt.subplots(p_baseline.N,figsize = (10,12),constrained_layout=True)

for i,origin in enumerate(p_baseline.countries):
    # origin = 'CHN'
    origin_index = p_baseline.countries.index(origin)
    
    df_temp = pd.DataFrame(columns=['date','event'])
    
    df_temp['date'] = sol_baseline.psi_m_star[:,origin_index,1]
    for n,destination in enumerate(p_baseline.countries):
        if c_of_n_i[n,i] != -1:
            df_temp.loc[n,'event'] = p_baseline.countries[n]+' '+str(
                    df.loc[(destination,origin,p_baseline.countries[int(c_of_n_i[n,i])]),'conditional probability data'].round(2))
        else:
            df_temp.loc[n,'event'] = p_baseline.countries[n]
    
    levels = np.tile(
        np.array([-5, 5, -3, 3, -1, 1])*0.1,
        int(np.ceil(len(df_temp)/6))
    )[:len(df_temp)]
    
    # fig, ax = plt.subplots(figsize=(12.8, 4), constrained_layout=True);
    ax[origin_index].set_title(origin,color='r',fontsize=10,loc='left')
    
    # ax[origin_index].set_ylabel(origin)
    
    ax[origin_index].vlines(df_temp['date'], 0, levels, color="tab:red");  # The vertical stems.
    ax[origin_index].plot(   # Baseline and markers on it.
        df_temp['date'],
        np.zeros_like(df_temp['date']),
        "-o",
        color="k",
        markerfacecolor="w"
    );
    
    # annotate lines
    for d, l, r in zip(df_temp['date'], levels, df_temp['event']):
        if r[:3 ] == origin:
            color = 'r'
        else:
            color='k'
        ax[origin_index].annotate(
            r,
            xy=(d, l),
            # xytext=(-3, np.sign(l)*1.5),
            xytext=(0, 0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom" if l > 0 else "top",
            color=color
        );
    
    # ax[origin_index].annotate(origin,xy=(1,0.5),color='r')
    
    # format xaxis with 4 month intervals
    
    
    # remove y axis and spines
    ax[origin_index].yaxis.set_visible(False);
    ax[origin_index].yaxis.set_visible(False);
    # ax.spines["left"].set_visible(False)
    ax[origin_index].spines["top"].set_visible(False)
    ax[origin_index].spines["right"].set_visible(False)    
    # ax[origin_index].margins(y=0.1);
    plt.setp(ax[origin_index].get_xticklabels(), rotation=30, ha="right");
    ax[origin_index].set_ylim((-1,1))
    ax[origin_index].set_xlim((1,1e3))
    ax[origin_index].set_xscale('log')
ax[origin_index].set_xlabel('Innovation quality',fontsize=12)
plt.suptitle('Where countries patent first',fontsize= 20)
plt.show()

#%% Model order of patenting with data annotation

fig,ax=plt.subplots(p_baseline.N,figsize = (10,12),constrained_layout=True)

for i,origin in enumerate(p_baseline.countries):
    # origin = 'CHN'
    origin_index = p_baseline.countries.index(origin)
    
    df_temp = pd.DataFrame(columns=['date','event'])
    
    df_temp['date'] = sol_baseline.psi_m_star[:,origin_index,1]
    for n,destination in enumerate(p_baseline.countries):
        if c_of_n_i[n,i] != -1:
            df_temp.loc[n,'event'] = p_baseline.countries[n]+' '+str(
                    df.loc[(destination,origin,p_baseline.countries[int(c_of_n_i[n,i])]),'conditional probability data'].round(2))
        else:
            df_temp.loc[n,'event'] = p_baseline.countries[n]
    
    levels = np.tile(
        np.array([-5, 5, -3, 3, -1, 1])*0.1,
        int(np.ceil(len(df_temp)/6))
    )[:len(df_temp)]
    
    # fig, ax = plt.subplots(figsize=(12.8, 4), constrained_layout=True);
    ax[origin_index].set_title(origin,color='r',fontsize=10,loc='left')
    
    # ax[origin_index].set_ylabel(origin)
    
    ax[origin_index].vlines(df_temp['date'], 0, levels, color="tab:red");  # The vertical stems.
    ax[origin_index].plot(   # Baseline and markers on it.
        df_temp['date'],
        np.zeros_like(df_temp['date']),
        "-o",
        color="k",
        markerfacecolor="w"
    );
    
    # annotate lines
    for d, l, r in zip(df_temp['date'], levels, df_temp['event']):
        if r[:3 ] == origin:
            color = 'r'
        else:
            color='k'
        ax[origin_index].annotate(
            r,
            xy=(d, l),
            # xytext=(-3, np.sign(l)*1.5),
            xytext=(0, 0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom" if l > 0 else "top",
            color=color
        );
    
    # ax[origin_index].annotate(origin,xy=(1,0.5),color='r')
    
    # format xaxis with 4 month intervals
    
    
    # remove y axis and spines
    ax[origin_index].yaxis.set_visible(False);
    ax[origin_index].yaxis.set_visible(False);
    # ax.spines["left"].set_visible(False)
    ax[origin_index].spines["top"].set_visible(False)
    ax[origin_index].spines["right"].set_visible(False)    
    # ax[origin_index].margins(y=0.1);
    plt.setp(ax[origin_index].get_xticklabels(), rotation=30, ha="right");
    ax[origin_index].set_ylim((-1,1))
    ax[origin_index].set_xlim((1,1e3))
    ax[origin_index].set_xscale('log')
ax[origin_index].set_xlabel('Innovation quality',fontsize=12)
plt.suptitle('Where countries patent first',fontsize= 20)
plt.show()

