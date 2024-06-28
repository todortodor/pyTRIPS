#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:21:19 2024

@author: slepot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classes import moments, parameters, var, var_with_entry_costs, dynamic_var
import scienceplots
from adjustText import adjust_text
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
# mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})

save_to_tex_options = dict(position_float='centering',
                             clines='all;index',
                            hrules=True)

baseline = '1300'
variation = 'baseline'

results_path = 'calibration_results_matched_economy/'

if variation == 'baseline':
    run_path = results_path+baseline+'/'
else:
    run_path = f'calibration_results_matched_economy/baseline_{baseline}_variations/{variation}/'

p_baseline = parameters()
p_baseline.load_run(run_path)

m_baseline = moments()
m_baseline.load_run(run_path)

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()

#%%

# fdi = pd.read_csv('data/fdi_longformat_2015.csv').set_index(
fdi = pd.read_csv('data/fdi_longformat_2015_South_imputed.csv').set_index(
    ['Rep_ccode', 'File_ccode']
).rename_axis(
    ['destination', 'origin']
).sort_index(
)

flows = pd.DataFrame(
    index = m_baseline.idx['SPFLOWDOM'],
    columns = ['Patent flows data'],
    data = m_baseline.cc_moments['patent flows'].values.ravel(),
    )

flows['Patent flows model'] = sol_baseline.pflow.ravel()
flows['Trade flows data'] = m_baseline.ccs_moments.xs(1,level=2)['trade'].values.ravel()

flows = flows.reset_index()[(flows.reset_index().origin!='ROW')&(flows.reset_index().destination!='ROW')]

flows['FDI stocks'] = fdi['FileToRep_Stock'].values
flows['FDI flows'] = fdi['FileToRep_Flow'].values

flows = flows[flows.origin!=flows.destination]

flows['Patent flows over trade flows data'] = flows['Patent flows data']/flows['Trade flows data']
flows['Patent flows over trade flows model'] = flows['Patent flows model']/flows['Trade flows data']
flows['FDI flow over trade flows'] = flows['FDI flows']/flows['Trade flows data']
flows['FDI stock over trade flows'] = flows['FDI stocks']/flows['Trade flows data']

deltas = pd.DataFrame(index=pd.Index(p_baseline.countries,name='destination'),
                      columns=['delta'],
                      data=p_baseline.delta[:,1]).reset_index()

etas = pd.DataFrame(index=pd.Index(p_baseline.countries,name='origin'),
                      columns=['eta'],
                      data=p_baseline.eta[:,1]).reset_index()

flows = pd.merge(flows,deltas,on='destination')
flows = pd.merge(flows,etas,on='origin')

#%% Comparing fdi stocks with patent flows
import seaborn as sns

fig,ax = plt.subplots(1,2,figsize = (8,3),dpi=1200)

x = flows['Patent flows over trade flows data'].values.ravel()
y_fdi = flows['FDI stock over trade flows'].values.ravel()

ax[0].scatter(x,
        y_fdi,
        label = 'FDI stock',
        marker='+'
        )

ax[0].set_xlabel('Patent flows (# of patents per Mio.$)')
ax[0].set_ylabel('Ratio to trade flow (Mio.\$ / Mio.\$)')

coef = np.polyfit(x,y_fdi,1)
poly1d_fn = np.poly1d(coef) 

ax[0].plot(np.sort(x), poly1d_fn(np.sort(x)), ls='--',color='grey',label='Affine fit')

ax[0].legend()

x = flows['Patent flows over trade flows data'].values.ravel()
y_fdi = flows['FDI flow over trade flows'].values.ravel()

ax[1].scatter(x,
        y_fdi,
        label = 'FDI flow',
        marker='^',
        color=sns.color_palette()[1])

ax[1].set_xlabel('Patent flows (# of patents per Mio.$)')
# ax[1].set_ylabel('Ratio of FDI flow to trade flow (Mio.\$ / Mio.\$)')

coef = np.polyfit(x,y_fdi,1)
poly1d_fn = np.poly1d(coef) 

ax[1].plot(np.sort(x), poly1d_fn(np.sort(x)), ls='--',color='grey',label='Affine fit')

ax[1].legend()
plt.savefig('../misc/summary.png',format='png')

plt.show()

#%%

import statsmodels.api as sm
from scipy.stats import linregress

# Assuming you have a DataFrame 'flows' with the necessary columns

# Prepare the MultiIndex for the DataFrame columns
arrays = [
    ['Stock', 'Flow', 'Delta and Eta', 'Delta and Eta'],
    ['Value', 'Value', 'Delta', 'Eta']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["Regression", "Variable"])

# Prepare the DataFrame to store the regression results
df = pd.DataFrame(
    columns=index,
    index=pd.Index(['Slope', 'Intercept', 'R value', 'P value', 'Standard error'])
)

# Linear regression for 'Stock'
slope, intercept, r_value, p_value, std_err = linregress(flows['FDI stock over trade flows'], flows['Patent flows over trade flows data'])
df[('Stock', 'Value')] = [slope, intercept, r_value, p_value, std_err]

# Linear regression for 'Flow'
slope, intercept, r_value, p_value, std_err = linregress(flows['FDI flow over trade flows'], flows['Patent flows over trade flows data'])
df[('Flow', 'Value')] = [slope, intercept, r_value, p_value, std_err]

# Multiple linear regression for 'Delta' and 'Eta' using statsmodels
X = flows[['delta', 'eta']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = flows['Patent flows over trade flows data']
model = sm.OLS(y, X).fit()

# Storing the results for delta
df[('Delta and Eta', 'Delta')] = [
    model.params['delta'],  # Slope for delta
    model.params['const'],  # Intercept
    model.rsquared,         # R value
    model.pvalues['delta'], # P value for delta
    model.bse['delta']      # Standard error for delta
]

# Storing the results for eta
df[('Delta and Eta', 'Eta')] = [
    model.params['eta'],    # Slope for eta
    model.params['const'],  # Intercept (same as for delta)
    model.rsquared,         # R value (same as for delta)
    model.pvalues['eta'],   # P value for eta
    model.bse['eta']        # Standard error for eta
]

print(df)

df.to_csv('../misc/fdi_fits.csv')

#%%

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

fig,ax = plt.subplots(1,2,figsize = (8,3),dpi=1200)

agg_flows = flows.groupby('origin').agg({
    'eta': 'mean',
    'FDI flows': 'sum',
    'Patent flows data':'sum',
    'Trade flows data':'sum',
})

agg_flows['Patent flows over trade flows data'] = agg_flows['Patent flows data']/agg_flows['Trade flows data']

# Plotting the relationship
sns.scatterplot(x=agg_flows['eta'], y=agg_flows['Patent flows over trade flows data'],ax=ax[0])
ax[0].set_xlabel('Eta origin')
ax[0].set_ylabel('Patent flows over trade flows origin')
# ax[0].set_title('Scatter plot of Eta vs. Patent flows over trade flows data')
# plt.show()

agg_flows = flows.groupby('origin').agg({
    'delta': 'mean',
    'FDI flows': 'sum',
    'Patent flows data':'sum',
    'Trade flows data':'sum',
})

agg_flows['Patent flows over trade flows data'] = agg_flows['Patent flows data']/agg_flows['Trade flows data']

# Plotting the relationship
sns.scatterplot(x=agg_flows['delta'], y=agg_flows['Patent flows over trade flows data'],ax=ax[1])
ax[1].set_xlabel('Delta destination')
ax[1].set_ylabel('Patent flows over trade flows destination')
# ax[1].title('Scatter plot of Delta vs. Patent flows over trade flows data')

plt.show()


#%%

import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'flows' DataFrame is already available

# Extract the relevant data
x = flows['delta']
y = flows['eta']
z = flows['Patent flows over trade flows data']

# Create a meshgrid for the surface plot
x_surf, y_surf = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
z_surf = np.zeros_like(x_surf)

# Fit a regression model to predict 'Patent flows over trade flows data'
X = np.column_stack((x, y))
X = sm.add_constant(X)  # Add constant term for intercept
model = sm.OLS(z, X).fit()

# Predict the Z values on the meshgrid
Z_pred = model.predict(np.column_stack((np.ones_like(x_surf.ravel()), x_surf.ravel(), y_surf.ravel())))
z_surf = Z_pred.reshape(x_surf.shape)

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(x, y, z, color='red', label='Data points')

# Surface plot of the fitted surface
# ax.plot_surface(x_surf, y_surf, z_surf, color='blue', alpha=0.6)

# Set viewing angle
ax.view_init(elev=10, azim=220)  # Adjust azim to rotate the plot

ax.set_xlabel('Delta')
ax.set_ylabel('Eta')
ax.set_zlabel('Patent flows over trade flows data')
ax.set_title('3D Surface Plot')

plt.show()

#%%

X = flows[['Trade flows data', 'FDI stocks', 'delta', 'eta']]
y = flows['Patent flows data']/flows['Patent flows data'].max()



# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Perform the logistic regression
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Extract the coefficients
coefficients = result.params
p_values = result.pvalues
std_err = result.bse

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Coefficient': coefficients,
    'Standard Error': std_err,
    'P-value': p_values
})

print(results_df)

results_df.to_csv('../misc/logit_reg.csv')