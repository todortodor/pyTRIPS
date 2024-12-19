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
p_baseline.load_data('data_smooth_5_years/data_12_countries_2015/')

m_baseline = moments()
m_baseline.load_run(run_path)
m_baseline.load_data('data_smooth_5_years/data_12_countries_2015/')

sol_baseline = var.var_from_vector(p_baseline.guess, p_baseline, compute=True, context = 'counterfactual')
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline)

m_baseline.compute_moments(sol_baseline,p_baseline)
m_baseline.compute_moments_deviations()


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

flows_bu=flows.copy()

flows['ln_Patent_flows'] = np.log(flows['Patent flows data'])
flows['ln_Trade_flows'] = np.log(flows['Trade flows data'])
flows['ln_FDI_stock'] = np.log(flows['FDI stocks'])

stats = flows[flows['FDI stocks']>0][['Patent flows data', 'Trade flows data',
       'FDI stocks', 'Patent flows over trade flows data',
       'FDI stock over trade flows','ln_Patent_flows',
       'ln_Trade_flows','ln_FDI_stock']].describe()

# stats.round(2).to_csv('../misc/stats.csv')

#%% without fixed effects

import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Prepare the MultiIndex for the DataFrame columns
arrays = [
    ['Log Regression'],
    ['Value']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["Regression", "Variable"])

# Prepare the DataFrame to store the regression results
df = pd.DataFrame(
    columns=index,
    index=pd.Index(['Slope_ln_Trade_flows', 
                    'Slope_ln_FDI_stock', 
                    'Intercept', 
                    'R value', 
                    'P value_ln_Trade_flows', 
                    'P value_ln_FDI_stock', 
                    'Standard error_ln_Trade_flows', 
                    'Standard error_ln_FDI_stock',
                    # 'Standard deviation_ln_Trade_flows', 
                    # 'Standard deviation_ln_FDI_stock',
                    ])
)

# OLS regression of ln(Patent flows) on ln(Trade flows), ln(FDI stock), and a constant
flows['ln_Patent_flows'] = np.log(flows['Patent flows data'])
flows['ln_Trade_flows'] = np.log(flows['Trade flows data'])
flows['ln_FDI_stock'] = np.log(flows['FDI stocks'])

X = flows[flows['FDI stocks']>0][['ln_Trade_flows', 'ln_FDI_stock']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = flows[flows['FDI stocks']>0]['ln_Patent_flows']
model = sm.OLS(y, X).fit()

# Storing the results for the log regression
df[('Log Regression', 'Value')] = [
    model.params['ln_Trade_flows'],    # Slope for ln(Trade flows)
    model.params['ln_FDI_stock'],      # Slope for ln(FDI stock)
    model.params['const'],             # Intercept
    model.rsquared,                    # R value
    model.pvalues['ln_Trade_flows'],   # P value for ln(Trade flows)
    model.pvalues['ln_FDI_stock'],     # P value for ln(FDI stock)
    model.bse['ln_Trade_flows'],       # Standard error for ln(Trade flows)
    model.bse['ln_FDI_stock'],          # Standard error for ln(FDI stock)
    # model.bse['ln_Trade_flows']*y.shape[0]**(1/2),       # Standard deviation for ln(Trade flows)
    # model.bse['ln_FDI_stock']*y.shape[0]**(1/2)         # Standard deviation for ln(FDI stock)
]

# # Display the results
# import ace_tools as tools

# tools.display_dataframe_to_user(name="Regression Results", dataframe=df)

# print(df)

# df.to_csv('../misc/fdi_fits.csv')

print('Without fixed effects')

print('the coefficient for trade flows is :',df.loc['Slope_ln_Trade_flows'].iloc[0])
print('the coefficient for FDI stock is :',df.loc['Slope_ln_FDI_stock'].iloc[0])
print('the standard deviation for trade flows is :',stats.loc['std','ln_Trade_flows'])
print('the standard deviation for FDI stock is :',stats.loc['std','ln_FDI_stock'])
print('the coefficient times the standard deviation for trade flows is :',df.loc['Slope_ln_Trade_flows'].iloc[0]*stats.loc['std','ln_Trade_flows'])
print('the coefficient times the standard deviation for FDI stock is :',df.loc['Slope_ln_FDI_stock'].iloc[0]*stats.loc['std','ln_FDI_stock'])

#%%  with fixed effects

flows=flows_bu.copy()
arrays = [
    ['Log Regression'],
    ['Value']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["Regression", "Variable"])

# OLS regression of ln(Patent flows) on ln(Trade flows), ln(FDI stock), and a constant
flows['ln_Patent_flows'] = np.log(flows['Patent flows data'])
flows['ln_Trade_flows'] = np.log(flows['Trade flows data'])
flows['ln_FDI_stock'] = np.log(flows['FDI stocks'])

# Generate dummy variables for 'destination' and 'origin'
flows = pd.get_dummies(flows, columns=['destination', 'origin'], drop_first=True, dtype=float)

# OLS regression of ln(Patent flows) on ln(Trade flows), ln(FDI stock), fixed effects for destination and origin, and a constant
X = flows[flows['FDI stocks']>0][['ln_Trade_flows', 'ln_FDI_stock'] + 
                                 [col for col in flows.columns if col.startswith('destination_') or col.startswith('origin_')]]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = flows[flows['FDI stocks']>0]['ln_Patent_flows']
model = sm.OLS(y, X).fit()

# Prepare the DataFrame to store the regression results
df = pd.DataFrame(
    columns=index,
    index=pd.Index(['Slope_ln_Trade_flows', 
                    'Slope_ln_FDI_stock', 
                    'Intercept', 
                    'R value', 
                    'P value_ln_Trade_flows', 
                    'P value_ln_FDI_stock', 
                    'Standard error_ln_Trade_flows', 
                    'Standard error_ln_FDI_stock',
                    'Standard deviation_ln_Trade_flows', 
                    'Standard deviation_ln_FDI_stock',
                    ])
)

# Storing the results for the log regression
df[('Log Regression', 'Value')] = [
    model.params['ln_Trade_flows'],    # Slope for ln(Trade flows)
    model.params['ln_FDI_stock'],      # Slope for ln(FDI stock)
    model.params['const'],             # Intercept
    model.rsquared,                    # R value
    model.pvalues['ln_Trade_flows'],   # P value for ln(Trade flows)
    model.pvalues['ln_FDI_stock'],     # P value for ln(FDI stock)
    model.bse['ln_Trade_flows'],       # Standard error for ln(Trade flows)
    model.bse['ln_FDI_stock'],         # Standard error for ln(FDI stock)
    model.bse['ln_Trade_flows']*y.shape[0]**(1/2),       # Standard deviation for ln(Trade flows)
    model.bse['ln_FDI_stock']*y.shape[0]**(1/2)         # Standard deviation for ln(FDI stock)

]

# # Prepare the MultiIndex for the DataFrame columns
# arrays = [
#     ['Log Regression', 'Log Regression', 'Log Regression'],
#     ['Slope', 'P value', 'Standard error']
# ]
# tuples = list(zip(*arrays))
# index = pd.MultiIndex.from_tuples(tuples, names=["Regression", "Variable"])

# # Prepare the DataFrame to store the regression results
# variables = ['ln_Trade_flows', 'ln_FDI_stock', 'const'] + [col for col in flows.columns if col.startswith('destination_') or col.startswith('origin_')]
# df = pd.DataFrame(index=variables, columns=index)

# # Add R-squared value as a separate row
# df.loc['R squared', ('Log Regression', 'Slope')] = model.rsquared

# # Storing the results for the log regression
# for var in variables:
#     df.loc[var, ('Log Regression', 'Slope')] = model.params[var]
#     df.loc[var, ('Log Regression', 'P value')] = model.pvalues[var]
#     df.loc[var, ('Log Regression', 'Standard error')] = model.bse[var]

# print(df)

# df.to_csv('../misc/fdi_fits_with_fixed_effects_dest_orig.csv')

print('With fixed effects')

print('the coefficient for trade flows is :',df.loc['Slope_ln_Trade_flows'].iloc[0])
print('the coefficient for FDI stock is :',df.loc['Slope_ln_FDI_stock'].iloc[0])
print('the standard deviation for trade flows is :',stats.loc['std','ln_Trade_flows'])
print('the standard deviation for FDI stock is :',stats.loc['std','ln_FDI_stock'])
print('the coefficient times the standard deviation for trade flows is :',df.loc['Slope_ln_Trade_flows'].iloc[0]*stats.loc['std','ln_Trade_flows'])
print('the coefficient times the standard deviation for FDI stock is :',df.loc['Slope_ln_FDI_stock'].iloc[0]*stats.loc['std','ln_FDI_stock'])

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
# plt.savefig('../misc/summary.png',format='png')

plt.show()

#%% Comparing fdi stocks with patent flows
import seaborn as sns
from scipy.stats import linregress

fig,ax = plt.subplots()

x = flows['Patent flows over trade flows data'].values.ravel()
y_fdi = flows['FDI stock over trade flows'].values.ravel()

ax.scatter(x,
        y_fdi,
        label = 'FDI stock',
        marker='+'
        )

ax.set_xlabel('Ratio patent flow to trade flow\n(#patents / Mio.$)')
ax.set_ylabel('Ratio FDI stock to trade flow\n(Mio.\$ / Mio.\$)')

coef = np.polyfit(x,y_fdi,1)
poly1d_fn = np.poly1d(coef) 

ax.plot(np.sort(x), poly1d_fn(np.sort(x)), ls='--',color='grey',label='Affine fit')

slope, intercept, r_value, p_value, std_err = linregress(x, 
                                                         y_fdi)

df = pd.DataFrame(
    index=pd.Index(['Slope', 
                    'Intercept', 
                    'R value', 
                    'P value', 
                    'Standard error'])
)

df['Regression'] = [slope, intercept, r_value, p_value, std_err]

# ax.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('../misc/fi_stock_to_patent_flow.png',format='png')
df.to_csv('../misc/graph_fit.csv')
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
X = flows[['FDI stocks', 'eta']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = flows['Patent flows data']
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