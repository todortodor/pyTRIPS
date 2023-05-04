#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:38:17 2023

@author: slepot
"""

import os
import matplotlib.pylab as pylab
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from classes import moments, parameters
import pandas as pd

legacy_data_path = 'data/'
pflows_path = '/Users/slepot/Documents/taff/datas/PATSTAT/results/'
WDI_data_path = '/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/Calibration data/'

moments_descriptions = pd.read_csv(
    legacy_data_path+'moments_descriptions.csv', sep=';', index_col=0)
scalar_moments = pd.read_csv(
    legacy_data_path+'scalar_moments.csv', index_col=0)


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

p = parameters()
m = moments()
m.load_data()

crosswalk_countries = pd.read_csv(
    '/Users/slepot/Dropbox/TRIPS/simon_version/code/data/crosswalk_countries_OECD.csv')
crosswalk_sectors = pd.read_csv(
    '/Users/slepot/Dropbox/TRIPS/simon_version/code/data/crosswalk_sectors_OECD.csv')

crosswalk_sectors['Code'] = crosswalk_sectors['Code'].str.replace('D', '')
crosswalk_sectors = crosswalk_sectors.set_index('Code')
crosswalk_countries = crosswalk_countries.set_index('country')

WDI_data = pd.read_csv(WDI_data_path+'WDI_10032023_data.csv')
gdp_WLD = WDI_data.loc[(WDI_data['Series Code'] == 'NY.GDP.MKTP.CD') & (
    WDI_data['Country Code'] == 'WLD')]
gdp = WDI_data.loc[WDI_data['Series Code'] == 'NY.GDP.MKTP.CD']
labor_WLD = WDI_data.loc[(WDI_data['Series Code'] == 'SP.POP.1564.TO') & (
    WDI_data['Country Code'] == 'WLD')]
labor = WDI_data.loc[WDI_data['Series Code'] == 'SP.POP.1564.TO']

# year_OECD = 2005
# for year_OECD in [1995,2005,2018]:
# config_dics = [
#     {'year': 1995, 'N': 7},
#     # {'year':1995,'N':13},
#     # {'year':2018,'N':13}
# ]

config_dics = [{
    'year': y, 
    'N': N}
    for y in range(1990,2019) 
    for N in [7,13]
    ]

write = True

for config_dic in config_dics:
    year = config_dic['year']
    year_OECD = np.maximum(config_dic['year'],1995)
    nbr_of_countries = config_dic['N']
    path = f'data_{nbr_of_countries}_countries_{year}/'
    dropbox_path = f'/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/Calibration data/calibration_data_folders/data_{nbr_of_countries}_countries_{year}/'

    try:
        os.mkdir(path)
    except:
        pass
    try:
        os.mkdir(dropbox_path)
    except:
        pass

    iot_OECD = pd.read_csv(
        f'/Users/slepot/Dropbox/Green Logistics/Global Sustainability Index/OECD_ICIO_data/yearly_CSV/datas{year_OECD}/input_output_{year_OECD}.csv')
    output_OECD = pd.read_csv(
        f'/Users/slepot/Dropbox/Green Logistics/Global Sustainability Index/OECD_ICIO_data/yearly_CSV/datas{year_OECD}/output_{year_OECD}.csv')
    consumption_OECD = pd.read_csv(
        f'/Users/slepot/Dropbox/Green Logistics/Global Sustainability Index/OECD_ICIO_data/yearly_CSV/datas{year_OECD}/consumption_{year_OECD}.csv')
    va_OECD = pd.read_csv(
        f'/Users/slepot/Dropbox/Green Logistics/Global Sustainability Index/OECD_ICIO_data/yearly_CSV/datas{year_OECD}/VA_{year_OECD}.csv')

    iot_OECD['row_country'] = iot_OECD['row_country'].map(
        crosswalk_countries['country_code'])
    iot_OECD['col_country'] = iot_OECD['col_country'].map(
        crosswalk_countries['country_code'])
    iot_OECD['row_sector'] = iot_OECD['row_sector'].map(
        crosswalk_sectors['Sectors'])
    iot_OECD['col_sector'] = iot_OECD['col_sector'].map(
        crosswalk_sectors['Sectors'])

    consumption_OECD['row_country'] = consumption_OECD['row_country'].map(
        crosswalk_countries['country_code'])
    consumption_OECD['col_country'] = consumption_OECD['col_country'].map(
        crosswalk_countries['country_code'])
    consumption_OECD['row_sector'] = consumption_OECD['row_sector'].map(
        crosswalk_sectors['Sectors'])

    va_OECD['country'] = va_OECD['country'].map(
        crosswalk_countries['country_code'])
    va_OECD['sector'] = va_OECD['sector'].map(crosswalk_sectors['Sectors'])

    #%% build full trade flows

    iot_OECD = iot_OECD.groupby(['row_country', 'row_sector', 'col_country'])[
        'value'].sum()
    consumption_OECD = consumption_OECD.groupby(
        ['row_country', 'row_sector', 'col_country'])['value'].sum()

    trade_OECD = iot_OECD+consumption_OECD
    trade_OECD = trade_OECD.reset_index()

    #%%
    trade_OECD_reduced = trade_OECD.copy()
    trade_OECD_reduced['row_country'] = np.minimum(
        trade_OECD_reduced['row_country'], nbr_of_countries)
    trade_OECD_reduced['col_country'] = np.minimum(
        trade_OECD_reduced['col_country'], nbr_of_countries)

    trade_OECD_reduced = trade_OECD_reduced.groupby(
        ['row_country', 'row_sector', 'col_country'])['value'].sum()
    trade_OECD_reduced = trade_OECD_reduced.reorder_levels(
        ['row_country', 'col_country', 'row_sector'])
    trade_OECD_reduced.sort_index(inplace=True)

    trade_OECD_reduced.columns = ['trade']

    trade_OECD_reduced.rename_axis(
        ['origin_code', 'destination_code', 'sector'], inplace=True)

    if write:
        trade_OECD_reduced.to_csv(path+'country_country_sector_moments.csv')
        trade_OECD_reduced.to_csv(
            dropbox_path+'country_country_sector_moments.csv')

    beta_s = trade_OECD_reduced.groupby(
        'sector').sum()/trade_OECD_reduced.sum()
    alpha_s = va_OECD.groupby('sector')['value'].sum()/va_OECD['value'].sum()
    sector_moments = pd.DataFrame(index=[0, 1])
    sector_moments['beta'] = beta_s
    sector_moments['alpha'] = alpha_s
    if write:
        sector_moments.to_csv(path+'sector_moments.csv')
        sector_moments.to_csv(dropbox_path+'sector_moments.csv')

    output = trade_OECD_reduced.groupby(
        'origin_code').sum().rename_axis('country')
    expenditure = trade_OECD_reduced.groupby(
        'destination_code').sum().rename_axis('country')

    trade_balance = trade_OECD_reduced.groupby('origin_code').sum().rename_axis('country')\
        - trade_OECD_reduced.groupby('destination_code').sum().rename_axis('country')

    trade_balance_as_ratio_world_output = trade_balance/trade_OECD_reduced.sum()

    country_qty = pd.concat([output, expenditure, trade_balance], axis=1)
    country_qty.columns = ['output', 'expenditure', 'deficit']
    country_qty.rename_axis('country_code', inplace=True)
    
    gdp_year = pd.merge(gdp[
        ['Country Code', f'{year} [YR{year}]']
                            ].rename(
                                columns = {f'{year} [YR{year}]':'gdp'}
    ).set_index(
        'Country Code'
    ).rename_axis(
        'country'
    ),
        crosswalk_countries,
        left_index=True,
        right_index=True
    )
    gdp_year['gdp'] = gdp_year['gdp'].str.replace('..','0')
    gdp_year['gdp'] = gdp_year['gdp'].astype(float)  
    gdp_year['country_code'] = np.minimum(
        gdp_year['country_code'], nbr_of_countries)
    
    gdp_year = gdp_year.groupby(
        'country_code'
    )['gdp'].sum()
    gdp_year.loc[nbr_of_countries] = gdp_WLD[f'{year} [YR{year}]'].astype(float).iloc[0] - gdp_year.loc[1:nbr_of_countries-1].sum()
    
    country_qty['gdp'] = gdp_year
    
    labor_year = pd.merge(labor[
        ['Country Code', f'{year} [YR{year}]']
                            ].rename(
                                columns = {f'{year} [YR{year}]':'labor'}
    ).set_index(
        'Country Code'
    ).rename_axis(
        'country'
    ),
        crosswalk_countries,
        left_index=True,
        right_index=True
    )
    labor_year['labor'] = labor_year['labor'].astype(float)  
    labor_year['country_code'] = np.minimum(
        labor_year['country_code'], nbr_of_countries)
    
    labor_year = labor_year.groupby(
        'country_code'
    )['labor'].sum()
    labor_year.loc[nbr_of_countries] = labor_WLD[f'{year} [YR{year}]'].astype(float).iloc[0] - labor_year.loc[1:nbr_of_countries-1].sum()
    
    country_qty['labor'] = labor_year
    
    country_qty[['price_level', 'rnd_gdp']] = ''
    

    if write:
        country_qty.to_csv(path+'country_moments.csv')
        country_qty.to_csv(dropbox_path+'country_moments.csv')

    if write:
        scalar_moments.to_csv(path+'scalar_moments.csv')
        scalar_moments.to_csv(dropbox_path+'scalar_moments.csv')

    #%% get patent flows

    pflows = pd.read_csv(
        pflows_path+f'{nbr_of_countries}_countries/flows_{nbr_of_countries}_countries_{year}.csv',index_col=[0,1]
        )
    for origin_code in range(1,nbr_of_countries+1):
        for destination_code in range(1,nbr_of_countries+1):
            if (origin_code,destination_code) not in pflows.index:
                pflows.loc[(origin_code,destination_code),'patent flows'] = pflows['patent flows'].min()
                
    pflows.sort_index(inplace=True)
    
    if write:
        pflows.to_csv(path+'country_country_moments.csv')
        pflows.to_csv(dropbox_path+'country_country_moments.csv')

