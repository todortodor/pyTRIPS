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
final_pat_fees = pd.read_csv(
    legacy_data_path+'final_pat_fees.csv', index_col=0)

na_values = ["",
             "#N/A",
             "#N/A N/A",
             "#NA",
             # "-1.#IND",
             "-1.#QNAN",
             "-NaN",
             "-nan",
             # "1.#IND",
             "1.#QNAN",
             "<NA>",
             "N/A",
             #"NA",
             "NULL",
             "NaN",
             "n/a",
             "nan",
             "null",
             ".."]

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

# config_dics = [
#     {'year': 2018, 'N': 13},
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
    year_OECD = np.maximum(config_dic['year'], 1995)
    year_rnd_gdp = np.maximum(config_dic['year'], 1996)
    nbr_of_countries = config_dic['N']
    path = f'data/data_{nbr_of_countries}_countries_{year}/'
    dropbox_path = f'/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/Calibration data/calibration_data_folders/data_{nbr_of_countries}_countries_{year}/'

    crosswalk_countries = pd.read_csv(
        '/Users/slepot/Dropbox/TRIPS/simon_version/code/data/countries_wdi.csv')
    crosswalk_sectors = pd.read_csv(
        '/Users/slepot/Dropbox/TRIPS/simon_version/code/data/crosswalk_sectors_OECD.csv')

    crosswalk_sectors['Code'] = crosswalk_sectors['Code'].str.replace('D', '')
    crosswalk_sectors = crosswalk_sectors.set_index('Code')
    crosswalk_countries = crosswalk_countries[
        ['countrycode', f'ccode{nbr_of_countries}']
    ].dropna().set_index('countrycode').rename_axis('country').rename(columns={f'ccode{nbr_of_countries}': 'country_code'})
    crosswalk_countries['country_code'] = np.minimum(crosswalk_countries['country_code'],nbr_of_countries)

    gdp_deflator = pd.read_csv(WDI_data_path+'WDI_gdp_deflator.csv',
                               keep_default_na=False,
                               na_values=na_values,
                               skiprows=4).set_index('Country Code')

    WDI_data = pd.read_csv(WDI_data_path+'WDI_10032023_data.csv',
                           keep_default_na=False,
                           na_values=na_values)
    gdp_WLD = WDI_data.loc[(WDI_data['Series Code'] == 'NY.GDP.MKTP.CD') & (
        WDI_data['Country Code'] == 'WLD')]
    gdp = WDI_data.loc[WDI_data['Series Code'] == 'NY.GDP.MKTP.CD']
    labor_WLD = WDI_data.loc[(WDI_data['Series Code'] == 'SP.POP.1564.TO') & (
        WDI_data['Country Code'] == 'WLD')]
    labor = WDI_data.loc[WDI_data['Series Code'] == 'SP.POP.1564.TO']

    rnd_gdp = WDI_data.loc[WDI_data['Series Code'].isin(
        ['GB.XPD.RSDV.GD.ZS', 'NY.GDP.MKTP.CD'])]
    rnd_gdp = rnd_gdp.melt(
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        var_name='year',
    )
    rnd_gdp['year'] = rnd_gdp['year'].str[:5].astype(int)
    rnd_gdp['Series Code'] = rnd_gdp['Series Code'].str.replace(
        'GB.XPD.RSDV.GD.ZS', 'rnd_gdp'
    ).str.replace(
        'NY.GDP.MKTP.CD', 'gdp'
    )
    rnd_gdp = rnd_gdp[['Country Code', 'Series Code', 'year', 'value']].pivot(
        columns='Series Code',
        index=['Country Code', 'year'],
        values='value'
    )

    rnd_gdp['rnd_gdp'] = rnd_gdp['rnd_gdp']/100
    rnd_gdp['rnd'] = rnd_gdp['gdp']*rnd_gdp['rnd_gdp']
    # rnd_gdp = rnd_gdp.dropna(subset=['rnd'])

    rnd_gdp = pd.merge(rnd_gdp.reset_index().set_index('Country Code').rename_axis('country'),
                       crosswalk_countries,
                       left_index=True,
                       right_index=True,
                       how='left'
                       ).reset_index().set_index(['country', 'year'])
    
    price_levels = WDI_data.loc[WDI_data['Series Code'].isin(
        ['PA.NUS.PPPC.RF', 'NY.GDP.MKTP.CD'])]
    price_levels = price_levels.melt(
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        var_name='year',
    )
    price_levels['year'] = price_levels['year'].str[:5].astype(int)
    price_levels['Series Code'] = price_levels['Series Code'].str.replace(
        'PA.NUS.PPPC.RF', 'price_level'
    ).str.replace(
        'NY.GDP.MKTP.CD', 'gdp'
    )
    price_levels = price_levels[['Country Code', 'Series Code', 'year', 'value']].pivot(
        columns='Series Code',
        index=['Country Code', 'year'],
        values='value'
    )
    price_levels = pd.merge(price_levels.reset_index().set_index('Country Code').rename_axis('country'),
                       crosswalk_countries,
                       left_index=True,
                       right_index=True,
                       how='left'
                       ).reset_index().set_index(['country', 'year'])
    
    if write:
        try:
            os.mkdir(path)
        except:
            pass
        try:
            os.mkdir(dropbox_path)
        except:
            pass

    if write:
        moments_descriptions.to_csv(path+'moments_descriptions.csv', sep=';')
        moments_descriptions.to_csv(
            dropbox_path+'moments_descriptions.csv', sep=';')

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

    # build full trade flows

    iot_OECD = iot_OECD.groupby(['row_country', 'row_sector', 'col_country'])[
        'value'].sum()
    consumption_OECD = consumption_OECD.groupby(
        ['row_country', 'row_sector', 'col_country'])['value'].sum()

    trade_OECD = iot_OECD+consumption_OECD
    trade_OECD = trade_OECD.reset_index()

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
        columns={f'{year} [YR{year}]': 'gdp'}
    ).set_index(
        'Country Code'
    ).rename_axis(
        'country'
    ),
        crosswalk_countries,
        left_index=True,
        right_index=True
    )
    # gdp_year['gdp'] = gdp_year['gdp'].str.replace('..','0')
    gdp_year['gdp'] = gdp_year['gdp'].astype(float)
    gdp_year['country_code'] = np.minimum(
        gdp_year['country_code'], nbr_of_countries)

    gdp_year = gdp_year.groupby(
        'country_code'
    )['gdp'].sum()
    gdp_year.loc[nbr_of_countries] = gdp_WLD[f'{year} [YR{year}]'].astype(
        float).iloc[0] - gdp_year.loc[1:nbr_of_countries-1].sum()

    country_qty['gdp'] = gdp_year

    labor_year = pd.merge(labor[
        ['Country Code', f'{year} [YR{year}]']
    ].rename(
        columns={f'{year} [YR{year}]': 'labor'}
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
    labor_year.loc[nbr_of_countries] = labor_WLD[f'{year} [YR{year}]'].astype(
        float).iloc[0] - labor_year.loc[1:nbr_of_countries-1].sum()

    country_qty['labor'] = labor_year

    for country_code in range(1, nbr_of_countries+1):
        if country_code == 2:  # EUR
            temp = rnd_gdp.loc[
                rnd_gdp.country_code == country_code
            ].copy()
            
            countries_with_rnd_data = temp.dropna().xs(
                year_rnd_gdp,
                level=1
                )['rnd_gdp'].index.get_level_values(0)
            
            country_qty.loc[country_code, 'rnd_gdp'] = (
                temp.xs(
                    year_rnd_gdp,
                    level=1).loc[countries_with_rnd_data]['rnd_gdp']*temp.xs(
                    year,
                    level=1
                ).loc[countries_with_rnd_data]['gdp']
            ).sum()/temp.xs(
                year,
                level=1
            ).loc[countries_with_rnd_data]['gdp'].sum()
                
        elif country_code == nbr_of_countries: # ROW
            world_rnd = rnd_gdp.loc[('WLD',year_rnd_gdp),'rnd_gdp']*rnd_gdp.loc[('WLD',year),'gdp']
            world_gdp = rnd_gdp.loc[('WLD',year),'gdp']
            row_gdp = world_gdp-country_qty.loc[[i for i in range(1, nbr_of_countries)],'gdp'].sum()
            row_rnd = world_rnd-(country_qty['rnd_gdp']*country_qty['gdp']).sum()
            country_qty.loc[country_code, 'rnd_gdp'] = row_rnd/row_gdp
        
        else:
            years_with_data = np.array(rnd_gdp.loc[
                rnd_gdp.country_code == country_code
                ].dropna().index.get_level_values('year'))
            best_year_with_data = years_with_data[np.argmin(np.abs(
                year - years_with_data
                ))]
            country_qty.loc[country_code, 'rnd_gdp'] = rnd_gdp.loc[
                rnd_gdp.country_code == country_code
                ].xs(
                    best_year_with_data,level=1
                )['rnd_gdp'].iloc[0]
    
    price_levels_year = price_levels.xs(year,level=1).dropna()
    price_levels_year['gdp_times_price'] = price_levels_year['gdp']*price_levels_year['price_level']
    price_levels_year = price_levels_year.groupby('country_code').sum()
    
    country_qty['price_level'] = price_levels_year['gdp_times_price']/price_levels_year['gdp']

    if write:
        country_qty.to_csv(path+'country_moments.csv')
        country_qty.to_csv(dropbox_path+'country_moments.csv')

    #%% get patent flows

    pflows = pd.read_csv(
        pflows_path+f'{nbr_of_countries}_countries/flows_{nbr_of_countries}_countries_{year}.csv', index_col=[0, 1]
    )
    for origin_code in range(1, nbr_of_countries+1):
        for destination_code in range(1, nbr_of_countries+1):
            if (origin_code, destination_code) not in pflows.index:
                pflows.loc[(origin_code, destination_code),
                           'patent flows'] = pflows['patent flows'].min()

    pflows.sort_index(inplace=True)
    
    # correct Indian patent flows with wipo data and deflate it
    indian_flows = pd.read_csv(pflows_path+f'patent_flows_from_wipo/unscaled_IN_{nbr_of_countries}_countries.csv'
                               ).set_index(['year', 'origin_code']).loc[year]

    for factor_type in ["app_per_family",
                        "first_applicant",
                        "sector_filter"]:
        indian_flows['patent flows'] = indian_flows[
            'patent flows'
        ]/pd.read_csv(
            pflows_path+f'factors/{factor_type}_{nbr_of_countries}_countries.csv'
        ).set_index(
            ['year', 'origin_code']
        ).loc[year]['factor']
            
    for origin in indian_flows.index:
        pflows.loc[(origin,6),'patent flows'] = indian_flows.loc[origin,'patent flows']

    if write:
        pflows.to_csv(path+'country_country_moments.csv')
        pflows.to_csv(dropbox_path+'country_country_moments.csv')

    scalar_moments.loc['UUPCOST', 'value'] = pflows.loc[(1, 1), 'patent flows']*final_pat_fees.loc[1, 'fee']\
        * gdp_deflator.loc['USA', str(year)]/gdp_deflator.loc['USA', '2005']/1e12

    scalar_moments.loc['JUPCOST', 'value'] = pflows.loc[(1, 3), 'patent flows']*final_pat_fees.loc[1, 'fee']\
        * gdp_deflator.loc['USA', str(year)]/gdp_deflator.loc['USA', '2005']/1e12

    if write:
        scalar_moments.to_csv(path+'scalar_moments.csv')
        scalar_moments.to_csv(dropbox_path+'scalar_moments.csv')