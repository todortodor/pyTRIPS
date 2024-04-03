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
# WDI_data_path = '/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/Calibration data/'
WDI_data_path = '/Users/slepot/Dropbox/TRIPS/Calibration data/'
tariff_data_path = 'data/'

moments_descriptions = pd.read_csv(
    legacy_data_path+'moments_descriptions.csv', sep=';', index_col=0)
scalar_moments = pd.read_csv(
    legacy_data_path+'scalar_moments.csv', index_col=0)
final_pat_fees = pd.read_csv(
    legacy_data_path+'final_pat_fees.csv', index_col=0)


exchange_rates = {
    1990: 1.066116,
    1991: 1.066116,
    1992: 1.066116,
    1993: 1.066116,
    1994: 1.066116,
    1995: 1.066116,
    1996: 1.066116,
    1997: 1.066116,
    1998: 1.066116,
    1999: 1.066116,
    2000: 0.923498,
    2001: 0.895969,
    2002: 0.942468,
    2003: 1.134134,
    2004: 1.244143,
    2005: 1.246376,
    2006: 1.256316,
    2007: 1.370412,
    2008: 1.471366,
    2009: 1.39448,
    2010: 1.327386,
    2011: 1.392705,
    2012: 1.285697,
    2013: 1.328464,
    2014: 1.329165,
    2015: 1.109729,
    2016: 1.10656,
    2017: 1.130051,
    2018: 1.181011,
    2019: 1.120129,
    2020: 1.142203,
    2021: 1.18318,
    2022: 1.053783,
    2023: 1.081941
}

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
#     {'year': 2005, 'N': 7},
# ]

# config_dics = [{
#     'year': y,
#     'N': N}
#     for y in range(1990,2019)
#     for N in [7,13]
#     ]
config_dics = [{
    'year': y,
    'N': N}
    for y in range(1990,2019)
    # for y in range(1990,1991)
    # for y in [1992]
    # for y in range(2005,2006)
    # for N in [7,12,13]
    for N in [12]
    ]

write = False
write_tariff = True

for config_dic in config_dics:
    print(config_dic)
    year = config_dic['year']
    year_OECD = np.maximum(config_dic['year'], 1995)
    year_rnd_gdp = np.maximum(config_dic['year'], 1996)
    nbr_of_countries = config_dic['N']
    path = f'data/data_{nbr_of_countries}_countries_{year}/'
    dropbox_path = f'/Users/slepot/Dropbox/TRIPS/Calibration data/calibration_data_folders/data_{nbr_of_countries}_countries_{year}/'
    # dropbox_path = f'/Users/slepot/Library/CloudStorage/Dropbox/TRIPS/Calibration data/calibration_data_folders/data_{nbr_of_countries}_countries_{year}/'
    
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
    
    crosswalk_countries = pd.read_csv(
        'data/countries_wdi.csv')
    crosswalk_sectors = pd.read_csv(
        'data/crosswalk_sectors_OECD.csv')

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

    iot_OECD = pd.read_csv(
        f'/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas{year_OECD}/input_output_{year_OECD}.csv')
    output_OECD = pd.read_csv(
        f'/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas{year_OECD}/output_{year_OECD}.csv')
    consumption_OECD = pd.read_csv(
        f'/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas{year_OECD}/consumption_{year_OECD}.csv')
    va_OECD = pd.read_csv(
        f'/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas{year_OECD}/VA_{year_OECD}.csv')
    
    trade_OECD_full = (iot_OECD.groupby(['row_country', 'row_sector', 'col_country']
        ).sum()+consumption_OECD.set_index(['row_country', 'row_sector', 'col_country'])
                       ).reset_index()
    
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
        ['row_country', 'row_sector', 'col_country'])['value'].sum().to_frame()
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
    alpha_s = va_OECD.groupby('sector')['value'].sum()/trade_OECD_reduced.groupby(
        'sector')['trade'].sum()
    sector_moments = pd.DataFrame(index=[0, 1])
    sector_moments['beta'] = beta_s
    sector_moments['alpha'] = alpha_s
    if write:
        sector_moments.to_csv(path+'sector_moments.csv')
        sector_moments.to_csv(dropbox_path+'sector_moments.csv')
    
    iot_OECD_weights = pd.read_csv(
        '/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas2010/input_output_2010.csv')
    consumption_OECD_weights = pd.read_csv(
        '/Users/slepot/Documents/taff/datas/OECD/yearly_CSV/datas2010/consumption_2010.csv')
    
    trade_weights = (iot_OECD_weights.groupby(['row_country', 'row_sector', 'col_country']
        ).sum()+consumption_OECD_weights.set_index(['row_country', 'row_sector', 'col_country'])
                       ).reset_index()
    
    trade_weights_total = trade_weights.copy()
    trade_weights_total['row_country'] = trade_weights_total['row_country'].map(
        crosswalk_countries['country_code'])
    trade_weights_total['col_country'] = trade_weights_total['col_country'].map(
        crosswalk_countries['country_code'])
    trade_weights_total['row_sector'] = trade_weights_total['row_sector'].map(
        crosswalk_sectors['Sectors'])
    trade_weights_total = trade_weights_total.groupby(['row_country', 'row_sector', 'col_country']
            ).sum().reset_index()
    
    trade_weights_total['row_country'] = np.minimum(
        trade_weights_total['row_country'], nbr_of_countries)
    trade_weights_total['col_country'] = np.minimum(
        trade_weights_total['col_country'], nbr_of_countries)

    trade_weights_total = trade_weights_total.groupby(
        ['row_country', 'row_sector', 'col_country'])['value'].sum().to_frame()
    trade_weights_total = trade_weights_total.reorder_levels(
        ['row_country', 'col_country', 'row_sector'])
    
    trade_weights_total.sort_index(inplace=True)

    trade_weights_total.columns = ['trade']

    trade_weights_total.rename_axis(
        ['origin_code', 'destination_code', 'sector'], inplace=True)
    
    trade_weights['row_sector'] = trade_weights['row_sector'].map({
        '01T02':'agri_fishing',
        '03':'agri_fishing',
        '05T06':'mining_quarrying',
        '07T08':'mining_quarrying',
        '09':'mining_quarrying'
        },na_action='ignore')
    
    trade_weights['row_country'] = trade_weights['row_country'].map(
        crosswalk_countries['country_code'])
    trade_weights['col_country'] = trade_weights['col_country'].map(
        crosswalk_countries['country_code'])
    
    trade_weights['row_country'] = np.minimum(
        trade_weights['row_country'], nbr_of_countries)
    trade_weights['col_country'] = np.minimum(
        trade_weights['col_country'], nbr_of_countries)

    trade_weights = trade_weights.groupby(
        ['row_country', 'row_sector', 'col_country'])['value'].sum().to_frame()
    trade_weights = trade_weights.reorder_levels(
        ['row_country', 'col_country', 'row_sector'])
    
    trade_weights.sort_index(inplace=True)

    trade_weights.columns = ['trade']

    trade_weights.rename_axis(
        ['origin_code', 'destination_code', 'sector'], inplace=True)
    
    tariff_all = pd.read_csv(tariff_data_path+f'tariffs_{nbr_of_countries}_countries.csv').set_index(
        ['origin_code', 'destination_code', 'sector', 'year', 'base_year']).sort_index().reset_index()
    
    tariff_all = tariff_all.loc[tariff_all['sector'].isin(['agri_fishing',
                                                           'mining_quarrying',
                                                           'patenting'])]
    
    tariff_all = tariff_all.pivot(index=['origin_code','destination_code','year'],
                                  columns = 'sector',values='tariff')
    
    tariff_all['trade_agri_fishing'] = trade_weights.loc[:,:,'agri_fishing']['trade']
    tariff_all['trade_mining_quarrying'] = trade_weights.loc[:,:,'mining_quarrying']['trade']
    tariff_all['trade_non_patent'] = trade_weights_total.loc[:,:,0]['trade']
    
    tariff_all = tariff_all.fillna(0)
    
    tariff_all['non_patenting'] = (tariff_all['trade_agri_fishing']*tariff_all['agri_fishing']
                                   +tariff_all['trade_mining_quarrying']*tariff_all['mining_quarrying']
                                   )/tariff_all['trade_non_patent']
    
    tariff_all = pd.melt(tariff_all[['patenting','non_patenting']].reset_index(),
                         id_vars = ['origin_code', 'destination_code', 'year'],
                         value_vars = ['patenting','non_patenting'],
                         var_name='sector',
                         value_name='tariff'
                         )
    
    tariff_all['sector'] = tariff_all['sector'].map({
        'patenting':1,
        'non_patenting':0,
        },na_action='ignore')
    
    # tariff_all = tariff_all.set_index(['year','origin_code', 'destination_code','sector'])
    # tariff = tariff_all.loc[year]
    
    # if correction:
    #     for tariff_year in range(1990,1996):
    #         tariff_all.loc[(tariff_all.year == tariff_year) &
    #                    (tariff_all.sector == 0),'tariff'] = tariff_all.loc[(tariff_all.year == tariff_year) &
    #                               (tariff_all.sector == 0),'tariff']+tariff_all[(tariff_all.year == 1996) &
    #                               (tariff_all.sector == 0)]['tariff'].values-tariff_all[(tariff_all.year == 1995) &
    #                               (tariff_all.sector == 0)]['tariff'].values
    
    tariff = tariff_all[tariff_all.year == year].groupby(['origin_code',
                                                            'destination_code',
                                                            'sector'])[['tariff']].mean()/100
    
    # #%%
    # import matplotlib.pyplot as plt
    
    # fig,ax= plt.subplots(figsize = (12,8))
    
    # for s in [0,1]:
    #     if s==0:
    #         ls='-'
    #     else:
    #         ls ='--'
    #     x = tariff_all[(tariff_all['sector']==s)].groupby('year').mean().index.get_level_values(0)
    #     y = tariff_all[(tariff_all['sector']==s)].groupby('year').mean()['tariff']
    #     ax.plot(x,y,label=f'sector {s}',ls=ls)
    # plt.legend()
    # plt.show()
    
    # #%%
    
    
    if write or write_tariff:
        tariff.to_csv(path+'tariff.csv')
        tariff.to_csv(
            dropbox_path+'tariff.csv')
    
    #%%
    final_pat_fees_year = final_pat_fees.copy()
    #!!! changing EUR patenting fee
    # final_pat_fees_year.loc[2,'fee'] = 30530*gdp_year.loc[2]/exchange_rates[year]/gdp.loc[
    #     gdp['Country Code'].isin(['GBR','FRA','ITA','NLD','ESP','DEU'])
    #     ][f'{year} [YR{year}]'].sum()
    # EUR_2003_gdp = pd.merge(gdp[
    #     ['Country Code', '2003 [YR2003]']
    # ].rename(
    #     columns={'2003 [YR2003]': 'gdp'}
    # ).set_index(
    #     'Country Code'
    # ).rename_axis(
    #     'country'
    # ),
    #     crosswalk_countries,
    #     left_index=True,
    #     right_index=True
    # ).groupby('country_code').sum().loc[2,'gdp']
    # final_pat_fees_year.loc[2,'fee'] = 30530*EUR_2003_gdp*exchange_rates[2003]/gdp.loc[
    #     gdp['Country Code'].isin(['GBR','FRA','ITA','NLD','ESP','DEU'])
    #     ]['2003 [YR2003]'].sum()*gdp_deflator.loc['USA', '2005']/gdp_deflator.loc['USA', '2003']
    final_pat_fees_year.loc[2,'fee'] = 32130.875144801288
    final_pat_fees_year['fee'] = final_pat_fees_year['fee']*gdp_deflator.loc['USA', str(year)]/gdp_deflator.loc['USA', '2005']
    if write:
        final_pat_fees_year.to_csv(path+'final_pat_fees.csv')
        final_pat_fees_year.to_csv(dropbox_path+'final_pat_fees.csv')
    
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

    # get patent flows

    pflows = pd.read_csv(
        pflows_path+f'{nbr_of_countries}_countries/flows_{nbr_of_countries}_countries_{year}.csv', index_col=[0, 1]
    )

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
        
    # correct Indonesia patent flows with wipo data and deflate it
    if nbr_of_countries > 12:
        indonesia_flows = pd.read_csv(pflows_path+f'patent_flows_from_wipo/unscaled_ID_{nbr_of_countries}_countries.csv'
                                    ).set_index(['year', 'origin_code']).loc[year]
    
        for factor_type in ["app_per_family",
                            "first_applicant",
                            "sector_filter"]:
            indonesia_flows['patent flows'] = indonesia_flows[
                'patent flows'
            ]/pd.read_csv(
            ).set_index(
                ['year', 'origin_code']
            ).loc[year]['factor']
                
        for origin in indonesia_flows.index:
            pflows.loc[(origin,12),'patent flows'] = indonesia_flows.loc[origin,'patent flows']
    
    year_after_us_correc = 2001
    if year<year_after_us_correc:
        us_correction = pd.read_csv(
            pflows_path+f'factors/granted_patents_US_factor_{nbr_of_countries}_countries.csv'
        ).set_index(['year','origin_code'])
        year_before_us_correc = max(year, 1997) 
        for origin_code in range(1, nbr_of_countries+1):
            pflows.loc[origin_code,1] = pflows.loc[origin_code,1].values/(
                us_correction.loc[year_after_us_correc]/us_correction.loc[year_before_us_correc]
                ).loc[origin_code].values
    
    for origin_code in range(1, nbr_of_countries+1):
        for destination_code in range(1, nbr_of_countries+1):
            if (origin_code, destination_code) not in pflows.index or pflows.loc[(origin_code, destination_code)].isna().any():
                pflows.loc[(origin_code, destination_code),
                            'patent flows'] = 0
                
    for origin_code in range(1, nbr_of_countries+1):
        for destination_code in range(1, nbr_of_countries+1):
            if pflows.loc[(origin_code, destination_code),
                        'patent flows'] == 0:
                pflows.loc[(origin_code, destination_code),
                            'patent flows'] = pflows.loc[
                                pflows['patent flows'] > 0, 'patent flows'
                                ].min(
                                    )*(pflows.xs(origin_code,level=0)['patent flows'].sum(
                                        )*pflows.xs(destination_code,level=1)['patent flows'].sum(
                                            )/pflows['patent flows'].sum())**(1/16)
            # pflows.loc[
            #     pflows['patent flows'] == 0, 'patent flows'
            #     ] = pflows.loc[
            #         pflows['patent flows'] > 0, 'patent flows'
            #         ].min()/10
    
    pflows = pflows.sort_index()
            
    if write:
        pflows.to_csv(path+'country_country_moments.csv')
        pflows.to_csv(dropbox_path+'country_country_moments.csv')

    scalar_moments.loc['UUPCOST', 'value'] = pflows.loc[(1, 1), 'patent flows']*final_pat_fees.loc[1, 'fee']\
        * gdp_deflator.loc['USA', str(year)]/gdp_deflator.loc['USA', '2005']/1e12

    scalar_moments.loc['JUPCOST', 'value'] = pflows.loc[(1, 3), 'patent flows']*final_pat_fees.loc[1, 'fee']\
        * gdp_deflator.loc['USA', str(year)]/gdp_deflator.loc['USA', '2005']/1e12
    # scalar_moments.loc['TO', 'value'] = 0.018546283
    scalar_moments.loc['TO', 'value'] = 0.017496806
    if write:
        scalar_moments.to_csv(path+'scalar_moments.csv')
        scalar_moments.to_csv(dropbox_path+'scalar_moments.csv')
