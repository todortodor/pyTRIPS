#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:40:47 2023

@author: slepot
"""

import pandas as pd

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

WDI_data_path = '/Users/slepot/Dropbox/TRIPS/Calibration data/'

WDI_data = pd.read_csv(WDI_data_path+'WDI_10032023_data.csv',
                        keep_default_na=False,
                        na_values=na_values)

gdp = WDI_data.loc[WDI_data['Series Code'] == 'NY.GDP.MKTP.CD'][['Country Name','2015 [YR2015]']].set_index(
    'Country Name').rename(columns={'2015 [YR2015]':'gdp'})
price = WDI_data.loc[WDI_data['Series Code'] == 'PA.NUS.PPPC.RF'][
    ['Country Name','2015 [YR2015]']
    ].set_index('Country Name').dropna().rename(columns={'2015 [YR2015]':'price'})

tot = pd.merge(price,
               gdp,
               how = 'left',
               left_index=True,
               right_index=True)

tot['real gdp'] = tot['gdp']/tot['price']
tot = tot.sort_values('real gdp',ascending=False)
tot['real gdp share'] = tot['real gdp']/tot['real gdp'].sum()

tot.to_csv('../misc/real_gdp_rank_2015.csv')