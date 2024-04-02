#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:06:54 2023

@author: slepot
"""

import pandas as pd
import os

delta_t = 3

data_path = 'data/'
smooth_data_path = f'data_smooth_{delta_t}_years/'

try:
    os.mkdir(smooth_data_path)
except:
    pass

years = [y for y in range(1990,2019)]
# years = [y for y in range(1990,1991)]
nbrs_countries = [11]
# nbrs_countries = [12]
# nbrs_countries = [7]

years_for_smoothing = {}

for y in years:
    years_for_smoothing[y] = [y]
    i=1
    while (delta_t/2)-i>0:
        if y-i >= years[0]:
            years_for_smoothing[y].insert(0,y-i)
        if y+i <= years[-1]:
            years_for_smoothing[y].append(y+i)
        i+=1

def average(year,name,index_col,years_smoothing,nbr_of_countries):
    smooth_data = pd.read_csv(f'data/data_{nbr_of_countries}_countries_{year}/'+name,index_col=index_col)
    for y in years_smoothing:
        if y!= year:
            smooth_data = smooth_data + pd.read_csv(f'data/data_{nbr_of_countries}_countries_{y}/'+name,index_col=index_col)
    return smooth_data/len(years_smoothing)

for y in years:
    for nbr_of_countries in nbrs_countries:
        path = smooth_data_path+f'data_{nbr_of_countries}_countries_{y}/'
        try:
            os.mkdir(path)
        except:
            pass
        print(y,nbr_of_countries)
        
        moments_descriptions = pd.read_csv(
            f'data/data_{nbr_of_countries}_countries_{y}/moments_descriptions.csv', sep=';', index_col=0)
        moments_descriptions.to_csv(path+'moments_descriptions.csv', sep=';')
        
        data_to_average_list = [
        # dict(name='country_country_moments.csv'
        #      ,index_col=[0,1]),
        # dict(name='country_country_sector_moments.csv'
        #      ,index_col=[0,1,2]),
        dict(name='tariff.csv'
             ,index_col=[0,1,2]),
        # dict(name='country_moments.csv'
        #      ,index_col=[0]),
        # dict(name='final_pat_fees.csv'
        #      ,index_col=[0,1]),
        # dict(name='scalar_moments.csv'
        #      ,index_col=[0]),
        # dict(name='sector_moments.csv'
        #      ,index_col=[0]),
        ]
        
        for data_to_average in data_to_average_list:
            smoothed_out = average(year=y,
                                   name=data_to_average['name'],
                                   index_col=data_to_average['index_col'],
                                   years_smoothing=years_for_smoothing[y],
                                   nbr_of_countries=nbr_of_countries)
            smoothed_out.to_csv(path+data_to_average['name'])
    