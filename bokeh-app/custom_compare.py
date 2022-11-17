#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 00:22:26 2022

@author: simonl
"""

import numpy as np
import pandas as pd
import os
from classes import moments, parameters, var
from solver_funcs import full_load_and_solve
from data_funcs import compare_params

list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
 'RD', 'RD_US', 'RD_RUS', 'RP', 'SPFLOWDOM', 'SPFLOW',
 'SPFLOW_US', 'SPFLOW_RUS', 'SRDUS', 'SRGDP', 'SRGDP_US',
 'SRGDP_RUS', 'JUPCOST','JUPCOSTRD', 'SINNOVPATUS', 'TO',
 'DOMPATUS','DOMPATEU']
comments_dic = {'1':'1 : drop South in RD targeting',
                '2.1':'2.1 : added domestic US to patent flow moment',
                '2.2':'2.2 : added domestic EU to patent flow moment',
                '2.3':'2.3 : added domestic US and EU to patent flow moment',
                '3.1':'3.1 : added DOMPATUS',
                '3.2':'3.2 : added DOMPATEU',
                '3.3':'3.3 : added DOMPATUS and DOMPATUS',
                '4.1':'4.1 : 21 and drop South in RD',
                '4.2':'4.2 : 22 and drop South in RD',
                '4.3':'4.3 : 23 and drop South in RD',
                '5':'5 : patent cost relative to RD_US (JUPCOSTRD)',
                '6':'6 : fix delta_US = 0.05 and drop JUPCOST',
                '7':'7 : drop SRDUS',
                '8.1':'8.1: drop South RD, DOMPAT moments, weight1 SPFLOW',
                '8.2':'8.2: drop South RD, DOMPAT moments, weight3 SPFLOW'
                }
#%%
# baseline_number = '101'
for baseline_number in ['101','102','104']:
    path = 'calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
    dropbox_path = '/Users/simonl/Dropbox/TRIPS/simon_version/code/calibration_results_matched_economy/baseline_'+baseline_number+'_variations/'
    
    moments_runs = []
    params_runs = []
    files_in_dir = next(os.walk(path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    # run_list = ['101','102','103','104']
    comments = []
    for run in run_list:
        p, sol, m = full_load_and_solve(path+run+'/')
        moments_runs.append(m)
        params_runs.append(p)
        comments.append(comments_dic[run])
    dic_m = dict(zip(comments, moments_runs))
    dic_p = dict(zip(comments, params_runs))
    save_path = dropbox_path+'summary/'
    try:
        os.mkdir(save_path)
    except:
        pass
    moments.compare_moments(dic_m,
        save_path = save_path,list_of_moments = list_of_moments, contin_cmap = True)
    compare_params(dic_p,color_gradient=True,save = True, save_path = save_path)
    # moments.compare_moments(dic_m,list_of_moments = list_of_moments)
    # compare_params(dic_p,color_gradient=True)
