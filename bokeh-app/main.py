from os.path import dirname, join
import os
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Button, Slider, Toggle, FactorRange, Div, ColumnDataSource, LabelSet, Select,Legend, LegendItem, DataTable, TableColumn, HoverTool, Slope
from bokeh.plotting import figure
from bokeh.events import ButtonClick
from classes import parameters, moments, var
from data_funcs import compute_rough_jacobian,rough_dyn_fixed_point_solver
import numpy as np
import itertools
from bokeh.palettes import Category10, Dark2
Category18 = Category10[10]+Dark2[8]
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)
# warnings.simplefilter('ignore', np.RuntimeWarning)
warnings.filterwarnings('ignore')

start = time.perf_counter()

def load(path, data_path=None, 
         dir_path = None, context = 'calibration'):
    # p = parameters(data_path=data_path)
    p = parameters()
    # p.load_data(path)
    p.load_run(path,dir_path=dir_path)
    sol = var.var_from_vector(p.guess, p, compute=True, context = context)
    sol.scale_P(p)
    sol.compute_price_indices(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    # m.load_data(data_path)
    m.load_run(path,dir_path=dir_path)
    m.compute_moments(sol, p)
    m.compute_moments_deviations()
    return p,m,sol

def init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,sol_baseline,list_of_moments):
    dic_df_param = {}
    dic_df_mom = {}
    dic_df_sol = {}
    params = p_baseline.calib_parameters
    params.append('kappa')
    params.append('r_hjort')
    if 'theta' not in params:
        params.append('theta')
    params.append('theta')
    # params.append('d*fe')
    # params.append('nu/deltaUS')
    df_scalar_params = pd.DataFrame(columns = ['baseline'])
    df_scalar_params.index.name='x'
    
    for param in params:
        if hasattr(p_baseline,param):
            if len(getattr(p_baseline,param)[p_baseline.mask[param]]) == 1:
                if param == 'k':
                    df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,param)[p_baseline.mask[param]])-1
                else:
                    df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,param)[p_baseline.mask[param]])
            if param in ['eta','delta']:
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,1])
                df.index.name='x'
                dic_df_param[param] = df
            if param in ['r_hjort']:
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param))
                df.index.name='x'
                dic_df_param[param] = df
            if param in ['T']:
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,0])
                df.index.name='x'
                dic_df_param[param+' non patent sector'] = df
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,1])
                df.index.name='x'
                dic_df_param[param+' patent sector'] = df
        elif param == 'd*fe':
            df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,'d')[p_baseline.mask['d']])*float(getattr(p_baseline,'fe')[p_baseline.mask['fe']])
        elif param == 'nu/deltaUS':
            df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,'nu')[1])/float(getattr(p_baseline,'delta')[0,1])
    dic_df_param['scalars'] = df_scalar_params
    
    df_scalar_moments = pd.DataFrame(columns = ['target','baseline'])
    df_scalar_moments.index.name='x'
    for mom in list_of_moments:
        if mom != 'objective':
            if len(m_baseline.idx[mom]) == 1:
                if mom != 'OUT':
                    try:
                        df_scalar_moments.loc[mom,'target'] = float(getattr(m_baseline,mom+'_target'))
                        df_scalar_moments.loc[mom,'baseline'] = float(getattr(m_baseline,mom))
                    except:
                        pass
            else:
                try:
                    df = pd.DataFrame(index = m_baseline.idx[mom], 
                                      columns = ['target','baseline'], 
                                      # data = np.array([getattr(m_baseline,mom+'_target').ravel(), getattr(m_baseline,mom).ravel()])
                                      )
                    df.index.name='x'
                    df['target'] = getattr(m_baseline,mom+'_target').ravel()
                    df['baseline'] = getattr(m_baseline,mom).ravel()
                    dic_df_mom[mom] = df
                except:
                    pass
            
    for sol_qty in ['semi_elast_patenting_delta','DT','psi_o_star']:
        df = pd.DataFrame(index = p_baseline.countries, 
                          columns = ['baseline'], 
                          )
        df.index.name='x'
        df['baseline'] = getattr(sol_baseline,sol_qty)[...,1]
        dic_df_sol[sol_qty] = df
        
    for sol_qty in ['l_R']:
        df = pd.DataFrame(index = p_baseline.countries, 
                          columns = ['baseline'], 
                          )
        df.index.name='x'
        df['baseline'] = getattr(sol_baseline,sol_qty)[...,1]/p_baseline.labor
        dic_df_sol[sol_qty] = df
        
    for sol_qty in ['min_psi_m_star_inward']:
        df = pd.DataFrame(index = p_baseline.countries, 
                          columns = ['baseline'], 
                          )
        df.index.name='x'
        df['baseline'] = getattr(sol_baseline,'psi_m_star')[:,:,1].min(axis=1)
        dic_df_sol[sol_qty] = df
        
    for sol_qty in ['min_psi_m_star_outward']:
        df = pd.DataFrame(index = p_baseline.countries, 
                          columns = ['baseline'], 
                          )
        df.index.name='x'
        df['baseline'] = getattr(sol_baseline,'psi_m_star')[:,:,1].min(axis=0)
        dic_df_sol[sol_qty] = df
    
    df_scalar_moments.loc['objective','target'] = 0.01
    df_scalar_moments.loc['objective','baseline'] = m_baseline.objective_function()*28
    dic_df_mom['scalars'] = df_scalar_moments
    return dic_df_param, dic_df_mom, dic_df_sol

def append_dic_of_dataframes_with_variation(dic_df_param, dic_df_mom, dic_df_sol, p, m, sol, run_name):
    for k in dic_df_param.keys():
        if k == 'scalars':
            for i in dic_df_param[k].index:
                if i == 'k':
                    dic_df_param[k].loc[i,run_name] = float(getattr(p,i)[p.mask[i]])-1
                elif i == 'd*fe':
                    dic_df_param[k].loc[i,run_name] = float(getattr(p,'d')[p.mask['d']])*float(getattr(p,'fe')[p.mask['fe']])
                elif i == 'nu/deltaUS':
                    dic_df_param[k].loc[i,run_name] = float(getattr(p,'nu')[1])/float(getattr(p,'delta')[0,1])
                else:
                    dic_df_param[k].loc[i,run_name] = float(getattr(p,i)[p.mask[i]])
                
        if k in ['eta','delta']:
            dic_df_param[k][run_name] = getattr(p,k)[...,1]
        if k in ['r_hjort']:
            dic_df_param[k][run_name] = getattr(p,k)
        if k == 'T non patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,0]
        if k == 'T patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,1]
        
    for k in dic_df_mom.keys():
        if k == 'scalars':
            for i in dic_df_mom[k].index:
                if i == 'objective':
                    dic_df_mom[k].loc[i,run_name] = m.objective_function()*28
                else:
                    dic_df_mom[k].loc[i,run_name] = float(getattr(m,i))
        if k == 'scalar deviations':
            for i in dic_df_mom[k].index:
                dic_df_mom[k].loc[i,run_name] = float(getattr(m,i+'_deviation'))/m.weights_dict[i]
        if k not in ['scalars','scalar deviations']:
            dic_df_mom[k][run_name] = getattr(m,k).ravel()
    
    for k in dic_df_sol.keys():
        if k in ['semi_elast_patenting_delta','DT','psi_o_star']:
            dic_df_sol[k][run_name] = getattr(sol,k)[...,1]
        if k in ['l_R']:
            dic_df_sol[k][run_name] = getattr(sol,k)[...,1]/p.labor
        if k in ['min_psi_m_star_outward']:
            dic_df_sol[k][run_name] = getattr(sol,'psi_m_star')[:,:,1].min(axis=0)
        if k in ['min_psi_m_star_inward']:
            dic_df_sol[k][run_name] = getattr(sol,'psi_m_star')[:,:,1].min(axis=1)
            
    return dic_df_param, dic_df_mom, dic_df_sol

#%% path
dir_path = dirname(__file__)+'/'
data_path = join(dirname(__file__), 'data/')
# dir_path = './'
# data_path = 'data/'
# results_path = 'calibration_results_matched_economy/'
results_path = join(dirname(__file__), 'calibration_results_matched_economy/')
cf_path = join(dirname(__file__), 'counterfactual_recaps/unilateral_patent_protection/')
around_dyn_eq_path = join(dirname(__file__), 'counterfactual_recaps/')
nash_eq_path = join(dirname(__file__), 'nash_eq_recaps/')
coop_eq_path = join(dirname(__file__), 'coop_eq_recaps/')


#%% moments / parameters for variations

list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
 'RD', 'RP', 'SPFLOWDOM', 'SPFLOW','STFLOW','STFLOWSDOM',
 'SRGDP','UUPCOST','SINNOVPATUS',
  'TO','TE','DOMPATINUS','DOMPATUS',
 'TWSPFLOW','TWSPFLOWDOM','SDOMTFLOW','objective']
# list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
#  'RD', 'RP', 'SPFLOWDOM', 'SPFLOW','STFLOW','STFLOWSDOM',
#  'SRDUS', 'SRGDP','UUPCOST', 'PCOST','PCOSTINTER','PCOSTNOAGG','PCOSTINTERNOAGG','SINNOVPATUS',
#  'SINNOVPATEU', 'TO','TP',
#  'DOMPATUS','DOMPATEU','DOMPATINUS','DOMPATINEU','TWSPFLOW','TWSPFLOWDOM','SDOMTFLOW','objective']

comments_dic = {}

# comments_dic['403'] = {'baseline':'bsln:TO:0.0183',
#     '1.0':'1.0: TO: 0.01',
# '1.1':'1.1: TO: 0.0105',
# '1.2':'1.2: TO: 0.011',
# '1.3':'1.3: TO: 0.0115',
# '1.4':'1.4: TO: 0.012',
# '1.5':'1.5: TO: 0.0125',
# '1.6':'1.6: TO: 0.013',
# '1.7':'1.7: TO: 0.0135',
# '1.8':'1.8: TO: 0.014',
# '1.9':'1.9: TO: 0.0145',
# '1.10':'1.10: TO: 0.015',
# '1.11':'1.11: TO: 0.0155',
# '1.12':'1.12: TO: 0.016',
# '1.13':'1.13: TO: 0.0165',
# '1.14':'1.14: TO: 0.017',
# '1.15':'1.15: TO: 0.0175',
# '1.16':'1.16: TO: 0.018',
# '1.17':'1.17: TO: 0.0185',
# '1.18':'1.18: TO: 0.019',
# '1.19':'1.19: TO: 0.0195',
# '1.20':'1.20: TO: 0.02',
# '1.21':'1.21: TO: 0.0205',
# '1.22':'1.22: TO: 0.021',
# '1.23':'1.23: TO: 0.0215',
# '1.24':'1.24: TO: 0.022',
# '1.25':'1.25: TO: 0.0225',
# '1.26':'1.26: TO: 0.023',
# '1.27':'1.27: TO: 0.0235',
# '1.28':'1.28: TO: 0.024',
# '1.29':'1.29: TO: 0.0245',
# '1.30':'1.30: TO: 0.025',
# '1.31':'1.31: TO: 0.0255',
# '1.32':'1.32: TO: 0.026',
# '1.33':'1.33: TO: 0.0265',
# '1.34':'1.34: TO: 0.027',
# '1.35':'1.35: TO: 0.0275',
# '1.36':'1.36: TO: 0.028',
# '1.37':'1.37: TO: 0.0285',
# '1.38':'1.38: TO: 0.029',
# '1.39':'1.39: TO: 0.0295',
# '1.40':'1.40: TO: 0.03'
#     }
# comments_dic['405'] = {'baseline':'bsln:TO:0.0183',
#     '1.0':'1.0: TO: 0.01',
# '1.1':'1.1: TO: 0.0105',
# '1.2':'1.2: TO: 0.011',
# '1.3':'1.3: TO: 0.0115',
# '1.4':'1.4: TO: 0.012',
# '1.5':'1.5: TO: 0.0125',
# '1.6':'1.6: TO: 0.013',
# '1.7':'1.7: TO: 0.0135',
# '1.8':'1.8: TO: 0.014',
# '1.9':'1.9: TO: 0.0145',
# '1.10':'1.10: TO: 0.015',
# '1.11':'1.11: TO: 0.0155',
# '1.12':'1.12: TO: 0.016',
# '1.13':'1.13: TO: 0.0165',
# '1.14':'1.14: TO: 0.017',
# '1.15':'1.15: TO: 0.0175',
# '1.16':'1.16: TO: 0.018',
# '1.17':'1.17: TO: 0.0185',
# '1.18':'1.18: TO: 0.019',
# '1.19':'1.19: TO: 0.0195',
# '1.20':'1.20: TO: 0.02',
# '1.21':'1.21: TO: 0.0205',
# '1.22':'1.22: TO: 0.021',
# '1.23':'1.23: TO: 0.0215',
# '1.24':'1.24: TO: 0.022',
# '1.25':'1.25: TO: 0.0225',
# '1.26':'1.26: TO: 0.023',
# '1.27':'1.27: TO: 0.0235',
# '1.28':'1.28: TO: 0.024',
# '1.29':'1.29: TO: 0.0245',
# '1.30':'1.30: TO: 0.025',
# '1.31':'1.31: TO: 0.0255',
# '1.32':'1.32: TO: 0.026',
# '1.33':'1.33: TO: 0.0265',
# '1.34':'1.34: TO: 0.027',
# '1.35':'1.35: TO: 0.0275',
# '1.36':'1.36: TO: 0.028',
# '1.37':'1.37: TO: 0.0285',
# '1.38':'1.38: TO: 0.029',
# '1.39':'1.39: TO: 0.0295',
# '1.40':'1.40: TO: 0.03'
#     }

# comments_dic['404'] = {
#     'baseline':'baseline',
#     '1.0':'1.0: SRDUS, UUPCOST, log loss',
#     '1.1':'1.1: SRDUS, UUPCOST, ratio loss',
#     '1.2':'1.2: SRDUS, PCOSTNOAGG, log loss',
#     '1.3':'1.3: SRDUS, PCOSTNOAGG, ratio loss',
#     '1.4':'1.4: no SRDUS, UUPCOST, log loss',
#     '1.5':'1.5: no SRDUS, UUPCOST, ratio loss',
#     '1.6':'1.6: no SRDUS, PCOSTNOAGG, log loss',
#     '1.7':'1.7: no SRDUS, PCOSTNOAGG, ratio loss',
#     '1.8':'1.8: no RD, UUPCOST, log loss',
#     '1.9':'1.9: no RD, UUPCOST, ratio loss',
#     '1.10':'1.10: no RD, PCOSTNOAGG, log loss',
#     '1.11':'1.11: no RD, PCOSTNOAGG, ratio loss',
#     '2.0':'2.0: sigma=2.7, SRDUS, UUPCOST',
#     '2.1':'2.1: sigma=2.7, no SRDUS, UUPCOST',
#     '2.2':'2.2: sigma=2.7, SRDUS, PCOSTNOAGG',
#     '2.3':'2.3: sigma=2.7, no SRDUS, PCOSTNOAGG',
#     }

# comments_dic['501'] = {
#     "baseline":"baseline",
#     '1.0':'1.0: Higher growth weight',
#     '2.0':'2.0: Hjort correc real GDP',
#     '3.0':'3.0: No drop RD South',
#     '4.0':'4.0: New data',
#     '5.0':'5.0: New data v2',
#     }

# comments_dic['601'] = {
#     "baseline":"baseline : 2005",
#     "1.0" : "1.0 : 1990",
#     "1.1" : "1.1 : 1991",
#     "1.2" : "1.2 : 1992",
#     "1.3" : "1.3 : 1993",
#     "1.4" : "1.4 : 1994",
#     "1.5" : "1.5 : 1995",
#     "1.6" : "1.6 : 1996",
#     "1.7" : "1.7 : 1997",
#     "1.8" : "1.8 : 1998",
#     "1.9" : "1.9 : 1999",
#     "1.10" : "1.10 : 2000",
#     "1.11" : "1.11 : 2001",
#     "1.12" : "1.12 : 2002",
#     "1.13" : "1.13 : 2003",
#     "1.14" : "1.14 : 2004",
#     "1.15" : "1.15 : 2005",
#     "1.16" : "1.16 : 2006",
#     "1.17" : "1.17 : 2007",
#     "1.18" : "1.18 : 2008",
#     "1.19" : "1.19 : 2009",
#     "1.20" : "1.20 : 2010",
#     "1.21" : "1.21 : 2011",
#     "1.22" : "1.22 : 2012",
#     "1.23" : "1.23 : 2013",
#     "1.24" : "1.24 : 2014",
#     "1.25" : "1.25 : 2015",
#     "1.26" : "1.26 : 2016",
#     "1.27" : "1.27 : 2017",
#     "1.28" : "1.28 : 2018",
# }

# comments_dic['602'] = comments_dic['601']
# comments_dic['603'] = comments_dic['601']

# comments_dic['606'] = {
#     "baseline":"baseline:SRGDP weight < RP weight",
#     "2.0" : "2.0:SRGDP weight = RP weight",
#     "3.0" : "3.0:SRGDP weight > RP weight",
# }
# comments_dic['607'] = comments_dic['601']
# comments_dic['608'] = comments_dic['601']
# comments_dic['609'] = comments_dic['601']
# comments_dic['610'] = comments_dic['601']
# comments_dic['614'] = comments_dic['601']
# comments_dic['615'] = comments_dic['601']
# comments_dic['616'] = comments_dic['601']
# comments_dic['617'] = comments_dic['601']
# comments_dic['620'] = comments_dic['601']
# comments_dic['619'] = comments_dic['601']


# comments_dic['611'] = {'baseline':'bsln:TO:0.0183',
#     '1.0':'1.0: TO: 0.01',
# '1.1':'1.1: TO: 0.0105',
# '1.2':'1.2: TO: 0.011',
# '1.3':'1.3: TO: 0.0115',
# '1.4':'1.4: TO: 0.012',
# '1.5':'1.5: TO: 0.0125',
# '1.6':'1.6: TO: 0.013',
# '1.7':'1.7: TO: 0.0135',
# '1.8':'1.8: TO: 0.014',
# '1.9':'1.9: TO: 0.0145',
# '1.10':'1.10: TO: 0.015',
# '1.11':'1.11: TO: 0.0155',
# '1.12':'1.12: TO: 0.016',
# '1.13':'1.13: TO: 0.0165',
# '1.14':'1.14: TO: 0.017',
# '1.15':'1.15: TO: 0.0175',
# '1.16':'1.16: TO: 0.018',
# '1.17':'1.17: TO: 0.0185',
# '1.18':'1.18: TO: 0.019',
# '1.19':'1.19: TO: 0.0195',
# '1.20':'1.20: TO: 0.02',
# '1.21':'1.21: TO: 0.0205',
# '1.22':'1.22: TO: 0.021',
# '1.23':'1.23: TO: 0.0215',
# '1.24':'1.24: TO: 0.022',
# '1.25':'1.25: TO: 0.0225',
# '1.26':'1.26: TO: 0.023',
# '1.27':'1.27: TO: 0.0235',
# '1.28':'1.28: TO: 0.024',
# '1.29':'1.29: TO: 0.0245',
# '1.30':'1.30: TO: 0.025',
# '1.31':'1.31: TO: 0.0255',
# '1.32':'1.32: TO: 0.026',
# '1.33':'1.33: TO: 0.0265',
# '1.34':'1.34: TO: 0.027',
# '1.35':'1.35: TO: 0.0275',
# '1.36':'1.36: TO: 0.028',
# '1.37':'1.37: TO: 0.0285',
# '1.38':'1.38: TO: 0.029',
# '1.39':'1.39: TO: 0.0295',
# '1.40':'1.40: TO: 0.03'
#     }

# comments_dic['618'] = {
#     'baseline':'baseline',
#     '1.0':'1.0:full calibration 2005',
#     '1.1':'1.1:full calibration 1992',
#     '2.0':'2.0:free f, target UUPCOST, 2005',
#     '2.1':'2.1:free f, target UUPCOST, 1992',
#     '3.0':'3.0:free f, target UUPCOST and TP, 2005',
#     '3.1':'3.1:free f, target UUPCOST and TP, 1992',
#     '4.0':'4.0:free f, target UUPCOST and inter_TP, 2005',
#     '4.1':'4.1:free f, target UUPCOST and inter_TP, 1992',
#     '5.0':'5.0:fixed f, target UUPCOST, 2005',
#     '5.1':'5.1:fixed f, target UUPCOST, 1992',
#     '6.0':'6.0:fixed f, target UUPCOST and TP, 2005',
#     '6.1':'6.1:fixed f, target UUPCOST and TP, 1992',
#     '7.0':'7.0:fixed f, target UUPCOST and inter_TP, 2005',
#     '7.1':'7.1:fixed f, target UUPCOST and inter_TP, 1992',
#     '8.0':'8.0:fixed f, drop UUPCOST, 2005',
#     '8.1':'8.1:fixed f, drop UUPCOST, 1992',
#     '9.0':'9.0:fixed f, drop UUPCOST, target TP, 2005',
#     '9.1':'9.1:fixed f, drop UUPCOST, target TP, 1992',
#     '10.0':'10.0:fixed f, drop UUPCOST,target inter_TP, 2005',
#     '10.1':'10.1:fixed f, drop UUPCOST,target inter_TP, 1992',
#     '11.0':'11.0:fixed f, target UUPCOST, drop SINNOV, 2005',
#     '11.1':'11.1:fixed f, target UUPCOST, drop SINNOV, 1992',
#     '12.0':'12.0:fixed f, target UUPCOST, inter_TP, drop SINNOV, 2005',
#     '12.1':'12.1:fixed f, target UUPCOST, inter_TP, drop SINNOV, 1992',
#     '13.0':'13.0:full calibration without SINNOV, 2005',
#     '15.0':'15.0:nu=0.1, drop TO',
#     '16.0':'16.0:fixed f, target UUPCOST and KM, drop SINNOV, 2005',
#     '16.1':'16.1:fixed f, target UUPCOST and KM, drop SINNOV, 1992',
#     '17.0':'17.0:fixed f, target UUPCOST and KM and inter_TP, drop SINNOV, 2005',
#     '17.1':'17.1:fixed f, target UUPCOST and KM and inter_TP, drop SINNOV, 1992',
#     '18.0':'18.0:fixed f, target UUPCOST and KM, 2005',
#     '18.1':'18.1:fixed f, target UUPCOST and KM, 1992',
#     '19.0':'19.0:fixed f, target UUPCOST and KM and inter_TP, 2005',
#     '19.1':'19.1:fixed f, target UUPCOST and KM and inter_TP, 1992',
#     '20.0':'20.0:fixed f, target UUPCOST and KM, drop SINNOVUS, 2005',
#     '20.1':'20.1:fixed f, target UUPCOST and KM, drop SINNOVUS, 1992',
#     '21.0':'21.0:fixed f, drop SINNOV, KM, UUPCOST, keep delta_north fixed 2005',
#     '21.1':'21.1:fixed f, drop SINNOV, KM, UUPCOST, keep delta_north fixed 1992',
#     '22.1':'22.1:full calibration 1992, scale up nbr of patents',
#     }

# comments_dic['701'] = {
#     'baseline':'baseline: same as 607/618 without SINNOVPATEU',
#     '1.0':'1.0:full calibration 2005',
#     '1.1':'1.1:full calibration 1992',
#     '2.0':'2.0:[delta], [SPFLOW], deltaUS fixed',
#     '2.1':'2.1:[delta], [SPFLOW], deltaUS fixed',
#     '3.0':'3.0:[delta], [SPFLOW,DOMPATIN], deltaUS fixed',
#     '3.1':'3.1:[delta], [SPFLOW,DOMPATIN], deltaUS fixed',
#     '4.0':'4.0:[delta], [SPFLOW,DOMPATIN], deltaUS fixed',
#     '4.1':'4.1:[delta], [SPFLOW,DOMPATIN], deltaUS_1995 = 1.17647 deltaUS_2005',
#     '5.0':'5.0:[delta,T], [SPFLOW,DOMPATIN,OUT], deltaUS fixed',
#     '5.1':'5.1:[delta,T], [SPFLOW,DOMPATIN,OUT], deltaUS fixed',
#     '6.0':'6.0:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP], deltaUS fixed',
#     '6.1':'6.1:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP], deltaUS fixed',
#     '7.0':'7.0:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP], delta,etaUS fixed',
#     '7.1':'7.1:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP], delta,etaUS fixed',
#     '8.0':'8.0:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP,KM], deltaUS fixed',
#     '8.1':'8.1:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP,KM], deltaUS fixed',
#     '9.0':'9.0:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP,KM], deltaUS fixed, KM weight=10',
#     '9.1':'9.1:[delta,T,eta], [SPFLOW,DOMPATIN,OUT,RD,RP,SRGDP,KM], deltaUS fixed, KM weight=10',
#     }

# comments_dic['702'] = {
#     'baseline':'baseline: same as 607/618 without SINNOVPATEU, DOMPATINEU',
#     '1.0':'1.0:full calibration 2005',
#     '1.1':'1.1:full calibration 1992',
#     '2.0':'2.0:[delta], [SPFLOW], deltaUS fixed',
#     '2.1':'2.1:[delta], [SPFLOW], deltaUS fixed',
#     '3.0':'3.0:[delta], [SPFLOW,DOMPATINUS], deltaUS fixed',
#     '3.1':'3.1:[delta], [SPFLOW,DOMPATINUS], deltaUS fixed',
#     '4.0':'4.0:[delta], [SPFLOW,DOMPATINUS], deltaUS fixed',
#     '4.1':'4.1:[delta], [SPFLOW,DOMPATINUS], deltaUS_1995 = 1.17647 deltaUS_2005',
#     '5.0':'5.0:[delta,T], [SPFLOW,DOMPATINUS,OUT], deltaUS fixed',
#     '5.1':'5.1:[delta,T], [SPFLOW,DOMPATINUS,OUT], deltaUS fixed',
#     '6.0':'6.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], deltaUS fixed',
#     '6.1':'6.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], deltaUS fixed',
#     '7.0':'7.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta,etaUS fixed',
#     '7.1':'7.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta,etaUS fixed',
#     '8.0':'8.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM], deltaUS fixed',
#     '8.1':'8.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM], deltaUS fixed',
#     '9.0':'9.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM], deltaUS fixed, KM weight=10',
#     '9.1':'9.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM], deltaUS fixed, KM weight=10',
#     '10.0':'10.0:[delta,eta], [SPFLOW,DOMPATINUS], deltaUS fixed',
#     '10.1':'10.1:[delta,eta], [SPFLOW,DOMPATINUS], deltaUS fixed',
#     }

# comments_dic['801'] = {
#     'baseline':'baseline',
#     '0.0':'0.0',
#     '0.1':'0.1',
#     '0.2':'0.2',
#     '0.3':'0.3',
#     '0.4':'0.4',
#     '0.5':'0.5',
#     }
# comments_dic['802'] = {
#     'baseline':'baseline,target RD US/EUR/JAP,theta=7',
#     '1.0':'1.0: 1992',
#     '2.0':'2.0: 2005 no Hjort correc',
#     '3.0':'3.0: target RD US/EU/JP/CA/KR/RU/AU/MX',
#     '4.0':'4.0: from now: target RD US/EU/JP/CA/KR/AU',
#     '4.1':'4.1: drop SRDUS',
#     '4.2':'4.2: drop SRDUS, higher weight on RD',
#     '4.3':'4.3: drop SRDUS, higher weight on SPFLOW',
#     '5.0':'5.0: lin loss',
#     '5.1':'5.1: weight on large pflows lin loss',
#     '5.2':'5.2: higher weight on large pflows lin loss',
#     '6.0':'6.0: weight on large pflows log loss',
#     '7.0':'7.0: higher weight on large pflows log loss',
#     }

# comments_dic['803'] = {
#     'baseline':'baseline: 802_7.0 with improved weights',
#     '1.0':'1.0: calibrated theta',
#     '1.1':'1.1: drop SRDUS',
#     '1.2':'1.2: drop SINNOVPATEU',
#     '1.3':'1.3: drop DOMPATINEU',
#     '1.4':'1.4: drop SINNOVPATEU and DOMPATINEU',
#     '1.5':'1.5: drop SRDUS, SINNOVPATEU and DOMPATINEU',
#     '1.5.0':'1.5.0: sigma = 2',
#     '1.5.1':'1.5.1: sigma = 2.25',
#     '1.5.2':'1.5.2: sigma = 2.5',
#     '1.5.3':'1.5.3: sigma = 2.75',
#     '1.5.4':'1.5.4: sigma = 3',
#     '1.5.5':'1.5.5: sigma = 3.5',
#     '1.5.6':'1.5.6: sigma = 4.5',
#     '1.6':'1.6: drop SINNOVPATUS',
#     '1.7':'1.7: drop DOMPATINUS',
#     '1.8':'1.8: drop SINNOVPATUS and DOMPATINUS',
#     '1.9':'1.9: drop SRDUS, SINNOVPATUS and DOMPATINUS',
#     }

# comments_dic['804'] = {'baseline':'bsln:2005',
#     '1.0':'1.0: TO: 0.01',
# '1.1':'1.1: TO: 0.0105',
# '1.2':'1.2: TO: 0.011',
# '1.3':'1.3: TO: 0.0115',
# '1.4':'1.4: TO: 0.012',
# '1.5':'1.5: TO: 0.0125',
# '1.6':'1.6: TO: 0.013',
# '1.7':'1.7: TO: 0.0135',
# '1.8':'1.8: TO: 0.014',
# '1.9':'1.9: TO: 0.0145',
# '1.10':'1.10: TO: 0.015',
# '1.11':'1.11: TO: 0.0155',
# '1.12':'1.12: TO: 0.016',
# '1.13':'1.13: TO: 0.0165',
# '1.14':'1.14: TO: 0.017',
# '1.15':'1.15: TO: 0.0175',
# '1.16':'1.16: TO: 0.018',
# '1.17':'1.17: TO: 0.0185',
# '1.18':'1.18: TO: 0.019',
# '1.19':'1.19: TO: 0.0195',
# '1.20':'1.20: TO: 0.02',
# '1.23':'1.23: TO: TO = 0.022',
# '1.24':'1.24: TO: TO = 0.024',
# '1.25':'1.25: TO: TO = 0.026',
# '1.26':'1.26: TO: TO = 0.028',
# '1.27':'1.27: TO: TO = 0.03',
# '1.40':'1.40: TO: 0.014603',
# '1.41':'1.41: TO: TO = 0.019661',
#     }

# comments_dic['805'] = {'baseline':'bsln:2015',
#     '1.0':'1.0: TO: 0.01',
# '1.1':'1.1: TO: 0.0105',
# '1.2':'1.2: TO: 0.011',
# '1.3':'1.3: TO: 0.0115',
# '1.4':'1.4: TO: 0.012',
# '1.5':'1.5: TO: 0.0125',
# '1.6':'1.6: TO: 0.013',
# '1.7':'1.7: TO: 0.0135',
# '1.8':'1.8: TO: 0.014',
# '1.9':'1.9: TO: 0.0145',
# '1.10':'1.10: TO: 0.015',
# '1.11':'1.11: TO: 0.0155',
# '1.12':'1.12: TO: 0.016',
# '1.13':'1.13: TO: 0.0165',
# '1.14':'1.14: TO: 0.017',
# '1.15':'1.15: TO: 0.0175',
# '1.16':'1.16: TO: 0.018',
# '1.17':'1.17: TO: 0.0185',
# '1.18':'1.18: TO: 0.019',
# '1.19':'1.19: TO: 0.0195',
# '1.20':'1.20: TO: 0.02',
# '1.23':'1.23: TO: TO = 0.022',
# '1.24':'1.24: TO: TO = 0.024',
# '1.25':'1.25: TO: TO = 0.026',
# '1.26':'1.26: TO: TO = 0.028',
# '1.27':'1.27: TO: TO = 0.03',
# '1.40':'1.40: TO: 0.014603',
# '1.41':'1.41: TO: TO = 0.019661',
#     }

# comments_dic['806'] = {
#     "baseline":"baseline : 2015",
#     "1.0" : "1.0 : 1990",
#     "1.1" : "1.1 : 1991",
#     "1.2" : "1.2 : 1992",
#     "1.3" : "1.3 : 1993",
#     "1.4" : "1.4 : 1994",
#     "1.5" : "1.5 : 1995",
#     "1.6" : "1.6 : 1996",
#     "1.7" : "1.7 : 1997",
#     "1.8" : "1.8 : 1998",
#     "1.9" : "1.9 : 1999",
#     "1.10" : "1.10 : 2000",
#     "1.11" : "1.11 : 2001",
#     "1.12" : "1.12 : 2002",
#     "1.13" : "1.13 : 2003",
#     "1.14" : "1.14 : 2004",
#     "1.15" : "1.15 : 2005",
#     "1.16" : "1.16 : 2006",
#     "1.17" : "1.17 : 2007",
#     "1.18" : "1.18 : 2008",
#     "1.19" : "1.19 : 2009",
#     "1.20" : "1.20 : 2010",
#     "1.21" : "1.21 : 2011",
#     "1.22" : "1.22 : 2012",
#     "1.23" : "1.23 : 2013",
#     "1.24" : "1.24 : 2014",
#     "1.25" : "1.25 : 2015",
#     "1.26" : "1.26 : 2016",
#     "1.27" : "1.27 : 2017",
#     "1.28" : "1.28 : 2018",
# }

# comments_dic['807'] = {
#     "baseline":"baseline : 2015",
#     "0.1" : "0.1 : dont drop RD in South",
#     "1.0" : "1.0 : ratio loss",
#     "1.1" : "1.1 : ratio loss and dont drop RD in South",
#     "2.0" : "2.0 : no weight on large flows",
#     "3.0" : "3.0 : ratio loss and no weight on large flows",
# }

# comments_dic['808'] = {
#     'baseline':'baseline',
#     '1.0':'1.0:full calibration 2015',
#     '1.1':'1.1:full calibration 1992',
#     '2.0':'2.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '2.1':'2.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '3.0':'3.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta_US fixed',
#     '3.1':'3.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta_US fixed',
#     '4.0':'4.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta,eta_US fixed',
#     '4.1':'4.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP], delta,eta_US fixed',
#     '5.0':'5.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM]',
#     '5.1':'5.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,KM]',
#     '6.0':'6.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,SINNOVPATUS]',
#     '6.1':'6.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,SINNOVPATUS]',
#     '7.0':'7.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,GROWTH]',
#     '7.1':'7.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,GROWTH]',
#     '8.0':'8.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '8.1':'8.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '9.0':'9.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO]',
#     '9.1':'9.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO]',
#     '17.0':'17.0:[delta,T,eta,g_0], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,GROWTH]',
#     '17.1':'17.1:[delta,T,eta,g_0], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,GROWTH]',
#     '18.0':'18.0:[delta,T,eta,fe,fo], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '18.1':'18.1:[delta,T,eta,fe,fo], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '19.0':'19.0:[delta,T,eta,nu], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO]',
#     '19.1':'19.1:[delta,T,eta,nu], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO]',
#     }

# comments_dic['901'] = {
#     "baseline":"baseline : 2015",
#     '1.0':'1.0:same as bsln',
#     '2.0':'2.0:calibrated theta, new weights',
#     '3.0':'3.0:more weights SPFLOW',
#     '4.0':'4.0:more weights high SPFLOW',
#     '5.0':'5.0:special weight on USA-EUR',
#     '6.0':'6.0:more weight on high SPFLOW',
#     '7.0':'7.0',
#     '8.0':'8.0',
#     '9.0':'9.0',
#     '10.0':'10.0:doubling eta IDN',
#     '11.0':'11.0',
#     '12.0':'12.0',
#     '13.0':'13.0',
#     '14.0':'14.0',
#     '15.0':'15.0:only TE, theta',
#     '16.0':'16.0',
#     '17.0':'17.0',
#     '18.0':'18.0',
#     '19.0':'19.0',
#     '20.0':'20.0',
#     '21.0':'21.0',
#     }

# comments_dic['902'] = {
#     "baseline":"baseline : 2015",
#     "1.0" : "1.0 : 1990",
#     "1.1" : "1.1 : 1991",
#     "1.2" : "1.2 : 1992",
#     "1.3" : "1.3 : 1993",
#     "1.4" : "1.4 : 1994",
#     "1.5" : "1.5 : 1995",
#     "1.6" : "1.6 : 1996",
#     "1.7" : "1.7 : 1997",
#     "1.8" : "1.8 : 1998",
#     "1.9" : "1.9 : 1999",
#     "1.10" : "1.10 : 2000",
#     "1.11" : "1.11 : 2001",
#     "1.12" : "1.12 : 2002",
#     "1.13" : "1.13 : 2003",
#     "1.14" : "1.14 : 2004",
#     "1.15" : "1.15 : 2005",
#     "1.16" : "1.16 : 2006",
#     "1.17" : "1.17 : 2007",
#     "1.18" : "1.18 : 2008",
#     "1.19" : "1.19 : 2009",
#     "1.20" : "1.20 : 2010",
#     "1.21" : "1.21 : 2011",
#     "1.22" : "1.22 : 2012",
#     "1.23" : "1.23 : 2013",
#     "1.24" : "1.24 : 2014",
#     "1.25" : "1.25 : 2015",
#     "1.26" : "1.26 : 2016",
#     "1.27" : "1.27 : 2017",
#     "1.28" : "1.28 : 2018",
# }

# comments_dic['903'] = {
#     "baseline":"baseline : 2015",
#     "1.0" : "1.0 : 1990 smooth 3y",
#     "1.1" : "1.1 : 1991 smooth 3y",
#     "1.2" : "1.2 : 1992 smooth 3y",
#     "1.3" : "1.3 : 1993 smooth 3y",
#     "1.4" : "1.4 : 1994 smooth 3y",
#     "1.5" : "1.5 : 1995 smooth 3y",
#     "1.6" : "1.6 : 1996 smooth 3y",
#     "1.7" : "1.7 : 1997 smooth 3y",
#     "1.8" : "1.8 : 1998 smooth 3y",
#     "1.9" : "1.9 : 1999 smooth 3y",
#     "1.10" : "1.10 : 2000 smooth 3y",
#     "1.11" : "1.11 : 2001 smooth 3y",
#     "1.12" : "1.12 : 2002 smooth 3y",
#     "1.13" : "1.13 : 2003 smooth 3y",
#     "1.14" : "1.14 : 2004 smooth 3y",
#     "1.15" : "1.15 : 2005 smooth 3y",
#     "1.16" : "1.16 : 2006 smooth 3y",
#     "1.17" : "1.17 : 2007 smooth 3y",
#     "1.18" : "1.18 : 2008 smooth 3y",
#     "1.19" : "1.19 : 2009 smooth 3y",
#     "1.20" : "1.20 : 2010 smooth 3y",
#     "1.21" : "1.21 : 2011 smooth 3y",
#     "1.22" : "1.22 : 2012 smooth 3y",
#     "1.23" : "1.23 : 2013 smooth 3y",
#     "1.24" : "1.24 : 2014 smooth 3y",
#     "1.25" : "1.25 : 2015 smooth 3y",
#     "1.26" : "1.26 : 2016 smooth 3y",
#     "1.27" : "1.27 : 2017 smooth 3y",
#     "1.28" : "1.28 : 2018 smooth 3y",
# }

# comments_dic['1001'] = {
#     "baseline":"baseline : 2015",
#     "1.0":"1.0:same as bsln",
#     "2.0":"2.0:less weights on big flows",
#     "3.0":"3.0:1 weight on all moments",
#     "4.0":"4.0:increase weight on SPFLOW",
#     "5.0":"5.0:corect RD",
#     "6.0":"6.0:no weight on high pflow",
# }

# comments_dic['1002'] = {
#     "baseline":"baseline : 2015",
#     "1.0" : "1.0 : 1990 smooth 3y",
#     "1.1" : "1.1 : 1991 smooth 3y",
#     "1.2" : "1.2 : 1992 smooth 3y",
#     "1.3" : "1.3 : 1993 smooth 3y",
#     "1.4" : "1.4 : 1994 smooth 3y",
#     "1.5" : "1.5 : 1995 smooth 3y",
#     "1.6" : "1.6 : 1996 smooth 3y",
#     "1.7" : "1.7 : 1997 smooth 3y",
#     "1.8" : "1.8 : 1998 smooth 3y",
#     "1.9" : "1.9 : 1999 smooth 3y",
#     "1.10" : "1.10 : 2000 smooth 3y",
#     "1.11" : "1.11 : 2001 smooth 3y",
#     "1.12" : "1.12 : 2002 smooth 3y",
#     "1.13" : "1.13 : 2003 smooth 3y",
#     "1.14" : "1.14 : 2004 smooth 3y",
#     "1.15" : "1.15 : 2005 smooth 3y",
#     "1.16" : "1.16 : 2006 smooth 3y",
#     "1.17" : "1.17 : 2007 smooth 3y",
#     "1.18" : "1.18 : 2008 smooth 3y",
#     "1.19" : "1.19 : 2009 smooth 3y",
#     "1.20" : "1.20 : 2010 smooth 3y",
#     "1.21" : "1.21 : 2011 smooth 3y",
#     "1.22" : "1.22 : 2012 smooth 3y",
#     "1.23" : "1.23 : 2013 smooth 3y",
#     "1.24" : "1.24 : 2014 smooth 3y",
#     "1.25" : "1.25 : 2015 smooth 3y",
#     "1.26" : "1.26 : 2016 smooth 3y",
#     "1.27" : "1.27 : 2017 smooth 3y",
#     "1.28" : "1.28 : 2018 smooth 3y",
# }

# comments_dic['1005'] = comments_dic['1002']
# comments_dic['1011'] = comments_dic['1002']

# comments_dic['1003'] = {
#     "baseline":"baseline : 2015",
#     # '0.1':'0.1:better RD targeting',
#     # '0.2':'0.2:better RD and GROWTH targeting',
#     # '0.3':'0.3:better RD and GROWTH/TO/TE targeting',
#     '0.4':'0.4:better RD targeting',
#     '0.5':'0.5:0.4 with different UUPCOST/DOMPATINUS  tension',
#     '1.0':'1.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US fixed',
#     '1.1':'1.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US fixed',
#     '2.0':'2.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '2.1':'2.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '3.0':'3.0:full calibration',
#     '3.1':'3.1:full calibration',
#     '4.0':'4.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TP]',
#     '4.1':'4.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TP]',
#     '5.0':'5.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,inter-TP]',
#     '5.1':'5.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,inter-TP]',
#     }

# comments_dic['1004'] = {
#     "baseline":"baseline : 2015, same as 1003_0.4",
#     '1.0':'1.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US fixed',
#     '1.1':'1.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US fixed',
#     '2.0':'2.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '2.1':'2.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '3.0':'3.0:full calibration',
#     '3.1':'3.1:full calibration',
#     '4.0':'4.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TP]',
#     '4.1':'4.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TP]',
#     '5.0':'5.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,inter-TP]',
#     '5.1':'5.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,inter-TP]',
#     '6.0':'6.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_North fixed',
#     '6.1':'6.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_North fixed',
#     '8.0':'8.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US bertolotti',
#     '8.1':'8.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP],delta_US bertolotti',
#     '9.0':'9.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '9.1':'9.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST]',
#     '9.2':'9.2:1995',
#     '10.0':'10.0:full calibration, delta_US fixed',
#     '10.1':'10.1:full calibration, delta_US fixed',
#     '11.0':'11.0:[delta,T,eta,nu], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO(updated)]',
#     '11.1':'11.1:[delta,T,eta,nu], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,TO(updated)]',
#     '12.0':'12.0:full calibration except delta_US fixed, KM and TO not targeted',
#     '12.1':'12.1:full calibration except delta_US fixed, KM and TO not targeted',
#     '13.0':'13.0:full calibration except delta_US and nu fixed, KM and TO not targeted',
#     '13.1':'13.1:full calibration except delta_US and nu fixed, KM and TO not targeted',
#     '14.0':'14.0:[delta,T,eta,fe,fo], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST], d_US fixed',
#     '14.1':'14.1:[delta,T,eta,fe,fo], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST], d_US fixed',
#     '15.0':'15.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST], d_US fixed',
#     '15.1':'15.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST], d_US fixed',
#     }

# comments_dic['1006'] = {
#     "baseline":"baseline : 2015, same as 1004",
#     '1.0':'1.0:SPFLOWDOM instead of SPFLOW',
#     '2.0':'2.0:DOMPATUS instead of DOMPATINUS',
#     '2.1':'2.1:1992 partial calibration',
#     '3.0':'3.0:DOMPATUS and DOMPATINUS',
#     '3.1':'3.1:1992 partial calibration',
#     '4.0':'4.0:2.0 with higher weight on DOMPATUS',
#     '4.1':'4.1:1992 partial calibration',
#     '5.0':'5.0:3.0 with higher weight on DOMPAT(IN)US',
#     '5.1':'5.1:1992 partial calibration',
#     }

# comments_dic['1010'] = {
#     "baseline":"baseline : 2015, new correction US flows and new TO",
#     '2.0':'2.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '2.1':'2.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '3.0':'3.0:full calibration 2015',
#     '3.1':'3.1:full calibration 1992',
#     '9.0':'9.0:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 2015',
#     '9.1':'9.1:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 1992',
#     '9.2':'9.2:same conditions, 3-year smoothed out data 1992',
#     '10.0':'10.0 corrected mistake denominator in the Gamma function',
#     }

# comments_dic['1020'] = {
#     "baseline":"baseline : 2015, with corrected term in Gamma function",
#     "1.0" : "1.0 : 1990 smooth 3y",
#     "1.1" : "1.1 : 1991 smooth 3y",
#     "1.2" : "1.2 : 1992 smooth 3y",
#     "1.3" : "1.3 : 1993 smooth 3y",
#     "1.4" : "1.4 : 1994 smooth 3y",
#     "1.5" : "1.5 : 1995 smooth 3y",
#     "1.6" : "1.6 : 1996 smooth 3y",
#     "1.7" : "1.7 : 1997 smooth 3y",
#     "1.8" : "1.8 : 1998 smooth 3y",
#     "1.9" : "1.9 : 1999 smooth 3y",
#     "1.10" : "1.10 : 2000 smooth 3y",
#     "1.11" : "1.11 : 2001 smooth 3y",
#     "1.12" : "1.12 : 2002 smooth 3y",
#     "1.13" : "1.13 : 2003 smooth 3y",
#     "1.14" : "1.14 : 2004 smooth 3y",
#     "1.15" : "1.15 : 2005 smooth 3y",
#     "1.16" : "1.16 : 2006 smooth 3y",
#     "1.17" : "1.17 : 2007 smooth 3y",
#     "1.18" : "1.18 : 2008 smooth 3y",
#     "1.19" : "1.19 : 2009 smooth 3y",
#     "1.20" : "1.20 : 2010 smooth 3y",
#     "1.21" : "1.21 : 2011 smooth 3y",
#     "1.22" : "1.22 : 2012 smooth 3y",
#     "1.23" : "1.23 : 2013 smooth 3y",
#     "1.24" : "1.24 : 2014 smooth 3y",
#     "1.25" : "1.25 : 2015 smooth 3y",
#     "1.26" : "1.26 : 2016 smooth 3y",
#     "1.27" : "1.27 : 2017 smooth 3y",
#     "1.28" : "1.28 : 2018 smooth 3y",
#     '2.0':'2.0:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '2.1':'2.1:[delta,T,eta], [SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP]',
#     '3.0':'3.0:full calibration 2015',
#     '3.1':'3.1:full calibration 1992',
#     '9.0':'9.0:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 2015',
#     '9.1':'9.1:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 1992',
#     '9.2':'9.2:same conditions, 3-year smoothed out data 1992',
#     '10.1':'10.1:2015 calibration with 1992 trade costs',
#     '10.2':'10.2:2015 calibration with doubled trade costs in pat sector',
#     }

comments_dic['1030'] = {
    "baseline":"baseline : 2015",
    # '0.102':'0.1020:old baseline',
    # '0.2':'0.2:high weight on prices',
    '3.0':'3.0:full calibration 2015',
    '3.1':'3.1:full calibration 1992',
    # '9.0':'9.0:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 2015',
    # '9.1':'9.1:[delta,T,eta],[SPFLOW,DOMPATINUS,OUT,RD,RP,SRGDP,UUPCOST] 1992',
    '9.0':'9.0: partial calibration 2015',
    '9.1':'9.1: partial calibration 1992',
    '9.2':'9.2:partial calib, 3-year smoothed data 1992',
    '10.2':'10.2:2015 doubled trade costs in pat sector',
    # "11.0" : "11.0 : 1990 smooth 3y",
    # "11.1" : "11.1 : 1991 smooth 3y",
    "11.2" : "11.2 : 1992 smooth 3y",
    "11.3" : "11.3 : 1993 smooth 3y",
    "11.4" : "11.4 : 1994 smooth 3y",
    "11.5" : "11.5 : 1995 smooth 3y",
    "11.6" : "11.6 : 1996 smooth 3y",
    "11.7" : "11.7 : 1997 smooth 3y",
    "11.8" : "11.8 : 1998 smooth 3y",
    "11.9" : "11.9 : 1999 smooth 3y",
    "11.10" : "11.10 : 2000 smooth 3y",
    "11.11" : "11.11 : 2001 smooth 3y",
    "11.12" : "11.12 : 2002 smooth 3y",
    "11.13" : "11.13 : 2003 smooth 3y",
    "11.14" : "11.14 : 2004 smooth 3y",
    "11.15" : "11.15 : 2005 smooth 3y",
    "11.16" : "11.16 : 2006 smooth 3y",
    "11.17" : "11.17 : 2007 smooth 3y",
    "11.18" : "11.18 : 2008 smooth 3y",
    "11.19" : "11.19 : 2009 smooth 3y",
    "11.20" : "11.20 : 2010 smooth 3y",
    "11.21" : "11.21 : 2011 smooth 3y",
    "11.22" : "11.22 : 2012 smooth 3y",
    "11.23" : "11.23 : 2013 smooth 3y",
    "11.24" : "11.24 : 2014 smooth 3y",
    "11.25" : "11.25 : 2015 smooth 3y",
    # "11.26" : "11.26 : 2016 smooth 3y",
    # "11.27" : "11.27 : 2017 smooth 3y",
    # "11.28" : "11.28 : 2018 smooth 3y",
    # '97.1':'97.1: Sensitivity SRGDP_EUR',
    # '97.3':'97.3: Sensitivity SRGDP_CHN',
    # '98.1':'98.1: Sensitivity RP_EUR',
    # '98.2':'98.2: Sensitivity RP_EUR with high weight on prices',
    '99.0':'99.0: Low TO',
    '99.1':'99.1: High TO',
    '99.2':'99.2: Low TE',
    '99.3':'99.3: High TE',
    '99.4':'99.4: Low KM',
    '99.5':'99.5: High KM',
    '99.6':'99.6: Low Sigma',
    '99.7':'99.7: High Sigma',
    '99.8':'99.8: Low Kappa',
    '99.9':'99.9: High Kappa',
    '99.10':'99.10: Low Growth',
    '99.11':'99.11: High Growth',
    '99.12':'99.12: Low rho',
    '99.13':'99.13: High rho',
    '99.14':'99.14: Low UUPCOST',
    '99.15':'99.15: High UUPCOST',
    # '199.0':'199.0: Low TO 1992',
    # '199.1':'199.1: High TO 1992',
    # '199.2':'199.2: Low TE 1992',
    # '199.3':'199.3: High TE 1992',
    # '199.4':'199.4: Low KM 1992',
    # '199.5':'199.5: High KM 1992',
    # '199.6':'199.6: Low Sigma 1992',
    # '199.7':'199.7: High Sigma 1992',
    # '199.8':'199.8: Low Kappa 1992',
    # '199.9':'199.9: High Kappa 1992',
    # '199.10':'199.10: Low Growth 1992',
    # '199.11':'199.11: High Growth 1992',
    # '199.12':'199.12: Low rho 1992',
    # '199.13':'199.13: High rho 1992',
    # '199.14':'199.14: Low UUPCOST 1992',
    # '199.15':'199.15: High UUPCOST 1992',
    }

comments_dic['1040'] = {
    "baseline":"baseline : tariff = 0",
    '1.0':'1.0:tariff = 1%',
    '2.0':'2.0:tariff = 5%',
    '3.0':'3.0:tariff = 10%',
    '4.0':'4.0:tariff = 50%',
    '5.0':'5.0:tariff = 100%',
    }

comments_dic['1050'] = {
    "baseline":"baseline : 2015 with data tariffs",
    '9.2':'9.2:partial calib, 3-year smoothed data 1992',
    # "11.2" : "11.2 : 1992 smooth 3y",
    # "11.3" : "11.3 : 1993 smooth 3y",
    # "11.4" : "11.4 : 1994 smooth 3y",
    # "11.5" : "11.5 : 1995 smooth 3y",
    # "11.6" : "11.6 : 1996 smooth 3y",
    # "11.7" : "11.7 : 1997 smooth 3y",
    # "11.8" : "11.8 : 1998 smooth 3y",
    # "11.9" : "11.9 : 1999 smooth 3y",
    # "11.10" : "11.10 : 2000 smooth 3y",
    # "11.11" : "11.11 : 2001 smooth 3y",
    # "11.12" : "11.12 : 2002 smooth 3y",
    # "11.13" : "11.13 : 2003 smooth 3y",
    # "11.14" : "11.14 : 2004 smooth 3y",
    # "11.15" : "11.15 : 2005 smooth 3y",
    # "11.16" : "11.16 : 2006 smooth 3y",
    # "11.17" : "11.17 : 2007 smooth 3y",
    # "11.18" : "11.18 : 2008 smooth 3y",
    # "11.19" : "11.19 : 2009 smooth 3y",
    # "11.20" : "11.20 : 2010 smooth 3y",
    # "11.21" : "11.21 : 2011 smooth 3y",
    # "11.22" : "11.22 : 2012 smooth 3y",
    # "11.23" : "11.23 : 2013 smooth 3y",
    # "11.24" : "11.24 : 2014 smooth 3y",
    # "11.25" : "11.25 : 2015 smooth 3y"
    }

comments_dic['1050'] = {
    "baseline":"baseline : 2015 with zero tariffs",
    '9.2':'9.2:partial calib, 3-year smoothed data 1992'
    }

baselines_dic_param = {}
baselines_dic_mom = {}
baselines_dic_sol_qty = {}

# baseline_list = ['311','312','401','402','403']    
# baseline_list = ['402','403','404']    
# baseline_list = ['403','404','405']    
# baseline_list = ['501','607','608','609','610','614','615','616','617']    
# baseline_list = ['618','701','702']    
# baseline_list = ['901','803','806','808']    
# baseline_list = ['1030','1040','1050']    
# baseline_list = ['1050']    
baseline_list = ['1200']    
baseline_mom = '1200'

def section(s):
     return [int(_) for _ in s.split(".")]
 
for baseline_nbr in baseline_list:
    print(baseline_nbr)
    print(time.perf_counter() - start)
    baseline_path = results_path+baseline_nbr+'/'
    baseline_variations_path = results_path+'baseline_'+baseline_nbr+'_variations/'
    p_baseline,m_baseline,sol_baseline = load(baseline_path,data_path = data_path,
                                              dir_path=dir_path)
    # print(baseline_nbr)
    baselines_dic_param[baseline_nbr], baselines_dic_mom[baseline_nbr], baselines_dic_sol_qty[baseline_nbr]\
        = init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,sol_baseline,list_of_moments)
    try:
        files_in_dir = next(os.walk(baseline_variations_path))[1]
        run_list = [f for f in files_in_dir if f[0].isnumeric()]
        # lists = sorted([s.split('.') for s in run_list], key=lambda x:map(int, x))  
        # run_list#.sort()
        run_list = sorted(run_list, key=section)
    
        for run in run_list:
            # if run not in ['2.1.9','99'] and not run.startswith('99'):
            if run in comments_dic[baseline_nbr]:
                p_to_add,m_to_add,sol_to_add = load(baseline_variations_path+run+'/',
                                                    data_path = data_path,
                                                    dir_path=dir_path)
                a, b, c  = append_dic_of_dataframes_with_variation(baselines_dic_param[baseline_nbr], 
                                                                baselines_dic_mom[baseline_nbr], 
                                                                baselines_dic_sol_qty[baseline_nbr],
                                                                p_to_add, 
                                                                m_to_add, 
                                                                sol_to_add,
                                                                run)
                baselines_dic_param[baseline_nbr] = a
                baselines_dic_mom[baseline_nbr] = b
                baselines_dic_sol_qty[baseline_nbr] = c
    except:
        pass

# gather full list run
full_run_list = []
for baseline_nbr in baseline_list:
    baseline_path = results_path+baseline_nbr+'/'
    baseline_variations_path = results_path+'baseline_'+baseline_nbr+'_variations/'
    files_in_dir = next(os.walk(baseline_variations_path))[1]
    for f in files_in_dir:
        if f[0].isnumeric() and f not in full_run_list and f in comments_dic[baseline_nbr]:
            full_run_list.append(f)
full_run_list = ['target','baseline']+sorted(full_run_list,key = section)
#add empty columns to dfs
for baseline_nbr in baseline_list:
    for df_name in baselines_dic_mom[baseline_nbr].keys():
        baselines_dic_mom[baseline_nbr][df_name] = baselines_dic_mom[baseline_nbr][df_name].reindex(columns=full_run_list)
    for df_name in baselines_dic_param[baseline_nbr].keys():
        baselines_dic_param[baseline_nbr][df_name] = baselines_dic_param[baseline_nbr][df_name].reindex(columns=full_run_list[1:])
    for df_name in baselines_dic_sol_qty[baseline_nbr].keys():
        baselines_dic_sol_qty[baseline_nbr][df_name] = baselines_dic_sol_qty[baseline_nbr][df_name].reindex(columns=full_run_list[1:])

countries = p_baseline.countries

TOOLS="pan,wheel_zoom,box_zoom,reset,save"

# baseline_mom = '101'
# baseline_mom = '618'

mom = 'SPFLOW'

baseline_mom_select = Select(value=baseline_mom, title='Baseline', options=sorted(baselines_dic_mom.keys()))
mom_select = Select(value=mom, title='Quantity', options=sorted(baselines_dic_mom[baseline_mom].keys()))
x_mom_select = Select(value='baseline', title='x-axis target', options=list(comments_dic[baseline_mom].keys()))
labels_mom_toggle = Toggle(label="Labels On/Off",align='end')

def update_x_axis_mom_matching_options(attr, old, new):
    x_mom_select.options = list(comments_dic[new].keys())

ds_mom = ColumnDataSource(baselines_dic_mom[baseline_mom][mom])
p_mom = figure(title="Moment matching", 
               width = 1200,
               height = 875,
                x_axis_type="log",
                y_axis_type="log",
                x_axis_label='Target', 
                y_axis_label='Model implied',
                tools = TOOLS)
hover_tool_mom = HoverTool()
hover_tool_mom.tooltips = [
    ("index", "@x"),
    ("(target,value)", "($x,$y)"),
    ]
labels_mom = LabelSet(x='target', y='baseline', text='x',
              x_offset=2, y_offset=2, source=ds_mom, text_font_size="7pt")
p_mom.add_layout(labels_mom)
p_mom.add_tools(hover_tool_mom)
slope1 = Slope(gradient=1, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=1)
# slope2 = Slope(gradient=1.4876, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)
# slope3 = Slope(gradient=0.5124, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)
# slope4 = Slope(gradient=0.756198, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)
# slope5 = Slope(gradient=1.546, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)
# slope6 = Slope(gradient=2.20, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)

for slope in [slope1]:
# for slope in [slope1,slope2,slope3,slope4,slope5,slope6]:
    p_mom.add_layout(slope)
    
# slope2.visible = False
# slope3.visible = False
# slope4.visible = False
# slope5.visible = False
# slope6.visible = False

colors_mom = itertools.cycle(Category18)

lines_mom = {}
# for i,col in enumerate(ds_mom.data.keys()):
for i,col in enumerate(ds_mom.data.keys()):
    if col not in ['x','target']:
        lines_mom[col] = p_mom.circle('target', col, 
                                      source = ds_mom, 
                                      size=5, color=next(colors_mom))
        if col != 'baseline':
            lines_mom[col].visible = False
            
legend_items_mom = [LegendItem(label=comments_dic[baseline_mom][col], renderers=[lin_mom]) 
                    for col, lin_mom in lines_mom.items() if col in comments_dic[baseline_mom]]
# legend_items_mom = [LegendItem(label=comments_dic[baseline_mom][col], renderers=[lines_mom[i]]) for i,col in enumerate(ds_mom.data)]
# legend_mom = Legend(items=legend_items_mom, click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0)
# p_mom.add_layout(legend_mom, 'right')

legend_mom_split_1 = Legend(items=legend_items_mom[:round((len(legend_items_mom)+1)/2)], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    # location=(10, -60)
                    )
legend_mom_split_2 = Legend(items=legend_items_mom[round((len(legend_items_mom)+1)/2):], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0
                    # , location=(10, -60)
                    )
p_mom.add_layout(legend_mom_split_1, 'right')
p_mom.add_layout(legend_mom_split_2, 'right')
# columns_mom = [TableColumn(field=col) for col in list(ds_mom.data.keys())]
columns_mom = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in ['target']+list(comments_dic[baseline_mom].keys())]
data_table_mom = DataTable(source=ds_mom, columns = columns_mom, width=1200, height=400)
    
def update_baseline_mom(attrname, old, new):
    mom = mom_select.value
    ds_mom.data = baselines_dic_mom[new][mom]
    
    # legend_items_mom = [LegendItem(label=comments_dic[new][col], 
    #                                 renderers=[lines_mom[i]]) for i,col in enumerate(ds_mom.data) if col not in ['x','target']]
    legend_items_mom = [LegendItem(label=comments_dic[new][col], renderers=[lines_mom[col]]) 
                        for col in ds_mom.data if col in comments_dic[new]]
    # legend_mom.items = legend_items_mom
    legend_mom_split_1.items = legend_items_mom[:round((len(legend_items_mom)+1)/2)]
    legend_mom_split_2.items = legend_items_mom[round((1+len(legend_items_mom))/2):]
    data_table_mom.columns = [
            TableColumn(field="x"),
        ]+[TableColumn(field=col) for col in ['target']+list(comments_dic[new].keys())]
    x_mom_select.value = 'baseline'
    
def update_mom(attrname, old, new):
    baseline_mom = baseline_mom_select.value
    ds_mom.data = baselines_dic_mom[baseline_mom][new]
    # if new == 'scalars':
    #     slope2.visible = True
    #     slope3.visible = True
    #     slope4.visible = True
    # else:
    #     slope2.visible = False
    #     slope3.visible = False
    #     slope4.visible = False
    x_mom_select.value = 'baseline'
        
def update_x_axis_target(attrname, old, new):
    baseline_mom = baseline_mom_select.value
    mom = mom_select.value
    df_temp = ds_mom.data.copy()
    if new == 'baseline':
        path_x_axis = results_path+baseline_mom+'/'
    else:
        path_x_axis = results_path+'baseline_'+baseline_mom+'_variations/'+new+'/'
    if mom != 'scalars':
        m_temp = moments()
        m_temp.load_run(path_x_axis,
        dir_path=dir_path)
        df_temp['target'] = getattr(m_temp,mom+'_target').ravel()
        # df_temp['target'] = pd.read_csv(path_x_axis+mom)['target']
    else:
        m_temp = moments()
        m_temp.load_run(path_x_axis,
        dir_path=dir_path)
        for i,x in enumerate(df_temp['x']):
            if x != 'objective':
                df_temp['target'][i] = float(getattr(m_temp,x+'_target'))
    ds_mom.data = df_temp
    
def toggle_labels(event):
    labels_mom.visible = not labels_mom.visible
    
controls_mom = row(baseline_mom_select, mom_select, x_mom_select, labels_mom_toggle)

baseline_mom_select.on_change('value', update_baseline_mom)
baseline_mom_select.on_change('value', update_x_axis_mom_matching_options)
labels_mom_toggle.on_click(toggle_labels)

mom_select.on_change('value', update_mom)
x_mom_select.on_change('value', update_x_axis_target)

baseline_par = baseline_mom
par = 'delta'

baseline_par_select = Select(value=baseline_par, title='Baseline', options=sorted(baselines_dic_param.keys()))
par_select = Select(value=par, title='Quantity', options=sorted(baselines_dic_param[baseline_par].keys()))

country_sort = {
    'USA':	1,
    'JAP':	2,
    'CAN':	3,
    'ZAF':	13,
    'EUR':	5,
    'KOR':	6,
    'MEX':	7,
    'RUS':	8,
    'BRA':	9,
    'ROW':	10,
    'CHN':	11,
    'IND':	12,
    'IDN':	14
    }

x_range = baselines_dic_param[baseline_par][par_select.value].index.to_list()
x_range = sorted(x_range, key = country_sort.get)
ds_par = ColumnDataSource(baselines_dic_param[baseline_par][par].loc[x_range])
p_par = figure(title="Parameters", 
               width = 1200,
               height = 875,
           x_range = x_range,
           y_axis_label='Model implied',
           tools = TOOLS)
hover_tool_par = HoverTool()
hover_tool_par.tooltips = [
    ("index", "@x"),
    ("value", "$y")
    ]

p_par.add_tools(hover_tool_par)
colors_par = itertools.cycle(Category18)
lines_par = {}

for col in baselines_dic_param[baseline_par][par].columns:
    lines_par[col] = p_par.line(x='x', y=col, source = ds_par, color=next(colors_par),
                                line_width = 2)
    if col != 'baseline':
        lines_par[col].visible = False

legend_items_par = [LegendItem(label=comments_dic[baseline_par][col], renderers=[lin_par])
                    for col, lin_par in lines_par.items() if col in comments_dic[baseline_par]]
# legend_par = Legend(items=legend_items_par, click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0, 
#                     )
# p_par.add_layout(legend_par, 'right')

legend_par_split_1 = Legend(items=legend_items_par[:round((len(legend_items_par)+1)/2)], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    )
legend_par_split_2 = Legend(items=legend_items_par[round((1+len(legend_items_par))/2):], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0
                    )
p_par.add_layout(legend_par_split_1, 'right')
p_par.add_layout(legend_par_split_2, 'right')

columns_par = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in list(comments_dic[baseline_par].keys())]

data_table_par = DataTable(source=ds_par, columns = columns_par, width=1200, height=400)

def update_baseline_par(attrname, old, new):
    par = par_select.value
    x_range_factors = baselines_dic_param[new][par].index.to_list()
    if new != 'scalars':
        x_range_factors = sorted(x_range_factors, key = country_sort.get)
    ds_par.data = baselines_dic_param[new][par].loc[x_range_factors]
    legend_items_par = [LegendItem(label=comments_dic[new][col], renderers=[lines_par[col]])
                        for col in ds_par.data if col in comments_dic[new]]
    # legend_par.items = legend_items_par
    legend_par_split_1.items = legend_items_par[:round((1+len(legend_items_par))/2)]
    legend_par_split_2.items = legend_items_par[round((len(legend_items_par)+1)/2):]
                      
    data_table_par.columns = [
            TableColumn(field="x"),
        ]+[TableColumn(field=col) for col in list(comments_dic[new].keys())]

def update_par(attrname, old, new):
    baseline_par = baseline_par_select.value
    x_range_factors = baselines_dic_param[baseline_par][new].index.to_list()
    if new != 'scalars':
        x_range_factors = sorted(x_range_factors, key = country_sort.get)
    p_par.x_range.factors = x_range_factors
    ds_par.data = baselines_dic_param[baseline_par][new].loc[x_range_factors]

controls_par = row(baseline_par_select, par_select)

baseline_par_select.on_change('value', update_baseline_par)
par_select.on_change('value', update_par)

baseline_sol_qty = baseline_mom
sol_qty = 'psi_o_star'

baseline_sol_qty_select = Select(value=baseline_sol_qty, title='Baseline', options=sorted(baselines_dic_sol_qty.keys()))
sol_qty_select = Select(value=sol_qty, title='Quantity', options=sorted(baselines_dic_sol_qty[baseline_sol_qty].keys()))
x_range_par = baselines_dic_sol_qty[baseline_sol_qty][sol_qty_select.value].index.to_list()
x_range_par = sorted(x_range_par, key = country_sort.get)
ds_sol_qty = ColumnDataSource(baselines_dic_sol_qty[baseline_sol_qty][sol_qty].loc[x_range_par])
p_sol_qty = figure(title="Solution quantities", 
                width = 1200,
                height = 875,
            x_range = x_range,
            y_axis_label='Model implied',
            tools = TOOLS)
hover_tool_sol_qty = HoverTool()
hover_tool_sol_qty.tooltips = [
    ("index", "@x"),
    ("value", "$y")
    ]

p_sol_qty.add_tools(hover_tool_sol_qty)
colors_sol_qty = itertools.cycle(Category18)
lines_sol_qty = {}

for col in baselines_dic_sol_qty[baseline_sol_qty][sol_qty].columns:
    lines_sol_qty[col] = p_sol_qty.line(x='x', y=col, source = ds_sol_qty, 
                                        color=next(colors_sol_qty),
                                line_width = 2)
    if col != 'baseline':
        lines_sol_qty[col].visible = False

legend_items_sol_qty = [LegendItem(label=comments_dic[baseline_sol_qty][col], renderers=[lin_sol_qty]) 
                        for col, lin_sol_qty in lines_sol_qty.items() if col in comments_dic[baseline_sol_qty]]

# legend_sol_qty = Legend(items=legend_items_sol_qty, click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0, 
#                     )
# p_sol_qty.add_layout(legend_sol_qty, 'right')

legend_sol_qty_split_1 = Legend(items=legend_items_sol_qty[:round((len(legend_items_sol_qty)+1)/2)], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    )
legend_sol_qty_split_2 = Legend(items=legend_items_sol_qty[round((len(legend_items_sol_qty)+1)/2):], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0
                    )
p_sol_qty.add_layout(legend_sol_qty_split_1, 'right')
p_sol_qty.add_layout(legend_sol_qty_split_2, 'right')


columns_sol_qty = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in list(comments_dic[baseline_sol_qty].keys())]

data_table_sol_qty = DataTable(source=ds_sol_qty, columns = columns_sol_qty, width=1200, height=400)

def update_baseline_sol_qty(attrname, old, new):
    sol_qty = sol_qty_select.value
    x_range_factors = baselines_dic_sol_qty[new][sol_qty].index.to_list()
    if new != 'scalars':
        x_range_factors = sorted(x_range_factors, key = country_sort.get)
    ds_sol_qty.data = baselines_dic_sol_qty[new][sol_qty].loc[x_range_factors]
    legend_items_sol_qty = [LegendItem(label=comments_dic[new][col], renderers=[lines_sol_qty[col]]) 
                            for col in ds_sol_qty.data  if col in comments_dic[new]]
    # legend_sol_qty.items = legend_items_sol_qty
    legend_sol_qty_split_1.items = legend_items_sol_qty[:round((len(legend_items_sol_qty)+1)/2)]
    legend_sol_qty_split_2.items = legend_items_sol_qty[round((len(legend_items_sol_qty)+1)/2):]
    data_table_sol_qty.columns = [TableColumn(field=col) for col in list(comments_dic[new].keys())]
    
def update_sol_qty(attrname, old, new):
    baseline_sol_qty = baseline_sol_qty_select.value
    # p_sol_qty.x_range.factors = baselines_dic_sol_qty[baseline_sol_qty][new].index.to_list()
    ds_sol_qty.data = baselines_dic_sol_qty[baseline_sol_qty][new].loc[x_range_par]

controls_sol_qty = row(baseline_sol_qty_select, sol_qty_select)

baseline_sol_qty_select.on_change('value', update_baseline_sol_qty)
sol_qty_select.on_change('value', update_sol_qty)

moment_report = column(controls_mom,p_mom,data_table_mom)
param_report = column(controls_par, p_par, data_table_par)
sol_qty_report = column(controls_sol_qty, p_sol_qty, data_table_sol_qty)

#!!! first panel
first_panel = row(moment_report,param_report,sol_qty_report)
# first_panel = row(moment_report,param_report)
print(time.perf_counter() - start)

#%% Time series

# baseline_time = '1050'
# # baseline_time_list = ['607','608','609','610','614','615','616','617']    
# # baseline_time_list = ['607','806','903']
# baseline_time_list = ['1050']
# par_time = 'delta'
# par_time_select = Select(value=par_time, title='Quantity', options=sorted(baselines_dic_param[baseline_time].keys()))
# baseline_time_select = Select(value=baseline_time, title='Baseline', options=baseline_time_list)


# years_time = [y for y in range(1992,2016)]
# runs_time = ['99.'+str(i) for i in range(2,26)]

# def build_time_series(baseline_time,par_time):
#     # df = baselines_dic_param[baseline_time][par_time].T.reindex(
#     #     columns=countries+baselines_dic_param[baseline_time]['scalars'].index.to_list()
#     #     )
#     df = baselines_dic_param[baseline_time][par_time].copy()
#     df = df[runs_time]
#     # print(df)
#     df.columns = years_time
#     df = df.T
#     df = df.reindex(
#         columns=countries+baselines_dic_param[baseline_time]['scalars'].index.to_list()
#         )
#     df.index.name = 'year'
#     return df

# df_par_time = build_time_series(baseline_time,par_time)
# ds_par_time = ColumnDataSource(df_par_time)
# p_par_time = figure(title="Time series", 
#                 width = 1500,
#                 height = 850,
#             y_axis_label='Parameter',
#             tools = TOOLS)
# hover_tool_par_time = HoverTool()
# hover_tool_par_time.tooltips = [
#     ("Year", "@year"),
#     ("value", "$y")
#     ]

# p_par_time.add_tools(hover_tool_par_time)
# colors_par_time = itertools.cycle(Category18)
# lines_par_time = {}

# for col in df_par_time.columns:
#     if col != 'kappa':
#         lines_par_time[col] = p_par_time.line(x='year', y=col, 
#                                         source = ds_par_time, 
#                                         color=next(colors_par_time),
#                                         line_width = 2,
#                                         # legend_label=col
#                                         )

# legend_items_par_time = [LegendItem(label=col, renderers=[lines_par_time[col]]) 
#                         for col in countries]
# legend_par_time = Legend(items=legend_items_par_time, click_policy="hide", 
#                     label_text_font_size="10pt",
#                     )
# p_par_time.add_layout(legend_par_time , 'right')
    
# def update_par_time(attrname, old, new):
#     df_par_time = build_time_series(baseline_time_select.value,new)
#     ds_par_time.data = df_par_time
#     if new!='scalars':
#         legend_items_par_time = [LegendItem(label=col, renderers=[lines_par_time[col]]) 
#                                 for col in countries]
#     else:
#         legend_items_par_time = [LegendItem(label=col, renderers=[lines_par_time[col]]) 
#                                 for col in baselines_dic_param[baseline_time]['scalars'].index.to_list() if col != 'kappa']
#     legend_par_time.items = legend_items_par_time
    
# def update_baseline_time(attrname, old, new):
#     df_par_time = build_time_series(new,par_time_select.value)
#     ds_par_time.data = df_par_time
#     if new!='scalars':
#         legend_items_par_time = [LegendItem(label=col, renderers=[lines_par_time[col]]) 
#                                 for col in countries]
#     else:
#         legend_items_par_time = [LegendItem(label=col, renderers=[lines_par_time[col]]) 
#                                 for col in baselines_dic_param[baseline_time]['scalars'].index.to_list() if col != 'kappa']
#     legend_par_time.items = legend_items_par_time

# controls_par_time = row(baseline_time_select,par_time_select)

# par_time_select.on_change('value', update_par_time)
# baseline_time_select.on_change('value', update_baseline_time)

# par_time_report = column(controls_par_time, p_par_time)    

# # explication_calib_params = Div(text=
# #                           "607 variations : <br> \
# #                               calibrated parameters : eta,k,fe,T,zeta,g_0,delta,nu,fo,theta <br> \
# #                                 targeted moments : GPDIFF,GROWTH,KM,OUT,RD,RP,SRDUS,SRGDP,SINNOVPATUS,\
# #                                     TO,SPFLOW,UUPCOST,SINNOVPATEU,DOMPATINUS,DOMPATINEU,TE<br> \
# #                           608 variations : <br> \
# #                               calibrated parameters : eta,<u><b>fe</b></u>,T,delta,<u><b>fo</b></u> <br> \
# #                                 targeted moments : OUT,RD,RP,SRGDP,SINNOVPATUS,\
# #                                     SPFLOW,<u><b>UUPCOST</b></u>,SINNOVPATEU,DOMPATINUS,DOMPATINEU<br> \
# #                           609 variations :<br> \
# #                               calibrated parameters : eta,T,delta <br> \
# #                                 targeted moments : OUT,RD,RP,SRGDP,SINNOVPATUS,\
# #                                     SPFLOW,SINNOVPATEU,DOMPATINUS,DOMPATINEU<br> \
# #                           610 variations :<br> \
# #                               calibrated parameters : eta,T,delta <br> \
# #                                 targeted moments : OUT,RD,RP,<u><b>SRDUS</b></u>,SRGDP,SINNOVPATUS,\
# #                                     SPFLOW,SINNOVPATEU,DOMPATINUS,DOMPATINEU<br> \
# #                           ")

# #!!! second_panel
# # second_panel = row(par_time_report, explication_calib_params)
# second_panel = row(par_time_report)


#%% counterfactuals

# baseline_cf = '101'
baseline_cf = '1200'
country_cf = 'USA'

def section_end(s):
      return [int(_) for _ in s.split("_")[-1].split(".")]
# cf_list = sorted([s for s in os.listdir(cf_path) 
#             if s[9:].startswith('604') and s.startswith('baseline')], key=section_end)+\
cf_list = sorted([s for s in os.listdir(cf_path) 
                if s[9:].startswith('1200') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #                 if s[9:].startswith('803') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #                 if s[9:].startswith('804') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('805') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('608') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('609') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('618') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #         if s[9:].startswith('501') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('601') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #         if s[9:].startswith('602') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #         if s[9:].startswith('603') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('404') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('312') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #         if s[9:].startswith('311') and s.startswith('baseline')], key=section_end)

baseline_cf_select = Select(value=baseline_cf, title='Baseline', options=[s[9:] for s in cf_list])
country_cf_select = Select(value=country_cf, 
                            title='Country', 
                            options=countries+['World','Harmonizing','Upper_harmonizing',
                                               'Uniform_delta','Upper_uniform_delta'])

def get_data_cf(baseline,country):
    df_cf = pd.read_csv(cf_path+'baseline_'+baseline+'/'+country+'.csv')
    if country != 'Harmonizing':
        df_cf['Growth rate'] = df_cf['growth']/df_cf.loc[np.argmin(np.abs(df_cf.delt-1))].growth
    if country == 'Harmonizing':
        df_cf['Growth rate'] = df_cf['growth']/df_cf.loc[np.argmin(np.abs(df_cf.delt))].growth
    df_cf.set_index('delt',inplace=True)
    return df_cf

def build_max(df_cf):
    df_max = pd.concat([df_cf.idxmax(),df_cf.max()],axis=1)
    df_max.index.name = 'label'
    df_max.columns = ['xmax','max'] 
    df_max = df_max.loc[countries]
    df_max['colors'] = Category18[:len(df_max)]
    return df_max

df_cf = get_data_cf(baseline_cf,country_cf)
ds_cf = ColumnDataSource(df_cf)
df_cf_max = build_max(df_cf)
ds_cf_max = ColumnDataSource(df_cf_max)

colors_cf = itertools.cycle(Category18)
colors_cf_max = itertools.cycle(Category18)

p_cf = figure(title="Patent protection counterfactual", 
                width = 1200,
                height = 850,
                x_axis_label='Change in delta',
                y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
                x_axis_type="log",
                tools = TOOLS) 

for col in df_cf.columns:
    if col not in [0,'delt','growth']:
        p_cf.line(x='delt', y=col, source = ds_cf, color=next(colors_cf),line_width = 2, legend_label=col)

p_cf.circle(x = 'xmax', y = 'max', source = ds_cf_max, size=4, color='colors')
     
p_cf.legend.click_policy="hide"
p_cf.legend.label_text_font_size = '8pt'
p_cf.add_layout(p_cf.legend[0], 'right')

def update_baseline_cf(attrname, old, new):
    country_cf = country_cf_select.value
    ds_cf.data = get_data_cf(new,country_cf)
    df_cf = get_data_cf(new,country_cf)
    ds_cf.data = df_cf
    ds_cf_max.data = build_max(df_cf)
    
def update_country_cf(attrname, old, new):
    baseline_cf = baseline_cf_select.value
    df_cf = get_data_cf(baseline_cf,new)
    ds_cf.data = df_cf
    ds_cf_max.data = build_max(df_cf)
    
controls_cf = row(baseline_cf_select, country_cf_select)

baseline_cf_select.on_change('value', update_baseline_cf)
country_cf_select.on_change('value', update_country_cf)

counterfactuals_report = column(controls_cf,p_cf)

#%% counterfactuals 805 TO target

# country_to_cf = 'USA'
# to_target = 0.0155
# baseline_to_cf = '804'

# # list_of_to_targets = np.linspace(0.01,0.03,41)
# list_of_to_targets = np.array(np.linspace(0.01,0.02,21).tolist()
#                               +[0.022,0.024,0.026,0.028,0.03])

# def section_end(s):
#       return [int(_) for _ in s.split("_")[-1].split(".")]
# cf_to_list = {'804':sorted([s for s in os.listdir(cf_path) 
#             if s[9:].startswith('804') and s.startswith('baseline')], key=section_end),
#               '805':sorted([s for s in os.listdir(cf_path) 
#             if s[9:].startswith('805') and s.startswith('baseline')], key=section_end)}

# def get_data_to_cf(to_target,country,baseline_to_cf):
#     idx_to_cf = np.argmin(np.abs(list_of_to_targets-to_target))
#     df_to_cf = pd.read_csv(cf_path+cf_to_list[baseline_to_cf][min(idx_to_cf,len(cf_to_list[baseline_to_cf])-1)]+'/'+country+'.csv')
#     if country == 'Harmonizing':
#         df_to_cf['Growth rate'] = df_to_cf['growth']/df_to_cf.loc[np.argmin(np.abs(df_to_cf.delt))].growth
#     elif country == 'Uniform_delta':
#         df_to_cf['Growth rate'] = np.nan
#     else:
#         df_to_cf['Growth rate'] = df_to_cf['growth']/df_to_cf.loc[np.argmin(np.abs(df_to_cf.delt-1))].growth
#     df_to_cf.set_index('delt',inplace=True)
#     return df_to_cf

# def build_max(df_to_cf):
#     df_max = pd.concat([df_to_cf.idxmax(),df_to_cf.max()],axis=1)
#     df_max.index.name = 'label'
#     df_max.columns = ['xmax','max'] 
#     df_max = df_max.loc[countries]
#     df_max['colors'] = Category18[:len(df_max)]
#     return df_max

# baseline_to_cf_select = Select(value=baseline_to_cf, title='Baseline', options=['804','805'])
# country_to_cf_select = Select(value=country_to_cf, 
#                             title='Country', 
#                             options=countries+['World','Harmonizing','Uniform_delta'])

# df_to_cf = get_data_to_cf(to_target,country_to_cf,baseline_to_cf)
# ds_to_cf = ColumnDataSource(df_to_cf)
# df_to_cf_max = build_max(df_to_cf)
# ds_to_cf_max = ColumnDataSource(df_to_cf_max)

# colors_to_cf = itertools.cycle(Category18)
# colors_to_cf_max = itertools.cycle(Category18)

# p_to_cf = figure(title="Patent protection counterfactual as function of TO target, baselines 804(2005) and 805 (2015)", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Change in delta',
#                 y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
#                 x_axis_type="log",
#                 tools = TOOLS) 

# for col in df_to_cf.columns:
#     if col not in [0,'delt','growth']:
#         p_to_cf.line(x='delt', y=col, source = ds_to_cf, color=next(colors_to_cf),line_width = 2, legend_label=col)

# p_to_cf.circle(x = 'xmax', y = 'max', source = ds_to_cf_max, size=4, color='colors')
     
# p_to_cf.legend.click_policy="hide"
# p_to_cf.legend.label_text_font_size = '8pt'
# p_to_cf.add_layout(p_to_cf.legend[0], 'right')

# def update_target_to_cf(attrname, old, new):
#     country_to_cf = country_to_cf_select.value
#     baseline_to_cf = baseline_to_cf_select.value
#     df_to_cf = get_data_to_cf(new/100,country_to_cf,baseline_to_cf)
#     ds_to_cf.data = df_to_cf
#     ds_to_cf_max.data = build_max(df_to_cf)
    
# def update_baseline_to_cf(attrname, old, new):
#     country_to_cf = country_to_cf_select.value
#     to_target = slider_to_cf.value/100
#     df_to_cf = get_data_to_cf(to_target,country_to_cf,new)
#     ds_to_cf.data = df_to_cf
#     ds_to_cf_max.data = build_max(df_to_cf)
    
# def update_country_to_cf(attrname, old, new):
#     to_target = slider_to_cf.value/100
#     baseline_to_cf = baseline_to_cf_select.value
#     df_to_cf = get_data_to_cf(to_target,new,baseline_to_cf)
#     ds_to_cf.data = df_to_cf
#     ds_to_cf_max.data = build_max(df_to_cf)
    
# slider_to_cf = Slider(start=1, end=3, value=1.55, step=0.05, title="Turnover target in %")    
    
# controls_to_cf = row(baseline_to_cf_select, slider_to_cf, country_to_cf_select)
# country_to_cf_select.on_change('value', update_country_to_cf)
# slider_to_cf.on_change('value', update_target_to_cf)
# baseline_to_cf_select.on_change('value', update_baseline_to_cf)

# counterfactuals_to_report = column(controls_to_cf,p_to_cf)

#%% dynamic counterfactuals

baseline_dyn_cf = '1200'
country_dyn_cf = 'USA'

# baseline_dyn_cf_select = Select(value=baseline_dyn_cf, title='Baseline', options=['1030',
#                                                                                   '1030_99.0',
#                                                                                   '1030_99.1',
#                                                                                   '1030_99.2',
#                                                                                   '1030_99.3',
#                                                                                   '1030_99.4',
#                                                                                   '1030_99.5',
#                                                                                   '1030_99.6',
#                                                                                   '1030_99.7',
#                                                                                   '1030_99.8',
#                                                                                   '1030_99.9',
#                                                                                   '1030_99.10',
#                                                                                   '1030_99.11',
#                                                                                   '1030_99.12',
#                                                                                   '1030_99.13'
#                                                                                   ])
baseline_dyn_cf_select = Select(value=baseline_dyn_cf, title='Baseline', options=['1200',
                                                                                  ])
country_dyn_cf_select = Select(value=country_dyn_cf, 
                            title='Country', 
                            options=countries+['World'])
                            # options=countries+['World','Harmonizing','Upper_harmonizing',
                            #                    'Uniform_delta','Upper_uniform_delta'])

def get_data_dyn_cf(baseline,country):
    df_dyn_cf = pd.read_csv(cf_path+'baseline_'+baseline+'/dyn_'+country+'.csv')
    df_dyn_cf.set_index('delt',inplace=True)
    return df_dyn_cf

def build_max(df_dyn_cf):
    df_max = pd.concat([df_dyn_cf.idxmax(),df_dyn_cf.max()],axis=1)
    df_max.index.name = 'label'
    df_max.columns = ['xmax','max'] 
    df_max = df_max.loc[countries]
    df_max['colors'] = Category18[:len(df_max)]
    return df_max

df_dyn_cf = get_data_dyn_cf(baseline_dyn_cf,country_dyn_cf)
ds_dyn_cf = ColumnDataSource(df_dyn_cf)
df_dyn_cf_max = build_max(df_dyn_cf)
ds_dyn_cf_max = ColumnDataSource(df_dyn_cf_max)

colors_dyn_cf = itertools.cycle(Category18)
colors_dyn_cf_max = itertools.cycle(Category18)

p_dyn_cf = figure(title="With transitional dynamics patent protection counterfactual", 
                width = 1200,
                height = 850,
                x_axis_label='Change in delta',
                y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
                x_axis_type="log",
                tools = TOOLS) 

for col in df_dyn_cf.columns:
    if col not in [0,'delt']:
        p_dyn_cf.line(x='delt', y=col, source = ds_dyn_cf, 
                      color=next(colors_dyn_cf),line_width = 2, legend_label=col)

p_dyn_cf.circle(x = 'xmax', y = 'max', source = ds_dyn_cf_max, size=4, color='colors')
     
p_dyn_cf.legend.click_policy="hide"
p_dyn_cf.legend.label_text_font_size = '8pt'
p_dyn_cf.add_layout(p_dyn_cf.legend[0], 'right')

def update_baseline_dyn_cf(attrname, old, new):
    country_dyn_cf = country_dyn_cf_select.value
    ds_dyn_cf.data = get_data_dyn_cf(new,country_dyn_cf)
    df_dyn_cf = get_data_dyn_cf(new,country_dyn_cf)
    ds_dyn_cf.data = df_dyn_cf
    ds_dyn_cf_max.data = build_max(df_dyn_cf)
    
def update_country_dyn_cf(attrname, old, new):
    baseline_dyn_cf = baseline_dyn_cf_select.value
    df_dyn_cf = get_data_dyn_cf(baseline_dyn_cf,new)
    ds_dyn_cf.data = df_dyn_cf
    ds_dyn_cf_max.data = build_max(df_dyn_cf)
    
controls_dyn_cf = row(baseline_dyn_cf_select, country_dyn_cf_select)

baseline_dyn_cf_select.on_change('value', update_baseline_dyn_cf)
country_dyn_cf_select.on_change('value', update_country_dyn_cf)

counterfactuals_dyn_report = column(controls_dyn_cf,p_dyn_cf)

#%% counterfactuals 405 TO target with dynamics

# country_to_cf_dyn = 'USA'
# to_target_dyn = 0.016

# list_of_to_targets_dyn = np.linspace(0.01,0.03,41)

# def section_end(s):
#       return [int(_) for _ in s.split("_")[-1].split(".")]
# cf_to_list = sorted([s for s in os.listdir(cf_path) 
#             if s[9:].startswith('405') and s.startswith('baseline')], key=section_end)

# def get_data_to_cf_dyn(to_target_dyn,country):
#     idx_to_cf_dyn = np.argmin(np.abs(list_of_to_targets_dyn-to_target_dyn))
#     df_to_cf_dyn = pd.read_csv(cf_path+cf_to_list[min(idx_to_cf_dyn,len(cf_to_list)-1)]+'/dyn_'+country+'.csv')
#     df_to_cf_dyn.set_index('delt',inplace=True)
#     if country not in ['World','Harmonizing']:
#         df_to_cf_dyn['static_for_main_country'] = pd.read_csv(
#             cf_path+cf_to_list[min(idx_to_cf_dyn,len(cf_to_list)-1)]+'/'+country+'.csv'
#             )[country].values
#     else:
#         df_to_cf_dyn['static_for_main_country'] = np.nan
#     return df_to_cf_dyn

# def build_max(df_to_cf):
#     df_max = pd.concat([df_to_cf.idxmax(),df_to_cf.max()],axis=1)
#     df_max.index.name = 'label'
#     df_max.columns = ['xmax','max'] 
#     df_max = df_max.loc[countries]
#     df_max['colors'] = Category18[:len(df_max)]
#     return df_max

# country_to_cf_dyn_select = Select(value=country_to_cf_dyn, 
#                             title='Country', 
#                             options=countries+['World','Harmonizing'])

# df_to_cf_dyn = get_data_to_cf_dyn(to_target_dyn,country_to_cf_dyn)
# ds_to_cf_dyn = ColumnDataSource(df_to_cf_dyn)
# df_to_cf_dyn_max = build_max(df_to_cf_dyn)
# ds_to_cf_dyn_max = ColumnDataSource(df_to_cf_dyn_max)

# colors_to_cf_dyn = itertools.cycle(Category18)
# colors_to_cf_dyn_max = itertools.cycle(Category18)

# p_to_cf_dyn = figure(title="With transitional dynamics patent protection counterfactual as function of TO target, baseline 405", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Change in delta',
#                 y_axis_label='Normalized Consumption equivalent welfare',
#                 x_axis_type="log",
#                 tools = TOOLS) 

# for col in df_to_cf_dyn.columns:
#     if col not in [0,'delt','static_for_main_country']:
#         p_to_cf_dyn.line(x='delt', y=col, source = ds_to_cf_dyn, 
#                          color=next(colors_to_cf_dyn),line_width = 2, legend_label=col)
#     if col == 'static_for_main_country':
#         p_to_cf_dyn.line(x='delt', y=col, source = ds_to_cf_dyn, 
#                          color='grey',line_width = 2, legend_label=col, 
#                          line_dash = 'dashed')

# p_to_cf_dyn.circle(x = 'xmax', y = 'max', source = ds_to_cf_dyn_max, size=4, color='colors')

# p_to_cf_dyn.legend.click_policy="hide"
# p_to_cf_dyn.legend.label_text_font_size = '8pt'
# p_to_cf_dyn.add_layout(p_to_cf_dyn.legend[0], 'right')

# def update_baseline_to_cf_dyn(attrname, old, new):
#     country_to_cf_dyn = country_to_cf_dyn_select.value
#     df_to_cf_dyn = get_data_to_cf_dyn(new/100,country_to_cf_dyn)
#     ds_to_cf_dyn.data = df_to_cf_dyn
#     ds_to_cf_dyn_max.data = build_max(df_to_cf_dyn)
    
# def update_country_to_cf_dyn(attrname, old, new):
#     to_target_dyn = slider_to_cf_dyn.value/100
#     df_to_cf_dyn = get_data_to_cf_dyn(to_target_dyn,new)
#     ds_to_cf_dyn.data = df_to_cf_dyn
#     ds_to_cf_dyn_max.data = build_max(df_to_cf_dyn)
    
# slider_to_cf_dyn = Slider(start=1, end=3, value=1.85, step=0.05, title="Turnover target in %")    
    
# controls_to_cf_dyn = row(slider_to_cf_dyn, country_to_cf_dyn_select)
# country_to_cf_dyn_select.on_change('value', update_country_to_cf_dyn)
# slider_to_cf_dyn.on_change('value', update_baseline_to_cf_dyn)

# counterfactuals_to_dyn_report = column(controls_to_cf_dyn,p_to_cf_dyn)

#!!! third panel
# third_panel = row(counterfactuals_dyn_report, counterfactuals_to_dyn_report,  dyn_report)
third_panel = row(counterfactuals_dyn_report,counterfactuals_report)

#%% Dynamic Nash / coop equilibrium and deviations from it

baseline_dyn_nash_coop = '1200'
variation_dyn_nash_coop = 'baseline'
equilibrium_type ='Nash'

baseline_dyn_nash_coop_select = Select(value=baseline_dyn_nash_coop, title='Baseline', options=[
    # '607','501'
    '1200'
    ])
dic_of_possible_variations_dyn_nash_coop = {
    '1200':['baseline'],
    # '1003':['baseline','0.4'],
    # '1030':['baseline','99.0','99.1','99.2','99.3','99.4','99.5','99.6','99.7',     
    #         '99.8',     '99.9',     '99.10',     '99.11',
    #         '99.12',     '99.13'],
    # '607':['baseline'],
    # '501':['1.0','2.0']
    }
variation_dyn_nash_coop_select = Select(value=variation_dyn_nash_coop, 
                            title='Variation', 
                            options=dic_of_possible_variations_dyn_nash_coop[baseline_dyn_nash_coop])
equilibrium_type_select = Select(value=equilibrium_type, title='Equilibrium', options=['Nash','Coop eq','Coop negishi'])

def get_dyn_eq_deltas_welfares(baseline_dyn_nash_coop,variation_dyn_nash_coop,equilibrium_type):
    if equilibrium_type == 'Nash':
        deltas = pd.read_csv(nash_eq_path+'dyn_deltas.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','method'],keep='last')
        eq_deltas = deltas.loc[
            (deltas.baseline == baseline_dyn_nash_coop)
            & (deltas.variation == variation_dyn_nash_coop)
            ][countries].values.squeeze()
        welfares = pd.read_csv(nash_eq_path+'dyn_cons_eq_welfares.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','method'],keep='last')
        eq_welfares = welfares.loc[
            (welfares.baseline == baseline_dyn_nash_coop)
            & (welfares.variation == variation_dyn_nash_coop)
            ][countries].values.squeeze()
    
    if equilibrium_type == 'Coop eq':
        deltas = pd.read_csv(coop_eq_path+'dyn_deltas.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','aggregation_method'],keep='last')
        eq_deltas = deltas.loc[
            (deltas.baseline == baseline_dyn_nash_coop)
            & (deltas.variation == variation_dyn_nash_coop)
            & (deltas.aggregation_method == 'pop_weighted')
            ][countries].values.squeeze()
        welfares = pd.read_csv(coop_eq_path+'dyn_cons_eq_welfares.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','aggregation_method'],keep='last')
        eq_welfares = welfares.loc[
            (welfares.baseline == baseline_dyn_nash_coop)
            & (welfares.variation == variation_dyn_nash_coop)
            & (welfares.aggregation_method == 'pop_weighted')
            ][countries].values.squeeze()
    
    if equilibrium_type == 'Coop negishi':
        deltas = pd.read_csv(coop_eq_path+'dyn_deltas.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','aggregation_method'],keep='last')
        eq_deltas = deltas.loc[
            (deltas.baseline == baseline_dyn_nash_coop)
            & (deltas.variation == variation_dyn_nash_coop)
            & (deltas.aggregation_method == 'negishi')
            ][countries].values.squeeze()
        welfares = pd.read_csv(coop_eq_path+'dyn_cons_eq_welfares.csv'
                              ,index_col=0
                              ,dtype={'baseline':str,'variation':str}).drop_duplicates(['baseline','variation','aggregation_method'],keep='last')
        eq_welfares = welfares.loc[
            (welfares.baseline == baseline_dyn_nash_coop)
            & (welfares.variation == variation_dyn_nash_coop)
            & (welfares.aggregation_method == 'negishi')
            ][countries].values.squeeze()
        
    df = pd.DataFrame(index = pd.Index(countries,name='country'))
    df['deltas'] = eq_deltas
    df['welfares'] = eq_welfares
    df['colors'] = Category18[:len(df)]
    
    return df

def get_dyn_deviation_recap(baseline_dyn_nash_coop,variation_dyn_nash_coop,equilibrium_type):
    if variation_dyn_nash_coop == 'baseline':
        temp_run = baseline_dyn_nash_coop
    else:
        temp_run = baseline_dyn_nash_coop+'_'+variation_dyn_nash_coop
    
    if equilibrium_type == 'Nash':
        dyn_deviation_recap = pd.read_csv(around_dyn_eq_path+f'around_dyn_nash_eq/baseline_{temp_run}/all_countries.csv')
    
    if equilibrium_type == 'Coop eq':
        dyn_deviation_recap = pd.read_csv(around_dyn_eq_path+f'around_dyn_coop_equal_eq/baseline_{temp_run}/all_countries.csv')
    
    if equilibrium_type == 'Coop negishi':
        dyn_deviation_recap = pd.read_csv(around_dyn_eq_path+f'around_dyn_coop_negishi_eq/baseline_{temp_run}/all_countries.csv')

    return dyn_deviation_recap
  
ds_dyn_eq = ColumnDataSource(get_dyn_eq_deltas_welfares(baseline_dyn_nash_coop,variation_dyn_nash_coop,equilibrium_type))
# ds_dyn_eq_dev = ColumnDataSource(get_dyn_deviation_recap(baseline_dyn_nash_coop,variation_dyn_nash_coop,equilibrium_type))

colors_dyn_eq_dev = itertools.cycle(Category18)

p_dyn_eq_dev = figure(title="With transitional dynamics Equilibria and unilateral deviations from it", 
                width = 1200,
                height = 850,
                x_axis_label='Delta',
                y_axis_label='Normalized Consumption equivalent welfare change',
                x_axis_type="log",
                tools = TOOLS) 

# for country_eq_dev in countries:
#     color = next(colors_dyn_eq_dev)
#     p_dyn_eq_dev.line(x=country_eq_dev+'_delta', 
#                       y=country_eq_dev+'_welfare', 
#                       source = ds_dyn_eq_dev, 
#                       color=color,
#                       line_width = 2, 
#                       legend_label=country_eq_dev+'_welfare')
#     p_dyn_eq_dev.line(x=country_eq_dev+'_delta', 
#                       y=country_eq_dev+'_world_negishi', 
#                       source = ds_dyn_eq_dev, 
#                       color=color,
#                       line_width = 2, 
#                       line_dash='dashed',
#                       legend_label=country_eq_dev+'_world_negishi')
#     p_dyn_eq_dev.line(x=country_eq_dev+'_delta', 
#                       y=country_eq_dev+'_world_equal', 
#                       source = ds_dyn_eq_dev, 
#                       color=color,
#                       line_width = 2, 
#                       line_dash='dotted',
#                       legend_label=country_eq_dev+'_world_equal')

p_dyn_eq_dev.circle(x = 'deltas', y = 'welfares', source = ds_dyn_eq, size=4,color = 'colors')

# p_dyn_eq_dev.legend.click_policy="hide"
# p_dyn_eq_dev.legend.label_text_font_size = '8pt'
# p_dyn_eq_dev.add_layout(p_dyn_eq_dev.legend[0], 'right')

hover_tool_eq = HoverTool()
hover_tool_eq.tooltips = [
    ("delta", "$x"),
    ("welfare", "$y")
    ] 
p_dyn_eq_dev.add_tools(hover_tool_eq)


labels_dyn_eq_dev = LabelSet(x='deltas', y='welfares', text='country',
              x_offset=2, y_offset=2, source=ds_dyn_eq, text_font_size="7pt")

p_dyn_eq_dev.add_layout(labels_dyn_eq_dev)


def update_baseline_dyn_nash(attrname, old, new):
    variation_dyn_nash_coop_select.value = dic_of_possible_variations_dyn_nash_coop[new][0]
    variation_dyn_nash_coop_select.options = dic_of_possible_variations_dyn_nash_coop[new]
    ds_dyn_eq.data = get_dyn_eq_deltas_welfares(new,
                                                variation_dyn_nash_coop_select.value,
                                                equilibrium_type_select.value)
    # ds_dyn_eq_dev.data = get_dyn_deviation_recap(new,
    #                                               variation_dyn_nash_coop_select.value,
    #                                               equilibrium_type_select.value)
    
def update_variation_dyn_nash_coop(attrname, old, new):
    ds_dyn_eq.data = get_dyn_eq_deltas_welfares(baseline_dyn_nash_coop_select.value,
                                                new,
                                                equilibrium_type_select.value)
    # ds_dyn_eq_dev.data = get_dyn_deviation_recap(baseline_dyn_nash_coop_select.value,
    #                                               new,
    #                                               equilibrium_type_select.value)
    
def update_equilibrium_type(attrname, old, new):
    ds_dyn_eq.data = get_dyn_eq_deltas_welfares(baseline_dyn_nash_coop_select.value,
                                                variation_dyn_nash_coop_select.value,
                                                new)
    # ds_dyn_eq_dev.data = get_dyn_deviation_recap(baseline_dyn_nash_coop_select.value,
    #                                               variation_dyn_nash_coop_select.value,
    #                                               new)

controls_dyn_eq_dev = row(baseline_dyn_nash_coop_select, variation_dyn_nash_coop_select, equilibrium_type_select)

baseline_dyn_nash_coop_select.on_change('value', update_baseline_dyn_nash)
variation_dyn_nash_coop_select.on_change('value', update_variation_dyn_nash_coop)
equilibrium_type_select.on_change('value', update_equilibrium_type)

dyn_eq_dev_report = column(controls_dyn_eq_dev,p_dyn_eq_dev)


#%% Nash / coop equilibrium
def section_ser(s):
      return pd.Series([[int(_) for _ in s_e.split(".")] for s_e in s])

baseline_nash_coop = '1200'

# dic_change_labels_for_405 = {'405, '+k:comments_dic['403'][k] for k in comments_dic['405']}

def get_data_nash_coop(baseline_nash_number):

    welf_coop = pd.read_csv(coop_eq_path+'dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation','aggregation_method'],keep='last').sort_values(['baseline','variation'])
    welf_nash = pd.read_csv(nash_eq_path+'dyn_cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation'],keep='last').sort_values(['baseline','variation'])
        
    welf_coop['run'] = welf_coop['baseline'].astype('str')+', '+welf_coop['variation']
    welf_nash['run'] = welf_nash['baseline'].astype('str')+', '+welf_nash['variation']

    # welf_coop['run'] = welf_coop['run'].replace(dic_change_labels_for_405)
    # welf_nash['run'] = welf_nash['run'].replace(dic_change_labels_for_405)
    
    welf_coop['sorting'] = welf_coop['variation'].str.replace('baseline','0')#.astype(float)
    welf_nash['sorting'] = welf_nash['variation'].str.replace('baseline','0')#.astype(float)
    
    welf_coop = welf_coop.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    welf_nash = welf_nash.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    
    welf_coop = welf_coop[welf_coop['baseline'].isin([int(baseline_nash_number)])]
    welf_nash = welf_nash[welf_nash['baseline'].isin([int(baseline_nash_number)])]
    
    welf_negishi = welf_coop[welf_coop['aggregation_method'] == 'negishi']
    welf_pop_weighted = welf_coop[welf_coop['aggregation_method'] == 'pop_weighted']
    
    return welf_pop_weighted, welf_negishi, welf_nash

baseline_nash_coop_select = Select(value=baseline_nash_coop, title='Baseline', 
                                    # options=['404','405','501','601'])
                                    # options=['501','607','618','619'])
                                    # options=['802','803','804','805','806'])
                                    options=['1200'])

welf_pop_weighted, welf_negishi, welf_nash = get_data_nash_coop(baseline_nash_coop)
    
ds_pop_weighted = ColumnDataSource(welf_pop_weighted)
ds_negishi = ColumnDataSource(welf_negishi)
ds_nash = ColumnDataSource(welf_nash)

colors_pop_weighted = itertools.cycle(Category18)
colors_negishi = itertools.cycle(Category18)
colors_nash = itertools.cycle(Category18)

x_range_nash = welf_nash['run'].to_list()

p_eq = figure(title="Cooperative and Nash equilibrium", 
                width = 1200,
                height = 900,
                x_range = x_range_nash,
                # x_axis_label='Run',
                y_axis_label='Consumption eqivalent welfare change',
                tools = TOOLS
                ) 
p_eq.xaxis.major_label_orientation = 3.14/3

lines_nash = {}
for col in p_baseline.countries+['Equal']+['Negishi']:
    lines_nash[col+' Nash'] = p_eq.line(x='run', y=col, source = ds_nash, color=next(colors_nash),line_width = 2, legend_label=col+' Nash')
    lines_nash[col+' coop equal'] = p_eq.line(x='run', y=col, source = ds_pop_weighted, color=next(colors_pop_weighted), line_dash='dashed', line_width = 2, legend_label=col+' coop equal')
    lines_nash[col+' coop negishi'] = p_eq.line(x='run', y=col, source = ds_negishi, color=next(colors_negishi), line_dash='dotted', line_width = 2, legend_label=col+' coop negishi')
    if col != 'Negishi' and col != 'Equal':
        lines_nash[col+' Nash'].visible = False
        lines_nash[col+' coop equal'].visible = False
        lines_nash[col+' coop negishi'].visible = False
        
        
p_eq.legend.click_policy="hide"
p_eq.legend.label_text_font_size = '8pt'
p_eq.legend.spacing = 0
p_eq.add_layout(p_eq.legend[0], 'right')    

hover_tool_eq = HoverTool()
hover_tool_eq.tooltips = [
    ("run", "@run"),
    ("value", "$y")
    ] 
p_eq.add_tools(hover_tool_eq)

columns = [
        TableColumn(field="runs", title="Runs"),
        TableColumn(field="comments", title="Description"),
    ]

# explication = Div(text="In the legend, first is the quantity displayed and last\
#                   is the quantity maximized <br> 'Negishi coop equal' means that: <br> \
#                       - we display the Change in cons equivalent of world welfare <br> according to Negishi weights aggregation<br>\
#                       - we maximize according to the Change in cons equivalent of world welfare <br> according to equal weights aggregation\
#                           ")

data_table_welfares = pd.concat([welf_nash.set_index('run'),
              welf_negishi.set_index('run'),
              welf_pop_weighted.set_index('run')],
            axis=0,
            keys=['Nash','Coop Negishi','Coop equal'],
            names=['type','run'],
            sort=False
            ).reset_index().sort_values('sorting',key=section_ser)[['run','type']+p_baseline.countries+['Equal']+['Negishi']]

source_table_welfares = ColumnDataSource(data_table_welfares)
columns_welf = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries+['Equal']+['Negishi']]

table_widget_welfares = DataTable(source=source_table_welfares, columns=columns_welf, width=1100, height=400,
                          )

def get_delta_nash_coop(baseline_number):
    deltas_coop = pd.read_csv(coop_eq_path+'dyn_deltas.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation','aggregation_method'],keep='last').sort_values(['baseline','variation'])
    deltas_nash = pd.read_csv(nash_eq_path+'dyn_deltas.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation'],keep='last').sort_values(['baseline','variation'])
    
    deltas_coop['run'] = deltas_coop['baseline'].astype('str')+', '+deltas_coop['variation']
    deltas_nash['run'] = deltas_nash['baseline'].astype('str')+', '+deltas_nash['variation']
    
    # deltas_coop['run'] = deltas_coop['run'].replace(dic_change_labels_for_405)
    # deltas_nash['run'] = deltas_nash['run'].replace(dic_change_labels_for_405)
    
    deltas_coop['sorting'] = deltas_coop['variation'].str.replace('baseline','0')#.astype(float)
    deltas_nash['sorting'] = deltas_nash['variation'].str.replace('baseline','0')#.astype(float)
    
    deltas_coop = deltas_coop.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    deltas_nash = deltas_nash.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    
    deltas_coop = deltas_coop[deltas_coop['baseline'].isin([int(baseline_number)])]
    deltas_nash = deltas_nash[deltas_nash['baseline'].isin([int(baseline_number)])]
    
    deltas_negishi = deltas_coop[deltas_coop['aggregation_method'] == 'negishi']
    deltas_pop_weighted = deltas_coop[deltas_coop['aggregation_method'] == 'pop_weighted']
    
    return deltas_pop_weighted, deltas_negishi, deltas_nash

deltas_pop_weighted, deltas_negishi, deltas_nash = get_delta_nash_coop(baseline_nash_coop)

ds_deltas_negishi = ColumnDataSource(deltas_negishi)
ds_deltas_pop_weighted = ColumnDataSource(deltas_pop_weighted)
ds_deltas_nash = ColumnDataSource(deltas_nash)

colors_deltas_negishi = itertools.cycle(Category18)
colors_deltas_pop_weighted = itertools.cycle(Category18)
colors_deltas_nash = itertools.cycle(Category18)

p_deltas_eq = figure(title="Cooperative and Nash equilibrium", 
                width = 1200,
                height = 900,
                x_range = x_range_nash,
                y_axis_type="log",
                y_axis_label='Delta',
                tools = TOOLS
                ) 
p_deltas_eq.xaxis.major_label_orientation = 3.14/3

lines_delta={}
for col in p_baseline.countries:
    lines_delta[col+' Nash'] = p_deltas_eq.line(x='run', y=col, 
                                            source = ds_deltas_nash, color=next(colors_deltas_nash),
                                            line_width = 2, legend_label=col+' Nash')
    lines_delta[col+' coop equal'] = p_deltas_eq.line(x='run', y=col, 
                                                source = ds_deltas_pop_weighted, color=next(colors_deltas_pop_weighted), line_dash='dashed', 
                                                line_width = 2, legend_label=col+' coop equal')
    lines_delta[col+' coop negishi'] = p_deltas_eq.line(x='run', y=col, 
                                                source = ds_deltas_negishi, color=next(colors_deltas_negishi), line_dash='dotted', 
                                                line_width = 2, legend_label=col+' coop negishi')
    lines_delta[col+' coop equal'].visible = False
    lines_delta[col+' coop negishi'].visible = False
    
p_deltas_eq.legend.click_policy="hide"
p_deltas_eq.legend.label_text_font_size = '8pt'
p_deltas_eq.legend.spacing = 0
p_deltas_eq.add_layout(p_deltas_eq.legend[0], 'right')   
hover_tool_deltas_eq = HoverTool()
hover_tool_deltas_eq.tooltips = [
    ("run", "@run"),
    ("value", "$y")
    ] 
p_deltas_eq.add_tools(hover_tool_deltas_eq)

data_table_deltas = pd.concat([deltas_nash.set_index('run'),
              deltas_negishi.set_index('run'),
              deltas_pop_weighted.set_index('run')],
            axis=0,
            keys=['Nash','Coop Negishi','Coop equal'],
            names=['type','run'],
            sort=False
            ).reset_index().sort_values('sorting',key=section_ser)[['run','type']+p_baseline.countries]

source_table_deltas = ColumnDataSource(data_table_deltas)
columns_deltas = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries+['Equal']+['Negishi']]

table_widget_deltas = DataTable(source=source_table_deltas, columns=columns_deltas, width=1100, height=400,
                          )

def update_baseline_nash(attrname, old, new):
    baseline_nash_number = new
    welf_pop_weighted, welf_negishi, welf_nash = get_data_nash_coop(baseline_nash_number)
        
    ds_pop_weighted.data = welf_pop_weighted
    ds_negishi.data = welf_negishi
    ds_nash.data = welf_nash
    
    deltas_pop_weighted, deltas_negishi, deltas_nash = get_delta_nash_coop(baseline_nash_number)

    ds_deltas_negishi.data = deltas_negishi
    ds_deltas_pop_weighted.data = deltas_pop_weighted
    ds_deltas_nash.data = deltas_nash
    
    p_eq.x_range.factors = welf_nash['run'].to_list()
    p_deltas_eq.x_range.factors = welf_nash['run'].to_list()

baseline_nash_coop_select.on_change('value', update_baseline_nash)

nash_coop_welfare_report = column(baseline_nash_coop_select,p_eq,table_widget_welfares)
nash_coop_deltas_report = column(p_deltas_eq,table_widget_deltas)

#!!! fourth panel
# fourth_panel = row(nash_coop_welfare_report, nash_coop_deltas_report)
fourth_panel = row(dyn_eq_dev_report, nash_coop_welfare_report, nash_coop_deltas_report)
# fourth_panel = row(nash_coop_welfare_report, nash_coop_deltas_report)

#%% dynamic solver

# baseline_dyn = '1020'
# country_dyn = 'USA'
# sector_dyn = 'Patent'

# baseline_dyn_select = Select(value=baseline_dyn, title='Baseline', 
#                               # options=['501','604','607','608','609','610']
#                               options=['1020']
#                               )

# baseline_dyn_path = results_path+'baseline_'+baseline_dyn+'_variations/'
# files_in_dir = next(os.walk(baseline_dyn_path))[1]
# run_list = [f for f in files_in_dir if f[0].isnumeric()]
# run_list = sorted(run_list, key=section)
# variation_dyn_select = Select(value='baseline', title='Variation', 
#                               options=['baseline']+run_list)

# def update_list_of_runs_dyn(attr, old, new):
#     baseline_dyn_path = results_path+'baseline_'+new+'_variations/'
#     files_in_dir = next(os.walk(baseline_dyn_path))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list = sorted(run_list, key=section)
#     variation_dyn_select.options = ['baseline']+run_list

# country_dyn_select = Select(value='USA', title='Country delta to change', options=['USA', 'EUR', 'JAP', 'CHN', 'BRA', 
#                                                                                     'IND','CAN','KOR','RUS','AUS',
#                                                                                     'MEX', 'ROW','World'])
# slider_dyn = Slider(start=-1, end=0.5, value=0, step=0.01, title="Log change of delta")    

# state_computation = Div(text="Done")

# def make_time_evolution_df(dyn_sol):
#     qties = ['w','l_R','l_Ae','l_Ao','price_indices','Z','g','r','profit']
#     df = pd.DataFrame(index = pd.Index(qties,name='Quantity'), 
#                       columns = ['Initial jump mean','Initial jump median',
#                                   'Typical time of evolution\nmean','Typical time of evolution\nmedian'])
#     for qty in qties:
#         a =  dyn_sol.get_jump(qty)
#         df.loc[qty,'Initial jump mean'] = a[0].round(2)
#         df.loc[qty,'Initial jump median'] = a[1].round(2)
#         b = dyn_sol.get_typical_time_evolution(qty)
#         df.loc[qty,'Typical time of evolution\nmean'] = b[0].round(2)
#         df.loc[qty,'Typical time of evolution\nmedian'] = b[1].round(2)
#     return df

# def fit_and_eval(vec,dyn_sol):
#     fit = np.polyval(np.polyfit(dyn_sol.t_real,
#                 vec,
#                 dyn_sol.Nt),np.linspace(0,dyn_sol.t_inf,2001))
#     return fit

# def create_column_data_source_from_dyn_sol(dyn_sol):
#     data_dyn = {}
#     data_dyn['time'] = np.linspace(0,dyn_sol.t_inf,2001)
#     for agg_qty in ['g']:
#         data_dyn[agg_qty] = fit_and_eval(getattr(dyn_sol,agg_qty),dyn_sol)
#     for c_qty in ['Z','r','price_indices','w','nominal_final_consumption','ratios_of_consumption_levels_change_not_normalized',
#                   'integrand_welfare','second_term_sum_welfare','integral_welfare']:
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn[c_qty+c] = fit_and_eval(getattr(dyn_sol,c_qty)[i,:].ravel(),dyn_sol)
#     for c_s_qty in ['l_R','psi_o_star','PSI_CD','l_Ao']:
#         for i,c in enumerate(dyn_sol.countries):
#             if c_s_qty in ['PSI_CD']:
#                 data_dyn[c_s_qty+c] = fit_and_eval(
#                     (getattr(dyn_sol,c_s_qty)+getattr(dyn_sol,c_s_qty+'_0')[...,None])[i,1,:].ravel(),dyn_sol)
#             else:
#                 data_dyn[c_s_qty+c] = fit_and_eval(
#                     getattr(dyn_sol,c_s_qty)[i,1,:].ravel(),dyn_sol)
#     for c_c_s_qty in ['l_Ae','PSI_MPD','PSI_MPND','PSI_MNP','profit']:
#         if c_c_s_qty in ['PSI_MPD','PSI_MPND','PSI_MNP']:
#             temp_sum_n = (getattr(dyn_sol,c_c_s_qty)+getattr(dyn_sol,c_c_s_qty+'_0')[...,None]).sum(axis=0)
#             temp_sum_i = (getattr(dyn_sol,c_c_s_qty)+getattr(dyn_sol,c_c_s_qty+'_0')[...,None]).sum(axis=1)
#         else:
#             temp_sum_n = getattr(dyn_sol,c_c_s_qty).sum(axis=0)
#             temp_sum_i = getattr(dyn_sol,c_c_s_qty).sum(axis=1)
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn['sum_n_'+c_c_s_qty+c] = fit_and_eval(temp_sum_n[i,1,:].ravel(),dyn_sol)
#             data_dyn['sum_i_'+c_c_s_qty+c] = fit_and_eval(temp_sum_i[i,1,:].ravel(),dyn_sol)
#     for i,c in enumerate(dyn_sol.countries):
#         data_dyn['real_final_consumption'+c] = fit_and_eval((getattr(dyn_sol,'nominal_final_consumption')[i,:]
#                                                 /getattr(dyn_sol,'price_indices')[i,:]).ravel(),dyn_sol)
    
#     data_dyn_init = {}
#     data_dyn_init['time'] = [0]
#     for agg_qty in ['g']:
#         data_dyn_init[agg_qty] = [getattr(dyn_sol.sol_init,agg_qty)]
#     for c_qty in ['Z','price_indices','w','nominal_final_consumption']:
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_init[c_qty+c] = [getattr(dyn_sol.sol_init,c_qty)[i]]
#     for c_s_qty in ['l_R','psi_o_star','PSI_CD','l_Ao']:
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_init[c_s_qty+c] = [getattr(dyn_sol.sol_init,c_s_qty)[i,1]]
#     for c_c_s_qty in ['l_Ae','PSI_MPD','PSI_MPND','PSI_MNP','profit']:
#         if c_c_s_qty == 'profit':
#             temp_sum_n = (getattr(dyn_sol.sol_init,c_c_s_qty)*getattr(dyn_sol.sol_init,'w')[None,:,None]).sum(axis=0)[:,1]
#             temp_sum_i = (getattr(dyn_sol.sol_init,c_c_s_qty)*getattr(dyn_sol.sol_init,'w')[None,:,None]).sum(axis=1)[:,1]
#         else:
#             temp_sum_n = getattr(dyn_sol.sol_init,c_c_s_qty).sum(axis=0)[:,1]
#             temp_sum_i = getattr(dyn_sol.sol_init,c_c_s_qty).sum(axis=1)[:,1]
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_init['sum_n_'+c_c_s_qty+c] = [temp_sum_n[i]]
#             data_dyn_init['sum_i_'+c_c_s_qty+c] = [temp_sum_i[i]]
#     for i,c in enumerate(dyn_sol.countries):
#         data_dyn_init['real_final_consumption'+c] = [getattr(dyn_sol.sol_init,'nominal_final_consumption')[i]/getattr(dyn_sol.sol_init,'price_indices')[i]]
#         data_dyn_init['r'+c] = [getattr(dyn_sol.sol_init,'r')]
#         data_dyn_init['integrand_welfare'+c] = [None]
#         data_dyn_init['integral_welfare'+c] = [None]
#         data_dyn_init['second_term_sum_welfare'+c] = [None]
#         data_dyn_init['ratios_of_consumption_levels_change_not_normalized'+c] = [None]
        
#     data_dyn_fin = {}
#     data_dyn_fin['time'] = [dyn_sol.t_inf]
#     for agg_qty in ['g']:
#         data_dyn_fin[agg_qty] = [getattr(dyn_sol.sol_fin,agg_qty)]
#     for c_qty in ['Z','price_indices','w','nominal_final_consumption']:
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_fin[c_qty+c] = [getattr(dyn_sol.sol_fin,c_qty)[i]]
#     for c_s_qty in ['l_R','psi_o_star','PSI_CD','l_Ao']:
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_fin[c_s_qty+c] = [getattr(dyn_sol.sol_fin,c_s_qty)[i,1]]
#     for c_c_s_qty in ['l_Ae','PSI_MPD','PSI_MPND','PSI_MNP','profit']:
#         if c_c_s_qty == 'profit':
#             temp_sum_n = (getattr(dyn_sol.sol_fin,c_c_s_qty)*getattr(dyn_sol.sol_fin,'w')[None,:,None]).sum(axis=0)[:,1]
#             temp_sum_i = (getattr(dyn_sol.sol_fin,c_c_s_qty)*getattr(dyn_sol.sol_fin,'w')[None,:,None]).sum(axis=1)[:,1]
#         else:
#             temp_sum_n = getattr(dyn_sol.sol_fin,c_c_s_qty).sum(axis=0)[:,1]
#             temp_sum_i = getattr(dyn_sol.sol_fin,c_c_s_qty).sum(axis=1)[:,1]
#         for i,c in enumerate(dyn_sol.countries):
#             data_dyn_fin['sum_n_'+c_c_s_qty+c] = [temp_sum_n[i]]
#             data_dyn_fin['sum_i_'+c_c_s_qty+c] = [temp_sum_i[i]]
#     for i,c in enumerate(dyn_sol.countries):
#         data_dyn_fin['real_final_consumption'+c] = [getattr(dyn_sol.sol_fin,'nominal_final_consumption')[i]/getattr(dyn_sol.sol_fin,'price_indices')[i]]
#         data_dyn_fin['r'+c] = [getattr(dyn_sol.sol_fin,'r')]
#         data_dyn_fin['integrand_welfare'+c] = [None]
#         data_dyn_fin['integral_welfare'+c] = [None]
#         data_dyn_fin['second_term_sum_welfare'+c] = [None]
#         data_dyn_fin['ratios_of_consumption_levels_change_not_normalized'+c] = [None]
        
#     return data_dyn, data_dyn_init, data_dyn_fin

# def compute_dyn(event):
#     if variation_dyn_select.value == 'baseline':
#         path = results_path+baseline_dyn_select.value+'/'
#     else:
#         path = results_path+'baseline_'+baseline_dyn_select.value+'_variations/'+variation_dyn_select.value+'/'
#     p_dyn, m_dyn, sol_dyn = load(path, data_path=data_path,
#                                   dir_path=dir_path)
#     p_dyn_cf = p_dyn.copy()
#     if country_dyn_select.value != 'World':
#         p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1] = p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1]*(10**slider_dyn.value)
#     else:
#         p_dyn_cf.delta[:,1] = p_dyn_cf.delta[:,1]*(10**slider_dyn.value)
#     start = time.perf_counter()
#     dyn_sol, sol_c, convergence = rough_dyn_fixed_point_solver(p_dyn_cf, sol_dyn, sol_fin = None,Nt=25,
#                                           t_inf=500, x0=None, tol = 1e-14, max_count=1e6, safe_convergence=0.1,damping=50, damping_post_acceleration=10)
#     end = time.perf_counter()
#     if country_dyn_select.value == 'World':
#         message = 'Done, computation for all deltas multiplied by a factor '+str(10**slider_dyn.value)+'<br>Convergence : '+str(convergence)+'<br>Computation time : '+str(end-start)
#     else:
#         message = 'Done, computation for delta '+country_dyn_select.value+' = '+str(p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1])+'<br>Convergence : '+str(convergence)+'<br>Computation time : '+str(end-start)
#     state_computation.text = message
#     temp = create_column_data_source_from_dyn_sol(dyn_sol)
#     ds_dyn.data = temp[0]
#     ds_dyn_init.data = temp[1]
#     ds_dyn_fin.data = temp[2]
#     source_table_time_evol.data = make_time_evolution_df(dyn_sol)
    
# if variation_dyn_select.value == 'baseline':
#     path = results_path+baseline_dyn_select.value+'/'
# else:
#     path = results_path+'baseline_'+baseline_dyn_select.value+'_variations/'+variation_dyn_select.value+'/'
# p_dyn, m_dyn, sol_dyn = load(path, data_path=data_path,
#                               dir_path=dir_path)
# p_dyn_cf = p_dyn.copy()
# if country_dyn_select.value != 'World':
#     p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1] = p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1]*10**slider_dyn.value
# else:
#     p_dyn_cf.delta[:,1] = p_dyn_cf.delta[:,1]*slider_dyn.value
# dyn_sol, sol_c, convergence = rough_dyn_fixed_point_solver(p_dyn_cf, sol_dyn, sol_fin = None,Nt=25,
#                                       t_inf=500, x0=None, tol = 1e-14, max_count=1e6, safe_convergence=0.1,damping=50, damping_post_acceleration=10)

# source_table_time_evol = ColumnDataSource(make_time_evolution_df(dyn_sol))
# columns_time_evol = [TableColumn(field=col) for col in 
#                       ['Quantity','Initial jump mean','Initial jump median',
#                                 'Typical time of evolution\nmean','Typical time of evolution\nmedian']]
# table_widget_time_evol = DataTable(source=source_table_time_evol, columns=columns_time_evol, width=600, height=750)

# button_compute_dyn = Button(label="Compute",align='end')
# button_compute_dyn.on_event(ButtonClick, compute_dyn)

# qty_dyn_display_select = Select(value='g', title='Quantity', options=['g','Z','r','price_indices','w','nominal_final_consumption',
#                                                     'real_final_consumption','ratios_of_consumption_levels_change_not_normalized',
#                                                     'l_R','l_Ao','psi_o_star',
#                                                     'PSI_CD','integrand_welfare','integral_welfare','second_term_sum_welfare',
#                                                     'sum_n_l_Ae','sum_n_PSI_MPD','sum_n_PSI_MPND','sum_n_PSI_MNP','sum_n_profit',
#                                                     'sum_i_l_Ae','sum_i_PSI_MPD','sum_i_PSI_MPND','sum_i_PSI_MNP','sum_i_profit'])
# country_dyn_display_select = Select(value='USA', title='Country', options=['USA', 'EUR', 'JAP', 'CHN', 'BRA', 
#                                                                                     'IND','CAN','KOR','RUS','AUS',
#                                                                                     'MEX', 'ROW'])

# temp = create_column_data_source_from_dyn_sol(dyn_sol)
# data_dyn_default = temp[0]
# data_dyn_init_default = temp[1]
# data_dyn_fin_default = temp[2]
# ds_dyn = ColumnDataSource(data_dyn_default)
# ds_dyn_init = ColumnDataSource(data_dyn_init_default)
# ds_dyn_fin = ColumnDataSource(data_dyn_fin_default)
# up_max = max([max(ds_dyn.data['g']), max(ds_dyn_fin.data['g']), max(ds_dyn_init.data['g'])])
# down_min = min([min(ds_dyn.data['g']), min(ds_dyn_fin.data['g']), min(ds_dyn_init.data['g'])])
# delta = up_max-down_min
# if delta == 0:
#     delta = 1
# p_dyn_figure = figure(title="Dynamic solver",
#                 width = 1200,
#                 height = 750,
#                 x_axis_label='Time',
#                 y_axis_label='Value',
#                 tools = TOOLS,
#                 x_range = (-20,dyn_sol.t_inf+20),
#                 y_range=(down_min-delta*0.1,up_max+delta*0.1)
#                 )

# hover_tool_eq = HoverTool()
# hover_tool_eq.tooltips = [
#     ("Time", "$x"),
#     ("value", "$y")
#     ] 
# p_dyn_figure.add_tools(hover_tool_eq)

# lines_dyn = {}
# for col in data_dyn_default.keys():
#     if col != time:
#         lines_dyn[col] = p_dyn_figure.line(x='time', y=col, source = ds_dyn)
#         if col != 'g':
#             lines_dyn[col].visible = False

# init_dyn = {}
# for col in data_dyn_init_default.keys():
#     if col != time:
#         init_dyn[col] = p_dyn_figure.circle(x='time', y=col, source = ds_dyn_init, color='red',size=8)
#         if col != 'g':
#             init_dyn[col].visible = False
            
# fin_dyn = {}
# for col in data_dyn_fin_default.keys():
#     if col != time:
#         fin_dyn[col] = p_dyn_figure.circle(x='time', y=col, source = ds_dyn_fin, color='red',size=8)
#         if col != 'g':
#             fin_dyn[col].visible = False

# def update_graph_dyn(event):
#     if qty_dyn_display_select.value in ['g']:
#         col = qty_dyn_display_select.value
#     elif qty_dyn_display_select.value in ['Z','r','price_indices','w','nominal_final_consumption','real_final_consumption',
#                                           'ratios_of_consumption_levels_change_not_normalized',
#                                 'l_R','l_Ao','psi_o_star','PSI_CD','integrand_welfare','integral_welfare','second_term_sum_welfare',
#                                 'sum_n_l_Ae','sum_n_PSI_MPD','sum_n_PSI_MPND','sum_n_PSI_MNP','sum_n_profit',
#                                 'sum_i_l_Ae','sum_i_PSI_MPD','sum_i_PSI_MPND','sum_i_PSI_MNP','sum_i_profit']:
#         col = qty_dyn_display_select.value+country_dyn_display_select.value
#     lines_dyn[col].visible = True
#     if qty_dyn_display_select.value not in ['integrand_welfare','integral_welfare',
#                                             'second_term_sum_welfare','ratios_of_consumption_levels_change_not_normalized']:
#         init_dyn[col].visible = True
#         fin_dyn[col].visible = True
#     else:
#         init_dyn[col].visible = False
#         fin_dyn[col].visible = False

#     for other_column in lines_dyn:
#         if other_column != col:
#             lines_dyn[other_column].visible = False
#             init_dyn[other_column].visible = False
#             fin_dyn[other_column].visible = False
#     try:
#         up_max = max([max(ds_dyn.data[col]), max(ds_dyn_fin.data[col]), max(ds_dyn_init.data[col])])
#         down_min = min([min(ds_dyn.data[col]), min(ds_dyn_fin.data[col]), min(ds_dyn_init.data[col])])
#     except:
#         up_max = max(ds_dyn.data[col])
#         down_min = min(ds_dyn.data[col])
#     delta = up_max-down_min
#     if delta == 0:
#         delta = 1
#     p_dyn_figure.y_range.start=down_min-delta*0.1
#     p_dyn_figure.y_range.end=up_max+delta*0.1
#     p_dyn_figure.x_range.start=-20
#     p_dyn_figure.x_range.end=dyn_sol.t_inf+20
        
# button_display_dyn = Button(label="Display",align='end')
# button_display_dyn.on_event(ButtonClick, update_graph_dyn)

# controls_dyn = row(baseline_dyn_select, variation_dyn_select, country_dyn_select, slider_dyn, button_compute_dyn, state_computation)
# controls_display_dyn = row(qty_dyn_display_select, 
#                             country_dyn_display_select,
#                             button_display_dyn)

# baseline_dyn_select.on_change('value', update_list_of_runs_dyn)

# dyn_report = column(controls_dyn,controls_display_dyn,p_dyn_figure)


#!!! fifth_panel
# fifth_panel = row(counterfactuals_report, counterfactuals_to_report)
# fifth_panel = row(dyn_report,table_widget_time_evol)

#%% sensitivities

# baselines_dic_sensi = {}

# for baseline_nbr in ['1004']:
#     baselines_dic_sensi[baseline_nbr] = {} 
#     baseline_sensi_path = results_path+'baseline_'+baseline_nbr+'_sensitivity_tables/'
#     files_in_dir = os.listdir(baseline_sensi_path)
#     files_in_dir = [ filename for filename in files_in_dir if filename.endswith('.csv') ]
#     for f in files_in_dir:
#         baselines_dic_sensi[baseline_nbr][f[:-4]] = pd.read_csv(baseline_sensi_path+f,index_col = 0)
    
# baseline_sensi = '1004'
# qty_sensi = 'objective'

# baseline_sensi_select = Select(value=baseline_sensi, title='Baseline', options=sorted(baselines_dic_sensi.keys()))
# qty_sensi_select = Select(value=qty_sensi, title='Quantity', options=sorted(baselines_dic_sensi[baseline_sensi].keys()))

# ds_sensi = ColumnDataSource(baselines_dic_sensi[baseline_sensi][qty_sensi])
# p_sensi = figure(title="Sensitivity", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Change in moment or parameter',
#                 y_axis_label='Value',
#                 tools = TOOLS)

# colors_sensi = itertools.cycle(Category18)

# for col in baselines_dic_sensi[baseline_sensi][qty_sensi].columns[1:]:
#     if col!='zeta':
#         p_sensi.line(x='Change', y=col, source = ds_sensi, color=next(colors_sensi),line_width = 2, legend_label=col)

# p_sensi.legend.click_policy="hide"
# p_sensi.legend.label_text_font_size = '8pt'
# p_sensi.add_layout(p_sensi.legend[0], 'right')

# def update_baseline_sensi(attrname, old, new):
#     qty_sensi = qty_sensi_select.value
#     ds_sensi.data = baselines_dic_sensi[new][qty_sensi]
    
# def update_qty_sensi(attrname, old, new):
#     baseline_sensi = baseline_sensi_select.value
#     ds_sensi.data = baselines_dic_sensi[baseline_sensi][new]

# controls_sensi = row(baseline_sensi_select, qty_sensi_select)

# baseline_sensi_select.on_change('value', update_baseline_sensi)
# qty_sensi_select.on_change('value', update_qty_sensi)

# sensitivity_report = column(controls_sensi,p_sensi)

# %% weights sensitivities

# baselines_dic_sensi_weights = {}

# # for baseline_nbr in ['101','102','104']:
# for baseline_nbr in ['802']:
#     baselines_dic_sensi_weights[baseline_nbr] = {}
#     baseline_sensi_weights_path = results_path+'baseline_'+baseline_nbr+'_sensitivity_weights_tables/'
#     files_in_dir = os.listdir(baseline_sensi_weights_path)
#     files_in_dir = [ filename for filename in files_in_dir if filename.endswith('.csv') ]
#     for f in files_in_dir:
#         # if f not in ['GPDIFF.csv','GROWTH.csv']:
#             baselines_dic_sensi_weights[baseline_nbr][f[:-4]] = pd.read_csv(baseline_sensi_weights_path+f,index_col = 0)
    
# baseline_sensi_weights = '802'
# qty_sensi_weights = 'objective'

# baseline_sensi_weights_select = Select(value=baseline_sensi_weights, title='Baseline', options=sorted(baselines_dic_sensi_weights.keys()))
# qty_sensi_weights_select = Select(value=qty_sensi_weights, title='Quantity', options=sorted(baselines_dic_sensi_weights[baseline_sensi_weights].keys()))

# ds_sensi_weights = ColumnDataSource(baselines_dic_sensi_weights[baseline_sensi_weights][qty_sensi_weights])
# p_sensi_weights = figure(title="Sensitivity to the weights", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Change in weight',
#                 y_axis_label='Objective function or contribution to objective function: loss(moment,target)',
#                 y_axis_type="log",
#                 tools = TOOLS)

# colors_sensi_weights = itertools.cycle(Category18)

# for col in baselines_dic_sensi_weights[baseline_sensi_weights][qty_sensi_weights].columns[1:]:
#     # if col not in ['zeta','GPDIFF_weight','GROWTH_weight']:
#         p_sensi_weights.line(x='Change', y=col, source = ds_sensi_weights, color=next(colors_sensi_weights),line_width = 2, 
#                               legend_label=col)

# p_sensi_weights.legend.click_policy="hide"
# p_sensi_weights.legend.label_text_font_size = '8pt'
# p_sensi_weights.add_layout(p_sensi_weights.legend[0], 'right')

# def update_baseline_sensi_weights(attrname, old, new):
#     qty_sensi_weights = qty_sensi_weights_select.value
#     ds_sensi_weights.data = baselines_dic_sensi_weights[new][qty_sensi_weights]
    
# def update_qty_sensi_weights(attrname, old, new):
#     baseline_sensi_weights = baseline_sensi_weights_select.value
#     ds_sensi_weights.data = baselines_dic_sensi_weights[baseline_sensi_weights][new]

# controls_sensi_weights = row(baseline_sensi_weights_select, qty_sensi_weights_select)

# baseline_sensi_weights_select.on_change('value', update_baseline_sensi_weights)
# qty_sensi_weights_select.on_change('value', update_qty_sensi_weights)

# sensitivity_weights_report = column(controls_sensi_weights,p_sensi_weights)

#%% Jacobian panel

# baseline_jac = '1010'
# country_jac = 'USA'
# sector_jac = 'Patent'

# # baseline_jac_select = Select(value=baseline_jac, title='Baseline', options=['501','604','607','608','609','610'])
# baseline_jac_select = Select(value=baseline_jac, title='Baseline', options=['1010'])

# baseline_jac_path = results_path+'baseline_'+baseline_jac+'_variations/'
# files_in_dir = next(os.walk(baseline_jac_path))[1]
# run_list = [f for f in files_in_dir if f[0].isnumeric()]
# run_list = sorted(run_list, key=section)
# variation_jac_select = Select(value='baseline', title='Variation', 
#                               options=['baseline']+run_list)

# def update_list_of_runs_jac(attr, old, new):
#     baseline_jac_path = results_path+'baseline_'+new+'_variations/'
#     files_in_dir = next(os.walk(baseline_jac_path))[1]
#     run_list = [f for f in files_in_dir if f[0].isnumeric()]
#     run_list = sorted(run_list, key=section)
#     variation_jac_select.options = ['baseline']+run_list

# if variation_jac_select.value == 'baseline':
#     path = results_path+baseline_jac_select.value+'/'
# else:
#     path = results_path+'baseline_'+baseline_jac_select.value+'_variations/'+variation_jac_select.value+'/'
    
# p_jac, m_jac, sol_jac = load(path, data_path=data_path,
#                               dir_path=dir_path)

# qty_jac_select = Select(value='delta', title='Parameter', options=p_jac.calib_parameters)
# country_jac_select = Select(value='USA', title='Country', options=p_jac.countries)
# sector_jac_select = Select(value='Patent', title='Sector', options=p_jac.sectors)

# if qty_jac_select.value in ['eta','T','delta','nu']:
#     idx_to_change_jac = p_jac.countries.index(country_jac_select.value),p_jac.sectors.index(sector_jac_select.value)
# if qty_jac_select.value in ['fe','zeta','nu', 'fo']:
#     idx_to_change_jac = 0,p_jac.sectors.index(sector_jac_select.value)
# if qty_jac_select.value in ['k','g_0']:
#     idx_to_change_jac = 0

# qty_to_change_jac = qty_jac_select.value

# x_jac = compute_rough_jacobian(p_jac, m_jac, qty_to_change_jac, idx_to_change_jac, 
#                             change_by = 0.25, tol = 1e-14, damping = 5,
#                             max_count = 5e3)

# p_jac_fig = figure(title="Rough jacobian computation", 
#                 y_range=FactorRange(factors=m_jac.get_signature_list()),
#                 width = 1500,
#                 height = 1200,
#                 x_axis_label='Change in contribution to objective function',
#                 y_axis_label='Moment',
#                 tools = TOOLS) 

# data_jac = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([np.array(m_jac.get_signature_list()),x_jac]).T)
# src_jac = ColumnDataSource(data_jac)

# p_jac_fig.hbar(y = 'Moment',right = 'Contribution', source = src_jac)

# hover_tool_jac = HoverTool()
# hover_tool_jac.tooltips = [
#     ("(Moment)", "(@Moment)"),
#     ]
# p_jac_fig.add_tools(hover_tool_jac)


# def update_jac(event):
#     if variation_jac_select.value == 'baseline':
#         path = results_path+baseline_jac_select.value+'/'
#     else:
#         path = results_path+'baseline_'+baseline_jac_select.value+'_variations/'+variation_jac_select.value+'/'
#     par_jac, m_jac, sol_jac = load(path, data_path=data_path,
#                                     dir_path=dir_path)
#     if qty_jac_select.value in ['eta','T','delta','nu']:
#         idx_to_change_jac = par_jac.countries.index(country_jac_select.value),par_jac.sectors.index(sector_jac_select.value)
#     if qty_jac_select.value in ['fe','zeta','nu', 'fo']:
#         idx_to_change_jac = par_jac.sectors.index(sector_jac_select.value)
#     if qty_jac_select.value in ['k','g_0']:
#         idx_to_change_jac = None
#     x_jac = compute_rough_jacobian(par_jac, m_jac, qty_jac_select.value, idx_to_change_jac, 
#                                 change_by = 0.1, tol = 1e-14, damping = 5,
#                                 max_count = 5e3)
#     data_jac = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([np.array(m_jac.get_signature_list()),x_jac]).T)
#     src_jac.data = data_jac
#     p_jac_fig.y_range.factors = m_jac.get_signature_list()

# button_jac = Button(label="Compute")
# button_jac.on_event(ButtonClick, update_jac)

# controls_jac = row(baseline_jac_select, variation_jac_select, qty_jac_select, 
#                     country_jac_select, sector_jac_select, button_jac)

# baseline_jac_select.on_change('value', update_list_of_runs_jac)

# jac_report = column(controls_jac,p_jac_fig)

#!!! sixth panel
# sixth_panel = row(sensitivity_report,sensitivity_weights_report,jac_report)
# sixth_panel = row(sensitivity_report,sensitivity_weights_report)

#%% Kogan paper

# colors_kog = itertools.cycle(Category18)

# df_kog = pd.read_csv(data_path+'koga_updated.csv')
# ds_kog = ColumnDataSource(df_kog)

# p_kog = figure(title="Kogan moment updated / extrapolated", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Issue Date',
#                 y_axis_type="log",
#                 tools = TOOLS) 

# l_kog = {}

# for i,col in enumerate(df_kog.columns):
#     if col not in ['issue_date']:
#         l_kog[i] = p_kog.line(x='issue_date', y=col, 
#                   source = ds_kog, 
#                   line_width = 2, legend_label=col, color=next(colors_kog),
#                   name = col)

# hover_tool_kog = HoverTool(
#     tooltips = [
#         ("Issue date", "$x"),
#         ('ValuePerPatent', '@ValuePerPatent'),
#         ('CostPerPatent', '@CostPerPatent'),
#         ('KM_article', '@KM_article'),
#         ('ValuePerPatentUpdated', '@ValuePerPatentUpdated'),
#         ('CostPerPatentExtrapolated', '@CostPerPatentExtrapolated'),
#         ('KM_extrapolatedCost', '@KM_extrapolatedCost')
#         ],
#     mode='vline',
#     renderers = [l_kog[4]]
# )
# p_kog.add_tools(hover_tool_kog)

# p_kog.legend.click_policy="hide"
# p_kog.legend.label_text_font_size = '8pt'
# p_kog.add_layout(p_kog.legend[0], 'right')


# # colors_kog2 = itertools.cycle(Category18)

# # df_kog2 = pd.read_csv(data_path+'KM_prior.csv')
# # ds_kog2 = ColumnDataSource(df_kog2)

# # p_kog2 = figure(title="Kogan moment", 
# #                 width = 1200,
# #                 height = 850,
# #                 x_axis_label='Market Prior',
# #                 tools = TOOLS) 

# # l_kog2 = {}

# # for i,col in enumerate(df_kog2.columns):
# #     if col not in ['market prior']:
# #         l_kog2[i] = p_kog2.line(x='market prior', y=col, 
# #                   source = ds_kog2, 
# #                   line_width = 2, legend_label=col, color=next(colors_kog2))

# # hover_tool_kog2 = HoverTool(
# #     tooltips = [
# #         ("market prior", "$x"),
# #         ('1950 to 2007', '@from1950to2007'),
# #         ('1980 to 2007', '@from1980to2007'),
# #         ('1995 to 2007', '@from1995to2007'),
# #         ('2002 to 2007', '@from2002to2007'),
# #         ('1950 to 2020', '@from1950to2020'),
# #         ('1980 to 2020', '@from1980to2020'),
# #         ('1995 to 2020', '@from1995to2020'),
# #         ('2002 to 2020', '@from2002to2020'),
# #         ],
# #     mode='vline',
# #     renderers = [l_kog2[4]]
# # )

# # p_kog2.legend.click_policy="hide"
# # p_kog2.legend.label_text_font_size = '8pt'
# # p_kog2.add_layout(p_kog2.legend[0], 'right')
# # p_kog2.add_tools(hover_tool_kog2)


# colors_to_data = itertools.cycle(Category18)

# df_to_data = pd.read_csv(data_path+'turnover_imports_weighted_11_countries.csv'
#                          )[['year','HS_digits','A3']].pivot(
#                                 columns= 'HS_digits',
#                                 index = 'year',
#                                 values = 'A3'
#                             )[[6,8,10]]
# df_to_data = df_to_data.rename(columns={6:'6',
#                                         8:'8',
#                                         10:'10'})
# ds_to_data = ColumnDataSource(df_to_data)

# p_to_data = figure(title="Turnover moment for rule A3, time window (y,y+5)", 
#                 width = 1200,
#                 height = 850,
#                 x_axis_label='Year',
#                 # y_axis_type="log",
#                 tools = TOOLS) 

# l_to_data = {}

# for i,col in enumerate(['6','8','10']):
#     if col not in ['year']:
#         l_to_data[i] = p_to_data.line(x='year', y=col, 
#                   source = ds_to_data, 
#                   line_width = 2, legend_label=col, color=next(colors_to_data),
#                   name = col)

# hover_tool_to_data = HoverTool(
#     tooltips = [
#         ("Year", "$x"),
#         ('HS6', '@6'),
#         ('HS8', '@8'),
#         ('HS10', '@10'),
#         ],
#     mode='vline',
#     renderers = [l_to_data[1]]
# )
# p_to_data.add_tools(hover_tool_to_data)

# p_to_data.legend.click_policy="hide"
# p_to_data.legend.label_text_font_size = '8pt'
# p_to_data.add_layout(p_to_data.legend[0], 'right')

# #!!! seventh_panel
# # seventh_panel = row(p_kog,p_kog2)
# seventh_panel = row(p_kog,p_to_data)

#%% 7 countries comparison of patent flows data

# # labels_leg_patstat = {
# #     'baseline':'pre IN treatment',
# #     'calibration data':'calibration data',
# #     'WIPO data':'WIPO data',
# #     'alternative 1':'alt 1 : no sector filtering',
# #     'alternative 2':'alt 2 : first applicant only',
# #     'alternative 3':'alt 3 : diff origin weight',
# #     'alternative 4':'alt 4 : no domestic allocation',
# #     'alternative 5':'alt 5 : only granted patents',
# #     'alternative 6':'alt 6 : no ML predi for EPO',
# #     'alternative 7':'alt 7 : with ML predi for WIPO',
# #     'after IN treatment':'baseline',
# #     'julian latest code':'julian latest code',
# #     }
# # tot = pd.read_csv(join(dirname(__file__),'patstat_compar.csv')).set_index(
# #     ['destination_code','origin_code']
# #     ).sort_index(
# #     ).round()

# # ds_patstat = ColumnDataSource(tot)
# # # TOOLS="pan,wheel_zoom,box_zoom,reset,save"
# # p_patstat = figure(title="Patent flows", 
# #                 width = 1200,
# #                 height = 850,
# #                 x_axis_type="log",
# #                 y_axis_type="log",
# #                 x_axis_label='Baseline', 
# #                 # y_axis_label='Model implied',
# #                 tools = TOOLS)
# # hover_tool = HoverTool()
# # hover_tool.tooltips = [
# #     ("index", "@x"),
# #     ("(baseline,alternative)", "($x,$y)"),
# #     ]
# # # labels_patstat = LabelSet(x='calibration data', y='baseline', text='x',
# # labels_patstat = LabelSet(y='WIPO data', x='after IN treatment', text='x',
# #               x_offset=2, y_offset=2, source=ds_patstat, text_font_size="7pt")
# # p_patstat.add_layout(labels_patstat)
# # p_patstat.add_tools(hover_tool)

# # slope_patstat = Slope(gradient=1, y_intercept=0,
# #               line_color='black', line_dash='dashed', line_width=1)
# # p_patstat.add_layout(slope_patstat)
# # lines_patstat = {}
# # colors_patstat = itertools.cycle(Category18)
# # for i,col in enumerate(tot.columns):
# #     if col not in ['x','after IN treatment']:
# #         # lines_patstat[col] = p_patstat.circle('calibration data', col, 
# #         lines_patstat[col] = p_patstat.circle('after IN treatment', col, 
# #                 source = ds_patstat, 
# #                 size=5, color=next(colors_patstat))
# #         if col != 'WIPO data':
# #             lines_patstat[col].visible = False
            
# # legend_items = [LegendItem(label=labels_leg_patstat[col], renderers=[lin_par])
# #                     for col, lin_par in lines_patstat.items() if col not in 
# #                     # ['x','calibration data']]
# #                     ['x','after IN treatment']]

# # legend = Legend(items=legend_items, click_policy="hide", 
# #                     label_text_font_size="8pt",
# #                     spacing = 0, 
# #                     )
# # p_patstat.add_layout(legend, 'right')

# # columns_patstat = [
# #         TableColumn(field="x"),
# #     ]+[TableColumn(field=col) for col in tot.columns]
# # data_table_patstat = DataTable(source=ds_patstat, columns = columns_patstat, width=1200, height=400)

# # #%% 13 countries comparison of patent flows data

# # tot_13 = pd.read_csv(join(dirname(__file__),'patstat_compar_13.csv')).set_index(
# #     ['destination_code','origin_code']
# #     ).sort_index(
# #     ).round()

# # ds_patstat_13 = ColumnDataSource(tot_13)
# # # TOOLS="pan,wheel_zoom,box_zoom,reset,save"
# # p_patstat_13 = figure(title="Patent flows", 
# #                 width = 1200,
# #                 height = 850,
# #                 x_axis_type="log",
# #                 y_axis_type="log",
# #                 x_axis_label='Baseline', 
# #                 # y_axis_label='Model implied',
# #                 tools = TOOLS)
# # hover_tool = HoverTool()
# # hover_tool.tooltips = [
# #     ("index", "@x"),
# #     ("(baseline,alternative)", "($x,$y)"),
# #     ]
# # # labels_patstat = LabelSet(x='calibration data', y='baseline', text='x',
# # labels_patstat_13 = LabelSet(y='WIPO data', x='baseline', text='x',
# #               x_offset=2, y_offset=2, source=ds_patstat_13, text_font_size="7pt")
# # p_patstat_13.add_layout(labels_patstat_13)
# # p_patstat_13.add_tools(hover_tool)

# # slope_patstat_13 = Slope(gradient=1, y_intercept=0,
# #               line_color='black', line_dash='dashed', line_width=1)
# # p_patstat_13.add_layout(slope_patstat_13)
# # lines_patstat_13 = {}
# # colors_patstat_13 = itertools.cycle(Category18)
# # for i,col in enumerate(tot_13.columns):
# #     if col not in ['x','baseline']:
# #         # lines_patstat[col] = p_patstat.circle('calibration data', col, 
# #         lines_patstat_13[col] = p_patstat_13.circle('baseline', col, 
# #                 source = ds_patstat_13, 
# #                 size=5, color=next(colors_patstat_13))
# # legend_items_13 = [LegendItem(label=labels_leg_patstat[col], renderers=[lin_par])
# #                     for col, lin_par in lines_patstat_13.items() if col not in 
# #                     # ['x','calibration data']]
# #                     ['x','baseline']]

# # legend_13 = Legend(items=legend_items_13, click_policy="hide", 
# #                     label_text_font_size="8pt",
# #                     spacing = 0, 
# #                     )
# # p_patstat_13.add_layout(legend_13, 'right')

# # columns_patstat_13 = [
# #         TableColumn(field="x"),
# #     ]+[TableColumn(field=col) for col in tot_13.columns]
# # data_table_patstat_13 = DataTable(source=ds_patstat_13, columns = columns_patstat_13, width=1200, height=400)


#!!! eigth_panel
# # eigth_panel = row(column(p_patstat,data_table_patstat),
# #                     column(p_patstat_13,data_table_patstat_13))

#%% build curdoc
print(time.perf_counter() - start)
curdoc().add_root(column(first_panel, 
                            # second_panel, 
                           third_panel, 
                            fourth_panel, 
                           #  fifth_panel, 
                           #  sixth_panel,
                           # seventh_panel,
                          # eigth_panel
                         )
                  )
