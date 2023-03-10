from os.path import dirname, join
import os
# import sys
# import __main__
# import datetime
# os.chdir(dirname(__file__))
# os.path.realpath("__file__")
# if __name__ == "__main__":
#     print("__file__")

import pandas as pd
# from scipy.signal import savgol_filter

from bokeh.io import curdoc
from bokeh.layouts import row, column
# from bokeh.models import DataRange1d,Button, Slider,CheckboxButtonGroup, LinearAxis, FactorRange, Text, Div,Toggle, ColumnDataSource, LabelSet, Select,Legend, LegendItem, DataTable, TableColumn, HoverTool, Slope
from bokeh.models import Button,Range1d, Slider, FactorRange, Div, ColumnDataSource, LabelSet, Select,Legend, LegendItem, DataTable, TableColumn, HoverTool, Slope
# from bokeh.models.formatters import NumeralTickFormatter
# from bokeh.models.widgets.tables import NumberFormatter
# from bokeh.palettes import Blues4
from bokeh.plotting import figure
from datetime import datetime
import random
from bokeh.events import ButtonClick
from classes import parameters, moments, var
from data_funcs import compute_rough_jacobian,rough_dyn_fixed_point_solver
import numpy as np
# from bokeh.models import LogScale, LinearScale
import itertools
from bokeh.palettes import Category10
# import numpy as np
import time


# print(1)

def load(path, data_path=None, context = 'calibration'):
    p = parameters(n=7,s=2,data_path=data_path)
    p.load_data(path)
    # if path.endswith('20.1/') or path.endswith('20.2/'):
    #     p.r_hjort[3] = 17.33029162
    # if path.endswith('19.1/') or path.endswith('19.2/'):
    #     p.r_hjort[4] = 17.33029162
    # print(path)
    sol = var.var_from_vector(p.guess, p, compute=True, context = context)
    # sol.compute_non_solver_aggregate_qualities(p)
    # sol.compute_non_solver_quantities(p)
    sol.scale_P(p)
    sol.compute_price_indices(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    m.load_data(data_path)
    m.load_run(path)
    m.compute_moments(sol, p)
    m.compute_moments_deviations()
    # print(m.STFLOWSDOM)
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
        # print(param)
        # print(getattr(p_baseline,param)[p_baseline.mask[param]].squeeze().shape == (14,))
        if hasattr(p_baseline,param):
            # print(param,getattr(p_baseline,param))
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
    # df_scalar_moments_deviation = pd.DataFrame(columns = ['baseline'])
    # df_scalar_moments_deviation.index.name='x'
    for mom in list_of_moments:
        if mom != 'objective':
            if len(m_baseline.idx[mom]) == 1:
                if mom != 'OUT':
                    try:
                        df_scalar_moments.loc[mom,'target'] = float(getattr(m_baseline,mom+'_target'))
                        df_scalar_moments.loc[mom,'baseline'] = float(getattr(m_baseline,mom))
                        # df_scalar_moments_deviation.loc[mom,'baseline'] = float(getattr(m_baseline,mom+'_deviation'))
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
            
    for sol_qty in ['semi_elast_RD_delta','DT','psi_o_star']:
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
    # df_scalar_moments.loc['objective','baseline'] = (m_baseline.deviation_vector()**2).sum()
    df_scalar_moments.loc['objective','baseline'] = m_baseline.objective_function()*28
    dic_df_mom['scalars'] = df_scalar_moments
    # dic_df_mom['scalar deviations'] = df_scalar_moments_deviation
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
        if k in ['semi_elast_RD_delta','DT','psi_o_star']:
            dic_df_sol[k][run_name] = getattr(sol,k)[...,1]
        if k in ['l_R']:
            dic_df_sol[k][run_name] = getattr(sol,k)[...,1]/p.labor
        if k in ['min_psi_m_star_outward']:
            dic_df_sol[k][run_name] = getattr(sol,'psi_m_star')[:,:,1].min(axis=0)
        if k in ['min_psi_m_star_inward']:
            dic_df_sol[k][run_name] = getattr(sol,'psi_m_star')[:,:,1].min(axis=1)
            
    return dic_df_param, dic_df_mom, dic_df_sol

#%% path

data_path = join(dirname(__file__), 'data/')
# data_path = 'data/'
# results_path = 'calibration_results_matched_economy/'
results_path = join(dirname(__file__), 'calibration_results_matched_economy/')
cf_path = join(dirname(__file__), 'counterfactual_recaps/unilateral_patent_protection/')
nash_eq_path = join(dirname(__file__), 'nash_eq_recaps/')
coop_eq_path = join(dirname(__file__), 'coop_eq_recaps/')


#%% moments / parameters for variations

list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
 'RD', 'RP', 'SPFLOWDOM', 'SPFLOW','STFLOW','STFLOWSDOM',
 'SRDUS', 'SRGDP','UUPCOST','SINNOVPATUS',
 'SINNOVPATEU', 'TO','TE','DOMPATINUS','DOMPATINEU',
 'TWSPFLOW','TWSPFLOWDOM','SDOMTFLOW','objective']
# list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
#  'RD', 'RP', 'SPFLOWDOM', 'SPFLOW','STFLOW','STFLOWSDOM',
#  'SRDUS', 'SRGDP','UUPCOST', 'PCOST','PCOSTINTER','PCOSTNOAGG','PCOSTINTERNOAGG','SINNOVPATUS',
#  'SINNOVPATEU', 'TO','TP',
#  'DOMPATUS','DOMPATEU','DOMPATINUS','DOMPATINEU','TWSPFLOW','TWSPFLOWDOM','SDOMTFLOW','objective']

comments_dic = {}

comments_dic['311'] = {"baseline":"baseline",
                "1.0":"1.0: kappa:0.5,TO:0.05,KM:0.06",
                "1.1":"1.1: kappa:0.5,TO:0.05,KM:0.09277",
                "1.2":"1.2: kappa:0.5,TO:0.05,KM:0.1322",
                "1.3":"1.3: kappa:0.5,TO:0.036,KM:0.06",
                "1.4":"1.4: kappa:0.5,TO:0.036,KM:0.09277",
                "1.5":"1.5: kappa:0.5,TO:0.036,KM:0.1322",
                "1.6":"1.6: kappa:0.5,TO:0.0124,KM:0.06",
                "1.7":"1.7: kappa:0.5,TO:0.0124,KM:0.09277",
                "1.8":"1.8: kappa:0.5,TO:0.0124,KM:0.1322",
                "1.9":"1.9: kappa:0.5,TO:0.0242,KM:0.06",
                "1.10":"1.10: kappa:0.5,TO:0.0242,KM:0.09277",
                "1.11":"1.11: kappa:0.5,TO:0.0242,KM:0.1322",
                "2.0":"2.0: Added SINNOVPATEU moment",
                "2.1":"2.1: 2.0 stronger weights DOMPATEU/US SINNOVPATEU/US",
                "2.1.0":"2.1.0: kappa:0.5,TO:0.05,KM:0.06",
                "2.1.1":"2.1.1: kappa:0.5,TO:0.05,KM:0.09277",
                "2.1.2":"2.1.2: kappa:0.5,TO:0.05,KM:0.1322",
                "2.1.3":"2.1.3: kappa:0.5,TO:0.036,KM:0.06",
                "2.1.4":"2.1.4: kappa:0.5,TO:0.036,KM:0.09277",
                "2.1.5":"2.1.5: kappa:0.5,TO:0.036,KM:0.1322",
                "2.1.6":"2.1.6: kappa:0.5,TO:0.0242,KM:0.06",
                "2.1.7":"2.1.7: kappa:0.5,TO:0.0242,KM:0.09277",
                "2.1.8":"2.1.8: kappa:0.5,TO:0.0242,KM:0.1322",
                "2.1.9":"2.1.9: kappa:0.5,TO:0.0124,KM:0.06",
                "2.1.9.2":"2.1.9.2: re-converging 2.1.9",
                "2.1.10":"2.1.10: kappa:0.5,TO:0.0124,KM:0.09277",
                "2.1.11":"2.1.11: kappa:0.5,TO:0.0124,KM:0.1322",
                "2.2":"2.2: 2.0 with ratio loss",
                "2.3":"2.3: 2.2 higher SPFLOW weight",
                "3.0":"3.0: Not calibrated ! only eta_US/2",
                "4.0":"4.0: drop SRDUS moment",
                "5.0":"5.0: no Hjort factors",
                "6.1":"6.1: JUPCOST instead of UUPCOST",
                "6.2":"6.2: both JUPCOST and UUPCOST",
                "6.3":"6.3: none of JUPCOST or UUPCOST",
                "7.0":"7.0: calibrated elasticities",
                "8.0":"8.0: ratio loss function for SPFLOW",
                # "8.1":"8.1: squared diff loss function for SPFLOW",            
                # "1.9":"1.9: kappa:0.7474,TO:0.05,KM:0.06",
                # "1.10":"1.10: kappa:0.7474,TO:0.05,KM:0.09277",
                # "1.11":"1.11: kappa:0.7474,TO:0.05,KM:0.1322",
                # "1.12":"1.12: kappa:0.7474,TO:0.036,KM:0.06",
                # "1.13":"1.13: kappa:0.7474,TO:0.036,KM:0.09277",
                # "1.14":"1.14: kappa:0.7474,TO:0.036,KM:0.1322",
                # "1.15":"1.15: kappa:0.7474,TO:0.0124,KM:0.06",
                # "1.16":"1.16: kappa:0.7474,TO:0.0124,KM:0.09277",
                # "1.17":"1.17: kappa:0.7474,TO:0.0124,KM:0.1322",
                }

comments_dic['312'] = {"baseline":"baseline",
                "1.0":"1.0: identical baseline, TO: 0.0242, KM:0.09277",
                "1.1":"1.1: TO: 0.036",
                "1.2":"1.2: TO: 0.0124",
                "1.3":"1.3: TO: 0.00972",
                "2.0":"2.0: with SINNOVPATEU",
                "2.1":"2.1: TO: 0.036",
                "2.2":"2.2: TO: 0.0124",
                "2.3":"2.3: TO: 0.00972",
                "3.0":"3.0: with DOMPATINUS/EU",
                "3.1":"3.1: TO: 0.036",
                "3.2":"3.2: TO: 0.0124",
                "3.3":"3.3: TO: 0.00972",
                "4.0":"4.0: with SINNOVPATEU and DOMPATINUS/EU",
                "4.1":"4.1: TO: 0.036",
                "4.2":"4.2: TO: 0.0124",
                "4.3":"4.3: TO: 0.00972",
                "4.4":"4.4: 4.0 new solver version",
                # "4.5":"4.5: 4.4 targeting SDOMTFLOW",
                "5.0":"5.0: drop SRDUS",
                "5.1":"5.1: TO: 0.036",
                "5.2":"5.2: TO: 0.0124",
                "5.3":"5.3: TO: 0.00972",
                "6.0":"6.0: drop SRDUS with SINNOVPATEU",
                "6.1":"6.1: TO: 0.036",
                "6.2":"6.2: TO: 0.0124",
                "6.3":"6.3: TO: 0.00972",
                "7.0":"7.0: drop SRDUS with DOMPATINUS/EU",
                "7.1":"7.1: TO: 0.036",
                "7.2":"7.2: TO: 0.0124",
                "7.3":"7.3: TO: 0.00972",
                "8.0":"8.0: drop SRDUS with SINNOVPATEU and DOMPATINUS/EU",
                "8.1":"8.1: TO: 0.036",
                "8.2":"8.2: TO: 0.0124",
                "8.3":"8.3: TO: 0.00972",
                "9.0":"9.0: drop UUPCOST",
                "9.1":"9.1: TO: 0.036",
                "9.2":"9.2: TO: 0.0124",
                "9.3":"9.3: TO: 0.00972",
                "10.0":"10.0: drop UUPCOST with SINNOVPATEU and DOMPATINUS/EU",
                "10.1":"10.1: TO: 0.036",
                "10.2":"10.2: TO: 0.0124",
                "10.3":"10.3: TO: 0.00972",
                "11.0":"11.0:drop SRDUS & UUPCOST with SINNOVPATEU & DOMPATIN",  
                "11.1":"11.1: TO: 0.036",
                "11.2":"11.2: TO: 0.0124",
                "11.3":"11.3: TO: 0.00972",
                }

comments_dic['401'] = {"baseline":"baseline",
                "1.0":"1.0: identical baseline, TO: 0.0242, KM:0.09277",
                "1.1":"1.1: TO: 0.036",
                "1.2":"1.2: TO: 0.0183",
                "1.3":"1.3: TO: 0.0124",
                "2.0":"2.0: with SINNOVPATEU",
                "2.1":"2.1: TO: 0.036",
                "2.2":"2.2: TO: 0.0183",
                "2.3":"2.3: TO: 0.0124",
                "3.0":"3.0: with DOMPATINUS/EU",
                "3.1":"3.1: TO: 0.036",
                "3.2":"3.2: TO: 0.0183",
                "3.3":"3.3: TO: 0.0124",
                "4.0":"4.0: with SINNOVPATEU and DOMPATINUS/EU",
                "4.1":"4.1: TO: 0.036",
                "4.2":"4.2: TO: 0.0183",
                "4.3":"4.3: TO: 0.0124",
                "5.0":"5.0: drop SRDUS",
                "5.1":"5.1: TO: 0.036",
                "5.2":"5.2: TO: 0.0183",
                "5.3":"5.3: TO: 0.0124",
                "6.0":"6.0: drop SRDUS with SINNOVPATEU",
                "6.1":"6.1: TO: 0.036",
                "6.2":"6.2: TO: 0.0183",
                "6.3":"6.3: TO: 0.0124",
                "7.0":"7.0: drop SRDUS with DOMPATINUS/EU",
                "7.1":"7.1: TO: 0.036",
                "7.2":"7.2: TO: 0.0183",
                "7.3":"7.3: TO: 0.0124",
                "8.0":"8.0: drop SRDUS with SINNOVPATEU and DOMPATINUS/EU",
                "8.1":"8.1: TO: 0.036",
                "8.2":"8.2: TO: 0.0183",
                "8.3":"8.3: TO: 0.0124",
                "9.0":"9.0: drop UUPCOST",
                "9.1":"9.1: TO: 0.036",
                "9.2":"9.2: TO: 0.0183",
                "9.3":"9.3: TO: 0.0124",
                "10.0":"10.0: drop UUPCOST with SINNOVPATEU and DOMPATINUS/EU",
                "10.1":"10.1: TO: 0.036",
                "10.2":"10.2: TO: 0.0183",
                "10.3":"10.3: TO: 0.0124",
                "11.0":"11.0:drop SRDUS & UUPCOST with SINNOVPATEU & DOMPATIN",  
                "11.1":"11.1: TO: 0.036",
                "11.2":"11.2: TO: 0.0183",
                "11.3":"11.3: TO: 0.0124",
                }

comments_dic['402'] = {"baseline":"baseline, SRDUS, UUPCOST, DOMPATOUT",
                "1.0":"1.0: identical baseline, TO: 0.0242, KM:0.09277",
                "1.1":"1.1: TO: 0.0183",
                "1.1.1":"1.1.1: calibrated elasticities",
                "1.2":"1.2: TO: 0.0124",
                "2.0":"2.0: SRDUS, UUPCOST, DOMPATIN",
                "2.1":"2.1: TO: 0.0183",
                "2.1.1":"2.1.1: calibrated elasticities",
                "2.1.2":"2.1.2: calibrated sigma",
                "2.1.3":"2.1.3: calibrated theta",
                "2.2":"2.2: TO: 0.0124",
                "3.0":"3.0: SRDUS, UUPCOST, no DOMPAT",
                "3.1":"3.1: TO: 0.0183",
                "3.1.1":"3.1.1: calibrated elasticities",
                "3.2":"3.2: TO: 0.0124",
                "4.0":"4.0: SRDUS, PCOST, DOMPATOUT",
                "4.1":"4.1: TO: 0.0183",
                "4.1.1":"4.1.1: calibrated elasticities",
                "4.2":"4.2: TO: 0.0124",
                "5.0":"5.0: SRDUS, PCOST, DOMPATIN",
                "5.1":"5.1: TO: 0.0183",
                "5.1.1":"5.1.1: calibrated elasticities",
                "5.2":"5.2: TO: 0.0124",
                "6.0":"6.0: SRDUS, PCOSTINTER, no DOMPAT",
                "6.1":"6.1: TO: 0.0183",
                "6.1.1":"6.1.1: calibrated elasticities",
                "6.2":"6.2: TO: 0.0124",
                "7.0":"7.0: SRDUS, PCOSTNOAGG, DOMPATOUT",
                "7.1":"7.1: TO: 0.0183",
                "7.1.1":"7.1.1: calibrated elasticities",
                "7.2":"7.2: TO: 0.0124",
                "8.0":"8.0: SRDUS, PCOSTNOAGG, DOMPATIN",
                "8.1":"8.1: TO: 0.0183",
                "8.1.1":"8.1.1: calibrated elasticities",
                "8.2":"8.2: TO: 0.0124",
                "9.0":"9.0: SRDUS, PCOSTINTERNOAGG, no DOMPAT",
                "9.1":"9.1: TO: 0.0183",
                "9.1.1":"9.1.1: calibrated elasticities",
                "9.2":"9.2: TO: 0.0124",
                "10.0":"10.0: no SRDUS, UUPCOST, DOMPATOUT",
                "10.1":"10.1: TO: 0.0183",
                "10.1.1":"10.1.1: calibrated elasticities",
                "10.2":"10.2: TO: 0.0124",
                "11.0":"11.0: no SRDUS, UUPCOST, DOMPATIN",
                "11.1":"11.1: TO: 0.0183",
                "11.1.1":"11.1.1: calibrated elasticities",
                "11.2":"11.2: TO: 0.0124",
                "12.0":"12.0: no SRDUS, UUPCOST, no DOMPAT",
                "12.1":"12.1: TO: 0.0183",
                "12.1.1":"12.1.1: calibrated elasticities",
                "12.2":"12.2: TO: 0.0124",
                "13.0":"13.0: no SRDUS, PCOST, DOMPATOUT",
                "13.1":"13.1: TO: 0.0183",
                "13.1.1":"13.1.1: calibrated elasticities",
                "13.2":"13.2: TO: 0.0124",
                "14.0":"14.0: no SRDUS, PCOST, DOMPATIN",
                "14.1":"14.1: TO: 0.0183",
                "14.1.1":"14.1.1: calibrated elasticities",
                "14.2":"14.2: TO: 0.0124",
                "15.0":"15.0: no SRDUS, PCOSTINTER, no DOMPAT",
                "15.1":"15.1: TO: 0.0183",
                "15.1.1":"15.1.1: calibrated elasticities",
                "15.2":"15.2: TO: 0.0124",
                "16.0":"16.0: no SRDUS, PCOSTNOAGG, DOMPATOUT",
                "16.1":"16.1: TO: 0.0183",
                "16.1.1":"16.1.1: calibrated elasticities",
                "16.2":"16.2: TO: 0.0124",
                "17.0":"17.0: no SRDUS, PCOSTNOAGG, DOMPATIN",
                "17.1":"17.1: TO: 0.0183",
                "17.1.1":"17.1.1: calibrated elasticities",
                "17.1.2":"17.1.2: calibrated sigma",
                "17.1.3":"17.1.3: calibrated theta",
                "17.2":"17.2: TO: 0.0124",
                "18.0":"18.0: no SRDUS, PCOSTINTERNOAGG, no DOMPAT",
                "18.1":"18.1: TO: 0.0183",
                "18.1.1":"18.1.1: calibrated elasticities",
                "18.2":"18.2: TO: 0.0124",
                }

comments_dic['403'] = {'baseline':'bsln:TO:0.0183',
    '1.0':'1.0: TO: 0.01',
'1.1':'1.1: TO: 0.0105',
'1.2':'1.2: TO: 0.011',
'1.3':'1.3: TO: 0.0115',
'1.4':'1.4: TO: 0.012',
'1.5':'1.5: TO: 0.0125',
'1.6':'1.6: TO: 0.013',
'1.7':'1.7: TO: 0.0135',
'1.8':'1.8: TO: 0.014',
'1.9':'1.9: TO: 0.0145',
'1.10':'1.10: TO: 0.015',
'1.11':'1.11: TO: 0.0155',
'1.12':'1.12: TO: 0.016',
'1.13':'1.13: TO: 0.0165',
'1.14':'1.14: TO: 0.017',
'1.15':'1.15: TO: 0.0175',
'1.16':'1.16: TO: 0.018',
'1.17':'1.17: TO: 0.0185',
'1.18':'1.18: TO: 0.019',
'1.19':'1.19: TO: 0.0195',
'1.20':'1.20: TO: 0.02',
'1.21':'1.21: TO: 0.0205',
'1.22':'1.22: TO: 0.021',
'1.23':'1.23: TO: 0.0215',
'1.24':'1.24: TO: 0.022',
'1.25':'1.25: TO: 0.0225',
'1.26':'1.26: TO: 0.023',
'1.27':'1.27: TO: 0.0235',
'1.28':'1.28: TO: 0.024',
'1.29':'1.29: TO: 0.0245',
'1.30':'1.30: TO: 0.025',
'1.31':'1.31: TO: 0.0255',
'1.32':'1.32: TO: 0.026',
'1.33':'1.33: TO: 0.0265',
'1.34':'1.34: TO: 0.027',
'1.35':'1.35: TO: 0.0275',
'1.36':'1.36: TO: 0.028',
'1.37':'1.37: TO: 0.0285',
'1.38':'1.38: TO: 0.029',
'1.39':'1.39: TO: 0.0295',
'1.40':'1.40: TO: 0.03'
    }
comments_dic['405'] = {'baseline':'bsln:TO:0.0183',
    '1.0':'1.0: TO: 0.01',
'1.1':'1.1: TO: 0.0105',
'1.2':'1.2: TO: 0.011',
'1.3':'1.3: TO: 0.0115',
'1.4':'1.4: TO: 0.012',
'1.5':'1.5: TO: 0.0125',
'1.6':'1.6: TO: 0.013',
'1.7':'1.7: TO: 0.0135',
'1.8':'1.8: TO: 0.014',
'1.9':'1.9: TO: 0.0145',
'1.10':'1.10: TO: 0.015',
'1.11':'1.11: TO: 0.0155',
'1.12':'1.12: TO: 0.016',
'1.13':'1.13: TO: 0.0165',
'1.14':'1.14: TO: 0.017',
'1.15':'1.15: TO: 0.0175',
'1.16':'1.16: TO: 0.018',
'1.17':'1.17: TO: 0.0185',
'1.18':'1.18: TO: 0.019',
'1.19':'1.19: TO: 0.0195',
'1.20':'1.20: TO: 0.02',
'1.21':'1.21: TO: 0.0205',
'1.22':'1.22: TO: 0.021',
'1.23':'1.23: TO: 0.0215',
'1.24':'1.24: TO: 0.022',
'1.25':'1.25: TO: 0.0225',
'1.26':'1.26: TO: 0.023',
'1.27':'1.27: TO: 0.0235',
'1.28':'1.28: TO: 0.024',
'1.29':'1.29: TO: 0.0245',
'1.30':'1.30: TO: 0.025',
'1.31':'1.31: TO: 0.0255',
'1.32':'1.32: TO: 0.026',
'1.33':'1.33: TO: 0.0265',
'1.34':'1.34: TO: 0.027',
'1.35':'1.35: TO: 0.0275',
'1.36':'1.36: TO: 0.028',
'1.37':'1.37: TO: 0.0285',
'1.38':'1.38: TO: 0.029',
'1.39':'1.39: TO: 0.0295',
'1.40':'1.40: TO: 0.03'
    }

comments_dic['404'] = {
    'baseline':'baseline',
    '1.0':'1.0: SRDUS, UUPCOST, log loss',
    '1.1':'1.1: SRDUS, UUPCOST, ratio loss',
    '1.2':'1.2: SRDUS, PCOSTNOAGG, log loss',
    '1.3':'1.3: SRDUS, PCOSTNOAGG, ratio loss',
    '1.4':'1.4: no SRDUS, UUPCOST, log loss',
    '1.5':'1.5: no SRDUS, UUPCOST, ratio loss',
    '1.6':'1.6: no SRDUS, PCOSTNOAGG, log loss',
    '1.7':'1.7: no SRDUS, PCOSTNOAGG, ratio loss',
    '1.8':'1.8: no RD, UUPCOST, log loss',
    '1.9':'1.9: no RD, UUPCOST, ratio loss',
    '1.10':'1.10: no RD, PCOSTNOAGG, log loss',
    '1.11':'1.11: no RD, PCOSTNOAGG, ratio loss',
    '2.0':'2.0: sigma=2.7, SRDUS, UUPCOST',
    '2.1':'2.1: sigma=2.7, no SRDUS, UUPCOST',
    '2.2':'2.2: sigma=2.7, SRDUS, PCOSTNOAGG',
    '2.3':'2.3: sigma=2.7, no SRDUS, PCOSTNOAGG',
    }

# comments_dic['401'] = {"baseline":"baseline"}

baselines_dic_param = {}
baselines_dic_mom = {}
baselines_dic_sol_qty = {}

# baseline_list = ['311','312','401','402','403']    
# baseline_list = ['402','403','404']    
# baseline_list = ['403','404','405']    
baseline_list = ['404','405']    

def section(s):
     return [int(_) for _ in s.split(".")]
for baseline_nbr in baseline_list:
    baseline_path = results_path+baseline_nbr+'/'
    baseline_variations_path = results_path+'baseline_'+baseline_nbr+'_variations/'
        
    p_baseline,m_baseline,sol_baseline = load(baseline_path,data_path = data_path)
    baselines_dic_param[baseline_nbr], baselines_dic_mom[baseline_nbr], baselines_dic_sol_qty[baseline_nbr]\
        = init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,sol_baseline,list_of_moments)
    try:
        files_in_dir = next(os.walk(baseline_variations_path))[1]
        run_list = [f for f in files_in_dir if f[0].isnumeric()]
        # lists = sorted([s.split('.') for s in run_list], key=lambda x:map(int, x))  
        # run_list#.sort()
        run_list = sorted(run_list, key=section)
    
        for run in run_list:
            # print(run)
            if run not in ['2.1.9','99']:
                p_to_add,m_to_add,sol_to_add = load(baseline_variations_path+run+'/',data_path = data_path)
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
        if f[0].isnumeric() and f not in full_run_list:
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
baseline_mom = '405'
mom = 'SPFLOW'

baseline_mom_select = Select(value=baseline_mom, title='Baseline', options=sorted(baselines_dic_mom.keys()))
mom_select = Select(value=mom, title='Quantity', options=sorted(baselines_dic_mom[baseline_mom].keys()))

ds_mom = ColumnDataSource(baselines_dic_mom[baseline_mom][mom])
p_mom = figure(title="Moment matching", 
               width = 1400,
               height = 850,
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
labels = LabelSet(x='target', y='baseline', text='x',
              x_offset=2, y_offset=2, source=ds_mom, text_font_size="7pt")
p_mom.add_layout(labels)
p_mom.add_tools(hover_tool_mom)
slope1 = Slope(gradient=1, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=1)
slope2 = Slope(gradient=1.4876, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=0.25)
slope3 = Slope(gradient=0.5124, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=0.25)
slope4 = Slope(gradient=0.756198, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=0.25)
# slope5 = Slope(gradient=1.546, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)
# slope6 = Slope(gradient=2.20, y_intercept=0,
#               line_color='black', line_dash='dashed', line_width=0.25)

for slope in [slope1,slope2,slope3,slope4]:
# for slope in [slope1,slope2,slope3,slope4,slope5,slope6]:
    p_mom.add_layout(slope)
    
slope2.visible = False
slope3.visible = False
slope4.visible = False
# slope5.visible = False
# slope6.visible = False

colors_mom = itertools.cycle(Category10[10])

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
#                     spacing = 0, location=(10, -60))
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

# p_mom.add_layout(legend_mom, 'right')
p_mom.add_layout(legend_mom_split_1, 'right')
p_mom.add_layout(legend_mom_split_2, 'right')
# columns_mom = [TableColumn(field=col) for col in list(ds_mom.data.keys())]
columns_mom = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in ['target']+list(comments_dic[baseline_mom].keys())]
data_table_mom = DataTable(source=ds_mom, columns = columns_mom, width=1400, height=400)
    
def update_baseline_mom(attrname, old, new):
    mom = mom_select.value
    ds_mom.data = baselines_dic_mom[new][mom]
    # legend_items_mom = [LegendItem(label=comments_dic[new][col], 
    #                                 renderers=[lines_mom[i]]) for i,col in enumerate(ds_mom.data) if col not in ['x','target']]
    legend_items_mom = [LegendItem(label=comments_dic[new][col], renderers=[lines_mom[col]]) 
                        for col in ds_mom.data if col in comments_dic[new]]
    # p_mom.legend.items = legend_items_mom
    legend_mom_split_1.items = legend_items_mom[:round((len(legend_items_mom)+1)/2)]
    legend_mom_split_2.items = legend_items_mom[round((1+len(legend_items_mom))/2):]
    data_table_mom.columns = [
            TableColumn(field="x"),
        ]+[TableColumn(field=col) for col in ['target']+list(comments_dic[new].keys())]
    
def update_mom(attrname, old, new):
    baseline_mom = baseline_mom_select.value
    ds_mom.data = baselines_dic_mom[baseline_mom][new]
    if new == 'scalars':
        slope2.visible = True
        slope3.visible = True
        slope4.visible = True
        # slope5.visible = True
        # slope6.visible = True
    else:
        slope2.visible = False
        slope3.visible = False
        slope4.visible = False
        # slope5.visible = False
        # slope6.visible = False

# def update_legend():
#     legend_items_mom = [LegendItem(label=comments_dic[new][col], renderers=[lines_mom[col]]) 
#                         for col in ds_mom.data if col in comments_dic[new]]
#     p_mom.legend.items = legend_items_mom
    
# checkbox_buttons_labels = ["Variations .0", "Variations .1", "Variations .2"]
# checkbox_button_group = CheckboxButtonGroup(labels=checkbox_buttons_labels, active=[0, 1],
                                            # )
# checkbox_button_group.on_change('value', update_legend)


# controls_mom = row(baseline_mom_select, mom_select, checkbox_button_group)
controls_mom = row(baseline_mom_select, mom_select)

baseline_mom_select.on_change('value', update_baseline_mom)
mom_select.on_change('value', update_mom)

# curdoc().add_root(row(p_par, controls))
   

# baseline_par = '101'
baseline_par = '405'
par = 'delta'

baseline_par_select = Select(value=baseline_par, title='Baseline', options=sorted(baselines_dic_param.keys()))
par_select = Select(value=par, title='Quantity', options=sorted(baselines_dic_param[baseline_par].keys()))
x_range = baselines_dic_param[baseline_par][par_select.value].index.to_list()
ds_par = ColumnDataSource(baselines_dic_param[baseline_par][par])
p_par = figure(title="Parameters", 
               width = 1400,
               height = 850,
           x_range = x_range,
           y_axis_label='Model implied',
           tools = TOOLS)
hover_tool_par = HoverTool()
hover_tool_par.tooltips = [
    ("index", "@x"),
    ("value", "$y")
    ]

p_par.add_tools(hover_tool_par)
# p_par.sizing_mode = 'scale_width'
# colors_par = itertools.cycle(Category20.values()(len(baselines_dic_param[baseline_par][par].columns)))
colors_par = itertools.cycle(Category10[10])
lines_par = {}

for col in baselines_dic_param[baseline_par][par].columns:
    # lines_par[col] = p_par.line(x='x', y=col, source = ds_par, color=next(colors_par),
    #                             line_width = 2, legend_label=comments_dic[col])
    lines_par[col] = p_par.line(x='x', y=col, source = ds_par, color=next(colors_par),
                                line_width = 2)
    if col != 'baseline':
        lines_par[col].visible = False

legend_items_par = [LegendItem(label=comments_dic[baseline_par][col], renderers=[lin_par])
                    for col, lin_par in lines_par.items() if col in comments_dic[baseline_par]]
# legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8pt",spacing = 0
#                     # , location=(10, -30)
#                     )

legend_par_split_1 = Legend(items=legend_items_par[:round((len(legend_items_par)+1)/2)], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    # location=(10, -60)
                    )
legend_par_split_2 = Legend(items=legend_items_par[round((1+len(legend_items_par))/2):], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0
                    # , location=(10, -60)
                    )
# p_par.add_layout(legend_par, 'right')
p_par.add_layout(legend_par_split_1, 'right')
p_par.add_layout(legend_par_split_2, 'right')
# p_par.legend.click_policy="hide"
# p_par.legend.label_text_font_size = '8pt'
# p_par.legend.spacing = 0
# p_par.add_layout(p_par.legend[0], 'right')



columns_par = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in list(comments_dic[baseline_par].keys())]
# columns_par = [
#         TableColumn(field="x"),
#     ]+[TableColumn(field=col) for col in baselines_dic_param[baseline_par][par].columns]

data_table_par = DataTable(source=ds_par, columns = columns_par, width=1400, height=400)

def update_baseline_par(attrname, old, new):
    par = par_select.value
    ds_par.data = baselines_dic_param[new][par]
    legend_items_par = [LegendItem(label=comments_dic[new][col], renderers=[lines_par[col]])
                        for col in ds_par.data if col in comments_dic[new]]
    # legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8px",spacing = 0)
    # p_par.legend.items = legend_items_par
    legend_par_split_1.items = legend_items_par[:round((1+len(legend_items_par))/2)]
    legend_par_split_2.items = legend_items_par[round((len(legend_items_par)+1)/2):]
                      
    data_table_par.columns = [
            TableColumn(field="x"),
        ]+[TableColumn(field=col) for col in list(comments_dic[new].keys())]
    
def update_par(attrname, old, new):
    baseline_par = baseline_par_select.value
    p_par.x_range.factors = baselines_dic_param[baseline_par][new].index.to_list()
    ds_par.data = baselines_dic_param[baseline_par][new]

controls_par = row(baseline_par_select, par_select)
# controls_par.sizing_mode = 'scale_width'

baseline_par_select.on_change('value', update_baseline_par)
par_select.on_change('value', update_par)
# p_par.add_layout(p_par.legend[0], 'bottom right')

# baseline_sol_qty = '101'
baseline_sol_qty = '405'
sol_qty = 'psi_o_star'

baseline_sol_qty_select = Select(value=baseline_sol_qty, title='Baseline', options=sorted(baselines_dic_sol_qty.keys()))
sol_qty_select = Select(value=sol_qty, title='Quantity', options=sorted(baselines_dic_sol_qty[baseline_sol_qty].keys()))
x_range = baselines_dic_sol_qty[baseline_sol_qty][sol_qty_select.value].index.to_list()
ds_sol_qty = ColumnDataSource(baselines_dic_sol_qty[baseline_sol_qty][sol_qty])
p_sol_qty = figure(title="Solution quantities", 
               width = 1400,
               height = 850,
           x_range = x_range,
           y_axis_label='Model implied',
           tools = TOOLS)
hover_tool_sol_qty = HoverTool()
hover_tool_sol_qty.tooltips = [
    ("index", "@x"),
    ("value", "$y")
    ]

p_sol_qty.add_tools(hover_tool_sol_qty)
# p_par.sizing_mode = 'scale_width'
# colors_par = itertools.cycle(Category20.values()(len(baselines_dic_param[baseline_par][par].columns)))
colors_sol_qty = itertools.cycle(Category10[10])
lines_sol_qty = {}

for col in baselines_dic_sol_qty[baseline_sol_qty][sol_qty].columns:
    # lines_par[col] = p_par.line(x='x', y=col, source = ds_par, color=next(colors_par),
    #                             line_width = 2, legend_label=comments_dic[col])
    lines_sol_qty[col] = p_sol_qty.line(x='x', y=col, source = ds_sol_qty, color=next(colors_sol_qty),
                                line_width = 2)
    if col != 'baseline':
        lines_sol_qty[col].visible = False

legend_items_sol_qty = [LegendItem(label=comments_dic[baseline_sol_qty][col], renderers=[lin_sol_qty]) 
                        for col, lin_sol_qty in lines_sol_qty.items() if col in comments_dic[baseline_sol_qty]]
# legend_sol_qty = Legend(items=legend_items_sol_qty, click_policy="hide", 
#                         label_text_font_size="8pt",spacing = 0
#                         , location=(10, -30))

legend_sol_qty_split_1 = Legend(items=legend_items_sol_qty[:round((len(legend_items_sol_qty)+1)/2)], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    # location=(10, -60)
                    )
legend_sol_qty_split_2 = Legend(items=legend_items_sol_qty[round((len(legend_items_sol_qty)+1)/2):], click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0
                    # , location=(10, -60)
                    )
# p_sol_qty.add_layout(legend_sol_qty, 'right')
p_sol_qty.add_layout(legend_sol_qty_split_1, 'right')
p_sol_qty.add_layout(legend_sol_qty_split_2, 'right')
# p_par.legend.click_policy="hide"
# p_par.legend.label_text_font_size = '8pt'
# p_par.legend.spacing = 0
# p_par.add_layout(p_par.legend[0], 'right')



columns_sol_qty = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in list(comments_dic[baseline_sol_qty].keys())]

data_table_sol_qty = DataTable(source=ds_sol_qty, columns = columns_sol_qty, width=1400, height=400)

def update_baseline_sol_qty(attrname, old, new):
    sol_qty = sol_qty_select.value
    ds_sol_qty.data = baselines_dic_sol_qty[new][sol_qty]
    legend_items_sol_qty = [LegendItem(label=comments_dic[new][col], renderers=[lines_sol_qty[col]]) 
                            for col in ds_sol_qty.data  if col in comments_dic[new]]
    # legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8px",spacing = 0)
    # p_sol_qty.legend.items = legend_items_sol_qty
    legend_sol_qty_split_1.items = legend_items_sol_qty[:round((len(legend_items_sol_qty)+1)/2)]
    legend_sol_qty_split_2.items = legend_items_sol_qty[round((len(legend_items_sol_qty)+1)/2):]
    data_table_sol_qty.columns = [TableColumn(field=col) for col in list(comments_dic[new].keys())]
    
def update_sol_qty(attrname, old, new):
    baseline_sol_qty = baseline_sol_qty_select.value
    p_sol_qty.x_range.factors = baselines_dic_sol_qty[baseline_sol_qty][new].index.to_list()
    ds_sol_qty.data = baselines_dic_sol_qty[baseline_sol_qty][new]

controls_sol_qty = row(baseline_sol_qty_select, sol_qty_select)
# controls_par.sizing_mode = 'scale_width'

baseline_sol_qty_select.on_change('value', update_baseline_sol_qty)
sol_qty_select.on_change('value', update_sol_qty)
# p_par.add_layout(p_par.legend[0], 'bottom right')


# moment_report = column(controls_mom,p_mom)
moment_report = column(controls_mom,p_mom,data_table_mom)
# param_report = column(controls_par, p_par)
param_report = column(controls_par, p_par, data_table_par)
sol_qty_report = column(controls_sol_qty, p_sol_qty, data_table_sol_qty)
first_panel = row(moment_report,param_report,sol_qty_report)

#%% sensitivities

baselines_dic_sensi = {}

# for baseline_nbr in ['101','102','104']:
for baseline_nbr in ['403']:
    baselines_dic_sensi[baseline_nbr] = {} 
    baseline_sensi_path = results_path+'baseline_'+baseline_nbr+'_sensitivity_tables/'
    files_in_dir = os.listdir(baseline_sensi_path)
    files_in_dir = [ filename for filename in files_in_dir if filename.endswith('.csv') ]
    for f in files_in_dir:
        baselines_dic_sensi[baseline_nbr][f[:-4]] = pd.read_csv(baseline_sensi_path+f,index_col = 0)
    
# baseline_sensi = '101'
baseline_sensi = '403'
qty_sensi = 'objective'

baseline_sensi_select = Select(value=baseline_sensi, title='Baseline', options=sorted(baselines_dic_sensi.keys()))
qty_sensi_select = Select(value=qty_sensi, title='Quantity', options=sorted(baselines_dic_sensi[baseline_sensi].keys()))

ds_sensi = ColumnDataSource(baselines_dic_sensi[baseline_sensi][qty_sensi])
p_sensi = figure(title="Sensitivity", 
                width = 1200,
                height = 850,
                x_axis_label='Change in moment or parameter',
                y_axis_label='Value',
                tools = TOOLS)

colors_sensi = itertools.cycle(Category10[10])

for col in baselines_dic_sensi[baseline_sensi][qty_sensi].columns[1:]:
    if col!='zeta':
        p_sensi.line(x='Change', y=col, source = ds_sensi, color=next(colors_sensi),line_width = 2, legend_label=col)

p_sensi.legend.click_policy="hide"
p_sensi.legend.label_text_font_size = '8pt'
p_sensi.add_layout(p_sensi.legend[0], 'right')

def update_baseline_sensi(attrname, old, new):
    qty_sensi = qty_sensi_select.value
    ds_sensi.data = baselines_dic_sensi[new][qty_sensi]
    
def update_qty_sensi(attrname, old, new):
    baseline_sensi = baseline_sensi_select.value
    ds_sensi.data = baselines_dic_sensi[baseline_sensi][new]

controls_sensi = row(baseline_sensi_select, qty_sensi_select)
# controls_mom.sizing_mode = 'scale_width'

baseline_sensi_select.on_change('value', update_baseline_sensi)
qty_sensi_select.on_change('value', update_qty_sensi)

sensitivity_report = column(controls_sensi,p_sensi)

#%% weights sensitivities

baselines_dic_sensi_weights = {}

# for baseline_nbr in ['101','102','104']:
for baseline_nbr in ['404']:
    baselines_dic_sensi_weights[baseline_nbr] = {}
    baseline_sensi_weights_path = results_path+'baseline_'+baseline_nbr+'_sensitivity_weights_tables/'
    files_in_dir = os.listdir(baseline_sensi_weights_path)
    files_in_dir = [ filename for filename in files_in_dir if filename.endswith('.csv') ]
    for f in files_in_dir:
        baselines_dic_sensi_weights[baseline_nbr][f[:-4]] = pd.read_csv(baseline_sensi_weights_path+f,index_col = 0)
    
baseline_sensi_weights = '404'
qty_sensi_weights = 'objective'

baseline_sensi_weights_select = Select(value=baseline_sensi_weights, title='Baseline', options=sorted(baselines_dic_sensi_weights.keys()))
qty_sensi_weights_select = Select(value=qty_sensi_weights, title='Quantity', options=sorted(baselines_dic_sensi_weights[baseline_sensi_weights].keys()))

ds_sensi_weights = ColumnDataSource(baselines_dic_sensi_weights[baseline_sensi_weights][qty_sensi_weights])
p_sensi_weights = figure(title="Sensitivity to the weights", 
                width = 1200,
                height = 850,
                x_axis_label='Change in weight',
                y_axis_label='Objective function or contribution to objective function: loss(moment,target)',
                y_axis_type="log",
                tools = TOOLS)

colors_sensi_weights = itertools.cycle(Category10[10])

for col in baselines_dic_sensi_weights[baseline_sensi_weights][qty_sensi_weights].columns[1:]:
    if col!='zeta':
        p_sensi_weights.line(x='Change', y=col, source = ds_sensi_weights, color=next(colors_sensi_weights),line_width = 2, legend_label=col)

p_sensi_weights.legend.click_policy="hide"
p_sensi_weights.legend.label_text_font_size = '8pt'
p_sensi_weights.add_layout(p_sensi_weights.legend[0], 'right')

def update_baseline_sensi_weights(attrname, old, new):
    qty_sensi_weights = qty_sensi_weights_select.value
    ds_sensi_weights.data = baselines_dic_sensi_weights[new][qty_sensi_weights]
    
def update_qty_sensi_weights(attrname, old, new):
    baseline_sensi_weights = baseline_sensi_weights_select.value
    ds_sensi_weights.data = baselines_dic_sensi_weights[baseline_sensi_weights][new]

controls_sensi_weights = row(baseline_sensi_weights_select, qty_sensi_weights_select)
# controls_mom.sizing_mode = 'scale_width'

baseline_sensi_weights_select.on_change('value', update_baseline_sensi_weights)
qty_sensi_weights_select.on_change('value', update_qty_sensi_weights)

sensitivity_weights_report = column(controls_sensi_weights,p_sensi_weights)

#%% Jacobian panel

baseline_jac = '405'
country_jac = 'USA'
sector_jac = 'Patent'

# baseline_jac_select = Select(value=baseline_jac, title='Baseline', options=['311','312','401','402','403','404','405'])
baseline_jac_select = Select(value=baseline_jac, title='Baseline', options=['404','405'])

baseline_jac_path = results_path+'baseline_'+baseline_jac+'_variations/'
files_in_dir = next(os.walk(baseline_jac_path))[1]
run_list = [f for f in files_in_dir if f[0].isnumeric()]
run_list = sorted(run_list, key=section)
variation_jac_select = Select(value='baseline', title='Variation', 
                              options=['baseline']+run_list)

def update_list_of_runs_jac(attr, old, new):
    baseline_jac_path = results_path+'baseline_'+new+'_variations/'
    files_in_dir = next(os.walk(baseline_jac_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list = sorted(run_list, key=section)
    variation_jac_select.options = ['baseline']+run_list

if variation_jac_select.value == 'baseline':
    path = results_path+baseline_jac_select.value+'/'
else:
    path = results_path+'baseline_'+baseline_jac_select.value+'_variations/'+variation_jac_select.value+'/'
    
p_jac, m_jac, sol_jac = load(path, data_path=data_path)

qty_jac_select = Select(value='delta', title='Parameter', options=p_jac.calib_parameters)
country_jac_select = Select(value='USA', title='Country', options=p_jac.countries)
sector_jac_select = Select(value='Patent', title='Sector', options=p_jac.sectors)

if qty_jac_select.value in ['eta','T','delta','nu']:
    idx_to_change_jac = p_jac.countries.index(country_jac_select.value),p_jac.sectors.index(sector_jac_select.value)
if qty_jac_select.value in ['fe','zeta','nu', 'fo']:
    idx_to_change_jac = 0,p_jac.sectors.index(sector_jac_select.value)
if qty_jac_select.value in ['k','g_0']:
    idx_to_change_jac = 0

qty_to_change_jac = qty_jac_select.value

x_jac = compute_rough_jacobian(p_jac, m_jac, qty_to_change_jac, idx_to_change_jac, 
                           change_by = 0.25, tol = 1e-14, damping = 5,
                           max_count = 5e3)

p_jac = figure(title="Rough jacobian computation", 
               y_range=FactorRange(factors=m_jac.get_signature_list()),
                width = 1200,
                height = 850,
                x_axis_label='Change in contribution to objective function',
                y_axis_label='Moment',
                tools = TOOLS) 

data_jac = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([m_jac.get_signature_list(),x_jac]).T)
src_jac = ColumnDataSource(data_jac)

# p_jac.hbar(y = 'Moment',right = 'Contribution', source = src_jac)
p_jac.hbar(y = 'Moment',right = 'Contribution', source = src_jac)

def update_jac(event):
    if variation_jac_select.value == 'baseline':
        path = results_path+baseline_jac_select.value+'/'
    else:
        path = results_path+'baseline_'+baseline_jac_select.value+'_variations/'+variation_jac_select.value+'/'
    par_jac, m_jac, sol_jac = load(path, data_path=data_path)
    if qty_jac_select.value in ['eta','T','delta','nu']:
        idx_to_change_jac = par_jac.countries.index(country_jac_select.value),par_jac.sectors.index(sector_jac_select.value)
    if qty_jac_select.value in ['fe','zeta','nu', 'fo']:
        idx_to_change_jac = par_jac.sectors.index(sector_jac_select.value)
    if qty_jac_select.value in ['k','g_0']:
        idx_to_change_jac = None
    x_jac = compute_rough_jacobian(par_jac, m_jac, qty_jac_select.value, idx_to_change_jac, 
                               change_by = 0.1, tol = 1e-14, damping = 5,
                               max_count = 5e3)
    data_jac = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([m_jac.get_signature_list(),x_jac]).T)
    src_jac.data = data_jac
    p_jac.y_range.factors = m_jac.get_signature_list()

button_jac = Button(label="Compute")
button_jac.on_event(ButtonClick, update_jac)

controls_jac = row(baseline_jac_select, variation_jac_select, qty_jac_select, 
                   country_jac_select, sector_jac_select, button_jac)

baseline_jac_select.on_change('value', update_list_of_runs_jac)

jac_report = column(controls_jac,p_jac)

second_panel = row(sensitivity_report,sensitivity_weights_report,jac_report)

#%% counterfactuals

# baseline_cf = '101'
baseline_cf = '404'
country_cf = 'USA'

# p_baseline,m_baseline,sol_baseline = load(results_path+baseline_cf+'/',data_path = data_path)
def section_end(s):
     return [int(_) for _ in s.split("_")[-1].split(".")]
cf_list = sorted([s for s in os.listdir(cf_path) 
            if s[9:].startswith('404') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('402') and s.startswith('baseline')], key=section_end)#+\
    # sorted([s for s in os.listdir(cf_path) 
    #             if s[9:].startswith('312') and s.startswith('baseline')], key=section_end)+\
    # sorted([s for s in os.listdir(cf_path) 
    #         if s[9:].startswith('311') and s.startswith('baseline')], key=section_end)

# baseline_cf_select = Select(value=baseline_cf, title='Baseline', options=[s[9:] for s in sorted(os.listdir(cf_path)) 
# baseline_cf_select = Select(value=baseline_cf, title='Baseline', options=[s[9:] for s in sorted(cf_list, key=section_end)])
baseline_cf_select = Select(value=baseline_cf, title='Baseline', options=[s[9:] for s in cf_list])
country_cf_select = Select(value=country_cf, 
                            title='Country', 
                            # options=countries+['World','Harmonizing','World_2','Harmonizing_2'])
                            options=countries+['World','Harmonizing'])

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
    df_max['colors'] = Category10[10][:len(df_max)]
    return df_max

df_cf = get_data_cf(baseline_cf,country_cf)
ds_cf = ColumnDataSource(df_cf)
df_cf_max = build_max(df_cf)
ds_cf_max = ColumnDataSource(df_cf_max)

colors_cf = itertools.cycle(Category10[10])
colors_cf_max = itertools.cycle(Category10[10])

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

# p_cf.extra_y_ranges['growth'] = DataRange1d()
# p_cf.add_layout(LinearAxis(y_range_name='growth', axis_label='Growth rate'), 'right')
# p_cf.line(x='delt', y='growth', source = ds_cf, color = 'black', legend_label = 'growth',y_range_name = 'growth')
     
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
    # ds_cf.data = get_data_cf(baseline_cf,new)
    
controls_cf = row(baseline_cf_select, country_cf_select)
# controls_mom.sizing_mode = 'scale_width'

baseline_cf_select.on_change('value', update_baseline_cf)
country_cf_select.on_change('value', update_country_cf)

counterfactuals_report = column(controls_cf,p_cf)

#%% counterfactuals 403 TO target

# baseline_cf = '101'
# baseline_to_cf = '403'
country_to_cf = 'USA'
to_target = 0.0185

list_of_to_targets = np.linspace(0.01,0.03,41)

def section_end(s):
     return [int(_) for _ in s.split("_")[-1].split(".")]
# cf_to_list = list(reversed(sorted([s for s in os.listdir(cf_path) 
#             if s[9:].startswith('403') and s.startswith('baseline')], key=section_end)))
cf_to_list = sorted([s for s in os.listdir(cf_path) 
            if s[9:].startswith('405') and s.startswith('baseline')], key=section_end)

def get_data_to_cf(to_target,country):
    idx_to_cf = np.argmin(np.abs(list_of_to_targets-to_target))
    df_to_cf = pd.read_csv(cf_path+cf_to_list[min(idx_to_cf,len(cf_to_list)-1)]+'/'+country+'.csv')
    if country != 'Harmonizing':
        df_to_cf['Growth rate'] = df_to_cf['growth']/df_to_cf.loc[np.argmin(np.abs(df_to_cf.delt-1))].growth
    if country == 'Harmonizing':
        df_to_cf['Growth rate'] = df_to_cf['growth']/df_to_cf.loc[np.argmin(np.abs(df_to_cf.delt))].growth
    df_to_cf.set_index('delt',inplace=True)
    return df_to_cf

def build_max(df_to_cf):
    df_max = pd.concat([df_to_cf.idxmax(),df_to_cf.max()],axis=1)
    df_max.index.name = 'label'
    df_max.columns = ['xmax','max'] 
    df_max = df_max.loc[countries]
    df_max['colors'] = Category10[10][:len(df_max)]
    return df_max

country_to_cf_select = Select(value=country_to_cf, 
                            title='Country', 
                            # options=countries+['World','Harmonizing','World_2','Harmonizing_2'])
                            options=countries+['World','Harmonizing'])

df_to_cf = get_data_to_cf(to_target,country_cf)
ds_to_cf = ColumnDataSource(df_to_cf)
df_to_cf_max = build_max(df_to_cf)
ds_to_cf_max = ColumnDataSource(df_to_cf_max)

colors_to_cf = itertools.cycle(Category10[10])
colors_to_cf_max = itertools.cycle(Category10[10])

p_to_cf = figure(title="Patent protection counterfactual as function of TO target, baseline 405", 
                width = 1200,
                height = 850,
                x_axis_label='Change in delta',
                y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
                x_axis_type="log",
                tools = TOOLS) 

for col in df_to_cf.columns:
    if col not in [0,'delt','growth']:
        p_to_cf.line(x='delt', y=col, source = ds_to_cf, color=next(colors_to_cf),line_width = 2, legend_label=col)

p_to_cf.circle(x = 'xmax', y = 'max', source = ds_to_cf_max, size=4, color='colors')

# p_cf.extra_y_ranges['growth'] = DataRange1d()
# p_cf.add_layout(LinearAxis(y_range_name='growth', axis_label='Growth rate'), 'right')
# p_cf.line(x='delt', y='growth', source = ds_cf, color = 'black', legend_label = 'growth',y_range_name = 'growth')
     
p_to_cf.legend.click_policy="hide"
p_to_cf.legend.label_text_font_size = '8pt'
p_to_cf.add_layout(p_to_cf.legend[0], 'right')

def update_baseline_to_cf(attrname, old, new):
    country_to_cf = country_to_cf_select.value
    # ds_to_cf.data = get_data_to_cf(new,country_to_cf)
    df_to_cf = get_data_to_cf(new/100,country_to_cf)
    ds_to_cf.data = df_to_cf
    ds_to_cf_max.data = build_max(df_to_cf)
    
def update_country_to_cf(attrname, old, new):
    # baseline_to_cf = baseline_cf_to_select.value
    to_target = slider_to_cf.value/100
    df_to_cf = get_data_to_cf(to_target,new)
    ds_to_cf.data = df_to_cf
    ds_to_cf_max.data = build_max(df_to_cf)
    # ds_cf.data = get_data_cf(baseline_cf,new)
    
slider_to_cf = Slider(start=1, end=3, value=1.85, step=0.05, title="Turnover target in %")    
    
controls_to_cf = row(slider_to_cf, country_to_cf_select)
country_to_cf_select.on_change('value', update_country_to_cf)
slider_to_cf.on_change('value', update_baseline_to_cf)

counterfactuals_to_report = column(controls_to_cf,p_to_cf)

# third_panel = row(counterfactuals_report, counterfactuals_to_report)

#%% dynamic solver

baseline_dyn = '405'
country_dyn = 'USA'
sector_dyn = 'Patent'

# baseline_dyn_select = Select(value=baseline_dyn, title='Baseline', options=['311','312','401','402','403','404','405'])
baseline_dyn_select = Select(value=baseline_dyn, title='Baseline', options=['404','405'])

baseline_dyn_path = results_path+'baseline_'+baseline_dyn+'_variations/'
files_in_dir = next(os.walk(baseline_dyn_path))[1]
run_list = [f for f in files_in_dir if f[0].isnumeric()]
run_list = sorted(run_list, key=section)
variation_dyn_select = Select(value='baseline', title='Variation', 
                              options=['baseline']+run_list)

def update_list_of_runs_dyn(attr, old, new):
    baseline_dyn_path = results_path+'baseline_'+new+'_variations/'
    files_in_dir = next(os.walk(baseline_dyn_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list = sorted(run_list, key=section)
    variation_dyn_select.options = ['baseline']+run_list

country_dyn_select = Select(value='USA', title='Country delta to change', options=['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW','World'])
# country_dyn_select = Select(value='USA', title='Country delta to change', options=['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'])
slider_dyn = Slider(start=-1, end=1, value=0, step=0.01, title="Log change of delta")    

state_computation = Div(text="Done")

def create_column_data_source_from_dyn_sol(dyn_sol):
    data_dyn = {}
    data_dyn['time'] = dyn_sol.t_real
    for agg_qty in ['g']:
        data_dyn[agg_qty] = getattr(dyn_sol,agg_qty)
    for c_qty in ['Z','r','price_indices','w','nominal_final_consumption']:
        for i,c in enumerate(dyn_sol.countries):
            data_dyn[c_qty+c] = getattr(dyn_sol,c_qty)[i,:].ravel()
    for c_s_qty in ['l_R','psi_o_star','PSI_CD','l_Ao']:
        for i,c in enumerate(dyn_sol.countries):
            data_dyn[c_s_qty+c] = getattr(dyn_sol,c_s_qty)[i,1,:].ravel()
    for c_c_s_qty in ['l_Ae','PSI_MPD','PSI_MPND','PSI_MNP','profit']:
        temp_sum_n = getattr(dyn_sol,c_c_s_qty).sum(axis=0)
        temp_sum_i = getattr(dyn_sol,c_c_s_qty).sum(axis=1)
        for i,c in enumerate(dyn_sol.countries):
            data_dyn['sum_n_'+c_c_s_qty+c] = temp_sum_n[i,1,:].ravel()
            data_dyn['sum_i_'+c_c_s_qty+c] = temp_sum_i[i,1,:].ravel()
    for i,c in enumerate(dyn_sol.countries):
        data_dyn['real_final_consumption'+c] = (getattr(dyn_sol,'nominal_final_consumption')[i,:]/getattr(dyn_sol,'price_indices')[i,:]).ravel()
    # for c_s_qty in ['l_R','l_Ae','l_Ao']:
    # ds_dyn = ColumnDataSource(data_dyn)
    return data_dyn

def compute_dyn(event):
    if variation_dyn_select.value == 'baseline':
        path = results_path+baseline_dyn_select.value+'/'
    else:
        path = results_path+'baseline_'+baseline_dyn_select.value+'_variations/'+variation_dyn_select.value+'/'
    p_dyn, m_dyn, sol_dyn = load(path, data_path=data_path)
    p_dyn_cf = p_dyn.copy()
    if country_dyn_select.value != 'World':
        p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1] = p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1]*(10**slider_dyn.value)
    else:
        p_dyn_cf.delta[:,1] = p_dyn_cf.delta[:,1]*(10**slider_dyn.value)
    start = time.perf_counter()
    dyn_sol, sol_c, convergence = rough_dyn_fixed_point_solver(p_dyn_cf, sol_dyn, sol_fin = None,Nt=25,
                                          t_inf=500, x0=None, tol = 1e-14, max_count=1e6, safe_convergence=0.1,damping=50, damping_post_acceleration=10)
    # print('Done')
    end = time.perf_counter()
    # print(slider_dyn.value)
    if country_dyn_select.value == 'World':
        message = 'Done, computation for all deltas multiplied by a factor '+str(10**slider_dyn.value)+'<br>Convergence : '+str(convergence)+'<br>Computation time : '+str(end-start)
    else:
        message = 'Done, computation for delta '+country_dyn_select.value+' = '+str(p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1])+'<br>Convergence : '+str(convergence)+'<br>Computation time : '+str(end-start)
    state_computation.text = message
    ds_dyn.data = create_column_data_source_from_dyn_sol(dyn_sol)
    
    # return dyn_sol, sol_c
if variation_dyn_select.value == 'baseline':
    path = results_path+baseline_dyn_select.value+'/'
else:
    path = results_path+'baseline_'+baseline_dyn_select.value+'_variations/'+variation_dyn_select.value+'/'
p_dyn, m_dyn, sol_dyn = load(path, data_path=data_path)
p_dyn_cf = p_dyn.copy()
if country_dyn_select.value != 'World':
    p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1] = p_dyn_cf.delta[p_dyn.countries.index(country_dyn_select.value),1]*10**slider_dyn.value
else:
    p_dyn_cf.delta[:,1] = p_dyn_cf.delta[:,1]*slider_dyn.value
dyn_sol, sol_c, convergence = rough_dyn_fixed_point_solver(p_dyn_cf, sol_dyn, sol_fin = None,Nt=25,
                                      t_inf=500, x0=None, tol = 1e-14, max_count=1e6, safe_convergence=0.1,damping=50, damping_post_acceleration=10)

button_compute_dyn = Button(label="Compute")
button_compute_dyn.on_event(ButtonClick, compute_dyn)

qty_dyn_display_select = Select(value='g', title='Quantity', options=['g','Z','r','price_indices','w','nominal_final_consumption','real_final_consumption',
                                                    'l_R','l_Ao','psi_o_star','PSI_CD',
                                                    'sum_n_l_Ae','sum_n_PSI_MPD','sum_n_PSI_MPND','sum_n_PSI_MNP','sum_n_profit',
                                                    'sum_i_l_Ae','sum_i_PSI_MPD','sum_i_PSI_MPND','sum_i_PSI_MNP','sum_i_profit'])
country_dyn_display_select = Select(value='USA', title='Country', options=['USA', 'EUR', 'JAP', 'CHN', 'BRA', 'IND', 'ROW'])

# df_dyn = pd.DataFrame(index=pd.Index(dyn_sol.t_real,name='time'))
# df_dyn['Value'] = dyn_sol.r[p_dyn.countries.index(country_dyn_display_select.value),:]
# ds_dyn = ColumnDataSource(df_dyn)
data_dyn_default = create_column_data_source_from_dyn_sol(dyn_sol)
ds_dyn = ColumnDataSource(data_dyn_default)
delta = max(ds_dyn.data['g'])-min(ds_dyn.data['g'])
if delta == 0:
    delta = 1
p_dyn_figure = figure(title="Dynamic solver",
                width = 1200,
                height = 850,
                x_axis_label='Time',
                y_axis_label='Value',
                # x_axis_type="log",
                tools = TOOLS,
                x_range = (0,100),
                y_range=(min(ds_dyn.data['g'])-delta*0.1,max(ds_dyn.data['g'])+delta*0.1)
                )

# p_dyn_figure.y_range=Range1d(min(ds_dyn.data['g']), max(ds_dyn.data['g']))

lines_dyn = {}
for col in data_dyn_default.keys():
    if col != time:
        lines_dyn[col] = p_dyn_figure.line(x='time', y=col, source = ds_dyn)
        if col != 'g':
            lines_dyn[col].visible = False
            
            
# line_dyn = p_dyn_figure.line(x='time', y='g', source = ds_dyn)

def update_graph_dyn(event):
    # p_dyn_figure.renderers = []
    if qty_dyn_display_select.value in ['g']:
        col = qty_dyn_display_select.value
    elif qty_dyn_display_select.value in ['Z','r','price_indices','w','nominal_final_consumption','real_final_consumption',
                                'l_R','l_Ao','psi_o_star','PSI_CD',
                                'sum_n_l_Ae','sum_n_PSI_MPD','sum_n_PSI_MPND','sum_n_PSI_MNP','sum_n_profit',
                                'sum_i_l_Ae','sum_i_PSI_MPD','sum_i_PSI_MPND','sum_i_PSI_MNP','sum_i_profit']:
        col = qty_dyn_display_select.value+country_dyn_display_select.value
    # print(col)
    lines_dyn[col].visible = True
    for other_column in lines_dyn:
        if other_column != col:
            lines_dyn[other_column].visible = False
    # print(min(ds_dyn.data[col])-(min(ds_dyn.data[col])>0)*min(ds_dyn.data[col])*0.1+(min(ds_dyn.data[col])<0)*min(ds_dyn.data[col])*0.1)
    # print(max(ds_dyn.data[col])+(max(ds_dyn.data[col])>0)*max(ds_dyn.data[col])*0.1-(max(ds_dyn.data[col])<0)*max(ds_dyn.data[col])*0.1)
    # p_dyn_figure.y_range=Range1d(min(ds_dyn.data[col])-(min(ds_dyn.data[col])>0)*min(ds_dyn.data[col])*0.1+(min(ds_dyn.data[col])<0)*min(ds_dyn.data[col])*0.1
    #                              ,max(ds_dyn.data[col])+(max(ds_dyn.data[col])>0)*max(ds_dyn.data[col])*0.1-(max(ds_dyn.data[col])<0)*max(ds_dyn.data[col])*0.1)
    delta = max(ds_dyn.data[col])-min(ds_dyn.data[col])
    if delta == 0:
        delta = 1
    p_dyn_figure.y_range.start=min(ds_dyn.data[col])-delta*0.1
    p_dyn_figure.y_range.end=max(ds_dyn.data[col])+delta*0.05
        
button_display_dyn = Button(label="Display")
button_display_dyn.on_event(ButtonClick, update_graph_dyn)

controls_dyn = row(baseline_dyn_select, variation_dyn_select, country_dyn_select, slider_dyn, button_compute_dyn, state_computation)
controls_display_dyn = row(qty_dyn_display_select, 
                           country_dyn_display_select,
                           button_display_dyn)

# data_dyn = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([m_dyn.get_signature_list(),x_dyn]).T)
# src_dyn = ColumnDataSource(data_dyn)

# # p_dyn.hbar(y = 'Moment',right = 'Contribution', source = src_dyn)
# p_dyn.hbar(y = 'Moment',right = 'Contribution', source = src_dyn)

# def update_dyn(event):
#     if variation_dyn_select.value == 'baseline':
#         path = results_path+baseline_dyn_select.value+'/'
#     else:
#         path = results_path+'baseline_'+baseline_dyn_select.value+'_variations/'+variation_dyn_select.value+'/'
#     par_dyn, m_dyn, sol_dyn = load(path, data_path=data_path)
#     if qty_dyn_select.value in ['eta','T','delta','nu']:
#         idx_to_change_dyn = par_dyn.countries.index(country_dyn_select.value),par_dyn.sectors.index(sector_dyn_select.value)
#     if qty_dyn_select.value in ['fe','zeta','nu', 'fo']:
#         idx_to_change_dyn = par_dyn.sectors.index(sector_dyn_select.value)
#     if qty_dyn_select.value in ['k','g_0']:
#         idx_to_change_dyn = None
#     x_dyn = compute_rough_dynobian(par_dyn, m_dyn, qty_dyn_select.value, idx_to_change_dyn, 
#                                change_by = 0.1, tol = 1e-14, damping = 5,
#                                max_count = 5e3)
#     data_dyn = pd.DataFrame(columns = ['Moment','Contribution'], data=np.array([m_dyn.get_signature_list(),x_dyn]).T)
#     src_dyn.data = data_dyn
#     p_dyn.y_range.factors = m_dyn.get_signature_list()

# button_dyn = Button(label="Compute")
# button_dyn.on_event(ButtonClick, update_dyn)

# controls_dyn = row(baseline_dyn_select, variation_dyn_select, qty_dyn_select, 
#                    country_dyn_select, sector_dyn_select, button_dyn)

# baseline_dyn_select.on_change('value', update_list_of_runs_dyn)

dyn_report = column(controls_dyn,controls_display_dyn,p_dyn_figure)


third_panel = row(counterfactuals_report, counterfactuals_to_report, dyn_report)


        
    
    

#%% Nash / coop equilibrium

# nash_eq_path = 'nash_eq_recaps/'
# coop_eq_path = 'coop_eq_recaps/'
def section_ser(s):
     return pd.Series([[int(_) for _ in s_e.split(".")] for s_e in s])

baseline_nash_coop = '405'

dic_change_labels_for_405 = {'405, '+k:comments_dic['403'][k] for k in comments_dic['405']}

def get_data_nash_coop(baseline_nash_number):

    welf_coop = pd.read_csv(coop_eq_path+'cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation','aggregation_method'],keep='last').sort_values(['baseline','variation'])
    welf_nash = pd.read_csv(nash_eq_path+'cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation'],keep='last').sort_values(['baseline','variation'])
    
    # welf_coop['pop w av'] = ((welf_coop[p_baseline.countries].T.values*p_baseline.data.labor.values[:,None]).sum(axis=0)/p_baseline.data.labor.values.sum())
    # welf_nash['pop w av'] = ((welf_nash[p_baseline.countries].T.values*p_baseline.data.labor.values[:,None]).sum(axis=0)/p_baseline.data.labor.values.sum())
    
    welf_coop['run'] = welf_coop['baseline'].astype('str')+', '+welf_coop['variation']
    welf_nash['run'] = welf_nash['baseline'].astype('str')+', '+welf_nash['variation']

    welf_coop['run'] = welf_coop['run'].replace(dic_change_labels_for_405)
    welf_nash['run'] = welf_nash['run'].replace(dic_change_labels_for_405)
    
    welf_coop['sorting'] = welf_coop['variation'].str.replace('baseline','0')#.astype(float)
    welf_nash['sorting'] = welf_nash['variation'].str.replace('baseline','0')#.astype(float)
    
    # welf_coop = welf_coop.sort_values(['baseline','sorting'])
    # welf_nash = welf_nash.sort_values(['baseline','sorting'])
    welf_coop = welf_coop.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    welf_nash = welf_nash.sort_values('sorting',key=section_ser)#.sort_values('baseline')
    
    welf_coop = welf_coop[welf_coop['baseline'].isin([int(baseline_nash_number)])]
    welf_nash = welf_nash[welf_nash['baseline'].isin([int(baseline_nash_number)])]
    
    welf_negishi = welf_coop[welf_coop['aggregation_method'] == 'negishi']
    welf_pop_weighted = welf_coop[welf_coop['aggregation_method'] == 'pop_weighted']
    
    return welf_pop_weighted, welf_negishi, welf_nash

# baseline_nash_coop_select = Select(value=baseline_nash_coop, title='Baseline', options=['311','312','401','402','403'])
# baseline_nash_coop_select = Select(value=baseline_nash_coop, title='Baseline', options=['402','403','404'])
# baseline_nash_coop_select = Select(value=baseline_nash_coop, title='Baseline', options=['403','404'])
baseline_nash_coop_select = Select(value=baseline_nash_coop, title='Baseline', options=['404','405'])

welf_pop_weighted, welf_negishi, welf_nash = get_data_nash_coop(baseline_nash_coop)
    
ds_pop_weighted = ColumnDataSource(welf_pop_weighted)
ds_negishi = ColumnDataSource(welf_negishi)
ds_nash = ColumnDataSource(welf_nash)

colors_pop_weighted = itertools.cycle(Category10[10])
colors_negishi = itertools.cycle(Category10[10])
colors_nash = itertools.cycle(Category10[10])

x_range_nash = welf_nash['run'].to_list()

p_eq = figure(title="Cooperative and Nash equilibrium", 
                width = 1400,
                height = 850,
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
p_eq.add_layout(p_eq.legend[0], 'right')    

hover_tool_eq = HoverTool()
hover_tool_eq.tooltips = [
    ("run", "@run"),
    ("value", "$y")
    ] 
p_eq.add_tools(hover_tool_eq)

# data_table_eq = dict(
#         runs=[run for run in comments_dic[baseline_sol_qty].keys()],
#         comments=[comment for comment in comments_dic[baseline_sol_qty].values()],
#     )
# source_table_eq = ColumnDataSource(data_table_eq)

columns = [
        TableColumn(field="runs", title="Runs"),
        TableColumn(field="comments", title="Description"),
    ]
# data_table_eq = DataTable(source=source_table_eq, columns=columns, width=400, height=600,
                            # autosize_mode="force_fit"
                          # )

# explication = Text(text="First is the quantity displayed\n'Negishi coop equal' means that we display the ")
explication = Div(text="In the legend, first is the quantity displayed and last\
                  is the quantity maximized <br> 'Negishi coop equal' means that: <br> \
                      - we display the Change in cons equivalent of world welfare <br> according to Negishi weights aggregation<br>\
                      - we maximize according to the Change in cons equivalent of world welfare <br> according to equal weights aggregation\
                          ")

# help_panel = column(explication,data_table_eq)

data_table_welfares = pd.concat([welf_nash.set_index('run'),
              welf_negishi.set_index('run'),
              welf_pop_weighted.set_index('run')],
            axis=0,
            keys=['Nash','Coop Negishi','Coop equal'],
            names=['type','run'],
            sort=False
            # ).reset_index().sort_values(['baseline','sorting','type'])[['run','type']+p_baseline.countries+['Equal']+['Negishi']]
            ).reset_index().sort_values('sorting',key=section_ser)[['run','type']+p_baseline.countries+['Equal']+['Negishi']]

source_table_welfares = ColumnDataSource(data_table_welfares)
columns_welf = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries+['Equal']+['Negishi']]

table_widget_welfares = DataTable(source=source_table_welfares, columns=columns_welf, width=850, height=400,
                            # autosize_mode="force_fit"
                          )

# p_eq.show()
def get_delta_nash_coop(baseline_number):
    deltas_coop = pd.read_csv(coop_eq_path+'deltas.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation','aggregation_method'],keep='last').sort_values(['baseline','variation'])
    deltas_nash = pd.read_csv(nash_eq_path+'deltas.csv',index_col=0).drop_duplicates(['baseline', 
                                'variation'],keep='last').sort_values(['baseline','variation'])
    
    deltas_coop['run'] = deltas_coop['baseline'].astype('str')+', '+deltas_coop['variation']
    deltas_nash['run'] = deltas_nash['baseline'].astype('str')+', '+deltas_nash['variation']
    
    deltas_coop['run'] = deltas_coop['run'].replace(dic_change_labels_for_405)
    deltas_nash['run'] = deltas_nash['run'].replace(dic_change_labels_for_405)
    
    deltas_coop['sorting'] = deltas_coop['variation'].str.replace('baseline','0')#.astype(float)
    deltas_nash['sorting'] = deltas_nash['variation'].str.replace('baseline','0')#.astype(float)
    
    # deltas_coop = deltas_coop.sort_values(['baseline','sorting'])
    # deltas_nash = deltas_nash.sort_values(['baseline','sorting'])
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

colors_deltas_negishi = itertools.cycle(Category10[10])
colors_deltas_pop_weighted = itertools.cycle(Category10[10])
colors_deltas_nash = itertools.cycle(Category10[10])

# x_range = deltas_nash['run'].to_list()

p_deltas_eq = figure(title="Cooperative and Nash equilibrium", 
                width = 1400,
                height = 850,
                x_range = x_range_nash,
                y_axis_type="log",
                # x_axis_label='Run',
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
            # ).reset_index().sort_values(['baseline','sorting','type'])[['run','type']+p_baseline.countries]
            ).reset_index().sort_values('sorting',key=section_ser)[['run','type']+p_baseline.countries]

source_table_deltas = ColumnDataSource(data_table_deltas)
columns_deltas = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries+['Equal']+['Negishi']]

table_widget_deltas = DataTable(source=source_table_deltas, columns=columns_deltas, width=850, height=400,
                            # autosize_mode="force_fit"
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
    
    # if new != '403':
    p_eq.x_range.factors = welf_nash['run'].to_list()
    p_deltas_eq.x_range.factors = welf_nash['run'].to_list()
    # else:
    #     p_eq.x_range.factors = [comments_dic['403'][k][4:] for k in comments_dic['403']][:len(p_eq.x_range.factors)]
    #     p_deltas_eq.x_range.factors = p_eq.x_range.factors
    # if new == '403':
    #     p_eq.xaxis.major_label_overrides = {'403, '+k:comments_dic['403'][k] for k in comments_dic['403']}
    #     p_deltas_eq.xaxis.major_label_overrides = {'403, '+k:comments_dic['403'][k] for k in comments_dic['403']}

baseline_nash_coop_select.on_change('value', update_baseline_nash)

nash_coop_welfare_report = column(baseline_nash_coop_select,p_eq,table_widget_welfares)
nash_coop_deltas_report = column(p_deltas_eq,table_widget_deltas)

fourth_panel = row(nash_coop_welfare_report, nash_coop_deltas_report)

#%% Kogan paper

# TOOLS="pan,wheel_zoom,box_zoom,reset"

colors_kog = itertools.cycle(Category10[10])

df_kog = pd.read_csv(data_path+'koga_updated.csv')
ds_kog = ColumnDataSource(df_kog)

p_kog = figure(title="Kogan moment updated / extrapolated", 
                width = 1200,
                height = 850,
                x_axis_label='Issue Date',
                y_axis_type="log",
                # y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
                tools = TOOLS) 

l_kog = {}

for i,col in enumerate(df_kog.columns):
    if col not in ['issue_date']:
        l_kog[i] = p_kog.line(x='issue_date', y=col, 
                  source = ds_kog, 
                  line_width = 2, legend_label=col, color=next(colors_kog),
                  name = col)

hover_tool_kog = HoverTool(
    # line_policy='nearest',
    tooltips = [
        ("Issue date", "$x"),
        ('ValuePerPatent', '@ValuePerPatent'),
        ('CostPerPatent', '@CostPerPatent'),
        ('KM_article', '@KM_article'),
        ('ValuePerPatentUpdated', '@ValuePerPatentUpdated'),
        ('CostPerPatentExtrapolated', '@CostPerPatentExtrapolated'),
        ('KM_extrapolatedCost', '@KM_extrapolatedCost')
        ],
    mode='vline',
    renderers = [l_kog[4]]
)
p_kog.add_tools(hover_tool_kog)
# hover_tool_kog.renderers.append(l_kog[0])

p_kog.legend.click_policy="hide"
p_kog.legend.label_text_font_size = '8pt'
p_kog.add_layout(p_kog.legend[0], 'right')


#
colors_kog2 = itertools.cycle(Category10[10])

df_kog2 = pd.read_csv(data_path+'KM_prior.csv')
ds_kog2 = ColumnDataSource(df_kog2)

p_kog2 = figure(title="Kogan moment", 
                width = 1200,
                height = 850,
                x_axis_label='Market Prior',
                # y_axis_type="log",
                # y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
                tools = TOOLS) 

l_kog2 = {}

for i,col in enumerate(df_kog2.columns):
    if col not in ['market prior']:
        l_kog2[i] = p_kog2.line(x='market prior', y=col, 
                  source = ds_kog2, 
                  line_width = 2, legend_label=col, color=next(colors_kog2))

hover_tool_kog2 = HoverTool(
    # line_policy='nearest',
    tooltips = [
        ("market prior", "$x"),
        ('1950 to 2007', '@from1950to2007'),
        ('1980 to 2007', '@from1980to2007'),
        ('1995 to 2007', '@from1995to2007'),
        ('2002 to 2007', '@from2002to2007'),
        ('1950 to 2020', '@from1950to2020'),
        ('1980 to 2020', '@from1980to2020'),
        ('1995 to 2020', '@from1995to2020'),
        ('2002 to 2020', '@from2002to2020'),
        ],
    mode='vline',
    renderers = [l_kog2[4]]
)

p_kog2.legend.click_policy="hide"
p_kog2.legend.label_text_font_size = '8pt'
p_kog2.add_layout(p_kog2.legend[0], 'right')
p_kog2.add_tools(hover_tool_kog2)

fifth_panel = row(p_kog,p_kog2)

curdoc().add_root(column(first_panel, second_panel, third_panel, fourth_panel, fifth_panel))
# curdoc().add_root(column(first_panel, second_panel))
