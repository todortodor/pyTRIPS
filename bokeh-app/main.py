from os.path import dirname, join
import os
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Button, Slider, Toggle, FactorRange, Div, ColumnDataSource, LabelSet, Select,Legend, LegendItem, DataTable, TableColumn, HoverTool, Slope
from bokeh.plotting import figure
from bokeh.events import ButtonClick
from classes import parameters, moments, var, var_with_entry_costs
from data_funcs import compute_rough_jacobian,rough_dyn_fixed_point_solver
import numpy as np
import itertools
from bokeh.palettes import Category10, Dark2
Category18 = Category10[10]+Dark2[8]
import time
import warnings

warnings.filterwarnings('ignore')

start = time.perf_counter()

def load(path, data_path=None, 
         dir_path = None, context = 'calibration'):
    p = parameters()
    p.load_run(path,dir_path=dir_path)
    sol = var.var_from_vector(p.guess, p, compute=True, context = context)
    sol.scale_P(p)
    sol.compute_price_indices(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    # m.load_data(data_path)
    m.load_run(path,dir_path=dir_path)
    m.aggregate_moments = True
    m.compute_moments(sol, p)
    m.compute_moments_deviations()
    return p,m,sol

def init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,sol_baseline,list_of_moments):
    dic_df_param = {}
    dic_df_mom = {}
    dic_df_sol = {}
    params = p_baseline.calib_parameters
    df_scalar_params = pd.DataFrame(columns = ['baseline'])
    df_scalar_params.index.name='x'
    
    for param in params:
        if hasattr(p_baseline,param):
            if len(getattr(p_baseline,param)[p_baseline.mask[param]]) == 1:
                if param == 'k':
                    df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,param)[p_baseline.mask[param]])-1
                else:
                    df_scalar_params.loc[param,'baseline'] = float(getattr(p_baseline,param)[p_baseline.mask[param]])
            if len(getattr(p_baseline,param)[p_baseline.mask[param]]) == 3 or len(getattr(p_baseline,param)[p_baseline.mask[param]]) == 4 or len(getattr(p_baseline,param)[p_baseline.mask[param]]) == 2:
                df = pd.DataFrame(index = p_baseline.sectors[1:], columns = ['baseline'], data = getattr(p_baseline,param)[1:])
                df.index.name='x'
                dic_df_param[param] = df
            # if param in ['eta','delta']:
            #     df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,1])
            #     df.index.name='x'
            #     dic_df_param[param] = df
            if param in ['T','eta','delta']:
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,0])
                df.index.name='x'
                dic_df_param[param+' non patent sector'] = df
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,1])
                df.index.name='x'
                dic_df_param[param+' patent sector'] = df
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,2])
                df.index.name='x'
                dic_df_param[param+' pharma'] = df
                df = pd.DataFrame(index = p_baseline.countries, columns = ['baseline'], data = getattr(p_baseline,param)[...,3])
                df.index.name='x'
                dic_df_param[param+' chemicals'] = df
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
                
        if k in ['r_hjort']:
            dic_df_param[k][run_name] = getattr(p,k)
        if k == 'T non patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,0]
        if k == 'T patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,1]
        if k == 'T pharma':
            dic_df_param[k][run_name] = getattr(p,'T')[...,2]
        if k == 'T chemicals':
            dic_df_param[k][run_name] = getattr(p,'T')[...,3]
        if k == 'eta non patent sector':
            dic_df_param[k][run_name] = getattr(p,'eta')[...,0]
        if k == 'eta patent sector':
            dic_df_param[k][run_name] = getattr(p,'eta')[...,1]
        if k == 'eta pharma':
            dic_df_param[k][run_name] = getattr(p,'eta')[...,2]
        if k == 'eta chemicals':
            dic_df_param[k][run_name] = getattr(p,'eta')[...,3]
        if k == 'delta non patent sector':
            dic_df_param[k][run_name] = getattr(p,'delta')[...,0]
        if k == 'delta patent sector':
            dic_df_param[k][run_name] = getattr(p,'delta')[...,1]
        if k == 'delta pharma':
            dic_df_param[k][run_name] = getattr(p,'delta')[...,2]
        if k == 'delta chemicals':
            dic_df_param[k][run_name] = getattr(p,'delta')[...,3]
        if k in ['fe','fo','nu','theta','zeta','k','sigma']:
            dic_df_param[k][run_name] = getattr(p,k)[1:]
        
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

list_of_moments = ['GPDIFF','GROWTH','KM','KMPHARMA','KMCHEM', 'OUT',
 'RD','RDPHARMA','RDCHEM', 'RP', 'SPFLOWDOM', 'SPFLOW','SDFLOW','STFLOW','STFLOWSDOM',
 'SRGDP','SGDP','RGDPPC','UUPCOSTS','SINNOVPATUS',
  'TO','TE','TOPHARMA','TEPHARMA','TOCHEM','TECHEM','DOMPATINUS','DOMPATUS',
 'TWSPFLOW','TWSPFLOWDOM','SDOMTFLOW',#'UUPCOST',
 'objective']
comments_dic = {}

comments_dic['5001'] = {
    "baseline":"bsline:GPDIFF,SDFLOW,UUPCOSTS,DOMPATINUS",
    "1.0":"1.0:added TEPH/CH",
    "2.0":"2.0:added TEPH/CH+TOPH/CH",
    "2.01":"2.01:2.0 with aggregate KM and SINNOVPATUS",
    "2.02":"2.02:2.01 and added RDPH/CH",
    "3.0":"3.0:added TEPH/CH+TOPH/CH+RDPH/CH",
    "4.0":"4.0:added TEPH/CH+TOPH/CH+KMPH/CH",
    "5.0":"5.0:added TEPH/CH+TOPH/CH+KMPH/CH+RDPH/CH",
    "6.0":"6.0:doubled eta and delta from 5.0",
    "7.0":"7.0:no crazy deltas (not recalibrated) from 5.0",
    "8.0":"8.0:higher bound on delta (same setup as 5.0)",
    "11.0":"11.0:added TOPH/CH",
    "12.0":"12.0:added TOPH/CH+RDPH/CH",
    "13.0":"13.0:added TOPH/CH+KMPH/CH",
    "14.0":"14.0:added TOPH/CH+KMPH/CH+RDPH/CH",
    "20.0":"20.0:removed GPDIFF",
    "21.0":"21.0:fixed fe acros sects,UUPCOST not sect-spec",
    }

comments_dic['5002'] = {
    "baseline":"bsline:same as 5001 variation 2.0",
    "1.0":"1.0:added RDPH/CH, weight 0",
    "2.0":"2.0:added RDPH/CH, weight 0.05",
    "3.0":"3.0:added RDPH/CH, weight 0.1",
    "4.0":"4.0:added RDPH/CH, weight 0.2",
    "5.0":"5.0:added RDPH/CH, weight 0.4",
    "6.0":"6.0:added RDPH/CH, weight 0.6",
    "7.0":"7.0:added RDPH/CH, weight 0.8",
    "8.0":"8.0:added RDPH/CH, weight 1",
    # "9.0":"9.0:added RDPH/CH, weight 1.5",
    # "10.0":"10.0:added RDPH/CH, weight 2",
    "11.0":"11.0:added RDPH/CH, weight 2.5",
    "12.0":"12.0:added RDPH/CH, weight 3",
    "13.0":"13.0:added RDPH/CH, weight 3.5",
    "14.0":"14.0:added RDPH/CH, weight 4",
    "15.0":"15.0:added RDPH/CH, weight 4.5",
    "16.0":"16.0:added RDPH/CH, weight 5",
    "17.0":"17.0:added RDPH/CH, weight 5.5",
    "18.0":"18.0:added RDPH/CH, weight 7.5",
    "19.0":"19.0:added RDPH/CH, weight 10",
    # "19.0":"19.0:added RDPH/CH, weight 6.5",
    # "20.0":"20.0:added RDPH/CH, weight 7",
    # "21.0":"21.0:added RDPH/CH, weight 7.5",
    # "22.0":"22.0:added RDPH/CH, weight 8",
    # "23.0":"23.0:added RDPH/CH, weight 8.5",
    # "24.0":"24.0:added RDPH/CH, weight 9",
    # "25.0":"25.0:added RDPH/CH, weight 9.5",
    # "26.0":"26.0:added RDPH/CH, weight 10",
    }

comments_dic['5003'] = {
    "baseline":"bsline:same as 5001 variation 2.01",
    "1.0":"1.0:calibrated k sector-specific",
    "2.0":"2.0:calibrated k and sigma sector-specific",
    "3.0":"3.0:calibrated k and sigma in new sectors only",
    "4.0":"4.0:3.0 adding KMPH/CH",
    "5.0":"5.0:3.0 adding RDPH/CH",
    "6.0":"6.0:3.0 adding RDPH/CH and KMPH/CH",
    "5.01":"5.01:5.0 higher weight RDPH/CH",
    "6.01":"6.01:6.0 higher weight RDPH/CH",
    "99.0":"99.0:increasing beta_pharma",
    }

baselines_dic_param = {}
baselines_dic_mom = {}
baselines_dic_sol_qty = {}

baseline_list = ['5003','5001','5002']    
baseline_mom = baseline_list[0]

def section(s):
     return [int(_) for _ in s.split(".")]
 
for baseline_nbr in baseline_list:
    print(baseline_nbr)
    print(time.perf_counter() - start)
    baseline_path = results_path+baseline_nbr+'/'
    baseline_variations_path = results_path+'baseline_'+baseline_nbr+'_variations/'
    p_baseline,m_baseline,sol_baseline = load(baseline_path,data_path = data_path,
                                              dir_path=dir_path)
    if 'sigma' not in p_baseline.calib_parameters:
        p_baseline.calib_parameters.append('sigma')
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
#%%
countries = p_baseline.countries

TOOLS="pan,wheel_zoom,box_zoom,reset,save"

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

for slope in [slope1]:
    p_mom.add_layout(slope)
    
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
legend_mom = Legend(items=legend_items_mom, click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0)
p_mom.add_layout(legend_mom, 'right')

# legend_mom_split_1 = Legend(items=legend_items_mom[:round((len(legend_items_mom)+1)/2)], click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0, 
#                     # location=(10, -60)
#                     )
# legend_mom_split_2 = Legend(items=legend_items_mom[round((len(legend_items_mom)+1)/2):], click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0
#                     # , location=(10, -60)
#                     )
# p_mom.add_layout(legend_mom_split_1, 'right')
# p_mom.add_layout(legend_mom_split_2, 'right')

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
    legend_mom.items = legend_items_mom
    # legend_mom_split_1.items = legend_items_mom[:round((len(legend_items_mom)+1)/2)]
    # legend_mom_split_2.items = legend_items_mom[round((1+len(legend_items_mom))/2):]
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
par = 'delta patent sector'

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
               y_axis_type="log",
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
legend_par = Legend(items=legend_items_par, click_policy="hide", 
                    label_text_font_size="8pt",
                    spacing = 0, 
                    )
p_par.add_layout(legend_par, 'right')

# legend_par_split_1 = Legend(items=legend_items_par[:round((len(legend_items_par)+1)/2)], click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0, 
#                     )
# legend_par_split_2 = Legend(items=legend_items_par[round((1+len(legend_items_par))/2):], click_policy="hide", 
#                     label_text_font_size="8pt",
#                     spacing = 0
#                     )
# p_par.add_layout(legend_par_split_1, 'right')
# p_par.add_layout(legend_par_split_2, 'right')

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
    legend_par.items = legend_items_par
    # legend_par_split_1.items = legend_items_par[:round((1+len(legend_items_par))/2)]
    # legend_par_split_2.items = legend_items_par[round((len(legend_items_par)+1)/2):]
                      
    data_table_par.columns = [
            TableColumn(field="x"),
        ]+[TableColumn(field=col) for col in list(comments_dic[new].keys())]

def update_par(attrname, old, new):
    baseline_par = baseline_par_select.value
    x_range_factors = baselines_dic_param[baseline_par][new].index.to_list()
    if new != 'scalars':
        try:
            x_range_factors = sorted(x_range_factors, key = country_sort.get)
        except:
            pass
    p_par.x_range.factors = x_range_factors
    ds_par.data = baselines_dic_param[baseline_par][new].loc[x_range_factors]

controls_par = row(baseline_par_select, par_select)

baseline_par_select.on_change('value', update_baseline_par)
par_select.on_change('value', update_par)

moment_report = column(controls_mom,p_mom,data_table_mom)
param_report = column(controls_par, p_par, data_table_par)

#!!! first panel
# first_panel = row(moment_report,param_report,sol_qty_report)
first_panel = row(moment_report,param_report)
print(time.perf_counter() - start)



#%% build curdoc
print(time.perf_counter() - start)
curdoc().add_root(column(first_panel, 
                         )
                  )
