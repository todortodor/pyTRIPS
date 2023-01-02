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
from bokeh.models import DataRange1d, LinearAxis, ColumnDataSource, LabelSet, Select,Legend, LegendItem, DataTable, TableColumn, HoverTool, Slope
# from bokeh.models.formatters import NumeralTickFormatter
# from bokeh.models.widgets.tables import NumberFormatter
# from bokeh.palettes import Blues4
from bokeh.plotting import figure
from classes import parameters, moments, var

import numpy as np
# from bokeh.models import LogScale, LinearScale
import itertools
from bokeh.palettes import Category10
# import numpy as np


def load(path, data_path=None):
    p = parameters(n=7,s=2,data_path=data_path)
    p.load_data(path)
    sol = var.var_from_vector(p.guess, p, compute=True)
    sol.compute_non_solver_aggregate_qualities(p)
    sol.compute_non_solver_quantities(p)
    sol.scale_P(p)
    sol.compute_price_indices(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    m.load_data(data_path)
    m.load_run(path)
    m.compute_moments(sol, p)
    m.compute_moments_deviations()
    return p,m,sol

def init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,list_of_moments):
    dic_df_param = {}
    dic_df_mom = {}
    params = p_baseline.calib_parameters
    params.append('d*fe')
    params.append('nu/deltaUS')
    df_scalar_params = pd.DataFrame(columns = ['baseline'])
    df_scalar_params.index.name='x'
    
    for param in params:
        # print(param)
        # print(getattr(p_baseline,param)[p_baseline.mask[param]].squeeze().shape == (14,))
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
    dic_df_mom['scalars'] = df_scalar_moments
    # dic_df_mom['scalar deviations'] = df_scalar_moments_deviation
    return dic_df_param, dic_df_mom

def append_dic_of_dataframes_with_variation(dic_df_param, dic_df_mom, p, m, run_name):
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
        if k == 'T non patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,0]
        if k == 'T patent sector':
            dic_df_param[k][run_name] = getattr(p,'T')[...,1]
        
    for k in dic_df_mom.keys():
        if k == 'scalars':
            for i in dic_df_mom[k].index:
                dic_df_mom[k].loc[i,run_name] = float(getattr(m,i))
        if k == 'scalar deviations':
            for i in dic_df_mom[k].index:
                dic_df_mom[k].loc[i,run_name] = float(getattr(m,i+'_deviation'))/m.weights_dict[i]
        if k not in ['scalars','scalar deviations']:
            dic_df_mom[k][run_name] = getattr(m,k).ravel()
    return dic_df_param, dic_df_mom

#%% path

data_path = join(dirname(__file__), 'data/')
results_path = join(dirname(__file__), 'calibration_results_matched_economy/')
cf_path = join(dirname(__file__), 'counterfactual_recaps/unilateral_patent_protection/')
nash_eq_path = join(dirname(__file__), 'nash_eq_recaps/')
coop_eq_path = join(dirname(__file__), 'coop_eq_recaps/')
# print(data_path)

#%% moments / parameters for variations

list_of_moments = ['GPDIFF','GROWTH','KM','KM_GDP', 'OUT',
 'RD', 'RD_US', 'RD_RUS', 'RP', 'SPFLOWDOM', 'SPFLOW',
 'SPFLOW_US', 'SPFLOW_RUS', 'SRDUS', 'SRGDP', 'SRGDP_US',
 'SRGDP_RUS', 'JUPCOST','JUPCOSTRD', 'SINNOVPATUS', 'TO',
 'DOMPATUS','DOMPATEU']

comments_dic = {'baseline':'baseline',
                '1':'1: drop South\nin RD targeting',
                '2.1':'2.1: added domestic US to patent flow moment',
                '2.2':'2.2: added domestic EU to patent flow moment',
                '2.3':'2.3: added domestic US and EU to patent flow moment',
                '3.1':'3.1: added DOMPATUS',
                '3.2':'3.2: added DOMPATEU',
                '3.3':'3.3: added DOMPATUS and DOMPATUS',
                '4.1':'4.1: 2.1 and drop South in RD',
                '4.2':'4.2: 2.2 and drop South in RD',
                '4.3':'4.3: 2.3 and drop South in RD',
                '5':'5: patent cost relative to RD_US (JUPCOSTRD)',
                '6':'6: fix delta_US = 0.05 and drop JUPCOST',
                '7':'7: drop SRDUS',
                '8.1':'8.1: drop South RD, DOMPAT moments, weight1 SPFLOW',
                '8.2':'8.2: drop South RD, DOMPAT moments, weight3 SPFLOW',
                '9.1':'9.1: drop KM moment, TO target divided by 2',
                '10.1':'10.1: SPFLOWDOM instead',
                '10.2':'10.2: SPFLOWDOM and drop South in RD',
                '11.1':'11.1: baseline with new parameter d',
                '11.2':'11.2: 10.1 with new parameter d',
                '11.3':'11.3: 1 with new parameter d',
                '11.4':'11.4: 10.2 with new parameter d',
                '11.5':'11.5: 2.3 with new param d',
                '11.6':'11.6: 4.3 with new param d',
                '11.7':'11.7: 8.1 with new param d',
                '12':'12: replace KM moment by KM_GDP',
                '12.1':'12.1: 11.7 but replace KM moment by KM_GDP',
                '12.2':'12.2: 11.7 but drop KM moment',
                '13.1':'13.1: 11.7 but preventing "bad" Nash',
                '14.1':'14.1: 11.7 with kappa=0.75'
                }

baselines_dic_param = {}
baselines_dic_mom = {}

for baseline_nbr in ['101','102','104']:
    baseline_path = results_path+baseline_nbr+'/'
    baseline_variations_path = results_path+'baseline_'+baseline_nbr+'_variations/'
        
    p_baseline,m_baseline,sol_baseline = load(baseline_path,data_path = data_path)
    baselines_dic_param[baseline_nbr], baselines_dic_mom[baseline_nbr] = init_dic_of_dataframes_with_baseline(p_baseline,m_baseline,list_of_moments)
    
    files_in_dir = next(os.walk(baseline_variations_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for run in run_list:
        p_to_add,m_to_add,sol_to_add = load(baseline_variations_path+run+'/',data_path = data_path)
        a, b  = append_dic_of_dataframes_with_variation(baselines_dic_param[baseline_nbr], 
                                                        baselines_dic_mom[baseline_nbr], p_to_add, m_to_add, run)
        baselines_dic_param[baseline_nbr] = a
        baselines_dic_mom[baseline_nbr] = b


TOOLS="pan,wheel_zoom,box_zoom,reset"

baseline_mom = '101'
mom = 'SPFLOW'

baseline_mom_select = Select(value=baseline_mom, title='Baseline', options=sorted(baselines_dic_mom.keys()))
mom_select = Select(value=mom, title='Quantity', options=sorted(baselines_dic_mom[baseline_mom].keys()))

ds_mom = ColumnDataSource(baselines_dic_mom[baseline_mom][mom])
p_mom = figure(title="Moment matching", 
               width = 1200,
               height = 900,
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
# p_mom.sizing_mode = 'scale_width'
slope = Slope(gradient=1, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=1)
p_mom.add_layout(slope)

# colors_mom = itertools.cycle(Category20.values()(len(baselines_dic_mom[baseline_mom][mom].columns)))
colors_mom = itertools.cycle(Category10[10])

# for col in baselines_dic_mom[baseline_mom][mom].columns[1:]:
lines_mom = {}
for col in ds_mom.data.keys():
    if col not in ['x','target']:
        # lines_mom[col] = p_mom.circle('target', col, 
        #                               source = ds_mom, 
        #                               size=5, color=next(colors_mom), 
        #                               legend_label=comments_dic[col])
        lines_mom[col] = p_mom.circle('target', col, 
                                      source = ds_mom, 
                                      size=5, color=next(colors_mom))
        if col != 'baseline':
            lines_mom[col].visible = False
            
legend_items_mom = [LegendItem(label=comments_dic[col], renderers=[lin_mom]) for col, lin_mom in lines_mom.items()]
legend_mom = Legend(items=legend_items_mom, click_policy="hide", label_text_font_size="8pt",spacing = 0)
p_mom.add_layout(legend_mom, 'right')
# p_mom.line('target','target',source = ds_mom , color = 'black', line_alpha = 0.1)
# p_mom.ray(x=1e-15, y=1e-15, length=0, angle_units = "deg",
#       angle = 45)

# p_mom.legend.click_policy="hide"
# p_mom.legend.label_text_font_size = '8pt'
# # p_mom.legend.label_height = 0
# # p_mom.legend.glyph_height = 0
# p_mom.legend.spacing = 0
# # p_mom.legend.
# p_mom.add_layout(p_mom.legend[0], 'right')

# columns_mom = [
#         TableColumn(field="x"),
#     ]+[TableColumn(field=col) for col in baselines_dic_mom[baseline_mom][mom].columns]
columns_mom = [TableColumn(field=col) for col in list(ds_mom.data.keys())]
data_table_mom = DataTable(source=ds_mom, columns = columns_mom, width=1200, height=400)
# data_table_mom = DataTable(source=ds_mom, width=900, height=400)
# data_table_mom = DataTable(source=ds_mom.data, width=900, height=400)

def update_baseline_mom(attrname, old, new):
    mom = mom_select.value
    ds_mom.data = baselines_dic_mom[new][mom]
    legend_items_mom = [LegendItem(label=comments_dic[col], renderers=[lines_mom[col]]) for col in ds_mom.data if col not in ['x','target']]
    # legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8px",spacing = 0)
    p_mom.legend.items = legend_items_mom
    
def update_mom(attrname, old, new):
    # p_mom.legend.items = []
    baseline_mom = baseline_mom_select.value
    ds_mom.data = baselines_dic_mom[baseline_mom][new]

controls_mom = row(baseline_mom_select, mom_select)
# controls_mom.sizing_mode = 'scale_width'

baseline_mom_select.on_change('value', update_baseline_mom)
mom_select.on_change('value', update_mom)

# curdoc().add_root(row(p_par, controls))
   

baseline_par = '101'
par = 'delta'

baseline_par_select = Select(value=baseline_par, title='Baseline', options=sorted(baselines_dic_param.keys()))
par_select = Select(value=par, title='Quantity', options=sorted(baselines_dic_param[baseline_par].keys()))
x_range = baselines_dic_param[baseline_par][par_select.value].index.to_list()
ds_par = ColumnDataSource(baselines_dic_param[baseline_par][par])
p_par = figure(title="Parameters", 
               width = 1200,
               height = 900,
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

legend_items_par = [LegendItem(label=comments_dic[col], renderers=[lin_par]) for col, lin_par in lines_par.items()]
legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8pt",spacing = 0)
p_par.add_layout(legend_par, 'right')
# p_par.legend.click_policy="hide"
# p_par.legend.label_text_font_size = '8pt'
# p_par.legend.spacing = 0
# p_par.add_layout(p_par.legend[0], 'right')



columns_par = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in baselines_dic_param[baseline_par][par].columns]

data_table_par = DataTable(source=ds_par, columns = columns_par, width=1200, height=400)

def update_baseline_par(attrname, old, new):
    par = par_select.value
    ds_par.data = baselines_dic_param[new][par]
    legend_items_par = [LegendItem(label=comments_dic[col], renderers=[lines_par[col]]) for col in ds_par.data if col not in ['x']]
    # legend_par = Legend(items=legend_items_par, click_policy="hide", label_text_font_size="8px",spacing = 0)
    p_par.legend.items = legend_items_par
    
def update_par(attrname, old, new):
    baseline_par = baseline_par_select.value
    p_par.x_range.factors = baselines_dic_param[baseline_par][new].index.to_list()
    ds_par.data = baselines_dic_param[baseline_par][new]

controls_par = row(baseline_par_select, par_select)
# controls_par.sizing_mode = 'scale_width'

baseline_par_select.on_change('value', update_baseline_par)
par_select.on_change('value', update_par)
# p_par.add_layout(p_par.legend[0], 'bottom right')

# moment_report = column(controls_mom,p_mom)
moment_report = column(controls_mom,p_mom,data_table_mom)
# param_report = column(controls_par, p_par)
param_report = column(controls_par, p_par, data_table_par)
first_panel = row(moment_report,param_report)

# curdoc().add_root(row(column(controls_mom,p_mom,data_table_mom),column(controls_par, p_par, data_table_par)))

#%% sensitivities

baselines_dic_sensi = {}

for baseline_nbr in ['101','102','104']:
    baselines_dic_sensi[baseline_nbr] = {}
    baseline_sensi_path = results_path+'baseline_'+baseline_nbr+'_sensitivity_tables/'
    files_in_dir = os.listdir(baseline_sensi_path)
    files_in_dir = [ filename for filename in files_in_dir if filename.endswith('.csv') ]
    for f in files_in_dir:
        baselines_dic_sensi[baseline_nbr][f[:-4]] = pd.read_csv(baseline_sensi_path+f,index_col = 0)
    
baseline_sensi = '101'
qty_sensi = 'delta US over nu'

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

#%% counterfactuals

baseline_cf = '101'
country_cf = 'USA'

p_baseline,m_baseline,sol_baseline = load(results_path+baseline_cf+'/',data_path = data_path)

baseline_cf_select = Select(value=baseline_cf, title='Baseline', options=[s[9:] for s in sorted(os.listdir(cf_path))])
country_cf_select = Select(value=country_cf, title='Country', options=p_baseline.countries+['World','Harmonizing'])

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
    df_max = df_max.loc[p_baseline.countries]
    df_max['colors'] = Category10[10][:len(df_max)]
    return df_max

df_cf = get_data_cf(baseline_cf,country_cf)
ds_cf = ColumnDataSource(df_cf)
df_cf_max = build_max(df_cf)
ds_cf_max = ColumnDataSource(df_cf_max)

colors_cf = itertools.cycle(Category10[10])
colors_cf_max = itertools.cycle(Category10[10])

p_cf = figure(title="Unilateral patent protection counterfactual", 
               width = 1200,
               height = 850,
               x_axis_label='Change in delta',
               y_axis_label='Normalized Consumption equivalent welfare / Growth rate',
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

second_panel = row(sensitivity_report,counterfactuals_report)

#%% Nash / coop equilibrium

# nash_eq_path = 'nash_eq_recaps/'
# coop_eq_path = 'coop_eq_recaps/'

welf_coop = pd.read_csv(coop_eq_path+'cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                           'variation'],keep='last').sort_values(['baseline','variation'])
welf_nash = pd.read_csv(nash_eq_path+'cons_eq_welfares.csv',index_col=0).drop_duplicates(['baseline', 
                           'variation'],keep='last').sort_values(['baseline','variation'])

welf_coop['pop w av'] = ((welf_coop[p_baseline.countries].T.values*p_baseline.data.labor.values[:,None]).sum(axis=0)/p_baseline.data.labor.values.sum())
welf_nash['pop w av'] = ((welf_nash[p_baseline.countries].T.values*p_baseline.data.labor.values[:,None]).sum(axis=0)/p_baseline.data.labor.values.sum())

welf_coop['run'] = welf_coop['baseline'].astype('str')+', '+welf_coop['variation']
welf_nash['run'] = welf_nash['baseline'].astype('str')+', '+welf_nash['variation']

welf_coop['sorting'] = welf_coop['variation'].str.replace('baseline','0').astype(float)
welf_nash['sorting'] = welf_nash['variation'].str.replace('baseline','0').astype(float)

welf_coop = welf_coop.sort_values(['baseline','sorting'])
welf_nash = welf_nash.sort_values(['baseline','sorting'])

ds_coop = ColumnDataSource(welf_coop)
ds_nash = ColumnDataSource(welf_nash)

colors_coop = itertools.cycle(Category10[10])
colors_nash = itertools.cycle(Category10[10])

x_range = welf_nash['run'].to_list()

p_eq = figure(title="Cooperative and Nash equilibrium", 
               width = 1400,
               height = 850,
               x_range = x_range,
               # x_axis_label='Run',
               y_axis_label='Consumption eqivalent welfare change',
                tools = TOOLS
               ) 
p_eq.xaxis.major_label_orientation = 3.14/3

for col in p_baseline.countries+['pop w av']:
    p_eq.line(x='run', y=col, source = ds_nash, color=next(colors_nash),line_width = 2, legend_label=col+' Nash')
    p_eq.line(x='run', y=col, source = ds_coop, color=next(colors_coop), line_dash='dashed', line_width = 2, legend_label=col+' coop')
    
p_eq.legend.click_policy="hide"
p_eq.legend.label_text_font_size = '8pt'
p_eq.add_layout(p_eq.legend[0], 'right')    

hover_tool_eq = HoverTool()
hover_tool_eq.tooltips = [
    ("run", "@run"),
    ("value", "$y")
    ] 
p_eq.add_tools(hover_tool_eq)

data_table_eq = dict(
        runs=[run for run in comments_dic.keys()],
        comments=[comment for comment in comments_dic.values()],
    )
source_table_eq = ColumnDataSource(data_table_eq)

columns = [
        TableColumn(field="runs", title="Runs"),
        TableColumn(field="comments", title="Description"),
    ]
data_table_eq = DataTable(source=source_table_eq, columns=columns, width=400, height=850,
                           # autosize_mode="force_fit"
                          )

data_table_welfares = pd.concat([welf_nash.set_index('run'),
             welf_coop.set_index('run')],
            axis=0,
            keys=['Nash','Coop'],
            names=['type','run'],
            sort=False
            ).reset_index().sort_values(['baseline','sorting','type'])[['run','type']+p_baseline.countries+['pop w av']]

source_table_welfares = ColumnDataSource(data_table_welfares)
columns_welf = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries+['pop w av']]

table_widget_welfares = DataTable(source=source_table_welfares, columns=columns_welf, width=850, height=400,
                           # autosize_mode="force_fit"
                          )

# columns_par = [
#         TableColumn(field="x"),
#     ]+[TableColumn(field=col) for col in baselines_dic_param[baseline_par][par].columns]

data_table_par = DataTable(source=ds_par, columns = columns_par, width=1200, height=400)

# p_eq.show()

deltas_coop = pd.read_csv(coop_eq_path+'deltas.csv',index_col=0).drop_duplicates(['baseline', 
                           'variation'],keep='last').sort_values(['baseline','variation'])
deltas_nash = pd.read_csv(nash_eq_path+'deltas.csv',index_col=0).drop_duplicates(['baseline', 
                           'variation'],keep='last').sort_values(['baseline','variation'])

deltas_coop['run'] = deltas_coop['baseline'].astype('str')+', '+deltas_coop['variation']
deltas_nash['run'] = deltas_nash['baseline'].astype('str')+', '+deltas_nash['variation']

deltas_coop['sorting'] = deltas_coop['variation'].str.replace('baseline','0').astype(float)
deltas_nash['sorting'] = deltas_nash['variation'].str.replace('baseline','0').astype(float)

deltas_coop = deltas_coop.sort_values(['baseline','sorting'])
deltas_nash = deltas_nash.sort_values(['baseline','sorting'])

ds_deltas_coop = ColumnDataSource(deltas_coop)
ds_deltas_nash = ColumnDataSource(deltas_nash)

colors_deltas_coop = itertools.cycle(Category10[10])
colors_deltas_nash = itertools.cycle(Category10[10])

x_range = deltas_nash['run'].to_list()

p_deltas_eq = figure(title="Cooperative and Nash equilibrium", 
               width = 1400,
               height = 850,
               x_range = x_range,
               y_axis_type="log",
               # x_axis_label='Run',
               y_axis_label='Delta',
                tools = TOOLS
               ) 
p_deltas_eq.xaxis.major_label_orientation = 3.14/3

for col in p_baseline.countries:
    p_deltas_eq.line(x='run', y=col, source = ds_deltas_nash, color=next(colors_deltas_nash),line_width = 2, legend_label=col+' Nash')
    p_deltas_eq.line(x='run', y=col, source = ds_deltas_coop, color=next(colors_deltas_coop), line_dash='dashed', line_width = 2, legend_label=col+' coop')
    
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
             deltas_coop.set_index('run')],
            axis=0,
            keys=['Nash','Coop'],
            names=['type','run'],
            sort=False
            ).reset_index().sort_values(['baseline','sorting','type'])[['run','type']+p_baseline.countries]

source_table_deltas = ColumnDataSource(data_table_deltas)
columns_deltas = [TableColumn(field=col) for col in ['run','type']+p_baseline.countries]

table_widget_deltas = DataTable(source=source_table_deltas, columns=columns_deltas, width=850, height=400,
                           # autosize_mode="force_fit"
                          )

second_panel_bis = column(row(p_eq,data_table_eq),table_widget_welfares,row(p_deltas_eq,data_table_eq),table_widget_deltas)

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

third_panel = row(p_kog,p_kog2)

curdoc().add_root(column(first_panel, second_panel, second_panel_bis, third_panel))