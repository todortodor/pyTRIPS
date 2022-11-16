from os.path import dirname, join
import os
import sys
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
from bokeh.models import ColumnDataSource, LabelSet, Select, DataTable, TableColumn, HoverTool, Slope
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.widgets.tables import NumberFormatter
# from bokeh.palettes import Blues4
from bokeh.plotting import figure, show
from classes import parameters, moments, var

import numpy as np
from bokeh.models import LogScale, LinearScale
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
    df_scalar_params = pd.DataFrame(columns = ['baseline'])
    df_scalar_params.index.name='x'
    
    for param in params:
        # print(param)
        # print(getattr(p_baseline,param)[p_baseline.mask[param]].squeeze().shape == (14,))
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
                dic_df_param[k].loc[i,run_name] = float(getattr(p,i)[p.mask[i]])
                if i == 'k':
                    dic_df_param[k].loc[i,run_name] = float(getattr(p,i)[p.mask[i]])-1
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

list_of_moments = ['GPDIFF','GROWTH','KM', 'OUT',
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
                '4.1':'4.1: 21 and drop South in RD',
                '4.2':'4.2: 22 and drop South in RD',
                '4.3':'4.3: 23 and drop South in RD',
                '5':'5: patent cost relative to RD_US (JUPCOSTRD)',
                '6':'6: fix delta_US = 0.05 and drop JUPCOST',
                '7':'7: drop SRDUS'
                }


baselines_dic_param = {}
baselines_dic_mom = {}
path_tfs = dirname(__file__)+'/'
data_path = join(dirname(__file__), 'data/')
results_path = join(dirname(__file__), 'calibration_results_matched_economy/')
print(data_path)
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

#%%
TOOLS="pan,wheel_zoom,box_zoom,reset"

baseline_mom = '101'
mom = 'SPFLOW'

baseline_mom_select = Select(value=baseline_mom, title='Baseline', options=sorted(baselines_dic_mom.keys()))
mom_select = Select(value=mom, title='Quantity', options=sorted(baselines_dic_mom[baseline_mom].keys()))

ds_mom = ColumnDataSource(baselines_dic_mom[baseline_mom][mom])
p_mom = figure(title="Moment matching", 
               width = 900,
               height = 600,
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
              x_offset=2, y_offset=2, source=ds_mom, text_font_size="6pt")
p_mom.add_layout(labels)
p_mom.add_tools(hover_tool_mom)
# p_mom.sizing_mode = 'scale_width'
slope = Slope(gradient=1, y_intercept=0,
              line_color='black', line_dash='dashed', line_width=1)
p_mom.add_layout(slope)

# colors_mom = itertools.cycle(Category20.values()(len(baselines_dic_mom[baseline_mom][mom].columns)))
colors_mom = itertools.cycle(Category10[10])

for col in baselines_dic_mom[baseline_mom][mom].columns[1:]:
    p_mom.circle('target', col, source = ds_mom , size=5, color=next(colors_mom), legend_label=comments_dic[col])
# p_mom.line('target','target',source = ds_mom , color = 'black', line_alpha = 0.1)
# p_mom.ray(x=1e-15, y=1e-15, length=0, angle_units = "deg",
#       angle = 45)

p_mom.legend.click_policy="hide"
p_mom.legend.label_text_font_size = '8pt'
p_mom.add_layout(p_mom.legend[0], 'right')

columns_mom = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in baselines_dic_mom[baseline_mom][mom].columns]

data_table_mom = DataTable(source=ds_mom, columns = columns_mom, width=900, height=400)

def update_baseline_mom(attrname, old, new):
    mom = mom_select.value
    ds_mom .data = baselines_dic_mom[new][mom]
    
def update_mom(attrname, old, new):
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
               width = 900,
               height = 600,
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

for col in baselines_dic_param[baseline_par][par].columns:
    p_par.line(x='x', y=col, source = ds_par, color=next(colors_par),line_width = 2, legend_label=comments_dic[col])
    
p_par.legend.click_policy="hide"
p_par.legend.label_text_font_size = '8pt'
p_par.add_layout(p_par.legend[0], 'right')

columns_par = [
        TableColumn(field="x"),
    ]+[TableColumn(field=col) for col in baselines_dic_param[baseline_par][par].columns]

data_table_par = DataTable(source=ds_par, columns = columns_par, width=900, height=400)

def update_baseline_par(attrname, old, new):
    par = par_select.value
    ds_par.data = baselines_dic_param[new][par]
    
def update_par(attrname, old, new):
    baseline_par = baseline_par_select.value
    p_par.x_range.factors = baselines_dic_param[baseline_par][new].index.to_list()
    ds_par.data = baselines_dic_param[baseline_par][new]

controls_par = row(baseline_par_select, par_select)
# controls_par.sizing_mode = 'scale_width'

baseline_par_select.on_change('value', update_baseline_par)
par_select.on_change('value', update_par)
# p_par.add_layout(p_par.legend[0], 'bottom right')

curdoc().add_root(row(column(controls_mom,p_mom,data_table_mom),column(controls_par, p_par, data_table_par)))


#%%

# STATISTICS = ['record_min_temp', 'actual_min_temp', 'average_min_temp', 'average_max_temp', 'actual_max_temp', 'record_max_temp']

# def get_dataset(src, name, distribution):
#     df = src[src.airport == name].copy()
#     del df['airport']
#     df['date'] = pd.to_datetime(df.date)
#     # timedelta here instead of pd.DateOffset to avoid pandas bug < 0.18 (Pandas issue #11925)
#     df['left'] = df.date - datetime.timedelta(days=0.5)
#     df['right'] = df.date + datetime.timedelta(days=0.5)
#     df = df.set_index(['date'])
#     df.sort_index(inplace=True)
#     if distribution == 'Smoothed':
#         window, order = 51, 3
#         for key in STATISTICS:
#             df[key] = savgol_filter(df[key], window, order)

#     return ColumnDataSource(data=df)

# def make_plot(source, title):
#     plot = figure(x_axis_type="datetime", width=800, tools="", toolbar_location=None)
#     plot.title.text = title

#     plot.quad(top='record_max_temp', bottom='record_min_temp', left='left', right='right',
#               color=Blues4[2], source=source)
#     plot.quad(top='average_max_temp', bottom='average_min_temp', left='left', right='right',
#               color=Blues4[1], source=source)
#     plot.quad(top='actual_max_temp', bottom='actual_min_temp', left='left', right='right',
#               color=Blues4[0], alpha=0.5, line_color="black", source=source)

#     # fixed attributes
#     plot.xaxis.axis_label = None
#     plot.yaxis.axis_label = "Temperature (F)"
#     plot.axis.axis_label_text_font_style = "bold"
#     plot.x_range = DataRange1d(range_padding=0.0)
#     plot.grid.grid_line_alpha = 0.3

#     return plot

# def update_plot(attrname, old, new):
#     city = city_select.value
#     plot.title.text = "TEST Weather data for " + cities[city]['title']

#     src = get_dataset(df, cities[city]['airport'], distribution_select.value)
#     source.data.update(src.data)

# city = 'Austin'
# distribution = 'Discrete'

# cities = {
#     'Austin': {
#         'airport': 'AUS',
#         'title': 'Austin, TX',
#     },
#     'Boston': {
#         'airport': 'BOS',
#         'title': 'Boston, MA',
#     },
#     'Seattle': {
#         'airport': 'SEA',
#         'title': 'Seattle, WA',
#     }
# }

# city_select = Select(value=city, title='City', options=sorted(cities.keys()))
# distribution_select = Select(value=distribution, title='Distribution', options=['Discrete', 'Smoothed'])

# df = pd.read_csv(join(dirname(__file__), 'bokeh-app/data/2015_weather.csv'))
# source = get_dataset(df, cities[city]['airport'], distribution)
# plot = make_plot(source, "Weather data for " + cities[city]['title'])

# city_select.on_change('value', update_plot)
# distribution_select.on_change('value', update_plot)

# controls = column(city_select, distribution_select)

# curdoc().add_root(row(plot, controls))
# curdoc().title = "Weather"

# show(plot)