#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:59:13 2023

@author: slepot
"""

from classes import moments, parameters,  var
from solver_funcs import fixed_point_solver
import numpy as np
import pandas as pd

baseline_number = '1010'
baseline_pre_trips_variation = '1010'
pre_trips_cf = True
pre_trips_variation = '9.2'
#%%
p_baseline = parameters()
p_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')
m_baseline = moments()
m_baseline.load_run('calibration_results_matched_economy/'+baseline_number+'/')

_, sol_baseline = fixed_point_solver(p_baseline,context = 'counterfactual',
                        x0=p_baseline.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_baseline.scale_P(p_baseline)
sol_baseline.compute_non_solver_quantities(p_baseline) 

delta_nash = pd.read_csv('nash_eq_recaps/deltas.csv',index_col=0)
delta_nash = delta_nash.loc[(delta_nash.baseline == int(baseline_number))
                            & (delta_nash.variation == 'baseline')
                            ].iloc[-1][p_baseline.countries].values

p_nash = p_baseline.copy()
p_nash.delta[...,1] = delta_nash

m_nash = m_baseline.copy()

_, sol_nash = fixed_point_solver(p_nash,context = 'counterfactual',
                        x0=p_nash.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_nash.scale_P(p_nash)
sol_nash.compute_non_solver_quantities(p_nash) 
sol_nash.compute_consumption_equivalent_welfare(p_nash,sol_baseline)
m_nash.compute_moments(sol_nash,p_nash)
m_nash.compute_moments_deviations()
#
delta_negishi = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
delta_negishi = delta_negishi.loc[(delta_negishi.baseline == int(baseline_number))
                            & (delta_negishi.variation == 'baseline')
                            & (delta_negishi.aggregation_method == 'negishi')
                            ].iloc[-1][p_baseline.countries].values

p_negishi = p_baseline.copy()
p_negishi.delta[...,1] = delta_negishi

m_negishi = m_baseline.copy()

_, sol_negishi = fixed_point_solver(p_negishi,context = 'counterfactual',
                        x0=p_negishi.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_negishi.scale_P(p_negishi)
sol_negishi.compute_non_solver_quantities(p_negishi) 
sol_negishi.compute_consumption_equivalent_welfare(p_negishi,sol_baseline)
m_negishi.compute_moments(sol_negishi,p_negishi)
m_negishi.compute_moments_deviations()
#
delta_equal = pd.read_csv('coop_eq_recaps/deltas.csv',index_col=0)
delta_equal = delta_equal.loc[(delta_equal.baseline == int(baseline_number))
                            & (delta_equal.variation == 'baseline')
                            & (delta_equal.aggregation_method == 'pop_weighted')
                            ].iloc[-1][p_baseline.countries].values

p_equal = p_baseline.copy()
p_equal.delta[...,1] = delta_equal

m_equal = m_baseline.copy()

_, sol_equal = fixed_point_solver(p_equal,context = 'counterfactual',
                        x0=p_equal.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_equal.scale_P(p_equal)
sol_equal.compute_non_solver_quantities(p_equal) 
sol_equal.compute_consumption_equivalent_welfare(p_equal,sol_baseline)
m_equal.compute_moments(sol_equal,p_equal)
m_equal.compute_moments_deviations()

p_uniform_delta = p_baseline.copy()
p_uniform_delta.delta[...,1] = np.minimum(p_baseline.delta[0,1],p_uniform_delta.delta[...,1])

m_uniform_delta = m_baseline.copy()

_, sol_uniform_delta = fixed_point_solver(p_uniform_delta,context = 'counterfactual',
                        x0=p_uniform_delta.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_uniform_delta.scale_P(p_uniform_delta)
sol_uniform_delta.compute_non_solver_quantities(p_uniform_delta) 
sol_uniform_delta.compute_consumption_equivalent_welfare(p_uniform_delta,sol_baseline)
m_uniform_delta.compute_moments(sol_uniform_delta,p_uniform_delta)
m_uniform_delta.compute_moments_deviations()

p_pre_trips = p_baseline.copy()
p_temp = p_baseline.copy()
p_temp.load_run(f'calibration_results_matched_economy/baseline_{baseline_pre_trips_variation}_variations/{pre_trips_variation}/')
p_pre_trips.delta[...,1] = p_temp.delta[...,1]

m_pre_trips = m_baseline.copy()

_, sol_pre_trips = fixed_point_solver(p_pre_trips,context = 'counterfactual',
                        x0=p_pre_trips.guess,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=True,
                        cobweb_qty='phi',
                        plot_convergence=False,
                        plot_cobweb=False,
                        safe_convergence=0.001,
                        disp_summary=True,
                        damping = 10,
                        max_count = 3e3,
                        accel_memory = 50, 
                        accel_type1=True, 
                        accel_regularization=1e-10,
                        accel_relaxation=0.5, 
                        accel_safeguard_factor=1, 
                        accel_max_weight_norm=1e6,
                        damping_post_acceleration=5
                        )
sol_pre_trips.scale_P(p_pre_trips)
sol_pre_trips.compute_non_solver_quantities(p_pre_trips) 
sol_pre_trips.compute_consumption_equivalent_welfare(p_pre_trips,sol_baseline)
m_pre_trips.compute_moments(sol_pre_trips,p_pre_trips)
m_pre_trips.compute_moments_deviations()

#%%
nodes = []
edges = []

long = {
        'USA':-97,
        'EUR':9,
        'JAP':145,
        'CHN':110,
        'BRA':-55,
        'IND':77,
        'CAN':-95,
        'KOR':127.5,
        'RUS':100,
        'MEX':-102,
        'ROW':0
        }
lat = {
        'USA':38,
        'EUR':51,
        'JAP':50,
        'CHN':20,
        'BRA':-10,
        'IND':5,
        'CAN':60,
        'KOR':37,
        'RUS':60,
        'MEX':23,
        'ROW':0
        }

def compute_node_hat(country_index,sol,sol_baseline,quantity):
    i = country_index
    if quantity == 'trade_flow_total':
        return sum([sol.X[j,i,:].sum() for j in range(p_baseline.N) if j!= i]
            )/sum([sol_baseline.X[j,i,:].sum() for j in range(p_baseline.N) if j!= i])
    if quantity == 'trade_flow_non_patent':
        return sum([sol.X[j,i,0] for j in range(p_baseline.N) if j!= i]
            )/sum([sol_baseline.X[j,i,0] for j in range(p_baseline.N) if j!= i])
    if quantity == 'trade_flow_patent':
        return sum([sol.X[j,i,1] for j in range(p_baseline.N) if j!= i]
            )/sum([sol_baseline.X[j,i,1] for j in range(p_baseline.N) if j!= i])
    if quantity == 'patent_flow':
        return sum([sol.pflow[i,j] for j in range(p_nash.N) if j!= i]
                        )/sum([sol_baseline.pflow[i,j] for j in range(p_nash.N) if j!= i])

def compute_node_hat_color(country_index,sol,sol_baseline,quantity):
    # max_color_absolute = max(
    #     np.abs( max([compute_node_hat(k,sol,sol_baseline,quantity) for k in range(p_baseline.N)])-1 ),
    #     np.abs( min([compute_node_hat(k,sol,sol_baseline,quantity) for k in range(p_baseline.N)])-1 ),
    #     )
    if quantity in ['patent_flow']:
        return np.tanh(0.5*np.log10(compute_node_hat(country_index,sol,sol_baseline,quantity)))/2+0.5
    if quantity in ['trade_flow_total','trade_flow_non_patent','trade_flow_patent']:
        temp = max([np.abs(np.tanh(0.5*np.log10(compute_node_hat(k,sol,sol_baseline,quantity)))) for k in range(p_baseline.N)])
        return np.tanh(0.5*np.log10(compute_node_hat(country_index,sol,sol_baseline,quantity)))/(1.5*temp)+0.5

def compute_node_value(country_index,sol,quantity):    
    i = country_index
    if quantity == 'trade_flow_total':
        return sum([sol.X[j,i,:].sum() for j in range(p_baseline.N) if j!= i])*1e6
    if quantity == 'trade_flow_non_patent':
        return sum([sol.X[j,i,0] for j in range(p_baseline.N) if j!= i])*1e6
    if quantity == 'trade_flow_patent':
        return sum([sol.X[j,i,1] for j in range(p_baseline.N) if j!= i])*1e6
    if quantity == 'patent_flow':
        return sum([sol.pflow[i,j] for j in range(p_baseline.N) if j!= i])*1e6

def compute_node_value_size(country_index,sol,quantity):
    return compute_node_value(country_index,sol,quantity)/max([compute_node_value(k,sol,quantity) for k in range(p_baseline.N)])

def compute_welfare(country_index,sol):
    return sol.cons_eq_welfare[i]*100-100

def compute_edge_value_size(destination_index,origin_index,sol,quantity):
    i = destination_index
    j = origin_index
    if quantity == 'trade_flow_total':
        return sol.X[i,j,:].sum()/max([sol.X[k,l,:].sum() for k in range(p_baseline.N) for l in range(p_baseline.N) if k!=l])
    if quantity == 'trade_flow_non_patent':
        return sol.X[i,j,0]/max([sol.X[k,l,0] for k in range(p_baseline.N) for l in range(p_baseline.N) if k!=l])
    if quantity == 'trade_flow_patent':
        return sol.X[i,j,1]/max([sol.X[k,l,1] for k in range(p_baseline.N) for l in range(p_baseline.N) if k!=l])
    if quantity == 'patent_flow':
        return sol.pflow[i,j]/max([sol.pflow[k,l] for k in range(p_baseline.N) for l in range(p_baseline.N) if k!=l])

def compute_edge_hat(destination_index,origin_index,sol,sol_baseline,quantity):
    i = destination_index
    j = origin_index
    if quantity == 'trade_flow_total':
        return sol.X[i,j,:].sum()/sol_baseline.X[i,j,:].sum()
    if quantity == 'trade_flow_non_patent':
        return sol.X[i,j,0]/sol_baseline.X[i,j,0]
    if quantity == 'trade_flow_patent':
        return sol.X[i,j,1]/sol_baseline.X[i,j,1]
    if quantity == 'patent_flow':
        return sol.pflow[i,j]/sol_baseline.pflow[i,j]

def compute_edge_hat_color(destination_index,origin_index,sol,sol_baseline,quantity):
    if quantity in ['patent_flow']:
        return np.tanh(0.5*np.log10(compute_edge_hat(destination_index,origin_index,sol,sol_baseline,quantity)))/2+0.5
    if quantity in ['trade_flow_total','trade_flow_non_patent','trade_flow_patent']:
        temp = max([np.abs(np.tanh(0.5*np.log10(compute_edge_hat(k,l,sol,sol_baseline,quantity)))) for k in range(p_baseline.N) for l in range(p_baseline.N) if k!=l])
        # print(temp)
        # return np.tanh(0.5*np.log10(compute_edge_hat(destination_index,origin_index,sol,sol_baseline,quantity)))/temp/2+0.5
        return np.tanh(0.5*np.log10(compute_edge_hat(destination_index,origin_index,sol,sol_baseline,quantity)))/(1.5*temp)+0.5

for i,country_destination in enumerate(p_baseline.countries):
    node = {
        'id':country_destination,
        'lng':long[country_destination],
        'lat':lat[country_destination],
        'BaselineTFHat':0,
        'BaselineTFHatColor':0.5,
        'BaselineTFValue':compute_node_value(i,sol_baseline,'trade_flow_total'),
        'BaselineTFValueSize':compute_node_value_size(i,sol_baseline,'trade_flow_total'),
        'BaselineTFWelfare':0,
        'BaselineTF0Hat':0,
        'BaselineTF0HatColor':0.5,
        'BaselineTF0Value':compute_node_value(i,sol_baseline,'trade_flow_non_patent'),
        'BaselineTF0ValueSize':compute_node_value_size(i,sol_baseline,'trade_flow_non_patent'),
        'BaselineTF0Welfare':0,
        'BaselineTF1Hat':0,
        'BaselineTF1HatColor':0.5,
        'BaselineTF1Value':compute_node_value(i,sol_baseline,'trade_flow_patent'),
        'BaselineTF1ValueSize':compute_node_value_size(i,sol_baseline,'trade_flow_patent'),
        'BaselineTF1Welfare':0,
        'BaselinePFHat':0,
        'BaselinePFHatColor':0.5,
        'BaselinePFValue':compute_node_value(i,sol_baseline,'patent_flow'),
        'BaselinePFValueSize':compute_node_value_size(i,sol_baseline,'patent_flow'),
        'BaselinePFWelfare':0,
        
        'NashTFHat':compute_node_hat(i,sol_nash,sol_baseline,'trade_flow_total')*100-100,
        'NashTFHatColor':compute_node_hat_color(i,sol_nash,sol_baseline,'trade_flow_total'),
        'NashTFValue':compute_node_value(i,sol_nash,'trade_flow_total'),
        'NashTFValueSize':compute_node_value_size(i,sol_nash,'trade_flow_total'),
        'NashTFWelfare':compute_welfare(i,sol_nash),
        'NashTF0Hat':compute_node_hat(i,sol_nash,sol_baseline,'trade_flow_non_patent')*100-100,
        'NashTF0HatColor':compute_node_hat_color(i,sol_nash,sol_baseline,'trade_flow_non_patent'),
        'NashTF0Value':compute_node_value(i,sol_nash,'trade_flow_non_patent'),
        'NashTF0ValueSize':compute_node_value_size(i,sol_nash,'trade_flow_non_patent'),
        'NashTF0Welfare':compute_welfare(i,sol_nash),
        'NashTF1Hat':compute_node_hat(i,sol_nash,sol_baseline,'trade_flow_patent')*100-100,
        'NashTF1HatColor':compute_node_hat_color(i,sol_nash,sol_baseline,'trade_flow_patent'),
        'NashTF1Value':compute_node_value(i,sol_nash,'trade_flow_patent'),
        'NashTF1ValueSize':compute_node_value_size(i,sol_nash,'trade_flow_patent'),
        'NashTF1Welfare':compute_welfare(i,sol_nash),
        'NashPFHat':compute_node_hat(i,sol_nash,sol_baseline,'patent_flow')*100-100,
        'NashPFHatColor':compute_node_hat_color(i,sol_nash,sol_baseline,'patent_flow'),
        'NashPFValue':compute_node_value(i,sol_nash,'patent_flow'),
        'NashPFValueSize':compute_node_value_size(i,sol_nash,'patent_flow'),
        'NashPFWelfare':compute_welfare(i,sol_nash),
        
        'EqualTFHat':compute_node_hat(i,sol_equal,sol_baseline,'trade_flow_total')*100-100,
        'EqualTFHatColor':compute_node_hat_color(i,sol_equal,sol_baseline,'trade_flow_total'),
        'EqualTFValue':compute_node_value(i,sol_equal,'trade_flow_total'),
        'EqualTFValueSize':compute_node_value_size(i,sol_equal,'trade_flow_total'),
        'EqualTFWelfare':compute_welfare(i,sol_equal),
        'EqualTF0Hat':compute_node_hat(i,sol_equal,sol_baseline,'trade_flow_non_patent')*100-100,
        'EqualTF0HatColor':compute_node_hat_color(i,sol_equal,sol_baseline,'trade_flow_non_patent'),
        'EqualTF0Value':compute_node_value(i,sol_equal,'trade_flow_non_patent'),
        'EqualTF0ValueSize':compute_node_value_size(i,sol_equal,'trade_flow_non_patent'),
        'EqualTF0Welfare':compute_welfare(i,sol_equal),
        'EqualTF1Hat':compute_node_hat(i,sol_equal,sol_baseline,'trade_flow_patent')*100-100,
        'EqualTF1HatColor':compute_node_hat_color(i,sol_equal,sol_baseline,'trade_flow_patent'),
        'EqualTF1Value':compute_node_value(i,sol_equal,'trade_flow_patent'),
        'EqualTF1ValueSize':compute_node_value_size(i,sol_equal,'trade_flow_patent'),
        'EqualTF1Welfare':compute_welfare(i,sol_equal),
        'EqualPFHat':compute_node_hat(i,sol_equal,sol_baseline,'patent_flow')*100-100,
        'EqualPFHatColor':compute_node_hat_color(i,sol_equal,sol_baseline,'patent_flow'),
        'EqualPFValue':compute_node_value(i,sol_equal,'patent_flow'),
        'EqualPFValueSize':compute_node_value_size(i,sol_equal,'patent_flow'),
        'EqualPFWelfare':compute_welfare(i,sol_equal),
        
        'NegishiTFHat':compute_node_hat(i,sol_negishi,sol_baseline,'trade_flow_total')*100-100,
        'NegishiTFHatColor':compute_node_hat_color(i,sol_negishi,sol_baseline,'trade_flow_total'),
        'NegishiTFValue':compute_node_value(i,sol_negishi,'trade_flow_total'),
        'NegishiTFValueSize':compute_node_value_size(i,sol_negishi,'trade_flow_total'),
        'NegishiTFWelfare':compute_welfare(i,sol_negishi),
        'NegishiTF0Hat':compute_node_hat(i,sol_negishi,sol_baseline,'trade_flow_non_patent')*100-100,
        'NegishiTF0HatColor':compute_node_hat_color(i,sol_negishi,sol_baseline,'trade_flow_non_patent'),
        'NegishiTF0Value':compute_node_value(i,sol_negishi,'trade_flow_non_patent'),
        'NegishiTF0ValueSize':compute_node_value_size(i,sol_negishi,'trade_flow_non_patent'),
        'NegishiTF0Welfare':compute_welfare(i,sol_negishi),
        'NegishiTF1Hat':compute_node_hat(i,sol_negishi,sol_baseline,'trade_flow_patent')*100-100,
        'NegishiTF1HatColor':compute_node_hat_color(i,sol_negishi,sol_baseline,'trade_flow_patent'),
        'NegishiTF1Value':compute_node_value(i,sol_negishi,'trade_flow_patent'),
        'NegishiTF1ValueSize':compute_node_value_size(i,sol_negishi,'trade_flow_patent'),
        'NegishiTF1Welfare':compute_welfare(i,sol_negishi),
        'NegishiPFHat':compute_node_hat(i,sol_negishi,sol_baseline,'patent_flow')*100-100,
        'NegishiPFHatColor':compute_node_hat_color(i,sol_negishi,sol_baseline,'patent_flow'),
        'NegishiPFValue':compute_node_value(i,sol_negishi,'patent_flow'),
        'NegishiPFValueSize':compute_node_value_size(i,sol_negishi,'patent_flow'),
        'NegishiPFWelfare':compute_welfare(i,sol_negishi),
        
        'UniformTFHat':compute_node_hat(i,sol_uniform_delta,sol_baseline,'trade_flow_total')*100-100,
        'UniformTFHatColor':compute_node_hat_color(i,sol_uniform_delta,sol_baseline,'trade_flow_total'),
        'UniformTFValue':compute_node_value(i,sol_uniform_delta,'trade_flow_total'),
        'UniformTFValueSize':compute_node_value_size(i,sol_uniform_delta,'trade_flow_total'),
        'UniformTFWelfare':compute_welfare(i,sol_uniform_delta),
        'UniformTF0Hat':compute_node_hat(i,sol_uniform_delta,sol_baseline,'trade_flow_non_patent')*100-100,
        'UniformTF0HatColor':compute_node_hat_color(i,sol_uniform_delta,sol_baseline,'trade_flow_non_patent'),
        'UniformTF0Value':compute_node_value(i,sol_uniform_delta,'trade_flow_non_patent'),
        'UniformTF0ValueSize':compute_node_value_size(i,sol_uniform_delta,'trade_flow_non_patent'),
        'UniformTF0Welfare':compute_welfare(i,sol_uniform_delta),
        'UniformTF1Hat':compute_node_hat(i,sol_uniform_delta,sol_baseline,'trade_flow_patent')*100-100,
        'UniformTF1HatColor':compute_node_hat_color(i,sol_uniform_delta,sol_baseline,'trade_flow_patent'),
        'UniformTF1Value':compute_node_value(i,sol_uniform_delta,'trade_flow_patent'),
        'UniformTF1ValueSize':compute_node_value_size(i,sol_uniform_delta,'trade_flow_patent'),
        'UniformTF1Welfare':compute_welfare(i,sol_uniform_delta),
        'UniformPFHat':compute_node_hat(i,sol_uniform_delta,sol_baseline,'patent_flow')*100-100,
        'UniformPFHatColor':compute_node_hat_color(i,sol_uniform_delta,sol_baseline,'patent_flow'),
        'UniformPFValue':compute_node_value(i,sol_uniform_delta,'patent_flow'),
        'UniformPFValueSize':compute_node_value_size(i,sol_uniform_delta,'patent_flow'),
        'UniformPFWelfare':compute_welfare(i,sol_uniform_delta),
        
        'PreTFHat':compute_node_hat(i,sol_pre_trips,sol_baseline,'trade_flow_total')*100-100,
        'PreTFHatColor':compute_node_hat_color(i,sol_pre_trips,sol_baseline,'trade_flow_total'),
        'PreTFValue':compute_node_value(i,sol_pre_trips,'trade_flow_total'),
        'PreTFValueSize':compute_node_value_size(i,sol_pre_trips,'trade_flow_total'),
        'PreTFWelfare':compute_welfare(i,sol_pre_trips),
        'PreTF0Hat':compute_node_hat(i,sol_pre_trips,sol_baseline,'trade_flow_non_patent')*100-100,
        'PreTF0HatColor':compute_node_hat_color(i,sol_pre_trips,sol_baseline,'trade_flow_non_patent'),
        'PreTF0Value':compute_node_value(i,sol_pre_trips,'trade_flow_non_patent'),
        'PreTF0ValueSize':compute_node_value_size(i,sol_pre_trips,'trade_flow_non_patent'),
        'PreTF0Welfare':compute_welfare(i,sol_pre_trips),
        'PreTF1Hat':compute_node_hat(i,sol_pre_trips,sol_baseline,'trade_flow_patent')*100-100,
        'PreTF1HatColor':compute_node_hat_color(i,sol_pre_trips,sol_baseline,'trade_flow_patent'),
        'PreTF1Value':compute_node_value(i,sol_pre_trips,'trade_flow_patent'),
        'PreTF1ValueSize':compute_node_value_size(i,sol_pre_trips,'trade_flow_patent'),
        'PreTF1Welfare':compute_welfare(i,sol_pre_trips),
        'PrePFHat':compute_node_hat(i,sol_pre_trips,sol_baseline,'patent_flow')*100-100,
        'PrePFHatColor':compute_node_hat_color(i,sol_pre_trips,sol_baseline,'patent_flow'),
        'PrePFValue':compute_node_value(i,sol_pre_trips,'patent_flow'),
        'PrePFValueSize':compute_node_value_size(i,sol_pre_trips,'patent_flow'),
        'PrePFWelfare':compute_welfare(i,sol_pre_trips),
        }
    nodes.append(node)
    
    for j,country_origin in enumerate(p_baseline.countries):
        if j!= i:
            edge = {
                'id':country_origin+country_destination,
                'source':country_origin,
                'target':country_destination,
                'totalShareInputTarget':sol_baseline.X[i,j].sum()/sum([sol_baseline.X[i,k].sum() for k in range(p_baseline.N) if k!= i]),
                'totalShareOutputSource':sol_baseline.X[i,j].sum()/sum([sol_baseline.X[k,i].sum() for k in range(p_baseline.N) if k!= i]),
                'BaselineTFHatColor':0.5,
                'BaselineTFValueSize':compute_edge_value_size(i,j,sol_baseline,'trade_flow_total'),
                'BaselineTF0HatColor':0.5,
                'BaselineTF0ValueSize':compute_edge_value_size(i,j,sol_baseline,'trade_flow_non_patent'),
                'BaselineTF1HatColor':0.5,
                'BaselineTF1ValueSize':compute_edge_value_size(i,j,sol_baseline,'trade_flow_patent'),
                'BaselinePFHatColor':0.5,
                'BaselinePFValueSize':compute_edge_value_size(i,j,sol_baseline,'patent_flow'),
                
                'NashTFHatColor':compute_edge_hat_color(i,j,sol_nash,sol_baseline,'trade_flow_total'),
                'NashTFValueSize':compute_edge_value_size(i,j,sol_nash,'trade_flow_total'),
                'NashTF0HatColor':compute_edge_hat_color(i,j,sol_nash,sol_baseline,'trade_flow_non_patent'),
                'NashTF0ValueSize':compute_edge_value_size(i,j,sol_nash,'trade_flow_non_patent'),
                'NashTF1HatColor':compute_edge_hat_color(i,j,sol_nash,sol_baseline,'trade_flow_patent'),
                'NashTF1ValueSize':compute_edge_value_size(i,j,sol_nash,'trade_flow_patent'),
                'NashPFHatColor':compute_edge_hat_color(i,j,sol_nash,sol_baseline,'patent_flow'),
                'NashPFValueSize':compute_edge_value_size(i,j,sol_nash,'patent_flow'),
                                
                'EqualTFHatColor':compute_edge_hat_color(i,j,sol_equal,sol_baseline,'trade_flow_total'),
                'EqualTFValueSize':compute_edge_value_size(i,j,sol_equal,'trade_flow_total'),
                'EqualTF0HatColor':compute_edge_hat_color(i,j,sol_equal,sol_baseline,'trade_flow_non_patent'),
                'EqualTF0ValueSize':compute_edge_value_size(i,j,sol_equal,'trade_flow_non_patent'),
                'EqualTF1HatColor':compute_edge_hat_color(i,j,sol_equal,sol_baseline,'trade_flow_patent'),
                'EqualTF1ValueSize':compute_edge_value_size(i,j,sol_equal,'trade_flow_patent'),
                'EqualPFHatColor':compute_edge_hat_color(i,j,sol_equal,sol_baseline,'patent_flow'),
                'EqualPFValueSize':compute_edge_value_size(i,j,sol_equal,'patent_flow'),
                                
                'NegishiTFHatColor':compute_edge_hat_color(i,j,sol_negishi,sol_baseline,'trade_flow_total'),
                'NegishiTFValueSize':compute_edge_value_size(i,j,sol_negishi,'trade_flow_total'),
                'NegishiTF0HatColor':compute_edge_hat_color(i,j,sol_negishi,sol_baseline,'trade_flow_non_patent'),
                'NegishiTF0ValueSize':compute_edge_value_size(i,j,sol_negishi,'trade_flow_non_patent'),
                'NegishiTF1HatColor':compute_edge_hat_color(i,j,sol_negishi,sol_baseline,'trade_flow_patent'),
                'NegishiTF1ValueSize':compute_edge_value_size(i,j,sol_negishi,'trade_flow_patent'),
                'NegishiPFHatColor':compute_edge_hat_color(i,j,sol_negishi,sol_baseline,'patent_flow'),
                'NegishiPFValueSize':compute_edge_value_size(i,j,sol_negishi,'patent_flow'),
                                                
                'UniformTFHatColor':compute_edge_hat_color(i,j,sol_uniform_delta,sol_baseline,'trade_flow_total'),
                'UniformTFValueSize':compute_edge_value_size(i,j,sol_uniform_delta,'trade_flow_total'),
                'UniformTF0HatColor':compute_edge_hat_color(i,j,sol_uniform_delta,sol_baseline,'trade_flow_non_patent'),
                'UniformTF0ValueSize':compute_edge_value_size(i,j,sol_uniform_delta,'trade_flow_non_patent'),
                'UniformTF1HatColor':compute_edge_hat_color(i,j,sol_uniform_delta,sol_baseline,'trade_flow_patent'),
                'UniformTF1ValueSize':compute_edge_value_size(i,j,sol_uniform_delta,'trade_flow_patent'),
                'UniformPFHatColor':compute_edge_hat_color(i,j,sol_uniform_delta,sol_baseline,'patent_flow'),
                'UniformPFValueSize':compute_edge_value_size(i,j,sol_uniform_delta,'patent_flow'),
                                                
                'PreTFHatColor':compute_edge_hat_color(i,j,sol_pre_trips,sol_baseline,'trade_flow_total'),
                'PreTFValueSize':compute_edge_value_size(i,j,sol_pre_trips,'trade_flow_total'),
                'PreTF0HatColor':compute_edge_hat_color(i,j,sol_pre_trips,sol_baseline,'trade_flow_non_patent'),
                'PreTF0ValueSize':compute_edge_value_size(i,j,sol_pre_trips,'trade_flow_non_patent'),
                'PreTF1HatColor':compute_edge_hat_color(i,j,sol_pre_trips,sol_baseline,'trade_flow_patent'),
                'PreTF1ValueSize':compute_edge_value_size(i,j,sol_pre_trips,'trade_flow_patent'),
                'PrePFHatColor':compute_edge_hat_color(i,j,sol_pre_trips,sol_baseline,'patent_flow'),
                'PrePFValueSize':compute_edge_value_size(i,j,sol_pre_trips,'patent_flow'),
                }
            edges.append(edge)
            
dic = {
       'nodes':nodes,
       'edges':edges       
       }


#%%

import json
import os
try:
    os.remove('../../TRIPS_maps/data/graph.json')
except:
    pass
# r = json.dumps(dic)
with open('../../TRIPS_maps/data/graph.json', 'w') as f:
    json.dump(dic, f)