#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:45:48 2022

@author: simonl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from classes import moments, parameters, var, dynamic_var

def write_calibration_results(path,p,m,sol_c,commentary = None):
    writer = pd.ExcelWriter(path+'.xlsx', engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Summary')
    writer.sheets['Summary'] = worksheet
    df1 = pd.DataFrame(index = p.get_list_of_params(), 
                      columns = ['state', 'lower bound', 'higher bound'])
    for pa_name in p.get_list_of_params():
        if pa_name in p.calib_parameters:
            df1.loc[pa_name] = ['calibrated',p.lb_dict[pa_name],p.ub_dict[pa_name]]
        else:
            df1.loc[pa_name] = ['fixed','','']
    df1.name = 'parameters'
    worksheet.write_string(0, 0, df1.name)
    df1.to_excel(writer,sheet_name='Summary',startrow=1 , startcol=0)
    
    
    df2 = pd.DataFrame(index = m.get_list_of_moments(), columns = ['weight','norm of deviation', 'description'])
    for mom in m.get_list_of_moments():
        if mom in m.list_of_moments:
            df2.loc[mom] = [m.weights_dict[mom],
                            np.linalg.norm(getattr(m,mom+'_deviation')),
                            m.description.loc[mom].description]
        else:
            df2.loc[mom] = [0,
                            np.linalg.norm(getattr(m,mom+'_deviation')),
                            m.description.loc[mom].description]
    df2.name = 'targeted moments : '+str(m.list_of_moments)
    worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
    df2.to_excel(writer,sheet_name='Summary',startrow=df1.shape[0] + 5 , startcol=0)
    
    worksheet.write_string(df1.shape[0] + df2.shape[0] + 6, 0, commentary)
    
    
    scalar_moments = pd.DataFrame(columns=['model','target'])
    for mom in m.get_list_of_moments():
        if np.array(getattr(m,mom)).size == 1:
            scalar_moments.loc[mom] = [getattr(m,mom),getattr(m,mom+'_target')]
        else:
            moment = getattr(m,mom)
            moment_target = getattr(m,mom+'_target')
            df = pd.DataFrame({'model':np.array(moment).ravel(),'target':np.array(moment_target).ravel()},
                              index=m.idx[mom])
            df.to_excel(writer,sheet_name=mom)
    scalar_moments.to_excel(writer,sheet_name='scalar_moments')
    scalar_parameters = pd.DataFrame(columns=['value'])
    for pa_name in p.get_list_of_params():
        if np.array(getattr(p,pa_name)).size == 1:
            scalar_parameters.loc[pa_name] = getattr(p,pa_name)
        else:
            par = getattr(p,pa_name)
            df = pd.DataFrame({'value':np.array(par).ravel()},index=p.idx[pa_name])
            df.to_excel(writer,sheet_name=pa_name)
    scalar_parameters.to_excel(writer,sheet_name='scalar_parameters')
    
    df_labor = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_labor['non patenting'] = sol_c.nominal_value_added[:,0]/sol_c.w
    df_labor['production patenting sector'] = sol_c.nominal_value_added[:,1]/sol_c.w
    df_labor['RD'] = sol_c.l_R[:,1]
    df_labor['patenting'] = sol_c.l_Ao[:,1]+sol_c.l_Ae[:,:,1].sum(axis=1)
    df_labor['total'] = df_labor['non patenting']+df_labor['production patenting sector']+df_labor['RD']+df_labor['patenting']
    df_labor['total data'] = p.labor
    df_labor.to_excel(writer,sheet_name='labor')
    
    df_psi_m_star = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries,p.sectors],names=['destination','origin','sector']))
    df_psi_m_star['psi_m_star'] = sol_c.psi_m_star.ravel()
    df_psi_m_star.to_excel(writer,sheet_name='psi_m_star')
    df_psi_o_star = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.sectors],names=['country','sector']))
    df_psi_o_star['psi_o_star'] = sol_c.psi_o_star.ravel()
    df_psi_o_star.to_excel(writer,sheet_name='psi_o_star')
    
    df_g_s = pd.DataFrame(index = pd.Index(p.sectors,name='sector'))
    df_g_s['g_s'] = sol_c.g_s
    df_g_s.to_excel(writer,sheet_name='sectoral_growth_rates')
    
    df_r_and_g = pd.DataFrame(index = ['aggregate'])
    df_r_and_g.loc['g','aggregate'] = sol_c.g
    df_r_and_g.loc['r','aggregate'] = sol_c.r
    df_r_and_g.to_excel(writer,sheet_name='growth_interest_rates')
    
    df_sales = pd.DataFrame(index=pd.MultiIndex.from_product([p.countries, p.countries],names=['destination','origin']))
    df_sales['M share of sales'] = sol_c.X_M[:,:,1].ravel()
    df_sales['CD share of sales'] = sol_c.X_CD[:,:,1].ravel()
    df_sales['total to check'] = df_sales['M share of sales'] + df_sales['CD share of sales']
    df_sales.to_excel(writer,sheet_name='monopolistic_competitive_shares')
    
    df_qualities = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_qualities['PSI_M'] = sol_c.PSI_M[...,1].sum(axis=1)
    df_qualities['PSI_CD'] = sol_c.PSI_CD[...,1]
    df_qualities['total check'] = df_qualities['PSI_M']+df_qualities['PSI_CD']
    df_qualities.to_excel(writer,sheet_name='aggregate_qualities')    
    
    df_prices = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_prices['P_M'] = sol_c.P_M[...,1]
    df_prices['P_CD'] = sol_c.P_CD[...,1]
    df_prices['P'] = sol_c.price_indices
    df_prices.to_excel(writer,sheet_name='prices')
    
    df_country = pd.DataFrame(index=pd.Index(p.countries,name='country'))
    df_country['wage'] = sol_c.w
    df_country['expenditure'] = sol_c.Z
    df_country['gdp'] = sol_c.gdp
    df_country.to_excel(writer,sheet_name='countries_macro_quantities')
    
    df_pflows = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries],names=['destination','origin']))
    df_pflows['pflow'] = sol_c.pflow.ravel()
    df_pflows.to_excel(writer,sheet_name='number of patent flows')
    
    df_tau = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries, p.sectors],names=['destination','origin','sector']))
    df_tau['tau'] = sol_c.tau.ravel()
    df_tau.to_excel(writer,sheet_name='tau')
    
    df_share_patented = pd.DataFrame(index = pd.MultiIndex.from_product([p.countries,p.countries],names=['destination','origin']))
    df_share_patented['share_innov_patented'] = sol_c.share_innov_patented.ravel()
    df_share_patented.to_excel(writer,sheet_name='share of innovations patented')
    
    writer.close()

def get_vec_qty(x,p):
    res = {'w':x[0:p.N],
           'Z':x[p.N:p.N+1],
           'l_R':x[p.N+1:p.N+1+p.N*(p.S-1)],
           'profit':x[p.N+1+p.N*(p.S-1):p.N+1+p.N*(p.S-1)+p.N**2],
           'phi':x[p.N+1+p.N*(p.S-1)+p.N**2:]
           }
    return res

def compute_rough_jacobian(p, m, qty_to_change, idx_to_change, context = 'calibration',
                           change_by = 1e-2, tol = 1e-14, damping = 5,
                           max_count = 5e3):
    m.compute_moments_deviations()
    m_diff = m.copy()
    p_diff = p.copy()
    qty = getattr(p,qty_to_change)
    qty[idx_to_change] = qty[idx_to_change]*(1+change_by)
    setattr(p_diff, qty_to_change, qty)
    try:
        x_old = p_diff.guess
    except:
        x_old = p_diff.guess_from_params()
    count = 0
    x_new = None
    condition = True
    convergence = []
    while condition:
        if count != 0:
            x_old = (x_new+(damping-1)*x_old)/damping
        init = var.var_from_vector(x_old,p_diff,compute=False, context = context)
        init.compute_solver_quantities(p_diff)
        w = init.compute_wage(p_diff)
        Z = init.compute_expenditure(p_diff)
        l_R = init.compute_labor_research(p_diff)[...,1:].ravel()
        profit = init.compute_profit(p_diff)[...,1:].ravel()
        phi = init.compute_phi(p_diff).ravel()
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)
        x_new_decomp = get_vec_qty(x_new,p_diff)
        x_old_decomp = get_vec_qty(x_old,p_diff)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions) and count<max_count
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        count += 1
    sol_c = var.var_from_vector(x_old,p_diff,compute=True, context = context)
    sol_c.scale_P(p_diff)
    sol_c.compute_non_solver_quantities(p_diff)
    m_diff.compute_moments(sol_c,p_diff)
    m_diff.compute_moments_deviations()
    
    return (m_diff.deviation_vector()-m.deviation_vector())*(1+change_by)*m.deviation_vector()/(change_by*qty[idx_to_change])
    
    
def load(path, data_path=None, context = 'calibration'):
    p = parameters()
    p.load_run(path)
    sol = var.var_from_vector(p.guess, p, compute=True, context = context)
    sol.scale_P(p)
    sol.compute_non_solver_quantities(p)
    m = moments()
    # m.load_data(data_path)
    m.load_run(path)
    m.compute_moments(sol, p)
    m.compute_moments_deviations()
    return p,m,sol

def get_path(baseline,variation,results_path):
    if variation == 'baseline':
        path = results_path+baseline+'/'
    else:
        path = results_path+'baseline_'+baseline+'_variations/'+variation+'/'
    return path

def repeat_for_all_times(array,Nt):
    return np.repeat(array[..., np.newaxis],Nt,axis=len(array.shape))

def guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin,C=20):
    def build_guess(fin,init,C=C):
        if len(fin.shape) == 2:
            return (fin-init)[...,1:,None]*(
                np.exp( -C* (dyn_var.t_cheby+1) )[None,None,:]-1
                )/(np.exp(-2*C)-1)
        elif len(fin.shape) == 3:
            return (fin-init)[...,1:,None]*(
                np.exp(-C* (dyn_var.t_cheby+1) )[None,None,None,:]-1
                )/(np.exp(-2*C)-1)
    guess = {}
    guess['PSI_CD'] = build_guess(sol_fin.PSI_CD,dyn_var.PSI_CD_0)
    guess['PSI_MNP'] = build_guess(sol_fin.PSI_MNP,dyn_var.PSI_MNP_0)
    guess['PSI_MPND'] = build_guess(sol_fin.PSI_MPND,dyn_var.PSI_MPND_0)
    guess['PSI_MPD'] = build_guess(sol_fin.PSI_MPD,dyn_var.PSI_MPD_0)
    return guess

def rough_fixed_point_solver(p, context, x0=None, tol = 1e-10, damping = 5, max_count=1e6,
                       safe_convergence=0.1, damping_post_acceleration=2,
                       accelerate_when_stable=True):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    x_new = None
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        
        if count != 0:
            x_old = (x_new+(damping-1)*x_old)/damping
        init = var.var_from_vector(x_old,p,context=context,compute=False)
        init.compute_solver_quantities(p)
        
        w = init.compute_wage(p)
        Z = init.compute_expenditure(p)
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()
      
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)

        
        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                damping = damping_post_acceleration

    return init

def rough_dyn_fixed_point_solver(p, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-14, max_count=1e6,
                       safe_convergence=0.1,damping=50, damping_post_acceleration=10):  
    if sol_fin is None:
        sol_fin = rough_fixed_point_solver(p,x0=p.guess,
                                        context = 'counterfactual',tol =1e-14,
                                safe_convergence=0.001,
                                damping = 10,
                                max_count = 5000,
                                damping_post_acceleration=2
                                )
        sol_fin.scale_P(p)
        sol_fin.compute_non_solver_quantities(p) 
    
    dyn_var = dynamic_var(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin,N=p.N)
    # print(dyn_var.t_inf)
    dyn_var.initiate_state_variables_0(sol_init)
    
    psis_guess = guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin)
    
    dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
                    'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
                    'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
                    'PSI_CD':psis_guess['PSI_CD'],
                    'PSI_MNP':psis_guess['PSI_MNP'],
                    'PSI_MPND':psis_guess['PSI_MPND'],
                    'PSI_MPD':psis_guess['PSI_MPD'],
                    'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
                    'V_NP':repeat_for_all_times(sol_fin.V_NP,dyn_var.Nt)[...,1:,:],
                    'DELTA_V':repeat_for_all_times(sol_fin.V_P-sol_fin.V_NP,dyn_var.Nt)[...,1:,:]
                    }
    dyn_var.guess_from_dic(dic_of_guesses)
    if x0 is not None:
        dyn_var.guess_from_vector(x0)
    
    x_old = dyn_var.vector_from_var()
        
    condition = True
    count = 0
    convergence = []
    x_new = None

    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            x_old = (x_new+(damping-1)*x_old)/damping
            dyn_var.guess_from_vector(x_old)
            numeraire = dyn_var.price_indices[0,:]
            for qty in ['price_indices','w','Z']:
                temp = getattr(dyn_var,qty)
                temp = temp/numeraire[None,:]
                setattr(dyn_var,qty,temp)
            for qty in ['V_PD','DELTA_V','V_NP']:
                temp = getattr(dyn_var,qty)
                temp = temp/numeraire[None,None,None:]
                setattr(dyn_var,qty,temp)
            x_old = dyn_var.vector_from_var()
        dyn_var.compute_solver_quantities(p)
        x_new = np.concatenate([
            dyn_var.compute_price_indices(p).ravel(),
            dyn_var.compute_wage(p).ravel(),
            dyn_var.compute_expenditure(p).ravel(),
            dyn_var.compute_PSI_CD(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MNP(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPND(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPD(p)[...,1:,:].ravel(),
            dyn_var.compute_V_PD(p)[...,1:,:].ravel(),
            dyn_var.compute_DELTA_V(p)[...,1:,:].ravel(),
            dyn_var.compute_V_NP(p)[...,1:,:].ravel(),
            ],axis=0)

        condition = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old) > tol
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        
        count += 1
    
    dyn_var.compute_non_solver_quantities(p)

    return dyn_var, sol_fin, convergence[-1]

def make_counterfactual_recap(p_baseline, sol_baseline, country,
                              local_path,recap_path,harmonizing_country='USA',
                              dynamics=False,Nt=25,t_inf=500):
    try:
        os.mkdir(recap_path)
    except:
        pass
    recap = pd.DataFrame(columns = ['delt','growth']+p_baseline.countries)
    recap_dyn = pd.DataFrame(columns = ['delt']+p_baseline.countries)
    if country in p_baseline.countries:
        idx_country = p_baseline.countries.index(country)
    if country == 'Harmonizing' or country == 'Upper_harmonizing':
        idx_country = p_baseline.countries.index(harmonizing_country)
    if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
        idx_country = p_baseline.countries.index(country[:3])
    country_path = local_path+country+'/'
    files_in_dir = next(os.walk(country_path))[1]
    run_list = [f for f in files_in_dir if f[0].isnumeric()]
    run_list.sort(key=float)
    
    for i,run in enumerate(run_list):
        # print(run)
        p = parameters()
        p.load_run(country_path+run+'/')
        
        sol_c = var.var_from_vector(p.guess, p, compute=True, context = 'counterfactual')
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p)
        sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
        if country in p_baseline.countries:
            recap.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
            recap_dyn.loc[run, 'delt'] = p.delta[idx_country,1]/p_baseline.delta[idx_country,1]
            # print(p.delta[idx_country,1])
            # print(idx_country)
        if country == 'World':
            recap.loc[run, 'delt'] = p.delta[0,1]/p_baseline.delta[0,1]
            recap_dyn.loc[run, 'delt'] = p.delta[0,1]/p_baseline.delta[0,1]
        if country == 'Harmonizing':
            recap.loc[run, 'delt'] = np.log(
                p.delta[1,1]/p_baseline.delta[1,1]
                )/np.log(
                    p_baseline.delta[idx_country,1]/p_baseline.delta[1,1]
                    )
            recap_dyn.loc[run, 'delt'] = np.log(
                p.delta[1,1]/p_baseline.delta[1,1]
                )/np.log(
                    p_baseline.delta[idx_country,1]/p_baseline.delta[1,1]
                    )
        if country == 'Upper_harmonizing':
            recap.loc[run, 'delt'] = np.log(
                p.delta[-1,1]/p_baseline.delta[-1,1]
                )/np.log(
                    p_baseline.delta[idx_country,1]/p_baseline.delta[-1,1]
                    )
            recap_dyn.loc[run, 'delt'] = np.log(
                p.delta[-1,1]/p_baseline.delta[-1,1]
                )/np.log(
                    p_baseline.delta[idx_country,1]/p_baseline.delta[-1,1]
                    )
        if country == 'trade_cost_eq_trips_all_countries_all_sectors':
            recap.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
            recap_dyn.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
        if country == 'trade_cost_eq_trips_all_countries_pat_sectors':
            recap.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
            recap_dyn.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
        if country == 'trade_cost_all_countries_all_sectors':
            recap.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
            recap_dyn.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
        if country == 'trade_cost_all_countries_pat_sectors':
            recap.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
            recap_dyn.loc[run, 'delt'] = p.tau[0,1,1]/p_baseline.tau[0,1,1]
        if country == 'Uniform_delta':
            recap.loc[run, 'delt'] = p.delta[0,1]
            recap_dyn.loc[run, 'delt'] = p.delta[0,1]
        if country == 'Upper_uniform_delta':
            recap.loc[run, 'delt'] = np.logspace(-2,0,101)[i]
            recap_dyn.loc[run, 'delt'] = np.logspace(-2,0,101)[i]
        if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
            recap.loc[run, 'delt'] = p.tau[idx_country,idx_country-1,1]/p_baseline.tau[idx_country,idx_country-1,1]
            recap_dyn.loc[run, 'delt'] = p.tau[idx_country,idx_country-1,1]/p_baseline.tau[idx_country,idx_country-1,1]
        recap.loc[run, 'growth'] = sol_c.g
        recap.loc[run,p_baseline.countries] = sol_c.cons_eq_welfare
        
        if dynamics:
            dyn_sol_c = dynamic_var.var_from_vector(p.dyn_guess, p, compute=True,
                                                    Nt=Nt,t_inf=t_inf,
                                                    sol_init = sol_baseline,
                                                    sol_fin = sol_c)
            dyn_sol_c.compute_non_solver_quantities(p)
            recap_dyn.loc[run,p_baseline.countries] = dyn_sol_c.cons_eq_welfare
        
            
    if not dynamics:
        recap.to_csv(recap_path+country+'.csv', index=False)
    if dynamics:
        recap_dyn.to_csv(recap_path+'dyn_'+country+'.csv', index=False)
        # print(country,recap_dyn)