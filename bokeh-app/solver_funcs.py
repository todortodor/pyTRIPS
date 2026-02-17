#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:37:41 2022

@author: simonl
"""

import numpy as np
import aa
import matplotlib.pyplot as plt
import time
from classes import cobweb, sol_class, moments, parameters, var, var_with_entry_costs,var_double_diff_double_delta, history_nash, dynamic_var, dynamic_var_double_diff_double_delta
from scipy import optimize
import os
from optimparallel import minimize_parallel
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter('ignore', RuntimeWarning)

def get_vec_qty(x,p):
    res = {'w':x[0:p.N],
           'Z':x[p.N:p.N+p.N],
           'l_R':x[p.N+p.N:p.N+p.N+p.N*(p.S-1)],
           'profit':x[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2],
           'phi':x[p.N+p.N+p.N*(p.S-1)+p.N**2:]
           }
    return res

def bound_zero(x, cutoff=1e-8, hit_the_bound=None):
    if np.any(x<=0):
        x[x <= 0] = cutoff
        if hit_the_bound is not None:
            hit_the_bound+=1
    return x,hit_the_bound

def fixed_point_solver(p, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False, keep_l_R_fixed=False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var.var_from_vector(x_old,p,context=context,compute=False)
        if count == 0 and keep_l_R_fixed:
            l_R_0 = init.l_R[...,1:].ravel().copy()
        init.compute_solver_quantities(p)
        
        # print(init.compute_labor_research(p)[...,1:])
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        if keep_l_R_fixed:
            l_R = l_R_0.copy()
        
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            # if count%5==0:
            #     plt.plot(convergence)
            #     plt.yscale('log')
            #     plt.show()
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
        
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init

def fixed_point_solver_with_exog_pat_and_rd(p, p_old, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var.var_from_vector(x_old,p,context=context,compute=False)
        if count == 0:
            sol_0 = init.copy()
            sol_0.compute_solver_quantities(p_old)
            # print(sol_0.PSI_CD)
            # time.sleep(5)
        # init.compute_solver_quantities(p)
        
        init.compute_growth(p)
        # init.compute_patenting_thresholds(p)
        
        init.psi_C = sol_0.psi_C.copy()
        init.psi_star = sol_0.psi_star.copy()
        init.psi_o_star = sol_0.psi_o_star.copy()
        init.psi_m_star = sol_0.psi_m_star.copy()
        
        # init.compute_aggregate_qualities(p)
        
        init.PSI_M = sol_0.PSI_M.copy()
        init.PSI_CD = sol_0.PSI_CD.copy()
        
        init.compute_sectoral_prices(p)
        
        # init.compute_labor_allocations(p)
        
        init.l_Ae = sol_0.l_Ae.copy()
        init.l_Ao = sol_0.l_Ao.copy()
        init.l_P = sol_0.l_P.copy()
        
        init.compute_trade_flows_and_shares(p)
        init.compute_price_indices(p)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        l_R = sol_0.l_R[...,1:].ravel().copy()
        
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            # plt.plot(convergence)
            # plt.show()
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()

    return sol_inst, init

def fixed_point_solver_exog_lr(p, p_old, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var.var_from_vector(x_old,p,context=context,compute=False)
        if count == 0:
            sol_0 = init.copy()
            sol_0.compute_solver_quantities(p_old)
            # print(sol_0.PSI_CD)
            # time.sleep(5)
        init.compute_solver_quantities(p)
        
        # init.compute_growth(p)
        # init.compute_patenting_thresholds(p)
        
        # # init.psi_C = sol_0.psi_C.copy()
        # # init.psi_star = sol_0.psi_star.copy()
        # # init.psi_o_star = sol_0.psi_o_star.copy()
        # # init.psi_m_star = sol_0.psi_m_star.copy()
        
        # init.compute_aggregate_qualities(p)
        
        # # init.PSI_M = sol_0.PSI_M.copy()
        # # init.PSI_CD = sol_0.PSI_CD.copy()
        
        # init.compute_sectoral_prices(p)
        
        # init.compute_labor_allocations(p)
        
        # # init.l_Ae = sol_0.l_Ae.copy()
        # # init.l_Ao = sol_0.l_Ao.copy()
        # # init.l_P = sol_0.l_P.copy()
        
        # init.compute_trade_flows_and_shares(p)
        # init.compute_price_indices(p)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        l_R = sol_0.l_R[...,1:].ravel().copy()
        
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            # plt.plot(convergence)
            # plt.show()
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()

    return sol_inst, init

def fixed_point_solver_with_entry_costs(p, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False, keep_l_R_fixed=False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params(for_solver_with_entry_costs=True)
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    
    
    while condition and count < max_count and np.all(x_old<1e40): 
        # print(count)
        
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var_with_entry_costs.var_from_vector(x_old,p,context=context,compute=False)
        if count == 0 and keep_l_R_fixed:
            l_R_0 = init.l_R[...,1:].ravel().copy()
        
        init.compute_solver_quantities(p)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        price_indices = init.compute_price_indices(p)
        
        if keep_l_R_fixed:
            l_R = l_R_0.copy()
        
        x_new = np.concatenate((w/price_indices[0],Z/price_indices[0],l_R,profit,phi*price_indices[0],price_indices/price_indices[0]), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        # print([np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty])
        #               for qty in ['w','Z','profit','l_R','phi']])
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if False and plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if False and plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            if count%100==0:
                plt.plot(convergence)
                plt.yscale('log')
                plt.show()
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
        
        # if count>100:
        #     plt.plot(np.abs(x_new/x_old - 1))
        #     plt.show()
        
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = 100)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init

def fixed_point_solver_double_diff_double_delta(p, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False, keep_l_R_fixed=False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params()
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var_double_diff_double_delta.var_from_vector(x_old,p,context=context,compute=False)
        if count == 0 and keep_l_R_fixed:
            l_R_0 = init.l_R[...,1:].ravel().copy()
        init.compute_solver_quantities(p)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        if keep_l_R_fixed:
            l_R = l_R_0.copy()
        
        # x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)
        x_new = np.concatenate((w/init.price_indices[0],Z/init.price_indices[0],l_R,profit,phi*init.price_indices[0]), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            # if count%5==0:
            #     plt.plot(convergence)
            #     plt.yscale('log')
            #     plt.show()
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
        
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init


def fixed_point_solver_with_entry_costs_cf(p, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
                       accelerate = False, safe_convergence=0.001,accelerate_when_stable=True, 
                       plot_cobweb = False, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = False, apply_bound_zero = False, 
                        apply_bound_research_labor = False, keep_l_R_fixed=False,
                       accel_memory = 50, accel_type1=True, accel_regularization=1e-10,
                       accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=False,damping_post_acceleration=5):   
    if x0 is None:
        x0 = p.guess_from_params(for_solver_with_entry_costs=True)
    x_old = x0 
        
    condition = True
    count = 0
    convergence = []
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    
    
    while condition and count < max_count and np.all(x_old<1e40): 
        # print(count)
        
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        init = var_with_entry_costs.var_from_vector_no_price_indices(x_old,p,context='counterfactual',compute=False)
        if count == 0 and keep_l_R_fixed:
            l_R_0 = init.l_R[...,1:].ravel().copy()
            
        init.compute_solver_quantities(p)
        init.price_indices = init.compute_price_indices(p)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        
        if keep_l_R_fixed:
            l_R = l_R_0.copy()
            
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        if plot_live and count>500 and count%500 == 0:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            # norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
            if count%1000==0:
                # plt.plot(convergence)
                # plt.yscale('log')
                # plt.show()
                # print(['w','Z','profit','l_R','phi'][np.argmax(np.array(conditions))],
                #       np.max(np.array([np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty])
                #                        for qty in ['w','Z','profit','l_R','phi']])
                #              )
                #       )
                # conv_bottleneck = ['w','Z','profit','l_R','phi'][np.argmax(np.array(conditions))]
                conv_bottleneck = 'l_R'
                conv_bottlneck_old =  x_old_decomp[conv_bottleneck]
                conv_bottlneck_new =  x_new_decomp[conv_bottleneck]
                max_index = np.argmax(np.abs(conv_bottlneck_new-conv_bottlneck_old)/conv_bottlneck_old)
                print(conv_bottleneck, conv_bottlneck_old[max_index],(conv_bottlneck_new-conv_bottlneck_old)[max_index] )
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
        
        # if count>100:
        #     conv_bottleneck = 'l_R'
        #     conv_bottlneck_old =  x_old_decomp[conv_bottleneck]
        #     conv_bottlneck_new =  x_new_decomp[conv_bottleneck]
        #     max_index = np.argmax(np.abs(conv_bottlneck_new-conv_bottlneck_old)/conv_bottlneck_old)
        #     max_index = 5
        #     print(conv_bottleneck, conv_bottlneck_old[max_index],(conv_bottlneck_new-conv_bottlneck_old)[max_index] )
        #     norm.append( conv_bottlneck_old[max_index] )
        #     # plt.plot((l_R.squeeze()/init.l_R[...,1:].squeeze() - 1))
        #     # plt.plot((Z.squeeze()/init.Z.squeeze() - 1))
        #     # plt.plot((Z.squeeze()/init.Z.squeeze() - 1))
        #     plt.plot(np.arange(len(norm)),norm)
        #     plt.yscale('log')
        #     # plt.xlim(len(norm)-100,len(norm)+1)
            # plt.show()
        
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40) and np.all(x_new > 0):
        status = 'successful'
    else:
        status = 'failed'
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = 100)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init

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
    
def dyn_fixed_point_solver(p, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-10, 
                           damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = True,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):  
    # print('called')
    # print(p.delta[:,1])
    if sol_fin is None:
        sol, sol_fin = fixed_point_solver(p,x0=p.guess,
                                        context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1000,
                                accel_memory =50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        sol_fin.scale_P(p)
        sol_fin.compute_non_solver_quantities(p) 
    
    dyn_var = dynamic_var(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin,
                          N=p.N)
    dyn_var.initiate_state_variables_0(sol_init)
    
    psis_guess = guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin)
    
    dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
                    'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
                    'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
                    'PSI_CD':psis_guess['PSI_CD'],
                    'PSI_MNP':psis_guess['PSI_MNP'],
                    'PSI_MPND':psis_guess['PSI_MPND'],
                    'PSI_MPD':psis_guess['PSI_MPD'],
                    # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
                    'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
                    # 'V_P':repeat_for_all_times(sol_fin.V_P,dyn_var.Nt)[...,1:,:],
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
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
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

        if plot_live:
            if count == 0:
                dyn_var.plot_country(0,title = str(count),initial = True)
        if plot_live:
            if count<70 and count >0:
                dyn_var.plot_country(0,title = str(count))
                
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
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'

    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40):
        status = 'successful'
    else:
        status = 'failed'
        
    if status == 'failed':
        print('Failed, report :')
        print('count',count)
        print('nans',np.isnan(x_new).sum())
        print('diverged',(x_new>1e40).sum())
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, dyn_var


def dyn_fixed_point_solver_exog_lr(p, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-10, 
                           damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = True,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):  
    # print('called')
    # print(p.delta[:,1])
    if sol_fin is None:
        sol, sol_fin = fixed_point_solver(p,x0=p.guess,
                                        context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1000,
                                accel_memory =50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        sol_fin.scale_P(p)
        sol_fin.compute_non_solver_quantities(p) 
    
    dyn_var = dynamic_var(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin,
                          N=p.N)
    dyn_var.initiate_state_variables_0(sol_init)
    
    psis_guess = guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin)
    
    dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
                    'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
                    'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
                    'PSI_CD':psis_guess['PSI_CD'],
                    'PSI_MNP':psis_guess['PSI_MNP'],
                    'PSI_MPND':psis_guess['PSI_MPND'],
                    'PSI_MPD':psis_guess['PSI_MPD'],
                    # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
                    'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
                    # 'V_P':repeat_for_all_times(sol_fin.V_P,dyn_var.Nt)[...,1:,:],
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
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
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

        if plot_live:
            if count == 0:
                dyn_var.plot_country(0,title = str(count),initial = True)
        if plot_live:
            if count<70 and count >0:
                dyn_var.plot_country(0,title = str(count))
        
        dyn_var.l_R[...,1:,:] = repeat_for_all_times(sol_init.l_R,dyn_var.Nt)[...,1:,:]
        dyn_var.compute_solver_quantities(p,exog_lr=True)

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
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'

    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40):
        status = 'successful'
    else:
        status = 'failed'
        
    if status == 'failed':
        print('Failed, report :')
        print('count',count)
        print('nans',np.isnan(x_new).sum())
        print('diverged',(x_new>1e40).sum())
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, dyn_var

def guess_PSIS_from_sol_init_and_sol_fin_double_diff_double_delta(dyn_var,sol_init,sol_fin,C=20):
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
    guess['PSI_CL'] = build_guess(sol_fin.PSI_CL,dyn_var.PSI_CL_0)
    guess['PSI_CD'] = build_guess(sol_fin.PSI_CD,dyn_var.PSI_CD_0)
    guess['PSI_MNP'] = build_guess(sol_fin.PSI_MNP,dyn_var.PSI_MNP_0)
    guess['PSI_MPND'] = build_guess(sol_fin.PSI_MPND,dyn_var.PSI_MPND_0)
    guess['PSI_MPL'] = build_guess(sol_fin.PSI_MPL,dyn_var.PSI_MPL_0)
    guess['PSI_MPD'] = build_guess(sol_fin.PSI_MPD,dyn_var.PSI_MPD_0)
    return guess

def dyn_fixed_point_solver_double_diff_double_delta(p, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-10, 
                           damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = True,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):  
    # print('called')
    # print(p.delta[:,1])
    if sol_fin is None:
        sol, sol_fin = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                        context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1000,
                                accel_memory =50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        sol_fin.scale_P(p)
        sol_fin.compute_non_solver_quantities(p) 
    
    dyn_var = dynamic_var_double_diff_double_delta(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin,
                          N=p.N)
    dyn_var.initiate_state_variables_0(sol_init)
    
    psis_guess = guess_PSIS_from_sol_init_and_sol_fin_double_diff_double_delta(dyn_var,sol_init,sol_fin)
    
    dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
                    'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
                    'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
                    'PSI_CL':psis_guess['PSI_CL'],
                    'PSI_CD':psis_guess['PSI_CD'],
                    'PSI_MNP':psis_guess['PSI_MNP'],
                    'PSI_MPND':psis_guess['PSI_MPND'],
                    'PSI_MPL':psis_guess['PSI_MPL'],
                    'PSI_MPD':psis_guess['PSI_MPD'],
                    # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
                    # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
                    'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
                    # 'V_P':repeat_for_all_times(sol_fin.V_P,dyn_var.Nt)[...,1:,:],
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
    hit_the_bound_count = 0
    if plot_cobweb:
        history_old = []
        history_new = []
    x_new = None
    aa_options = {'dim': len(x_old),
                'mem': accel_memory,
                'type1': accel_type1,
                'regularization': accel_regularization,
                'relaxation': accel_relaxation,
                'safeguard_factor': accel_safeguard_factor,
                'max_weight_norm': accel_max_weight_norm}
    aa_wrk = aa.AndersonAccelerator(**aa_options)
    start = time.perf_counter()
    cob = cobweb(cobweb_qty)
    if plot_convergence:
        norm = []
    damping = damping
    
    while condition and count < max_count and np.all(x_old<1e40): 
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
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

        if plot_live:
            if count == 0:
                dyn_var.plot_country(0,title = str(count),initial = True)
        if plot_live:
            if count<70 and count >0:
                dyn_var.plot_country(0,title = str(count))
                
        dyn_var.compute_solver_quantities(p)
        
        numeraire = 1/(dyn_var.price_indices[0,-1]/dyn_var.compute_price_indices(p)[0,-1])
        
        x_new = np.concatenate([
            dyn_var.compute_price_indices(p).ravel(),
            dyn_var.compute_wage(p).ravel(),
            dyn_var.compute_expenditure(p).ravel(),
            dyn_var.compute_PSI_CL(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_CD(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MNP(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPND(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPL(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPD(p)[...,1:,:].ravel(),
            dyn_var.compute_V_PD(p)[...,1:,:].ravel(),
            dyn_var.compute_DELTA_V(p)[...,1:,:].ravel(),
            dyn_var.compute_V_NP(p)[...,1:,:].ravel(),
            ],axis=0)
        # x_new = np.concatenate([
        #     dyn_var.compute_price_indices(p).ravel()/numeraire,
        #     dyn_var.compute_wage(p).ravel()/numeraire,
        #     dyn_var.compute_expenditure(p).ravel()/numeraire,
        #     dyn_var.compute_PSI_CL(p)[...,1:,:].ravel(),
        #     dyn_var.compute_PSI_CD(p)[...,1:,:].ravel(),
        #     dyn_var.compute_PSI_MNP(p)[...,1:,:].ravel(),
        #     dyn_var.compute_PSI_MPND(p)[...,1:,:].ravel(),
        #     dyn_var.compute_PSI_MPL(p)[...,1:,:].ravel(),
        #     dyn_var.compute_PSI_MPD(p)[...,1:,:].ravel(),
        #     dyn_var.compute_V_PD(p)[...,1:,:].ravel()/numeraire,
        #     dyn_var.compute_DELTA_V(p)[...,1:,:].ravel()/numeraire,
        #     dyn_var.compute_V_NP(p)[...,1:,:].ravel()/numeraire,
        #     ],axis=0)
        
        # print(dyn_var.compute_price_indices(p).ravel()/dyn_var.price_indices.ravel(),'prices')
        # print(dyn_var.compute_wage(p).ravel()/dyn_var.w.ravel(),'wages')
        # print(dyn_var.compute_expenditure(p).ravel()/dyn_var.Z.ravel(),'expenditure')
        # print(dyn_var.compute_PSI_MPL(p)[...,1:,:].ravel(),dyn_var.PSI_MPL[...,1:,:].ravel(),'PSI_MPL')
        # print((dyn_var.compute_V_NP(p)[...,1:,:].ravel()/dyn_var.V_NP[...,1:,:].ravel()).min(),'V_NP')
        
        # time.sleep(5)
        
        condition = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old) > tol
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        
        count += 1
        if np.all(np.array(convergence[-5:])<safe_convergence):
            if accelerate_when_stable:
                accelerate = True
                damping = damping_post_acceleration
                
        if plot_convergence:
            norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
        if plot_cobweb:
            history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
            history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
    finish = time.perf_counter()
    solving_time = finish-start
    dev_norm = 'TODO'

    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40):
        status = 'successful'
    else:
        status = 'failed'
        
    if status == 'failed':
        print('Failed, report :')
        print('count',count)
        print('nans',np.isnan(x_new).sum())
        print('diverged',(x_new>1e40).sum())
    
    x_sol = x_new
        
    sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
                   status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
    if disp_summary:
        sol_inst.run_summary()
    
    if plot_cobweb:
        cob = cobweb(cobweb_qty)
        for i,c in enumerate(convergence):
            cob.append_old_new(history_old[i],history_new[i])
            if cobweb_anim:
                cob.plot(count=i, window = 5000,pause = 0.01) 
        cob.plot(count = count, window = None)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, dyn_var

# def dyn_fixed_point_solver_with_exog_pat_and_rd(p, p_old, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-10, 
#                            damping = 10, max_count=1e6,
#                        accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
#                        plot_cobweb = True, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
#                        cobweb_coord = 1, plot_convergence = True,
#                        accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
#                        accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
#                        disp_summary=True,damping_post_acceleration=5):  
    
#     if sol_fin is None:
#         sol, sol_fin = fixed_point_solver(p,x0=p.guess,
#                                         context = 'counterfactual',
#                                 cobweb_anim=False,tol =1e-14,
#                                 accelerate=False,
#                                 accelerate_when_stable=True,
#                                 cobweb_qty='l_R',
#                                 plot_convergence=False,
#                                 plot_cobweb=False,
#                                 safe_convergence=0.001,
#                                 disp_summary=False,
#                                 damping = 10,
#                                 max_count = 1000,
#                                 accel_memory =50, 
#                                 accel_type1=True, 
#                                 accel_regularization=1e-10,
#                                 accel_relaxation=0.5, 
#                                 accel_safeguard_factor=1, 
#                                 accel_max_weight_norm=1e6,
#                                 damping_post_acceleration=10
#                                 )
#         sol_fin.scale_P(p)
#         sol_fin.compute_non_solver_quantities(p) 
    
#     dyn_var = dynamic_var(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin,
#                           N=p.N)
#     dyn_var.initiate_state_variables_0(sol_init)
    
#     psis_guess = guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin)
    
#     dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
#                     'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
#                     'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
#                     'PSI_CD':psis_guess['PSI_CD'],
#                     'PSI_MNP':psis_guess['PSI_MNP'],
#                     'PSI_MPND':psis_guess['PSI_MPND'],
#                     'PSI_MPD':psis_guess['PSI_MPD'],
#                     # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
#                     # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
#                     # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
#                     # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
#                     'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
#                     # 'V_P':repeat_for_all_times(sol_fin.V_P,dyn_var.Nt)[...,1:,:],
#                     'V_NP':repeat_for_all_times(sol_fin.V_NP,dyn_var.Nt)[...,1:,:],
#                     'DELTA_V':repeat_for_all_times(sol_fin.V_P-sol_fin.V_NP,dyn_var.Nt)[...,1:,:]
#                     }

#     dyn_var.guess_from_dic(dic_of_guesses)

#     if x0 is not None:
#         dyn_var.guess_from_vector(x0)
    
#     x_old = dyn_var.vector_from_var()
        
#     condition = True
#     count = 0
#     convergence = []
#     hit_the_bound_count = 0
#     if plot_cobweb:
#         history_old = []
#         history_new = []
#     x_new = None
#     aa_options = {'dim': len(x_old),
#                 'mem': accel_memory,
#                 'type1': accel_type1,
#                 'regularization': accel_regularization,
#                 'relaxation': accel_relaxation,
#                 'safeguard_factor': accel_safeguard_factor,
#                 'max_weight_norm': accel_max_weight_norm}
#     aa_wrk = aa.AndersonAccelerator(**aa_options)
#     start = time.perf_counter()
#     cob = cobweb(cobweb_qty)
#     if plot_convergence:
#         norm = []
#     damping = damping
    
#     while condition and count < max_count and np.all(x_old<1e40): 
#         if count != 0:
#             if accelerate:
#                 aa_wrk.apply(x_new, x_old)
#             x_old = (x_new+(damping-1)*x_old)/damping
#             dyn_var.guess_from_vector(x_old)

#             numeraire = dyn_var.price_indices[0,:]
#             for qty in ['price_indices','w','Z']:
#                 temp = getattr(dyn_var,qty)
#                 temp = temp/numeraire[None,:]
#                 setattr(dyn_var,qty,temp)
                
#             for qty in ['V_PD','DELTA_V','V_NP']:
#                 temp = getattr(dyn_var,qty)
#                 temp = temp/numeraire[None,None,None:]
#                 setattr(dyn_var,qty,temp)
                
#             x_old = dyn_var.vector_from_var()

#         if plot_live:
#             if count == 0:
#                 dyn_var.plot_country(0,title = str(count),initial = True)
#         if plot_live:
#             if count<70 and count >0:
#                 dyn_var.plot_country(0,title = str(count))
#         dyn_var.compute_solver_quantities(p)

#         x_new = np.concatenate([
#             dyn_var.compute_price_indices(p).ravel(),
#             dyn_var.compute_wage(p).ravel(),
#             dyn_var.compute_expenditure(p).ravel(),
#             dyn_var.compute_PSI_CD(p)[...,1:,:].ravel(),
#             dyn_var.compute_PSI_MNP(p)[...,1:,:].ravel(),
#             dyn_var.compute_PSI_MPND(p)[...,1:,:].ravel(),
#             dyn_var.compute_PSI_MPD(p)[...,1:,:].ravel(),
#             dyn_var.compute_V_PD(p)[...,1:,:].ravel(),
#             dyn_var.compute_DELTA_V(p)[...,1:,:].ravel(),
#             dyn_var.compute_V_NP(p)[...,1:,:].ravel(),
#             ],axis=0)

#         condition = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old) > tol
#         convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        
#         count += 1
#         if np.all(np.array(convergence[-5:])<safe_convergence):
#             if accelerate_when_stable:
#                 accelerate = True
#                 damping = damping_post_acceleration
                
#         if plot_convergence:
#             norm.append( (get_vec_qty(x_new,p)[cobweb_qty]).mean() )
#         if plot_cobweb:
#             history_old.append(get_vec_qty(x_old,p)[cobweb_qty].mean())
#             history_new.append(get_vec_qty(x_new,p)[cobweb_qty].mean())
    
#     finish = time.perf_counter()
#     solving_time = finish-start
#     dev_norm = 'TODO'

#     if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40):
#         status = 'successful'
#     else:
#         status = 'failed'
        
#     if status == 'failed':
#         print('Failed, report :')
#         print('count',count)
#         print('nans',np.isnan(x_new).sum())
#         print('diverged',(x_new>1e40).sum())
    
#     x_sol = x_new
        
#     sol_inst = sol_class(x_sol, p, solving_time=solving_time, iterations=count, deviation_norm=dev_norm, 
#                    status=status, hit_the_bound_count=hit_the_bound_count, x0=x0, tol = tol)
        
#     if disp_summary:
#         sol_inst.run_summary()
    
#     if plot_cobweb:
#         cob = cobweb(cobweb_qty)
#         for i,c in enumerate(convergence):
#             cob.append_old_new(history_old[i],history_new[i])
#             if cobweb_anim:
#                 cob.plot(count=i, window = 5000,pause = 0.01) 
#         cob.plot(count = count, window = None)
            
#     if plot_convergence:
#         plt.semilogy(convergence, label = 'convergence')
#         plt.semilogy(norm, label = 'norm')
#         plt.legend()
#         plt.show()
#     return sol_inst, dyn_var

def compute_deriv_welfare_to_patent_protec_US(sol_baseline,p,v0=None):
    epsilon = 1e-2
    back_up_delta = p.delta[0,1]
    p.delta[0,1] = p.delta[0,1]*(1+epsilon)
    sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-14,
                                    context = 'counterfactual',
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=2e3,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=5
                                  )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    p.delta[0,1] = back_up_delta
    
    return (sol_c.cons_eq_welfare[0]-1)/epsilon

def compute_deriv_welfare_to_patent_protec(sol_baseline,p_init,
                                        country = 'USA',
                                        dynamics=False):
    epsilon = 1e-2
    country_index = p_init.countries.index(country)
    back_up_delta = p_init.delta[country_index,1]
    p = p_init.copy()
    p.delta[country_index,1] = p.delta[country_index,1]*(1+epsilon)
    sol, sol_c = fixed_point_solver(p,x0=p.guess,tol=1e-14,
                                    context = 'counterfactual',
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=2e3,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=5
                                  )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    res = ((sol_c.cons_eq_welfare-1)/(epsilon*back_up_delta)).tolist(
        )+[(sol_c.cons_eq_pop_average_welfare_change-1)/(epsilon*back_up_delta)
           ,(sol_c.cons_eq_negishi_welfare_change-1)/(epsilon*back_up_delta)]
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_fin = sol_c, sol_init=sol_baseline,
                               Nt=23,
                               t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        res = ((dyn_sol_c.cons_eq_welfare-1)/(epsilon*back_up_delta)).tolist(
            )+[(dyn_sol_c.cons_eq_pop_average_welfare_change-1)/(epsilon*back_up_delta),
               (dyn_sol_c.cons_eq_negishi_welfare_change-1)/(epsilon*back_up_delta)]
        
    p.delta[0,1] = back_up_delta
    
    return res

def compute_deriv_growth_to_patent_protec_US(sol_baseline,p,v0=None):
    epsilon = 1e-2
    back_up_delta = p.delta[0,1]
    p.delta[0,1] = p.delta[0,1]*(1+epsilon)
    sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-14,
                                    context = 'counterfactual',
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=2e3,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=5
                                  )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    p.delta[0,1] = back_up_delta
    
    return (sol_c.g-sol_baseline.g)/epsilon

def calibration_func(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    # print(p.make_p_vector())
    p.update_parameters(vec_parameters)
    if p.fix_fe_across_sectors:
        p.fe[2:] = p.fe[1]
        # print(p.fe[1:])
    # if p.S>2:
    #     for s in range(2,p.S):
    #         p.delta[:,s] = p.delta[:,1]
    # print('min delta',p.delta.min())
    if 'khi' in p.calib_parameters:
        p.update_khi_and_r_hjort(p.khi)
    try:
        v0 = p.guess
    except:
        pass
    
    sol, sol_c = fixed_point_solver(p,context = 'calibration', x0=v0,
                            cobweb_anim=False,tol =1e-10, #!!!! to change back to 14
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping =2,
                            max_count = 1000,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=1
                            )
    # sol, sol_c = fixed_point_solver(p,context = 'calibration', x0=v0,
    #                         cobweb_anim=False,tol =1e-14,
    #                         accelerate=False,
    #                         accelerate_when_stable=True,
    #                         cobweb_qty='l_R',
    #                         plot_convergence=False,
    #                         plot_cobweb=False,
    #                         safe_convergence=0.1,
    #                         disp_summary=False,
    #                         damping =2,
    #                         max_count = 1000,
    #                         accel_memory = 50, 
    #                         accel_type1=True, 
    #                         accel_regularization=1e-10,
    #                         accel_relaxation=0.5, 
    #                         accel_safeguard_factor=1, 
    #                         accel_max_weight_norm=1e6,
    #                         damping_post_acceleration=1
    #                         )

    if sol.status == 'failed': 
        print('trying safer')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.001,
                                  max_count=2e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=2
                                  )
    if sol.status == 'failed': 
        print('trying strong damp')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=5e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=10
                                  )
    if sol.status == 'failed': 
        print('trying less precise')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=v0,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with standard guess')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=None,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying longer convergence')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with no acceleration')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=False,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    m.compute_moments_deviations()
    if hist is not None:
        if hist.count%1 == 0:
            hist_dic = {mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments}
            hist_dic['objective'] = np.linalg.norm(m.deviation_vector())
            hist.append(**hist_dic)
            hist.time = time.perf_counter() - start_time
        if hist.count%100 == 0:
            hist.plot()
        if hist.count%200==0:
            print('fe : ',p.fe[1:],'fo : ',p.fo[1:], 'delta : ', p.delta[:,1:]
                  , 'nu : ', p.nu[1:], 'k :', p.k
                  , 'theta :', p.theta[1:], 'sigma :', p.sigma[1:], 'zeta :', p.zeta[1:]
                  , 'rho :', p.rho, 'kappa :', p.kappa,
                  # 'd : ', p.d, 'r_hjort : ', p.r_hjort, 'nu_tilde : ', p.nu_tilde[1]
                  )
    hist.count += 1
    # print(hist.count)
    p.guess = sol_c.vector_from_var()
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        return np.full_like(m.deviation_vector(),1e10)
    else:
        return m.deviation_vector() 
    

def calibration_func_with_entry_costs(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    p.update_parameters(vec_parameters)
    if 'khi' in p.calib_parameters:
        p.update_khi_and_r_hjort(p.khi)
    try:
        v0 = p.guess
    except:
        pass
    sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration', x0=v0,
                            cobweb_anim=False,tol =1e-12,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping =2,
                            max_count = 1000,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=1
                            )
    
    if sol.status == 'failed': 
        print('trying safer')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.001,
                                  max_count=2e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=2
                                  )
    if sol.status == 'failed': 
        print('trying strong damp')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=5e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=10
                                  )
    if sol.status == 'failed': 
        print('trying less precise')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',
                                                         x0=v0,tol=1e-6,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with standard guess')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',x0=None,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying longer convergence')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with no acceleration')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=False,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    m.compute_PROBINNOVENT(sol_c, p)
    m.compute_moments_deviations()
    if hist is not None:
        if hist.count%1 == 0:
            hist_dic = {mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments}
            hist_dic['objective'] = np.linalg.norm(m.deviation_vector())
            hist.append(**hist_dic)
            hist.time = time.perf_counter() - start_time
        if hist.count%100 == 0:
            hist.plot()
        if hist.count%100==0:
            print('fe : ',p.fe[1],'fo : ',p.fo[1], 'delta : ', p.delta[:,1]
                  , 'nu : ', p.nu[1], 'nu_tilde : ', p.nu_tilde[1], 'k :', p.k
                  , 'theta :', p.theta[1], 'sigma :', p.sigma[1], 'zeta :', p.zeta[1]
                  , 'rho :', p.rho, 'kappa :', p.kappa, 'd : ', p.d, 'r_hjort : ', p.r_hjort,
                  'a :', p.a)
    hist.count += 1
    # print(hist.count)
    p.guess = sol_c.vector_from_var()
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        return np.full_like(m.deviation_vector(),1e10)
    else:
        return m.deviation_vector() 
    
def calibration_func_double_diff_double_delta(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    p.update_parameters(vec_parameters)
    p.delta_dom = p.delta_int.copy()
    # p.delta_int = p.delta_dom.copy()
    # p.delta_dom.ravel()[np.s_[np.r_[0:7, 8:p.N*p.S]]] = p.delta_int.ravel()[np.s_[np.r_[0:7, 8:p.N*p.S]]]
    p.update_delta_eff()
    # p.nu_tilde = p.nu.copy()
    if 'khi' in p.calib_parameters:
        p.update_khi_and_r_hjort(p.khi)
    try:
        v0 = p.guess
    except:
        pass
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration', x0=v0,
                            cobweb_anim=False,tol =1e-12,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping =2,
                            max_count = 1000,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=1
                            )
    
    if sol.status == 'failed': 
        print('trying safer')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.001,
                                  max_count=2e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=2
                                  )
    if sol.status == 'failed': 
        print('trying strong damp')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',x0=v0,tol=1e-14,
                                  accelerate=False,
                                  accelerate_when_stable=True,
                                  plot_cobweb=False,
                                  plot_convergence=False,
                                  cobweb_qty='phi',
                                  disp_summary=False,
                                  safe_convergence=0.1,
                                  max_count=5e3,
                                  damping = 10,
                                  accel_memory = 50, 
                                  accel_type1=True, 
                                  accel_regularization=1e-10,
                                  accel_relaxation=0.5, 
                                  accel_safeguard_factor=1, 
                                  accel_max_weight_norm=1e6,
                                  damping_post_acceleration=10
                                  )
    if sol.status == 'failed': 
        print('trying less precise')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',
                                                         x0=v0,tol=1e-6,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with standard guess')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',x0=None,tol=1e-12,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.001,
                                      max_count=2e3,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying longer convergence')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=True,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    if sol.status == 'failed':
        print('trying with no acceleration')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,context = 'calibration',x0=v0,tol=1e-13,
                                      accelerate=False,
                                      accelerate_when_stable=False,
                                      plot_cobweb=False,
                                      plot_convergence=False,
                                      cobweb_qty='phi',
                                      disp_summary=False,
                                      safe_convergence=0.1,
                                      max_count=5e4,
                                      accel_memory = 50, 
                                      accel_type1=True, 
                                      accel_regularization=1e-10,
                                      accel_relaxation=0.5, 
                                      accel_safeguard_factor=1, 
                                      accel_max_weight_norm=1e6,
                                      damping_post_acceleration=5
                                      )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c,p)
    m.compute_moments_deviations()
    if hist is not None:
        if hist.count%1 == 0:
            hist_dic = {mom : np.linalg.norm(getattr(m,mom+'_deviation')) for mom in m.list_of_moments}
            hist_dic['objective'] = np.linalg.norm(m.deviation_vector())
            hist.append(**hist_dic)
            hist.time = time.perf_counter() - start_time
        if hist.count%100 == 0:
            hist.plot()
        if hist.count%200==0:
            print('fe : ',p.fe[1],'fo : ',p.fo[1], 'delta_dom : ', p.delta_dom[:,1], 'delta_int : ', p.delta_int[:,1]
                  , 'nu : ', p.nu[1], 'nu_tilde : ', p.nu_tilde[1], 'k :', p.k
                  , 'theta :', p.theta[1], 'sigma :', p.sigma[1], 'zeta :', p.zeta[1]
                  , 'rho :', p.rho, 'kappa :', p.kappa, 'd : ', p.d, 'r_hjort : ', p.r_hjort,
                  'a :', p.a)
    hist.count += 1
    # print(hist.count)
    p.guess = sol_c.vector_from_var()
    
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        return np.full_like(m.deviation_vector(),1e10)
    else:
        return m.deviation_vector() 

#%% compute nash equilibrium for deltas

def minus_welfare_of_delta(delta,p,c,sol_it_baseline,sector=1, hist = None,
                           dynamics=False):
    back_up_delta_value = p.delta[p.countries.index(c),sector]
    p.delta[p.countries.index(c),sector] = delta
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            ) 
    if sol.status != 'successful':
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status != 'successful':
            print(p.delta,'failed2')
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    
    welfare = -sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_fin = sol_c, sol_init=sol_it_baseline,Nt=21,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        welfare = -dyn_sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if hist is not None:
        fig, ax = plt.subplots(figsize=(16,12))
        hist.delta.append(p.delta[p.countries.index(c),sector])
        hist.welfare.append(welfare)
        ax.scatter(np.log(hist.delta),np.log(-np.array(hist.welfare)),color='grey')
        ax.scatter(np.log(hist.expected_deltas),np.log(-np.array(hist.expected_welfare)),color='grey',label='expected change')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.expected_deltas)[i],np.log(-hist.expected_welfare[i])),color='grey')
        ax.scatter(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1]), color = 'red',label='search optimization')
        ax.annotate(c,(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1])),color='red')
        ax.scatter(np.log(hist.current_deltas),np.log(-hist.current_welfare), color= 'blue',label='current state of the world')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.current_deltas)[i],np.log(-hist.current_welfare[i])))
        plt.xlabel('Delta (log)')
        plt.ylabel('Consumption equivalent welfare change (log)')
        plt.legend()
        plt.show()
        if hist.make_a_pause:
            # input("Press Enter to run next iteration")
            hist.make_a_pause = False
    p.delta[p.countries.index(c),sector] = back_up_delta_value
    p.guess = sol_c.vector_from_var()
    
    print(delta,c,welfare)
    
    return welfare

def minimize_delta(args):
    p, c, sol_it_baseline, hist_nash, dynamics, bounds = args
    if dynamics:
        delta_min = optimize.shgo(func=minus_welfare_of_delta,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
                                            method='bounded',
                                            bounds=bounds,
                                            args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                            tol=1e-15
                                            )
    return delta_min.x

def compute_new_deltas_fixed_point(p, sol_it_baseline, lb_delta, ub_delta, hist_nash = None, 
                                    dynamics=False,max_workers=12,parallel=True):
    
    bounds=(lb_delta, ub_delta)
    
    if not parallel:
    # monoprocess
        # new_deltas = np.zeros(len(p.countries))
        new_deltas = np.zeros((len(p.countries),p.S-1))
        for sector in range(1,p.S):
            for i,c in enumerate(p.countries):
                if dynamics:
                    # print('doing that')
                    delta_min = optimize.shgo(func=minus_welfare_of_delta,
                                                          # sampling_method='halton',
                                                          bounds=[bounds],
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                                                          minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
                                                          # options = dict(ftol=1e-8)
                                                          )
                else:
                    delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
                                                          method='bounded',
                                                            bounds=bounds,
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          # options={'disp':3},
                                                          tol=1e-15
                                                          )
        
                new_deltas[i,sector-1] = delta_min.x
            if hist_nash is not None:
                hist_nash.expected_deltas[i] = new_deltas[i,sector-1]
                hist_nash.expected_welfare[i] = delta_min.fun
            print(c,new_deltas)
    
    if parallel:
        
        args_list = [(p.copy(), c, sol_it_baseline.copy(), hist_nash, dynamics, bounds) for c in p.countries]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the function to be executed in parallel
            results = list(executor.map(minimize_delta, args_list))
    
        new_deltas = np.array(results).squeeze()
        print(new_deltas)
    
    if hist_nash is not None:
        hist_nash.expected_deltas = new_deltas
        # hist_nash.expected_welfare[i] = delta_min.fun
            
    if hist_nash is not None:
        # input("Press Enter to run next iteration")
        hist_nash.make_a_pause = True
                
    return new_deltas.ravel()

def find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',dynamics=False,
                 plot_convergence = False,solver_options=None,tol=5e-5,
                 damping = 1,plot_history = False,delta_init=None,max_workers=6,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    

    condition = True
    deltas = p_baseline.delta[...,1][:,None]
    welfares = np.ones(len(p_baseline.countries))[:,None]
    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    if delta_init is None:
        x_old = p_baseline.delta[...,1:].ravel()
    else:
        x_old = delta_init.ravel()
        p_it_baseline.delta[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
    convergence = []
    new_deltas = None
    
    if plot_history:
        hist_nash = history_nash()
        hist_nash.update_current_deltas(x_old)
        hist_nash.update_current_welfare(-np.ones(len(x_old)))
    else:
        hist_nash = None
        
    while condition:
        print(it)
        if it != 0:
            # print(type(new_deltas))
            # print(type(x_old))
            x_old = (new_deltas+(damping-1)*x_old)/damping
            p_it_baseline.delta[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
        
        sol, sol_it_baseline = fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_deltas = compute_new_deltas_fixed_point(p_it_baseline, sol_baseline, lb_delta, 
                                                        ub_delta, hist_nash = hist_nash,
                                                        dynamics=dynamics,parallel=parallel,
                                                        max_workers=max_workers)
            
        new_deltas[new_deltas>5] = 12
        p_it_baseline.delta[...,1:] = new_deltas.reshape(p_baseline.N,p_baseline.S-1)
        sol, sol_it= fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
            
        
        if plot_history:
            hist_nash.update_current_deltas(new_deltas)
            hist_nash.update_current_welfare(-sol_it.cons_eq_welfare)
        
        
        if dynamics:
            welfares = np.concatenate([welfares,dyn_sol_it.cons_eq_welfare[:,None]],axis=1)
        else:
            welfares = np.concatenate([welfares,sol_it.cons_eq_welfare[:,None]],axis=1)
        
        condition = np.linalg.norm((new_deltas-x_old)/x_old)> tol
        
        convergence.append(np.linalg.norm((new_deltas - x_old)/x_old))
        
        print(np.linalg.norm((new_deltas-x_old)/x_old))
        
        it += 1
        
        if it>5:
            damping = 5
        
        if plot_convergence:
            deltas = np.concatenate([deltas,new_deltas[:,None]],axis=1)
            fig,ax = plt.subplots()
            
            ax2 = ax.twinx()
            ax.semilogy(deltas.transpose())
            ax2.plot(welfares.transpose(), ls = '--')
            plt.legend(labels = p_baseline.countries)
            
            plt.show()

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it

#%% compute nash equilibrium for double deltas

def minus_welfare_of_delta_double_delta(delta,p,c,sol_it_baseline,sector=1, hist = None,
                           dynamics=False):
    back_up_delta_value = np.array([p.delta_dom[p.countries.index(c),sector],p.delta_int[p.countries.index(c),sector]])
    # print('here',delta)
    p.delta_dom[p.countries.index(c),sector] = delta[0]
    p.delta_int[p.countries.index(c),sector] = delta[1]
    p.update_delta_eff()
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            ) 
    if sol.status != 'successful':
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status != 'successful':
            print(p.delta,'failed2')
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    
    welfare = -sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if hist is not None:
        fig, ax = plt.subplots(figsize=(16,12))
        hist.delta.append(p.delta[p.countries.index(c),sector])
        hist.welfare.append(welfare)
        ax.scatter(np.log(hist.delta),np.log(-np.array(hist.welfare)),color='grey')
        ax.scatter(np.log(hist.expected_deltas),np.log(-np.array(hist.expected_welfare)),color='grey',label='expected change')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.expected_deltas)[i],np.log(-hist.expected_welfare[i])),color='grey')
        ax.scatter(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1]), color = 'red',label='search optimization')
        ax.annotate(c,(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1])),color='red')
        ax.scatter(np.log(hist.current_deltas),np.log(-hist.current_welfare), color= 'blue',label='current state of the world')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.current_deltas)[i],np.log(-hist.current_welfare[i])))
        plt.xlabel('Delta (log)')
        plt.ylabel('Consumption equivalent welfare change (log)')
        plt.legend()
        plt.show()
        if hist.make_a_pause:
            # input("Press Enter to run next iteration")
            hist.make_a_pause = False
    p.delta_dom[p.countries.index(c),sector] = back_up_delta_value[0]
    p.delta_int[p.countries.index(c),sector] = back_up_delta_value[1]
    p.update_delta_eff()
    p.guess = sol_c.vector_from_var()
    
    print(delta,c,welfare)
    
    return welfare

def minimize_delta_double_delta(args):
    p, c, sol_it_baseline, hist_nash, dynamics, bounds = args
    if dynamics:
        delta_min = optimize.shgo(func=minus_welfare_of_delta_double_delta,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta_double_delta,
                                            method='bounded',
                                            bounds=bounds,
                                            args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                            tol=1e-15
                                            )
    return delta_min.x

def compute_new_deltas_fixed_point_double_delta(p, sol_it_baseline, lb_delta, ub_delta, hist_nash = None, 
                                    dynamics=False,max_workers=12,parallel=True):
    
    bounds=(lb_delta, ub_delta)
    
    if not parallel:
    # monoprocess
        # new_deltas = np.zeros(len(p.countries))
        new_deltas_dom = np.zeros((len(p.countries),p.S-1))
        new_deltas_int = np.zeros((len(p.countries),p.S-1))
        for sector in range(1,p.S):
            for i,c in enumerate(p.countries):
                if dynamics:
                    # print('doing that')
                    delta_min = optimize.shgo(func=minus_welfare_of_delta_double_delta,
                                                          # sampling_method='halton',
                                                          bounds=[bounds,bounds],
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                                                          minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
                                                          # options = dict(ftol=1e-8)
                                                          )
                else:
                    # delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta_double_delta,
                    #                                       method='bounded',
                    #                                       bounds=[bounds,bounds],
                    #                                       args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                    #                                       # options={'disp':3},
                    #                                       tol=1e-15
                    #                                       )
                    delta_min = optimize.shgo(func=minus_welfare_of_delta_double_delta,
                                                          # method='bounded',
                                                          bounds=[bounds,bounds],
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          # options={'disp':3},
                                                          options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                                                          minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2},
                                                          # tol=1e-15
                                                          )
        
                new_deltas_dom[i,sector-1] = delta_min.x[0]
                new_deltas_int[i,sector-1] = delta_min.x[1]
            if hist_nash is not None:
                hist_nash.expected_deltas[i] = new_deltas_dom[i,sector-1]
                hist_nash.expected_welfare[i] = delta_min.fun
            print(c,new_deltas_dom,new_deltas_int)
                
    return np.concatenate([new_deltas_dom.ravel(), new_deltas_int.ravel()])

def find_nash_eq_double_delta(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',dynamics=False,
                 plot_convergence = False,solver_options=None,tol=5e-5,
                 damping = 1,plot_history = False,delta_init=None,max_workers=6,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
    
    sol, sol_baseline = fixed_point_solver_double_diff_double_delta(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    

    condition = True
    deltas = np.concatenate([p_baseline.delta_dom[...,1],p_baseline.delta_int[...,1]])
    welfares = np.ones(len(p_baseline.countries))[:,None]
    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    if delta_init is None:
        x_old = np.concatenate([p_baseline.delta_dom[...,1],p_baseline.delta_int[...,1]])
    else:
        x_old = delta_init.ravel()
        p_it_baseline.delta[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
    convergence = []
    new_deltas = None
    
    # if plot_history:
    #     hist_nash = history_nash()
    #     hist_nash.update_current_deltas(x_old)
    #     hist_nash.update_current_welfare(-np.ones(len(x_old)))
    # else:
    #     hist_nash = None
    hist_nash = None
        
    while condition:
        print(it)
        if it != 0:
            # print(type(new_deltas))
            # print(type(x_old))
            x_old = (new_deltas+(damping-1)*x_old)/damping
            p_it_baseline.delta_dom[...,1:] = x_old[:p_baseline.N].reshape(p_baseline.N,p_baseline.S-1)
            p_it_baseline.delta_int[...,1:] = x_old[p_baseline.N:].reshape(p_baseline.N,p_baseline.S-1)
            p_it_baseline.update_delta_eff()
        
        sol, sol_it_baseline = fixed_point_solver_double_diff_double_delta(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_deltas = compute_new_deltas_fixed_point_double_delta(p_it_baseline, sol_baseline, lb_delta, 
                                                        ub_delta, hist_nash = hist_nash,
                                                        dynamics=dynamics,parallel=parallel,
                                                        max_workers=max_workers)
            
        new_deltas[new_deltas>5] = 12
        # p_it_baseline.delta[...,1:] = new_deltas.reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.delta_dom[...,1:] = new_deltas[:p_baseline.N].reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.delta_int[...,1:] = new_deltas[p_baseline.N:].reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.update_delta_eff()
        sol, sol_it= fixed_point_solver_double_diff_double_delta(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
            
        
        if plot_history:
            hist_nash.update_current_deltas(new_deltas)
            hist_nash.update_current_welfare(-sol_it.cons_eq_welfare)
        
        
        if dynamics:
            welfares = np.concatenate([welfares,dyn_sol_it.cons_eq_welfare[:,None]],axis=1)
        else:
            welfares = np.concatenate([welfares,sol_it.cons_eq_welfare[:,None]],axis=1)
        
        condition = np.linalg.norm((new_deltas-x_old)/x_old)> tol
        
        convergence.append(np.linalg.norm((new_deltas - x_old)/x_old))
        
        print(np.linalg.norm((new_deltas-x_old)/x_old))
        
        it += 1
        
        if it>5:
            damping = 5
        
        if plot_convergence:
                deltas = np.concatenate([deltas,new_deltas[:,None]],axis=1)
                fig,ax = plt.subplots()
                
                ax2 = ax.twinx()
                ax.semilogy(deltas.transpose())
                ax2.plot(welfares.transpose(), ls = '--')
                plt.legend(labels = p_baseline.countries)
                
                plt.show()

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it

#%% compute nash equilibrium for double diffusion

def minus_welfare_of_delta_double_diffusion(delta,p,c,sol_it_baseline,sector=1, hist = None,
                           dynamics=False):
    back_up_delta_value = p.delta_dom[p.countries.index(c),sector].copy()
    # print('here',delta)
    p.delta_dom[p.countries.index(c),sector] = delta
    p.delta_int[p.countries.index(c),sector] = delta
    p.update_delta_eff()
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            ) 
    if sol.status != 'successful':
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status != 'successful':
            print(p.delta,'failed2')
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    
    welfare = -sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if hist is not None:
        fig, ax = plt.subplots(figsize=(16,12))
        hist.delta.append(p.delta[p.countries.index(c),sector])
        hist.welfare.append(welfare)
        ax.scatter(np.log(hist.delta),np.log(-np.array(hist.welfare)),color='grey')
        ax.scatter(np.log(hist.expected_deltas),np.log(-np.array(hist.expected_welfare)),color='grey',label='expected change')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.expected_deltas)[i],np.log(-hist.expected_welfare[i])),color='grey')
        ax.scatter(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1]), color = 'red',label='search optimization')
        ax.annotate(c,(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1])),color='red')
        ax.scatter(np.log(hist.current_deltas),np.log(-hist.current_welfare), color= 'blue',label='current state of the world')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.current_deltas)[i],np.log(-hist.current_welfare[i])))
        plt.xlabel('Delta (log)')
        plt.ylabel('Consumption equivalent welfare change (log)')
        plt.legend()
        plt.show()
        if hist.make_a_pause:
            # input("Press Enter to run next iteration")
            hist.make_a_pause = False
    p.delta_dom[p.countries.index(c),sector] = back_up_delta_value
    p.delta_int[p.countries.index(c),sector] = back_up_delta_value
    p.update_delta_eff()
    p.guess = sol_c.vector_from_var()
    
    print(delta,c,welfare)
    
    return welfare

def minimize_delta_double_diffusion(args):
    p, c, sol_it_baseline, hist_nash, dynamics, bounds = args
    if dynamics:
        delta_min = optimize.shgo(func=minus_welfare_of_delta_double_diffusion,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta_double_diffusion,
                                            method='bounded',
                                            bounds=bounds,
                                            args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                            tol=1e-15
                                            )
    return delta_min.x

def compute_new_deltas_fixed_point_double_diffusion(p, sol_it_baseline, lb_delta, ub_delta, hist_nash = None, 
                                    dynamics=False,max_workers=12,parallel=True):
    
    bounds=(lb_delta, ub_delta)
    
    if not parallel:
    # monoprocess
        # new_deltas = np.zeros(len(p.countries))
        new_deltas_dom = np.zeros((len(p.countries),p.S-1))
        new_deltas_int = np.zeros((len(p.countries),p.S-1))
        for sector in range(1,p.S):
            for i,c in enumerate(p.countries):
                if dynamics:
                    # print('doing that')
                    delta_min = optimize.shgo(func=minus_welfare_of_delta_double_diffusion,
                                                          # sampling_method='halton',
                                                          bounds=[bounds],
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                                                          minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
                                                          # options = dict(ftol=1e-8)
                                                          )
                else:
                    # delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta_double_delta,
                    #                                       method='bounded',
                    #                                       bounds=[bounds,bounds],
                    #                                       args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                    #                                       # options={'disp':3},
                    #                                       tol=1e-15
                    #                                       )
                    delta_min = optimize.shgo(func=minus_welfare_of_delta_double_diffusion,
                                                          # method='bounded',
                                                          bounds=[bounds],
                                                          args = (p,c,sol_it_baseline, sector, hist_nash, dynamics),
                                                          # options={'disp':3},
                                                          options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                                                          minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2},
                                                          # tol=1e-15
                                                          )
        
                new_deltas_dom[i,sector-1] = delta_min.x
                new_deltas_int[i,sector-1] = delta_min.x
            if hist_nash is not None:
                hist_nash.expected_deltas[i] = new_deltas_dom[i,sector-1]
                hist_nash.expected_welfare[i] = delta_min.fun
            print(c,new_deltas_dom,new_deltas_int)
                
    return new_deltas_dom.ravel()

def find_nash_eq_double_diffusion(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',dynamics=False,
                 plot_convergence = False,solver_options=None,tol=5e-5,
                 damping = 1,plot_history = False,delta_init=None,max_workers=6,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
    
    sol, sol_baseline = fixed_point_solver_double_diff_double_delta(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    

    condition = True
    deltas = np.concatenate([p_baseline.delta_dom[...,1],p_baseline.delta_int[...,1]])
    welfares = np.ones(len(p_baseline.countries))[:,None]
    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    if delta_init is None:
        x_old = p_baseline.delta_dom[...,1].copy()
    else:
        x_old = delta_init.ravel()
        p_it_baseline.delta[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
    convergence = []
    new_deltas = None
    
    # if plot_history:
    #     hist_nash = history_nash()
    #     hist_nash.update_current_deltas(x_old)
    #     hist_nash.update_current_welfare(-np.ones(len(x_old)))
    # else:
    #     hist_nash = None
    hist_nash = None
        
    while condition:
        print(it)
        if it != 0:
            # print(type(new_deltas))
            # print(type(x_old))
            x_old = (new_deltas+(damping-1)*x_old)/damping
            p_it_baseline.delta_dom[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
            p_it_baseline.delta_int[...,1:] = x_old.reshape(p_baseline.N,p_baseline.S-1)
            p_it_baseline.update_delta_eff()
        
        sol, sol_it_baseline = fixed_point_solver_double_diff_double_delta(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_deltas = compute_new_deltas_fixed_point_double_diffusion(p_it_baseline, sol_baseline, lb_delta, 
                                                        ub_delta, hist_nash = hist_nash,
                                                        dynamics=dynamics,parallel=parallel,
                                                        max_workers=max_workers)
            
        new_deltas[new_deltas>5] = 12
        # p_it_baseline.delta[...,1:] = new_deltas.reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.delta_dom[...,1:] = new_deltas.reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.delta_int[...,1:] = new_deltas.reshape(p_baseline.N,p_baseline.S-1)
        p_it_baseline.update_delta_eff()
        sol, sol_it= fixed_point_solver_double_diff_double_delta(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
            
        
        if plot_history:
            hist_nash.update_current_deltas(new_deltas)
            hist_nash.update_current_welfare(-sol_it.cons_eq_welfare)
        
        
        if dynamics:
            welfares = np.concatenate([welfares,dyn_sol_it.cons_eq_welfare[:,None]],axis=1)
        else:
            welfares = np.concatenate([welfares,sol_it.cons_eq_welfare[:,None]],axis=1)
        
        condition = np.linalg.norm((new_deltas-x_old)/x_old)> tol
        
        convergence.append(np.linalg.norm((new_deltas - x_old)/x_old))
        
        print(np.linalg.norm((new_deltas-x_old)/x_old))
        
        it += 1
        
        if it>5:
            damping = 5
        
        if plot_convergence:
                deltas = np.concatenate([deltas,new_deltas[:,None]],axis=1)
                fig,ax = plt.subplots()
                
                ax2 = ax.twinx()
                ax.semilogy(deltas.transpose())
                ax2.plot(welfares.transpose(), ls = '--')
                plt.legend(labels = p_baseline.countries)
                
                plt.show()

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it


#%% compute nash equilibrium for tariffs

def minus_welfare_of_tariff(country_tariff,p,c,sol_it_baseline, hist = None,
                           dynamics=False):

    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    
    back_up_tariff_value = p.tariff[p.countries.index(c),mask,1]
    p.tariff[p.countries.index(c),mask,1] = country_tariff
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            ) 
    if sol.status != 'successful':
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
        if sol.status != 'successful':
            print(p.delta,'failed2')
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    
    welfare = -sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_fin = sol_c, sol_init=sol_it_baseline,Nt=21,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        welfare = -dyn_sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if hist is not None:
        fig, ax = plt.subplots(figsize=(16,12))
        hist.delta.append(p.delta[p.countries.index(c),1])
        hist.welfare.append(welfare)
        ax.scatter(np.log(hist.delta),np.log(-np.array(hist.welfare)),color='grey')
        ax.scatter(np.log(hist.expected_deltas),np.log(-np.array(hist.expected_welfare)),color='grey',label='expected change')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.expected_deltas)[i],np.log(-hist.expected_welfare[i])),color='grey')
        ax.scatter(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1]), color = 'red',label='search optimization')
        ax.annotate(c,(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1])),color='red')
        ax.scatter(np.log(hist.current_deltas),np.log(-hist.current_welfare), color= 'blue',label='current state of the world')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.current_deltas)[i],np.log(-hist.current_welfare[i])))
        plt.xlabel('Delta (log)')
        plt.ylabel('Consumption equivalent welfare change (log)')
        plt.legend()
        plt.show()
        if hist.make_a_pause:
            # input("Press Enter to run next iteration")
            hist.make_a_pause = False

    p.tariff[p.countries.index(c),mask,1] = back_up_tariff_value
    p.guess = sol_c.vector_from_var()
    
    return welfare

def minimize_tariff(args):
    p, c, sol_it_baseline, hist_nash, dynamics, bounds = args
    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    if dynamics:
        #!!!! TODO dynamics
        tariff_min = optimize.shgo(func=minus_welfare_of_tariff,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        tariff_min = optimize.minimize(fun = minus_welfare_of_tariff,
                                x0 = p.tariff[p.countries.index(c),mask,1],
                                tol = 1e-14,
                                args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                options = {'disp':True,'f_tol':1e-14},
                                bounds=bounds)
        # optimize.minimize_scalar(fun=minus_welfare_of_tariff,
        #                                     method='bounded',
        #                                     bounds=bounds,
        #                                     args=(p, c, sol_it_baseline, hist_nash, dynamics),
        #                                     tol=1e-15
        #                                     )
    return tariff_min.x

def compute_new_tariff_fixed_point(p, sol_it_baseline, lb_tariff, ub_tariff, hist_nash = None, 
                                    dynamics=False,max_workers=6,parallel=True):
    
    bounds=[(lb_tariff, ub_tariff)]
    
    if not parallel:
    #!!! TODO (or delete) not parallel calculation
    # monoprocess
        new_tariff = np.zeros_like(p.tariff)
        
        for i,c in enumerate(p.countries):
            mask = np.ones(len(p.countries),bool)
            mask[p.countries.index(c)] = False
            if dynamics:
                pass
                # print('doing that')
                # delta_min = optimize.shgo(func=minus_welfare_of_delta,
                #                                       # sampling_method='halton',
                #                                       bounds=[bounds],
                #                                       args = (p,c,sol_it_baseline, hist_nash, dynamics),
                #                                       options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
                #                                       minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
                #                                       # options = dict(ftol=1e-8)
                #                                       )
            else:
                tariff_min = optimize.minimize(fun = minus_welfare_of_tariff,
                                        x0 = p.tariff[p.countries.index(c),mask,1],
                                        tol = 1e-14,
                                        args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                        options = {'disp':True,'f_tol':1e-14},
                                        bounds=bounds)
    
            new_tariff[i,mask,1] = tariff_min.x
    
    if parallel:
        
        args_list = [(p.copy(), c, sol_it_baseline.copy(), hist_nash, dynamics, bounds) for c in p.countries]
        
        # print(args_list[0])
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the function to be executed in parallel
            results = list(executor.map(minimize_tariff, args_list))
        
        new_tariff = np.zeros_like(p.tariff)
        
        for j in range(len(p.countries)):
            mask = np.ones(len(p.countries),bool)
            mask[j] = False
            new_tariff[j,mask,1] = results[j]

        print(new_tariff)
                
    return new_tariff

def find_nash_eq_tariff(p_baseline,lb_tariff=0,ub_tariff=1,method='fixed_point',dynamics=False,
                 solver_options=None,tol=1e-15,
                 damping = 1,max_workers=6,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    
    
    condition = True

    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    
    x_old = p_baseline.tariff.copy()
    
    convergence = []
    new_tariff = None
        
    while condition:
        print(it)
        if it != 0:
            x_old = (new_tariff+(damping-1)*x_old)/damping
            p_it_baseline.tariff[:] = x_old
        
        sol, sol_it_baseline = fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_tariff = compute_new_tariff_fixed_point(p_it_baseline, sol_baseline, lb_tariff, 
                                                    ub_tariff, hist_nash = None,
                                                    dynamics=dynamics,parallel=parallel,
                                                    max_workers=max_workers)

        p_it_baseline.tariff = new_tariff
        sol, sol_it= fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
        
        condition = np.linalg.norm(new_tariff-x_old) > tol
        
        convergence.append(np.linalg.norm(new_tariff - x_old))
        
        print(convergence)
        
        it += 1
        
        if it>5:
            damping = 5
        
        if not parallel:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it
    
#%% compute nash tariffs of cooperative deltas

def minus_welfare_of_tariff_coop_delta(country_tariff,p,c,sol_it_baseline, aggregation_method, hist = None,
                           dynamics=False):
    
    lb_delta=0.01
    ub_delta=12
    
    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    
    back_up_tariff_value = p.tariff[p.countries.index(c),mask,1]
    p.tariff[p.countries.index(c),mask,1] = country_tariff
    
    if aggregation_method == 'negishi':
        custom_x0 = np.array([lb_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta])
    if aggregation_method == 'pop_weighted':
        custom_x0 = np.array([lb_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta,lb_delta,lb_delta,ub_delta,lb_delta,ub_delta])
    
    p_opti, sol_opti = find_coop_eq(p,aggregation_method,parallel = False,
                     lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                     solver_options=None,tol=1e-4,
                     static_eq_deltas = None,custom_weights=None,
                     # custom_x0 = np.ones(p_baseline.N)*12,
                     custom_x0 = custom_x0,
                     max_workers=12)
    
    welfare = -sol_opti.cons_eq_welfare[p.countries.index(c)]
    
    # if dynamics:
    #     sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_fin = sol_c, sol_init=sol_it_baseline,Nt=21,
    #                                           t_inf=500,
    #                             cobweb_anim=False,tol =1e-14,
    #                             accelerate=False,
    #                             accelerate_when_stable=False,
    #                             cobweb_qty='l_R',
    #                             plot_convergence=False,
    #                             plot_cobweb=False,
    #                             plot_live = False,
    #                             safe_convergence=1e-8,
    #                             disp_summary=False,
    #                             damping = 60,
    #                             max_count = 50000,
    #                             accel_memory =5, 
    #                             accel_type1=True, 
    #                             accel_regularization=1e-10,
    #                             accel_relaxation=1, 
    #                             accel_safeguard_factor=1, 
    #                             accel_max_weight_norm=1e6,
    #                             damping_post_acceleration=10
    #                             )
    #     dyn_sol_c.compute_non_solver_quantities(p)
        
    #     welfare = -dyn_sol_c.cons_eq_welfare[p.countries.index(c)]

    p.tariff[p.countries.index(c),mask,1] = back_up_tariff_value
    p.guess = p_opti.guess
    
    return welfare

def minimize_tariff_coop_delta(args):
    p, c, sol_it_baseline, aggregation_method, hist_nash, dynamics, bounds = args
    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    if dynamics:
        #!!!! TODO dynamics
        tariff_min = optimize.shgo(func=minus_welfare_of_tariff_coop_delta,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, aggregation_method, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        tariff_min = optimize.minimize(fun = minus_welfare_of_tariff_coop_delta,
                                x0 = p.tariff[p.countries.index(c),mask,1],
                                tol = 1e-12,
                                args=(p, c, sol_it_baseline, aggregation_method, hist_nash, dynamics),
                                options = {'disp':True,'f_tol':1e-12},
                                bounds=bounds)
        # optimize.minimize_scalar(fun=minus_welfare_of_tariff,
        #                                     method='bounded',
        #                                     bounds=bounds,
        #                                     args=(p, c, sol_it_baseline, hist_nash, dynamics),
        #                                     tol=1e-15
        #                                     )
    return tariff_min.x

def compute_new_tariff_fixed_point_coop_delta(p, sol_it_baseline, lb_tariff, ub_tariff,
                                              aggregation_method, hist_nash = None, 
                                    dynamics=False,max_workers=6,parallel=True):
    
    bounds=[(lb_tariff, ub_tariff)]
    
    # if not parallel:
    #!!! TODO (or delete) not parallel calculation
    # # monoprocess
    #     new_deltas = np.zeros(len(p.countries))
    #     for i,c in enumerate(p.countries):
    #         if dynamics:
    #             # print('doing that')
    #             delta_min = optimize.shgo(func=minus_welfare_of_delta,
    #                                                   # sampling_method='halton',
    #                                                   bounds=[bounds],
    #                                                   args = (p,c,sol_it_baseline, hist_nash, dynamics),
    #                                                   options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
    #                                                   minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
    #                                                   # options = dict(ftol=1e-8)
    #                                                   )
    #         else:
    #             delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
    #                                                   method='bounded',
    #                                                   bounds=bounds,
    #                                                   args = (p,c,sol_it_baseline, hist_nash, dynamics),
    #                                                   # options={'disp':3},
    #                                                   tol=1e-15
    #                                                   )
    
    #         new_deltas[i] = delta_min.x
    #     if hist_nash is not None:
    #         hist_nash.expected_deltas[i] = new_deltas[i]
    #         hist_nash.expected_welfare[i] = delta_min.fun
    #     print(c,new_deltas)
    
    if parallel:
        
        args_list = [(p.copy(), c, sol_it_baseline.copy(), aggregation_method, hist_nash, dynamics, bounds) for c in p.countries]
        
        # print(args_list[0])
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the function to be executed in parallel
            results = list(executor.map(minimize_tariff_coop_delta, args_list))
        
        new_tariff = np.zeros_like(p.tariff)
        
        for j in range(len(p.countries)):
            mask = np.ones(len(p.countries),bool)
            mask[j] = False
            new_tariff[j,mask,1] = results[j]

        print(new_tariff)
                
    return new_tariff

def find_nash_eq_tariff_coop_delta(p_baseline,aggregation_method,lb_tariff=0,ub_tariff=1,method='fixed_point',dynamics=False,
                 solver_options=None,tol=1e-15,
                 damping = 1,max_workers=12, max_count=15,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    
    
    condition = True

    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    
    x_old = p_baseline.tariff.copy()
    
    convergence = []
    new_tariff = None
        
    while condition:
        print(it)
        if it != 0:
            x_old = (new_tariff+(damping-1)*x_old)/damping
            p_it_baseline.tariff[:] = x_old
        
        sol, sol_it_baseline = fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_tariff = compute_new_tariff_fixed_point_coop_delta(p_it_baseline, sol_baseline, lb_tariff, 
                                                    ub_tariff, aggregation_method, hist_nash = None,
                                                    dynamics=dynamics,parallel=parallel,
                                                    max_workers=max_workers)

        p_it_baseline.tariff = new_tariff
        
        lb_delta = 0.01
        ub_delta = 12
        
        if aggregation_method == 'negishi':
            custom_x0 = np.array([lb_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta])
        if aggregation_method == 'pop_weighted':
            custom_x0 = np.array([lb_delta,lb_delta,lb_delta,ub_delta,ub_delta,ub_delta,lb_delta,lb_delta,ub_delta,lb_delta,ub_delta])
        
        p_it_baseline, sol_opti = find_coop_eq(p_it_baseline,aggregation_method,parallel = True,
                         lb_delta=lb_delta,ub_delta=ub_delta,dynamics=False,
                         solver_options=None,tol=1e-10,
                         static_eq_deltas = None,custom_weights=None,
                         # custom_x0 = np.ones(p_baseline.N)*12,
                         custom_x0 = custom_x0,
                         max_workers=12)
        
        sol, sol_it= fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
        
        condition = np.linalg.norm(new_tariff-x_old) > tol and it < max_count
        
        convergence.append(np.linalg.norm(new_tariff - x_old))
        
        print(convergence)
        
        it += 1
        
        if it>5:
            damping = 5
        
        if not parallel:
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it

#%% compute nash equilibrium for tariffs and deltas

def minus_welfare_of_tariff_delta(country_tariff_delta,p,c,sol_it_baseline, hist = None,
                           dynamics=False):

    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    
    back_up_tariff_value = p.tariff[p.countries.index(c),mask,1]
    back_up_delta_value = p.delta[p.countries.index(c),1]
    p.tariff[p.countries.index(c),mask,1] = country_tariff_delta[1:]
    p.delta[p.countries.index(c),1] = country_tariff_delta[0]
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            # plot_live=True,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping = 10,
                            max_count = 1e4,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=2
                            ) 
    if sol.status != 'successful':
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
        if sol.status != 'successful':
            print(p.delta,'failed2')
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_it_baseline)
    
    welfare = -sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_fin = sol_c, sol_init=sol_it_baseline,Nt=21,
                                              t_inf=500,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                plot_live = False,
                                safe_convergence=1e-8,
                                disp_summary=False,
                                damping = 60,
                                max_count = 50000,
                                accel_memory =5, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=1, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=10
                                )
        dyn_sol_c.compute_non_solver_quantities(p)
        welfare = -dyn_sol_c.cons_eq_welfare[p.countries.index(c)]
    
    if hist is not None:
        fig, ax = plt.subplots(figsize=(16,12))
        hist.delta.append(p.delta[p.countries.index(c),1])
        hist.welfare.append(welfare)
        ax.scatter(np.log(hist.delta),np.log(-np.array(hist.welfare)),color='grey')
        ax.scatter(np.log(hist.expected_deltas),np.log(-np.array(hist.expected_welfare)),color='grey',label='expected change')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.expected_deltas)[i],np.log(-hist.expected_welfare[i])),color='grey')
        ax.scatter(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1]), color = 'red',label='search optimization')
        ax.annotate(c,(np.log(hist.delta)[-1],np.log(-np.array(hist.welfare)[-1])),color='red')
        ax.scatter(np.log(hist.current_deltas),np.log(-hist.current_welfare), color= 'blue',label='current state of the world')
        for i,country in enumerate(p.countries):
            ax.annotate(country,(np.log(hist.current_deltas)[i],np.log(-hist.current_welfare[i])))
        plt.xlabel('Delta (log)')
        plt.ylabel('Consumption equivalent welfare change (log)')
        plt.legend()
        plt.show()
        if hist.make_a_pause:
            # input("Press Enter to run next iteration")
            hist.make_a_pause = False

    p.tariff[p.countries.index(c),mask,1] = back_up_tariff_value
    p.delta[p.countries.index(c),1] = back_up_delta_value
    p.guess = sol_c.vector_from_var()
    
    return welfare

def minimize_tariff_delta(args):
    p, c, sol_it_baseline, hist_nash, dynamics, bounds = args
    mask = np.ones(len(p.countries),bool)
    mask[p.countries.index(c)] = False
    if dynamics:
        #!!!! TODO dynamics
        tariff_delta_min = optimize.shgo(func=minus_welfare_of_tariff_delta,
                                  bounds=[bounds],
                                  args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                  options={'disp': True,'f_tol':1e-15},
                                  )
    else:
        tariff_delta_min = optimize.minimize(fun = minus_welfare_of_tariff_delta,
                                x0 = np.concatenate(
                                    [[p.delta[p.countries.index(c),1]],
                                     p.tariff[p.countries.index(c),mask,1]]
                                    ),
                                tol = 1e-15,
                                args=(p, c, sol_it_baseline, hist_nash, dynamics),
                                options = {'disp':True,'f_tol':1e-15},
                                bounds=bounds)
        # optimize.minimize_scalar(fun=minus_welfare_of_tariff,
        #                                     method='bounded',
        #                                     bounds=bounds,
        #                                     args=(p, c, sol_it_baseline, hist_nash, dynamics),
        #                                     tol=1e-15
        #                                     )
    return tariff_delta_min.x

def compute_new_tariff_delta_fixed_point(p, sol_it_baseline, lb_tariff, ub_tariff, 
                                         hist_nash = None, 
                                    dynamics=False,max_workers=6,parallel=True):
    
    bounds=[(0.01,12)]+[(lb_tariff, ub_tariff)]*(len(p.countries)-1)
    
    # if not parallel:
    #!!! TODO (or delete) not parallel calculation
    # # monoprocess
    #     new_deltas = np.zeros(len(p.countries))
    #     for i,c in enumerate(p.countries):
    #         if dynamics:
    #             # print('doing that')
    #             delta_min = optimize.shgo(func=minus_welfare_of_delta,
    #                                                   # sampling_method='halton',
    #                                                   bounds=[bounds],
    #                                                   args = (p,c,sol_it_baseline, hist_nash, dynamics),
    #                                                   options={'disp':True,'f_tol':1e-4,'minimize_every_iter':False},
    #                                                   minimizer_kwargs={'f_tol':1e-4,'eps':1e-4,'finite_diff_rel_step':1e-2}
    #                                                   # options = dict(ftol=1e-8)
    #                                                   )
    #         else:
    #             delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
    #                                                   method='bounded',
    #                                                   bounds=bounds,
    #                                                   args = (p,c,sol_it_baseline, hist_nash, dynamics),
    #                                                   # options={'disp':3},
    #                                                   tol=1e-15
    #                                                   )
    
    #         new_deltas[i] = delta_min.x
    #     if hist_nash is not None:
    #         hist_nash.expected_deltas[i] = new_deltas[i]
    #         hist_nash.expected_welfare[i] = delta_min.fun
    #     print(c,new_deltas)
    
    if parallel:
        
        args_list = [(p.copy(), c, sol_it_baseline.copy(), hist_nash, dynamics, bounds) for c in p.countries]
        
        # print(args_list[0])
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the function to be executed in parallel
            results = list(executor.map(minimize_tariff_delta, args_list))
        
        new_delta = np.zeros_like(p.delta[:,1])
        new_tariff = np.zeros_like(p.tariff)
        
        for j in range(len(p.countries)):
            mask = np.ones(len(p.countries),bool)
            mask[j] = False
            new_tariff[j,mask,1] = results[j][1:]
            new_delta[j] = results[j][0]

        print(new_delta)
                
    return new_tariff, new_delta

def find_nash_eq_tariff_delta(p_baseline,lb_tariff=0,ub_tariff=1,method='fixed_point',dynamics=False,
                 solver_options=None,tol=1e-15,
                 damping = 1,max_workers=6,
                 parallel=True,max_count=50):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=True,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    
    
    condition = True

    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()

    it = 0
    
    x_old = np.concatenate([p_baseline.delta[:,1],
                            p_baseline.tariff.copy().ravel()])
    
    convergence = []
    new_tariff = None
    new_delta = None
    x_new = None
        
    while condition:
        print(it)
        if it != 0:
            x_old = (x_new+(damping-1)*x_old)/damping
            p_it_baseline.delta[:,1] = x_old[:len(p_baseline.countries)]
            p_it_baseline.tariff = x_old[len(p_baseline.countries):].reshape(p_it_baseline.tariff.shape)
        
        sol, sol_it_baseline = fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )

        sol_it_baseline.scale_P(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        new_tariff, new_delta = compute_new_tariff_delta_fixed_point(p_it_baseline, sol_baseline, lb_tariff, 
                                                    ub_tariff, hist_nash = None,
                                                    dynamics=dynamics,parallel=parallel,
                                                    max_workers=max_workers)
        new_delta[new_delta>5] = 12
        
        p_it_baseline.tariff = new_tariff
        p_it_baseline.delta[:,1] = new_delta
        sol, sol_it= fixed_point_solver(p_it_baseline,x0=p_it_baseline.guess,
                                                  context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )   
        sol_it.scale_P(p_it_baseline)
        sol_it.compute_non_solver_quantities(p_it_baseline)
        sol_it.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        sol_it.compute_world_welfare_changes(p_it_baseline,sol_baseline)
        
        if dynamics:
            sol, dyn_sol_it = dyn_fixed_point_solver(p_it_baseline,  sol_baseline, sol_fin=sol_it, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 50,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            dyn_sol_it.compute_non_solver_quantities(p_it_baseline)
        
        x_new = np.concatenate([new_delta,
                                new_tariff.ravel()])
        
        condition = np.linalg.norm(x_new-x_old) > tol and it < max_count
        
        convergence.append(np.linalg.norm(x_new - x_old))
        
        print(convergence)
        if not parallel:
            plt.plot(x_new[len(p_baseline.countries):]-x_old[len(p_baseline.countries):])
            plt.show()
            plt.plot(convergence)
            plt.yscale('log')
            plt.show()
        
        it += 1
        
        if it>5:
            damping = 5

    if dynamics:
        return p_it_baseline, dyn_sol_it
    else:
        return p_it_baseline, sol_it


#%% compute cooperative equilibrium for deltas

def minus_world_welfare_of_delta(deltas,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    p.delta[...,1:] = deltas.reshape((p.N,p.S-1))
    print(p.delta[...,1])
    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print(p.delta,'failed')
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(p.delta,'failed2')
            p.guess = None
    # p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)
        # if np.all(deltas>5):
        #     custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        #     accelerate=False,
        #     accelerate_when_stable=False,
        #     cobweb_qty='l_R',
        #     plot_convergence=False,
        #     plot_cobweb=False,
        #     plot_live = False,
        #     safe_convergence=1e-8,
        #     disp_summary=False,
        #     damping = 0,
        #     max_count = 50000,
        #     accel_memory =5, 
        #     accel_type1=True, 
        #     accel_regularization=1e-10,
        #     accel_relaxation=1, 
        #     accel_safeguard_factor=1, 
        #     accel_max_weight_norm=1e6,
        #     damping_post_acceleration=10)
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(deltas,welfare)
    
    return -welfare

def find_coop_eq(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_deltas = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = True,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    if dynamics and static_eq_deltas is not None:
        x0 = static_eq_deltas
    else:
        x0 = p.delta[...,1:].ravel()
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_delta,ub_delta)]*len(p.countries)*(p.S-1)
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    
    # sol = optimize.shgo(func=minus_world_welfare_of_delta,
    #                                       # sampling_method='halton',
    #                                       bounds=bounds,
    #                                       args = (p,sol_baseline,dynamics,aggregation_method,
    #                                             custom_weights,custom_sol_options),
    #                                       options={'disp':True},
    #                                        # tol=1e-8,
    #                                       workers=-1
    #                                       )

    p.delta[...,1:] = sol.x.reshape((p.N,p.S-1))
    solution_welfare = -sol.fun
    
    #make a 'corner check'
    for sector in range(1,p_baseline.S):
        corner_corrected_deltas = p.delta[...,sector].copy()
        for i,c in enumerate(p_baseline.countries):
            if p.delta[i,sector] > 1 or c=='MEX':
                print('checking on ',c)
                p_corner = p.copy()
                p_corner.delta[i,sector] = ub_delta
                
                sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
                                                context = 'counterfactual',
                                                **solver_options
                                                )
                sol_corner.compute_non_solver_quantities(p_corner)
                sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
                sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                
                if aggregation_method == 'negishi':
                    corner_welfare = sol_corner.cons_eq_negishi_welfare_change
                if aggregation_method == 'pop_weighted':
                    corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                if aggregation_method == 'custom_weights':
                    sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
                    corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
                
                if dynamics:
                    sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                                 sol_fin=sol_corner,
                                                                 Nt=23,
                                                          t_inf=500,
                                            **custom_dyn_sol_options
                                            )
            
                    dyn_sol_corner.compute_non_solver_quantities(p)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    # if aggregation_method == 'custom_weights':
                    #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
                
                if corner_welfare > solution_welfare:
                    print('upper corner was better for ',c)
                    corner_corrected_deltas[i] = ub_delta
        
        p.delta[...,sector] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver(p_corner,x0=p_corner.guess,
                                    context = 'counterfactual',
                                    **solver_options
                                    )
    sol_c.compute_non_solver_quantities(p_corner)
    sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
    
    if aggregation_method == 'negishi':
        solution_welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        solution_welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
        solution_welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
                                                     Nt=23,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
    
        dyn_sol.compute_non_solver_quantities(p)
        
        if aggregation_method == 'negishi':
            solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
    
    # corner_corrected_deltas = p.delta[...,1].copy()
    for sector in range(1,p_baseline.S):
        for i,c in enumerate(p_baseline.countries):
            if p.delta[i,sector] < 2*lb_delta or c=='MEX':
                print('checking on ',c)
                p_corner = p.copy()
                p_corner.delta[i,sector] = lb_delta
                
                sol, sol_corner = fixed_point_solver(p_corner,x0=p_corner.guess,
                                                context = 'counterfactual',
                                                **solver_options
                                                )
                sol_corner.compute_non_solver_quantities(p_corner)
                sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
                sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
                
                if aggregation_method == 'negishi':
                    corner_welfare = sol_corner.cons_eq_negishi_welfare_change
                if aggregation_method == 'pop_weighted':
                    corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
                if aggregation_method == 'custom_weights':
                    sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
                    corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
                
                if dynamics:
                    sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                                 sol_fin=sol_corner,
                                                                 Nt=23,
                                                          t_inf=500,
                                            **custom_dyn_sol_options
                                            )
            
                    dyn_sol_corner.compute_non_solver_quantities(p)
                    
                    if aggregation_method == 'negishi':
                        corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                    if aggregation_method == 'pop_weighted':
                        corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                    # if aggregation_method == 'custom_weights':
                    #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
                
                # print(corner_welfare,solution_welfare)
                if corner_welfare > solution_welfare:
                    print('lower corner was better for ',c)
                    corner_corrected_deltas[i] = lb_delta
                
        p.delta[...,sector] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c

#%% compute cooperative equilibrium for deltas with entry costs

def minus_world_welfare_of_delta_with_entry_costs(deltas,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    p.delta[...,1] = deltas
    print(p.delta[...,1])
    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print(p.delta,'failed')
        sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(p.delta,'failed2')
            p.guess = None
    # p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)
        # if np.all(deltas>5):
        #     custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        #     accelerate=False,
        #     accelerate_when_stable=False,
        #     cobweb_qty='l_R',
        #     plot_convergence=False,
        #     plot_cobweb=False,
        #     plot_live = False,
        #     safe_convergence=1e-8,
        #     disp_summary=False,
        #     damping = 0,
        #     max_count = 50000,
        #     accel_memory =5, 
        #     accel_type1=True, 
        #     accel_regularization=1e-10,
        #     accel_relaxation=1, 
        #     accel_safeguard_factor=1, 
        #     accel_max_weight_norm=1e6,
        #     damping_post_acceleration=10)
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(deltas,welfare)
    
    return -welfare

def find_coop_eq_with_entry_costs(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_deltas = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = True,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver_with_entry_costs(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    if dynamics and static_eq_deltas is not None:
        x0 = static_eq_deltas
    else:
        x0 = p.delta[...,1]
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_delta,ub_delta)]*len(p.countries)
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_delta_with_entry_costs,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_delta_with_entry_costs,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    
    # sol = optimize.shgo(func=minus_world_welfare_of_delta,
    #                                       # sampling_method='halton',
    #                                       bounds=bounds,
    #                                       args = (p,sol_baseline,dynamics,aggregation_method,
    #                                             custom_weights,custom_sol_options),
    #                                       options={'disp':True},
    #                                        # tol=1e-8,
    #                                       workers=-1
    #                                       )

    p.delta[...,1] = sol.x
    solution_welfare = -sol.fun
    
    #make a 'corner check'
    corner_corrected_deltas = p.delta[...,1].copy()
    for i,c in enumerate(p_baseline.countries):
        if p.delta[i,1] > 1 or c=='MEX':
            print('checking on ',c)
            p_corner = p.copy()
            p_corner.delta[i,1] = ub_delta
            
            sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
                                            context = 'counterfactual',
                                            **solver_options
                                            )
            sol_corner.compute_non_solver_quantities(p_corner)
            sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
            sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
            if aggregation_method == 'negishi':
                corner_welfare = sol_corner.cons_eq_negishi_welfare_change
            if aggregation_method == 'pop_weighted':
                corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
            if aggregation_method == 'custom_weights':
                sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
                corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
            if dynamics:
                sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                             sol_fin=sol_corner,
                                                             Nt=23,
                                                      t_inf=500,
                                        **custom_dyn_sol_options
                                        )
        
                dyn_sol_corner.compute_non_solver_quantities(p)
                
                if aggregation_method == 'negishi':
                    corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                if aggregation_method == 'pop_weighted':
                    corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                # if aggregation_method == 'custom_weights':
                #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
            if corner_welfare > solution_welfare:
                print('upper corner was better for ',c)
                corner_corrected_deltas[i] = ub_delta
    
    p.delta[...,1] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
                                    context = 'counterfactual',
                                    **solver_options
                                    )
    sol_c.compute_non_solver_quantities(p_corner)
    sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
    
    if aggregation_method == 'negishi':
        solution_welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        solution_welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
        solution_welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
                                                     Nt=23,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
    
        dyn_sol.compute_non_solver_quantities(p)
        
        if aggregation_method == 'negishi':
            solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
    
    # corner_corrected_deltas = p.delta[...,1].copy()
    for i,c in enumerate(p_baseline.countries):
        if p.delta[i,1] < 2*lb_delta or c=='MEX':
            print('checking on ',c)
            p_corner = p.copy()
            p_corner.delta[i,1] = lb_delta
            
            sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
                                            context = 'counterfactual',
                                            **solver_options
                                            )
            sol_corner.compute_non_solver_quantities(p_corner)
            sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
            sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
            if aggregation_method == 'negishi':
                corner_welfare = sol_corner.cons_eq_negishi_welfare_change
            if aggregation_method == 'pop_weighted':
                corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
            if aggregation_method == 'custom_weights':
                sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
                corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
            if dynamics:
                sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
                                                             sol_fin=sol_corner,
                                                             Nt=23,
                                                      t_inf=500,
                                        **custom_dyn_sol_options
                                        )
        
                dyn_sol_corner.compute_non_solver_quantities(p)
                
                if aggregation_method == 'negishi':
                    corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
                if aggregation_method == 'pop_weighted':
                    corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
                # if aggregation_method == 'custom_weights':
                #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
            # print(corner_welfare,solution_welfare)
            if corner_welfare > solution_welfare:
                print('lower corner was better for ',c)
                corner_corrected_deltas[i] = lb_delta
            
    p.delta[...,1] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c    

#%% compute cooperative equilibrium with double delta

def minus_world_welfare_of_delta_double_delta(deltas,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    p.delta_dom[...,1] = deltas[:12]
    p.delta_int[...,1] = deltas[12:]
    p.update_delta_eff()
    print(p.delta_dom[...,1])
    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-12,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print(p.delta,'failed')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(p.delta,'failed2')
            p.guess = None
    p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
        
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)
        # if np.all(deltas>5):
        #     custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        #     accelerate=False,
        #     accelerate_when_stable=False,
        #     cobweb_qty='l_R',
        #     plot_convergence=False,
        #     plot_cobweb=False,
        #     plot_live = False,
        #     safe_convergence=1e-8,
        #     disp_summary=False,
        #     damping = 0,
        #     max_count = 50000,
        #     accel_memory =5, 
        #     accel_type1=True, 
        #     accel_regularization=1e-10,
        #     accel_relaxation=1, 
        #     accel_safeguard_factor=1, 
        #     accel_max_weight_norm=1e6,
        #     damping_post_acceleration=10)
        sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(deltas,welfare)
    
    return -welfare

def find_coop_eq_double_delta(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_deltas = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = True,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver_double_diff_double_delta(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    x0 = np.concatenate([p.delta_dom[...,1],p.delta_int[...,1]])
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_delta,ub_delta)]*(len(p.countries)*2)
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_delta_double_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_delta_double_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    
    # sol = optimize.shgo(func=minus_world_welfare_of_delta,
    #                                       # sampling_method='halton',
    #                                       bounds=bounds,
    #                                       args = (p,sol_baseline,dynamics,aggregation_method,
    #                                             custom_weights,custom_sol_options),
    #                                       options={'disp':True},
    #                                        # tol=1e-8,
    #                                       workers=-1
    #                                       )
    
    p.delta_dom[...,1] = sol.x[:12]
    p.delta_int[...,1] = sol.x[12:]
    solution_welfare = -sol.fun
    
    # #make a 'corner check'
    # corner_corrected_deltas = p.delta[...,1].copy()
    # for i,c in enumerate(p_baseline.countries):
    #     if p.delta[i,1] > 1 or c=='MEX':
    #         print('checking on ',c)
    #         p_corner = p.copy()
    #         p_corner.delta[i,1] = ub_delta
            
    #         sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                         context = 'counterfactual',
    #                                         **solver_options
    #                                         )
    #         sol_corner.compute_non_solver_quantities(p_corner)
    #         sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    #         sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
    #         if aggregation_method == 'negishi':
    #             corner_welfare = sol_corner.cons_eq_negishi_welfare_change
    #         if aggregation_method == 'pop_weighted':
    #             corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
    #         if aggregation_method == 'custom_weights':
    #             sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #             corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if dynamics:
    #             sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
    #                                                          sol_fin=sol_corner,
    #                                                          Nt=23,
    #                                                   t_inf=500,
    #                                     **custom_dyn_sol_options
    #                                     )
        
    #             dyn_sol_corner.compute_non_solver_quantities(p)
                
    #             if aggregation_method == 'negishi':
    #                 corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
    #             if aggregation_method == 'pop_weighted':
    #                 corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
    #             # if aggregation_method == 'custom_weights':
    #             #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if corner_welfare > solution_welfare:
    #             print('upper corner was better for ',c)
    #             corner_corrected_deltas[i] = ub_delta
    
    # p.delta[...,1] = corner_corrected_deltas
    
    # sol, sol_c = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                 context = 'counterfactual',
    #                                 **solver_options
    #                                 )
    # sol_c.compute_non_solver_quantities(p_corner)
    # sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    # sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
    
    # if aggregation_method == 'negishi':
    #     solution_welfare = sol_c.cons_eq_negishi_welfare_change
    # if aggregation_method == 'pop_weighted':
    #     solution_welfare = sol_c.cons_eq_pop_average_welfare_change
    # if aggregation_method == 'custom_weights':
    #     sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #     solution_welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    # if dynamics:
    #     sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
    #                                                  Nt=23,
    #                                           t_inf=500,
    #                             **custom_dyn_sol_options
    #                             )
    
    #     dyn_sol.compute_non_solver_quantities(p)
        
    #     if aggregation_method == 'negishi':
    #         solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
    #     if aggregation_method == 'pop_weighted':
    #         solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
    
    # # corner_corrected_deltas = p.delta[...,1].copy()
    # for i,c in enumerate(p_baseline.countries):
    #     if p.delta[i,1] < 2*lb_delta or c=='MEX':
    #         print('checking on ',c)
    #         p_corner = p.copy()
    #         p_corner.delta[i,1] = lb_delta
            
    #         sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                         context = 'counterfactual',
    #                                         **solver_options
    #                                         )
    #         sol_corner.compute_non_solver_quantities(p_corner)
    #         sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    #         sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
    #         if aggregation_method == 'negishi':
    #             corner_welfare = sol_corner.cons_eq_negishi_welfare_change
    #         if aggregation_method == 'pop_weighted':
    #             corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
    #         if aggregation_method == 'custom_weights':
    #             sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #             corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if dynamics:
    #             sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
    #                                                          sol_fin=sol_corner,
    #                                                          Nt=23,
    #                                                   t_inf=500,
    #                                     **custom_dyn_sol_options
    #                                     )
        
    #             dyn_sol_corner.compute_non_solver_quantities(p)
                
    #             if aggregation_method == 'negishi':
    #                 corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
    #             if aggregation_method == 'pop_weighted':
    #                 corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
    #             # if aggregation_method == 'custom_weights':
    #             #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
    #         # print(corner_welfare,solution_welfare)
    #         if corner_welfare > solution_welfare:
    #             print('lower corner was better for ',c)
    #             corner_corrected_deltas[i] = lb_delta
            
    # p.delta[...,1] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c    
    
#%% compute cooperative equilibrium with double diffusion

def minus_world_welfare_of_delta_double_diffusion(deltas,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    p.delta_dom[...,1] = deltas
    p.delta_int[...,1] = deltas
    p.update_delta_eff()
    print(p.delta_dom[...,1])
    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-12,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print(p.delta,'failed')
        sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(p.delta,'failed2')
            p.guess = None
    p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
        
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)
        # if np.all(deltas>5):
        #     custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        #     accelerate=False,
        #     accelerate_when_stable=False,
        #     cobweb_qty='l_R',
        #     plot_convergence=False,
        #     plot_cobweb=False,
        #     plot_live = False,
        #     safe_convergence=1e-8,
        #     disp_summary=False,
        #     damping = 0,
        #     max_count = 50000,
        #     accel_memory =5, 
        #     accel_type1=True, 
        #     accel_regularization=1e-10,
        #     accel_relaxation=1, 
        #     accel_safeguard_factor=1, 
        #     accel_max_weight_norm=1e6,
        #     damping_post_acceleration=10)
        sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(deltas,welfare)
    
    return -welfare

def find_coop_eq_double_diffusion(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_deltas = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = True,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver_double_diff_double_delta(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    x0 = np.concatenate([p.delta_int[...,1]])
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_delta,ub_delta)]
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_delta_double_diffusion,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_delta_double_diffusion,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    
    # sol = optimize.shgo(func=minus_world_welfare_of_delta,
    #                                       # sampling_method='halton',
    #                                       bounds=bounds,
    #                                       args = (p,sol_baseline,dynamics,aggregation_method,
    #                                             custom_weights,custom_sol_options),
    #                                       options={'disp':True},
    #                                        # tol=1e-8,
    #                                       workers=-1
    #                                       )
    
    p.delta_dom[...,1] = sol.x
    p.delta_int[...,1] = sol.x
    p.update_delta_eff()
    solution_welfare = -sol.fun
    
    # #make a 'corner check'
    # corner_corrected_deltas = p.delta[...,1].copy()
    # for i,c in enumerate(p_baseline.countries):
    #     if p.delta[i,1] > 1 or c=='MEX':
    #         print('checking on ',c)
    #         p_corner = p.copy()
    #         p_corner.delta[i,1] = ub_delta
            
    #         sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                         context = 'counterfactual',
    #                                         **solver_options
    #                                         )
    #         sol_corner.compute_non_solver_quantities(p_corner)
    #         sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    #         sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
    #         if aggregation_method == 'negishi':
    #             corner_welfare = sol_corner.cons_eq_negishi_welfare_change
    #         if aggregation_method == 'pop_weighted':
    #             corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
    #         if aggregation_method == 'custom_weights':
    #             sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #             corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if dynamics:
    #             sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
    #                                                          sol_fin=sol_corner,
    #                                                          Nt=23,
    #                                                   t_inf=500,
    #                                     **custom_dyn_sol_options
    #                                     )
        
    #             dyn_sol_corner.compute_non_solver_quantities(p)
                
    #             if aggregation_method == 'negishi':
    #                 corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
    #             if aggregation_method == 'pop_weighted':
    #                 corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
    #             # if aggregation_method == 'custom_weights':
    #             #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if corner_welfare > solution_welfare:
    #             print('upper corner was better for ',c)
    #             corner_corrected_deltas[i] = ub_delta
    
    # p.delta[...,1] = corner_corrected_deltas
    
    # sol, sol_c = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                 context = 'counterfactual',
    #                                 **solver_options
    #                                 )
    # sol_c.compute_non_solver_quantities(p_corner)
    # sol_c.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    # sol_c.compute_world_welfare_changes(p_corner,sol_baseline)
    
    # if aggregation_method == 'negishi':
    #     solution_welfare = sol_c.cons_eq_negishi_welfare_change
    # if aggregation_method == 'pop_weighted':
    #     solution_welfare = sol_c.cons_eq_pop_average_welfare_change
    # if aggregation_method == 'custom_weights':
    #     sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #     solution_welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    # if dynamics:
    #     sol, dyn_sol = dyn_fixed_point_solver(p, sol_init=sol_baseline, 
    #                                                  Nt=23,
    #                                           t_inf=500,
    #                             **custom_dyn_sol_options
    #                             )
    
    #     dyn_sol.compute_non_solver_quantities(p)
        
    #     if aggregation_method == 'negishi':
    #         solution_welfare = dyn_sol.cons_eq_negishi_welfare_change
    #     if aggregation_method == 'pop_weighted':
    #         solution_welfare = dyn_sol.cons_eq_pop_average_welfare_change
    
    # # corner_corrected_deltas = p.delta[...,1].copy()
    # for i,c in enumerate(p_baseline.countries):
    #     if p.delta[i,1] < 2*lb_delta or c=='MEX':
    #         print('checking on ',c)
    #         p_corner = p.copy()
    #         p_corner.delta[i,1] = lb_delta
            
    #         sol, sol_corner = fixed_point_solver_with_entry_costs(p_corner,x0=p_corner.guess,
    #                                         context = 'counterfactual',
    #                                         **solver_options
    #                                         )
    #         sol_corner.compute_non_solver_quantities(p_corner)
    #         sol_corner.compute_consumption_equivalent_welfare(p_corner,sol_baseline)
    #         sol_corner.compute_world_welfare_changes(p_corner,sol_baseline)
            
    #         if aggregation_method == 'negishi':
    #             corner_welfare = sol_corner.cons_eq_negishi_welfare_change
    #         if aggregation_method == 'pop_weighted':
    #             corner_welfare = sol_corner.cons_eq_pop_average_welfare_change
    #         if aggregation_method == 'custom_weights':
    #             sol_corner.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    #             corner_welfare = sol_corner.cons_eq_custom_weights_welfare_change
            
    #         if dynamics:
    #             sol, dyn_sol_corner = dyn_fixed_point_solver(p_corner, sol_init=sol_baseline, 
    #                                                          sol_fin=sol_corner,
    #                                                          Nt=23,
    #                                                   t_inf=500,
    #                                     **custom_dyn_sol_options
    #                                     )
        
    #             dyn_sol_corner.compute_non_solver_quantities(p)
                
    #             if aggregation_method == 'negishi':
    #                 corner_welfare = dyn_sol_corner.cons_eq_negishi_welfare_change
    #             if aggregation_method == 'pop_weighted':
    #                 corner_welfare = dyn_sol_corner.cons_eq_pop_average_welfare_change
    #             # if aggregation_method == 'custom_weights':
    #             #     corner_welfare = dyn_sol_corner.cons_eq_custom_weights_welfare_change
            
    #         # print(corner_welfare,solution_welfare)
    #         if corner_welfare > solution_welfare:
    #             print('lower corner was better for ',c)
    #             corner_corrected_deltas[i] = lb_delta
            
    # p.delta[...,1] = corner_corrected_deltas
    
    sol, sol_c = fixed_point_solver_double_diff_double_delta(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c


#%% compute cooperative equilibrium for tariffs

def minus_world_welfare_of_tariff(tariff,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    mask = np.ones_like(p.tariff,bool)
    mask[:,:,0] = False
    np.einsum('iis->is',mask)[:] = False
    p.tariff[mask] = tariff

    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print('failed')
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print('failed2')
            p.guess = None
    # p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)
        # if np.all(deltas>5):
        #     custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        #     accelerate=False,
        #     accelerate_when_stable=False,
        #     cobweb_qty='l_R',
        #     plot_convergence=False,
        #     plot_cobweb=False,
        #     plot_live = False,
        #     safe_convergence=1e-8,
        #     disp_summary=False,
        #     damping = 0,
        #     max_count = 50000,
        #     accel_memory =5, 
        #     accel_type1=True, 
        #     accel_regularization=1e-10,
        #     accel_relaxation=1, 
        #     accel_safeguard_factor=1, 
        #     accel_max_weight_norm=1e6,
        #     damping_post_acceleration=10)
        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(welfare)
    
    return -welfare

def find_coop_eq_tariff(p_baseline,aggregation_method,
                 lb_tariff=0,ub_tariff=1,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_tariff = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = False,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                                    **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    mask = np.ones_like(p.tariff,bool)
    mask[:,:,0] = False
    np.einsum('iis->is',mask)[:] = False
    
    if dynamics and static_eq_tariff is not None:
        x0 = static_eq_tariff[mask].ravel()
    else:
        x0 = p.tariff[mask].ravel()
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_tariff,ub_tariff)]*len(x0)
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_tariff,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_tariff,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    p.tariff[mask] = sol.x
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c

#%% compute cooperative equilibrium for deltas and tariffs

def minus_world_welfare_of_tariff_delta(x,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    mask = np.ones_like(p.tariff,bool)
    mask[:,:,0] = False
    np.einsum('iis->is',mask)[:] = False
    
    p.delta[:,1] = x[:len(p.countries)]
    p.tariff[mask] = x[len(p.countries):]

    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print('failed')
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print('failed2')
            p.guess = None
    # p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)

    if aggregation_method == 'negishi':
        welfare = sol_c.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_c.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_c.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)

        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(welfare)
    
    return -welfare        
        
def find_coop_eq_tariff_delta(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,
                 lb_tariff=0,ub_tariff=1,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_tariff = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = False,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                                    **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    mask = np.ones_like(p.tariff,bool)
    mask[:,:,0] = False
    np.einsum('iis->is',mask)[:] = False
    
    x0 = np.concatenate([p.delta[:,1],p.tariff[mask]])
    
    bounds = [(lb_delta,ub_delta)]*p.delta[:,1].shape[0]+[(lb_tariff,ub_tariff)]*p.tariff[mask].ravel().shape[0]

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_tariff_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_tariff_delta,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)
        
    p.tariff[mask] = sol.x[len(p.countries):]
    p.delta[:,1] = sol.x[:len(p.countries)]
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c

#%% compute cooperative deltas of Nash tariffs

def minus_world_welfare_of_delta_nash_tariff(deltas,p,sol_baseline,dynamics,aggregation_method,
                                 custom_weights=None,custom_sol_options=None,
                                 custom_dyn_sol_options=None):
    p.delta[...,1] = deltas
    print(p.delta[...,1])
    if custom_sol_options is None:
        custom_sol_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 5,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **custom_sol_options
                            )
    if sol.status == 'successful':
        p.guess = sol_c.vector_from_var()
    else:
        print(p.delta,'failed')
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=False,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2
                                )
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(p.delta,'failed2')
            p.guess = None
    # p.guess = sol_c.vector_from_var()
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    p_nash, sol_nash = find_nash_eq_tariff(p,lb_tariff=0,ub_tariff=1,method='fixed_point',
                     solver_options=None,tol=1e-4,
                     max_workers=12,parallel=False
                     )
    
    if aggregation_method == 'custom_weights':
        sol_nash.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)

    if aggregation_method == 'negishi':
        welfare = sol_nash.cons_eq_negishi_welfare_change
    if aggregation_method == 'pop_weighted':
        welfare = sol_nash.cons_eq_pop_average_welfare_change
    if aggregation_method == 'custom_weights':
        welfare = sol_nash.cons_eq_custom_weights_welfare_change
    
    if dynamics:
        if custom_dyn_sol_options is None:
            custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
            accelerate=False,
            accelerate_when_stable=False,
            cobweb_qty='l_R',
            plot_convergence=False,
            plot_cobweb=False,
            plot_live = False,
            safe_convergence=1e-8,
            disp_summary=False,
            damping = 60,
            max_count = 50000,
            accel_memory =5, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=1, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=10)

        sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                x0 = p.dyn_guess,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        if sol.status == 'failed':
            p.dyn_guess=None
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_init=sol_baseline, Nt=23,
                                                    x0 = p.dyn_guess,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=False,
                                    damping = 60,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=10
                                    )
            
        dyn_sol_c.compute_non_solver_quantities(p)
        p.dyn_guess = dyn_sol_c.vector_from_var()
        if aggregation_method == 'custom_weights':
            dyn_sol_c.compute_world_welfare_changes_custom_weights(p, custom_weights)

        if aggregation_method == 'negishi':
            welfare = dyn_sol_c.cons_eq_negishi_welfare_change
        if aggregation_method == 'pop_weighted':
            welfare = dyn_sol_c.cons_eq_pop_average_welfare_change
        if aggregation_method == 'custom_weights':
            welfare = dyn_sol_c.cons_eq_custom_weights_welfare_change
    
    print(deltas,welfare)
    
    return -welfare

def find_coop_eq_delta_nash_tariff(p_baseline,aggregation_method,
                 lb_delta=0.01,ub_delta=12,dynamics=False,
                 solver_options=None,tol=1e-15,
                 static_eq_deltas = None,custom_weights=None,
                 custom_x0 = None,max_workers=6,
                 custom_dyn_sol_options=None, displays = True,
                 parallel=True):
    
    if solver_options is None:
        solver_options = dict(cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=2)
        
    custom_sol_options = solver_options
    
    if custom_dyn_sol_options is None:
        custom_dyn_sol_options = dict(cobweb_anim=False,tol =1e-14,
        accelerate=False,
        accelerate_when_stable=False,
        cobweb_qty='l_R',
        plot_convergence=False,
        plot_cobweb=False,
        plot_live = False,
        safe_convergence=1e-8,
        disp_summary=False,
        damping = 60,
        max_count = 50000,
        accel_memory =5, 
        accel_type1=True, 
        accel_regularization=1e-10,
        accel_relaxation=1, 
        accel_safeguard_factor=1, 
        accel_max_weight_norm=1e6,
        damping_post_acceleration=10)
    
    sol, sol_baseline = fixed_point_solver(p_baseline,x0=p_baseline.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    
    sol_baseline.scale_P(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline) 
    
    p = p_baseline.copy()
    
    if dynamics and static_eq_deltas is not None:
        x0 = static_eq_deltas
    else:
        x0 = p.delta[...,1]
        
    if custom_x0 is not None:
        x0 = custom_x0
    
    bounds = [(lb_delta,ub_delta)]*len(p.countries)
    # bounds = (lb_delta,ub_delta)

    if parallel:
        print('parallel')
        sol = minimize_parallel(fun = minus_world_welfare_of_delta_nash_tariff,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options,custom_dyn_sol_options),
                                # options = {'disp':True},
                                bounds=bounds, 
                                parallel={'max_workers':max_workers,
                                          'loginfo': displays,
                                          'time':displays,
                                          'verbose':displays}
            )
    else:
        print('not parallel')
        sol = optimize.minimize(fun = minus_world_welfare_of_delta_nash_tariff,
                                x0 = x0,
                                tol = tol,
                                args=(p,sol_baseline,dynamics,aggregation_method,
                                      custom_weights,custom_sol_options),
                                options = {'disp':True},
                                bounds=bounds)

    p.delta[...,1] = sol.x
    
    p, sol_nash = find_nash_eq_tariff(p,lb_tariff=0,ub_tariff=1,method='fixed_point',
                     solver_options=None,tol=1e-4,
                     max_workers=12,parallel=True
                     )
    
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            **solver_options
                            )
    sol_c.scale_P(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
    sol_c.compute_world_welfare_changes(p,sol_baseline)
    if aggregation_method == 'custom_weights':
        sol_c.compute_world_welfare_changes_custom_weights(p, sol_baseline, custom_weights)
    
    if dynamics:
        sol, dyn_sol_c = dyn_fixed_point_solver(p,  sol_baseline, sol_fin=sol_c, Nt=25,
                                              t_inf=500,
                                **custom_dyn_sol_options
                                )
        dyn_sol_c.compute_non_solver_quantities(p)

    if dynamics:
        return p, dyn_sol_c
    else:
        return p, sol_c

#%% compute counterfactuals

def make_counterfactual(p_baseline,country,local_path,
                        delta_factor_array=None,dynamics=False,
                        sol_baseline=None,harmonizing_country='USA',
                        Nt=25,t_inf=500,alt_delta=None):
    country_path = local_path+country+'/'
    try:
        os.mkdir(country_path)
    except:
        pass

    print(country)
    p = p_baseline.copy()
    if delta_factor_array is None:
        delta_factor_array = np.logspace(-1,1,111)
        if country == 'Harmonizing' or country == 'Upper_harmonizing':
            delta_factor_array = np.linspace(0,1,101)
        if country == 'Uniform_delta' or country == 'Upper_uniform_delta':
            delta_factor_array = np.logspace(-2,0,101)
        if country == 'trade_cost_eq_trips_all_countries_all_sectors':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_eq_trips_all_countries_pat_sectors':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_all_countries_all_sectors':
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_all_countries_pat_sectors':
            delta_factor_array = np.linspace(0.5,2,151)
        if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
            assert alt_delta is not None
            delta_factor_array = np.logspace(0,2,151)
        if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect_additive':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0,0.5,101)
    if country in p_baseline.countries:
        idx_country = p_baseline.countries.index(country)
    if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
        assert alt_delta is not None
        idx_country = p_baseline.countries.index(country[:3])
    if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
        assert alt_delta is not None
        idx_country = p_baseline.countries.index(country[:3])
    if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect_additive':
        assert alt_delta is not None
        idx_country = p_baseline.countries.index(country[:3])
        
    for i,delt in enumerate(delta_factor_array):
        # print(delt)
        if country in p_baseline.countries:
            p.delta[idx_country,1] = p_baseline.delta[idx_country,1] * delt
        if country == 'World':
            p.delta[:,1] = p_baseline.delta[:,1] * delt
        if country == 'Harmonizing' or country == 'Upper_harmonizing':
            p.delta[:,1] = p_baseline.delta[:,1]**(1-delt) * p_baseline.delta[
                p_baseline.countries.index(harmonizing_country),1
                ]**delt
            if country == 'Upper_harmonizing':
                p.delta[:,1][p_baseline.delta[:,1]<p_baseline.delta[p_baseline.countries.index(harmonizing_country),1]
                             ] = p_baseline.delta[:,1][
                                 p_baseline.delta[:,1]<p_baseline.delta[p_baseline.countries.index(harmonizing_country),1]
                                 ]
        if country == 'trade_cost_eq_trips_all_countries_all_sectors':
            p.delta[:,1] = alt_delta
            p.tau = p_baseline.tau * delt
            for n in range(p.N):
                p.tau[n,n,:] = 1
        if country == 'trade_cost_eq_trips_all_countries_pat_sectors':
            p.delta[:,1] = alt_delta
            p.tau[:,:,1] = p_baseline.tau[:,:,1] * delt
            for n in range(p.N):
                p.tau[n,n,:] = 1
        if country == 'trade_cost_all_countries_all_sectors':
            p.tau = p_baseline.tau * delt
            for n in range(p.N):
                p.tau[n,n,:] = 1
        if country == 'trade_cost_all_countries_pat_sectors':
            p.tau[:,:,1] = p_baseline.tau[:,:,1] * delt
            for n in range(p.N):
                p.tau[n,n,:] = 1
        if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
            p.delta[:,1] = alt_delta
            p.tau[:,idx_country,1] = p_baseline.tau[:,idx_country,1] * delt
            p.tau[idx_country,:,1] = p_baseline.tau[idx_country,:,1] * delt
            p.tau[idx_country,idx_country,:] = 1
        if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
            p.delta[:,1] = alt_delta
            p.tariff[:,idx_country,1] = p_baseline.tariff[:,idx_country,1] * delt
            p.tariff[idx_country,idx_country,:] = 0
        if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect_additive':
            p.delta[:,1] = alt_delta
            p.tariff[:,idx_country,1] = p_baseline.tariff[:,idx_country,1] + delt
            p.tariff[idx_country,idx_country,:] = 0
        if country == 'Uniform_delta':
            p.delta[:,1] = delt
        if country == 'Upper_uniform_delta':
            p.delta[:,1] = np.minimum(p_baseline.delta[:,1], delt)
            
        sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='phi',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.001,
                                disp_summary=False,
                                damping = 10,
                                max_count = 1e4,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=5
                                )
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p)
        if sol.status == 'successful':
            p.guess = sol_c.vector_from_var()
        else:
            print(country,delt,'failed')
            sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='phi',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    safe_convergence=0.001,
                                    disp_summary=False,
                                    damping = 10,
                                    max_count = 1e4,
                                    accel_memory = 50, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=0.5, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            sol_c.scale_P(p)
            sol_c.compute_non_solver_quantities(p)
            if sol.status == 'successful':
                p.guess = sol_c.vector_from_var()
            else:
                print(country,delt,'failed2')
                p.guess = None
            
        if dynamics:
            sol, dyn_sol_c = dyn_fixed_point_solver(p, sol_baseline,sol_fin=sol_c,
                                    Nt=Nt,t_inf=t_inf,x0=p.dyn_guess,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=False,
                                    plot_cobweb=False,
                                    plot_live = False,
                                    safe_convergence=1e-8,
                                    disp_summary=True,
                                    damping = 20,
                                    max_count = 50000,
                                    accel_memory =5, 
                                    accel_type1=True, 
                                    accel_regularization=1e-10,
                                    accel_relaxation=1, 
                                    accel_safeguard_factor=1, 
                                    accel_max_weight_norm=1e6,
                                    damping_post_acceleration=5
                                    )
            if sol.status == 'successful':
                p.dyn_guess = dyn_sol_c.vector_from_var()
            else:
                p.dyn_guess = None
                print('failed',country_path+'/'+str(i)+'/')
        
        p.write_params(country_path+'/'+str(i)+'/') 

#%% compute counterfactuals with entry costs

def make_counterfactual_with_entry_costs(p_baseline,country,local_path,
                        delta_factor_array=None,dynamics=False,
                        sol_baseline=None,harmonizing_country='USA',
                        Nt=25,t_inf=500,alt_delta=None):
    country_path = local_path+country+'/'
    try:
        os.mkdir(country_path)
    except:
        pass

    print(country)
    p = p_baseline.copy()
    if delta_factor_array is None:
        delta_factor_array = np.logspace(-1,1,11)
        # delta_factor_array = np.logspace(-np.log10(2),np.log10(2),51)
        # delta_factor_array = np.logspace(-np.log10(2),np.log10(2),51)
        if country == 'Harmonizing' or country == 'Upper_harmonizing':
            delta_factor_array = np.linspace(0,1,101)
        if country == 'Uniform_delta' or country == 'Upper_uniform_delta':
            delta_factor_array = np.logspace(-2,0,101)
        if country == 'trade_cost_eq_trips_all_countries_all_sectors':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_eq_trips_all_countries_pat_sectors':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_all_countries_all_sectors':
            delta_factor_array = np.linspace(0.5,2,151)
        if country == 'trade_cost_all_countries_pat_sectors':
            delta_factor_array = np.linspace(0.5,2,151)
        if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
            assert alt_delta is not None
            delta_factor_array = np.linspace(0.5,2,151)
        if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
            assert alt_delta is not None
            delta_factor_array = np.logspace(0,2,151)
    if country in p_baseline.countries:
        idx_country = p_baseline.countries.index(country)
    if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
        assert alt_delta is not None
        idx_country = p_baseline.countries.index(country[:3])
    if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
        assert alt_delta is not None
        idx_country = p_baseline.countries.index(country[:3])
        
    for i,delt in enumerate(delta_factor_array):
        print(i)
        if i>-1:
            if country in p_baseline.countries:
                p.delta[idx_country,1] = p_baseline.delta[idx_country,1] * delt
            if country == 'World':
                p.delta[:,1] = p_baseline.delta[:,1] * delt
            if country == 'Harmonizing' or country == 'Upper_harmonizing':
                p.delta[:,1] = p_baseline.delta[:,1]**(1-delt) * p_baseline.delta[
                    p_baseline.countries.index(harmonizing_country),1
                    ]**delt
                if country == 'Upper_harmonizing':
                    p.delta[:,1][p_baseline.delta[:,1]<p_baseline.delta[p_baseline.countries.index(harmonizing_country),1]
                                 ] = p_baseline.delta[:,1][
                                     p_baseline.delta[:,1]<p_baseline.delta[p_baseline.countries.index(harmonizing_country),1]
                                     ]
            if country == 'trade_cost_eq_trips_all_countries_all_sectors':
                p.delta[:,1] = alt_delta
                p.tau = p_baseline.tau * delt
                for n in range(p.N):
                    p.tau[n,n,:] = 1
            if country == 'trade_cost_eq_trips_all_countries_pat_sectors':
                p.delta[:,1] = alt_delta
                p.tau[:,:,1] = p_baseline.tau[:,:,1] * delt
                for n in range(p.N):
                    p.tau[n,n,:] = 1
            if country == 'trade_cost_all_countries_all_sectors':
                p.tau = p_baseline.tau * delt
                for n in range(p.N):
                    p.tau[n,n,:] = 1
            if country == 'trade_cost_all_countries_pat_sectors':
                p.tau[:,:,1] = p_baseline.tau[:,:,1] * delt
                for n in range(p.N):
                    p.tau[n,n,:] = 1
            if country[:3] in p_baseline.countries and country[3:] == '_trade_cost_eq_trips_exp_imp_pat_sect':
                p.delta[:,1] = alt_delta
                p.tau[:,idx_country,1] = p_baseline.tau[:,idx_country,1] * delt
                p.tau[idx_country,:,1] = p_baseline.tau[idx_country,:,1] * delt
                p.tau[idx_country,idx_country,:] = 1
            if country[:3] in p_baseline.countries and country[3:] == '_tariff_eq_trips_exp_pat_sect':
                p.delta[:,1] = alt_delta
                p.tariff[:,idx_country,1] = p_baseline.tariff[:,idx_country,1] * delt
                p.tariff[idx_country,idx_country,:] = 0
            if country == 'Uniform_delta':
                p.delta[:,1] = delt
            if country == 'Upper_uniform_delta':
                p.delta[:,1] = np.minimum(p_baseline.delta[:,1], delt)
                
            sol, sol_c = fixed_point_solver_with_entry_costs(p,
                                                             x0=p.guess,
                                    # context = 'counterfactual',
                                    # cobweb_anim=False,tol =1e-10,
                                    # accelerate=False,
                                    # accelerate_when_stable=True,
                                    # cobweb_qty='phi',
                                    # plot_convergence=True,
                                    # plot_cobweb=False,
                                    # safe_convergence=0.001,
                                    # disp_summary=False,
                                    # damping = 10,
                                    # max_count = 1e3,
                                    # accel_memory = 50, 
                                    # accel_type1=True, 
                                    # accel_regularization=1e-10,
                                    # accel_relaxation=0.5, 
                                    # accel_safeguard_factor=1, 
                                    # accel_max_weight_norm=1e6,
                                    # damping_post_acceleration=4
                                    # )
            context = 'counterfactual',
            # context = 'calibration',
            cobweb_anim=False,tol =1e-10,
            accelerate=True,
            accelerate_when_stable=True,
            cobweb_qty='phi',
            plot_convergence=True,
            plot_cobweb=False,
            safe_convergence=0.001,
            disp_summary=True,
            damping = 10,
            max_count = 1000,
            accel_memory =50, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=0.5, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=5
            )
            sol_c.scale_P(p)
            # sol_c.compute_non_solver_quantities(p)
            # if sol.status != 'successful':
            #     sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
            #                             context = 'counterfactual',
            #                             cobweb_anim=False,tol =1e-4,
            #                             accelerate=False,
            #                             accelerate_when_stable=True,
            #                             cobweb_qty='phi',
            #                             plot_convergence=True,
            #                             plot_cobweb=False,
            #                             safe_convergence=0.01,
            #                             disp_summary=False,
            #                             damping = 10,
            #                             max_count = 1e3,
            #                             accel_memory = 50, 
            #                             accel_type1=True, 
            #                             accel_regularization=1e-10,
            #                             accel_relaxation=0.5, 
            #                             accel_safeguard_factor=1, 
            #                             accel_max_weight_norm=1e6,
            #                             damping_post_acceleration=4
            #                             )
            # sol_c.scale_P(p)
            # sol_c.compute_non_solver_quantities(p)
            # if sol.status == 'successful':
            #     p.guess = sol_c.vector_from_var()
            # else:
            #     print(country,delt,'failed')
            #     sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
            #                             context = 'counterfactual',
            #                             cobweb_anim=False,tol =1e-10,
            #                             accelerate=False,
            #                             accelerate_when_stable=False,
            #                             cobweb_qty='phi',
            #                             plot_convergence=False,
            #                             plot_cobweb=False,
            #                             safe_convergence=0.001,
            #                             disp_summary=False,
            #                             damping = 10,
            #                             max_count = 1.5e3,
            #                             accel_memory = 50, 
            #                             accel_type1=True, 
            #                             accel_regularization=1e-10,
            #                             accel_relaxation=0.5, 
            #                             accel_safeguard_factor=1, 
            #                             accel_max_weight_norm=1e6,
            #                             damping_post_acceleration=5
            #                             )
            #     sol_c.scale_P(p)
            #     sol_c.compute_non_solver_quantities(p)
            # if sol.status == 'successful':
            #     p.guess = sol_c.vector_from_var()
            # else:
            #     print(country,delt,'failed2')
            #     sol, sol_c = fixed_point_solver_with_entry_costs(p,x0=p.guess,
            #                             context = 'counterfactual',
            #                             cobweb_anim=False,tol =1e-4,
            #                             accelerate=False,
            #                             accelerate_when_stable=False,
            #                             cobweb_qty='phi',
            #                             plot_convergence=False,
            #                             plot_cobweb=False,
            #                             safe_convergence=0.001,
            #                             disp_summary=False,
            #                             damping = 10,
            #                             max_count = 1e3,
            #                             accel_memory = 50, 
            #                             accel_type1=True, 
            #                             accel_regularization=1e-10,
            #                             accel_relaxation=0.5, 
            #                             accel_safeguard_factor=1, 
            #                             accel_max_weight_norm=1e6,
            #                             damping_post_acceleration=5
            #                             )
            #     sol_c.scale_P(p)
            #     sol_c.compute_non_solver_quantities(p)
            #     if sol.status == 'successful':
            #         p.guess = sol_c.vector_from_var()
            #     else:
            #         print(country,delt,'failed3')
            #         p.guess = None
            p.guess = sol_c.vector_from_var()
            
            if sol_baseline is not None:
                sol_c.compute_non_solver_quantities(p)
                sol_c.compute_consumption_equivalent_welfare(p, sol_baseline)
                print(sol_c.g)
                print(sol_c.cons)
                print(sol_c.price_indices)
                print(sol_c.cons_eq_welfare)
        p.write_params(country_path+'/'+str(i)+'/') 
        
#%% compute counterfactuals with double delta

def make_counterfactual_double_delta(p_baseline,country,local_path,
                        delta_int_factor_array=None,
                        delta_dom_factor_array=None,
                        delta_to_change='both',#can be 'dom','int',or 'both'
                        sol_baseline=None,
                        dynamics=False,
                        Nt=25,t_inf=500):
    country_path = local_path+country+'/'
    try:
        os.mkdir(country_path)
    except:
        pass

    print(country)
    p = p_baseline.copy()
    delta_factor_array = np.logspace(-1,1,11)
    if country in p_baseline.countries:
        idx_country = p_baseline.countries.index(country)
        
    for i,delt in enumerate(delta_factor_array):
        print(i)
        if i>-1:
            if country in p_baseline.countries:
                if delta_to_change == 'dom' or delta_to_change == 'both':
                    p.delta_dom[idx_country,1] = p_baseline.delta_dom[idx_country,1] * delt
                    print(p.delta_dom[idx_country,1])
                if delta_to_change == 'int' or delta_to_change == 'both':
                    p.delta_int[idx_country,1] = p_baseline.delta_int[idx_country,1] * delt
                p.update_delta_eff()
                
            sol, sol_c = fixed_point_solver_double_diff_double_delta(p,
                                                             x0=p.guess,
            context = 'counterfactual',
            cobweb_anim=False,tol =1e-10,
            accelerate=True,
            accelerate_when_stable=True,
            cobweb_qty='phi',
            plot_convergence=False,
            plot_cobweb=False,
            safe_convergence=0.001,
            disp_summary=True,
            damping = 10,
            max_count = 1000,
            accel_memory =50, 
            accel_type1=True, 
            accel_regularization=1e-10,
            accel_relaxation=0.5, 
            accel_safeguard_factor=1, 
            accel_max_weight_norm=1e6,
            damping_post_acceleration=5
            )
            sol_c.scale_P(p)
            p.guess = sol_c.vector_from_var()
            sol_c.compute_non_solver_quantities(p)
            
            if dynamics:
                # sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_init=sol_baseline,
                #                                                                  sol_fin=sol_c,
                #                         Nt=Nt,t_inf=t_inf,x0=p.dyn_guess,
                #                         cobweb_anim=False,tol =1e-6,
                #                         accelerate=False,
                #                         accelerate_when_stable=False,
                #                         cobweb_qty='l_R',
                #                         plot_convergence=False,
                #                         plot_cobweb=False,
                #                         plot_live = False,
                #                         safe_convergence=1e-8,
                #                         disp_summary=True,
                #                         damping = 20,
                #                         max_count = 50000,
                #                         accel_memory =5, 
                #                         accel_type1=True, 
                #                         accel_regularization=1e-10,
                #                         accel_relaxation=1, 
                #                         accel_safeguard_factor=1, 
                #                         accel_max_weight_norm=1e6,
                #                         damping_post_acceleration=5
                #                         )
                sol, dyn_sol_c = dyn_fixed_point_solver_double_diff_double_delta(p, sol_baseline, Nt=25,
                                                      t_inf=500,x0=p.dyn_guess,
                                        cobweb_anim=False,tol =1e-12,
                                        accelerate=False,
                                        accelerate_when_stable=False,
                                        cobweb_qty='l_R',
                                        plot_convergence=False,
                                        plot_cobweb=False,
                                        plot_live = False,
                                        safe_convergence=1e-4,
                                        disp_summary=True,
                                        damping = 60,
                                        max_count = 10000,
                                        accel_memory =5, 
                                        accel_type1=True, 
                                        accel_regularization=1e-10,
                                        accel_relaxation=1, 
                                        accel_safeguard_factor=1, 
                                        accel_max_weight_norm=1e6,
                                        damping_post_acceleration=5
                                        )
                if sol.status == 'successful':
                    p.dyn_guess = dyn_sol_c.vector_from_var()
                else:
                    p.dyn_guess = None
                    p.dyn_guess = dyn_sol_c.vector_from_var()
                    print('failed',country_path+'/'+str(i)+'/')
            
        p.write_params(country_path+'/'+str(i)+'/') 