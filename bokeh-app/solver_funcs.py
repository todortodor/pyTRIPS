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
from classes import cobweb, var, sol_class, moments, parameters

def get_vec_qty(x,p):
    res = {'w':x[0:p.N],
           'Z':x[p.N:p.N*2],
           'l_R':x[p.N*2:p.N*2+p.N*(p.S-1)],
           'psi_star':x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2],
           'phi':x[p.N*2+p.N*(p.S-1)+p.N**2:]
           }
    return res

def bound_psi_star(x,p,hit_the_bound=None):
    x_psi_star = x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2]
    if np.any(x_psi_star<1):
        hit_the_bound += 1
        x_psi_star[x_psi_star<1] = 1
    x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2] = x_psi_star
    return x, hit_the_bound

def bound_research_labor(x,p,hit_the_bound=None):
    x_l_R = x[p.N*3:p.N*3+p.N*(p.S-1)]
    if np.any(x_l_R > p.labor.max()):
        if hit_the_bound is not None:
            hit_the_bound+=1
        x_l_R[x_l_R > p.labor.max()] = p.labor.max()
    x[p.N*3:p.N*3+p.N*(p.S-1)] = x_l_R
    return x,hit_the_bound

def bound_zero(x, cutoff=1e-8, hit_the_bound=None):
    if np.any(x<=0):
        x[x <= 0] = cutoff
        if hit_the_bound is not None:
            hit_the_bound+=1
    return x,hit_the_bound

def smooth_large_jumps(x_new,x_old):
    high_jumps_too_big = x_new > 1000*x_old
    while np.any(high_jumps_too_big):
        x_new = x_old*1/2+x_new*1/2
        high_jumps_too_big = x_new > 1000*x_old
    low_jumps_too_big = x_new < x_old/1000
    while np.any(low_jumps_too_big):
        x_new = x_old*1/2+x_new*1/2
        low_jumps_too_big = x_new < x_old/1000
    return x_new

def fixed_point_solver(p, x0=None, tol = 1e-10, damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, cobweb_anim=False, cobweb_qty='psi_star',
                       cobweb_coord = 1, plot_convergence = True, apply_bound_zero = True, 
                       apply_bound_psi_star = False, apply_bound_research_labor = False,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):   
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
            x_new = smooth_large_jumps(x_new,x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        if apply_bound_psi_star:
            x_old, hit_the_bound_count = bound_psi_star(x_old, p, hit_the_bound_count)
        if apply_bound_research_labor:
            x_old, hit_the_bound_count = bound_research_labor(x_old, p, hit_the_bound_count) 
            
        init = var.var_from_vector(x_old,p,compute=False)
        
        init.compute_growth(p)
        init.compute_aggregate_qualities(p)
        init.compute_sectoral_prices(p)
        init.compute_trade_shares(p)
        init.compute_labor_allocations(p)
        init.compute_price_indices(p)
        
        w = init.compute_wage(p)
        Z = init.compute_expenditure(p)
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        psi_star = init.compute_psi_star(p)[...,1:].ravel()
        phi = init.compute_phi(p).ravel()
        
        x_new = np.concatenate((w,Z,l_R,psi_star,phi), axis=0)

        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','psi_star','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        count += 1
        if np.all(np.array(convergence[-10:])<safe_convergence):
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
    # dev_norm = deviation_norm(x_new,p)
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
                cob.plot(count=i, window = 100,pause = 0.05) 
        cob.plot(count = count, window = 200)
            
    if plot_convergence:
        plt.semilogy(convergence, label = 'convergence')
        plt.semilogy(norm, label = 'norm')
        plt.legend()
        plt.show()
    return sol_inst, init

def calibration_func(vec_parameters,p,m,v0=None,hist=None,start_time=0):
    p.update_parameters(vec_parameters)
    # print(vec_parameters)
    try:
        v0 = p.guess
    except:
        pass
    sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-14,
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
    if sol.status == 'failed': 
        print('trying less precise')
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-12,
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
        sol, sol_c = fixed_point_solver(p,x0=None,tol=1e-12,
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
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-13,
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
        sol, sol_c = fixed_point_solver(p,x0=v0,tol=1e-13,
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
    sol_c.compute_price_indices(p)
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
            print('fe : ',p.fe[1],'fo : ',p.fo[1], 'delta : ', p.delta[:,1]
                  , 'nu : ', p.nu[1], 'nu_tilde : ', p.nu_tilde[1], 'k :', p.k
                  , 'theta :', p.theta[1], 'sigma :', p.sigma[1], 'zeta :', p.zeta[1]
                  , 'rho :', p.rho, 'kappa :', p.kappa, 'd : ', p.d)
    hist.count += 1
    p.guess = sol_c.vector_from_var()
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        return np.full_like(m.deviation_vector(),1e10)
    else:
        return m.deviation_vector() 
    
def full_load_and_solve(path,list_of_moments = None):
    if list_of_moments is None:
        list_of_moments = moments.get_list_of_moments()
    p = parameters()
    p.load_data(path)
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='phi',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.001,
                            disp_summary=False,
                            damping = 10,
                            max_count = 3e3,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=5
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    sol_c.compute_price_indices
    sol_c.compute_non_solver_quantities(p)
    m = moments(list_of_moments)
    m.load_data()
    m.load_run(path)
    m.compute_moments(sol_c,p)
    # m.compute_Z(sol_c,p)
    m.compute_moments_deviations()
    return p,sol_c,m    