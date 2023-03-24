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
from classes import cobweb, sol_class, moments, parameters, var, history_nash, dynamic_var
# import pandas as pd
from scipy import optimize

def get_vec_qty(x,p):
    res = {'w':x[0:p.N],
           'Z':x[p.N:p.N+p.N],
           'l_R':x[p.N+p.N:p.N+p.N+p.N*(p.S-1)],
           'profit':x[p.N+p.N+p.N*(p.S-1):p.N+p.N+p.N*(p.S-1)+p.N**2],
           'phi':x[p.N+p.N+p.N*(p.S-1)+p.N**2:]
           }
    return res

# def bound_psi_star(x,p,hit_the_bound=None):
#     x_psi_star = x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2]
#     if np.any(x_psi_star<1):
#         hit_the_bound += 1
#         x_psi_star[x_psi_star<1] = 1
#     x[p.N*2+p.N*(p.S-1):p.N*2+p.N*(p.S-1)+p.N**2] = x_psi_star
#     return x, hit_the_bound

# def bound_research_labor(x,p,hit_the_bound=None):
#     x_l_R = x[p.N*3:p.N*3+p.N*(p.S-1)]
#     if np.any(x_l_R > p.labor.max()):
#         if hit_the_bound is not None:
#             hit_the_bound+=1
#         x_l_R[x_l_R > p.labor.max()] = p.labor.max()
#     x[p.N*3:p.N*3+p.N*(p.S-1)] = x_l_R
#     return x,hit_the_bound

def bound_zero(x, cutoff=1e-8, hit_the_bound=None):
    if np.any(x<=0):
        x[x <= 0] = cutoff
        if hit_the_bound is not None:
            hit_the_bound+=1
    return x,hit_the_bound

def smooth_large_jumps(x_new,x_old):
    high_jumps_too_big = x_new > 1000*x_old
    while np.any(high_jumps_too_big):
        # print(x_new.max())
        x_new = x_old*1/2+x_new*1/2
        high_jumps_too_big = x_new > 1000*x_old
    low_jumps_too_big = x_new < x_old/1000
    while np.any(low_jumps_too_big):
        # print('bip')
        x_new = x_old*1/2+x_new*1/2
        low_jumps_too_big = x_new < x_old/1000
    return x_new

def fixed_point_solver(p, context, x0=None, tol = 1e-15, damping = 10, max_count=1e4,
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
            # x_new = smooth_large_jumps(x_new,x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
        if apply_bound_zero:
            x_old, hit_the_bound_count = bound_zero(x_old,1e-12, hit_the_bound_count)
        # if apply_bound_research_labor:
        #     x_old, hit_the_bound_count = bound_research_labor(x_old, p, hit_the_bound_count) 
        init = var.var_from_vector(x_old,p,context=context,compute=False)
        # init.phi = init.phi/np.diagonal(init.phi).transpose()[:,None,:]
        # x_old = init.vector_from_var()
        # init.compute_growth(p)
        # init.compute_aggregate_qualities(p)
        # init.compute_sectoral_prices(p)
        # init.compute_trade_shares(p)
        # init.compute_labor_allocations(p)
        # init.compute_price_indices(p)
        init.compute_solver_quantities(p)
        # init.scale_tau(p)
        # init.scale_P(p)
        # print(init.price_indices)
        
        w = init.compute_wage(p)#/init.price_indices[0]
        Z = init.compute_expenditure(p)#/init.price_indices[0]
        l_R = init.compute_labor_research(p)[...,1:].ravel()
        profit = init.compute_profit(p)[...,1:].ravel()
        # psi_star = init.compute_psi_star(p)[...,1:].ravel()
        # psi_star[psi_star<1] = 1
        phi = init.compute_phi(p).ravel()#*init.price_indices[0]
        
        x_new = np.concatenate((w,Z,l_R,profit,phi), axis=0)
        
        # print(x_new)
        
        x_new_decomp = get_vec_qty(x_new,p)
        x_old_decomp = get_vec_qty(x_old,p)
        conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
                      for qty in ['w','Z','profit','l_R','phi']]
        condition = np.any(conditions)
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        # print(count)
        if plot_live and count>100:
            # plt.semilogy(convergence)
            plt.plot(x_new)
            plt.show()
        
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
    # dev_norm = deviation_norm(x_new,p)
    dev_norm = 'TODO'
    # print(w,Z,l_R,psi_star,phi)
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

def repeat_for_all_times(array,Nt):
    return np.repeat(array[..., np.newaxis],Nt,axis=len(array.shape))

def guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin,C=20):
    def build_guess(fin,init,C=C):
        if len(fin.shape) == 2:
            return (fin-init)[...,1:,None]*(
                np.exp( -C* (dyn_var.t_cheby+1) )[None,None,:]-1
                )/(np.exp(-2*C)-1)
            # return (fin-init)[...,1:,None]*(
            #     np.exp( -C* (np.linspace(-1,1,dyn_var.Nt)+1) )[None,None,:]-1
            #     )/(np.exp(-2*C)-1)
            # return (fin-init)[...,1:,None]*(
            #     1 - np.exp( -C*dyn_var.map_parameter*(1+dyn_var.t_cheby)/(1-dyn_var.t_cheby) )[None,None,:]
            #     )
        elif len(fin.shape) == 3:
            return (fin-init)[...,1:,None]*(
                np.exp(-C* (dyn_var.t_cheby+1) )[None,None,None,:]-1
                )/(np.exp(-2*C)-1)
            # return (fin-init)[...,1:,None]*(
            #     np.exp(-C* (np.linspace(-1,1,dyn_var.Nt)+1) )[None,None,None,:]-1
            #     )/(np.exp(-2*C)-1)
            # return (fin-init)[...,1:,None]*(
            #     1 - np.exp( -C*dyn_var.map_parameter*(1+dyn_var.t_cheby)/(1-dyn_var.t_cheby) )[None,None,None,:]
            #     )
    guess = {}
    # guess['PSI_CD'] = (dyn_var.PSI_CD_0-sol_fin.PSI_CD)[:,:,None]*(np.exp(1-np.linspace(-1,1,dyn_var.Nt))-np.exp(2))[None,None,:]
    # guess['PSI_MNP'] = (dyn_var.PSI_MNP_0-sol_fin.PSI_MNP)[...,None]*(np.exp(1-np.linspace(-1,1,dyn_var.Nt))-np.exp(2))[None,None,None,:]
    # guess['PSI_MPND'] = (dyn_var.PSI_MPND_0-sol_fin.PSI_MPND)[...,None]*(np.exp(1-np.linspace(-1,1,dyn_var.Nt))-np.exp(2))[None,None,None,:]
    # guess['PSI_MPD'] = (dyn_var.PSI_MPD_0-sol_fin.PSI_MPD)[...,None]*(np.exp(1-np.linspace(-1,1,dyn_var.Nt))-np.exp(2))[None,None,None,:]
    guess['PSI_CD'] = build_guess(sol_fin.PSI_CD,dyn_var.PSI_CD_0)
    guess['PSI_MNP'] = build_guess(sol_fin.PSI_MNP,dyn_var.PSI_MNP_0)
    guess['PSI_MPND'] = build_guess(sol_fin.PSI_MPND,dyn_var.PSI_MPND_0)
    guess['PSI_MPD'] = build_guess(sol_fin.PSI_MPD,dyn_var.PSI_MPD_0)
    return guess
    
def dyn_fixed_point_solver(p, sol_init, sol_fin = None,t_inf=200, Nt=500, x0=None, tol = 1e-10, damping = 10, max_count=1e6,
                       accelerate = False, safe_convergence=0.1,accelerate_when_stable=True, 
                       plot_cobweb = True, plot_live = False, cobweb_anim=False, cobweb_qty='profit',
                       cobweb_coord = 1, plot_convergence = True,
                       accel_memory = 10, accel_type1=False, accel_regularization=1e-12,
                       accel_relaxation=1, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
                       disp_summary=True,damping_post_acceleration=5):  
    
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
    
    dyn_var = dynamic_var(nbr_of_time_points = Nt,t_inf=t_inf,sol_init=sol_init,sol_fin=sol_fin)
    dyn_var.initiate_state_variables_0(sol_init)
    
    psis_guess = guess_PSIS_from_sol_init_and_sol_fin(dyn_var,sol_init,sol_fin)
    
    dic_of_guesses = {'price_indices':repeat_for_all_times(sol_fin.price_indices,dyn_var.Nt),
                    'w':repeat_for_all_times(sol_fin.w,dyn_var.Nt),
                    'Z':repeat_for_all_times(sol_fin.Z,dyn_var.Nt),
                    # 'PSI_CD':np.zeros((p.N,p.S,dyn_var.Nt)),(dyn_var.PSI_CD_0-sol_fin.PSI_CD)[:,:,None]*np.exp(np.linspace(-1,1,))
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
    # dic_of_guesses = {'price_indices':repeat_for_all_times(sol_init.price_indices,dyn_var.Nt),
    #                 'w':repeat_for_all_times(sol_init.w,dyn_var.Nt),
    #                 'Z':repeat_for_all_times(sol_init.Z,dyn_var.Nt),
    #                 # 'PSI_CD':np.zeros((p.N,p.S,dyn_var.Nt)),(dyn_var.PSI_CD_0-sol_fin.PSI_CD)[:,:,None]*np.exp(np.linspace(-1,1,))
    #                 'PSI_CD':np.zeros((p.N,p.S,dyn_var.Nt))[...,1:,:],
    #                 'PSI_MNP':np.zeros((p.N,p.N,p.S,dyn_var.Nt))[...,1:,:],
    #                 'PSI_MPND':np.zeros((p.N,p.N,p.S,dyn_var.Nt))[...,1:,:],
    #                 'PSI_MPD':np.zeros((p.N,p.N,p.S,dyn_var.Nt))[...,1:,:],
    #                 # 'PSI_CD':repeat_for_all_times(sol_fin.PSI_CD-sol_init.PSI_CD,dyn_var.Nt)[...,1:,:],
    #                 # 'PSI_MNP':repeat_for_all_times(sol_fin.PSI_MNP-sol_init.PSI_MNP,dyn_var.Nt)[...,1:,:],
    #                 # 'PSI_MPND':repeat_for_all_times(sol_fin.PSI_MPND-sol_init.PSI_MPND,dyn_var.Nt)[...,1:,:],
    #                 # 'PSI_MPD':repeat_for_all_times(sol_fin.PSI_MPD-sol_init.PSI_MPD,dyn_var.Nt)[...,1:,:],
    #                 'V_PD':repeat_for_all_times(sol_init.V_PD,dyn_var.Nt)[...,1:,:],
    #                 'V_P':repeat_for_all_times(sol_init.V_P,dyn_var.Nt)[...,1:,:],
    #                 'V_NP':repeat_for_all_times(sol_init.V_NP,dyn_var.Nt)[...,1:,:]}
    dyn_var.guess_from_dic(dic_of_guesses)
    # dyn_var.Z = dyn_var.Z*0.9
    # plt.plot(dyn_var.PSI_CD[0,1,:])
    # plt.show()
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
        # print(count)
        if count != 0:
            if accelerate:
                aa_wrk.apply(x_new, x_old)
            x_old = (x_new+(damping-1)*x_old)/damping
            dyn_var.guess_from_vector(x_old)
            # for qty in ['PSI_CD','PSI_MNP','PSI_MPND','PSI_MPD']:
            #     temp = getattr(dyn_var,qty)
            #     temp[...,0] = getattr(sol_fin,qty) - getattr(sol_init,qty)
            #     temp[...,-1] = 0
            #     setattr(dyn_var,qty,temp)
            #     # dyn_var.PSI_CD[...,0] = sol_fin.PSI_CD-dyn_var.PSI_CD_0
            #     # dyn_var.PSI_CD[...,-1] = 0
            # for qty in ['price_indices','w','Z','V_PD','V_P','V_NP']:
            #     temp = getattr(dyn_var,qty)
            #     temp[...,0] = getattr(sol_fin,qty)
            #     temp[...,-1] = getattr(sol_init,qty)
            #     setattr(dyn_var,qty,temp)
            numeraire = dyn_var.price_indices[0,:]
            for qty in ['price_indices','w','Z']:
                temp = getattr(dyn_var,qty)
                temp = temp/numeraire[None,:]
                # temp[...,:] = getattr(sol_init,qty)
                setattr(dyn_var,qty,temp)
            # for qty in ['V_PD','V_P','V_NP']:
            for qty in ['V_PD','DELTA_V','V_NP']:
                temp = getattr(dyn_var,qty)
                temp = temp/numeraire[None,None,None:]
                # temp[...,:] = getattr(sol_init,qty)
                setattr(dyn_var,qty,temp)
            x_old = dyn_var.vector_from_var()
        # dyn_var.plot_country(0,title = str(count))
        # dyn_var.compute_phi(p)
        # dyn_var.compute_PSI_M(p)
        # dyn_var.compute_sectoral_prices(p)
        # dyn_var.compute_patenting_thresholds(p)
        # dyn_var.compute_V(p)
        # dyn_var.compute_labor_research(p)
        # dyn_var.compute_growth(p)
        # dyn_var.compute_labor_allocations(p)
        # dyn_var.compute_trade_flows_and_shares(p)
        # dyn_var.compute_profit(p)
        # dyn_var.compute_nominal_final_consumption(p)
        # dyn_var.compute_interest_rate(p)
        # x_old[:dyn_var.Nt] = 1
        if plot_live:
            if count == 0:
                dyn_var.plot_country(0,title = str(count),initial = True)
                # input('enter')
        if plot_live:
            if count<70 and count >0:
                dyn_var.plot_country(0,title = str(count))
        dyn_var.compute_solver_quantities(p)
        
        # input('enter')
        # x_new = dyn_var.vector_from_var()
        x_new = np.concatenate([
            dyn_var.compute_price_indices(p).ravel(),
            dyn_var.compute_wage(p).ravel(),
            dyn_var.compute_expenditure(p).ravel(),
            dyn_var.compute_PSI_CD(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MNP(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPND(p)[...,1:,:].ravel(),
            dyn_var.compute_PSI_MPD(p)[...,1:,:].ravel(),
            dyn_var.compute_V_PD(p)[...,1:,:].ravel(),
            # dyn_var.compute_V_P(p)[...,1:,:].ravel(),
            dyn_var.compute_DELTA_V(p)[...,1:,:].ravel(),
            dyn_var.compute_V_NP(p)[...,1:,:].ravel(),
            ],axis=0)

        # x_new_decomp = get_vec_qty(x_new,p)
        # x_old_decomp = get_vec_qty(x_old,p)
        # conditions = [np.linalg.norm(x_new_decomp[qty] - x_old_decomp[qty])/np.linalg.norm(x_old_decomp[qty]) > tol
        #               for qty in ['w','Z','profit','l_R','phi']]
        # condition = np.any(conditions)
        condition = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old) > tol
        convergence.append(np.linalg.norm(x_new - x_old)/np.linalg.norm(x_old))
        # dyn_var.plot_country(0,title = str(count))
        # dyn_var.plot_numerical_derivatives(title = str(count))
        # plt.plot(x_new)
        # plt.show()
        # print(x_new)
        # if count%20 == 0:
        #     plt.semilogy(convergence)
        #     plt.show()
        # if plot_live and count>100:
        #     # plt.semilogy(convergence)
        #     plt.plot(x_new)
        #     plt.show()
        
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
            
        # plt.semilogy(convergence, label = 'convergence')
        # plt.semilogy(norm, label = 'norm')
        # plt.legend()
        # plt.show()
    
    finish = time.perf_counter()
    solving_time = finish-start
    # dev_norm = deviation_norm(x_new,p)
    dev_norm = 'TODO'
    # print(w,Z,l_R,psi_star,phi)
    if count < max_count and np.isnan(x_new).sum()==0 and np.all(x_new<1e40):
        status = 'successful'
    else:
        status = 'failed'
        
    if status == 'failed':
        print('Failed, report :')
        print('count',count)
        print('nans',np.isnan(x_new).sum())
        print('diverged',(x_new>1e40).sum())
    
    # print(x_new)
    # dyn_var.compute_non_solver_quantities(p)
    # dyn_var.plot_country(0,title = 'USA')
    # dyn_var.plot_numerical_derivatives()
    
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
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    p.delta[0,1] = back_up_delta
    
    return (sol_c.cons_eq_welfare[0]-1)/epsilon

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
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    p.delta[0,1] = back_up_delta
    
    return (sol_c.g-sol_baseline.g)/epsilon

def calibration_func(vec_parameters,p,m,v0=None,hist=None,start_time=0,
                     avoid_bad_nash=False,bad_nash_weight = None):
    p.update_parameters(vec_parameters)
    if 'khi' in p.calib_parameters:
        p.update_khi_and_r_hjort(p.khi)
    # print(vec_parameters)
    try:
        v0 = p.guess
    except:
        pass
    # print('here')
    sol, sol_c = fixed_point_solver(p,context = 'calibration', x0=v0,
                            cobweb_anim=False,tol =1e-14,
                            accelerate=False,
                            accelerate_when_stable=True,
                            cobweb_qty='l_R',
                            plot_convergence=False,
                            plot_cobweb=False,
                            safe_convergence=0.1,
                            disp_summary=False,
                            damping = 2,
                            max_count = 1000,
                            accel_memory = 50, 
                            accel_type1=True, 
                            accel_regularization=1e-10,
                            accel_relaxation=0.5, 
                            accel_safeguard_factor=1, 
                            accel_max_weight_norm=1e6,
                            damping_post_acceleration=1
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    # print('here')
    if sol.status == 'failed': 
        print('trying standard guess')
        sol, sol_c = fixed_point_solver(p,context = 'calibration',x0=None,
                                cobweb_anim=False,tol =1e-14,
                                accelerate=False,
                                accelerate_when_stable=True,
                                cobweb_qty='l_R',
                                plot_convergence=False,
                                plot_cobweb=False,
                                safe_convergence=0.1,
                                disp_summary=False,
                                damping = 2,
                                max_count = 1000,
                                accel_memory = 50, 
                                accel_type1=True, 
                                accel_regularization=1e-10,
                                accel_relaxation=0.5, 
                                accel_safeguard_factor=1, 
                                accel_max_weight_norm=1e6,
                                damping_post_acceleration=1
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
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
    if avoid_bad_nash:
        US_deriv_w_to_d = np.array(compute_deriv_welfare_to_patent_protec_US(sol_c,p,v0))
        US_cost_w_to_d = (US_deriv_w_to_d<0)*np.abs(US_deriv_w_to_d)*bad_nash_weight
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
                  , 'rho :', p.rho, 'kappa :', p.kappa, 'd : ', p.d, 'r_hjort : ', p.r_hjort)
        if avoid_bad_nash and hist.count%10==0:
            print('Nash : ',US_deriv_w_to_d, 'Cost : ', US_cost_w_to_d)
    hist.count += 1
    p.guess = sol_c.vector_from_var()
    # print(p.guess.shape)
    if np.any(np.isnan(p.guess)) or sol.status == 'failed':
        print('failed')
        p.guess = None
        return np.full_like(m.deviation_vector(),1e10)
    else:
        if avoid_bad_nash:
            return np.concatenate([m.deviation_vector(), US_cost_w_to_d[None]])
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
    # sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices
    sol_c.compute_non_solver_quantities(p)
    m = moments(list_of_moments)
    m.load_data()
    m.load_run(path)
    m.compute_moments(sol_c,p)
    # m.compute_Z(sol_c,p)
    m.compute_moments_deviations()
    return p,sol_c,m    
    
def is_oscillating(deltas,column,window):
    a = deltas[column].values[-window:]
    a = a-a.mean()
    sign_changes = np.where(np.sign(a[:-1]) != np.sign(a[1:]))
    # return ((window_values-window_values.mean())>=0).sum() >= window/2 and ((window_values-window_values.mean())<=0).sum() >= window/2
    # if a.shape == sign_changes.shape:
    #     return np.allclose(np.where(np.sign(a[:-1]) != np.sign(a[1:])) , np.arange(window-1))
    return (sign_changes[0].shape[0] >= window-2) or (np.all(a == 0))

def minus_welfare_of_delta(delta,p,c,sol_it_baseline, hist = None, reverse_search=False,
                           dynamics=False):
    # print('solving')
    back_up_delta_value = p.delta[p.countries.index(c),1]
    # p.delta[p.countries.index(c),1] = 10**delta
    # p.delta[p.countries.index(c),1] = -delta
    p.delta[p.countries.index(c),1] = delta
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
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    # sol_c = var.var_from_vector(sol.x, p)    
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
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
        
    # print(hist.current_deltas)
    
    
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
        # time.sleep(5)
    p.delta[p.countries.index(c),1] = back_up_delta_value
    p.guess = sol_c.vector_from_var()
    
    return welfare
    # return [-sol_c.cons_eq_welfare[p.countries.index(c)]]
     

def minus_welfare_of_delta_pop_weighted(deltas,p,sol_baseline):
    p.delta[...,1] = deltas
    print(p.guess)
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                                    context = 'counterfactual',
                            cobweb_anim=False,tol =1e-15,
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
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    
    
    return -sol_c.pop_average_welfare_change

def minus_welfare_of_delta_negishi_weighted(deltas,p,sol_baseline):
    p.delta[...,1] = deltas
    sol, sol_c = fixed_point_solver(p,x0=p.guess,
                            cobweb_anim=False,tol =1e-15,
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
                            # damping=10
                              # apply_bound_psi_star=True
                            )
    sol_c = var.var_from_vector(sol.x, p)    
    # sol_c.scale_tau(p)
    sol_c.scale_P(p)
    # sol_c.compute_price_indices(p)
    sol_c.compute_non_solver_quantities(p)
    sol_c.compute_consumption_equivalent_welfare(p,sol_baseline)
    sol_c.compute_world_welfare_changes(p, sol_baseline)
    return -sol_c.negishi_welfare_change
    
def compute_new_deltas_fixed_point(p, sol_it_baseline, lb_delta, ub_delta, hist_nash = None, 
                                   reverse_search = False, dynamics=False):
    new_deltas = np.zeros(len(p.countries))
    # if reverse_search:
    #     bounds = (-ub_delta, -lb_delta)
    # else:
    bounds=(lb_delta, ub_delta)
    for i,c in enumerate(p.countries):
        
        
        if dynamics:
            delta_min = optimize.shgo(func=minus_welfare_of_delta,
                                                  # sampling_method='halton',
                                                   bounds=[bounds],
                                                  args = (p,c,sol_it_baseline, hist_nash, reverse_search, dynamics),
                                                    options={'disp':True},
                                                  # tol=1e-8
                                                  )
        else:
            delta_min = optimize.minimize_scalar(fun=minus_welfare_of_delta,
                                                  method='bounded',
                                                    bounds=bounds,
                                                  args = (p,c,sol_it_baseline, hist_nash, reverse_search, dynamics),
                                                    # options={'disp':3},
                                                  tol=1e-8
                                                  )
        # rranges = (slice(lb_delta, ub_delta, 0.01),)
        # delta_min = optimize.brute(func=minus_welfare_of_delta,#x0=np.array(p.delta[p.countries.index(c),1]),
        #                                      # method='brute',
        #                                      # bounds=(np.log10(lb_delta), np.log10(ub_delta)),
        #                                        # bounds=(lb_delta, ub_delta),
        #                                       ranges=(slice(lb_delta, ub_delta, 0.01),),
        #                                      args = (p,c,sol_it_baseline, hist_nash),
        #                                        # options={'disp':3},
        #                                       # tol=1e-8
        #                                      )
        # new_deltas[i] = 10**delta_min.x
        # new_deltas[i] = (1-2*reverse_search)*delta_min.x
        new_deltas[i] = delta_min.x
        if hist_nash is not None:
            hist_nash.expected_deltas[i] = new_deltas[i]
            # hist_nash.expected_deltas[i] = -delta_min.x
            hist_nash.expected_welfare[i] = delta_min.fun
        print(c,new_deltas)
            
    if hist_nash is not None:
        # input("Press Enter to run next iteration")
        hist_nash.make_a_pause = True
                
    return new_deltas
    # return -new_deltas

def find_nash_eq(p_baseline,lb_delta=0.01,ub_delta=100,method='fixed_point',dynamics=False,
                 plot_convergence = False,solver_options=None,tol=5e-5,window=4,
                 damping = 1,plot_history = False,reverse_search=False):
    
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
                                max_count = 3e3,
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
    sol_baseline.compute_price_indices(p_baseline)
    sol_baseline.compute_non_solver_quantities(p_baseline)    

    condition = True
    deltas = p_baseline.delta[...,1][:,None]
    welfares = np.ones(len(p_baseline.countries))[:,None]
    p_it_baseline = p_baseline.copy()
    sol_it_baseline = sol_baseline.copy()
    # all_oscillating = False
    # buffer = 0

    it = 0
    x_old = p_baseline.delta[...,1]
    convergence = []
    new_deltas = None
    
    # accel_memory = 10
    # accel_type1=False
    # accel_regularization=1e-12
    # accel_relaxation=1
    # accel_safeguard_factor=1 
    # accel_max_weight_norm=1e6
    # aa_options = {'dim': len(x_old),
    #             'mem': accel_memory,
    #             'type1': accel_type1,
    #             'regularization': accel_regularization,
    #             'relaxation': accel_relaxation,
    #             'safeguard_factor': accel_safeguard_factor,
    #             'max_weight_norm': accel_max_weight_norm}
    # aa_wrk = aa.AndersonAccelerator(**aa_options)
    
    if plot_history:
        hist_nash = history_nash()
        hist_nash.update_current_deltas(x_old)
        hist_nash.update_current_welfare(-np.ones(len(x_old)))
    else:
        hist_nash = None
        
    while condition:
        print(it)
        if it != 0:
            # aa_wrk.apply(new_deltas, x_old)
            x_old = (new_deltas+(damping-1)*x_old)/damping
            # x_old = new_deltas
            p_it_baseline.delta[...,1] = x_old
        
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        # sol_it_baseline = var.var_from_vector(sol.x, p_it_baseline)    
        sol_it_baseline.scale_P(p_it_baseline)
        # sol_it_baseline.compute_price_indices(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
            
        # new_deltas = compute_new_deltas_fixed_point(p_it_baseline, sol_it_baseline, lb_delta, ub_delta)
        new_deltas = compute_new_deltas_fixed_point(p_it_baseline, sol_baseline, lb_delta, 
                                                    ub_delta, hist_nash = hist_nash,reverse_search = reverse_search,
                                                    dynamics=dynamics)
        
        p_it_baseline.delta[...,1] = new_deltas
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
                                # damping=10
                                  # apply_bound_psi_star=True
                                )
        # sol_it_baseline = var.var_from_vector(sol.x, p_it_baseline)    
        sol_it_baseline.scale_P(p_it_baseline)
        # sol_it_baseline.compute_price_indices(p_it_baseline)
        sol_it_baseline.compute_non_solver_quantities(p_it_baseline)
        sol_it_baseline.compute_consumption_equivalent_welfare(p_it_baseline, sol_baseline)
        
        if dynamics:
            sol, dyn_sol_c = dyn_fixed_point_solver(p_it_baseline, sol_baseline, Nt=25,
                                                  t_inf=500,
                                    cobweb_anim=False,tol =1e-14,
                                    accelerate=False,
                                    accelerate_when_stable=False,
                                    cobweb_qty='l_R',
                                    plot_convergence=True,
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
            dyn_sol_c.compute_non_solver_quantities(p_it_baseline)
            
        
        if plot_history:
            # print(np.log(new_deltas))
            # print(sol_it_baseline.cons_eq_welfare)
            hist_nash.update_current_deltas(new_deltas)
            hist_nash.update_current_welfare(-sol_it_baseline.cons_eq_welfare)
        
        deltas = np.concatenate([deltas,new_deltas[:,None]],axis=1)
        
        
        if dynamics:
            welfares = np.concatenate([welfares,dyn_sol_c.cons_eq_welfare[:,None]],axis=1)
        else:
            welfares = np.concatenate([welfares,sol_it_baseline.cons_eq_welfare[:,None]],axis=1)
        
        condition = np.linalg.norm((new_deltas-x_old)/x_old)> tol
        
        convergence.append(np.linalg.norm((new_deltas - x_old)/x_old))
        
        # print(convergence)
        # print((new_deltas-x_old)/x_old)
        
        it += 1
        
        if it>5:
            damping = 5
        
        if plot_convergence:
                fig,ax = plt.subplots()
                
                ax2 = ax.twinx()
                ax.semilogy(deltas.transpose())
                ax2.plot(welfares.transpose(), ls = '--')
                plt.legend(labels = p_baseline.countries)
                # deltas.plot(logy=True,ax=ax, xlabel = 'Iteration', 
                #               ylabel = 'Delta', 
                #               title = 'Convergence to Nash equilibrium')
                # welfares.plot(ax=ax2, ls = '--', ylabel = 'Consumption eq. welfare')
                
                plt.show()
    return deltas, welfares

# def find_coop_eq(p_baseline,lb_delta=0.01,ub_delta=100,tol=5e-5):
    