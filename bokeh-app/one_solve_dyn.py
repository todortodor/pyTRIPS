#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:42:42 2023

@author: slepot
"""

from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, guess_PSIS_from_sol_init_and_sol_fin
import numpy as np

p = parameters(n=7,s=2)
p.load_data('calibration_results_matched_economy/baseline_404_variations/1.0/')

# sol, sol_c = fixed_point_solver(p,x0=p.guess,
#                                 context = 'calibration',
#                         cobweb_anim=False,tol =1e-14,
#                         accelerate=False,
#                         accelerate_when_stable=True,
#                         cobweb_qty='l_R',
#                         plot_convergence=True,
#                         plot_cobweb=True,
#                         safe_convergence=0.001,
#                         disp_summary=True,
#                         damping = 10,
#                         max_count = 1000,
#                         accel_memory =50, 
#                         accel_type1=True, 
#                         accel_regularization=1e-10,
#                         accel_relaxation=0.5, 
#                         accel_safeguard_factor=1, 
#                         accel_max_weight_norm=1e6,
#                         damping_post_acceleration=10
#                         )
# sol_c.scale_P(p)
# sol_c.compute_non_solver_quantities(p) 

#%%
from classes import moments, parameters, var, dynamic_var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver, guess_PSIS_from_sol_init_and_sol_fin


def repeat_for_all_times(array,Nt):
    return np.repeat(array[..., np.newaxis],Nt,axis=len(array.shape))

p = parameters(n=7,s=2)
p.load_data('calibration_results_matched_economy/baseline_403_variations/1.40/')

p_init = parameters(n=7,s=2)
p_init.load_data('calibration_results_matched_economy/baseline_404_variations/1.0/')

sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
                                context = 'calibration',
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
sol_init.scale_P(p_init)
sol_init.compute_non_solver_quantities(p_init) 

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

dyn_var = dynamic_var(nbr_of_time_points = 500)
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
                'V_PD':repeat_for_all_times(sol_fin.V_PD,dyn_var.Nt)[...,1:,:],
                'V_P':repeat_for_all_times(sol_fin.V_P,dyn_var.Nt)[...,1:,:],
                'V_NP':repeat_for_all_times(sol_fin.V_NP,dyn_var.Nt)[...,1:,:]}
dyn_var.guess_from_dic(dic_of_guesses)
dyn_var.compute_PSI_M(p)

# for i in range(7):
#     for j in range(7): 
#         plt.plot((dyn_var.PSI_M[i,j,1:,:]+dyn_var.PSI_M_0[i,j,1:,None]).ravel())

#%%
dyn_var.compute_phi(p)
dyn_var.compute_PSI_M(p)
dyn_var.compute_sectoral_prices(p)
dyn_var.compute_patenting_thresholds(p)
dyn_var.compute_V(p)
dyn_var.compute_labor_research(p)
dyn_var.compute_growth(p)
dyn_var.compute_labor_allocations(p)
dyn_var.compute_trade_flows_and_shares(p)
dyn_var.compute_profit(p)
dyn_var.compute_nominal_final_consumption(p)
dyn_var.compute_interest_rate(p)
# dyn_var.compute_solver_quantities(p)

for qty_to_check in ['phi','PSI_M','P_M','P_CD','psi_m_star','psi_o_star',
                     'V','l_R','g','g_s','l_Ae','l_Ao','X_M','X_CD','X','profit',
                     'nominal_final_consumption','r']:
    if qty_to_check == 'PSI_M':
        print(qty_to_check, 
              np.all(
                    [np.allclose(getattr(sol_c,'PSI_M'),  getattr(dyn_var,'PSI_M')[...,i]+getattr(dyn_var,'PSI_M_0')) 
                     for i in range(0,dyn_var.Nt)]
                    ) 
            )
    elif qty_to_check == 'profit':
        print(qty_to_check, 
              np.all(
                    [np.allclose(getattr(sol_c,'profit')*sol_c.w[None,:,None],  getattr(dyn_var,'profit')[...,i]) 
                     for i in range(0,dyn_var.Nt)]
                    ) 
            )
    else:
        print(qty_to_check, 
              np.all(
                    [np.allclose(getattr(sol_c,qty_to_check),  getattr(dyn_var,qty_to_check)[...,i]) 
                     for i in range(0,dyn_var.Nt)]
                    ) 
            )

Z_dyn_var = dyn_var.compute_expenditure(p)
print('Z', np.all(
                  [np.allclose(getattr(sol_c,'Z'),  Z_dyn_var[...,i]) 
                   for i in range(0,dyn_var.Nt)]
                  )
    )

w_dyn_var = dyn_var.compute_wage(p)
print('w', np.all(
                  [np.allclose(getattr(sol_c,'w'),  w_dyn_var[...,i]) 
                   for i in range(0,dyn_var.Nt)]
                  )
    )

P_dyn_var = dyn_var.compute_price_indices(p)
print('price_indices', np.all(
                  [np.allclose(getattr(sol_c,'price_indices'),  P_dyn_var[...,i]) 
                   for i in range(0,dyn_var.Nt)]
                  )
    )

print('PSI_CD', np.allclose(dyn_var.compute_PSI_CD(p)[...,1:,:],dyn_var.PSI_CD[...,1:,:]))
print('PSI_MNP', np.allclose(dyn_var.compute_PSI_MNP(p)[...,1:,:],dyn_var.PSI_MNP[...,1:,:]))
print('PSI_MPND', np.allclose(dyn_var.compute_PSI_MPND(p)[...,1:,:],dyn_var.PSI_MPND[...,1:,:]))
print('PSI_MPD', np.allclose(dyn_var.compute_PSI_MPD(p)[...,1:,:],dyn_var.PSI_MPD[...,1:,:]))
print('V_PD', np.allclose(dyn_var.compute_V_PD(p)[...,1:,:],dyn_var.V_PD[...,1:,:]))
print('V_NP', np.allclose(dyn_var.compute_V_NP(p)[...,1:,:],dyn_var.V_NP[...,1:,:]))
print('V_P', np.allclose(dyn_var.compute_V_P(p)[...,1:,:],dyn_var.V_P[...,1:,:]))
print('P', np.allclose(dyn_var.compute_price_indices(p),dyn_var.price_indices))
print('w', np.allclose(dyn_var.compute_wage(p),dyn_var.w))
print('Z', np.allclose(dyn_var.compute_expenditure(p),dyn_var.Z))

#%%

def cheb(N):
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D, x

def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**i)
    return np.array(alt)


from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
from classes import moments, parameters, var, dynamic_var
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

p_init = parameters(n=7,s=2)
p_init.load_data('calibration_results_matched_economy/baseline_405_variations/1.12/')
# p_init.load_data('calibration_results_matched_economy/405/')
# p_init.delta = p_init.delta*2
sol, sol_init = fixed_point_solver(p_init,x0=p_init.guess,
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
sol_init.scale_P(p_init)
sol_init.compute_non_solver_quantities(p_init) 

# p = parameters(n=7,s=2)
# p.load_data('calibration_results_matched_economy/baseline_403_variations/1.40/')
p = p_init.copy()
# p.delta[-2,1] = p.delta[-2,1]/10
p.delta[:,1] = 12
p.delta[0,1] = 1e-2
# p.delta = p.delta*100
# p.delta[:] = 12

sol, sol_fin = fixed_point_solver(p,x0=p.guess,
                                context = 'counterfactual',
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
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
sol_fin.compute_consumption_equivalent_welfare(p,sol_init)


#%%     
# for Nt in range(23,27):
Nt = 25
print(Nt)
# p.zeta[1] = p.zeta[1]*10
sol, dyn_sol = dyn_fixed_point_solver(p, sol_init,Nt=Nt,
                                      t_inf=500,
                        cobweb_anim=False,tol =1e-14,
                        accelerate=False,
                        accelerate_when_stable=False,
                        cobweb_qty='l_R',
                        plot_convergence=True,
                        plot_cobweb=False,
                        plot_live = False,
                        safe_convergence=1e-8,
                        disp_summary=True,
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
dyn_sol.compute_non_solver_quantities(p)

#%%

fig,ax = plt.subplots(figsize = (15,10))

for i,c in enumerate(p.countries): 
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                dyn_sol.nominal_final_consumption[i,:]/dyn_sol.price_indices[i,:]/(dyn_sol.sol_init.cons[i]),
                dyn_sol.Nt),np.linspace(0,dyn_sol.t_inf,1001))
    ax.plot(np.linspace(0,dyn_sol.t_inf,1001),fit
             , label=c) 
plt.legend()
plt.xscale('log')
plt.title('Real consumption')
plt.show()

fig,ax = plt.subplots(figsize = (15,10))

for i,c in enumerate(p.countries): 
    fit = np.polyval(np.polyfit(dyn_sol.t_real,
                dyn_sol.integrand_welfare[i,:],
                dyn_sol.Nt),np.linspace(0,dyn_sol.t_inf,1001))
    ax.plot(np.linspace(0,dyn_sol.t_inf,1001),fit
             , label=c) 
ax.legend()
plt.title('Integrand welfare')
plt.xscale('log')
plt.show()

#%%

from scipy.special import roots_jacobi

def rat_cheb_diff_matrix(N, alpha, beta):
    # Compute the collocation points using the roots of the Jacobi polynomial
    # with parameters alpha and beta
    x, _ = roots_jacobi(N+1, alpha, beta)

    # Compute the Chebyshev differentiation matrix
    D = np.zeros((N+1, N+1))
    c = np.zeros(N+1)
    c[0] = c[N] = 2
    for i in range(1, N):
        c[i] = 1
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                if i == 0 or i == N:
                    D[i, j] = (2*N**2 + 1)*(1-x[i])**2/(6*x[i]**2)
                else:
                    D[i, j] = -2*(alpha+beta+2)*x[i]/((alpha+beta+3)*((1-x[i])**2)-alpha**2+beta**2)
            else:
                if i == 0:
                    D[i, j] = (-1)**j*c[j]*np.exp(beta*np.log(1-x[j]))/((alpha+beta+2)*x[j])
                elif i == N:
                    D[i, j] = (-1)**j*c[j]*np.exp(alpha*np.log(x[j]))/((alpha+beta+2)*(1-x[j]))
                else:
                    if i < j:
                        s = -1
                    else:
                        s = 1
                    d = np.abs(i-j)
                    p = (2*(alpha+beta+d+2)*(alpha+beta+d+1))/((2*d+alpha+beta+1)*(2*d+alpha+beta+3))
                    q = (alpha**2-beta**2)/((2*d+alpha+beta+1)*(2*d+alpha+beta+3))
                    r = (2*(d+alpha)*(d+beta)*(2*d+alpha+beta+2))/((2*d+alpha+beta+1)**2*(2*d+alpha+beta+3))
                    D[i, j] = s*c[j]*np.exp((beta-alpha)*np.log(1-x[j])+alpha*np.log(x[i]))*(p+q*(1-x[j])**2)/(r*(x[i]-x[j])**2)
    
    return D, x

D, x = rat_cheb_diff_matrix(10, 1, 1)

plt.plot(x,np.exp(-x**2))
plt.plot(x,np.matmul(D,np.exp(-x**2)))

#%%
from scipy.optimize import root

def rational_cheb_diff_matrix(N, alpha, beta):
    # Chebyshev collocation points
    x = -np.cos(np.pi*np.arange(N+1)/N)

    # Rational Chebyshev functions
    r = (x+1)**(-alpha) * (x-1)**(-beta)

    # Chebyshev differentiation matrix
    c = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                c[i,j] = r[i]/r[j] * (-1)**(i+j) / (x[i]-x[j])
            elif i == 0:
                c[i,j] = 2*N**2*(1+beta)*alpha
            elif i == N:
                c[i,j] = -2*N**2*(1+alpha)*beta
            else:
                c[i,j] = -x[i]/(2*(1-x[i]**2))

    # Return differentiation matrix and collocation points
    return c, x

def f(y):
    return y**2 - y**3

def solve_chebyshev_rational(N, alpha, beta):
    # Differentiation matrix and collocation points
    D, x = rational_cheb_diff_matrix(N, alpha, beta)

    # Residual function
    def residual(y):
        dydx = np.dot(D, y)
        return dydx - f(y)

    # Solve for y
    y_guess = np.zeros(N+1)
    sol = root(residual, y_guess)

    # Add boundary conditions
    y = np.append(sol.x, 0)
    dydx_inf = np.dot(D, y)[-1]
    y += y[0] / (1-dydx_inf)

    # Return solution
    return y, x

c,x = rational_cheb_diff_matrix(20, 2, 1)

# y, x = solve_chebyshev_rational(20, 2, 1)
# print(y)


#%%


fit = np.polyfit(dyn_sol.t_cheby,
                  dyn_sol.PSI_CD[0,1,:],
                  dyn_sol.Nt)

fig,ax = plt.subplots(figsize = (15,10))

ax.scatter(dyn_sol.t_real, dyn_sol.PSI_CD[0,1,:], label = 'values')
ax.plot(((cheb(5000)[1]+1)*dyn_sol.t_inf/2), 
        np.polyval(fit, cheb(5000)[1]),label='interpolation')
plt.suptitle('Chebyshev grid')
plt.title('PSI_CD')
# plt.xscale('log')
# ax[1].scatter(dyn_sol.t_real, dyn_sol.PSI_CD[0,1,:], label = 'values')
# ax[1].plot(dyn_sol.t_inf-np.linspace(0,dyn_sol.t_inf,501), 
#         np.polyval(fit, cheb(500)[1]),label='interpolation')
# ax[1].set_title('Real time')

plt.show()

# plt.plot([np.linalg.norm(np.polyval(np.polyfit(dic_dyn_sol[n-1].t_cheby,
#                   dic_dyn_sol[n-1].w[0,:],
#                   dyn_sol.Nt), cheb(500)[1]) 
#                 - np.polyval(np.polyfit(dic_dyn_sol[n].t_cheby,
#                                   dic_dyn_sol[n].w[0,:],
#                                   dyn_sol.Nt), cheb(500)[1])) for n in range(21,26)])

# fig,ax = plt.subplots(figsize = (15,10))

# idx_t_truncated = np.where(dyn_sol.t_real<300)
# idx_t_interp_truncated = np.where((cheb(10000)[1]+1)*dyn_sol.t_inf/2<300)

# fig,ax = plt.subplots(figsize = (15,10))

# ax.scatter(dyn_sol.t_real[idx_t_truncated], dyn_sol.PSI_CD[0,1,:].squeeze()[idx_t_truncated], label = 'values')
# ax.plot(((cheb(10000)[1]+1)*dyn_sol.t_inf/2)[idx_t_interp_truncated], 
#         np.polyval(fit, cheb(10000)[1])[idx_t_interp_truncated],label='interpolation')
# plt.suptitle('Chebyshev grid')
# plt.title('PSI_CD')

# # ax[1].scatter(dyn_sol.t_real, dyn_sol.PSI_CD[0,1,:], label = 'values')
# # ax[1].plot(dyn_sol.t_inf-np.linspace(0,dyn_sol.t_inf,501), 
# #         np.polyval(fit, cheb(500)[1]),label='interpolation')
# # ax[1].set_title('Real time')

# plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt
from classes import cheb

fit = np.polyfit(dyn_sol.t_cheby,
                 dyn_sol.PSI_CD[0,1,:],
                 dyn_sol.Nt)

fig,ax = plt.subplots(2,1,figsize = (15,10))

ax[0].scatter(dyn_sol.t_cheby, dyn_sol.PSI_CD[0,1,:], label = 'values')
ax[0].plot(cheb(500)[1], 
        np.polyval(fit, cheb(500)[1]),label='interpolation')
ax[0].set_title('Chebyshev grid')

ax[1].scatter(dyn_sol.t_real, dyn_sol.PSI_CD[0,1,:], label = 'values')
ax[1].plot(np.linspace(0,dyn_sol.t_inf,501), 
        np.polyval(fit, cheb(500)[1]),label='interpolation')
ax[1].set_title('Real time')

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from classes import cheb

fit = np.polyfit(dyn_sol.t_real,
                 dyn_sol.PSI_CD[0,1,:],
                 dyn_sol.Nt)

fig,ax = plt.subplots(2,1,figsize = (15,10))

ax[0].scatter(dyn_sol.t_cheby, dyn_sol.PSI_CD[0,1,:], label = 'values')
ax[0].plot(cheb(500)[1], 
        np.polyval(fit, np.linspace(0,dyn_sol.t_inf,501)),label='interpolation')
ax[0].set_title('Chebyshev grid')

ax[1].scatter(dyn_sol.t_real, dyn_sol.PSI_CD[0,1,:], label = 'values')
ax[1].plot(np.linspace(0,dyn_sol.t_inf,501), 
        np.polyval(fit, np.linspace(0,dyn_sol.t_inf,501)),label='interpolation')
ax[1].set_title('Real time')

plt.show()


#%%

import matplotlib.pyplot as plt

def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**i)
    return np.array(alt)

# def _alt(n):
#     alt = []
#     for i in range(n):
#         alt.append((-1)**(i+1))
#     return np.array(alt)

# def consec(n):
#     return list(range(n))

# def cesnoc(n):
#     cesnoc = []
#     for i in range(n):
#         cesnoc.append(-n+i)
#     return cesnoc

# def arrconsec(n):
#     return np.array(list(range(n)))

def cheb(N):
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D, x

def cheb_neuman_right(N):
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    D[-1,:] = 0
    return D, x

# def cheb_neuman_right(N):
#     x = np.cos(np.pi*np.linspace(0,1,N+1))
#     c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
#     X = np.outer(x, np.ones(N+1))
#     dX = X-X.T
#     D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
#     D = D - np.diag(np.sum(D,axis=1))
#     D[-1,:] = 0
#     return D, x

# def cheb_dirichlet_neuman(N):
    

D,x = cheb_neuman_right(dyn_sol.Nt-1)

fig,ax = plt.subplots(figsize = (16,12))
ax1 = ax.twinx()

ax.plot(dyn_var.t_real,np.einsum('tu,nisu->nist',D,dyn_var.PSI_M)[0,0,1,:],color = 'r')
# ax.scatter(x,np.einsum('tu,nisu->nist',D,dyn_var.PSI_M)[0,0,1,:],color='r')
ax1.plot(dyn_var.t_real,dyn_var.PSI_M[0,0,1,:])
# ax1.scatter(x,dyn_var.PSI_M[0,0,1,:])

plt.show()

#%%

def cheb(N):
    '''Chebushev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x      = np.cos(np.pi*np.arange(0,N+1)/N)
    if N%2 == 0:
        x[N//2] = 0.0 # only when N is even!
    c      = np.ones(N+1); c[0] = 2.0; c[N] = 2.0
    c      = c * (-1.0)**np.arange(0,N+1)
    c      = c.reshape(N+1,1)
    X      = np.tile(x.reshape(N+1,1), (1,N+1))
    dX     = X - X.T
    D      = np.dot(c, 1.0/c.T) / (dX+np.eye(N+1))
    D      = D - np.diag( D.sum(axis=1) )
    return D,x


N = dyn_sol.Nt
D,x = cheb(N)
D = D[1:N+1,1:N+1]

t = dyn_sol.t

#%%%

# from chebPy import *
# from numpy import dot,exp,zeros,max,linspace,polyval,polyfit,inf
import numpy as np
import matplotlib.pyplot as plt
# from numpy.linalg import norm
from scipy.linalg import solve
# from matplotlib.pyplot import title,plot,xlabel,ylabel,grid


def cheb(N):
    '''Chebushev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x      = np.cos(np.pi*np.arange(0,N+1)/N)
    if N%2 == 0:
        x[N//2] = 0.0 # only when N is even!
    c      = np.ones(N+1); c[0] = 2.0; c[N] = 2.0
    c      = c * (-1.0)**np.arange(0,N+1)
    c      = c.reshape(N+1,1)
    X      = np.tile(x.reshape(N+1,1), (1,N+1))
    dX     = X - X.T
    D      = np.dot(c, 1.0/c.T) / (dX+np.eye(N+1))
    D      = D - np.diag( D.sum(axis=1) )
    return D,x

N = 16

# Build matrix
D,x = cheb(N)
D2 = np.dot(D,D)
D2[N,:] = D[N,:] # Last eqn has neumann bc
D2 = D2[1:N+1,1:N+1]

# RHS
f = np.zeros(N)
f[0:-1] = np.exp(4.0*x[1:N])

# Solve
u = solve(D2,f)
s = np.zeros(N+1)
s[1:N+1] = u

# Compute error
xx = np.linspace(-1.0,1.0,200)
uu = np.polyval(np.polyfit(x,s,N),xx)    # interpolate grid data
exact = (np.exp(4.0*xx) - 4.0*np.exp(-4.0)*(xx-1.0) - np.exp(4.0))/16.0
maxerr = np.linalg.norm(uu-exact,np.inf)

plt.title('max err = %e' % maxerr)
plt.plot(x,s,'o',xx,exact)
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)

#%%
from numpy import dot,exp,zeros,max,linspace,polyval,polyfit,inf
from numpy.linalg import norm
from scipy.linalg import solve
from matplotlib.pyplot import title,plot,xlabel,ylabel,grid

N = 16

# Build matrix
D,x = cheb(N)
D2 = dot(D,D)
D2[N,:] = D[N,:] # Last eqn has neumann bc
D2 = D2[1:N+1,1:N+1]

# RHS
f = zeros(N)
f[0:-1] = exp(4.0*x[1:N])

# Solve
u = solve(D2,f)
s = zeros(N+1)
s[1:N+1] = u

# Compute error
xx = linspace(-1.0,1.0,200)
uu = polyval(polyfit(x,s,N),xx)    # interpolate grid data

exact = (exp(4.0*xx) - 4.0*exp(4.0)*(xx-1.0) - exp(4.0))/16.0

# exact = (exp(-4.0*xx) - 4.0*exp(4.0)*(xx-1.0) - exp(-4.0))/16.0
maxerr = norm(uu-exact,inf)

fig, ax = plt.subplots()
plt.title('max err = %e' % maxerr)
ax.plot(x,s,'o')
ax.plot(xx,exact)
# ax.xlabel('x'); ax.ylabel('u'); plt.grid(True);
plt.show()

#%%

def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**i)
    return np.array(alt)

def _alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**(i+1))
    return np.array(alt)

def consec(n):
    return list(range(n))

def cesnoc(n):
    cesnoc = []
    for i in range(n):
        cesnoc.append(-n+i)
    return cesnoc

def arrconsec(n):
    return np.array(list(range(n)))

def cheb(N):
    if N==0:
        return 0, 1
    
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D, x


D,x = cheb(dyn_var.Nt-1)

plt.plot(x,np.sin(x))
plt.plot(x,np.einsum('tu,u->t',D,np.sin(x)),c='r')
plt.scatter(x,np.cos(x),c='g')

plt.show()


