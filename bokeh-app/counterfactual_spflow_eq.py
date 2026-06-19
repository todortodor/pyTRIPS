#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual: shift the patent-flow (SPFLOW) targets, recalibrate baseline 2000
(single 2-sector, no FDI), and recompute the Nash and cooperative (Negishi +
Equal/pop-weighted) equilibria WITH DYNAMICS.

Scenarios:
  16.0 (a) "more patents in developed, fewer in developing", by DESTINATION:
        scale international (off-diagonal) flows Pat_{n,i} whose DESTINATION n is
        developed (USA,EUR,JAP,CAN,KOR) by 2.0 (+100%) and developing
        (CHN,BRA,IND,RUS,MEX,ZAF,ROW) by 0.5 (-50%), then renormalize the shares;
        AND increase DOMPATINUS, SINNOVPATUS, UUPCOST targets by 100% (x2.0).
  16.1 (b) "US becomes more the center of the world", by ORIGIN:
        scale international flows Pat_{n,i} whose ORIGIN i is the US by 2.5 (+150%),
        renormalize; increase US RD target by 150% (x2.5); reduce the turnover (TO)
        target by 30% (x0.7) with nu kept free; increase UUPCOST by 150% (x2.5),
        DOMPATINUS by 30% (x1.3), and the US SRGDP share by 15% (x1.15, renormalized).

Calibration -> baseline_2000_variations/{16.0,16.1}/
Equilibria  -> coop_eq_direct_saves/2000_{16.0,16.1}_{nash,negishi,pop_weighted}/

Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import optimize

from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver, find_nash_eq, find_coop_eq

warnings.filterwarnings('ignore')

DRY_RUN   = False
CALIB_ONLY = True          # only recalibrate to the shifted targets; skip Nash/coop
BASELINE  = '2000'
BASE_PATH = f'calibration_results_matched_economy/{BASELINE}/'
VARDIR    = f'calibration_results_matched_economy/baseline_{BASELINE}_variations/'
SAVE_ROOT = 'coop_eq_direct_saves/'

# 12-country order: USA0 EUR1 JAP2 CHN3 BRA4 IND5 CAN6 KOR7 RUS8 MEX9 ZAF10 ROW11
DEVELOPED  = [0, 1, 2, 6, 7]                 # USA, EUR, JAP, CAN, KOR
DEVELOPING = [3, 4, 5, 8, 9, 10, 11]         # CHN, BRA, IND, RUS, MEX, ZAF, ROW
US = 0

LB_DELTA, UB_DELTA = 0.01, 12
MAX_ITER = 8

SOLVER_KWARGS = dict(
    cobweb_anim=False, tol=1e-14, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=False, damping=10, max_count=3e3,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)

SCENARIOS = [
    # (a) developed-centric by destination, plus US-patenting scalar bumps
    dict(variation='16.0', kind='destination',
         dev_scale=2.0, deving_scale=0.5,
         scale_scalar={'DOMPATINUS': 2.0, 'SINNOVPATUS': 2.0, 'UUPCOST': 2.0}),
    # (b) US-centric by origin, plus US R&D / turnover / cost / GDP adjustments
    dict(variation='16.1', kind='us_origin',
         us_origin_scale=2.5,
         scale_scalar={'UUPCOST': 2.5, 'DOMPATINUS': 1.3, 'TO': 0.7},
         us_rd_scale=2.5, us_srgdp_scale=1.15),
]


# ─────────────────────────────────────────────────────────────────────────────
# Target modification (SPFLOW, and RD for scenario b)
# ─────────────────────────────────────────────────────────────────────────────
def shift_spflow_target(m, scn):
    """Rebuild the off-diagonal flow matrix from cc_moments, scale, renormalize,
    and write it back into m.SPFLOW_target. Returns (before, after) per-group
    share diagnostics."""
    kind = scn['kind']
    N = m.N
    F = m.cc_moments['patent flows'].values.reshape(N, N).astype(float)   # [dest, origin]
    offmask = ~np.eye(N, dtype=bool)

    # sanity: reconstructed normalized off-diagonal must equal the loaded target
    base_off = F[offmask].reshape(N, N - 1)
    assert np.allclose(base_off / base_off.sum(), m.SPFLOW_target), \
        'cc_moments reconstruction does not match SPFLOW_target'

    Fn = F.copy()
    if kind == 'destination':
        for d in DEVELOPED:
            Fn[d, :] *= scn['dev_scale']        # rows = destination
        for d in DEVELOPING:
            Fn[d, :] *= scn['deving_scale']
    elif kind == 'us_origin':
        Fn[:, US] *= scn['us_origin_scale']     # columns = origin
    else:
        raise ValueError(kind)

    new_off = Fn[offmask].reshape(N, N - 1)
    m.SPFLOW_target = new_off / new_off.sum()

    # diagnostics on the share captured by developed destinations and US origin
    def dest_share(M):
        s = M[offmask].reshape(N, N - 1)
        s = s / s.sum()
        full = np.zeros((N, N))
        full[offmask] = s.ravel()
        return full.sum(axis=1)        # inflow share by destination (row)
    def us_origin_share(M):
        s = M[offmask].reshape(N, N - 1); s = s / s.sum()
        full = np.zeros((N, N)); full[offmask] = s.ravel()
        return full[:, US].sum()       # share of flows with origin US
    return dict(dest_share_before=dest_share(F), dest_share_after=dest_share(Fn),
                us_share_before=us_origin_share(F), us_share_after=us_origin_share(Fn))


def setup_scenario(scn):
    p = parameters(); p.correct_eur_patent_cost = True
    p.load_run(BASE_PATH)
    m = moments(); m.load_run(BASE_PATH)

    diag = shift_spflow_target(m, scn)
    print(f"\n[{scn['variation']}] kind={scn['kind']}")
    print("  dest inflow share before:", np.round(diag['dest_share_before'], 4))
    print("  dest inflow share after :", np.round(diag['dest_share_after'], 4))
    print(f"  US-origin share before/after: {diag['us_share_before']:.4f} -> {diag['us_share_after']:.4f}")

    # scalar (US-patenting) target bumps: DOMPATINUS / SINNOVPATUS / UUPCOST / TO
    for name, f in scn.get('scale_scalar', {}).items():
        before = float(np.asarray(getattr(m, name + '_target')))
        setattr(m, name + '_target', getattr(m, name + '_target') * f)
        print(f"  {name}_target: {before:.5f} -> {float(np.asarray(getattr(m, name + '_target'))):.5f} (x{f})")

    # US R&D target bump (also refresh the US / relative-to-US helper targets)
    if 'us_rd_scale' in scn:
        rd_before = float(m.RD_target[US])
        m.RD_target[US] *= scn['us_rd_scale']
        m.RD_US_target = m.RD_target[US]
        m.RD_RUS_target = m.RD_target / m.RD_US_target
        print(f"  US RD target: {rd_before:.5f} -> {float(m.RD_target[US]):.5f} (x{scn['us_rd_scale']})")

    # US SRGDP share bump, then renormalize (SRGDP_target is a share summing to 1)
    if 'us_srgdp_scale' in scn:
        sg_before = float(m.SRGDP_target[US])
        m.SRGDP_target[US] *= scn['us_srgdp_scale']
        m.SRGDP_target = m.SRGDP_target / m.SRGDP_target.sum()
        m.SRGDP_US_target = m.SRGDP_target[US]
        m.SRGDP_RUS_target = m.SRGDP_target / m.SRGDP_US_target
        print(f"  US SRGDP share: {sg_before:.5f} -> {float(m.SRGDP_target[US]):.5f} "
              f"(x{scn['us_srgdp_scale']}, renormalized)")

    # assert SPFLOW shares still sum to 1 and shape unchanged
    assert np.isclose(m.SPFLOW_target.sum(), 1.0)
    assert m.SPFLOW_target.shape == (m.N, m.N - 1)
    assert np.isclose(m.SRGDP_target.sum(), 1.0)
    print(f"  calib_parameters={p.calib_parameters}")
    print(f"  list_of_moments={m.list_of_moments}")
    return p, m


# ─────────────────────────────────────────────────────────────────────────────
# Recalibration to the shifted targets
# ─────────────────────────────────────────────────────────────────────────────
def recalibrate(p, m):
    hist = history(*tuple(m.list_of_moments + ['objective']))
    bounds = p.make_parameters_bounds()
    t0 = time.perf_counter()
    for it in range(MAX_ITER + 1):
        xtol = 1e-10 if it < MAX_ITER - 4 else 1e-14
        res = optimize.least_squares(
            fun=calibration_func, x0=p.make_p_vector(), args=(p, m, p.guess, hist, t0),
            bounds=bounds, max_nfev=1e8, xtol=xtol, verbose=2)
        p.update_parameters(res.x)
    p_sol = p.copy(); p_sol.update_parameters(res.x)
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
    p_sol.guess = sol.x
    sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    return p_sol, float(np.linalg.norm(m.deviation_vector())), sol.status


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(scn):
    var_id = scn['variation']
    p, m = setup_scenario(scn)
    if DRY_RUN:
        # cheap check: one solve at the loaded (pre-recalibration) params
        sol, s = fixed_point_solver(p, x0=p.guess, context='calibration', **SOLVER_KWARGS)
        print(f"  [{var_id}] initial solve: {sol.status}")
        return

    print(f"\n[{var_id}] recalibrating ...")
    p_sol, obj, status = recalibrate(p, m)
    cal_out = VARDIR + var_id + '/'
    os.makedirs(cal_out, exist_ok=True)
    p_sol.write_params(cal_out); m.write_moments(cal_out)
    print(f"[{var_id}] calibration done obj={obj:.4f} ({status}) -> {cal_out}")

    if CALIB_ONLY:
        print(f"[{var_id}] CALIB_ONLY -> skipping Nash / cooperative equilibria")
        return

    # Nash (dynamic)
    try:
        t = time.perf_counter()
        p_nash, _ = find_nash_eq(p_sol, lb_delta=LB_DELTA, ub_delta=UB_DELTA,
                                 method='fixed_point', dynamics=True,
                                 plot_convergence=False, solver_options=None, tol=1e-4,
                                 delta_init=np.ones(p_sol.N * (p_sol.S - 1)) * UB_DELTA,
                                 parallel=False)
        p_nash.write_params(SAVE_ROOT + f'{BASELINE}_{var_id}_nash/')
        print(f"[{var_id}] NASH done in {time.perf_counter()-t:.0f}s -> {BASELINE}_{var_id}_nash/")
    except Exception as e:
        print(f"[{var_id}] NASH FAILED: {e}")

    # Cooperative (dynamic), both aggregations
    for agg in ['negishi', 'pop_weighted']:
        try:
            t = time.perf_counter()
            p_opti, _ = find_coop_eq(p_sol, agg, lb_delta=LB_DELTA, ub_delta=UB_DELTA,
                                     dynamics=True, solver_options=None, tol=1e-12,
                                     static_eq_deltas=None, custom_weights=None,
                                     parallel=False, custom_x0=None)
            p_opti.write_params(SAVE_ROOT + f'{BASELINE}_{var_id}_{agg}/')
            print(f"[{var_id}] COOP {agg} done in {time.perf_counter()-t:.0f}s -> {BASELINE}_{var_id}_{agg}/")
        except Exception as e:
            print(f"[{var_id}] COOP {agg} FAILED: {e}")


if __name__ == '__main__':
    for scn in SCENARIOS:
        run_scenario(scn)
    if DRY_RUN:
        print("\nDRY_RUN=True -> targets validated + initial solves, nothing calibrated. "
              "Set DRY_RUN=False to run the full exercise.")
