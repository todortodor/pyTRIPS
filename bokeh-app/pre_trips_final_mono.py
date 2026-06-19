#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final PRE-TRIPS (1992) MONO-sector lever calibrations (baseline 2000), OLD patent
fee, with the CORRECTED 1992 turnover target (1989-1995): TO_target = 0.125433.
Saved as NUMERIC variations: 17.0 base / 17.1 +fe,fo / 17.2 +nu / 17.3 +all.
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import os, time, warnings
import numpy as np, pandas as pd
from scipy import optimize
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver
warnings.filterwarnings('ignore')

DRY_RUN = False
SEED_PATH = 'calibration_results_matched_economy/2000/'
DATA_DIR = 'data_smooth_3_years/data_12_countries_1992/'
VARDIR = 'calibration_results_matched_economy/baseline_2000_variations/'
TO_1992 = 0.125433
BASE_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP', 'UUPCOST']
BASE_WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1, 'SRGDP': 1, 'UUPCOST': 1, 'TO': 5}
MAX_ITER = 5
LEVERS = [dict(name='base', out='17.0', extra_calib=[],                 extra_moments=[]),
          dict(name='fefo', out='17.1', extra_calib=['fe', 'fo'],       extra_moments=[]),
          dict(name='nu',   out='17.2', extra_calib=['nu'],             extra_moments=['TO']),
          dict(name='all',  out='17.3', extra_calib=['fe', 'fo', 'nu'], extra_moments=['TO'])]
SOLVER_KWARGS = dict(cobweb_anim=False, tol=1e-13, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False, safe_convergence=0.001,
    disp_summary=False, damping=10, max_count=3e3, accel_memory=50, accel_type1=True,
    accel_regularization=1e-10, accel_relaxation=0.5, accel_safeguard_factor=1,
    accel_max_weight_norm=1e6, damping_post_acceleration=5)


def setup(cfg):
    p = parameters(); p.correct_eur_patent_cost = True; p.load_run(SEED_PATH)
    p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=2)
    p.calib_parameters = ['delta', 'T', 'eta'] + cfg['extra_calib']
    m = moments(); m.load_run(SEED_PATH); m.load_data(DATA_DIR)
    m.aggregate_moments = True
    m.list_of_moments = BASE_MOMENTS + cfg['extra_moments']
    m.weights_dict.update(BASE_WEIGHTS)
    m.drop_CHN_IND_BRA_ROW_from_RD = True
    if 'TO' in m.list_of_moments:
        m.TO_target = np.float64(TO_1992)
    dmask = p.mask['delta'].reshape(p.N, p.S).sum(axis=0)
    print(f"  [{cfg['name']}->{cfg['out']}] calib={p.calib_parameters} delta_free={dmask.tolist()} moments={m.list_of_moments}"
          + (f"  TO_target={float(m.TO_target)}" if 'TO' in m.list_of_moments else ""))
    assert dmask[0] == 0 and dmask[1] == p.N
    return p, m


def run(cfg):
    p, m = setup(cfg)
    sol, sol_i = fixed_point_solver(p, x0=p.guess, context='calibration', **SOLVER_KWARGS)
    sol_i.scale_P(p); sol_i.compute_non_solver_quantities(p); p.guess = sol.x
    m.compute_moments(sol_i, p); m.compute_moments_deviations()
    print(f"     initial obj={np.linalg.norm(m.deviation_vector()):.3f} ({sol.status})")
    if DRY_RUN:
        return
    hist = history(*tuple(m.list_of_moments + ['objective'])); bounds = p.make_parameters_bounds(); t0 = time.perf_counter()
    for it in range(MAX_ITER + 1):
        res = optimize.least_squares(calibration_func, p.make_p_vector(), args=(p, m, p.guess, hist, t0),
                                     bounds=bounds, max_nfev=1e8, xtol=(1e-10 if it < MAX_ITER - 2 else 1e-16), verbose=2)
        p.update_parameters(res.x)
    p_sol = p.copy(); p_sol.update_parameters(res.x)
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
    p_sol.guess = sol.x; sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    out = VARDIR + cfg['out'] + '/'; os.makedirs(out, exist_ok=True)
    p_sol.write_params(out); m.write_moments(out)
    print(f"  [{cfg['name']}->{cfg['out']}] DONE obj={np.linalg.norm(m.deviation_vector()):.3f} -> {out}")


if __name__ == '__main__':
    print(f"{'DRY-RUN' if DRY_RUN else 'RUN'} {len(LEVERS)} mono configs (old fee, corrected 1992 TO)\n")
    for cfg in LEVERS:
        run(cfg)
    if DRY_RUN:
        print("\nDRY_RUN -> validated. set DRY_RUN=False to run.")
