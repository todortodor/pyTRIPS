#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRE-TRIPS (1992) PARTIAL-CALIBRATION CONFIG SEARCH (baseline 2002, multi-sector).

Sweeps which PARAMETERS are calibrated and which MOMENTS are targeted, scoring
each run against the desired delta criteria on the PATENT sector (sector 1):
  - developed economies: 1992 delta similar to 2015  (small |ratio-1|, CAN excluded)
  - developing economies: 1992 delta HIGHER than 2015 (less protection pre-TRIPS)

Identification is respected: nu is only ever calibrated together with an
identifying moment (turnover and/or GROWTH); turnover, when used, is set to the
corrected US 1992 targets. Core calib params {delta,T,eta} are always on.

Run from inside bokeh-app/.  Writes per-run folders + a ranked summary CSV to
  calibration_results_matched_economy/baseline_2002_variations/config_search/
"""
import matplotlib; matplotlib.use('Agg')
import os, time, warnings, itertools
import numpy as np, pandas as pd
from scipy import optimize
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver
warnings.filterwarnings('ignore')

VAR = 'calibration_results_matched_economy/baseline_2002_variations/'
DATA_DIR = 'data_smooth_3_years/data_12_countries_3_sectors_1992_RDPHARMACHEM_fix/'
OUTDIR = VAR + 'config_search/'
MAX_ITER = 5

TOPATENT_1992 = 0.127978       # corrected US 1992 turnover targets
TOPHARMACHEM_1992 = 0.094810

BASE_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP', 'UUPCOSTS', 'RDPHARMACHEM']
BASE_WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1, 'SRGDP': 1,
                'UUPCOSTS': 1, 'RDPHARMACHEM': 1, 'TOPATENT': 5, 'TOPHARMACHEM': 5, 'GROWTH': 5}
TURNOVER = ['TOPATENT', 'TOPHARMACHEM']

DEV = ['USA', 'EUR', 'JAP', 'KOR']                     # developed (CAN excluded)
DEVING = ['CHN', 'BRA', 'IND', 'RUS', 'MEX', 'ZAF']    # developing

REGIMES = [dict(tag='s',  seed='8.0', fix_delta=False),
           dict(tag='c',  seed='8.8', fix_delta=True),
           dict(tag='cd', seed='8.8', fix_delta=False)]
# param add-ons paired with identifying moments (nu always has turnover OR GROWTH).
# Note: targeting BOTH turnover and growth for a single nu is over-identified and
# ill-conditioned (it stalled least_squares), so those combos are excluded.
PARAM_MOMENT = [
    dict(name='base',     calib=[],               mom=[]),
    dict(name='fefo',     calib=['fe', 'fo'],     mom=[]),
    dict(name='nu_to',    calib=['nu'],           mom=TURNOVER),
    dict(name='nu_gr',    calib=['nu'],           mom=['GROWTH']),
    dict(name='all_to',   calib=['fe','fo','nu'], mom=TURNOVER),
    dict(name='all_gr',   calib=['fe','fo','nu'], mom=['GROWTH']),
]
MAX_NFEV = 400             # cap per least_squares call to prevent runaway configs

SK = dict(cobweb_anim=False, tol=1e-13, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False, safe_convergence=0.001,
    disp_summary=False, damping=10, max_count=3e3, accel_memory=50, accel_type1=True,
    accel_regularization=1e-10, accel_relaxation=0.5, accel_safeguard_factor=1,
    accel_max_weight_norm=1e6, damping_post_acceleration=5)


def seed_patent_delta(seed):
    p = parameters(); p.correct_eur_patent_cost = True; p.load_run(VAR + seed + '/')
    return pd.Series(np.asarray(p.delta).reshape(p.N, p.S)[:, 1], index=list(p.countries))


def setup(reg, cfg):
    p = parameters(); p.correct_eur_patent_cost = True
    p.fix_delta_across_sectors = reg['fix_delta']
    p.load_run(VAR + reg['seed'] + '/')
    p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=3)
    p.calib_parameters = ['delta', 'T', 'eta'] + cfg['calib']
    if reg['fix_delta']:
        for s in range(2, p.S):
            p.delta[:, s] = p.delta[:, 1]
    m = moments(); m.load_run(VAR + reg['seed'] + '/'); m.load_data(DATA_DIR)
    m.aggregate_moments = True
    m.list_of_moments = BASE_MOMENTS + cfg['mom']
    m.weights_dict.update(BASE_WEIGHTS)
    m.drop_CHN_IND_BRA_ROW_from_RD = True
    if 'TOPATENT' in m.list_of_moments:
        m.TOPATENT_target = np.float64(TOPATENT_1992)
        m.TOPHARMACHEM_target = np.float64(TOPHARMACHEM_1992)
    return p, m


def run_one(reg, cfg):
    label = f"{reg['tag']}_{cfg['name']}"
    p, m = setup(reg, cfg)
    sol, sol_i = fixed_point_solver(p, x0=p.guess, context='calibration', **SK)
    sol_i.scale_P(p); sol_i.compute_non_solver_quantities(p); p.guess = sol.x
    hist = history(*tuple(m.list_of_moments + ['objective']))
    bounds = p.make_parameters_bounds(); t0 = time.perf_counter()
    for it in range(MAX_ITER + 1):
        res = optimize.least_squares(calibration_func, p.make_p_vector(), args=(p, m, p.guess, hist, t0),
                                     bounds=bounds, max_nfev=MAX_NFEV,
                                     xtol=(1e-10 if it < MAX_ITER - 2 else 1e-16), verbose=0)
        p.update_parameters(res.x)
    p_sol = p.copy(); p_sol.update_parameters(res.x)
    if reg['fix_delta']:
        for s in range(2, p_sol.S):
            p_sol.delta[:, s] = p_sol.delta[:, 1]
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SK)
    p_sol.guess = sol.x; sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    out = OUTDIR + label + '/'; os.makedirs(out, exist_ok=True)
    p_sol.write_params(out); m.write_moments(out)
    obj = float(np.linalg.norm(m.deviation_vector()))
    return label, p_sol, obj


def score(reg, p_sol, d2015):
    d92 = pd.Series(np.asarray(p_sol.delta).reshape(p_sol.N, p_sol.S)[:, 1], index=list(p_sol.countries))
    ratio = d92 / d2015[reg['seed']]
    dev_dev = float((ratio[DEV] - 1).abs().mean())
    deving_cnt = int((d2015[reg['seed']][DEVING] < d92[DEVING]).sum())
    deving_ratio = float(ratio[DEVING].mean())
    return dev_dev, deving_cnt, deving_ratio


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    d2015 = {s: seed_patent_delta(s) for s in ['8.0', '8.8']}
    rows = []
    n = len(REGIMES) * len(PARAM_MOMENT); k = 0; t_all = time.perf_counter()
    for reg in REGIMES:
        for cfg in PARAM_MOMENT:
            k += 1; lab = f"{reg['tag']}_{cfg['name']}"; t0 = time.perf_counter()
            try:
                label, p_sol, obj = run_one(reg, cfg)
                dev_dev, dcnt, dratio = score(reg, p_sol, d2015)
                rows.append(dict(regime=reg['tag'], config=cfg['name'], label=label,
                                 calib='+'.join(cfg['calib']) or 'core', moments='+'.join(cfg['mom']) or 'base',
                                 objective=round(obj, 4), dev_ratio_minus1=round(dev_dev, 4),
                                 deving_lower_cnt=dcnt, deving_ratio=round(dratio, 4)))
                print(f"[{k}/{n}] {label:14s} obj={obj:.3f} dev|r-1|={dev_dev:.3f} "
                      f"deving={dcnt}/6 ratio={dratio:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[{k}/{n}] {lab:14s} FAILED: {e}", flush=True)
    df = pd.DataFrame(rows)
    # rank: criteria first (all 6 developing lower, then small developed deviation), obj as tiebreak
    df['meets_deving'] = df['deving_lower_cnt'] == len(DEVING)
    df = df.sort_values(['meets_deving', 'dev_ratio_minus1'], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(OUTDIR + 'config_search_summary.csv', index=False)
    print('\n=== RANKED (criteria: deving all-lower, then developed similarity) ===')
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    print(f"\nsaved {OUTDIR}config_search_summary.csv   total {time.perf_counter()-t_all:.0f}s")
