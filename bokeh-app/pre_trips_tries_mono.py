#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONO patenting-sector counterpart of pre_trips_tries.py — same pre-TRIPS lever
exercise for the single-sector baseline 2000, kept consistent with the
multi-sector run.

No separate/common-delta dimension here (one patenting sector). Levers on top of
the base calib delta,T,eta:
  base, +fe/fo, +nu (target the turnover moment TO), all(+fe/fo/nu)
All runs use the NEW 1992 US patenting fee ($14,370 in 1994 dollars). For the
mono model the patenting-cost moment is UUPCOST (scalar, no S); it is linear in
the US fee, so UUPCOST_target is rescaled by the same deflator-based ratio as
the multi-sector UUPCOSTS. Hjort factors are recomputed from 1992 GDP (asserted).

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
from solver_funcs import calibration_func, fixed_point_solver

warnings.filterwarnings('ignore')

DRY_RUN   = False
BASELINE  = '2000'
YEAR      = 1992
SEED_PATH = f'calibration_results_matched_economy/{BASELINE}/'          # 2015 single-sector
DATA_DIR  = f'data_smooth_3_years/data_12_countries_{YEAR}/'            # smoothed 2-sector 1992
VARDIR    = f'calibration_results_matched_economy/baseline_{BASELINE}_variations/'
OUT_ROOT  = VARDIR + 'pre_trips_tries_mono_newfee/'
COMPARISON_CSV = OUT_ROOT + 'pre_trips_tries_mono_comparison.csv'

# base moments mirror the single-sector pre-TRIPS (9.2) + 9.2-aligned weights;
# the +nu lever adds the (single-sector) turnover moment TO.
BASE_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP', 'UUPCOST']
BASE_WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1,
                'SRGDP': 1, 'UUPCOST': 1, 'TO': 5}
TURNOVER_MOMENTS = ['TO']

MAX_ITER = 5
DEVELOPED = ['USA', 'EUR', 'JAP', 'CAN', 'KOR']

# --- new 1992 US patenting fee rescale (same as multi; UUPCOST is linear in fee) ---
NEW_FEE_1994, BASE_FEE_2005 = 14370.0, 17078.0
_defl = pd.read_csv('/Users/slepot/Dropbox/TRIPS/Calibration data/WDI_gdp_deflator.csv',
                    keep_default_na=False, na_values=['..', ''], skiprows=4
                    ).set_index('Country Code').loc['USA']
def _d(yr):
    return float(_defl[[c for c in _defl.index if c.startswith(yr)][0]])
NEW_FEE_RATIO = (NEW_FEE_1994 / _d('1994')) / (BASE_FEE_2005 / _d('2005'))
print(f"new-fee UUPCOST rescale ratio = {NEW_FEE_RATIO:.5f}")

SOLVER_KWARGS = dict(
    cobweb_anim=False, tol=1e-13, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=False, damping=10, max_count=3e3,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)

LEVERS = [
    dict(name='base', extra_calib=[],                 extra_moments=[]),
    dict(name='fefo', extra_calib=['fe', 'fo'],       extra_moments=[]),
    dict(name='nu',   extra_calib=['nu'],             extra_moments=TURNOVER_MOMENTS),
    dict(name='all',  extra_calib=['fe', 'fo', 'nu'], extra_moments=TURNOVER_MOMENTS),
]


def setup_config(cfg):
    p = parameters()
    p.correct_eur_patent_cost = True
    p.load_run(SEED_PATH)
    seed_rhjort = np.asarray(p.r_hjort).copy()
    p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=2)   # -> r_hjort(1992)
    p.calib_parameters = ['delta', 'T', 'eta'] + cfg['extra_calib']

    m = moments()
    m.load_run(SEED_PATH)
    m.load_data(DATA_DIR)
    m.aggregate_moments = True
    m.list_of_moments = BASE_MOMENTS + cfg['extra_moments']
    m.weights_dict.update(BASE_WEIGHTS)
    m.drop_CHN_IND_BRA_ROW_from_RD = True                  # match the multi-sector exercise

    # new 1992 US patenting fee -> rescale the (scalar) UUPCOST target.
    # keep it as a numpy scalar (deviation code calls .size on the target).
    m.UUPCOST_target = np.float64(m.UUPCOST_target) * NEW_FEE_RATIO

    dmask = p.mask['delta'].reshape(p.N, p.S).sum(axis=0)
    rhjort_changed = not np.allclose(np.asarray(p.r_hjort), seed_rhjort)
    cond = {
        'label': cfg['name'], 'calib_parameters': p.calib_parameters,
        'delta_free_by_sector': dmask.tolist(),
        'fe_free': int(p.mask['fe'].sum()) if 'fe' in p.calib_parameters else 0,
        'fo_free': int(p.mask['fo'].sum()) if 'fo' in p.calib_parameters else 0,
        'nu_free': int(p.mask['nu'].sum()) if 'nu' in p.calib_parameters else 0,
        'moments': m.list_of_moments,
        'weights': {mm: m.weights_dict[mm] for mm in m.list_of_moments},
        'r_hjort_recomputed_for_1992': bool(rhjort_changed),
        'UUPCOST_target_newfee': round(float(m.UUPCOST_target), 7),
        'nu_seed_patent': round(float(p.nu[1]), 5),
    }
    print('\n' + '=' * 72)
    for k, v in cond.items():
        print(f"  {k}: {v}")
    assert dmask[0] == 0 and dmask[1] == p.N, 'delta mask wrong (sector0 fixed, sector1 free)'
    assert rhjort_changed, 'r_hjort NOT recomputed from 1992 GDP'
    if 'fe' in p.calib_parameters:
        assert cond['fe_free'] >= 1 and cond['fo_free'] >= 1
    if 'nu' in p.calib_parameters:
        assert cond['nu_free'] >= 1 and 'TO' in m.list_of_moments
    return p, m


def run_config(cfg):
    p, m = setup_config(cfg)
    sol, sol_init = fixed_point_solver(p, x0=p.guess, context='calibration', **SOLVER_KWARGS)
    sol_init.scale_P(p); sol_init.compute_non_solver_quantities(p)
    p.guess = sol.x
    m.compute_moments(sol_init, p); m.compute_moments_deviations()
    print(f"  [{cfg['name']}] initial solve: {sol.status} | obj={np.linalg.norm(m.deviation_vector()):.4f}")
    if DRY_RUN:
        return None

    hist = history(*tuple(m.list_of_moments + ['objective']))
    bounds = p.make_parameters_bounds()
    t0 = time.perf_counter()
    for it in range(MAX_ITER + 1):
        xtol = 1e-10 if it < MAX_ITER - 2 else 1e-16
        res = optimize.least_squares(
            fun=calibration_func, x0=p.make_p_vector(), args=(p, m, p.guess, hist, t0),
            bounds=bounds, max_nfev=1e8, xtol=xtol, verbose=2)
        p.update_parameters(res.x)
    p_sol = p.copy(); p_sol.update_parameters(res.x)
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
    p_sol.guess = sol.x
    sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    obj = float(np.linalg.norm(m.deviation_vector()))

    out = OUT_ROOT + cfg['name'] + '/'
    os.makedirs(out, exist_ok=True)
    p_sol.write_params(out); m.write_moments(out)
    d = np.asarray(p_sol.delta).reshape(p_sol.N, p_sol.S)[:, 1]
    nu_patent = float(np.asarray(p_sol.nu)[1])
    print(f"  [{cfg['name']}] DONE obj={obj:.4f} ({sol.status}) nu_patent={nu_patent:.4f} -> {out}")
    return dict(label=cfg['name'], obj=obj, delta=d.copy(), nu_patent=nu_patent)


if __name__ == '__main__':
    os.makedirs(OUT_ROOT, exist_ok=True)
    p15 = parameters(); p15.load_run(SEED_PATH)
    countries = list(p15.countries)
    delta_2015 = np.asarray(p15.delta).reshape(p15.N, p15.S)[:, 1]

    print(f"\n{'DRY-RUN' if DRY_RUN else 'RUNNING'} {len(LEVERS)} mono configs\n")
    rows = []
    for cfg in LEVERS:
        res = run_config(cfg)
        if res is None:
            continue
        for i, c in enumerate(countries):
            rows.append({'config': res['label'], 'country': c, 'obj': res['obj'],
                         'delta_patent_1992': res['delta'][i],
                         'delta_patent_2015': delta_2015[i],
                         'ratio': res['delta'][i] / delta_2015[i],
                         'nu_patent': res['nu_patent']})
        pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False)

    if DRY_RUN:
        print("\nDRY_RUN=True -> validated, nothing calibrated. Set DRY_RUN=False to run.")
    else:
        tab = pd.DataFrame(rows)
        tab.to_csv(COMPARISON_CSV, index=False)
        print(f"\nsaved {COMPARISON_CSV}")
        piv = tab.pivot(index='country', columns='config', values='ratio')
        pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print("\n=== ratio delta_1992/delta_2015 (patenting sector) by config ===")
        print(piv.to_string())
        print("\n=== calibrated nu_patent by config + objective ===")
        print(tab.drop_duplicates('config').set_index('config')[['nu_patent', 'obj']].to_string())
