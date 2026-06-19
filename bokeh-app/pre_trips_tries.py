#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-variant pre-TRIPS calibration of the multi-sector model (baseline 2002),
to test which levers bring the 1992 (pre-TRIPS) deltas of DEVELOPED countries
down (so we can argue protection didn't rise much there post-TRIPS).

Levers requested:
  (1) calibrate the patenting cost params fe, fo in 1992
  (2) Hjort factors recomputed from 1992 GDP        -> ALREADY in effect:
      load_data recomputes r_hjort (classes.py:200) from the loaded (1992) data
      with the same formula as update_khi_and_r_hjort; asserted per config.
  (3) target the turnover moments (TOPATENT, TOPHARMACHEM) and calibrate nu
  (4) common delta across sectors in 2015 but differentiated in 1992
      (seed the common-delta 8.8 run, but do NOT tie delta in the 1992 fit)

Run for BOTH 2015 bases (8.0 separate, 8.8 common). Regimes:
  s  : seed 8.0 (separate 2015), 1992 separate delta
  c  : seed 8.8 (common 2015),   1992 EQUAL delta (fix_delta_across_sectors)
  cd : seed 8.8 (common 2015),   1992 separate delta            <- lever (4)
Lever sets (added on top of the base calib delta,T,eta):
  base, +fe/fo, +nu(+turnover moments), all(+fe/fo/nu(+turnover))

Everything else matches the 8.1/8.81 runs: smoothed 3-year 1992 data,
9.2-aligned base weights, CAN/KOR RDPHARMACHEM backfill from earliest ANBERD.

Set DRY_RUN=True to load + print/assert every config's conditions + one solve,
then stop (cheap validation). DRY_RUN=False runs all calibrations.

Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')   # never block on plt.show() in the solver

import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from scipy import optimize

from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Fixed setup (shared across configs)
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN   = False
BASELINE  = '2002'
YEAR      = 1992
VARDIR    = f'calibration_results_matched_economy/baseline_{BASELINE}_variations/'
RAW_DATA_DIR = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}/'
DATA_DIR     = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}_RDPHARMACHEM_fix/'
ANBERD_FILE  = 'data/anberd.csv'
OUT_ROOT  = VARDIR + 'pre_trips_tries_newfee/'
COMPARISON_CSV = OUT_ROOT + 'pre_trips_tries_comparison.csv'

# --- new 1992 US patenting fee: $14,370 in 1994 dollars (replaces the base
# $17,078 in 2005 dollars that the UUPCOSTS formula deflates). UUPCOSTS is linear
# in the US fee, so we rescale the 1992 UUPCOSTS_target by new_fee/old_fee, both
# expressed in 1992 dollars via the US GDP deflator (the maker's deflator). ---
NEW_FEE_1994  = 14370.0
BASE_FEE_2005 = 17078.0
_defl = pd.read_csv('/Users/slepot/Dropbox/TRIPS/Calibration data/WDI_gdp_deflator.csv',
                    keep_default_na=False, na_values=['..', ''], skiprows=4
                    ).set_index('Country Code').loc['USA']
def _d(yr):
    return float(_defl[[c for c in _defl.index if c.startswith(yr)][0]])
NEW_FEE_RATIO = (NEW_FEE_1994 / _d('1994')) / (BASE_FEE_2005 / _d('2005'))
print(f"new-fee UUPCOSTS rescale ratio = {NEW_FEE_RATIO:.5f} "
      f"(${NEW_FEE_1994:.0f} 1994$ vs ${BASE_FEE_2005:.0f} 2005$)")

BASE_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP',
                'UUPCOSTS', 'RDPHARMACHEM']
BASE_WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1,
                'SRGDP': 1, 'UUPCOSTS': 1, 'RDPHARMACHEM': 1}
TURNOVER_MOMENTS = ['TOPATENT', 'TOPHARMACHEM']
TURNOVER_WEIGHTS = {'TOPATENT': 5, 'TOPHARMACHEM': 5}   # matches 8.0's TO weights

MAX_ITER = 5
FILL_REF_AREA = {7: 'CAN', 8: 'KOR'}
DEVELOPED = ['USA', 'EUR', 'JAP', 'CAN', 'KOR']

SOLVER_KWARGS = dict(
    cobweb_anim=False, tol=1e-13, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=False, damping=10, max_count=3e3,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)

# ─────────────────────────────────────────────────────────────────────────────
# The 12 configs: regime (seed + 1992 delta tie) x lever set
# ─────────────────────────────────────────────────────────────────────────────
REGIMES = [
    dict(tag='s',  seed='8.0', fix_delta=False),   # separate 2015 / separate 1992
    dict(tag='c',  seed='8.8', fix_delta=True),    # common 2015 / equal 1992
    dict(tag='cd', seed='8.8', fix_delta=False),   # common 2015 / differentiated 1992 (lever 4)
]
LEVERS = [
    dict(name='base', extra_calib=[],                  extra_moments=[]),
    dict(name='fefo', extra_calib=['fe', 'fo'],        extra_moments=[]),
    dict(name='nu',   extra_calib=['nu'],              extra_moments=TURNOVER_MOMENTS),
    dict(name='all',  extra_calib=['fe', 'fo', 'nu'],  extra_moments=TURNOVER_MOMENTS),
]
CONFIGS = [dict(label=f"{r['tag']}_{lv['name']}", **r, **lv)
           for r in REGIMES for lv in LEVERS]


# ─────────────────────────────────────────────────────────────────────────────
# 1992 RDPHARMACHEM backfill (CAN row 7, KOR row 8) from earliest ANBERD year
# ─────────────────────────────────────────────────────────────────────────────
def earliest_rd_ratios(ref_area):
    d = pd.read_csv(ANBERD_FILE)
    d = d[(d.CRITERIA == 'MA') & (d.UNIT_MEASURE == 'USD_PPP') &
          (d.PRICE_BASE == 'V') & (d.REF_AREA == ref_area)]
    g = d.groupby(['TIME_PERIOD', 'ACTIVITY'])['OBS_VALUE'].sum().unstack('ACTIVITY')
    g = g.dropna(subset=['C20', 'C21', '_T']).sort_index()
    yr = g.index[0]
    return g.loc[yr, 'C21'] / g.loc[yr, '_T'], g.loc[yr, 'C20'] / g.loc[yr, '_T'], int(yr)


def build_patched_data_dir():
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    shutil.copytree(RAW_DATA_DIR, DATA_DIR)
    needed, cols = [1, 2, 3, 7, 8], ['RD ratio pharma', 'RD ratio chemicals']
    base = pd.read_csv(RAW_DATA_DIR + 'country_sector_moments.csv', index_col=0)
    base.index = base.index.astype(int)
    base = base.reindex(base.index.union(needed))
    filled = []
    for r in needed:
        if any(pd.isna(base.loc[r, c]) for c in cols):
            pharma, chem, yr = earliest_rd_ratios(FILL_REF_AREA[r])
            vals = {'RD ratio pharma': pharma, 'RD ratio chemicals': chem}
            for c in cols:
                if pd.isna(base.loc[r, c]):
                    base.loc[r, c] = vals[c]; filled.append((r, c, round(vals[c], 4), yr))
    base.sort_index().to_csv(DATA_DIR + 'country_sector_moments.csv')
    print('RDPHARMACHEM backfill (earliest ANBERD):', filled)


# ─────────────────────────────────────────────────────────────────────────────
# Per-config setup + (optional) calibration. Prints/asserts run conditions.
# ─────────────────────────────────────────────────────────────────────────────
def setup_config(cfg):
    seed_path = VARDIR + cfg['seed'] + '/'
    p = parameters()
    p.correct_eur_patent_cost = True
    p.fix_delta_across_sectors = cfg['fix_delta']   # BEFORE load -> mask built right
    p.load_run(seed_path)
    seed_rhjort = np.asarray(p.r_hjort).copy()       # 2015 value loaded from csv
    p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=3)  # -> recompute r_hjort(1992)
    p.calib_parameters = ['delta', 'T', 'eta'] + cfg['extra_calib']
    if cfg['fix_delta']:
        for s in range(2, p.S):
            p.delta[:, s] = p.delta[:, 1]

    m = moments()
    m.load_run(seed_path)
    m.load_data(DATA_DIR)
    m.aggregate_moments = True
    m.list_of_moments = BASE_MOMENTS + cfg['extra_moments']
    m.weights_dict.update({**BASE_WEIGHTS, **TURNOVER_WEIGHTS})
    m.drop_CHN_IND_BRA_ROW_from_RD = True

    # new 1992 US patenting fee -> rescale the UUPCOSTS target (linear in the fee)
    uup_before = np.asarray(m.UUPCOSTS_target, dtype=float).copy()
    m.UUPCOSTS_target = uup_before * NEW_FEE_RATIO

    # ---- assert / print run conditions ----
    dmask = p.mask['delta'].reshape(p.N, p.S).sum(axis=0)
    rhjort_changed = not np.allclose(np.asarray(p.r_hjort), seed_rhjort)
    expect_dmask_s2 = 0 if cfg['fix_delta'] else p.N
    cond = {
        'label': cfg['label'], 'seed': cfg['seed'], 'fix_delta': cfg['fix_delta'],
        'calib_parameters': p.calib_parameters,
        'delta_free_by_sector': dmask.tolist(),
        'fe_free': int(p.mask['fe'].sum()) if 'fe' in p.calib_parameters else 0,
        'fo_free': int(p.mask['fo'].sum()) if 'fo' in p.calib_parameters else 0,
        'nu_free': int(p.mask['nu'].sum()) if 'nu' in p.calib_parameters else 0,
        'moments': m.list_of_moments,
        'weights': {mm: m.weights_dict[mm] for mm in m.list_of_moments},
        'r_hjort_recomputed_for_1992': bool(rhjort_changed),
        'UUPCOSTS_target_newfee': np.round(np.asarray(m.UUPCOSTS_target, float), 6).tolist(),
        'nu_seed(rest,pharma)': [round(float(p.nu[1]), 5), round(float(p.nu[2]), 5)],
    }
    print('\n' + '=' * 72)
    for k, v in cond.items():
        print(f"  {k}: {v}")
    # hard checks
    assert dmask[0] == 0, 'sector-0 delta should be fixed'
    assert dmask[2] == expect_dmask_s2, \
        f"sector-2 delta free count {dmask[2]} != expected {expect_dmask_s2} for fix_delta={cfg['fix_delta']}"
    assert dmask[1] == p.N, 'sector-1 delta should be free for all countries'
    assert rhjort_changed, 'r_hjort NOT recomputed from 1992 GDP (Hjort lever broken)'
    if 'fe' in p.calib_parameters:
        assert cond['fe_free'] >= p.S - 1 and cond['fo_free'] >= p.S - 1
    if 'nu' in p.calib_parameters:
        assert cond['nu_free'] >= p.S - 1
        assert all(t in m.list_of_moments for t in TURNOVER_MOMENTS), 'turnover moments missing for nu'
    return p, m


def run_config(cfg):
    p, m = setup_config(cfg)
    # initial solve at seeded params on 1992 data (warm-start + sanity)
    sol, sol_init = fixed_point_solver(p, x0=p.guess, context='calibration', **SOLVER_KWARGS)
    sol_init.scale_P(p); sol_init.compute_non_solver_quantities(p)
    p.guess = sol.x
    m.compute_moments(sol_init, p); m.compute_moments_deviations()
    print(f"  [{cfg['label']}] initial solve: {sol.status} | "
          f"obj={np.linalg.norm(m.deviation_vector()):.4f}")
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
    if cfg['fix_delta']:                              # exact equality for the final solve
        for s in range(2, p_sol.S):
            p_sol.delta[:, s] = p_sol.delta[:, 1]
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
    p_sol.guess = sol.x
    sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    obj = float(np.linalg.norm(m.deviation_vector()))

    out = OUT_ROOT + cfg['label'] + '/'
    os.makedirs(out, exist_ok=True)
    p_sol.write_params(out); m.write_moments(out)
    d = np.asarray(p_sol.delta).reshape(p_sol.N, p_sol.S)
    nu = np.asarray(p_sol.nu, dtype=float)
    print(f"  [{cfg['label']}] DONE  obj={obj:.4f}  status={sol.status}  "
          f"nu_rest={nu[1]:.4f} nu_pharma={nu[2]:.4f}  -> {out}")
    return dict(label=cfg['label'], seed=cfg['seed'], status=sol.status, obj=obj,
                rest=d[:, 1].copy(), pharma=d[:, 2].copy(),
                nu_rest=float(nu[1]), nu_pharma=float(nu[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_ROOT, exist_ok=True)
    build_patched_data_dir()

    # 2015 reference deltas (separate base 8.0, common base 8.8)
    ref = {}
    for s in ['8.0', '8.8']:
        ps = parameters(); ps.load_run(VARDIR + s + '/')
        ds = np.asarray(ps.delta).reshape(ps.N, ps.S)
        ref[s] = dict(rest=ds[:, 1].copy(), pharma=ds[:, 2].copy())
        countries = list(ps.countries)

    print(f"\n{'DRY-RUN: validating' if DRY_RUN else 'RUNNING'} {len(CONFIGS)} configs\n")
    rows = []
    for cfg in CONFIGS:
        res = run_config(cfg)
        if res is None:
            continue
        ref15 = ref['8.0'] if cfg['seed'] == '8.0' else ref['8.8']
        for i, c in enumerate(countries):
            rows.append({
                'config': res['label'], 'country': c, 'obj': res['obj'], 'status': res['status'],
                'delta_rest_1992': res['rest'][i], 'delta_pharma_1992': res['pharma'][i],
                'delta_rest_2015': ref15['rest'][i], 'delta_pharma_2015': ref15['pharma'][i],
                'ratio_rest': res['rest'][i] / ref15['rest'][i],
                'ratio_pharma': res['pharma'][i] / ref15['pharma'][i],
                'nu_rest': res['nu_rest'], 'nu_pharma': res['nu_pharma'],
            })
        pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False)   # incremental

    if DRY_RUN:
        print("\nDRY_RUN=True -> all configs validated, no calibration run. "
              "Set DRY_RUN=False to run.")
    else:
        tab = pd.DataFrame(rows)
        tab.to_csv(COMPARISON_CSV, index=False)
        print(f"\nsaved {COMPARISON_CSV}")
        # developed-country focus: post-TRIPS ratio (1992/2015) of the rest-of-patenting delta
        dev = tab[tab.country.isin(DEVELOPED)]
        piv = dev.pivot(index='country', columns='config', values='ratio_rest')
        pd.set_option('display.width', 240); pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print("\n=== developed-country ratio delta_1992/delta_2015 (rest-of-patenting) by config ===")
        print(piv.to_string())
        # per-config nu (rest, pharma) and objective
        nutab = tab.drop_duplicates('config').set_index('config')[['nu_rest', 'nu_pharma', 'obj']]
        print("\n=== calibrated nu by config (rest, pharma) + objective ===")
        print(nutab.to_string())
