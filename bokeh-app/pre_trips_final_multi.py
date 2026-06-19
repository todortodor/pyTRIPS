#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final PRE-TRIPS (1992) multi-sector lever calibrations (baseline 2002), OLD patent
fee, with the CORRECTED 1992 turnover targets (1989-1995 window):
  TOPATENT_target = 0.127978 , TOPHARMACHEM_target = 0.094810
Saved as NUMERIC variations so main.py can load them:
  separate (seed 8.0): 17.0 base / 17.1 +fe,fo / 17.2 +nu / 17.3 +all
  common   (seed 8.8): 18.0 / 18.1 / 18.2 / 18.3
  common2015/diff1992 (seed 8.8): 19.0 / 19.1 / 19.2 / 19.3
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import os, shutil, time, warnings
import numpy as np, pandas as pd
from scipy import optimize
from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver
warnings.filterwarnings('ignore')

DRY_RUN = False
YEAR = 1992
VARDIR = 'calibration_results_matched_economy/baseline_2002_variations/'
RAW_DATA_DIR = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}/'
DATA_DIR = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}_RDPHARMACHEM_fix/'
ANBERD_FILE = 'data/anberd.csv'
FILL_REF_AREA = {7: 'CAN', 8: 'KOR'}

TOPATENT_1992 = 0.127978
TOPHARMACHEM_1992 = 0.094810

BASE_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP', 'UUPCOSTS', 'RDPHARMACHEM']
BASE_WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1, 'SRGDP': 1,
                'UUPCOSTS': 1, 'RDPHARMACHEM': 1, 'TOPATENT': 5, 'TOPHARMACHEM': 5}
TURNOVER_MOMENTS = ['TOPATENT', 'TOPHARMACHEM']
MAX_ITER = 5

REGIMES = [dict(tag='s', seed='8.0', fix_delta=False, num=17),
           dict(tag='c', seed='8.8', fix_delta=True,  num=18),
           dict(tag='cd', seed='8.8', fix_delta=False, num=19)]
LEVERS = [dict(name='base', sub=0, extra_calib=[],                 extra_moments=[]),
          dict(name='fefo', sub=1, extra_calib=['fe', 'fo'],       extra_moments=[]),
          dict(name='nu',   sub=2, extra_calib=['nu'],             extra_moments=TURNOVER_MOMENTS),
          dict(name='all',  sub=3, extra_calib=['fe', 'fo', 'nu'], extra_moments=TURNOVER_MOMENTS)]
CONFIGS = [dict(label=f"{r['tag']}_{lv['name']}", out=f"{r['num']}.{lv['sub']}", **r, **lv)
           for r in REGIMES for lv in LEVERS]

SOLVER_KWARGS = dict(cobweb_anim=False, tol=1e-13, accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False, safe_convergence=0.001,
    disp_summary=False, damping=10, max_count=3e3, accel_memory=50, accel_type1=True,
    accel_regularization=1e-10, accel_relaxation=0.5, accel_safeguard_factor=1,
    accel_max_weight_norm=1e6, damping_post_acceleration=5)


def earliest_rd_ratios(ref_area):
    d = pd.read_csv(ANBERD_FILE)
    d = d[(d.CRITERIA == 'MA') & (d.UNIT_MEASURE == 'USD_PPP') & (d.PRICE_BASE == 'V') & (d.REF_AREA == ref_area)]
    g = d.groupby(['TIME_PERIOD', 'ACTIVITY'])['OBS_VALUE'].sum().unstack('ACTIVITY').dropna(subset=['C20', 'C21', '_T']).sort_index()
    yr = g.index[0]
    return g.loc[yr, 'C21'] / g.loc[yr, '_T'], g.loc[yr, 'C20'] / g.loc[yr, '_T']


def build_patched_data_dir():
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    shutil.copytree(RAW_DATA_DIR, DATA_DIR)
    needed, cols = [1, 2, 3, 7, 8], ['RD ratio pharma', 'RD ratio chemicals']
    base = pd.read_csv(RAW_DATA_DIR + 'country_sector_moments.csv', index_col=0); base.index = base.index.astype(int)
    base = base.reindex(base.index.union(needed))
    for r in needed:
        if any(pd.isna(base.loc[r, c]) for c in cols):
            pharma, chem = earliest_rd_ratios(FILL_REF_AREA[r])
            for c, v in zip(cols, [pharma, chem]):
                if pd.isna(base.loc[r, c]):
                    base.loc[r, c] = v
    base.sort_index().to_csv(DATA_DIR + 'country_sector_moments.csv')


def setup(cfg):
    p = parameters(); p.correct_eur_patent_cost = True
    p.fix_delta_across_sectors = cfg['fix_delta']
    p.load_run(VARDIR + cfg['seed'] + '/')
    p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=3)
    p.calib_parameters = ['delta', 'T', 'eta'] + cfg['extra_calib']
    if cfg['fix_delta']:
        for s in range(2, p.S):
            p.delta[:, s] = p.delta[:, 1]
    m = moments(); m.load_run(VARDIR + cfg['seed'] + '/'); m.load_data(DATA_DIR)
    m.aggregate_moments = True
    m.list_of_moments = BASE_MOMENTS + cfg['extra_moments']
    m.weights_dict.update(BASE_WEIGHTS)
    m.drop_CHN_IND_BRA_ROW_from_RD = True
    if 'TOPATENT' in m.list_of_moments:               # corrected 1992 turnover targets
        m.TOPATENT_target = np.float64(TOPATENT_1992)
        m.TOPHARMACHEM_target = np.float64(TOPHARMACHEM_1992)
    dmask = p.mask['delta'].reshape(p.N, p.S).sum(axis=0)
    print(f"  [{cfg['label']}->{cfg['out']}] calib={p.calib_parameters} delta_free={dmask.tolist()} "
          f"moments={m.list_of_moments}")
    if 'TOPATENT' in m.list_of_moments:
        print(f"     turnover targets: TOPATENT={float(m.TOPATENT_target)} TOPHARMACHEM={float(m.TOPHARMACHEM_target)}")
    assert dmask[2] == (0 if cfg['fix_delta'] else p.N)
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
    if cfg['fix_delta']:
        for s in range(2, p_sol.S):
            p_sol.delta[:, s] = p_sol.delta[:, 1]
    sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
    p_sol.guess = sol.x; sol_c.scale_P(p_sol); sol_c.compute_non_solver_quantities(p_sol); p_sol.tau = sol_c.tau
    m.compute_moments(sol_c, p_sol); m.compute_moments_deviations()
    out = VARDIR + cfg['out'] + '/'; os.makedirs(out, exist_ok=True)
    p_sol.write_params(out); m.write_moments(out)
    print(f"  [{cfg['label']}->{cfg['out']}] DONE obj={np.linalg.norm(m.deviation_vector()):.3f} -> {out}")


if __name__ == '__main__':
    build_patched_data_dir()
    print(f"{'DRY-RUN' if DRY_RUN else 'RUN'} {len(CONFIGS)} multi configs (old fee, corrected 1992 turnover)\n")
    for cfg in CONFIGS:
        run(cfg)
    if DRY_RUN:
        print("\nDRY_RUN -> validated. set DRY_RUN=False to run.")
