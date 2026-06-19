#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick report: equilibrium deltas and (dynamic, consumption-equivalent) welfare
changes for baseline-2002 multi-sector calibrations 8.0 (separate delta) and
8.8 (common delta), at the Nash and cooperative (Negishi / Equal) equilibria.

For each case x equilibrium it loads the saved (dynamic) equilibrium deltas and
re-solves the model WITH DYNAMICS relative to that case's OWN calibrated economy,
to get per-country welfare gains and the two world-welfare measures.

Reads:  coop_eq_direct_saves/dyn_2002_{case}_{nash,negishi,pop_weighted}/
Writes: output/report_2002_8.0_8.8_equilibria_{deltas,welfare}.csv  (+ console)
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import os, warnings, pickle
import numpy as np, pandas as pd
from classes import parameters, var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver
warnings.filterwarnings('ignore')

CASES = {                       # label -> (calibration folder, dyn-save key)
    '8.0': ('calibration_results_matched_economy/baseline_2002_variations/8.0/', '2002_8.0'),
    '8.8': ('calibration_results_matched_economy/baseline_2002_variations/8.8/', '2002_8.8'),
}
EQUILIBRIA = {'Nash': 'nash', 'Coop (Negishi)': 'negishi', 'Coop (Equal)': 'pop_weighted'}
SAVE = 'coop_eq_direct_saves/'
OUTDIR = 'output/'

SOLVER_KWARGS = dict(
    context='counterfactual', cobweb_anim=False, tol=1e-14, accelerate=False,
    accelerate_when_stable=True, cobweb_qty='phi', plot_convergence=False,
    plot_cobweb=False, safe_convergence=0.001, disp_summary=False, damping=10,
    max_count=3e3, accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5)
DYN_KWARGS = dict(
    Nt=23, t_inf=500, cobweb_anim=False, tol=1e-14, accelerate=False,
    accelerate_when_stable=False, cobweb_qty='l_R', plot_convergence=False,
    plot_cobweb=False, plot_live=False, safe_convergence=1e-8, disp_summary=False,
    damping=60, max_count=50000, accel_memory=5, accel_type1=True,
    accel_regularization=1e-10, accel_relaxation=1, accel_safeguard_factor=1,
    accel_max_weight_norm=1e6, damping_post_acceleration=10)


def eq_folder(dyn_key, eq_suffix):
    return f"{SAVE}dyn_{dyn_key}_{eq_suffix}/"


def static_solve(p):
    sol, sol_c = fixed_point_solver(p, x0=p.guess, **SOLVER_KWARGS)
    sol_c.scale_P(p); sol_c.compute_non_solver_quantities(p)
    return sol, sol_c


def check_inputs():
    missing = [cal for ck, (cal, _) in CASES.items() if not os.path.isdir(cal)]
    missing += [eq_folder(dk, es) for _, (_, dk) in CASES.items() for es in EQUILIBRIA.values()
                if not os.path.isdir(eq_folder(dk, es))]
    if missing:
        raise SystemExit("MISSING folders:\n  " + "\n  ".join(missing))


CACHE = OUTDIR + '_report_2002_8_cache.pkl'


def compute():
    if os.path.exists(CACHE):
        print(f"(loading cached dynamic solves from {CACHE}; delete it to recompute)")
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    out = {}
    for ck, (cal, dyn_key) in CASES.items():
        print(f"\n===== case {ck} =====")
        p_cal = parameters(); p_cal.correct_eur_patent_cost = True; p_cal.load_run(cal)
        countries, S = list(p_cal.countries), p_cal.S
        sol0, sol_base = static_solve(p_cal)        # calibrated economy = welfare reference
        p_cal.guess = sol0.x
        dcal = np.asarray(p_cal.delta).reshape(p_cal.N, S)
        rec = dict(countries=countries, S=S,
                   delta_cal_pat=dcal[:, 1].copy(),
                   delta_cal_pharma=(dcal[:, 2].copy() if S > 2 else None),
                   g_cal=float(sol_base.g), eq={})
        for eq_name, es in EQUILIBRIA.items():
            p_load = parameters(); p_load.load_run(eq_folder(dyn_key, es))
            delta_eq = np.asarray(p_load.delta).reshape(p_load.N, S)
            p_eq = p_cal.copy(); p_eq.delta = delta_eq.copy()
            sol_eq0, sol_eq = static_solve(p_eq)     # new BGP (growth + dyn warm-start)
            p_eq.guess = sol_eq0.x
            _, dyn_sol = dyn_fixed_point_solver(p_eq, sol_init=sol_base, sol_fin=sol_eq, **DYN_KWARGS)
            dyn_sol.compute_non_solver_quantities(p_eq)
            rec['eq'][eq_name] = dict(
                delta_pat=delta_eq[:, 1].copy(),
                delta_pharma=(delta_eq[:, 2].copy() if S > 2 else None),
                welfare=np.asarray(dyn_sol.cons_eq_welfare).copy(),
                world_negishi=float(dyn_sol.cons_eq_negishi_welfare_change),
                world_equal=float(dyn_sol.cons_eq_pop_average_welfare_change),
                g=float(sol_eq.g))
            print(f"  {eq_name:16s} world(Negishi)={(rec['eq'][eq_name]['world_negishi']-1)*100:+.2f}%  "
                  f"world(Equal)={(rec['eq'][eq_name]['world_equal']-1)*100:+.2f}%  g={rec['eq'][eq_name]['g']*100:.3f}%")
        out[ck] = rec
    os.makedirs(OUTDIR, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    return out


def report(data):
    os.makedirs(OUTDIR, exist_ok=True)
    cases = list(CASES); countries = data['8.0']['countries']
    multi = data['8.0']['S'] > 2
    pd.set_option('display.width', 400, 'display.max_columns', None,
                  'display.float_format', lambda x: f'{x:.3f}')

    eqs = ['calibrated'] + list(EQUILIBRIA)
    qtys = ['d patent'] + (['d pharma'] if multi else []) + ['welfare %']
    cols = pd.MultiIndex.from_tuples([(ck, eq, q) for ck in cases for eq in eqs for q in qtys],
                                     names=['case', 'equilibrium', 'quantity'])
    agg_rows = ['WORLD (Negishi) %', 'WORLD (Equal) %', 'GROWTH rate %']
    df = pd.DataFrame(index=list(countries) + agg_rows, columns=cols, dtype=float)

    for ck in cases:
        rec = data[ck]
        for eq in eqs:
            # per-country deltas + welfare
            if eq == 'calibrated':
                dpat, dpha, welf = rec['delta_cal_pat'], rec['delta_cal_pharma'], np.ones(len(countries))
                g, wn, we = rec['g_cal'], 1.0, 1.0
            else:
                ee = rec['eq'][eq]
                dpat, dpha, welf = ee['delta_pat'], ee['delta_pharma'], ee['welfare']
                g, wn, we = ee['g'], ee['world_negishi'], ee['world_equal']
            for i, c in enumerate(countries):
                df.loc[c, (ck, eq, 'd patent')] = dpat[i]
                if multi:
                    df.loc[c, (ck, eq, 'd pharma')] = dpha[i]
                df.loc[c, (ck, eq, 'welfare %')] = (welf[i] - 1) * 100
            # aggregate rows go in the 'welfare %' sub-column
            df.loc['WORLD (Negishi) %', (ck, eq, 'welfare %')] = np.nan if eq == 'calibrated' else (wn - 1) * 100
            df.loc['WORLD (Equal) %',   (ck, eq, 'welfare %')] = np.nan if eq == 'calibrated' else (we - 1) * 100
            df.loc['GROWTH rate %',     (ck, eq, 'welfare %')] = g * 100

    df.index.name = 'country'
    out = OUTDIR + 'report_2002_8.0_8.8_equilibria.csv'
    df.to_csv(out)
    print('\n===== baseline 2002 (8.0 / 8.8): equilibrium deltas & welfare changes =====')
    print('  d = delta (lower = stronger protection; equilibrium bounds 0.01 .. 12)')
    print('  welfare % = consumption-equivalent gain vs. each case\'s own calibration\n')
    print(df.to_string(na_rep=''))
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    check_inputs()
    report(compute())
