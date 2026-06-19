#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF comparison (TABLES) of baseline 2000 vs SPFLOW counterfactuals 16.0 and 16.1.

For each of the 3 cases and 3 equilibria (Nash, cooperative-Negishi,
cooperative-Equal/pop_weighted) it loads the saved equilibrium deltas and
re-solves the model WITH DYNAMICS (9 dynamic solves), each relative to that
case's own calibrated economy, to get consumption-equivalent welfare gains and
growth rates. Output: a multi-page PDF of tables.

Requires the saved equilibria in coop_eq_direct_saves/dyn_2000_{case}_{eq}/.
Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from classes import parameters, var
from solver_funcs import fixed_point_solver, dyn_fixed_point_solver

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CASES = {                       # label -> (calibration folder, dyn-save key)
    'baseline': ('calibration_results_matched_economy/2000/',                          '2000_baseline'),
    '16.0':     ('calibration_results_matched_economy/baseline_2000_variations/16.0/', '2000_16.0'),
    '16.1':     ('calibration_results_matched_economy/baseline_2000_variations/16.1/', '2000_16.1'),
}
EQUILIBRIA = {                  # display name -> dyn-save suffix
    'Nash':           'nash',
    'Coop (Negishi)': 'negishi',
    'Coop (Equal)':   'pop_weighted',
}
# explicit names for the counterfactual runs
CASE_DEF = {
    'baseline': 'Baseline calibration (2000).',
    '16.0': 'More patents to developed economies (by destination): international '
            'patent-flow shares Pat(n,i) x2.0 (+100%) to developed destinations '
            '(USA, EUR, JAP, CAN, KOR) and x0.5 (-50%) to developing (CHN, BRA, IND, '
            'RUS, MEX, ZAF, ROW), then renormalized; plus DOMPATINUS, SINNOVPATUS '
            'and UUPCOST targets x2.0 (+100%).',
    '16.1': 'US more central to the world (by origin): US-origin international '
            'patent-flow shares Pat(n,i) x2.5 (+150%), renormalized; US R&D x2.5 '
            '(+150%); turnover target x0.7 (-30%, with nu kept free); UUPCOST x2.5 '
            '(+150%); DOMPATINUS x1.3 (+30%); and the US SRGDP share x1.15 (+15%, '
            'renormalized).',
}
CAPTION = ('16.0 = more patents to developed economies (intl. patent-flow shares x2.0 to developed '
           'destinations, x0.5 to developing; DOMPATINUS/SINNOVPATUS/UUPCOST x2.0).    '
           '16.1 = US more central (US-origin intl. flows x2.5, US R&D x2.5, turnover x0.7, UUPCOST x2.5, '
           'DOMPATINUS x1.3, US SRGDP x1.15).')

SAVE = 'coop_eq_direct_saves/'
OUT_PDF = 'output/compare_baseline_16.0_16.1.pdf'

# delta cell-coloring (equilibrium tables only): only the corner/extreme values
# are highlighted -> 0.01 (strong protection) green, 12 (no protection) red.
DELTA_LB, DELTA_UB = 0.01, 12
GREEN, RED = '#7fbf7b', '#e06666'

SOLVER_KWARGS = dict(
    context='counterfactual', cobweb_anim=False, tol=1e-14, accelerate=False,
    accelerate_when_stable=True, cobweb_qty='phi', plot_convergence=False,
    plot_cobweb=False, safe_convergence=0.001, disp_summary=False, damping=10,
    max_count=3e3, accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)
DYN_KWARGS = dict(               # matches coop_eq_dyn / find_coop_eq dynamics
    Nt=23, t_inf=500, cobweb_anim=False, tol=1e-14, accelerate=False,
    accelerate_when_stable=False, cobweb_qty='l_R', plot_convergence=False,
    plot_cobweb=False, plot_live=False, safe_convergence=1e-8, disp_summary=False,
    damping=60, max_count=50000, accel_memory=5, accel_type1=True,
    accel_regularization=1e-10, accel_relaxation=1, accel_safeguard_factor=1,
    accel_max_weight_norm=1e6, damping_post_acceleration=10,
)


def eq_folder(case_key, eq_suffix):
    return f"{SAVE}dyn_{CASES[case_key][1]}_{eq_suffix}/"


def check_inputs():
    missing = [CASES[ck][0] for ck in CASES if not os.path.isdir(CASES[ck][0])]
    missing += [eq_folder(ck, es) for ck in CASES for es in EQUILIBRIA.values()
                if not os.path.isdir(eq_folder(ck, es))]
    if missing:
        print("MISSING required folders:")
        for m in missing:
            print("  ", m)
        raise SystemExit("Cannot run until all calibration + equilibrium folders exist.")


def load_delta_patent(folder):
    p = parameters(); p.load_run(folder)
    return np.asarray(p.delta).reshape(p.N, p.S)[:, 1]


def static_solve(p):
    sol, sol_c = fixed_point_solver(p, x0=p.guess, **SOLVER_KWARGS)
    sol_c.scale_P(p); sol_c.compute_non_solver_quantities(p)
    return sol, sol_c


# ─────────────────────────────────────────────────────────────────────────────
# Compute everything (9 dynamic solves)
# ─────────────────────────────────────────────────────────────────────────────
def compute():
    out = {}
    for ck, (cal, _) in CASES.items():
        print(f"\n===== case {ck} =====")
        p_cal = parameters(); p_cal.correct_eur_patent_cost = True; p_cal.load_run(cal)
        countries = list(p_cal.countries)
        sol0, sol_base = static_solve(p_cal)        # calibrated economy = welfare reference
        p_cal.guess = sol0.x
        rec = dict(
            countries=countries,
            delta_cal=np.asarray(p_cal.delta).reshape(p_cal.N, p_cal.S)[:, 1].copy(),
            eta_cal=np.asarray(p_cal.eta).reshape(p_cal.N, p_cal.S)[:, 1].copy(),
            g_cal=float(sol_base.g),
            eq={},
        )
        for eq_name, es in EQUILIBRIA.items():
            delta_eq = load_delta_patent(eq_folder(ck, es))
            p_eq = p_cal.copy()
            p_eq.delta[..., 1] = delta_eq
            sol_eq0, sol_eq = static_solve(p_eq)     # new BGP (growth + dyn warm-start)
            p_eq.guess = sol_eq0.x
            _, dyn_sol = dyn_fixed_point_solver(p_eq, sol_init=sol_base, sol_fin=sol_eq, **DYN_KWARGS)
            dyn_sol.compute_non_solver_quantities(p_eq)
            rec['eq'][eq_name] = dict(
                delta=delta_eq.copy(),
                welfare=np.asarray(dyn_sol.cons_eq_welfare).copy(),
                world_negishi=float(dyn_sol.cons_eq_negishi_welfare_change),
                world_equal=float(dyn_sol.cons_eq_pop_average_welfare_change),
                g=float(sol_eq.g),
            )
            print(f"  {eq_name:16s} world(Negishi)={rec['eq'][eq_name]['world_negishi']:.4f} "
                  f"world(Equal)={rec['eq'][eq_name]['world_equal']:.4f}  g={rec['eq'][eq_name]['g']:.4f}")
        out[ck] = rec
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Table rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_table(pdf, df, title, caption=CAPTION, fmt='{:.3f}', color_extremes=False):
    """color_extremes: if True, highlight only the corner delta values -> cells at
    the lower bound (0.01) green, cells at the upper bound (12) red; all other
    cells left plain."""
    fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis('off')
    col_labels = (['\n'.join(map(str, t)) for t in df.columns]
                  if isinstance(df.columns, pd.MultiIndex) else [str(c) for c in df.columns])
    row_labels = ([' / '.join(map(str, t)) for t in df.index]
                  if isinstance(df.index, pd.MultiIndex) else [str(i) for i in df.index])
    cell = [[('' if pd.isna(v) else fmt.format(v)) for v in row] for row in df.values]
    ax.set_title(title, fontsize=12, pad=26)
    t = ax.table(cellText=cell, rowLabels=row_labels, colLabels=col_labels,
                 loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.45)
    if color_extremes:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i][j]
                if pd.isna(v):
                    continue
                if v <= DELTA_LB + 1e-3:        # at lower bound -> strong protection
                    t[(i + 1, j)].set_facecolor(GREEN)
                elif v >= DELTA_UB - 1e-1:       # at upper bound -> no protection
                    t[(i + 1, j)].set_facecolor(RED)
    if caption:
        fig.text(0.5, 0.05, caption, ha='center', fontsize=7.5, wrap=True)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def build_pdf(data):
    countries = data['baseline']['countries']
    cases = list(CASES)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        # definitions / title page
        fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis('off')
        ax.set_title('Patent-flow counterfactuals: baseline 2000 vs 16.0 vs 16.1', fontsize=15, pad=20)
        txt = "Scenario definitions:\n\n" + "\n\n".join(f"  {k}:  {v}" for k, v in CASE_DEF.items())
        txt += ("\n\n\nWelfare gain = consumption-equivalent change (with transition dynamics) from each "
                "case's\nOWN calibrated economy to the equilibrium; reported in %.  Growth in %.\n"
                "delta: lower = stronger patent protection.  Equilibria solved with dynamics.")
        ax.text(0.04, 0.86, txt, va='top', ha='left', fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        # calibrated delta and eta
        render_table(pdf, pd.DataFrame({ck: data[ck]['delta_cal'] for ck in cases}, index=countries),
                     'Calibrated delta (patenting sector)   [lower = stronger protection]')
        render_table(pdf, pd.DataFrame({ck: data[ck]['eta_cal'] for ck in cases}, index=countries),
                     'Calibrated eta (patenting sector)')

        # equilibrium deltas: countries x (equilibrium, case)
        cols = pd.MultiIndex.from_tuples([(eq, ck) for eq in EQUILIBRIA for ck in cases])
        deq = pd.DataFrame({(eq, ck): data[ck]['eq'][eq]['delta'] for eq in EQUILIBRIA for ck in cases},
                           index=countries)[cols]
        render_table(pdf, deq, 'Equilibrium delta (patenting sector):  equilibrium x case'
                     '   [green = 0.01 (full protection), red = 12 (no protection)]',
                     color_extremes=True)

        # welfare gains by country (%)
        w = pd.DataFrame({(eq, ck): (data[ck]['eq'][eq]['welfare'] - 1) * 100
                          for eq in EQUILIBRIA for ck in cases}, index=countries)[cols]
        render_table(pdf, w, 'Welfare gain by country (% consumption-equivalent):  equilibrium x case', fmt='{:.2f}')

        # world welfare, two measures
        rows = {}
        for ck in cases:
            for eq in EQUILIBRIA:
                ee = data[ck]['eq'][eq]
                rows[(ck, eq)] = {'World Negishi (%)': (ee['world_negishi'] - 1) * 100,
                                  'World Equal (%)':   (ee['world_equal'] - 1) * 100}
        wdf = pd.DataFrame(rows).T
        wdf.index = pd.MultiIndex.from_tuples(wdf.index)
        render_table(pdf, wdf, 'World welfare gain (%), two measures:  case x equilibrium', fmt='{:.2f}')

        # growth rates
        grow = {}
        for ck in cases:
            grow[ck] = {'calibrated': data[ck]['g_cal'] * 100}
            for eq in EQUILIBRIA:
                grow[ck][eq] = data[ck]['eq'][eq]['g'] * 100
        gdf = pd.DataFrame(grow).T[['calibrated'] + list(EQUILIBRIA)]
        render_table(pdf, gdf, 'Growth rate (%): calibrated and at each equilibrium   (rows = case)', fmt='{:.3f}')

    print(f"\nwrote {OUT_PDF}")


if __name__ == '__main__':
    check_inputs()
    data = compute()
    build_pdf(data)
