#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse-engineering sensitivity on baseline 2000 (single-sector, no FDI).
Perturb parameters, re-solve (static), recompute all moments, and report which
moments move the most (vs the calibrated baseline) — with a focus on patent
flows, DOMPATINUS, Kogan (KM), share patented (SINNOVPATUS), RD, growth.

Perturbations:
  (a) delta: developed economies /2 (stronger protection), developing x2 (weaker)
  (b) eta:   US x2, all other countries x0.7
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd
from classes import parameters, moments, var
from solver_funcs import fixed_point_solver

DEV   = [0, 1, 2, 6, 7]                 # USA, EUR, JAP, CAN, KOR
DEVING = [3, 4, 5, 8, 9, 10, 11]        # CHN, BRA, IND, RUS, MEX, ZAF, ROW
US = 0
MOMS = ['GPDIFF', 'GROWTH', 'KM', 'OUT', 'RD', 'RP', 'SRGDP', 'SINNOVPATUS',
        'TO', 'SPFLOW', 'UUPCOST', 'DOMPATINUS', 'TE']
SK = dict(context='counterfactual', cobweb_anim=False, tol=1e-14, accelerate=False,
          accelerate_when_stable=True, cobweb_qty='phi', plot_convergence=False,
          plot_cobweb=False, safe_convergence=0.001, disp_summary=False, damping=10,
          max_count=3e3, accel_memory=50, accel_type1=True, accel_regularization=1e-10,
          accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
          damping_post_acceleration=5)


def solve_and_moments(p, m):
    sol, sol_c = fixed_point_solver(p, x0=p.guess, **SK)
    sol_c.scale_P(p); sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c, p)
    snap = {mm: np.asarray(getattr(m, mm), dtype=float).copy() for mm in MOMS}
    pf = np.asarray(sol_c.pflow, dtype=float)
    pf = pf[..., 0] if pf.ndim == 3 else pf       # [dest, origin] patenting flows
    return sol_c, snap, pf


def pflow_breakdown(pf):
    tot = pf.sum()
    offdiag = pf - np.diag(np.diag(pf))
    return {
        'total_patents': tot,
        'dest_developed_share': pf[DEV, :].sum() / tot,
        'dest_developing_share': pf[DEVING, :].sum() / tot,
        'orig_developed_share': pf[:, DEV].sum() / tot,
        'orig_developing_share': pf[:, DEVING].sum() / tot,
        'international_share': offdiag.sum() / tot,
        'US_origin_share': pf[:, US].sum() / tot,
    }


def rel_change(base, pert):
    """% change: scalar -> signed %; array -> relative L2 norm %."""
    base = np.atleast_1d(base); pert = np.atleast_1d(pert)
    if base.size == 1:
        b = base.item()
        return (pert.item() / b - 1) * 100 if b != 0 else np.nan
    denom = np.linalg.norm(base)
    return np.linalg.norm(pert - base) / denom * 100 if denom != 0 else np.nan


# ── baseline ──
p0 = parameters(); p0.correct_eur_patent_cost = True
p0.load_run('calibration_results_matched_economy/2000/')
m = moments(); m.load_run('calibration_results_matched_economy/2000/')
co = list(p0.countries)
sol0, base, pf0 = solve_and_moments(p0, m)
print('baseline growth g =', round(float(sol0.g), 5))

# ── (a) delta perturbation ──
pa = p0.copy()
pa.delta[DEV, 1] /= 2
pa.delta[DEVING, 1] *= 2
sola, snap_a, pfa = solve_and_moments(pa, m)

# ── (b) eta perturbation ──
pb = p0.copy()
pb.eta[US, 1] *= 2
others = [i for i in range(p0.N) if i != US]
pb.eta[others, 1] *= 0.7
solb, snap_b, pfb = solve_and_moments(pb, m)

# ── moment-change table ──
rows = []
for mm in MOMS:
    rows.append({'moment': mm,
                 '%chg_(a)_delta': rel_change(base[mm], snap_a[mm]),
                 '%chg_(b)_eta':  rel_change(base[mm], snap_b[mm])})
mt = pd.DataFrame(rows).set_index('moment')
mt['size'] = [('scalar' if np.asarray(base[mm]).size == 1 else 'array(relL2%)') for mm in MOMS]
pd.set_option('display.width', 160); pd.set_option('display.float_format', lambda x: f'{x:.1f}')
print('\n=== moment % change vs baseline (scalars signed; arrays = relative L2 %) ===')
print(mt.sort_values('%chg_(a)_delta', key=lambda s: s.abs(), ascending=False).to_string())

# ── patent-flow breakdown ──
bd = pd.DataFrame({'baseline': pflow_breakdown(pf0), '(a) delta': pflow_breakdown(pfa),
                   '(b) eta': pflow_breakdown(pfb)})
bd['%chg_(a)'] = (bd['(a) delta'] / bd['baseline'] - 1) * 100
bd['%chg_(b)'] = (bd['(b) eta'] / bd['baseline'] - 1) * 100
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print('\n=== patent-flow breakdown (shares of total patenting) ===')
print(bd.to_string())

# ── growth ──
print('\n=== growth rate g ===')
print(f"baseline {float(sol0.g):.4f} | (a) delta {float(sola.g):.4f} "
      f"({(sola.g/sol0.g-1)*100:+.1f}%) | (b) eta {float(solb.g):.4f} ({(solb.g/sol0.g-1)*100:+.1f}%)")

mt.to_csv('output/reverse_eng_moment_changes.csv')
bd.to_csv('output/reverse_eng_pflow_breakdown.csv')
print('\nsaved output/reverse_eng_moment_changes.csv and output/reverse_eng_pflow_breakdown.csv')
