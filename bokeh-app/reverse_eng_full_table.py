#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full long table of ALL moments (every dimension expanded) for baseline 2000 and
the two reverse-engineering perturbations:
  (a) delta developed /2, developing x2
  (b) eta US x2, others x0.7
Columns: target, baseline value, value_(a), %chg_(a) vs baseline, value_(b), %chg_(b).
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd
from classes import parameters, moments
from solver_funcs import fixed_point_solver

DEV   = [0, 1, 2, 6, 7]
DEVING = [3, 4, 5, 8, 9, 10, 11]
US = 0
SK = dict(context='counterfactual', cobweb_anim=False, tol=1e-14, accelerate=False,
          accelerate_when_stable=True, cobweb_qty='phi', plot_convergence=False,
          plot_cobweb=False, safe_convergence=0.001, disp_summary=False, damping=10,
          max_count=3e3, accel_memory=50, accel_type1=True, accel_regularization=1e-10,
          accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
          damping_post_acceleration=5)

p0 = parameters(); p0.correct_eur_patent_cost = True
p0.load_run('calibration_results_matched_economy/2000/')
m = moments(); m.load_run('calibration_results_matched_economy/2000/')
ALL_MOMS = list(m.list_of_moments)   # only the moments targeted in baseline 2000


def solve(p):
    sol, sol_c = fixed_point_solver(p, x0=p.guess, **SK)
    sol_c.scale_P(p); sol_c.compute_non_solver_quantities(p)
    m.compute_moments(sol_c, p)
    return sol_c


def snap(which):
    """return dict mom -> flat np.array of model values (and labels/target on first call)."""
    out = {}
    for mom in ALL_MOMS:
        try:
            v = np.atleast_1d(np.asarray(getattr(m, mom), dtype=float).ravel())
        except Exception:
            continue
        out[mom] = v
    return out


def labels_for(mom, n):
    idx = m.idx.get(mom)
    if idx is not None and len(idx) == n:
        return ['|'.join(map(str, t)) if isinstance(t, tuple) else str(t) for t in idx]
    return [str(i) for i in range(n)] if n > 1 else ['scalar']


sol0 = solve(p0); base = snap('base')
# targets (data), snapshot once
tgt = {}
for mom in ALL_MOMS:
    try:
        tgt[mom] = np.atleast_1d(np.asarray(getattr(m, mom + '_target'), dtype=float).ravel())
    except Exception:
        tgt[mom] = None

pa = p0.copy(); pa.delta[DEV, 1] /= 2; pa.delta[DEVING, 1] *= 2
sola = solve(pa); va = snap('a')

pb = p0.copy(); pb.eta[US, 1] *= 2
others = [i for i in range(p0.N) if i != US]; pb.eta[others, 1] *= 0.7
solb = solve(pb); vb = snap('b')

rows = []
for mom in ALL_MOMS:
    if mom not in base:
        continue
    b = base[mom]
    if not np.isfinite(b).any():          # skip moments not applicable to this model
        continue
    a = va.get(mom); bb = vb.get(mom)
    t = tgt.get(mom)
    n = b.size
    labs = labels_for(mom, n)
    for j in range(n):
        bj = b[j]
        rows.append({
            'moment': mom, 'dimension': labs[j] if j < len(labs) else str(j),
            'target': (t[j] if t is not None and j < t.size else np.nan),
            'baseline': bj,
            'value_a_delta': (a[j] if a is not None and j < a.size else np.nan),
            'pct_chg_a': ((a[j] / bj - 1) * 100 if a is not None and j < a.size and bj != 0 else np.nan),
            'value_b_eta': (bb[j] if bb is not None and j < bb.size else np.nan),
            'pct_chg_b': ((bb[j] / bj - 1) * 100 if bb is not None and j < bb.size and bj != 0 else np.nan),
        })

tab = pd.DataFrame(rows)
out = 'output/reverse_eng_all_moments_full.csv'
tab.to_csv(out, index=False)
print(f"saved {out}  ({len(tab)} rows, {tab.moment.nunique()} moments)")
pd.set_option('display.max_rows', 2000); pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.5g}')
print(tab.to_string(index=False))
