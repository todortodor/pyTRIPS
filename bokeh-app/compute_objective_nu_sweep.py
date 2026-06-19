#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective-function value at each saved nu in the nu / nu_tilde (double-diffusion)
exercise, baseline 2003.

For each variation, nu (patenting sector) is FIXED at a grid value and nu_tilde
(+ other params) was calibrated. We re-solve the double-diff equilibrium and
report the objective = ||m.deviation_vector()||_2, the same quantity the
calibration logs as 'objective' (calibration_func_double_diff_double_delta).

Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from classes import parameters, moments, var_double_diff_double_delta
from solver_funcs import fixed_point_solver_double_diff_double_delta

VARIATIONS = ['1.0', '2.0', '2.01', '2.02', '2.03', '2.04', '2.05', '2.06',
              '2.07', '2.08', '3.0', '4.0']
BASE = 'calibration_results_matched_economy/baseline_2003_variations/'

SOLVER_KWARGS = dict(
    context='calibration', cobweb_anim=False, tol=1e-13,
    accelerate=True, accelerate_when_stable=True, cobweb_qty='l_R',
    plot_convergence=False, plot_cobweb=False, safe_convergence=0.001,
    disp_summary=False, damping=100, max_count=10000,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=2,
)

rows = []
for v in VARIATIONS:
    path = BASE + v + '/'
    try:
        p = parameters()
        p.load_run(path)
        m = moments()
        m.load_run(path)

        sol, sol_c = fixed_point_solver_double_diff_double_delta(p, x0=p.guess, **SOLVER_KWARGS)
        sol_c.scale_P(p)
        sol_c.compute_non_solver_quantities(p)
        m.compute_moments(sol_c, p)
        m.compute_moments_deviations()

        dev = m.deviation_vector()
        obj = float(np.linalg.norm(dev))
        rows.append({
            'variation': v,
            'nu': float(np.asarray(p.nu)[1]),
            'nu_tilde': float(np.asarray(p.nu_tilde)[1]),
            'objective_norm': obj,
            'objective_sq': obj ** 2,
            'status': sol.status,
            'n_moments': len(m.list_of_moments),
            'moments': ','.join(sorted(m.list_of_moments)),
        })
        print(f"{v:5s}  nu={float(np.asarray(p.nu)[1]):.2e}  obj={obj:.6f}  ({sol.status})")
    except Exception as e:
        print(f"{v:5s}  FAILED: {e}")
        rows.append({'variation': v, 'objective_norm': np.nan, 'status': f'error: {e}'})

tab = pd.DataFrame(rows).set_index('variation')

# flag whether the moment set is identical across runs (needed for comparability)
msets = tab['moments'].dropna().unique()
print(f"\ndistinct moment sets across variations: {len(msets)} "
      f"({'comparable' if len(msets) == 1 else 'NOT identical - see moments column'})")

pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')
print("\n=== objective by nu (double-diffusion, baseline 2003) ===")
print(tab[['nu', 'nu_tilde', 'objective_norm', 'objective_sq', 'n_moments', 'status']]
      .sort_values('nu').to_string())

tab.to_csv('objective_nu_sweep_2003.csv')
print("\nsaved objective_nu_sweep_2003.csv")
