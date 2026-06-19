#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Footnote 34 (Section 4.5) check for the new calibration (baseline 2000):
"if we double iceberg trade costs in the patenting sector, then the US provides
complete patent protection in the Nash equilibrium."

We scale the OFF-DIAGONAL iceberg trade costs tau in the patenting sector
(sector 1) by a factor (diagonal kept at 1, as in the codebase's doubled-trade-
cost convention), recompute the STATIC (no-dynamics) Nash equilibrium in delta,
and report each country's Nash delta as a function of the factor. "Complete
patent protection" = US Nash delta at the floor lb_delta.

Quick first look: static Nash (dynamics=False).
Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from classes import parameters
from solver_funcs import find_nash_eq

BASELINE_PATH = 'calibration_results_matched_economy/2000/'
SECTOR        = 1                      # patenting sector
LB_DELTA      = 0.01                   # floor = complete protection
UB_DELTA      = 12                     # ceiling = no protection
FACTORS       = [2.1, 2.2, 2.3, 2.4]   # refine the dynamic threshold (in 2.0-2.5)
FLOOR_TOL     = 0.05                   # "complete protection" if Nash delta < this
DYNAMICS      = True                   # transition-path Nash (slow) vs static

# parallel=True path in find_nash_eq is broken (worker error); use serial.
NASH_KWARGS = dict(lb_delta=LB_DELTA, ub_delta=UB_DELTA, method='fixed_point',
                   dynamics=DYNAMICS, plot_convergence=False, plot_history=False,
                   tol=1e-4, parallel=False)
OUT_CSV = "nash_trade_cost_deviation_2000_dynamic_refine.csv"


def scaled_tau_params(p_base, factor):
    """Return a copy of p_base with off-diagonal patenting-sector tau scaled."""
    p = p_base.copy()
    tau = p.tau.copy()
    tau[:, :, SECTOR] = p_base.tau[:, :, SECTOR] * factor
    di = np.arange(p.N)
    tau[di, di, SECTOR] = 1.0           # keep diagonal (domestic) at 1
    p.tau = tau
    return p


if __name__ == '__main__':
    p_baseline = parameters()
    p_baseline.load_run(BASELINE_PATH)
    countries = list(p_baseline.countries)
    print(f"baseline {BASELINE_PATH} | countries {countries}")

    # start each Nash search from no protection (delta = ub) and let countries best-respond
    delta_init = np.ones(p_baseline.N * (p_baseline.S - 1)) * UB_DELTA

    rows = []
    for f in FACTORS:
        print(f"\n===== tau off-diagonal x {f} (patenting sector) =====")
        p = scaled_tau_params(p_baseline, f)
        p_nash, _ = find_nash_eq(p, delta_init=delta_init, **NASH_KWARGS)
        nash_delta = np.asarray(p_nash.delta)[:, SECTOR]
        us = float(nash_delta[0])
        rows.append({'tau_factor': f, 'US_nash_delta': us,
                     'US_complete_protection': bool(us < FLOOR_TOL),
                     **{c: float(nash_delta[i]) for i, c in enumerate(countries)}})
        print(f"  US Nash delta = {us:.4f}  -> complete protection: {us < FLOOR_TOL}")
        # save incrementally so partial progress survives if the run is long
        pd.DataFrame(rows).set_index('tau_factor').to_csv(OUT_CSV)

    tab = pd.DataFrame(rows).set_index('tau_factor')
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')

    print("\n=== US Nash delta vs off-diagonal patenting-sector trade-cost factor ===")
    print(tab[['US_nash_delta', 'US_complete_protection']].to_string())
    print("\n=== full Nash delta matrix (rows = tau factor, cols = country) ===")
    print(tab[countries].to_string())

    dev = tab.index[tab['US_complete_protection']]
    if len(dev):
        print(f"\nUS first provides complete patent protection at tau factor = {dev.min()}")
    else:
        print("\nUS does not reach complete protection within the tested factor range.")

    tab.to_csv('nash_trade_cost_deviation_2000.csv')
    print("\nsaved nash_trade_cost_deviation_2000.csv")
