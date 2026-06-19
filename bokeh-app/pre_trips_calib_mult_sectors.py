#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-TRIPS calibration of the MULTI-SECTOR model (baseline 2002), with a
SEPARATE patent-protection parameter delta in the pharma-chemicals sector vs.
the other patenting sector.

Motivation (referee R1): the headline multi-sector run currently used (8.8)
forces delta to be equal across the two patenting sectors, so it cannot speak
to the cross-sector heterogeneity in patent enforcement that TRIPS targeted.
This script redoes the pre-TRIPS exercise of pre_trips_calib.py but:
  * seeds from the SEPARATE-delta 2015 run 8.0 (structural params held fixed),
  * loads 1992 (pre-TRIPS) 3-sector data,
  * re-calibrates ONLY ['delta','T','eta'] to 1992 moments.

With the default mask, delta is free in BOTH patenting sectors (sectors 1 and 2),
so the pharma-chem delta is estimated separately. SPFLOW is sector-resolved
(N,N-1,S-1) for S>2, which is what identifies a sector-specific delta; we
additionally include the sector-resolved patent-cost (UUPCOSTS) and pharma R&D
(RDPHARMACHEM) moments.

The win condition: pre-TRIPS pharma-chem delta is much higher (weaker protection)
than the other patenting sector in countries like India, and the 1992->2015 gap
narrows -- evidence the model's sector-specific delta captures the TRIPS shock.

Run from inside bokeh-app/.
"""
# Force a non-interactive backend BEFORE importing classes/solver_funcs (which
# import matplotlib.pyplot). The solver has unconditional plt.show() calls that
# otherwise open a window and block for input in a headless/background run.
import matplotlib
matplotlib.use('Agg')

from scipy import optimize
import time
import os
import shutil
import warnings
import numpy as np
import pandas as pd

from classes import moments, parameters, var, history
from solver_funcs import calibration_func, fixed_point_solver

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASELINE        = '2002'
# --- which 2015 multi-sector run to seed from, and whether to impose equal
# delta across the two patenting sectors (the "8.8" version) ---
# 8.0 -> 8.1 : separate delta across patenting sectors
# 8.8 -> 8.81: delta EQUAL across patenting sectors (fix_delta_across_sectors)
SEED_VARIATION  = '8.8'
OUT_VARIATION   = '8.81'
FIX_DELTA_ACROSS_SECTORS = True
YEAR            = 1992     # pre-TRIPS
# Use the SMOOTHED 3-year data (built by smooth_out_data_mult_sectors.py), the
# apples-to-apples counterpart of the single-sector pre-TRIPS run. Raw single-
# year 1992 data inflated the delta levels across the board.
RAW_DATA_DIR    = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}/'
ANBERD_FILE     = 'data/anberd.csv'

# The 1992 country_sector_moments (used only by the RDPHARMACHEM moment, which
# hardcodes countries [1,2,3,7,8] = USA,EUR,JAP,CAN,KOR) is incomplete:
#   - KOR (row 8) is missing entirely
#   - CAN (row 7) is present but blank (-> NaN, which would poison the residual)
# The sector R&D breakdown in OECD ANBERD simply does not go back to 1992 for
# these two (CAN sector data starts 1994, KOR data starts 1995). We fill BOTH
# from the EARLIEST AVAILABLE year (the closest proxy to pre-TRIPS) computed
# directly from anberd.csv, and write a patched copy of the data folder so the
# smoothed 1992 data is left untouched. DATA_DIR points at the patched copy.
DATA_DIR        = f'data_smooth_3_years/data_12_countries_3_sectors_{YEAR}_RDPHARMACHEM_fix/'

# Internal country code -> ANBERD REF_AREA, for the rows we need to backfill.
FILL_REF_AREA   = {7: 'CAN', 8: 'KOR'}


def earliest_rd_ratios(ref_area):
    """Return (RD ratio pharma, RD ratio chemicals, year) for the earliest year
    in ANBERD where C21 (pharma), C20 (chemicals) and _T (total) all exist,
    using the same filter as make_data_folders_mult_sectors.py."""
    d = pd.read_csv(ANBERD_FILE)
    d = d[(d.CRITERIA == 'MA') & (d.UNIT_MEASURE == 'USD_PPP') &
          (d.PRICE_BASE == 'V') & (d.REF_AREA == ref_area)]
    g = d.groupby(['TIME_PERIOD', 'ACTIVITY'])['OBS_VALUE'].sum().unstack('ACTIVITY')
    g = g.dropna(subset=['C20', 'C21', '_T']).sort_index()
    yr = g.index[0]
    return g.loc[yr, 'C21'] / g.loc[yr, '_T'], g.loc[yr, 'C20'] / g.loc[yr, '_T'], int(yr)


def build_patched_data_dir():
    """Copy the 1992 data folder and fill missing/blank RDPHARMACHEM country
    rows (CAN, KOR) with their earliest-available-year ANBERD values."""
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    shutil.copytree(RAW_DATA_DIR, DATA_DIR)

    needed = [1, 2, 3, 7, 8]            # USA, EUR, JAP, CAN, KOR
    cols = ['RD ratio pharma', 'RD ratio chemicals']
    base = pd.read_csv(RAW_DATA_DIR + 'country_sector_moments.csv', index_col=0)
    base.index = base.index.astype(int)

    base = base.reindex(base.index.union(needed))
    filled = []
    for r in needed:
        if any(pd.isna(base.loc[r, c]) for c in cols):
            if r not in FILL_REF_AREA:
                raise ValueError(f"row {r} needs backfill but has no ANBERD mapping")
            pharma, chem, yr = earliest_rd_ratios(FILL_REF_AREA[r])
            fill_vals = {'RD ratio pharma': pharma, 'RD ratio chemicals': chem}
            for c in cols:
                if pd.isna(base.loc[r, c]):
                    base.loc[r, c] = fill_vals[c]
                    filled.append((r, FILL_REF_AREA[r], c, round(fill_vals[c], 4), yr))
    base = base.sort_index()
    base.to_csv(DATA_DIR + 'country_sector_moments.csv')
    print("patched RDPHARMACHEM rows (filled from earliest-available ANBERD year):")
    for r, ref, c, val, yr in filled:
        print(f"  row {r} ({ref})  {c} = {val}  [from {yr}]")


build_patched_data_dir()

# Moments used for the pre-TRIPS recalibration. Mirrors the 2-sector pre-TRIPS
# list (pre_trips_calib.py) but with sector-resolved augmentation:
#   SPFLOW  -> automatically (N,N-1,S-1) for S>2  [identifies sector-specific delta]
#   UUPCOSTS-> sector-resolved patent costs (replaces aggregate UUPCOST)
#   RDPHARMACHEM -> pharma/chem R&D for advanced countries
LIST_OF_MOMENTS = ['SPFLOW', 'DOMPATINUS', 'OUT', 'RD', 'RP', 'SRGDP',
                   'UUPCOSTS', 'RDPHARMACHEM']

# Weights aligned to the single-sector pre-TRIPS run (9.2): SPFLOW/RP/SRGDP = 1
# rather than 8.0's heavy 10/10/5, which (together with raw data) was pushing the
# delta levels up. RDPHARMACHEM kept at 1 (sector identification, light touch).
WEIGHTS = {'SPFLOW': 1, 'DOMPATINUS': 1, 'OUT': 5, 'RD': 10, 'RP': 1,
           'SRGDP': 1, 'UUPCOSTS': 1, 'RDPHARMACHEM': 1}

MAX_ITER        = 5        # outer least_squares iterations (as in pre_trips_calib.py)
DROP_CHN_IND_BRA_ROW_FROM_RD = True   # matches the 8.0 baseline calibration

# If True: load, do a single equilibrium solve, print shapes/targets and STOP.
# Use this to validate the setup cheaply before the full calibration.
DRY_RUN = False

SEED_PATH = f'calibration_results_matched_economy/baseline_{BASELINE}_variations/{SEED_VARIATION}/'
OUT_PATH  = f'calibration_results_matched_economy/baseline_{BASELINE}_variations/{OUT_VARIATION}/'

SOLVER_KWARGS = dict(
    cobweb_anim=False, tol=1e-13,
    accelerate=False, accelerate_when_stable=True,
    cobweb_qty='phi', plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=True,
    damping=10, max_count=3e3,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load seed params/moments, then swap in the 1992 data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"PRE-TRIPS multi-sector calibration  (seed {SEED_VARIATION} -> {OUT_VARIATION})")
print(f"data: {DATA_DIR}")
print("=" * 70)

p = parameters()
p.correct_eur_patent_cost = True
# set BEFORE load_run so the parameter mask is built with delta tied across
# patenting sectors (calibrate only sector 1; sector >=2 synced in calibration_func)
p.fix_delta_across_sectors = FIX_DELTA_ACROSS_SECTORS
p.load_run(SEED_PATH)
p.load_data(DATA_DIR, keep_already_calib_params=True, nbr_sectors=3)
p.calib_parameters = ['delta', 'T', 'eta']
if FIX_DELTA_ACROSS_SECTORS:                 # ensure equal at the seed too
    for s in range(2, p.S):
        p.delta[:, s] = p.delta[:, 1]

m = moments()
m.load_run(SEED_PATH)
m.load_data(DATA_DIR)
m.aggregate_moments = True
m.list_of_moments = LIST_OF_MOMENTS
m.weights_dict.update(WEIGHTS)
m.drop_CHN_IND_BRA_ROW_from_RD = DROP_CHN_IND_BRA_ROW_FROM_RD

print(f"\nN={p.N}  S={p.S}  sectors={p.sectors}")
print(f"calib_parameters = {p.calib_parameters}")
print(f"delta shape = {np.asarray(p.delta).shape}  (free entries per sector via mask)")
print(f"delta mask sums by column = {p.mask['delta'].reshape(p.N, p.S).sum(axis=0)}")

# ─────────────────────────────────────────────────────────────────────────────
# Initial solve at the seeded params on 1992 data (warm-starts the guess)
# ─────────────────────────────────────────────────────────────────────────────
sol, sol_init = fixed_point_solver(p, x0=p.guess, context='calibration', **SOLVER_KWARGS)
sol_init.scale_P(p)
sol_init.compute_non_solver_quantities(p)
p.guess = sol.x
m.compute_moments(sol_init, p)
m.compute_moments_deviations()

print("\n--- moment target shapes (1992) ---")
for mom in LIST_OF_MOMENTS:
    tgt = getattr(m, mom + '_target')
    print(f"  {mom:14s} target shape {np.asarray(tgt).shape}")
print(f"SPFLOW_target.ndim = {np.asarray(m.SPFLOW_target).ndim} "
      f"(expect 3 for sector-resolved)")

if DRY_RUN:
    print("\nDRY_RUN=True -> stopping before the full calibration. "
          "Set DRY_RUN=False to run it.")
    raise SystemExit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Calibration: recalibrate delta (separate by sector), T, eta to 1992 moments
# ─────────────────────────────────────────────────────────────────────────────
hist = history(*tuple(m.list_of_moments + ['objective']))
bounds = p.make_parameters_bounds()
start_time = time.perf_counter()
iterations = 0
cond = True
while cond:
    xtol = 1e-10 if iterations < MAX_ITER - 2 else 1e-16
    test_ls = optimize.least_squares(
        fun=calibration_func,
        x0=p.make_p_vector(),
        args=(p, m, p.guess, hist, start_time),
        bounds=bounds, max_nfev=1e8, xtol=xtol, verbose=2)
    cond = iterations < MAX_ITER
    iterations += 1
    p.update_parameters(test_ls.x)
print('minimizing time', time.perf_counter() - start_time)

p_sol = p.copy()
p_sol.update_parameters(test_ls.x)

sol, sol_c = fixed_point_solver(p_sol, x0=p_sol.guess, context='calibration', **SOLVER_KWARGS)
p_sol.guess = sol.x
sol_c.scale_P(p_sol)
sol_c.compute_non_solver_quantities(p_sol)
p_sol.tau = sol_c.tau
m.compute_moments(sol_c, p_sol)
m.compute_moments_deviations()

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_PATH, exist_ok=True)
p_sol.write_params(OUT_PATH)
m.write_moments(OUT_PATH)
print(f"\nwrote pre-TRIPS result to {OUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Deliverable: pre-TRIPS (1992) vs 2015 delta. With equal delta across patenting
# sectors the two sectors coincide, so we report the single patenting-sector
# delta and the post-TRIPS change (1992 -> 2015).
# ─────────────────────────────────────────────────────────────────────────────
p_2015 = parameters()
p_2015.load_run(SEED_PATH)
d92 = np.asarray(p_sol.delta).reshape(p_sol.N, p_sol.S)
d15 = np.asarray(p_2015.delta).reshape(p_2015.N, p_2015.S)
rows = []
for i, c in enumerate(p_sol.countries):
    row = {
        'country': c,
        'delta_1992': d92[i, 1], 'delta_2015': d15[i, 1],
        'change_1992_to_2015': d92[i, 1] - d15[i, 1],   # >0 means protection rose post-TRIPS
        'ratio_1992_over_2015': d92[i, 1] / d15[i, 1],
    }
    if not FIX_DELTA_ACROSS_SECTORS:                      # keep pharma gap if separate
        row.update({'d_pharma_1992': d92[i, 2], 'gap_1992': d92[i, 2] / d92[i, 1],
                    'd_pharma_2015': d15[i, 2], 'gap_2015': d15[i, 2] / d15[i, 1]})
    rows.append(row)
tab = pd.DataFrame(rows).set_index('country')
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print("\n=== patenting-sector delta: pre-TRIPS (1992) vs 2015"
      f" (seed {SEED_VARIATION}) ===")
print("(higher delta = WEAKER protection; ratio>1 means weaker pre-TRIPS, i.e. TRIPS raised protection)")
print(tab)
out_name = ('delta_pre_vs_post_trips.csv' if FIX_DELTA_ACROSS_SECTORS
            else 'delta_pharma_gap_1992_vs_2015.csv')
tab.to_csv(OUT_PATH + out_name)
print(f"\nsaved comparison table to {OUT_PATH}{out_name}")
