#!/usr/bin/env python3
"""
FDI calibration runner — Romer-EK model with FDI extension.

Loads parameter seeds AND moment targets from a single SOURCE folder
(no separate "params from X / moments from Y" split). Writes the
calibrated result (best-so-far during the run, final at the end) to
a single DEST folder.

To warm-start from a previous result, just point SOURCE at that folder.
"""
import sys, os, time, warnings, traceback
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
from scipy import optimize

from classes import moments, parameters, history
from solver_funcs import calibration_func_with_fdi, fixed_point_solver_with_fdi


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these paths and run controls in place
# ─────────────────────────────────────────────────────────────────────────────

# Folder holding BOTH the parameter seeds (eta.csv, k.csv, ...) AND the
# moment targets (GROWTH.csv, RD.csv, ..., list_of_moments.csv).
# For a cold start, this is the bundled cold_start_fdi/ folder.
# To warm-start, point this at a previous result folder (e.g. fdi_v9/).
SOURCE = 'calibration_results_matched_economy/baseline_2000_variations/cold_start_fdi/'

# Where to write the calibrated result (and best-so-far checkpoints).
DEST   = 'calibration_results_matched_economy/baseline_2000_variations/fdi_calibrated/'

# Number of outer least_squares iterations.
MAX_ITER = 4

# Max function evaluations per outer iteration. With ~60 fevals per Jacobian,
# 200 covers ~3 Jacobians per outer iter and bounds wall time to ~5-10 min
# per outer iter (depending on per-call solve time).
MAX_NFEV_PER_OUTER = 200

# Initial values for the FDI-specific parameters (used only for cold start).
# For warm-start, the values loaded from SOURCE take precedence.
INITIAL_A         = 0.1
INITIAL_D_FRAC    = 0.6      # d = d_frac * (k[1] - 1 - 1e-6); 0.6 ≈ d~0.15
INITIAL_POWER_FDI = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FDI CALIBRATION — Romer-EK with FDI extension")
print("=" * 70)
print(f"SOURCE: {SOURCE}")
print(f"DEST:   {DEST}")

# Detect whether this is a cold start (SOURCE has no FDI-tuned values) or
# a warm start (SOURCE was previously written by this script).
is_warm_start = os.path.exists(os.path.join(SOURCE, 'd_frac.csv'))
print(f"warm start: {is_warm_start}")

p = parameters()
p.correct_eur_patent_cost = True
p.load_run(SOURCE)

m = moments()
m.load_run(SOURCE)
m.drop_CHN_IND_BRA_ROW_from_RD = True

# Append FDI moments if not already there (idempotent for warm-start)
for _name in ('FDI_FLOW_N', 'FDI_ELAST'):
    if _name not in m.list_of_moments:
        m.list_of_moments.append(_name)

# Append FDI-specific calibrated parameters (idempotent for warm-start).
# We use 'd_frac' rather than 'd' to enforce k > d + 1 via reparameterization:
#   d = d_frac * (k[1] - 1 - 1e-6),  with d_frac in (1e-6, 1 - 1e-6).
# See parameters._sync_d_from_dfrac in classes.py.
for _name in ('a', 'd_frac', 'power_fdi'):
    if _name not in p.calib_parameters:
        p.calib_parameters.append(_name)
# Remove 'd' if it's there (replaced by d_frac).
if 'd' in p.calib_parameters:
    p.calib_parameters.remove('d')

# For cold start, set the initial values explicitly.
# For warm start, the loaded p.d_frac (or derived from p.d via load_run's
# backward-compat sync) is the right starting point — don't overwrite.
if not is_warm_start:
    p.a = INITIAL_A
    p.d_frac = np.float64(INITIAL_D_FRAC)
    p._sync_d_from_dfrac()    # update p.d to match
    p.power_fdi = INITIAL_POWER_FDI

p.guess = None

start_time = time.perf_counter()

print(f"N={p.N}, S={p.S}")
print(f"calib_parameters ({len(p.calib_parameters)}): {p.calib_parameters}")
print(f"list_of_moments  ({len(m.list_of_moments)}): {m.list_of_moments}")
print(f"initial p.k[1]={p.k[1]:.4f}, p.d_frac={float(p.d_frac):.4f}, "
      f"p.d={float(p.d):.4f}")

hist = history(*tuple(m.list_of_moments + ['objective']))
bounds = p.make_parameters_bounds()
print(f"parameter vector length: {len(p.make_p_vector())}")


# ─────────────────────────────────────────────────────────────────────────────
# Outer least_squares loop with per-iteration checkpointing
# ─────────────────────────────────────────────────────────────────────────────
# The destination folder is also used as CALIB_CKPT_PATH so the inner
# calibration_func_with_fdi can write best-so-far snapshots after every
# function eval. This means the process can be killed at any moment and
# the latest best result is on disk.
os.makedirs(DEST, exist_ok=True)
os.environ['CALIB_CKPT_PATH'] = DEST
print(f"CALIB_CKPT_PATH set to: {DEST}")
print(f"\nRunning least_squares loop, MAX_ITER={MAX_ITER}, "
      f"MAX_NFEV_PER_OUTER={MAX_NFEV_PER_OUTER}")

test_ls = None
for iteration in range(MAX_ITER):
    xtol = 1e-10 if iteration == MAX_ITER - 1 else 1e-8
    print(f"\n--- Outer iteration {iteration+1}/{MAX_ITER}, xtol={xtol} ---")
    t_start = time.perf_counter()
    try:
        test_ls = optimize.least_squares(
            fun=calibration_func_with_fdi,
            x0=p.make_p_vector(),
            args=(p, m, p.guess, hist, start_time),
            bounds=bounds,
            max_nfev=MAX_NFEV_PER_OUTER,
            xtol=xtol,
            verbose=2,
        )
    except Exception as e:
        print(f"\n!!! Outer iter {iteration+1} CRASHED: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        break

    t_elapsed = time.perf_counter() - t_start
    print(f"Outer iteration {iteration+1} done in {t_elapsed:.1f}s")
    print(f"  status: {test_ls.status}, message: {test_ls.message}")
    print(f"  cost: {test_ls.cost:.6e}")
    print(f"  nfev: {test_ls.nfev}, njev: {test_ls.njev}")
    p.update_parameters(test_ls.x)

finish_time = time.perf_counter()
print(f"\nTotal minimizing time: {finish_time - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Final solve and write results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Final solve and result writing")
print("=" * 70)

if test_ls is not None:
    p.update_parameters(test_ls.x)

print("\nFinal solve with calibration context...")
sol, sol_c = fixed_point_solver_with_fdi(
    p, context='counterfactual',
    x0=p.guess,
    cobweb_anim=False, tol=1e-10, accelerate=False,
    accelerate_when_stable=True,
    cobweb_qty='phi',
    plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=True,
    damping=10, max_count=2000,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)
p.guess = sol.x
sol_c.scale_P(p)
sol_c.compute_non_solver_quantities(p)
p.tau = sol_c.tau
m.compute_moments(sol_c, p)
m.compute_moments_deviations()

print(f"\nWriting final results to: {DEST}")
p.write_params(DEST)
m.write_moments(DEST)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 70)

print(f"\nFinal solver status:        {sol.status}")
print(f"Final solver iterations:    {sol.iter if hasattr(sol, 'iter') else 'n/a'}")
print(f"Final solver wallclock:     {sol.time:.2f}s")
print(f"Hit-the-bound count:        {sol.hit_the_bound_count}")

dev_vec = m.deviation_vector()
print(f"\nFinal ||deviation_vector||: {np.linalg.norm(dev_vec):.6f}")
print(f"Final number of moments:    {len(dev_vec)}")

print(f"\nPer-moment deviation norms:")
for mom in m.list_of_moments:
    dev = getattr(m, mom + '_deviation', None)
    if dev is None:
        continue
    norm = np.linalg.norm(dev)
    print(f"  {mom:15s}: ||dev|| = {norm:.4e}")

print(f"\nFinal FDI-specific parameter values:")
print(f"  a        = {float(p.a):.6f}")
print(f"  d        = {float(p.d):.6f}   (= d_frac * (k[1] - 1 - 1e-6))")
print(f"  d_frac   = {float(p.d_frac):.6f}")
print(f"  k[1]     = {float(p.k[1]):.6f}")
print(f"  k - d - 1 = {float(p.k[1] - p.d - 1):.6e}  (> 0 required)")
print(f"  power_fdi= {float(p.power_fdi):.6f}")

print(f"\nDONE. Output at: {DEST}")
