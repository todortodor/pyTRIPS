#!/usr/bin/env python3
"""
FDI calibration runner — BILATERAL a variant.

Same as run_calibration_FDI.py, with two key changes:
  1. p.a is promoted to a bilateral (N, N, S) array via
     parameters.enable_bilateral_a() before calibration. Each off-diagonal
     sector-1 entry is an independent free parameter, allowing the FDI
     setup cost to vary by origin-destination pair.
  2. The new bilateral moment FDI_FLOW (X^{M,F}_{ni} / net_absorption_n)
     replaces the destination-aggregated FDI_FLOW_N in list_of_moments.
     FDI_FLOW_N stays computed (for diagnostics) but is NOT in the
     calibration target list.

Writes to a separate DEST so it can run in parallel with the scalar-a
calibration without collision.

To compare the two calibration runs (scalar-a vs bilateral-a), point a
diagnostic notebook at both DEST folders. Backwards compatibility is
preserved on both sides:
  - p.load_run() on this DEST auto-detects bilateral a from CSV shape
    and promotes the loaded parameters via enable_bilateral_a().
  - p.load_run() on the scalar-a DEST leaves p.a scalar as before.
  - var_with_fdi.compute_entry_costs uses p.a * supply_potential via
    numpy broadcasting — scalar and array both work without code change.
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

# Folder holding parameter seeds and moment targets.
SOURCE = 'calibration_results_matched_economy/baseline_2000_variations/fdi_calibrated_bilateral_a_v3/'

# Where to write the calibrated result.  DIFFERENT from the scalar-a runner
# so both runs can coexist on disk.
DEST   = 'calibration_results_matched_economy/baseline_2000_variations/fdi_calibrated_bilateral_a_v4/'

# Number of outer least_squares iterations.
MAX_ITER = 4

# Max function evaluations per outer iteration.  Bilateral a adds N*(N-1) =
# 132 parameters (for N=12), so a Jacobian costs ~190 fevals.  Bump nfev
# budget proportionally to allow at least one full Jacobian + line search.
MAX_NFEV_PER_OUTER = 400

# Initial values for the FDI-specific parameters (used only for cold start).
INITIAL_A         = 0.1      # bilateral a starts uniform at this value
INITIAL_D_FRAC    = 0.6      # d = d_frac * (k[1] - 1 - margin); 0.6 ≈ d~0.15
INITIAL_POWER_FDI = 1.0

# Threshold for masking "data-zero" bilateral FDI flow cells.
# Cells with raw data FDI flow ratio below this value are excluded from
# the FDI_FLOW calibration residual, and the corresponding bilateral-a
# entries are held fixed (not calibrated). With threshold=1e-6 only the
# 12 exact-zero cells in the AAMNE data are masked.
# Set to 0 to disable masking (calibrate all 132 cells / parameters).
FDI_FLOW_ZERO_THRESHOLD = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FDI CALIBRATION — BILATERAL a variant (Romer-EK with FDI)")
print("=" * 70)
print(f"SOURCE: {SOURCE}")
print(f"DEST:   {DEST}")

# Detect whether DEST already has a bilateral checkpoint (warm-start).
# If a.csv on disk has N*N*S entries, load_run will auto-promote.
is_warm_start_bilateral = False
_dest_a_csv = os.path.join(DEST, 'a.csv')
if os.path.exists(_dest_a_csv):
    try:
        import pandas as _pd
        _nrows = len(_pd.read_csv(_dest_a_csv, header=None))
        # Assume N=12, S=2 for the warm-start check (will be re-verified
        # by load_run with the actual N, S from the data_path csv).
        if _nrows > 1:
            is_warm_start_bilateral = True
    except Exception:
        pass

# Also accept warm-start from a scalar-a checkpoint in SOURCE.  We will
# promote to bilateral after loading.
is_warm_start_scalar = (not is_warm_start_bilateral and
                        os.path.exists(os.path.join(SOURCE, 'd_frac.csv')))
print(f"warm start (bilateral, from DEST): {is_warm_start_bilateral}")
print(f"warm start (scalar,    from SOURCE): {is_warm_start_scalar}")

# Load parameters: from DEST if a bilateral checkpoint exists, else SOURCE.
load_from = DEST if is_warm_start_bilateral else SOURCE

p = parameters()
p.correct_eur_patent_cost = True
p.load_run(load_from)

# Load moments from SOURCE (the targets don't change between runs).
m = moments()
m.load_run(SOURCE)
m.drop_CHN_IND_BRA_ROW_from_RD = True

# Update list_of_moments: keep FDI_FLOW_N in the codebase but NOT in the
# calibration target list; add FDI_FLOW (bilateral) instead.
# FDI_ELAST stays.
if 'FDI_FLOW_N' in m.list_of_moments:
    m.list_of_moments.remove('FDI_FLOW_N')
for _name in ('FDI_FLOW', 'FDI_ELAST'):
    if _name not in m.list_of_moments:
        m.list_of_moments.append(_name)

# Promote p.a to bilateral if not already.  enable_bilateral_a is idempotent
# and preserves disk-loaded values when bilateral_a is already True.
if not p.bilateral_a:
    print(f"\nPromoting p.a from scalar to bilateral (N={p.N}, S={p.S})...")
    # Use the loaded scalar a as the uniform seed for all off-diagonal entries.
    _seed = float(np.asarray(p.a).reshape(-1)[0]) if np.size(p.a) > 0 else 0.0
    if _seed <= 0:
        _seed = INITIAL_A
    p.enable_bilateral_a(init_value=_seed)
    print(f"  Promoted with uniform seed = {_seed}")
else:
    print(f"\np.a already bilateral, shape {p.a.shape}")

# Append FDI-specific calibrated parameters (idempotent for warm-start).
# NOTE: power_fdi is intentionally EXCLUDED here. With bilateral a, each
# (n,i) pair has its own setup-cost parameter, which subsumes any cross-pair
# pattern that power_fdi (a curvature parameter on the supply-potential
# component) could capture. Calibrating both would be double-dipping and
# leave power_fdi underidentified. We hold p.power_fdi at its default (1.0).
for _name in ('a', 'd_frac'):
    if _name not in p.calib_parameters:
        p.calib_parameters.append(_name)
# Remove 'd' if it's there (replaced by d_frac).
if 'd' in p.calib_parameters:
    p.calib_parameters.remove('d')
# Defensive: ensure power_fdi is NOT in the calibration list, even if a
# warm-start checkpoint loaded it there.
if 'power_fdi' in p.calib_parameters:
    p.calib_parameters.remove('power_fdi')

# Drop "data-zero" FDI flow cells from the calibration (and freeze the
# corresponding bilateral a entries). With FDI_FLOW_ZERO_THRESHOLD > 0
# this masks both the moment residual and the parameter directions.
# The frozen a entries are set to the upper bound (p.ub_dict['a']) to
# encode the prior "no FDI for this pair" - i.e., setup cost is as high
# as the model allows, driving model FDI flow for that pair toward 0.
if FDI_FLOW_ZERO_THRESHOLD > 0:
    print(f"\nApplying FDI_FLOW data-zero mask (threshold "
          f"{FDI_FLOW_ZERO_THRESHOLD:.0e})...")
    m.set_fdi_flow_zero_threshold(FDI_FLOW_ZERO_THRESHOLD)
    p.mask_a_from_fdi_flow_mask(m)
    # Force the frozen a entries to the upper bound so the model produces
    # near-zero FDI for those pairs (matching the data zeros).
    _frozen = ~p.mask['a']
    _frozen[..., 0] = False                          # don't touch sector 0
    np.fill_diagonal(_frozen[..., 1], False)         # don't touch diagonal
    p.a = p.a.copy()                                  # ensure writeable
    p.a[_frozen] = 1000
    print(f"  Set {int(_frozen.sum())} frozen a entries to upper bound "
          f"({p.ub_dict['a']}).")
else:
    print("\nFDI_FLOW masking disabled (threshold=0): "
          "all 132 cells / a entries active.")

# For cold start (no prior FDI calibration), set initial values explicitly.
# For warm start, loaded values take precedence.
if not is_warm_start_bilateral and not is_warm_start_scalar:
    # Already set bilateral a above with INITIAL_A.
    p.d_frac = np.float64(INITIAL_D_FRAC)
    p._sync_d_from_dfrac()
    p.power_fdi = INITIAL_POWER_FDI  # held fixed at default during calibration

p.guess = None

start_time = time.perf_counter()

print(f"\nN={p.N}, S={p.S}")
print(f"p.a shape: {p.a.shape} (bilateral)")
print(f"p.mask['a'].sum() = {p.mask['a'].sum()} free entries "
      f"(of {p.N * (p.N - 1)} off-diagonal sector-1 cells)")
print(f"calib_parameters ({len(p.calib_parameters)}): {p.calib_parameters}")
print(f"list_of_moments  ({len(m.list_of_moments)}): {m.list_of_moments}")
print(f"initial p.k[1]={p.k[1]:.4f}, p.d_frac={float(p.d_frac):.4f}, "
      f"p.d={float(p.d):.4f}, p.power_fdi={float(p.power_fdi):.4f}")
print(f"initial p.a sample: a[0,1,1]={p.a[0,1,1]:.4f}, "
      f"a[5,3,1]={p.a[5,3,1]:.4f}")

hist = history(*tuple(m.list_of_moments + ['objective']))
bounds = p.make_parameters_bounds()
print(f"parameter vector length: {len(p.make_p_vector())}")
print(f"  (= {p.mask['a'].sum()} bilateral a + 1 [d_frac] + "
      f"{len(p.make_p_vector()) - p.mask['a'].sum() - 1} others; "
      f"power_fdi held fixed at {float(p.power_fdi):.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Outer least_squares loop with per-iteration checkpointing
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(DEST, exist_ok=True)
os.environ['CALIB_CKPT_PATH'] = DEST
print(f"\nCALIB_CKPT_PATH set to: {DEST}")
print(f"Running least_squares loop, MAX_ITER={MAX_ITER}, "
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
            x_scale='jac',
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

# Reload best-so-far from DEST (written by the in-call checkpoint mechanism).
print(f"\nReloading best-so-far parameters from {DEST}...")
try:
    p_best = parameters()
    p_best.correct_eur_patent_cost = True
    p_best.load_run(DEST)   # auto-promotes bilateral a if a.csv has N*N*S entries
    for _name in ('a', 'd_frac'):
        if _name not in p_best.calib_parameters:
            p_best.calib_parameters.append(_name)
    if 'd' in p_best.calib_parameters:
        p_best.calib_parameters.remove('d')
    # power_fdi is held fixed in this runner; remove it if a warm-start
    # checkpoint had it in the calib list.
    if 'power_fdi' in p_best.calib_parameters:
        p_best.calib_parameters.remove('power_fdi')
    p_best.guess = None
    p = p_best
    # Re-apply the FDI_FLOW data-zero mask on the reloaded parameters so
    # the held-fixed a entries don't drift back into the free set.
    if FDI_FLOW_ZERO_THRESHOLD > 0:
        p.mask_a_from_fdi_flow_mask(m)
    print(f"  Reloaded. bilateral_a={p.bilateral_a}, p.a.shape={p.a.shape}")
    print(f"  p.d_frac={float(p.d_frac):.4f}, p.d={float(p.d):.4f}, "
          f"k-d-1={float(p.k[1] - p.d - 1):.4f}")
except Exception as e:
    print(f"  Could not reload best-so-far ({e}); using last test_ls.x.")
    if test_ls is not None:
        p.update_parameters(test_ls.x)

print("\nFinal solve...")
# Bootstrap: refresh p.tau by running ONE calibration-context solve at the
# current best parameters. This is required because counterfactual context's
# compute_phi and compute_entry_costs both read p.tau (the "structural" trade
# resistance bundle). p.tau on the parameter object is stale at this point
# (either the value loaded from SOURCE at startup, or from the previous
# best-so-far reload). Without this bootstrap, the counterfactual solver is
# trying to solve a DIFFERENT model from the one we just calibrated and will
# typically fail to converge.
print("  [bootstrap] refreshing p.tau via calibration-context solve...")
sol_bs, sol_c_bs = fixed_point_solver_with_fdi(
    p, context='calibration',
    x0=p.guess,
    cobweb_anim=False, tol=1e-10, accelerate=False,
    accelerate_when_stable=False,
    cobweb_qty='phi',
    plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=False,
    damping=10, max_count=10000,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5,
)

sol_c_bs.scale_P(p)
sol_c_bs.compute_non_solver_quantities(p)
sol_c_bs.compute_tau(p)
p.tau = sol_c_bs.tau
p.guess = sol_bs.x   # warm-start the counterfactual from the converged cal state
print(f"  [bootstrap] OK ({sol_bs.iter} iters). p.tau refreshed.")
m.compute_moments(sol_c_bs, p)
m.compute_moments_deviations()
p.write_params(DEST)
m.write_moments(DEST)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

if sol_bs.status == 'successful':
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
else:
    print(f"\n!!! Final solve failed; reporting parameter values only.")
    print(f"!!! Best moments are in checkpoint.txt at: {DEST}")
    try:
        with open(os.path.join(DEST, 'checkpoint.txt')) as _f:
            print("\n--- contents of checkpoint.txt ---")
            print(_f.read())
    except Exception:
        pass

print(f"\nFinal FDI-specific parameter values:")
print(f"  bilateral_a = {p.bilateral_a}")
print(f"  a shape     = {np.asarray(p.a).shape}")
if p.bilateral_a:
    _a_off = p.a[..., 1][~np.eye(p.N, dtype=bool)]
    print(f"  a stats     : min={_a_off.min():.4f}, max={_a_off.max():.4f}, "
          f"mean={_a_off.mean():.4f}, median={np.median(_a_off):.4f}")
else:
    print(f"  a           = {float(p.a):.6f}")
print(f"  d           = {float(p.d):.6f}   (= d_frac * (k[1] - 1 - margin))")
print(f"  d_frac      = {float(p.d_frac):.6f}")
print(f"  k[1]        = {float(p.k[1]):.6f}")
print(f"  k - d - 1   = {float(p.k[1] - p.d - 1):.6e}  "
      f"(>= {parameters._D_MARGIN} required)")
print(f"  power_fdi   = {float(p.power_fdi):.6f}")

print(f"\nDONE. Output at: {DEST}")