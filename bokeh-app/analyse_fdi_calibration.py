#!/usr/bin/env python3
"""
analyse_fdi_calibration.py
==========================

End-to-end diagnostic of the bilateral-a FDI calibration.

Workflow:
  1. Load the calibration folder.
  2. Refresh p.tau via a calibration-context solve (in case tau.csv is stale).
  3. Solve in counterfactual context as the working baseline.
  4. Compute and plot all moment fits (model vs target).
  5. Compute and visualise FDI-related structural quantities:
       - Bilateral FDI cost heatmap (the 132 calibrated a^s_{ni}).
       - Case-2 incidence (where is FDI activated?).
       - FDI quality share by destination (Σ Ψ^{M,F} / Σ Ψ^{M}).
       - FDI sales share by destination (Σ X^{M,F} / Σ X).
       - Affiliate setup labor share by country.
       - Net FDI receipts (outward affiliate sales − inward affiliate cost).
  6. Run two FDI counterfactuals and decompose the welfare/growth/price impact:
       - "No FDI"    (a -> infinity, FDI banned).
       - "Free FDI"  (a -> ~0,  FDI cost-free).
  7. Per-origin FDI semi-elasticity decomposition (who contributes most
     to the aggregate FDI_ELAST = 0.024?).
  8. A few qualitative cross-cuts: top FDI corridors, "missed" data zeros,
     R&D vs affiliate-setup-labor allocations.

Output: a single multi-page PDF (`fdi_analysis.pdf`) plus a printed summary.

Usage:
  python analyse_fdi_calibration.py  RUN_FOLDER  [OUT_PDF]
  defaults: RUN_FOLDER = ./calibration_results_matched_economy/baseline_2000_variations/fdi_calibrated_bilateral_a/
            OUT_PDF    = ./fdi_analysis.pdf
"""
import sys, os, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, SymLogNorm

from classes import moments, parameters
from solver_funcs import fixed_point_solver_with_fdi


# ─────────────────────────────────────────────────────────────────────────────
# 0. CLI / paths
# ─────────────────────────────────────────────────────────────────────────────
RUN  = (sys.argv[1] if len(sys.argv) > 1 else
        './calibration_results_matched_economy/baseline_2000_variations/'
        'fdi_calibrated_bilateral_a/')
OUT  = sys.argv[2] if len(sys.argv) > 2 else './fdi_analysis.pdf'
if not RUN.endswith('/'): RUN += '/'

print("=" * 78)
print("FDI bilateral-a calibration — diagnostic analysis")
print("=" * 78)
print(f"Loading from: {RUN}")
print(f"Writing to:   {OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Solve at the saved parameters
# ─────────────────────────────────────────────────────────────────────────────
def _solve(p, context, x0, max_count=10000, tol=1e-10):
    sol, var = fixed_point_solver_with_fdi(
        p, context=context, x0=x0,
        cobweb_anim=False, tol=tol, accelerate=False,
        accelerate_when_stable=False, cobweb_qty='phi',
        plot_convergence=False, plot_cobweb=False,
        safe_convergence=0.001, disp_summary=False,
        damping=10, max_count=max_count,
        accel_memory=50, accel_type1=True, accel_regularization=1e-10,
        accel_relaxation=0.5, accel_safeguard_factor=1,
        accel_max_weight_norm=1e6, damping_post_acceleration=5)
    return sol, var

t0 = time.perf_counter()
p = parameters(); p.correct_eur_patent_cost = True
p.load_run(RUN)
print(f"\nLoaded: N={p.N}, S={p.S}, bilateral_a={p.bilateral_a}, "
      f"a.shape={np.asarray(p.a).shape}")
print(f"  d_frac={float(p.d_frac):.4f}, d={float(p.d):.4f}, k={np.array(p.k)}, "
      f"power_fdi={float(p.power_fdi):.3f}")

guess = np.loadtxt(RUN + 'guess.csv')
p.guess = guess

print("\n[1/3] Calibration-context solve (refreshes p.tau)...")
sol_cal, var_cal = _solve(p, 'calibration', x0=p.guess)
print(f"      status={sol_cal.status}, iters={sol_cal.iter}")
var_cal.scale_P(p); var_cal.compute_non_solver_quantities(p)
var_cal.compute_tau(p)
p.tau   = var_cal.tau
p.guess = sol_cal.x

print("\n[2/3] Counterfactual-context solve (working baseline)...")
sol_cf, var = _solve(p, 'counterfactual', x0=p.guess)
print(f"      status={sol_cf.status}, iters={sol_cf.iter}")
var.scale_P(p); var.compute_non_solver_quantities(p)
p.tau = var.tau

print("\n[3/3] Compute moments + deviations.")
m = moments(); m.load_run(RUN); m.drop_CHN_IND_BRA_ROW_from_RD = True
m.compute_moments(var, p)
m.compute_moments_deviations()
print(f"      total ||dev|| = {np.linalg.norm(m.deviation_vector()):.4f}")

COUNTRIES = list(p.countries)
N, S = p.N, p.S


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helpers for plotting
# ─────────────────────────────────────────────────────────────────────────────
def _annot_diag(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], '--', color='0.4', lw=0.8, zorder=0)

def _scatter_panel(ax, model, target, title, *, log=True, mask=None):
    x = np.array(target).ravel(); y = np.array(model).ravel()
    if mask is not None: m = mask.ravel(); x = x[m]; y = y[m]
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite]; y = y[finite]
    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    if len(x):
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        _annot_diag(ax, lo*0.9, hi*1.1)
        ax.scatter(x, y, s=14, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.set_xlim(lo*0.9, hi*1.1); ax.set_ylim(lo*0.9, hi*1.1)
        # correlation in logs
        if log and len(x) > 2:
            r = np.corrcoef(np.log(x), np.log(y))[0, 1]
            ax.text(0.04, 0.96, f'log-corr = {r:.3f}\nn={len(x)}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, lw=0))
    ax.set_xlabel('target'); ax.set_ylabel('model')
    ax.set_title(title, fontsize=10)

def _bar_panel(ax, model, target, title, labels=None):
    x = np.arange(len(np.array(target).ravel()))
    width = 0.4
    ax.bar(x - width/2, np.array(target).ravel(), width, label='target',
           color='steelblue', alpha=0.85)
    ax.bar(x + width/2, np.array(model).ravel(),  width, label='model',
           color='tomato',    alpha=0.85)
    if labels is not None:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right',
                                              fontsize=8)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PDF
# ─────────────────────────────────────────────────────────────────────────────
pdf = PdfPages(OUT)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: cover & moment-fit summary table
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.axis('off')
ax.text(0.5, 0.95, 'FDI bilateral-a calibration', ha='center', fontsize=18,
        weight='bold', transform=ax.transAxes)
ax.text(0.5, 0.91, RUN, ha='center', fontsize=8, family='monospace',
        transform=ax.transAxes)

# Build a per-moment fit table
rows = []
for mom in m.list_of_moments:
    dev = getattr(m, mom + '_deviation', None)
    if dev is None: continue
    dev_flat = np.array(dev).ravel()
    n = dev_flat.size
    mod = np.array(getattr(m, mom)).ravel()
    tgt = np.array(getattr(m, mom + '_target')).ravel()
    rows.append({
        'moment': mom, 'n': n,
        'weight': m.weights_dict.get(mom, 1.0),
        '|dev|_2': np.linalg.norm(dev_flat),
        'mean|dev|': np.abs(dev_flat).mean() if n else 0,
        'max|dev|': np.abs(dev_flat).max() if n else 0,
    })
df_fit = pd.DataFrame(rows).set_index('moment')
total_sse = (df_fit['|dev|_2']**2).sum()
df_fit['SSE_share_%'] = 100 * df_fit['|dev|_2']**2 / total_sse
df_fit = df_fit.round({'weight': 1, '|dev|_2': 3, 'mean|dev|': 3,
                       'max|dev|': 3, 'SSE_share_%': 1})

# Render table
ax.text(0.5, 0.86, f'Per-moment fit — total ||dev|| = '
        f'{np.linalg.norm(m.deviation_vector()):.3f}',
        ha='center', fontsize=11, transform=ax.transAxes)
tbl = ax.table(cellText=df_fit.reset_index().values.astype(str),
               colLabels=['moment'] + list(df_fit.columns),
               loc='center', cellLoc='center',
               bbox=[0.05, 0.20, 0.9, 0.62])
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.15)

# Params summary
ptxt = (f"N={N}, S={S}, bilateral a (132 free) + d_frac (1)\n"
        f"d_frac = {float(p.d_frac):.4f}  →  d = {float(p.d):.4f}\n"
        f"k = [{p.k[0]:.3f}, {p.k[1]:.3f}],  power_fdi = "
        f"{float(p.power_fdi):.3f} (fixed)\n"
        f"k[1] − d − 1 = {float(p.k[1] - p.d - 1):.4f}  (margin)\n"
        f"calibration-solve iters: {sol_cal.iter},  CF-solve iters: "
        f"{sol_cf.iter}")
ax.text(0.05, 0.03, ptxt, family='monospace', fontsize=8,
        transform=ax.transAxes)
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: moment fits — scalars + small vectors
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('Moment fit — scalars and small vectors '
             '(blue=target, red=model)', fontsize=12, weight='bold')

idx_ax = 0
scalar_or_vec = [mom for mom in m.list_of_moments
                 if mom not in ('SPFLOW', 'FDI_FLOW')]
for mom in scalar_or_vec:
    ax = axes.flat[idx_ax]; idx_ax += 1
    target = np.array(getattr(m, mom + '_target'))
    model  = np.array(getattr(m, mom))
    if target.size <= 1:
        tgt_v = float(np.array(target).ravel()[0])
        mod_v = float(np.array(model ).ravel()[0])
        ax.bar([0, 1], [tgt_v, mod_v],
               color=['steelblue', 'tomato'], alpha=0.85)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['target', 'model'],
                                                    fontsize=9)
        ax.set_title(f'{mom}  (scalar)', fontsize=10)
    else:
        # Vector — bar with country labels if length matches
        labels = COUNTRIES if len(np.array(target).ravel()) == N else None
        # RD was trimmed to 5 entries -- detect by length
        if mom == 'RD' and len(np.array(target).ravel()) == 5:
            labels = ['USA', 'EUR', 'JAP', 'KOR', 'CAN']  # rough — depends on
            # drop_CHN_IND_BRA_ROW order, just label by index if unsure
            labels = [f'idx{i}' for i in range(5)]
        _bar_panel(ax, model, target, mom, labels=labels)

for k in range(idx_ax, len(axes.flat)): axes.flat[k].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: SPFLOW + FDI_FLOW scatter fits (the bilateral moments)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# SPFLOW is (N, N-1) — off-diagonal patent flows by origin
ax = axes[0]
spf_tgt = np.array(getattr(m, 'SPFLOW_target'))
spf_mod = np.array(getattr(m, 'SPFLOW'))
_scatter_panel(ax, spf_mod, spf_tgt, 'SPFLOW (132 bilateral patent flows)',
               log=True)

# FDI_FLOW is (N, N) with diagonal=1 -- exclude diagonal
ax = axes[1]
fdi_tgt = np.array(getattr(m, 'FDI_FLOW_target'))
fdi_mod = np.array(getattr(m, 'FDI_FLOW'))
off = ~np.eye(N, dtype=bool)
_scatter_panel(ax, fdi_mod, fdi_tgt, 'FDI_FLOW (132 bilateral FDI ratios, '
               'off-diagonal)', log=True, mask=off)
fig.suptitle('Bilateral moment fits — log–log scatter', fontsize=12,
             weight='bold')
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4: FDI_FLOW heatmaps (target, model, log-residual)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
data_panels = [
    (fdi_tgt, 'FDI_FLOW target', 'viridis'),
    (fdi_mod, 'FDI_FLOW model',  'viridis'),
]
for ax, (mat, title, cmap) in zip(axes[:2], data_panels):
    masked = np.where(np.eye(N, dtype=bool), np.nan, mat)
    im = ax.imshow(masked, cmap=cmap,
                   norm=LogNorm(vmin=max(1e-6, np.nanmin(masked)),
                                vmax=np.nanmax(masked)),
                   aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=90,
                                                  fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(COUNTRIES, fontsize=8)
    ax.set_xlabel('origin i'); ax.set_ylabel('destination n')
    ax.set_title(title, fontsize=10)

# Log-residual
ax = axes[2]
log_res = np.where(np.eye(N, dtype=bool), np.nan,
                   np.log(np.maximum(fdi_mod, 1e-30)
                          / np.maximum(fdi_tgt, 1e-30)))
vmax = np.nanmax(np.abs(log_res))
im = ax.imshow(log_res, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.04)
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=90, fontsize=8)
ax.set_yticks(range(N)); ax.set_yticklabels(COUNTRIES, fontsize=8)
ax.set_xlabel('origin i'); ax.set_ylabel('destination n')
ax.set_title('log(model / target)\nred = model > target', fontsize=10)
fig.suptitle('FDI_FLOW heatmaps', fontsize=12, weight='bold')
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5: Bilateral FDI setup-cost a^s_{ni}
# ─────────────────────────────────────────────────────────────────────────────
a_arr = np.asarray(p.a)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
a1 = np.where(np.eye(N, dtype=bool), np.nan, a_arr[..., 1])
im = ax.imshow(a1, cmap='magma',
               norm=LogNorm(vmin=max(1e-3, np.nanmin(a1)),
                            vmax=np.nanmax(a1)), aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.04, label='a (log scale)')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=90, fontsize=8)
ax.set_yticks(range(N)); ax.set_yticklabels(COUNTRIES, fontsize=8)
ax.set_xlabel('origin i'); ax.set_ylabel('destination n')
ax.set_title('Calibrated bilateral FDI cost  a^1_{ni}', fontsize=10)

# Histogram + bound markers
ax = axes[1]
a_off = a1[~np.isnan(a1)]
ax.hist(np.log10(a_off), bins=30, color='steelblue', alpha=0.8,
        edgecolor='k')
ax.axvline(np.log10(1e-6), color='red', ls='--', label='lb (1e-6)')
ax.axvline(np.log10(10),   color='red', ls='--', label='ub (10)')
ax.set_xlabel('log10(a^1_{ni})'); ax.set_ylabel('count')
ax.set_title(f'Distribution of 132 calibrated a values\n'
             f'min={a_off.min():.3g}, max={a_off.max():.3g}, '
             f'median={np.median(a_off):.3g}', fontsize=10)
ax.legend()
fig.suptitle('Bilateral FDI setup costs', fontsize=12, weight='bold')
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6: How important is FDI in the model? (destination-level)
# ─────────────────────────────────────────────────────────────────────────────
# All metrics computed for sector 1 (the only sector with FDI/patenting).

# A) Share of monopolistic quality from FDI, by destination
psi_M_O_sum  = var.PSI_M_O[..., 1].sum(axis=1)        # (N,) — sum over origins
psi_M_F_sum  = var.PSI_M_F[..., 1].sum(axis=1)
fdi_quality_share = np.where(psi_M_O_sum + psi_M_F_sum > 0,
                              psi_M_F_sum / (psi_M_O_sum + psi_M_F_sum), 0)

# B) FDI sales share of monopolistic absorption, by destination
X_M_sum   = var.X_M[..., 1].sum(axis=1)               # exports inflow
X_M_F_sum = var.X_M_F[..., 1].sum(axis=1)             # affiliate sales inflow
fdi_sales_share = np.where(X_M_sum + X_M_F_sum > 0,
                            X_M_F_sum / (X_M_sum + X_M_F_sum), 0)

# C) Case-2 incidence per destination (fraction of foreign origins that
#    have FDI activated, i.e. some firms become affiliates)
off_diag = ~np.eye(N, dtype=bool)
case2 = var.case2[..., 1]
case2_per_n = np.where(off_diag, case2, np.nan)       # NaN on diagonal
c2_share = np.nanmean(case2_per_n, axis=1)            # (N,)

# D) FDI quality share by ORIGIN — how much of i's "monopolistic
#    presence abroad" is via affiliate, not via export?
psi_M_O_origin = var.PSI_M_O[..., 1].sum(axis=0)      # sum over destinations
psi_M_F_origin = var.PSI_M_F[..., 1].sum(axis=0)
fdi_quality_share_origin = np.where(
    psi_M_O_origin + psi_M_F_origin > 0,
    psi_M_F_origin / (psi_M_O_origin + psi_M_F_origin), 0)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.bar(range(N), fdi_quality_share, color='teal', alpha=0.85, edgecolor='k')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45, ha='right',
                                              fontsize=9)
ax.set_ylabel('Σ_i Ψ^{M,F}_{ni} / Σ_i (Ψ^{M,F} + Ψ^{M,O})')
ax.set_title('A. Monopolistic quality stock — FDI share\n(destination n)',
             fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(fdi_quality_share):
    ax.text(i, v + 0.005, f'{v:.2f}', ha='center', fontsize=8)

ax = axes[0, 1]
ax.bar(range(N), fdi_sales_share, color='goldenrod', alpha=0.85,
       edgecolor='k')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('Σ_i X^{M,F}_{ni} / Σ_i (X^{M,F} + X^{M,O})')
ax.set_title('B. Monopolistic SALES — FDI share\n(destination n)',
             fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(fdi_sales_share):
    ax.text(i, v + 0.005, f'{v:.2f}', ha='center', fontsize=8)

ax = axes[1, 0]
ax.bar(range(N), c2_share, color='indianred', alpha=0.85, edgecolor='k')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('fraction of (n, i≠n) pairs in Case 2')
ax.set_title('C. Case-2 (FDI active) incidence\n(destination n)',
             fontsize=10)
ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(c2_share):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

ax = axes[1, 1]
ax.bar(range(N), fdi_quality_share_origin, color='mediumpurple', alpha=0.85,
       edgecolor='k')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('Σ_n Ψ^{M,F}_{ni} / Σ_n (Ψ^{M,F} + Ψ^{M,O})')
ax.set_title('D. FDI share of foreign presence — by ORIGIN i\n'
             '(how much of country i\'s abroad-presence is FDI?)',
             fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(fdi_quality_share_origin):
    ax.text(i, v + 0.005, f'{v:.2f}', ha='center', fontsize=8)

fig.suptitle('How important is FDI? — Destination and origin perspectives',
             fontsize=12, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 7: Labor & net FDI receipts
# ─────────────────────────────────────────────────────────────────────────────
# Total labor allocations to R&D, original-patenting, export-patenting,
# affiliate-setup (per country, sector 1).
L_R  = var.l_R[..., 1]                                # (N,) destination
L_o  = var.l_Ao[..., 1]                               # (N,) origin
# l_Ae is indexed (i_origin, n_destination, s) per construction
L_e  = var.l_Ae[..., 1].sum(axis=1)                   # sum over destinations
                                                      # → labor at origin i
                                                      # for export patenting
L_F  = var.l_F[..., 1].sum(axis=0)                    # labor used at DEST
                                                      # to host affiliates
# Aggregate into innovation labor at the COUNTRY level:
#   - "innovating-as-origin": L_R + L_o + L_e  (research + patent fixed costs)
#   - "hosting affiliates":   L_F              (sunk setup, paid at dest)
innov_labor = L_R + L_o + L_e
total_labor_in_block = innov_labor + L_F
share_LF = np.where(total_labor_in_block > 0,
                    L_F / total_labor_in_block, 0)

# Net FDI receipts: outward affiliate sales minus inward affiliate cost.
#   Affiliate revenue earned by origin i:  Σ_n (1/σ) X^{M,F}_{nis}
#   Setup-labor paid by destination n:     w_n * Σ_i a^s_{ni} * (...) — but
#   `l_F` already aggregates this in labor units (multiplied by w_n it is in
#   nominal terms).  We use simple revenue and labor-cost measures.
sigma1 = float(p.sigma[1])
rev_outward = var.X_M_F[..., 1].sum(axis=0) / sigma1   # received by origin i
rev_inward  = var.X_M_F[..., 1].sum(axis=1) / sigma1   # paid by destination n
labor_cost_host = var.w * L_F                          # wage bill, dest n

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
xs = np.arange(N); width = 0.22
ax.bar(xs - 1.5*width, L_R, width, label='L^R (research)',
       color='steelblue')
ax.bar(xs - 0.5*width, L_o, width, label='L^o (orig-patent fixed)',
       color='goldenrod')
ax.bar(xs + 0.5*width, L_e, width, label='L^e (export-patent fixed)',
       color='indianred')
ax.bar(xs + 1.5*width, L_F, width, label='L^F (affiliate setup, host)',
       color='mediumpurple')
ax.set_xticks(xs); ax.set_xticklabels(COUNTRIES, rotation=45, ha='right',
                                       fontsize=9)
ax.set_yscale('log')
ax.set_ylabel('labor in patent block (log)')
ax.set_title('A. Labor allocation across innovation activities',
             fontsize=10)
ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
ax.bar(range(N), 100 * share_LF, color='mediumpurple', alpha=0.85,
       edgecolor='k')
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('% of country\'s innovation+FDI labor used for hosting '
              'affiliates')
ax.set_title('B. Affiliate-hosting share of country\'s patent-block labor',
             fontsize=10)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(100 * share_LF):
    ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=8)

ax = axes[1, 0]
net_fdi = rev_outward - rev_inward
colors_net = ['seagreen' if v >= 0 else 'tomato' for v in net_fdi]
ax.bar(range(N), net_fdi, color=colors_net, alpha=0.85, edgecolor='k')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('outward − inward affiliate revenue (model units)')
ax.set_title('C. Net affiliate revenue\n'
             '(green: net FDI sender; red: net receiver)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
gdp = var.w * np.array(p.labor) + (var.w * 0)  # nominal GDP ~ wL is rough
# Better: use the model's nominal_value_added if available, else fall back
try:
    nva = var.nominal_value_added.sum(axis=1)            # (N,) sector-sum
except AttributeError:
    nva = var.w * np.array(p.labor)
ax.bar(range(N), 100 * net_fdi / nva, color=colors_net, alpha=0.85,
       edgecolor='k')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(range(N)); ax.set_xticklabels(COUNTRIES, rotation=45,
                                              ha='right', fontsize=9)
ax.set_ylabel('% of nominal value added')
ax.set_title('D. Net affiliate revenue, % of nominal value added',
             fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Labor and net affiliate-revenue flows', fontsize=12,
             weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 8: COUNTERFACTUAL — shut down FDI, free FDI
# ─────────────────────────────────────────────────────────────────────────────
print("\n[CF] Running 'no-FDI' counterfactual (a × 10000)...")
import copy
p_no = copy.deepcopy(p)
p_no.a = (np.asarray(p.a) * 10000.0)
np.einsum('nns->ns', p_no.a)[:] = 0  # preserve diagonal=0
p_no.guess = sol_cf.x.copy()
sol_no, var_no = _solve(p_no, 'counterfactual', x0=p_no.guess, max_count=20000)
print(f"      status={sol_no.status}, iters={sol_no.iter}")

print("[CF] Running 'free-FDI' counterfactual (a × 0.01)...")
p_lo = copy.deepcopy(p)
p_lo.a = (np.asarray(p.a) * 0.01)
np.einsum('nns->ns', p_lo.a)[:] = 0
p_lo.guess = sol_cf.x.copy()
sol_lo, var_lo = _solve(p_lo, 'counterfactual', x0=p_lo.guess, max_count=20000)
print(f"      status={sol_lo.status}, iters={sol_lo.iter}")

# Process whichever converged
def _proc(var_, p_):
    var_.scale_P(p_); var_.compute_non_solver_quantities(p_)
    return var_

variants = [('Baseline', var, p)]
if sol_no.status == 'successful':
    variants.append(('No FDI',   _proc(var_no, p_no), p_no))
if sol_lo.status == 'successful':
    variants.append(('Free FDI', _proc(var_lo, p_lo), p_lo))

# Tabulate aggregate metrics
cf_rows = []
for name, v_, p_ in variants:
    fdi_sales = v_.X_M_F[..., 1].sum() / v_.X[..., 1].sum()
    fdi_quality = (v_.PSI_M_F[..., 1].sum()
                   / (v_.PSI_M_F[..., 1].sum() + v_.PSI_M_O[..., 1].sum()))
    growth = float(v_.g)
    avg_w  = float(np.mean(v_.w))
    avg_P  = float(np.mean(v_.price_indices))
    c2_count = int(v_.case2[..., 1].sum())
    cf_rows.append({
        'scenario': name,
        'aggregate growth g': growth,
        'mean wage w': avg_w,
        'mean price index P': avg_P,
        'FDI sales / total sales': fdi_sales,
        'FDI quality share': fdi_quality,
        'pairs in Case 2': c2_count,
    })
df_cf = pd.DataFrame(cf_rows).round(4)

fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')
ax.set_title('Counterfactual aggregates', fontsize=12, weight='bold')
tbl = ax.table(cellText=df_cf.values.astype(str),
               colLabels=list(df_cf.columns), loc='center',
               cellLoc='center', bbox=[0.02, 0.0, 0.96, 0.9])
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

# Cross-scenario per-country: wage and price index
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
xs = np.arange(N); w_total = len(variants); width = 0.8 / w_total
colors = ['steelblue', 'tomato', 'seagreen']

for k_, (name, v_, _) in enumerate(variants):
    axes[0,0].bar(xs + (k_ - (w_total-1)/2)*width, v_.w, width,
                  label=name, color=colors[k_])
    axes[0,1].bar(xs + (k_ - (w_total-1)/2)*width, v_.price_indices, width,
                  label=name, color=colors[k_])
    fdi_sh = (v_.X_M_F[..., 1].sum(axis=1)
              / np.maximum(v_.X_M[..., 1].sum(axis=1)
                           + v_.X_M_F[..., 1].sum(axis=1), 1e-30))
    axes[1,0].bar(xs + (k_ - (w_total-1)/2)*width, fdi_sh, width,
                  label=name, color=colors[k_])
    psi_share = (v_.PSI_M_F[..., 1].sum(axis=1)
                 / np.maximum(v_.PSI_M_F[..., 1].sum(axis=1)
                              + v_.PSI_M_O[..., 1].sum(axis=1), 1e-30))
    axes[1,1].bar(xs + (k_ - (w_total-1)/2)*width, psi_share, width,
                  label=name, color=colors[k_])

for ax, ttl, yl in zip(axes.flat,
        ['Wage w_n', 'Price index P_n',
         'FDI sales share (destination)',
         'FDI quality share (destination)'],
        ['wage', 'price index', 'share', 'share']):
    ax.set_xticks(xs); ax.set_xticklabels(COUNTRIES, rotation=45,
                                            ha='right', fontsize=8)
    ax.set_title(ttl, fontsize=10); ax.set_ylabel(yl); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Counterfactual comparison — baseline vs no-FDI vs free-FDI',
             fontsize=12, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 10: Per-origin contribution to aggregate FDI semi-elasticity
# ─────────────────────────────────────────────────────────────────────────────
# Re-run _vartheta_ni_perturbed to get vartheta_ni at baseline and at dagger,
# then decompose the change in bar_vartheta_US into per-origin contributions.
us_idx = 0
delta_b = 0.1
vartheta_base = m._vartheta_ni(var, p)            # (N, N, S-1)
vartheta_pert = m._vartheta_ni_perturbed(var, p, us_idx, delta_b)
# Aggregation weight per origin: η_i * (L^R_i)^{1-κ}, excluding i = n
eta_w_R = (p.eta[:, 1:] * var.l_R[:, 1:]**(1 - p.kappa))   # (N, S-1)
weights = eta_w_R[:, 0]                                    # sector 1
US_row_base = vartheta_base[us_idx, :, 0]                  # (N,)
US_row_pert = vartheta_pert[us_idx, :, 0]                  # (N,)
# Exclude diagonal
not_us = np.ones(N, bool); not_us[us_idx] = False
num_base = (weights[not_us] * US_row_base[not_us]).sum()
num_pert = (weights[not_us] * US_row_pert[not_us]).sum()
denom = weights[not_us].sum()
bar_base = num_base / denom
bar_pert = num_pert / denom
# Per-origin contribution to (bar_pert − bar_base) / delta_b
contrib = weights[not_us] * (US_row_pert[not_us] - US_row_base[not_us]) \
          / (denom * delta_b)
# Map to origins
origins = [COUNTRIES[i] for i in range(N) if i != us_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
order = np.argsort(-contrib)
ax.barh(np.arange(len(contrib)), contrib[order],
        color=['steelblue' if c >= 0 else 'tomato' for c in contrib[order]],
        alpha=0.85, edgecolor='k')
ax.set_yticks(np.arange(len(contrib)))
ax.set_yticklabels(np.array(origins)[order], fontsize=9)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('contribution to FDI semi-elasticity (per-origin)')
ax.set_title(f'A. Origin decomposition of FDI_ELAST\n'
             f'sum = {contrib.sum():.4f} ≈ model FDI_ELAST = '
             f'{np.array(m.FDI_ELAST).ravel()[0]:.4f}', fontsize=10)
ax.grid(axis='x', alpha=0.3)

ax = axes[1]
ax.bar(np.arange(len(US_row_base[not_us])), US_row_base[not_us],
       label='baseline ϑ_{US,i}', alpha=0.7, color='steelblue')
ax.bar(np.arange(len(US_row_pert[not_us])), US_row_pert[not_us],
       label='perturbed ϑ_{US,i}^†', alpha=0.7, color='tomato')
ax.set_xticks(range(len(origins))); ax.set_xticklabels(origins, rotation=45,
                                                         ha='right',
                                                         fontsize=9)
ax.set_ylabel('FDI probability ϑ_{US,i}')
ax.set_title('B. Per-origin FDI probability (at US) — baseline vs perturbed',
             fontsize=10)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('FDI semi-elasticity (Blonigen 2002 target 0.08) — '
             'origin decomposition',
             fontsize=12, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 11: top corridors & "data-zero" diagnosis
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Top 15 FDI flow target corridors (off-diagonal)
mask = ~np.eye(N, dtype=bool)
flat_tgt = fdi_tgt[mask]
flat_mod = fdi_mod[mask]
n_idx, i_idx = np.where(mask)
order = np.argsort(-flat_tgt)[:15]
ax = axes[0]
labels = [f'{COUNTRIES[n_idx[k]]}←{COUNTRIES[i_idx[k]]}' for k in order]
yy = np.arange(len(order))
ax.barh(yy + 0.2, flat_tgt[order], 0.4, label='target', color='steelblue')
ax.barh(yy - 0.2, flat_mod[order], 0.4, label='model', color='tomato')
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('FDI_FLOW value')
ax.set_title('A. Top-15 FDI corridors by target magnitude\n'
             '(destination ← origin)', fontsize=10)
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Histogram of target — bimodality of FDI data
ax = axes[1]
ax.hist(np.log10(np.maximum(flat_tgt, 1e-7)), bins=40, color='steelblue',
        alpha=0.7, edgecolor='k', label='target')
ax.hist(np.log10(np.maximum(flat_mod, 1e-7)), bins=40, color='tomato',
        alpha=0.5, edgecolor='k', label='model')
n_at_floor = int((flat_tgt <= 1.05e-6).sum())
ax.axvline(np.log10(1e-6), color='black', ls='--', lw=1)
ax.text(np.log10(1e-6) + 0.1, ax.get_ylim()[1]*0.9,
        f'{n_at_floor} entries floored at 1e-6 in data',
        fontsize=8)
ax.set_xlabel('log10(FDI_FLOW)')
ax.set_ylabel('count')
ax.set_title('B. Distribution of bilateral FDI ratios',
             fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('Top corridors & where the FDI data sits', fontsize=12,
             weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# done
# ─────────────────────────────────────────────────────────────────────────────
pdf.close()
print(f"\n[OK] wrote {OUT}  ({time.perf_counter()-t0:.1f}s)")
print()
print("=" * 78)
print("SUMMARY OF FDI'S IMPORTANCE IN THE CALIBRATED MODEL")
print("=" * 78)
print(f"  Aggregate FDI sales share (sector 1):  "
      f"{var.X_M_F[..., 1].sum() / var.X[..., 1].sum():.3f}")
print(f"  Aggregate FDI quality share (sector 1):"
      f" {var.PSI_M_F[..., 1].sum() / (var.PSI_M_F[..., 1].sum() + var.PSI_M_O[..., 1].sum()):.3f}")
print(f"  Bilateral pairs in Case 2 (out of {N*(N-1)}):  "
      f"{int(var.case2[..., 1].sum())}")
print(f"  Aggregate growth g:                     {float(var.g):.4f}")
if sol_no.status == 'successful':
    g_no = float(var_no.g)
    print(f"  Growth WITHOUT FDI:                     {g_no:.4f}   "
          f"(Δ = {g_no - float(var.g):+.4f})")
if sol_lo.status == 'successful':
    g_lo = float(var_lo.g)
    print(f"  Growth with FREE FDI:                   {g_lo:.4f}   "
          f"(Δ = {g_lo - float(var.g):+.4f})")
print("=" * 78)
