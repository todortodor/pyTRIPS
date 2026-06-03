#!/usr/bin/env python3
"""
analyse_fdi_fit.py
==================

Generates a multi-page PDF report on the v4 FDI calibration:

  1. Per-moment model-vs-target scatter plots (one panel per moment).
     Outliers are highlighted with annotated red circles. Diagonal,
     +/- 2x and +/- 10x bands give visual sense of fit.
  2. "Importance of FDI" — data vs model panels for the FDI-related
     aggregates that have data analogs.

Usage:
    python analyse_fdi_fit.py  RUN_FOLDER  [OUT_PDF]
"""
import sys, os, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from classes import moments, parameters
from solver_funcs import fixed_point_solver_with_fdi


# ─────────────────────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────────────────────
def scatter_panel(ax, target, model, title, *, labels=None,
                  log=True, n_outliers=5, point_size=55, exclude_mask=None,
                  exclude_label=None):
    """
    Generic scatter: model on y, target on x, log-log if log=True.
    - 45-degree dashed line.
    - +/- 2x band (dark grey) and +/- 10x band (light grey).
    - Up to `n_outliers` worst-fit points circled in red and labelled.
    - If labels=None and target is 1D, points are numbered.
    - If exclude_mask given (True = exclude), shows excluded points in light grey
      and computes fit stats only on included points.
    """
    tgt = np.array(target).ravel()
    mod = np.array(model).ravel()
    if labels is None:
        labels = [str(k) for k in range(len(tgt))]
    labels = list(labels)

    if exclude_mask is None:
        included = np.ones(len(tgt), bool)
    else:
        included = ~np.array(exclude_mask).ravel()

    # Strip non-finite and non-positive for log axes
    if log:
        valid = (tgt > 0) & (mod > 0) & np.isfinite(tgt) & np.isfinite(mod)
    else:
        valid = np.isfinite(tgt) & np.isfinite(mod)

    use = valid & included
    tgt_in = tgt[use]; mod_in = mod[use]
    labels_in = [labels[k] for k in np.where(use)[0]]

    # Background: excluded points (greyed)
    excl = valid & ~included
    if excl.any():
        ax.scatter(tgt[excl], mod[excl], s=point_size*0.6, alpha=0.25,
                   c='lightgrey', edgecolor='grey', linewidth=0.3, zorder=1,
                   label=exclude_label or 'excluded')

    if len(tgt_in) == 0:
        ax.text(0.5, 0.5, 'no valid data', transform=ax.transAxes,
                ha='center', va='center'); return

    # Axis bounds
    all_finite = np.concatenate([tgt_in, mod_in])
    if log:
        lo = max(all_finite[all_finite > 0].min() * 0.5, 1e-30)
        hi = all_finite.max() * 2
    else:
        lo = all_finite.min() - 0.1 * (all_finite.max() - all_finite.min() + 1e-9)
        hi = all_finite.max() + 0.1 * (all_finite.max() - all_finite.min() + 1e-9)

    # Bands
    if log:
        ax.fill_between([lo, hi], [lo/2, hi/2], [lo*2, hi*2],
                        color='grey', alpha=0.18, zorder=0,
                        label='±2× band')
        ax.fill_between([lo, hi], [lo/10, hi/10], [lo*10, hi*10],
                        color='grey', alpha=0.08, zorder=0,
                        label='±10× band')
    # Diagonal
    ax.plot([lo, hi], [lo, hi], '--', color='0.45', lw=1, zorder=0)

    # Main scatter
    ax.scatter(tgt_in, mod_in, s=point_size, alpha=0.75, c='steelblue',
               edgecolor='k', linewidth=0.4, zorder=2)

    # Outliers
    if log:
        resid = np.abs(np.log(mod_in / tgt_in))
    else:
        resid = np.abs(mod_in - tgt_in)
    order = np.argsort(-resid)
    n_show = min(n_outliers, len(order))
    for k in order[:n_show]:
        ax.scatter(tgt_in[k], mod_in[k], s=point_size+70, facecolor='none',
                   edgecolor='red', linewidth=1.4, zorder=3)
        ax.annotate(labels_in[k], (tgt_in[k], mod_in[k]),
                    xytext=(7, -2), textcoords='offset points',
                    fontsize=7.5, color='darkred', zorder=4)

    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('target'); ax.set_ylabel('model')
    ax.grid(True, which='both', alpha=0.25)

    # Stats annotation
    if log and len(tgt_in) > 2:
        log_resid = np.log(mod_in / tgt_in)
        r = np.corrcoef(np.log(tgt_in), np.log(mod_in))[0, 1]
        rmse = np.sqrt((log_resid**2).mean())
        bias = log_resid.mean()
        stat_txt = f'n={len(tgt_in)}\nlog-corr={r:.3f}\nRMSE(log)={rmse:.2f}\nbias(log)={bias:+.2f}'
    elif len(tgt_in) > 2:
        rmse = np.sqrt(((mod_in - tgt_in)**2).mean())
        stat_txt = f'n={len(tgt_in)}\nRMSE={rmse:.3g}'
    else:
        stat_txt = f'n={len(tgt_in)}'
    ax.text(0.04, 0.96, stat_txt, transform=ax.transAxes, va='top',
            fontsize=8, family='monospace',
            bbox=dict(facecolor='white', alpha=0.85, lw=0.3))

    ax.set_title(title, fontsize=10)


def bar_pair_panel(ax, label, target, model, *, fmt='.3f'):
    """Scalar moment: two adjacent bars (target vs model)."""
    tv = float(np.array(target).ravel()[0])
    mv = float(np.array(model).ravel()[0])
    ax.bar([0, 1], [tv, mv], color=['steelblue', 'tomato'],
           alpha=0.85, edgecolor='k')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['target', 'model'], fontsize=9)
    ax.set_title(f'{label}', fontsize=10)
    # Annotate values
    ax.text(0, tv, f'{tv:{fmt}}', ha='center', va='bottom', fontsize=8)
    ax.text(1, mv, f'{mv:{fmt}}', ha='center', va='bottom', fontsize=8)
    # Highlight if model is far off
    rel_err = abs(mv - tv) / max(abs(tv), 1e-12)
    if rel_err > 0.2:
        ax.set_facecolor('#fff5f5')
    ax.grid(axis='y', alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
RUN = (sys.argv[1] if len(sys.argv) > 1 else
       '/tmp/v4c/fdi_calibrated_bilateral_a_v4')
OUT = sys.argv[2] if len(sys.argv) > 2 else '/mnt/user-data/outputs/fdi_fit_report.pdf'
if not RUN.endswith('/'): RUN += '/'

t0 = time.perf_counter()
print(f"Loading {RUN}")
print(f"Writing {OUT}")

p = parameters(); p.correct_eur_patent_cost = True; p.load_run(RUN)
m = moments(); m.load_run(RUN); m.drop_CHN_IND_BRA_ROW_from_RD = True
if 'FDI_FLOW_N' in m.list_of_moments: m.list_of_moments.remove('FDI_FLOW_N')
for _ in ('FDI_FLOW', 'FDI_ELAST'):
    if _ not in m.list_of_moments: m.list_of_moments.append(_)
m.set_fdi_flow_zero_threshold(1e-6)
p.mask_a_from_fdi_flow_mask(m)

guess = np.loadtxt(RUN + 'guess.csv'); p.guess = guess
print("Solving (calibration context)...")
sol, var = fixed_point_solver_with_fdi(
    p, context='calibration', x0=p.guess,
    cobweb_anim=False, tol=1e-10, accelerate=False,
    accelerate_when_stable=False, cobweb_qty='phi',
    plot_convergence=False, plot_cobweb=False,
    safe_convergence=0.001, disp_summary=False,
    damping=10, max_count=15000,
    accel_memory=50, accel_type1=True, accel_regularization=1e-10,
    accel_relaxation=0.5, accel_safeguard_factor=1, accel_max_weight_norm=1e6,
    damping_post_acceleration=5)
print(f"  status={sol.status}, iters={sol.iter}")
var.scale_P(p); var.compute_non_solver_quantities(p); var.compute_tau(p); p.tau = var.tau
m.compute_moments(var, p); m.compute_moments_deviations()

COUNTRIES = list(p.countries)
N = p.N

# ─────────────────────────────────────────────────────────────────────────────
# Begin PDF
# ─────────────────────────────────────────────────────────────────────────────
pdf = PdfPages(OUT)

# ── Page 1: cover with summary ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.axis('off')
ax.text(0.5, 0.95, 'FDI calibration v4 — fit & importance report',
        ha='center', fontsize=18, weight='bold', transform=ax.transAxes)
ax.text(0.5, 0.91, RUN, ha='center', fontsize=8, family='monospace',
        transform=ax.transAxes)

# Per-moment fit summary table
rows = []
for mom in m.list_of_moments:
    dev = getattr(m, mom + '_deviation', None)
    if dev is None: continue
    dev = np.array(dev).ravel()
    rows.append([mom, dev.size, m.weights_dict.get(mom, 1.0),
                 np.linalg.norm(dev), (dev**2).sum()])
total_sse = sum(r[4] for r in rows)
rows.sort(key=lambda r: -r[4])
ax.text(0.5, 0.86, f'Per-moment fit — total ||dev|| = '
        f'{np.linalg.norm(m.deviation_vector()):.4f}',
        ha='center', fontsize=11, transform=ax.transAxes)

table_data = [['moment', 'n', 'w', '||dev||', 'SSE share']]
for mom, n, w, norm, sse in rows:
    table_data.append([mom, str(n), f'{w:.0f}',
                       f'{norm:.3f}', f'{100*sse/total_sse:.1f}%'])
tbl = ax.table(cellText=table_data, loc='center', cellLoc='center',
               bbox=[0.20, 0.30, 0.60, 0.55])
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
# Bold header
for j in range(5):
    tbl[(0, j)].set_text_props(weight='bold')

# Param/setup info
ptxt = (f"N={N}, S={p.S}\n"
        f"bilateral a: 120 free (12 frozen at 1000 for data-zero pairs)\n"
        f"d_frac={float(p.d_frac):.4f}  =>  d={float(p.d):.4f}\n"
        f"k=[{p.k[0]:.3f}, {p.k[1]:.3f}],  k-d-1={float(p.k[1]-p.d-1):.4f}\n"
        f"power_fdi={float(p.power_fdi):.2f} (fixed)\n"
        f"solver: {sol.status}, {sol.iter} iters")
ax.text(0.05, 0.20, ptxt, family='monospace', fontsize=8,
        transform=ax.transAxes, va='top')
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 2: SCALAR moments (bar pairs) ──────────────────────────────────────
scalar_moms = [mom for mom in m.list_of_moments
               if np.array(getattr(m, mom)).size == 1]
print(f"Scalar moments: {scalar_moms}")
n_scalar = len(scalar_moms)
ncols = 5
nrows = int(np.ceil(n_scalar / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2*nrows))
fig.suptitle('Scalar moments — target (blue) vs model (red)',
             fontsize=12, weight='bold')
axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
for i, mom in enumerate(scalar_moms):
    bar_pair_panel(axes_flat[i], mom,
                   getattr(m, mom + '_target'), getattr(m, mom))
for i in range(n_scalar, nrows*ncols):
    axes_flat[i].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 3: VECTOR-12 moments (RD, RP, SRGDP) ───────────────────────────────
vec_moms = [(mom, COUNTRIES) for mom in m.list_of_moments
            if np.array(getattr(m, mom)).shape == (12,)]
print(f"Vector-12 moments: {[v[0] for v in vec_moms]}")
n_vec = len(vec_moms)
fig, axes = plt.subplots(1, n_vec, figsize=(5.5*n_vec, 5.5))
fig.suptitle('Country-level moments — model vs target',
             fontsize=12, weight='bold')
if n_vec == 1: axes = [axes]
for ax, (mom, labels) in zip(axes, vec_moms):
    # RD has 7 dropped entries (CHN, BRA, IND, RUS, MEX, ZAF, ROW — indices 3,4,5,8,9,10,11)
    exclude_mask = None
    excl_lbl = None
    if mom == 'RD':
        exclude_mask = np.array([False, False, False, True, True, True,
                                 False, False, True, True, True, True])
        excl_lbl = '7 dropped from residual'
    scatter_panel(ax, getattr(m, mom + '_target'), getattr(m, mom), mom,
                  labels=labels, log=True, n_outliers=3,
                  exclude_mask=exclude_mask, exclude_label=excl_lbl)
    if exclude_mask is not None:
        ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 4: BILATERAL — SPFLOW (12 x 11) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
# Build (dest, origin) labels — SPFLOW is (12, 11), with origin axis excluding dest
spflow_tgt = np.array(m.SPFLOW_target)
spflow_mod = np.array(m.SPFLOW)
labels_sp = []
for n in range(N):
    for i in range(N):
        if i == n: continue
        labels_sp.append(f'{COUNTRIES[n]}<-{COUNTRIES[i]}')
# But SPFLOW is shape (N, N-1) where 2nd axis is in-order origins != n.
# We need to map (n, j) where j in 0..N-2 -> the right origin label.
scatter_panel(ax, spflow_tgt, spflow_mod,
              'SPFLOW — 132 bilateral patent flows (dest <- origin)',
              labels=labels_sp, log=True, n_outliers=8)
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 5: BILATERAL — FDI_FLOW (12 x 12) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
fdi_tgt = m._FDI_FLOW_target_original.copy()  # pristine (pre-mask) target
fdi_mod = m.FDI_FLOW.copy()
# Recompute the raw model FDI_FLOW (since the saved one has 1.0 on masked cells)
X_M_F_s1 = var.X_M_F[..., 1]
X_M_F_sum_n = X_M_F_s1.sum(axis=1)
X_nn = np.einsum('nns->n', var.X[..., 1:2]).squeeze()
denom = X_nn - X_M_F_sum_n
safe_denom = np.where(denom > 0, denom, 1.0)
fdi_mod_raw = np.where(denom[:, None] > 0,
                        X_M_F_s1 / safe_denom[:, None], 1e-6)
fdi_mod_raw = np.maximum(fdi_mod_raw, 1e-6)
np.fill_diagonal(fdi_mod_raw, 1.0)
fdi_mod = fdi_mod_raw

# Build the (n, i) labels and the exclusion mask
labels_fdi = []
exclude = np.zeros((N, N), bool)
for n in range(N):
    for i in range(N):
        labels_fdi.append(f'{COUNTRIES[n]}<-{COUNTRIES[i]}')
        if n == i:
            exclude[n, i] = True
        elif m._FDI_FLOW_raw_data[n, i] == 0:
            exclude[n, i] = True

scatter_panel(ax, fdi_tgt, fdi_mod,
              'FDI_FLOW — 132 bilateral FDI ratios (dest <- origin)\n'
              '(grey: 12 diagonal + 12 data-zero cells, not in residual)',
              labels=labels_fdi, log=True, n_outliers=8,
              exclude_mask=exclude, exclude_label='excluded (24 cells)')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 6: FDI importance — data vs model aggregates ──────────────────────
# Compute data-side aggregates from m.fdi_matrix (raw AAMNE) and trade_flows
# fdi_matrix is in p.unit; trade_flows are in p.unit too (per load_data).

print("\nComputing FDI importance metrics for data vs model...")
fdi_data = m.fdi_matrix.copy()                              # (N, N), dest<-origin
X_data = m.ccs_moments.trade.values.reshape(N, N, p.S) / p.unit  # (N, N, S)
X_M_F_model = var.X_M_F[..., 1]                              # (N, N)
X_model = var.X[..., 1]                                      # (N, N) — total flows
X_M_O_model = var.X_M[..., 1]                                # (N, N) — exports

# Metric A: bilateral FDI receipts by destination (sum over origins)
fdi_in_data = fdi_data.sum(axis=1)                # (N,)
fdi_in_model = X_M_F_model.sum(axis=1)            # (N,)

# Metric B: bilateral FDI sent by origin (sum over destinations)
fdi_out_data = fdi_data.sum(axis=0)               # (N,)
fdi_out_model = X_M_F_model.sum(axis=0)           # (N,)

# Metric C: FDI as % of net domestic absorption, by destination
#   Data: sum_i fdi_data[n, i] / (X_data[n, n, 1] - sum_i fdi_data[n, i])
denom_data = np.einsum('nns->n', X_data[..., 1:2]).squeeze() - fdi_in_data
denom_data = np.where(denom_data > 0, denom_data, 1e-30)
fdi_share_in_data = fdi_in_data / denom_data
denom_mod = X_nn - X_M_F_sum_n
denom_mod = np.where(denom_mod > 0, denom_mod, 1e-30)
fdi_share_in_model = fdi_in_model / denom_mod

# Metric D: FDI as % of country's monopolistic ABSORPTION (= FDI + exports)
#   data: sum_i fdi_data[n,i] / (sum_i fdi_data[n,i] + sum_i X^{M,O,data}_{n,i})
#   But we don't have X^{M,O,data}, so use total sector-1 trade as proxy
#   model: X_M_F_sum_n / (X_M_F_sum_n + X_M_O_sum_n)
X_M_O_sum_n = X_M_O_model.sum(axis=1)
fdi_in_share_model = fdi_in_model / (fdi_in_model + X_M_O_sum_n)
# For data: use trade flows minus the diagonal
trade_in_data = X_data[..., 1].sum(axis=1) - np.einsum('nns->n', X_data[..., 1:2]).squeeze()
fdi_in_share_data = fdi_in_data / (fdi_in_data + trade_in_data)

# Metric E: net affiliate revenue by country (outward - inward)
sigma1 = float(p.sigma[1])
net_fdi_data = (fdi_out_data - fdi_in_data) / sigma1
net_fdi_model = (fdi_out_model - fdi_in_model) / sigma1

# Build 4-panel page
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('FDI importance — data vs model aggregates (scatter view)',
             fontsize=12, weight='bold')

# Panel A: FDI receipts by destination (log-log)
scatter_panel(axes[0, 0], fdi_in_data, fdi_in_model,
              'A. Inward FDI by destination — Σ_i FDI_{ni}\n'
              '(raw AAMNE units; model X^{M,F}_{ni} sums)',
              labels=COUNTRIES, log=True, n_outliers=3)

# Panel B: FDI sent by origin
scatter_panel(axes[0, 1], fdi_out_data, fdi_out_model,
              'B. Outward FDI by origin — Σ_n FDI_{ni}\n'
              '(raw AAMNE units; model X^{M,F}_{ni} sums)',
              labels=COUNTRIES, log=True, n_outliers=3)

# Panel C: FDI as share of FDI + sectoral trade inward
scatter_panel(axes[1, 0], fdi_in_share_data, fdi_in_share_model,
              'C. FDI share of inward sector-1 absorption\n'
              'FDI_{n,·} / (FDI_{n,·} + foreign trade_{n,·})',
              labels=COUNTRIES, log=False, n_outliers=3)

# Panel D: net affiliate revenue (linear, signed)
ax = axes[1, 1]
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
# Use linear scale here because values are signed
all_finite = np.concatenate([net_fdi_data, net_fdi_model])
m_ = max(abs(all_finite).max(), 1e-9) * 1.15
ax.fill_between([-m_, m_], [-m_/2, m_/2], [-m_*2, m_*2],
                color='grey', alpha=0.18, zorder=0, label='±2× band')
ax.plot([-m_, m_], [-m_, m_], '--', color='0.45', lw=1, zorder=0)
ax.scatter(net_fdi_data, net_fdi_model, s=55, alpha=0.75, c='steelblue',
           edgecolor='k', linewidth=0.4, zorder=2)
resid = np.abs(net_fdi_model - net_fdi_data)
for k in np.argsort(-resid)[:4]:
    ax.scatter(net_fdi_data[k], net_fdi_model[k], s=125, facecolor='none',
               edgecolor='red', linewidth=1.4, zorder=3)
    ax.annotate(COUNTRIES[k], (net_fdi_data[k], net_fdi_model[k]),
                xytext=(7, -2), textcoords='offset points',
                fontsize=8, color='darkred')
for k in range(N):
    if resid[k] < np.sort(resid)[-4]:
        ax.annotate(COUNTRIES[k], (net_fdi_data[k], net_fdi_model[k]),
                    xytext=(5, -2), textcoords='offset points',
                    fontsize=7, color='0.4')
ax.set_xlim(-m_, m_); ax.set_ylim(-m_, m_)
ax.set_xlabel('target: net affiliate revenue (outward−inward, data)')
ax.set_ylabel('model')
ax.set_title('D. Net affiliate revenue by country\n'
             '(green quadrants: sign agrees; red quadrants: disagrees)',
             fontsize=10)
ax.grid(True, alpha=0.25)
# Color background to indicate quadrants
ax.axvspan(0, m_, ymin=0.5, ymax=1.0, color='lightgreen', alpha=0.1, zorder=-1)
ax.axvspan(-m_, 0, ymin=0, ymax=0.5, color='lightgreen', alpha=0.1, zorder=-1)
ax.axvspan(0, m_, ymin=0, ymax=0.5, color='lightcoral', alpha=0.1, zorder=-1)
ax.axvspan(-m_, 0, ymin=0.5, ymax=1.0, color='lightcoral', alpha=0.1, zorder=-1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ── Page 7: FDI bilateral — by-corridor scatter (alternative view) ─────────
# This panel shows ALL 144 cells of FDI_FLOW with detailed labels for the
# top 12 corridors AND the worst-fit corridors.
fig, ax = plt.subplots(figsize=(11, 9))
scatter_panel(ax, fdi_tgt, fdi_mod,
              'FDI_FLOW — all 144 cells, top 12 corridors & worst-fit '
              'highlighted',
              labels=labels_fdi, log=True, n_outliers=12,
              exclude_mask=exclude, exclude_label='diagonal + data-zero')
# Also highlight biggest targets
tgt_flat = fdi_tgt.ravel()
included = ~exclude.ravel()
big_idx = np.argsort(-np.where(included, tgt_flat, -np.inf))[:8]
for k in big_idx:
    n, i = k // N, k % N
    ax.scatter(fdi_tgt[n, i], fdi_mod[n, i], s=140, facecolor='none',
               edgecolor='darkgreen', linewidth=1.5, zorder=3)
    ax.annotate(labels_fdi[k], (fdi_tgt[n, i], fdi_mod[n, i]),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=8, color='darkgreen',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='darkgreen'))
ax.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markeredgecolor='k', markersize=8, label='cell in residual'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
           markeredgecolor='grey', markersize=8, label='excluded (24 cells)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='darkgreen', markersize=10, label='top-8 corridors'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='red', markersize=10, label='12 worst-fit')
], fontsize=9, loc='lower right')
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


pdf.close()
print(f"\n[OK] wrote {OUT}  ({time.perf_counter()-t0:.1f}s)")
