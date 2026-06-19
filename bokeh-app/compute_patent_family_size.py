#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patent family-size statistics for the baseline (2-sector, no FDI) model.

Implements Section 10.5 of RomerEKnewversionwithtariffs21_11_25.tex
("statistics on family size = number of countries covered by a given family").

Model object used: psi_m_star[n, i, s] = the patenting threshold psi^{sm*}_{ni}
(destination n, origin i, sector s); ideas with quality above it are patented
in n. Quality is Pareto(shape k[s]), so (psi^{sm*}_{ni})^{-k} is proportional to
the mass of ideas (from origin i) that clear destination n's threshold.

For a given origin i and the patenting sector, order destinations by threshold
ascending: n(1) smallest, ..., n(N); set the (N+1)-th tail to 0. With
T_j := (psi^{sm*}_{n(j) i})^{-k} (so T_1 >= T_2 >= ... and T_{N+1}=0):

  P(family covers >= j countries | patented somewhere) = T_j / T_1
  P(family size = j)                                   = (T_j - T_{j+1}) / T_1
  E[size]                                              = sum_j T_j / T_1

Three variants (text a/b/c):
  a) all patents (domestic + international): order all destinations.
  b) international only, EXCLUDING the domestic country from the count: drop the
     domestic destination (n=i) from the ordering, then variant-a logic.
  c) international families, INCLUDING the domestic country in the size count.
     Per origin, two cases:
       i)  domestic is patented first (psi^*_{ii} is the smallest threshold):
           distribution starts at size 2, normalised by T_{n(2)}. Equivalently
           variant (b) + 1 for the home patent.
       ii) all patents are international (some foreign threshold < psi^*_{ii}):
           same distribution as (a).

Text notes applied:
  - ROW and India are EXCLUDED, both as origin i and as destination n.
  - EUR is already a single country in the model, so no fractional EPO/GDP
    counting is needed here (that adjustment is data-side only).
  - Patent-family weights are data-side; to aggregate the MODEL statistic across
    origins we weight each origin by its model family count (= pflow at its
    smallest-threshold destination, i.e. the mass of families originating in i).

Run from inside bokeh-app/.
"""
import matplotlib
matplotlib.use('Agg')   # never block on plt.show() in the solver

import numpy as np
import pandas as pd

from classes import parameters, var

BASELINE   = '2000'
SECTOR     = 1                 # patenting sector index (0 = non-patent)
EXCLUDE    = ['IND', 'ROW']    # excluded as both origin and destination (text)
OUT_CSV    = f'patent_family_size_baseline_{BASELINE}.csv'


# ─────────────────────────────────────────────────────────────────────────────
# Load + solve the baseline
# ─────────────────────────────────────────────────────────────────────────────
p = parameters()
p.load_run(f'calibration_results_matched_economy/{BASELINE}/')
sol = var.var_from_vector(p.guess, p, compute=True, context='calibration')
sol.scale_P(p)
sol.compute_price_indices(p)
sol.compute_non_solver_quantities(p)

countries = list(p.countries)
k = float(np.asarray(p.k)[SECTOR])
psi = sol.psi_m_star[:, :, SECTOR]                      # [destination n, origin i]
pflow = np.asarray(sol.pflow)
pflow = pflow[:, :, -1] if pflow.ndim == 3 else pflow   # patenting-sector flows [n, i]

incl = [c for c in countries if c not in EXCLUDE]
idx = {c: countries.index(c) for c in countries}
incl_pos = [idx[c] for c in incl]                       # destination/origin positions kept

print(f"baseline {BASELINE} | patenting sector = '{p.sectors[SECTOR]}' | k = {k:.4f}")
print(f"countries: {countries}")
print(f"excluded (origin & destination): {EXCLUDE}")


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def size_distribution(tk_desc, norm_idx, start_size):
    """Given tk values sorted DESCENDING (T_1>=T_2>=...), return (sizes, probs)
    with P(size=j) = (T_j - T_{j+1}) / T_{norm_idx+1}, j running from start_size.
    A zero tail T_{M+1}=0 is appended."""
    tk = np.concatenate([tk_desc, [0.0]])
    norm = tk[norm_idx]
    sizes = np.arange(start_size, len(tk_desc) + 1)
    probs = (tk[sizes - 1] - tk[sizes]) / norm
    return sizes, probs


def mean_size(sizes, probs):
    return float(np.sum(sizes * probs))


def family_stats_for_origin(i_pos):
    """Return per-origin family-size stats (means + distributions) for the three
    variants, plus the model family-count weights used for aggregation."""
    dest_pos = incl_pos
    t_all = psi[dest_pos, i_pos]                       # thresholds over kept destinations
    tk_all = t_all ** (-k)
    order = np.argsort(t_all)                          # ascending threshold = descending tk
    tk_desc = tk_all[order]
    min_thr = t_all.min()

    # ---- variant a: all patents (incl domestic) ----
    sa, pa = size_distribution(tk_desc, norm_idx=0, start_size=1)

    # ---- variant b: international only, domestic excluded from the ordering ----
    dest_b = [d for d in dest_pos if d != i_pos]
    t_b = psi[dest_b, i_pos]
    tk_b_desc = np.sort(t_b ** (-k))[::-1]
    sb, pb = size_distribution(tk_b_desc, norm_idx=0, start_size=1)

    # ---- variant c: international families, domestic counted in the size ----
    # case i = domestic is (tied) the cheapest threshold -> patented first;
    # case ii = some foreign threshold strictly below domestic.
    domestic_first = np.isclose(psi[i_pos, i_pos], min_thr)
    if domestic_first:                                  # case i  (= variant b + 1)
        sc, pc = size_distribution(tk_desc, norm_idx=1, start_size=2)
        case = 'i (domestic first)'
    else:                                               # case ii (= variant a)
        sc, pc = sa, pa
        case = 'ii (intl first)'

    # model family-count weights (pflow ~ threshold^{-k}, so flow at the cheapest
    # threshold = number of families; foreign-only version for variants b & c)
    n_all = float(pflow[dest_pos, i_pos].max())
    n_intl = float(pflow[dest_b, i_pos].max())

    return {
        'mean_a_all': mean_size(sa, pa),
        'mean_b_intl_excl_dom': mean_size(sb, pb),
        'mean_c_intl_incl_dom': mean_size(sc, pc),
        'P_size1_a': float(pa[0]),                      # share covering only 1 country
        'c_case': case,
        'n_families_all': n_all,
        'n_families_intl': n_intl,
        '_dist_a': (sa, pa),
    }


def weighted_dist(rows, dist_key, weights):
    """Aggregate per-origin (sizes, probs) into one distribution over sizes."""
    maxsize = max(int(r[dist_key][0][-1]) for r in rows.values())
    agg = np.zeros(maxsize)
    for (r, w_i) in zip(rows.values(), weights):
        s, p_ = r[dist_key]
        agg[s - 1] += w_i * p_
    agg /= agg.sum()
    return pd.Series(agg, index=np.arange(1, maxsize + 1), name='prob')


# ─────────────────────────────────────────────────────────────────────────────
# Per-origin table
# ─────────────────────────────────────────────────────────────────────────────
rows = {c: family_stats_for_origin(idx[c]) for c in incl}
tab = pd.DataFrame(rows).T
tab.index.name = 'origin'

means = ['mean_a_all', 'mean_b_intl_excl_dom', 'mean_c_intl_incl_dom']
w_all = tab['n_families_all'].astype(float).values
w_intl = tab['n_families_intl'].astype(float).values
weight_for = {'mean_a_all': w_all,                      # variant a: all families
              'mean_b_intl_excl_dom': w_intl,           # variants b & c: intl families
              'mean_c_intl_incl_dom': w_intl}
agg = pd.DataFrame({
    'unweighted_mean': tab[means].astype(float).mean(),
    'family_weighted_mean': [np.average(tab[m].astype(float).values, weights=weight_for[m])
                             for m in means],
}).T

# aggregate variant-a distribution (weighted by total families per origin)
dist_a = weighted_dist(rows, '_dist_a', w_all)

pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print("\n=== mean family size by origin (patenting sector) ===")
print(tab[means + ['P_size1_a', 'c_case']].to_string())
print("\n=== aggregate mean across origins ===")
print(agg.to_string())
print("\n=== aggregate family-size distribution, variant a (family-weighted) ===")
print(dist_a.round(4).to_string())

out = tab.drop(columns=['_dist_a'])
out.to_csv(OUT_CSV)
dist_a.to_csv(OUT_CSV.replace('.csv', '_distribution_a.csv'))
print(f"\nsaved per-origin table to {OUT_CSV}")
print(f"saved variant-a distribution to {OUT_CSV.replace('.csv', '_distribution_a.csv')}")
print("variants: a = all patents (incl domestic); "
      "b = international, excl domestic; c = international, incl domestic")
