#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Country-indexed parameter sheet for the old-fee multi-sector pre-TRIPS lever
exercise (baseline_2002_variations/pre_trips_tries/), with 2015 seeds 8.0/8.8.

Rows = countries. Columns = MultiIndex (run [explicit name], quantity), where
quantity is the country-varying parameters: eta, delta, T by sector, and r_hjort.
(Sector-only / scalar params fe, fo, nu, k, sigma, ... are not country-indexed
and are reported in the run x parameter sheet instead.)
Run from inside bokeh-app/.
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd
from classes import parameters

VARDIR = 'calibration_results_matched_economy/baseline_2002_variations/'
ROOT = VARDIR + 'pre_trips_tries/'
RUNS = [('seed_8.0', VARDIR + '8.0/'), ('seed_8.8', VARDIR + '8.8/')] + \
       [(c, ROOT + c + '/') for c in
        ['s_base', 's_fefo', 's_nu', 's_all', 'c_base', 'c_fefo', 'c_nu', 'c_all',
         'cd_base', 'cd_fefo', 'cd_nu', 'cd_all']]
NAMES = {
    'seed_8.0': 'Seed 2015 - separate delta (8.0)',
    'seed_8.8': 'Seed 2015 - common delta (8.8)',
    's_base': 'Separate-delta base',
    's_fefo': 'Separate-delta + calibrate fe,fo in 1992',
    's_nu': 'Separate-delta + target turnover & calibrate nu',
    's_all': 'Separate-delta + calibrate fe,fo + target turnover & nu',
    'c_base': 'Common-delta (equal 2015 & 1992) base',
    'c_fefo': 'Common-delta (equal 2015 & 1992) + calibrate fe,fo in 1992',
    'c_nu': 'Common-delta (equal 2015 & 1992) + target turnover & calibrate nu',
    'c_all': 'Common-delta (equal 2015 & 1992) + calibrate fe,fo + target turnover & nu',
    'cd_base': 'Common in 2015 / differentiated in 1992 base',
    'cd_fefo': 'Common in 2015 / differentiated in 1992 + calibrate fe,fo in 1992',
    'cd_nu': 'Common in 2015 / differentiated in 1992 + target turnover & calibrate nu',
    'cd_all': 'Common in 2015 / differentiated in 1992 + calibrate fe,fo + target turnover & nu',
}
COUNTRY_SECTOR_PARAMS = ['eta', 'delta', 'T']    # country x sector
COUNTRY_PARAMS = ['r_hjort']                      # country only


def country_quantities(p):
    q = {}
    for prm in COUNTRY_SECTOR_PARAMS:
        arr = np.asarray(getattr(p, prm), dtype=float).reshape(p.N, p.S)
        for si, s in enumerate(p.sectors):
            q[f'{prm}|{s}'] = arr[:, si]
    for prm in COUNTRY_PARAMS:
        q[prm] = np.asarray(getattr(p, prm), dtype=float).ravel()
    return q


data = {}
countries = None
for code, folder in RUNS:
    p = parameters(); p.correct_eur_patent_cost = True; p.load_run(folder)
    countries = list(p.countries)
    for ql, arr in country_quantities(p).items():
        data[(NAMES[code], ql)] = arr

tab = pd.DataFrame(data, index=countries)
tab.columns = pd.MultiIndex.from_tuples(tab.columns, names=['run', 'quantity'])
tab.index.name = 'country'

out = ROOT + 'pre_trips_tries_all_params.csv'
tab.to_csv(out)
print(f"saved {out}  ({tab.shape[0]} countries x {tab.shape[1]} run-quantity columns)")
print('quantities per run:', list(dict.fromkeys(q for _, q in tab.columns)))
