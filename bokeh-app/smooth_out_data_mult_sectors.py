#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-sector (3-sector) analog of smooth_out_data.py.

Builds data_smooth_3_years/data_{N}_countries_3_sectors_{year}/ by averaging the
per-year 3-sector data folders over a centered 3-year window (e.g. 1992 uses
1991, 1992, 1993). This is the apples-to-apples counterpart of the smoothed
data the single-sector pre-TRIPS calibration uses, which the multi-sector
pre-TRIPS run was missing (it had to use raw single-year 1992 data, inflating
the delta levels).

Same averaging as the 2-sector script for every file, EXCEPT
country_sector_moments.csv (the RD-ratio file), which has missing/blank country
rows in early years; that one is averaged NaN-aware (mean over available years
per cell) so a missing year doesn't wipe out a country. Countries whose sector
R&D simply doesn't exist in the window (CAN pre-1994, KOR pre-1995) stay NaN and
are handled by the calibration script's earliest-ANBERD backfill.

Run from inside bokeh-app/.
"""
import os
import pandas as pd

DELTA_T          = 3
SECTORS_SUFFIX   = '_3_sectors'
NBRS_COUNTRIES   = [12]
YEARS            = list(range(1990, 2019))
SMOOTH_DATA_PATH = f'data_smooth_{DELTA_T}_years/'

os.makedirs(SMOOTH_DATA_PATH, exist_ok=True)

# centered window of width DELTA_T around each year (identical logic to the
# 2-sector smooth_out_data.py)
years_for_smoothing = {}
for y in YEARS:
    years_for_smoothing[y] = [y]
    i = 1
    while (DELTA_T / 2) - i > 0:
        if y - i >= YEARS[0]:
            years_for_smoothing[y].insert(0, y - i)
        if y + i <= YEARS[-1]:
            years_for_smoothing[y].append(y + i)
        i += 1


def folder(nbr_countries, year):
    return f'data/data_{nbr_countries}_countries{SECTORS_SUFFIX}_{year}/'


def average_additive(year, name, index_col, years_smoothing, nbr_countries):
    """Plain mean across the window (assumes aligned indices) — matches the
    original 2-sector script exactly."""
    s = pd.read_csv(folder(nbr_countries, year) + name, index_col=index_col)
    for y in years_smoothing:
        if y != year:
            s = s + pd.read_csv(folder(nbr_countries, y) + name, index_col=index_col)
    return s / len(years_smoothing)


def average_skipna(year, name, index_col, years_smoothing, nbr_countries):
    """NaN-aware mean across the window (for country_sector_moments, which has
    missing country rows in some years)."""
    dfs = [pd.read_csv(folder(nbr_countries, y) + name, index_col=index_col)
           for y in years_smoothing]
    return pd.concat(dfs).groupby(level=list(range(pd.concat(dfs).index.nlevels))).mean()


DATA_TO_AVERAGE = [
    dict(name='country_country_moments.csv',        index_col=[0, 1],    fn=average_additive),
    dict(name='country_country_sector_moments.csv', index_col=[0, 1, 2], fn=average_additive),
    dict(name='tariff.csv',                          index_col=[0, 1, 2], fn=average_additive),
    dict(name='country_moments.csv',                 index_col=[0],       fn=average_additive),
    dict(name='final_pat_fees.csv',                  index_col=[0, 1],    fn=average_additive),
    dict(name='scalar_moments.csv',                  index_col=[0],       fn=average_additive),
    dict(name='sector_moments.csv',                  index_col=[0],       fn=average_additive),
    dict(name='country_sector_moments.csv',          index_col=[0],       fn=average_skipna),
]

for y in YEARS:
    for nbr_countries in NBRS_COUNTRIES:
        src = folder(nbr_countries, y)
        if not os.path.isdir(src):
            continue
        path = SMOOTH_DATA_PATH + f'data_{nbr_countries}_countries{SECTORS_SUFFIX}_{y}/'
        os.makedirs(path, exist_ok=True)
        print(y, nbr_countries, '<-', years_for_smoothing[y])

        md = pd.read_csv(src + 'moments_descriptions.csv', sep=';', index_col=0)
        md.to_csv(path + 'moments_descriptions.csv', sep=';')

        for item in DATA_TO_AVERAGE:
            try:
                out = item['fn'](year=y, name=item['name'], index_col=item['index_col'],
                                 years_smoothing=years_for_smoothing[y],
                                 nbr_countries=nbr_countries)
                out.to_csv(path + item['name'])
            except Exception as e:
                print(f"  WARN {item['name']}: {e}")
print('done')
