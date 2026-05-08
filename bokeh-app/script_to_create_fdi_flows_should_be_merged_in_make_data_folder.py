#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:48:39 2026

@author: slepot
"""

"""
transform_aamne_to_fdi.py
--------------------------
Transforms AAMNE bilateral FDI data into the long-format used by
fdi_longformat_2015_South_imputed.csv.

Usage:
    python transform_aamne_to_fdi.py

Inputs (all expected in ./data/):
    AAMNE_bilateral_output.csv
    crosswalk_sectors_OECD.csv
    countries_wdi.csv

Output:
    ./output/fdi_longformat_2015_AAMNE.csv
"""

import pandas as pd
import os

# ── Paths ────────────────────────────────────────────────────────────────────
AAMNE_PATH      = "./data/AAMNE_bilateral_output.csv"
CROSSWALK_PATH  = "./data/crosswalk_sectors_OECD.csv"
COUNTRIES_PATH  = "./data/countries_wdi.csv"
OUTPUT_PATH     = "./output/fdi_longformat_2015_AAMNE.csv"
TARGET_YEAR     = 2015

os.makedirs("./output", exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
aamne     = pd.read_csv(AAMNE_PATH)
crosswalk = pd.read_csv(CROSSWALK_PATH)
countries = pd.read_csv(COUNTRIES_PATH)

# ── 2. Filter AAMNE to year 2015 ─────────────────────────────────────────────
aamne = aamne[aamne["year"] == TARGET_YEAR].copy()
print(f"  AAMNE rows for {TARGET_YEAR}: {len(aamne)}")

# ── 3. Build sector filter from crosswalk ────────────────────────────────────
# Crosswalk codes look like "D10T12"; AAMNE isic codes look like "C10T12".
# Strip the leading letter and compare the numeric part.
crosswalk["isic_num"] = crosswalk["Code"].str[1:]           # e.g. "10T12"
kept_isic_nums = set(crosswalk.loc[crosswalk["Sectors"] == 1, "isic_num"])

aamne["isic_num"] = aamne["isic"].str[1:]                   # strip section letter
aamne_filtered = aamne[aamne["isic_num"].isin(kept_isic_nums)].copy()

print(f"  Sectors tagged 1 in crosswalk : {len(kept_isic_nums)}")
print(f"  AAMNE rows after sector filter : {len(aamne_filtered)}")
print(f"  Matched ISIC codes             : {sorted(aamne_filtered['isic'].unique())}")

# ── 4. Build ISO3 → ccode12 mapping ─────────────────────────────────────────
# ccode12 == 100  →  aggregated into ROW (12)
# TWN is not in crosswalk  →  also ROW
iso_to_ccode = dict(zip(countries["countrycode"], countries["ccode12"]))

# ROW ccode will be 12
ROW_CCODE = 12

def map_ccode(iso3):
    """Return ccode12 for known countries, ROW_CCODE for everything else."""
    c = iso_to_ccode.get(iso3, 100)
    return c if c not in (100, 999) else ROW_CCODE

# ── 5. Melt AAMNE to long format ─────────────────────────────────────────────
# Destination countries are all columns except year, cou, isic, isic_num
dest_cols = [c for c in aamne_filtered.columns
             if c not in ("year", "cou", "isic", "isic_num")]

aamne_long = aamne_filtered.melt(
    id_vars=["year", "cou", "isic"],
    value_vars=dest_cols,
    var_name="dest_iso3",
    value_name="flow"
)

# ── 6. Map ISO3 → ccode12 for both source and destination ────────────────────
aamne_long["File_ccode"] = aamne_long["cou"].apply(map_ccode)
aamne_long["Rep_ccode"]  = aamne_long["dest_iso3"].apply(map_ccode)

# ── 7. Aggregate: sum flows across sectors AND across grouped countries ───────
# This collapses:
#   - multiple sectors → single total per country-pair
#   - multiple ISO3s inside the same ccode group → single aggregate
agg = (
    aamne_long
    .groupby(["File_ccode", "Rep_ccode"], as_index=False)["flow"]
    .sum()
)

# ── 8. Build ROW rows ─────────────────────────────────────────────────────────
# ROW as source: for every named destination (ccode 1-11), sum flows from all
#                source countries that mapped to ROW_CCODE.
# ROW as destination: for every named source (ccode 1-11), sum flows to all
#                     destination countries that mapped to ROW_CCODE.
# ROW→ROW: sum of flows where both source and destination are ROW.
# These are already captured in agg once ROW_CCODE is used consistently,
# so no extra step is needed — ROW appears as File_ccode=12 and Rep_ccode=12
# in agg naturally.

# ── 9. Build country-name lookup (ccode → display name) ──────────────────────
# Derive from the target template names observed in the reference file.
ccode_names = {
    1:  "United States",
    2:  "Europe",
    3:  "Japan",
    4:  "China",
    5:  "Brazil",
    6:  "India",
    7:  "Canada",
    8:  "Korea, Republic of",
    9:  "Russian Federation",
    10: "Mexico",
    11: "South Africa",
    12: "Rest of World",
}

agg["FileCountry"] = agg["File_ccode"].map(ccode_names)
agg["RepCountry"]  = agg["Rep_ccode"].map(ccode_names)
agg["Year"]        = TARGET_YEAR
agg.rename(columns={"flow": "FileToRep_Flow"}, inplace=True)

# ── 10. Final column order, drop self-pairs optionally ───────────────────────
output = agg[["File_ccode", "Rep_ccode", "FileCountry", "RepCountry",
              "Year", "FileToRep_Flow"]].copy()

# Sort for readability
output.sort_values(["File_ccode", "Rep_ccode"], inplace=True)
output.reset_index(drop=True, inplace=True)

# ── 11. Save ──────────────────────────────────────────────────────────────────
output.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Output written to: {OUTPUT_PATH}")
print(f"Output shape: {output.shape}")
print()
print(output.to_string())