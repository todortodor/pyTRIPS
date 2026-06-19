import pandas as pd, numpy as np
co = ['USA','EUR','JAP','CHN','BRA','IND','CAN','KOR','RUS','MEX','ZAF','ROW']
g = lambda x: float(f'{x:.3g}') if pd.notna(x) else np.nan
mono = pd.read_csv('calibration_results_matched_economy/baseline_2000_variations/pre_trips_tries_mono_newfee/pre_trips_tries_mono_comparison.csv')
multi = pd.read_csv('calibration_results_matched_economy/baseline_2002_variations/pre_trips_tries_newfee/pre_trips_tries_comparison.csv')

names = {
 'MONO_base':'Mono base','MONO_fefo':'Mono + calibrate fe,fo in 1992',
 'MONO_nu':'Mono + target turnover & calibrate nu','MONO_all':'Mono + calibrate fe,fo + target turnover & nu',
 's_base':'Separate-delta base','s_fefo':'Separate-delta + calibrate fe,fo in 1992',
 's_nu':'Separate-delta + target turnover & calibrate nu','s_all':'Separate-delta + calibrate fe,fo + target turnover & nu',
 'c_base':'Common-delta (equal 2015 & 1992) base','c_fefo':'Common-delta (equal 2015 & 1992) + calibrate fe,fo in 1992',
 'c_nu':'Common-delta (equal 2015 & 1992) + target turnover & calibrate nu','c_all':'Common-delta (equal 2015 & 1992) + calibrate fe,fo + target turnover & nu',
 'cd_base':'Common in 2015 / differentiated in 1992 base','cd_fefo':'Common in 2015 / differentiated in 1992 + calibrate fe,fo in 1992',
 'cd_nu':'Common in 2015 / differentiated in 1992 + target turnover & calibrate nu','cd_all':'Common in 2015 / differentiated in 1992 + calibrate fe,fo + target turnover & nu',
}
blocks, nu_rest, nu_pharma = {}, {}, {}

def add(name, restcol92, restcol15, ph92, ph15, df, nr92, nr15, np92, np15):
    blocks[name] = pd.DataFrame({
        'delta_rest_1992':  df[restcol92].reindex(co).map(g),
        'delta_rest_2015':  df[restcol15].reindex(co).map(g),
        'delta_pharma_1992': (df[ph92].reindex(co).map(g) if ph92 else pd.Series(np.nan, index=co)),
        'delta_pharma_2015': (df[ph15].reindex(co).map(g) if ph15 else pd.Series(np.nan, index=co)),
    }, index=co)
    nu_rest[name] = (g(nr92), g(nr15))
    nu_pharma[name] = (g(np92), g(np15))

mono_seed_nu = mono[mono.config == 'base']['nu_patent'].iloc[0]
for cfg in ['base','fefo','nu','all']:
    s = mono[mono.config == cfg].set_index('country')
    add(names[f'MONO_{cfg}'], 'delta_patent_1992','delta_patent_2015', None, None, s,
        s['nu_patent'].iloc[0], mono_seed_nu, np.nan, np.nan)

seed_nu = {fam: (multi[multi.config == f'{fam}_base']['nu_rest'].iloc[0],
                 multi[multi.config == f'{fam}_base']['nu_pharma'].iloc[0]) for fam in ['s','c','cd']}
for cfg in ['s_base','s_fefo','s_nu','s_all','c_base','c_fefo','c_nu','c_all','cd_base','cd_fefo','cd_nu','cd_all']:
    fam = cfg.split('_')[0]; s = multi[multi.config == cfg].set_index('country')
    add(names[cfg], 'delta_rest_1992','delta_rest_2015','delta_pharma_1992','delta_pharma_2015', s,
        s['nu_rest'].iloc[0], seed_nu[fam][0], s['nu_pharma'].iloc[0], seed_nu[fam][1])

tab = pd.concat(blocks, axis=1)
tab.columns.names = ['try', 'delta']
# nu rows: nu_rest under the rest delta fields (1992 calibrated, 2015 seed); nu_pharma under pharma fields
nr = {(nm,'delta_rest_1992'): nu_rest[nm][0] for nm in blocks}
nr.update({(nm,'delta_rest_2015'): nu_rest[nm][1] for nm in blocks})
npx = {(nm,'delta_pharma_1992'): nu_pharma[nm][0] for nm in blocks}
npx.update({(nm,'delta_pharma_2015'): nu_pharma[nm][1] for nm in blocks})
tab.loc['nu_rest'] = pd.Series(nr).reindex(tab.columns)
tab.loc['nu_pharma'] = pd.Series(npx).reindex(tab.columns)

out = 'calibration_results_matched_economy/pre_trips_tries_deltas_newfee.csv'
tab.to_csv(out)
print('saved', out, 'shape', tab.shape)
print('rows:', list(tab.index))
print('configs:', [c for c in tab.columns.get_level_values(0).unique()])
