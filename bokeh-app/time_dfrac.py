import sys, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np

from classes import parameters, moments, history
from solver_funcs import calibration_func_with_fdi

p = parameters()
p.correct_eur_patent_cost = True
p.load_run('calibration_results_matched_economy/baseline_2000_variations/15.0/')
m = moments()
m.load_run('calibration_results_matched_economy/baseline_2000_variations/11.02/')

m.drop_CHN_IND_BRA_ROW_from_RD = True
m.list_of_moments.append('FDI_FLOW_N')
m.list_of_moments.append('FDI_ELAST')
p.calib_parameters.append('a')
p.calib_parameters.append('d_frac')        # NEW: was 'd'
p.calib_parameters.append('power_fdi')
p.a = 0.1
p.d_frac = np.float64(0.6)                 # corresponds to d ~ 0.15 with k[1]~1.25
p.power_fdi = 1.0
p.guess = None

hist = history(*tuple(m.list_of_moments+['objective']))

# Call once
print("Calling calibration_func_with_fdi with d_frac...")
t0 = time.perf_counter()
dev = calibration_func_with_fdi(p.make_p_vector(), p, m, None, hist, t0)
print(f"Took {time.perf_counter()-t0:.1f}s")
print(f"Deviation vector: shape={dev.shape}, ||dev||={np.linalg.norm(dev):.4f}")
print(f"Any nan: {np.isnan(dev).any()}, any inf: {np.isinf(dev).any()}")
print()
print(f"Final d:      {p.d:.6f}")
print(f"Final d_frac: {p.d_frac:.6f}")
print(f"k[1]:         {p.k[1]:.6f}")
print(f"k[1] - 1 - 1e-6 = {p.k[1] - 1 - 1e-6:.6f}")
print(f"d_frac * (k[1] - 1 - 1e-6) = {p.d_frac * (p.k[1] - 1 - 1e-6):.6f}  (should == d)")
print()
# Test bounds
lb, ub = p.make_parameters_bounds()
print(f"d_frac bound: lb[-2]={lb[-2]}, ub[-2]={ub[-2]}  (should be near 0 and 1)")
