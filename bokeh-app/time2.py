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
p.calib_parameters.append('d')
p.calib_parameters.append('power_fdi')
p.a = 0.1; p.d = 0.15; p.power_fdi = 1.0
p.guess = None

hist = history(*tuple(m.list_of_moments+['objective']))

# Call once
print("Calling calibration_func_with_fdi once...")
t0 = time.perf_counter()
try:
    dev = calibration_func_with_fdi(p.make_p_vector(), p, m, None, hist, t0)
    t = time.perf_counter() - t0
    print(f"Took {t:.1f}s")
    print(f"Deviation vector: shape={dev.shape}, ||dev||={np.linalg.norm(dev):.4f}")
    print(f"Any nan: {np.isnan(dev).any()}, any inf: {np.isinf(dev).any()}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
