"""
Real-world problems for the DM-ABC study (import by dmabc_benchmark.py, --suite real).

PV parameter extraction (residual-form RMSE, standard in the PV literature)
  RTC France cell (26 points, 33 C) and Photowatt-PWP201 module (36 cells, 25 points, 45 C);
  data cross-checked in two public repositories; this implementation reproduces the
  best-known RMSE values 9.8602e-4 (RTC SDM), 9.8248e-4 (RTC DDM) and 2.4251e-3 (PWP SDM).
  x = [Iph, Rs, Rsh, I01, n1, (I02, n2), (I03, n3)]
CEC2011 problems (Das & Suganthan, 2010 technical report)
  P1 FM sound wave estimation (D=6) and P2 Lennard-Jones, 10 atoms (D=30): values and bounds
  match the minionpy port to machine precision; LJ is reported as f - (-28.422532).
  P7 radar polyphase code design (D=20): f = max(phi, -phi) as written in the report;
  minionpy omits the -phi terms, so values can differ from that port.
"""
import numpy as np
# ---- CEC2011 P1: FM sound wave parameter estimation (D=6) ----
_t = np.arange(101); _th = 2*np.pi/100
def _fmwave(x):
    a1,w1,a2,w2,a3,w3 = x
    return a1*np.sin(w1*_t*_th + a2*np.sin(w2*_t*_th + a3*np.sin(w3*_t*_th)))
_y0 = _fmwave(np.array([1.0,5.0,-1.5,4.8,2.0,4.9]))
def fm(x): return float(np.sum((_fmwave(x)-_y0)**2))
# ---- CEC2011 P2: Lennard-Jones, 10 atoms (D=30), V = r^-12 - 2 r^-6 ----
_iu = np.triu_indices(10, 1)
def lj(x):
    p = x.reshape(10,3)
    d = p[:,None,:]-p[None,:,:]
    r2 = np.sum(d*d, axis=2)[_iu]
    r2 = np.maximum(r2, 1e-12)
    ir6 = 1.0/r2**3
    return float(np.sum(ir6*ir6 - 2*ir6))
# ---- CEC2011 P7: spread spectrum radar polyphase code design (D=20) ----
# f = max(phi_1..phi_m, -phi_1..-phi_m), m = 2D-1, as defined in the CEC2011 technical report.
# (minionpy's implementation omits the -phi terms; we follow the report.)
def _radar_index(D):
    rows, lo, hi, const = [], [], [], []
    r = 0
    for i in range(1, D + 1):            # phi_{2i-1}
        for j in range(i, D + 1):
            rows.append(r); lo.append(abs(2 * i - j - 1)); hi.append(j)
        const.append(0.0); r += 1
    for i in range(1, D):                # phi_{2i}
        for j in range(i + 1, D + 1):
            rows.append(r); lo.append(abs(2 * i - j)); hi.append(j)
        const.append(0.5); r += 1
    return np.array(rows), np.array(lo), np.array(hi), np.array(const)
_RADAR = _radar_index(20)
def radar(x):
    rows, lo, hi, const = _RADAR
    c = np.concatenate([[0.0], np.cumsum(x)])
    phi = np.bincount(rows, weights=np.cos(c[hi] - c[lo]), minlength=const.size) + const
    return float(max(phi.max(), -phi.min()))

# ---- PV parameter extraction (residual-form RMSE, the common literature convention) ----
_K, _Q = 1.380649e-23, 1.602176634e-19
RTC_V = np.array([-0.2057,-0.1291,-0.0588,0.0057,0.0646,0.1185,0.1678,0.2132,0.2545,0.2924,
                  0.3269,0.3585,0.3873,0.4137,0.4373,0.4590,0.4784,0.4960,0.5119,0.5265,
                  0.5398,0.5521,0.5633,0.5736,0.5833,0.5900])
RTC_I = np.array([0.7640,0.7620,0.7605,0.7605,0.7600,0.7590,0.7570,0.7570,0.7555,0.7540,
                  0.7505,0.7465,0.7385,0.7280,0.7065,0.6755,0.6320,0.5730,0.4990,0.4130,
                  0.3165,0.2120,0.1035,-0.0100,-0.1230,-0.2100])
PWP_V = np.array([0.1248,1.8093,3.3511,4.7622,6.0538,7.2364,8.3189,9.3097,10.2163,11.0449,
                  11.8018,12.4929,13.1231,13.6983,14.2221,14.6995,15.1346,15.5311,15.8929,
                  16.2229,16.5241,16.7987,17.0499,17.2793,17.4885])
PWP_I = np.array([1.0315,1.0300,1.0260,1.0220,1.0180,1.0155,1.0140,1.0100,1.0035,0.9880,
                  0.9630,0.9255,0.8725,0.8075,0.7265,0.6345,0.5345,0.4275,0.3185,0.2085,
                  0.1010,-0.0080,-0.1110,-0.2090,-0.3030])
PV_DATA = {"RTC": (RTC_V, RTC_I, 306.15, 1), "PWP": (PWP_V, PWP_I, 318.15, 36)}

def pv_rmse(x, data, ndiode):
    """x = [Iph, Rs, Rsh, I01, n1, (I02, n2), (I03, n3)]"""
    V, I, T, Ns = PV_DATA[data]
    Vt = _K * T / _Q * Ns
    Iph, Rs, Rsh = x[0], x[1], max(x[2], 1e-9)   # Rsh lower bound 0 would divide by zero
    Vd = V + I * Rs
    Imodel = Iph - Vd / Rsh
    for k in range(ndiode):
        I0, n = x[3 + 2 * k], x[4 + 2 * k]
        Imodel = Imodel - I0 * np.expm1(np.minimum(Vd / (n * Vt), 100.0))  # cap never binds inside the bounds
    return float(np.sqrt(np.mean((Imodel - I) ** 2)))

def pv_bounds(data, ndiode):
    if data == "RTC":
        lo, hi = [0.0, 0.0, 0.0], [1.0, 0.5, 100.0]
        d_lo, d_hi = [0.0, 1.0], [1e-6, 2.0]
    else:  # PWP201 module, 36 cells in series; n is per cell
        lo, hi = [0.0, 0.0, 0.0], [2.0, 2.0, 2000.0]
        d_lo, d_hi = [0.0, 1.0], [50e-6, 2.0]
    return np.array(lo + d_lo * ndiode), np.array(hi + d_hi * ndiode)


# ---- registry: name -> (objective, lb, ub, offset); reported value = objective - offset ----
LJ10_BEST = -28.422532
def _cec11_lj_bounds():
    # bounds as in the CEC2011 code (cross-checked with the minionpy port)
    lo, hi = [0.0, 0.0, 0.0], [4.0, 4.0, np.pi]
    for pos in range(3, 30):                  # 0-based position
        b = 4.0 + 0.25 * max(0, (pos - 4) // 3)
        lo.append(-b); hi.append(b)
    return np.array(lo), np.array(hi)

def build_real():
    reg = {}
    for data in ("RTC", "PWP"):
        for nd, tag in ((1, "SDM"), (2, "DDM"), (3, "TDM")):
            if data == "PWP" and nd == 3:
                continue
            lo, hi = pv_bounds(data, nd)
            reg[f"PV_{data}_{tag}"] = (lambda x, d=data, n=nd: pv_rmse(x, d, n), lo, hi, 0.0)
    reg["CEC11_FM"] = (fm, np.full(6, -6.4), np.full(6, 6.35), 0.0)
    lo, hi = _cec11_lj_bounds()
    reg["CEC11_LJ10"] = (lj, lo, hi, LJ10_BEST)
    reg["CEC11_Radar20"] = (radar, np.zeros(20), np.full(20, 2 * np.pi), 0.0)
    return reg
REAL = build_real()
REAL_MAIN = list(REAL)          # default set for --suite real (keeps earlier results resumable)

# ---- supplementary: same PV problems with log10-encoded saturation currents ----
# The optimizer searches log10(I0k) in [log10(1e-12), log10(I0_max)] instead of I0k in
# [0, I0_max]; all other parameters, the data and the RMSE are unchanged, so values are
# directly comparable with the linear encoding.
LOG_I0_MIN = 1e-12

def _pv_log(data, nd):
    lo, hi = pv_bounds(data, nd)
    idx = [3 + 2 * k for k in range(nd)]
    lo, hi = lo.copy(), hi.copy()
    lo[idx], hi[idx] = np.log10(LOG_I0_MIN), np.log10(hi[idx])
    def f(x, d=data, n=nd, idx=idx):
        z = np.array(x, float)
        z[idx] = 10.0 ** z[idx]
        return pv_rmse(z, d, n)
    return f, lo, hi

for _name in [k for k in REAL_MAIN if k.startswith("PV_")]:
    _, _data, _tag = _name.split("_")
    _f, _lo, _hi = _pv_log(_data, {"SDM": 1, "DDM": 2, "TDM": 3}[_tag])
    REAL[_name + "_log"] = (_f, _lo, _hi, 0.0)
REAL_LOG = [k for k in REAL if k.endswith("_log")]
