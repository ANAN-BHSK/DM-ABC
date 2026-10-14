"""
DM-ABC: Direction-Memory Artificial Bee Colony, benchmarked against ABC, GABC, PSO, Bat.

Every algorithm gets the same budget of function evaluations (FE).
Usage:
    python dmabc_benchmark.py --quick                         # fast sanity check
    python dmabc_benchmark.py --dim 30 --runs 30 --workers 8  # full experiment
    python dmabc_benchmark.py --algos DM-Onl ABC --runs 30    # subset
    python dmabc_benchmark.py --suite cec2017 --dim 30 --runs 51 --workers 8 --out cec_D30
    python dmabc_benchmark.py --from-raw results/raw_final_errors.csv --out stats  # stats only
Outputs go to ./results/ (or --out)
"""
import argparse, os, time, csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy import stats
try:
    from realprob import REAL, REAL_MAIN   # real-world problem definitions (same folder)
except ImportError:
    REAL, REAL_MAIN = {}, []


# ============================================================
# Benchmark functions (minimization, f* = 0 at x = o)
# ============================================================
class BudgetExhausted(Exception):
    pass


def _rotation(D, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    return q * np.sign(np.diag(r))


def sphere(z):     return np.sum(z ** 2)
def elliptic(z):
    D = z.size
    w = 10 ** (6 * np.arange(D) / max(D - 1, 1))
    return np.sum(w * z ** 2)
def rosenbrock(z):
    z = z + 1.0
    return np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)
def rastrigin(z):  return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
def ackley(z):
    D = z.size
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(z ** 2) / D))
            - np.exp(np.sum(np.cos(2 * np.pi * z)) / D) + 20 + np.e)
def griewank(z):
    i = np.arange(1, z.size + 1)
    return np.sum(z ** 2) / 4000 - np.prod(np.cos(z / np.sqrt(i))) + 1
def schwefel12(z): return np.sum(np.cumsum(z) ** 2)

# name: (base function, bound, rotated?, separable?)
FUNCTIONS = {
    "F1_Sphere":          (sphere,     100.0, False, "S"),
    "F2_Elliptic":        (elliptic,   100.0, False, "S"),
    "F3_Rastrigin":       (rastrigin,  5.12,  False, "S"),
    "F4_Rosenbrock":      (rosenbrock, 30.0,  False, "N"),
    "F5_Ackley":          (ackley,     32.0,  False, "N"),
    "F6_Griewank":        (griewank,   600.0, False, "N"),
    "F7_Schwefel1.2":     (schwefel12, 100.0, False, "N"),
    "F8_RotElliptic":     (elliptic,   100.0, True,  "N"),
    "F9_RotRastrigin":    (rastrigin,  5.12,  True,  "N"),
}


# CEC2017 suite (opfunu provides F1-F29; F2 is excluded as in the official
# competition because of its numerical instability). Valid dims: 10, 30, 50, 100.
# Official CEC2017 numbering (Awad et al., 2016): F1, F3-F30 (F2 removed by the organizers).
# opfunu already drops the official F2 and renumbers the rest, so official Fk (k >= 3) is opfunu F(k-1).
CEC2017 = [f"CEC17_F{i}" for i in range(1, 31) if i != 2]


def _opfunu_index(official):
    if official == 2:
        raise ValueError("CEC2017 F2 was removed from the official suite")
    return official if official == 1 else official - 1
CEC17_NAN_AT_D10 = ["CEC17_F18", "CEC17_F30"]   # official numbering


class Problem:
    """Shifted (and optionally rotated) problem with FE counter and convergence log."""
    def __init__(self, name, D, max_fe, n_ckpt=100, inst_seed=12345):
        self.D, self.max_fe = D, max_fe
        if name in REAL:
            # real-world problems (realprob.py): PV parameter extraction, CEC2011 problems
            f, lo, hi, off = REAL[name]
            self.base = lambda z, _f=f, _off=off: _f(z) - _off
            self.lb, self.ub = np.array(lo, float), np.array(hi, float)
            self.o, self.M = np.zeros(self.lb.size), None
        elif name.startswith("CEC17_"):
            # official CEC2017 definitions (shift, rotation, hybrid, composition) via opfunu;
            # error = f(x) - f*
            from opfunu.cec_based import cec2017
            obj = getattr(cec2017, f"F{_opfunu_index(int(name.split('_F')[1]))}2017")(ndim=D)
            self.base = lambda z, _o=obj: _o.evaluate(z) - _o.f_global
            self.lb = np.asarray(obj.bounds[:, 0], float)
            self.ub = np.asarray(obj.bounds[:, 1], float)
            self.o, self.M = np.zeros(D), None
        else:
            f, b, rot, _ = FUNCTIONS[name]
            self.base = f
            self.lb, self.ub = -b * np.ones(D), b * np.ones(D)
            rng = np.random.default_rng(inst_seed + sum(map(ord, name)))
            self.o = rng.uniform(0.8 * self.lb, 0.8 * self.ub)   # shifted optimum
            self.M = _rotation(D, inst_seed + 7 + sum(map(ord, name))) if rot else None
        self.fe, self.best = 0, np.inf
        self.ckpt = np.linspace(max_fe / n_ckpt, max_fe, n_ckpt).astype(int)
        self.hist, self._k = np.full(n_ckpt, np.nan), 0

    def __call__(self, x):
        if self.fe >= self.max_fe:
            raise BudgetExhausted
        z = x - self.o
        if self.M is not None:
            z = self.M @ z
        val = float(self.base(z))
        if not np.isfinite(val):
            raise ValueError(f"objective returned {val}; check the benchmark definition")
        self.fe += 1
        if val < self.best:
            self.best = val
        while self._k < len(self.ckpt) and self.fe >= self.ckpt[self._k]:
            self.hist[self._k] = self.best
            self._k += 1
        return val

    def finish(self):
        self.hist[self._k:] = self.best


# ============================================================
# Algorithms
# ============================================================
def _abc_fitness(F):
    return np.where(F >= 0, 1.0 / (1.0 + F), 1.0 + np.abs(F))


def abc_family(prob, rng, SN=20, limit=None, mode="abc",
               dir_phase="onlooker",      # where direction memory is USED: "onlooker" or "both"
               alpha=0.3,                 # direction-memory learning rate
               pmin_emp=0.2, pmin_onl=0.05, sel_prob=0.5,
               eta_up=1.2, eta_down=0.95, eta_min=0.2, eta_max=3.0,
               mr=0.0, mr_theta=0.2,      # multi-dimension directed move (onlooker)
               p_path=0.0, path_c=0.3, path_shrink=0.5,  # success-path (pattern) move (onlooker)
               gabc_C=1.5,                # GABC baseline only
               emp_gbest_C=0.0):          # DM: gbest pull for employed bees (GABC-style)
    """
    mode = "abc"  : standard ABC (Karaboga)
    mode = "gabc" : gbest-guided ABC (Zhu & Kwong, 2010)
    mode = "dabc" : directed ABC (Kiran & Findik, 2015), re-implemented from the paper's
                    description: verify against the original before publishing
    mode = "dm"   : direction-memory ABC. Per (source, dimension) memory Dm in [-1, 1]
                    learned from every move: s = sign(step) * sign(improvement).
      mr > 0      : onlooker also moves other dimensions whose |Dm| > mr_theta,
                    each with probability mr, in their remembered direction
      emp_gbest_C : employed bees add a GABC-style pull U(0, C) * (gbest - x)
      p_path > 0  : onlooker, with probability p_path, makes a full-vector move along the
                    source's success path P (EMA of successful displacements), which captures
                    correlations between dimensions; P shrinks after a failed path move
    """
    D, lb, ub = prob.D, prob.lb, prob.ub
    limit = limit or SN * D
    X = lb + rng.random((SN, D)) * (ub - lb)
    F = np.array([prob(x) for x in X])
    trial = np.zeros(SN, int)
    Dm = np.zeros((SN, D))       # direction memory
    eta = np.ones((SN, D))       # per-dimension step multiplier
    P = np.zeros((SN, D))        # success path
    directed = mode == "dm"
    dd = np.zeros((SN, D), int)  # dABC direction information in {-1, 0, +1}
    if mode in ("dl", "ll"):     # LLABC direction-learning rule: random initial sign per dimension
        dl = rng.choice([-1.0, 1.0], size=(SN, D))

    diag = {"moves": 0, "success": 0, "scouts": 0, "extra_dims": 0,
            "path_moves": 0, "path_success": 0}
    prob.diag = diag

    def accept(i, v, fv):
        if fv <= F[i]:
            X[i], F[i], trial[i] = v, fv, 0
        else:
            trial[i] += 1

    def path_move(i):
        u = rng.uniform(0.5, 2.0)
        v = np.clip(X[i] + u * P[i], lb, ub)
        fv = prob(v)
        improved = fv < F[i]
        diag["moves"] += 1; diag["success"] += int(improved)
        diag["path_moves"] += 1; diag["path_success"] += int(improved)
        if improved:
            P[i] = (1 - path_c) * P[i] + path_c * (v - X[i])
        else:
            P[i] *= path_shrink
        accept(i, v, fv)

    def try_move(i, onlooker):
        if (directed and onlooker and p_path > 0 and rng.random() < p_path
                and np.any(P[i] != 0)):
            path_move(i)
            return
        if mode == "ll":
            # LL-ABC elite learning (Gao et al., 2019, Eqs. 19-21): partner drawn from the top M
            # sources, M decreasing linearly from SN to mu*SN with the evaluations used (mu = 0.2)
            M = max(int(np.ceil(-prob.fe / prob.max_fe * SN + SN)), int(np.ceil(0.2 * SN)))
            top = [t for t in np.argsort(F)[:M] if t != i] or [t for t in range(SN) if t != i]
            k = top[rng.integers(len(top))]
        else:
            k = rng.integers(SN - 1); k += k >= i
        use_dir = directed and (dir_phase == "both" or onlooker)
        if use_dir and onlooker and rng.random() < sel_prob:
            w = np.abs(Dm[i]) + 0.05
            j = rng.choice(D, p=w / w.sum())
        else:
            j = rng.integers(D)
        v = X[i].copy()
        moved = [j]
        if mode in ("dl", "ll"):
            # direction learning of LL-ABC (Gao et al., 2019, Eq. 17):
            # v = x + delta * U(0,1) * |x - x_k|; delta kept after a success, reversed after a failure
            v[j] += dl[i, j] * rng.random() * abs(X[i, j] - X[k, j])
        elif mode == "dabc":
            # dABC (Kiran & Findik, 2015): remembered direction from the last successful move;
            # 0 -> standard bidirectional search
            if dd[i, j] == 0:
                v[j] += rng.uniform(-1, 1) * (X[i, j] - X[k, j])
            else:
                v[j] += dd[i, j] * rng.random() * abs(X[i, j] - X[k, j])
        elif not use_dir:
            v[j] += rng.uniform(-1, 1) * (X[i, j] - X[k, j])
            C = gabc_C if mode == "gabc" else (emp_gbest_C if directed else 0.0)
            if C > 0:
                g = int(np.argmin(F))
                v[j] += rng.uniform(0, C) * (X[g, j] - X[i, j])
        else:
            if onlooker and mr > 0:
                cand = np.flatnonzero(np.abs(Dm[i]) > mr_theta)
                cand = cand[cand != j]
                if cand.size:
                    extra = cand[rng.random(cand.size) < mr]
                    moved += list(extra)
                    diag["extra_dims"] += extra.size
            pmin = pmin_onl if onlooker else pmin_emp
            for d in moved:
                p_up = np.clip((1 + Dm[i, d]) / 2, pmin, 1 - pmin)
                delta = 1.0 if rng.random() < p_up else -1.0
                base = max(abs(X[i, d] - X[k, d]), 1e-12 * (ub[d] - lb[d]))
                v[d] += delta * rng.random() * eta[i, d] * base
        for d in moved:
            v[d] = np.clip(v[d], lb[d], ub[d])
        fv = prob(v)
        improved = fv < F[i]
        diag["moves"] += 1; diag["success"] += int(improved)

        if mode == "dabc":
            dd[i, j] = int(np.sign(v[j] - X[i, j])) if improved else 0
        if mode in ("dl", "ll") and not improved:
            dl[i, j] = -dl[i, j]
        if directed:
            sgn_imp = np.sign(F[i] - fv)
            for d in moved:
                s = np.sign(v[d] - X[i, d]) * sgn_imp
                Dm[i, d] = (1 - alpha) * Dm[i, d] + alpha * s
                if use_dir:
                    eta[i, d] = (min(eta[i, d] * eta_up, eta_max) if improved
                                 else max(eta[i, d] * eta_down, eta_min))
            if p_path > 0 and improved:
                P[i] = (1 - path_c) * P[i] + path_c * (v - X[i])
        accept(i, v, fv)

    def snapshot():
        diag["absDm"] = float(np.mean(np.abs(Dm))) if directed else np.nan
    prob.snapshot = snapshot

    while True:
        for i in range(SN):                         # employed bees
            try_move(i, onlooker=False)
        fit = _abc_fitness(F); p = fit / fit.sum()
        for _ in range(SN):                         # onlooker bees
            try_move(rng.choice(SN, p=p), onlooker=True)
        i = int(np.argmax(trial))                   # scout bee
        if trial[i] > limit:
            diag["scouts"] += 1
            X[i] = lb + rng.random(D) * (ub - lb)
            F[i] = prob(X[i]); trial[i] = 0
            Dm[i] = 0.0; eta[i] = 1.0; P[i] = 0.0; dd[i] = 0
            if mode in ("dl", "ll"): dl[i] = rng.choice([-1.0, 1.0], size=D)


def abc_dl(prob, rng, **kw):         return abc_family(prob, rng, mode="dl", **kw)


def ll_abc(prob, rng, SN=100, limit=200, **kw):
    """LL-ABC (Gao et al., 2019): direction learning (Eq. 17) and elite learning (Eqs. 19-21) in
    classic ABC, with the settings of the original paper (SN = 100, limit = 200, mu = 0.2)."""
    return abc_family(prob, rng, SN=SN, limit=limit, mode="ll", **kw)


def ddabc(prob, rng, SN=50, limit=100, p_elite=0.1, step=0.25):
    """DDABC (Zhou et al., 2024): DFSABC_elite framework (depth-first search and elite-guided
    equations, Cui et al., 2016) with the direction-information-learning (DIL) mechanism.
    Implemented from Algorithms 1-2 and Eqs. (11)-(12) of the paper; common parameters (SN, limit)
    use the settings of the original paper (SN = 50, limit = 100)."""
    D, lb, ub = prob.D, prob.lb, prob.ub
    limit = limit or SN * D
    X = lb + rng.random((SN, D)) * (ub - lb)
    F = np.array([prob(x) for x in X])
    trial = np.zeros(SN, int)
    delta = rng.choice([-1.0, 1.0], size=(SN, D))   # direction factor in [-1, 1]
    cnt = np.zeros((SN, D), int)                    # consecutive failures per dimension
    n_el = max(1, int(np.ceil(p_elite * SN)))
    prob.diag = {"moves": 0, "success": 0, "scouts": 0}

    def dil(i, j, improved):                        # Algorithm 1
        if improved:
            cnt[i, j] = 0
            if delta[i, j] > step: delta[i, j] -= step
            elif delta[i, j] < -step: delta[i, j] += step
        else:
            cnt[i, j] += 1
            if cnt[i, j] == 1:
                delta[i, j] = -delta[i, j]
            elif cnt[i, j] >= 2:
                if delta[i, j] > 0: delta[i, j] = min(1.0, delta[i, j] + step)
                elif delta[i, j] < 0: delta[i, j] = max(-1.0, delta[i, j] - step)

    def try_update(i, v, j):
        fv = prob(v)
        improved = fv < F[i]
        prob.diag["moves"] += 1; prob.diag["success"] += int(improved)
        if improved:
            X[i], F[i], trial[i] = v, fv, 0
        else:
            trial[i] += 1
        dil(i, j, improved)
        return improved

    while True:
        elite = np.argsort(F)[:n_el]
        best = int(np.argmin(F))
        # employed bee phase with depth-first search: Eq. (11)
        dfs, r = True, 0
        for _ in range(SN):
            if dfs:
                r = rng.integers(SN)
            e = elite[rng.integers(n_el)]
            k = rng.integers(SN - 1); k += k >= r
            j = rng.integers(D)
            v = X[r].copy()
            v[j] = np.clip(X[e, j] + delta[r, j] * rng.random() * (X[e, j] - X[k, j]), lb[j], ub[j])
            dfs = not try_update(r, v, j)
        # onlooker bee phase on the elite group with depth-first search: Eq. (12)
        elite = np.argsort(F)[:n_el]; best = int(np.argmin(F))
        dfs, e = True, elite[0]
        for _ in range(n_el):
            if dfs:
                e = elite[rng.integers(n_el)]
            k = rng.integers(SN - 1); k += k >= e
            j = rng.integers(D)
            v = X[e].copy()
            v[j] = np.clip(0.5 * (X[e, j] + X[best, j]) + delta[e, j] * rng.random() * (X[best, j] - X[k, j]), lb[j], ub[j])
            dfs = not try_update(e, v, j)
        # scout bee phase
        i = int(np.argmax(trial))
        if trial[i] > limit:
            prob.diag["scouts"] += 1
            X[i] = lb + rng.random(D) * (ub - lb); F[i] = prob(X[i]); trial[i] = 0
            delta[i] = rng.choice([-1.0, 1.0], size=D); cnt[i] = 0


def dabc(prob, rng, **kw):           return abc_family(prob, rng, mode="dabc", **kw)
def abc(prob, rng, **kw):            return abc_family(prob, rng, mode="abc", **kw)
def gabc(prob, rng, **kw):           return abc_family(prob, rng, mode="gabc", **kw)
def dm_both(prob, rng, **kw):        return abc_family(prob, rng, mode="dm", dir_phase="both", **kw)
def dm_onl(prob, rng, **kw):         return abc_family(prob, rng, mode="dm", **kw)
def dm_onl_mr(prob, rng, **kw):      return abc_family(prob, rng, mode="dm", mr=0.3, **kw)
def dm_onl_mr_g(prob, rng, **kw):    return abc_family(prob, rng, mode="dm", mr=0.3, emp_gbest_C=1.5, **kw)
def dm_onl_mr_g075(prob, rng, **kw): return abc_family(prob, rng, mode="dm", mr=0.3, emp_gbest_C=0.75, **kw)
def dm_onl_path(prob, rng, **kw):    return abc_family(prob, rng, mode="dm", p_path=0.3, **kw)
def dm_onl_mr_path(prob, rng, **kw): return abc_family(prob, rng, mode="dm", mr=0.3, p_path=0.3, **kw)


def pso(prob, rng, N=40, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4, vfrac=0.2):
    """Global-best PSO with linearly decreasing inertia weight."""
    D, lb, ub = prob.D, prob.lb, prob.ub
    vmax = vfrac * (ub - lb)
    X = lb + rng.random((N, D)) * (ub - lb)
    V = rng.uniform(-vmax, vmax, (N, D))
    F = np.array([prob(x) for x in X])
    P, PF = X.copy(), F.copy()
    g = int(np.argmin(PF))
    while True:
        w = w_max - (w_max - w_min) * prob.fe / prob.max_fe
        for i in range(N):
            r1, r2 = rng.random(D), rng.random(D)
            V[i] = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (P[g] - X[i])
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], lb, ub)
            fx = prob(X[i])
            if fx < PF[i]:
                P[i], PF[i] = X[i].copy(), fx
                if fx < PF[g]:
                    g = i


def bat(prob, rng, N=40, fmin=0.0, fmax=2.0, A0=1.0, r0=0.5, alpha=0.9, gamma=0.9):
    """Bat Algorithm (Yang, 2010) with loudness/pulse-rate update."""
    D, lb, ub = prob.D, prob.lb, prob.ub
    X = lb + rng.random((N, D)) * (ub - lb)
    V = np.zeros((N, D))
    F = np.array([prob(x) for x in X])
    A, r = np.full(N, A0), np.zeros(N)
    b = int(np.argmin(F)); xbest, fbest = X[b].copy(), F[b]
    t = 0
    while True:
        t += 1
        for i in range(N):
            freq = fmin + (fmax - fmin) * rng.random()
            V[i] = V[i] + (X[i] - xbest) * freq
            xn = np.clip(X[i] + V[i], lb, ub)
            if rng.random() > r[i]:                    # local random walk around best
                xn = np.clip(xbest + rng.uniform(-1, 1, D) * A.mean(), lb, ub)
            fn = prob(xn)
            if fn <= F[i] and rng.random() < A[i]:
                X[i], F[i] = xn, fn
                A[i] *= alpha
                r[i] = r0 * (1 - np.exp(-gamma * t))
            if fn <= fbest:
                xbest, fbest = xn.copy(), fn


ALGORITHMS = {
    # ---- direction-memory ABC family ----
    "DM-Onl": dm_onl,                 # memory learned everywhere, used by onlookers
    "DM-Both": dm_both,               # memory used by employed + onlookers
    "DM-Onl-MR": dm_onl_mr,           # + multi-dimension directed moves
    "DM-Onl-Path": dm_onl_path,       # + success-path (pattern) moves
    "DM-Onl-MRPath": dm_onl_mr_path,  # + both
    "DM-Onl-MR-G": dm_onl_mr_g,       # MR + gbest pull for employed bees (C = 1.5)
    "DM-Onl-MR-G075": dm_onl_mr_g075, # MR + gbest pull for employed bees (C = 0.75)
    # ---- baselines ----
    "ABC": abc, "GABC": gabc, "dABC": dabc, "L-ABC": abc_dl, "ABC-DL": abc_dl, "LL-ABC": ll_abc, "DDABC": ddabc, "PSO": pso, "BA": bat,
}
ABC_FAMILY = {"dABC", "ABC-DL", "L-ABC", "DM-Onl-MR-G", "DM-Onl-MR-G075", "DM-Onl", "DM-Both", "DM-Onl-MR", "DM-Onl-Path", "DM-Onl-MRPath", "ABC", "GABC"}
# Names used in the paper (aliases of the internal names above)
ALGORITHMS["DM-ABC"] = ALGORITHMS["DM-Onl-MR-G"]
ALGORITHMS["DM-ABC-NG"] = ALGORITHMS["DM-Onl-MR"]
ALGORITHMS["DM-ABC-NGM"] = ALGORITHMS["DM-Onl"]
ABC_FAMILY |= {"DM-ABC", "DM-ABC-NG", "DM-ABC-NGM"}
DEFAULT_ALGOS = ["DM-Onl-MR-G", "DM-Onl-MR", "DM-Onl", "ABC", "GABC", "dABC", "PSO", "BA"]
PROPOSED = "DM-Onl-MR-G"


# ============================================================
# Experiment runner
# ============================================================
def run_one(job):
    fname, aname, run, D, max_fe, n_ckpt, limit = job
    prob = Problem(fname, D, max_fe, n_ckpt)
    rng = np.random.default_rng(1000 * run + 17)   # same seed per run across algorithms
    kw = {"limit": limit} if (aname in ABC_FAMILY and limit) else {}
    t0 = time.time()
    try:
        ALGORITHMS[aname](prob, rng, **kw)
    except BudgetExhausted:
        pass
    prob.finish()
    diag = getattr(prob, "diag", None)
    if diag is not None and hasattr(prob, "snapshot"):
        prob.snapshot()
    return fname, aname, run, prob.best, prob.hist, time.time() - t0, diag


def fmt(x):
    return f"{x:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--maxfe", type=int, default=None,
                    help="default: 5000*dim (classic) or 10000*dim (cec2017, official budget)")
    ap.add_argument("--suite", choices=["classic", "cec2017", "real"], default="classic")
    ap.add_argument("--fe-per-dim", type=int, default=None,
                    help="budget = fe_per_dim * D per problem (default 5000 classic, 10000 cec2017/real)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--funcs", nargs="*", default=None,
                    help="default: whole suite; e.g. --funcs CEC17_F1 CEC17_F5")
    ap.add_argument("--algos", nargs="*", default=DEFAULT_ALGOS)
    ap.add_argument("--limit", type=int, default=None, help="ABC scout limit for all ABC variants (default SN*D)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--proposed", default=None, help="algorithm used as reference in Wilcoxon tests")
    ap.add_argument("--quick", action="store_true", help="dim=10, runs=5, maxfe=20000")
    ap.add_argument("--err-floor", type=float, default=1e-8,
                    help="errors below this are treated as 0 in statistics (CEC convention); raw CSV keeps raw values")
    ap.add_argument("--from-raw", default=None,
                    help="path to an existing raw_final_errors.csv: recompute statistics only, no runs")
    a = ap.parse_args()
    if a.quick:
        a.dim, a.runs, a.maxfe = 10, 5, 20000
    if a.funcs is None:
        a.funcs = {"cec2017": CEC2017, "real": list(REAL_MAIN)}.get(a.suite, list(FUNCTIONS))
    if a.suite == "real" and not REAL:
        raise SystemExit("realprob.py not found next to dmabc_benchmark.py")
    if a.dim == 10 and any(f in CEC17_NAN_AT_D10 for f in a.funcs):
        a.funcs = [f for f in a.funcs if f not in CEC17_NAN_AT_D10]
        print(f"Note: {CEC17_NAN_AT_D10} return NaN in opfunu at D=10 "
              "(1-dimensional elliptic component divides by zero) and were skipped.")
    if any(f.startswith("CEC17_") for f in a.funcs) and a.dim not in (10, 30, 50, 100):
        raise SystemExit("CEC2017 is defined only for dim = 10, 30, 50, 100")
    dims = {f: (REAL[f][1].size if f in REAL else a.dim) for f in a.funcs}
    fpd = a.fe_per_dim or (5000 if a.suite == "classic" else 10000)
    max_fe = {f: (a.maxfe or fpd * dims[f]) for f in a.funcs}
    a.dims = dims
    global PROPOSED
    if a.proposed:
        PROPOSED = a.proposed
    n_ckpt = 100
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    diags, curves = {}, None
    if a.from_raw:
        raw = {}
        for row in csv.DictReader(open(a.from_raw)):
            raw.setdefault((row["function"], row["algorithm"]), []).append(float(row["error"]))
        a.funcs = [f for f in a.funcs if any(k[0] == f for k in raw)]
        a.algos = [al for al in a.algos if any(k[1] == al for k in raw)]
        final = {k: np.array(v) for k, v in raw.items() if k[0] in a.funcs and k[1] in a.algos}
        print(f"Statistics only from {a.from_raw}: {len(a.funcs)} functions, algorithms {a.algos}")
    else:
        final, curves, diags = run_experiment(a, max_fe, n_ckpt)
        # ---------- raw results ----------
        with open(os.path.join(a.out, "raw_final_errors.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["function", "algorithm", "run", "error"])
            for (f, al), v in final.items():
                for r, e in enumerate(v):
                    w.writerow([f, al, r, e])
    report(a, final, curves, diags, max_fe, n_ckpt, t0)


def run_experiment(a, max_fe, n_ckpt):
    """Runs all jobs. Every finished job is appended to <out>/progress.jsonl, so an
    interrupted experiment resumes where it stopped when the same command is rerun."""
    import json
    meta = {"dims": a.dims, "max_fe": max_fe, "limit": a.limit, "n_ckpt": n_ckpt}
    meta_path = os.path.join(a.out, "meta.json")
    prog_path = os.path.join(a.out, "progress.jsonl")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if "dim" in old and "dims" not in old:      # folder written by the earlier version
            old = {"dims": {f: old["dim"] for f in a.funcs},
                   "max_fe": {f: old["max_fe"] for f in a.funcs},
                   "limit": old["limit"], "n_ckpt": old["n_ckpt"]}
        if old != meta:
            raise SystemExit(f"{a.out} holds results for {old}, not {meta}; use another --out")
    json.dump(meta, open(meta_path, "w"))

    final = {(f, al): np.full(a.runs, np.nan) for f in a.funcs for al in a.algos}
    curves = {(f, al): np.full((a.runs, n_ckpt), np.nan) for f in a.funcs for al in a.algos}
    diags = {}
    done = set()

    def store(f, al, r, best, hist, dg):
        if (f, al) not in final or r >= a.runs:
            return
        final[(f, al)][r] = best
        curves[(f, al)][r] = hist
        if dg is not None:
            diags.setdefault((f, al), []).append(dg)
        done.add((f, al, r))

    if os.path.exists(prog_path):
        for line in open(prog_path):
            try:
                d = json.loads(line)
            except ValueError:
                continue                      # partially written last line
            store(d["f"], d["al"], d["r"], d["best"], np.array(d["hist"], float), d["diag"])

    jobs = [(f, al, r, a.dims[f], max_fe[f], n_ckpt, a.limit)
            for f in a.funcs for al in a.algos for r in range(a.runs)
            if (f, al, r) not in done]
    fe_txt = (f"{min(max_fe.values())}-{max(max_fe.values())}"
              if len(set(max_fe.values())) > 1 else f"{next(iter(max_fe.values()))}")
    dim_txt = "per problem" if len(set(a.dims.values())) > 1 else f"{next(iter(a.dims.values()))}"
    print(f"D={dim_txt}, runs={a.runs}, maxFE={fe_txt}, limit={a.limit or 'SN*D'}, "
          f"jobs={len(jobs)} (already done: {len(done)}), workers={a.workers}", flush=True)
    t0 = time.time()
    if jobs:
        with open(prog_path, "a") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, (f, al, r, best, hist, dt, dg) in enumerate(
                    ex.map(run_one, jobs, chunksize=1), 1):
                store(f, al, r, best, hist, dg)
                fh.write(json.dumps({"f": f, "al": al, "r": r, "best": best,
                                     "hist": [float(h) for h in hist], "diag": dg}) + "\n")
                fh.flush()
                if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                    el = time.time() - t0
                    print(f"  {n}/{len(jobs)} done ({el:.0f}s, ~{el / n * (len(jobs) - n) / 60:.0f} min left)",
                          flush=True)
    return final, curves, diags


def report(a, final, curves, diags, max_fe, n_ckpt, t0):
    floor = lambda v: np.where(np.asarray(v) < a.err_floor, 0.0, np.asarray(v))
    final = {k: floor(v) for k, v in final.items()}
    print(f"(statistics use error floor {a.err_floor:g}: smaller errors count as 0)")
    # ---------- summary + Wilcoxon rank-sum (proposed vs others) ----------
    others = [al for al in a.algos if al != PROPOSED]
    wtl = {al: [0, 0, 0] for al in others}
    rows = []
    for f in a.funcs:
        for al in a.algos:
            v = final[(f, al)]
            sig = ""
            if al != PROPOSED and PROPOSED in a.algos:
                p_dm = final[(f, PROPOSED)]
                if np.array_equal(np.sort(p_dm), np.sort(v)):
                    p = 1.0
                else:
                    p = stats.ranksums(p_dm, v).pvalue
                if p < 0.05:
                    if np.median(p_dm) < np.median(v):
                        sig = "+"; wtl[al][0] += 1    # proposed significantly better
                    else:
                        sig = "-"; wtl[al][2] += 1
                else:
                    sig = "="; wtl[al][1] += 1
            rows.append([f, al, np.mean(v), np.std(v), np.min(v), np.median(v), sig])
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "algorithm", "mean", "std", "best", "median", "vs_proposed"])
        w.writerows(rows)

    # ---------- Friedman test on mean errors ----------
    M = np.array([[np.mean(final[(f, al)]) for al in a.algos] for f in a.funcs])
    ranks = np.array([stats.rankdata(r) for r in M])
    mean_rank = ranks.mean(axis=0)
    fr_p = stats.friedmanchisquare(*M.T).pvalue if len(a.algos) > 2 and len(a.funcs) > 1 else np.nan

    # ---------- print ----------
    print("\n" + "=" * 92)
    print(f"{'Function':<18}{'Algorithm':<15}{'Mean':>12}{'Std':>12}{'Best':>12}{'Median':>12}  Sig")
    print("-" * 92)
    for f, al, mean, std, best, med, sig in rows:
        print(f"{f:<18}{al:<15}{fmt(mean):>12}{fmt(std):>12}{fmt(best):>12}{fmt(med):>12}  {sig}")
    print("-" * 92)
    print(f"Wilcoxon rank-sum (alpha=0.05), {PROPOSED} vs others  [+ better / = tie / - worse]:")
    for al, (b, t, l) in wtl.items():
        print(f"  vs {al:<15}: {b}/{t}/{l}")
    print("Friedman mean rank (lower is better):")
    for al, mr in sorted(zip(a.algos, mean_rank), key=lambda x: x[1]):
        print(f"  {al:<15} {mr:.2f}")
    print(f"Friedman p-value: {fr_p:.3e}")
    print(f"Total time: {time.time() - t0:.0f}s, results in {a.out}")

    # ---------- internal diagnostics (ABC family) ----------
    if diags:
        cols = ["success_rate", "scouts", "mean_absDm", "extra_dims_per_move",
                "path_move_share", "path_success_rate"]
        def agg(lst):
            s = lambda key: float(np.sum([d.get(key, 0) for d in lst]))
            m = lambda key: float(np.nanmean([d.get(key, np.nan) for d in lst]))
            mv = max(s("moves"), 1)
            return [s("success") / mv, m("scouts"), m("absDm"), s("extra_dims") / mv,
                    s("path_moves") / mv,
                    s("path_success") / s("path_moves") if s("path_moves") else np.nan]
        with open(os.path.join(a.out, "diagnostics.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["function", "algorithm"] + cols)
            for (f, al), lst in diags.items():
                w.writerow([f, al] + agg(lst))
        print("\nDiagnostics (mean over functions):")
        print(f"{'Algorithm':<15}{'succ':>7}{'scouts':>8}{'|Dm|':>7}{'extra':>7}{'path%':>7}{'pathSucc':>9}")
        for al in a.algos:
            rowsd = [agg(diags[(f, al)]) for f in a.funcs if (f, al) in diags]
            if rowsd:
                mval = np.nanmean(np.array(rowsd, float), axis=0)
                print(f"{al:<15}{mval[0]:>7.3f}{mval[1]:>8.1f}{mval[2]:>7.3f}{mval[3]:>7.3f}"
                      f"{mval[4]:>7.3f}{mval[5]:>9.3f}")

    # ---------- convergence plots ----------
    if curves is None:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(a.funcs); cols = 3; rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(5 * cols, 3.8 * rws), squeeze=False)
        x = np.linspace(1 / n_ckpt, 1, n_ckpt)
        for ax, f in zip(axes.flat, a.funcs):
            for al in a.algos:
                ax.semilogy(x, np.maximum(np.mean(curves[(f, al)], axis=0), 1e-300), label=al)
            ax.set_title(f"{f} (D={a.dims.get(f, '?')}, FE={max_fe.get(f, '?')})", fontsize=9)
            ax.set_xlabel("fraction of FE budget"); ax.set_ylabel("mean error")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "convergence.png"), dpi=150)
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
