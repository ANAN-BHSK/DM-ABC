"""
Run every DM-ABC experiment with one command and pack all results into one zip file.

    python run_all.py --workers 8            # full set of experiments
    python run_all.py --quick                # 2-3 minute check that everything works
    python run_all.py --workers 8 --only real classic_D30
    python run_all.py --only real_pvlog      # supplementary PV experiment (log10 I0 encoding)
    python run_all.py --only cec_D30_ablation cec_D50_ablation   # CEC2017 ablation
    python run_all.py --only real_dir classic_D30_dir classic_D50_dir cec_D30_dir cec_D50_dir   # DDABC, LL-ABC
    python run_all.py --only real_labc classic_D30_labc classic_D50_labc cec_D30_labc cec_D50_labc   # L-ABC
    python run_all.py --only cec_D30_F3 cec_D50_F3   # official CEC2017 F3 (Zakharov), all algorithms

Put this file next to dmabc_benchmark.py and realprob.py. If it is interrupted (Ctrl+C, power cut),
run the same command again: finished runs are kept and only the rest are computed.
At the end it writes dmabc_results_<date>.zip, which is the single file to send back.
"""
import argparse, datetime, hashlib, os, platform, subprocess, sys, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "dmabc_benchmark.py")
CEC_ALGOS = ["DM-Onl-MR-G", "ABC", "GABC", "dABC", "PSO", "BA"]
REAL_ALGOS = ["DM-Onl-MR-G", "DM-Onl-MR", "ABC", "GABC", "dABC", "PSO", "BA"]
ABL_ALGOS = ["DM-Onl-MR", "DM-Onl"]
DIR_ALGOS = ["DDABC", "LL-ABC"]
F3_ALGOS = ["DM-Onl-MR-G", "DM-Onl-MR", "DM-Onl", "ABC", "GABC", "dABC", "PSO", "BA", "DDABC", "LL-ABC", "L-ABC"]
REALPROB = os.path.join(HERE, "realprob.py")

# name -> extra arguments for dmabc_benchmark.py
EXPERIMENTS = {
    "real":        ["--suite", "real", "--algos", *REAL_ALGOS],   # PV + CEC2011, 10000*D FE
    "real_pvlog":  ["--suite", "real", "--algos", *REAL_ALGOS,     # supplementary: log10(I0) encoding
                    "--funcs", "PV_RTC_SDM_log", "PV_RTC_DDM_log", "PV_RTC_TDM_log",
                    "PV_PWP_SDM_log", "PV_PWP_DDM_log"],
    "classic_D30": ["--suite", "classic", "--dim", "30"],
    "classic_D50": ["--suite", "classic", "--dim", "50"],
    "cec_D30":     ["--suite", "cec2017", "--dim", "30", "--algos", *CEC_ALGOS],
    "cec_D50":     ["--suite", "cec2017", "--dim", "50", "--algos", *CEC_ALGOS],
    # ablation on CEC2017 (DM-ABC-NG = DM-Onl-MR, DM-ABC-NGM = DM-Onl); same seeds as cec_D30/cec_D50
    "cec_D30_ablation": ["--suite", "cec2017", "--dim", "30", "--algos", *ABL_ALGOS],
    "cec_D50_ablation": ["--suite", "cec2017", "--dim", "50", "--algos", *ABL_ALGOS],
    # direction-learning competitors: DDABC (Zhou et al., 2024) and LL-ABC (Gao et al., 2019)
    "real_dir":        ["--suite", "real", "--algos", *DIR_ALGOS],
    "classic_D30_dir": ["--suite", "classic", "--dim", "30", "--algos", *DIR_ALGOS],
    "classic_D50_dir": ["--suite", "classic", "--dim", "50", "--algos", *DIR_ALGOS],
    "cec_D30_dir":     ["--suite", "cec2017", "--dim", "30", "--algos", *DIR_ALGOS],
    "cec_D50_dir":     ["--suite", "cec2017", "--dim", "50", "--algos", *DIR_ALGOS],
    # L-ABC (direction learning of LL-ABC only); can run on a separate machine
    "real_labc":        ["--suite", "real", "--algos", "L-ABC"],
    "classic_D30_labc": ["--suite", "classic", "--dim", "30", "--algos", "L-ABC"],
    "classic_D50_labc": ["--suite", "classic", "--dim", "50", "--algos", "L-ABC"],
    "cec_D30_labc":     ["--suite", "cec2017", "--dim", "30", "--algos", "L-ABC"],
    "cec_D50_labc":     ["--suite", "cec2017", "--dim", "50", "--algos", "L-ABC"],
    # official CEC2017 F3 (Zakharov), missing from the earlier runs; all algorithms
    "cec_D30_F3": ["--suite", "cec2017", "--dim", "30", "--funcs", "CEC17_F3", "--algos", *F3_ALGOS],
    "cec_D50_F3": ["--suite", "cec2017", "--dim", "50", "--funcs", "CEC17_F3", "--algos", *F3_ALGOS],
}
QUICK = {
    "quick_classic": ["--suite", "classic", "--dim", "10", "--runs", "2", "--maxfe", "2000",
                      "--funcs", "F1_Sphere", "F7_Schwefel1.2", "F9_RotRastrigin"],
    "quick_cec":     ["--suite", "cec2017", "--dim", "10", "--runs", "2", "--maxfe", "2000",
                      "--funcs", "CEC17_F1", "CEC17_F10", "CEC17_F21", "--algos", *CEC_ALGOS],
    "quick_real":    ["--suite", "real", "--runs", "2", "--fe-per-dim", "200",
                      "--funcs", "PV_RTC_SDM", "CEC11_FM", "CEC11_Radar20", "--algos", *REAL_ALGOS],
}


def ensure_packages():
    missing = []
    for mod, pkg in [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                     ("opfunu", "opfunu")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def environment_info(workers, runs):
    import numpy, scipy
    try:
        import opfunu; opv = opfunu.__version__
    except Exception:
        opv = "not installed"
    sha = hashlib.sha256(open(BENCH, "rb").read()).hexdigest()[:16]
    return "\n".join([
        f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"python: {platform.python_version()}  numpy: {numpy.__version__}  "
        f"scipy: {scipy.__version__}  opfunu: {opv}",
        f"os: {platform.platform()}  cpu_count: {os.cpu_count()}  workers: {workers}",
        f"runs per (function, algorithm): {runs}",
        f"dmabc_benchmark.py sha256[:16]: {sha}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {list(EXPERIMENTS)}")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the setup")
    ap.add_argument("--root", default="dmabc_runs", help="folder for all experiment outputs")
    a = ap.parse_args()

    for p in (BENCH, REALPROB):
        if not os.path.exists(p):
            sys.exit(f"{os.path.basename(p)} must be in the same folder as run_all.py")
    ensure_packages()
    exps = QUICK if a.quick else {k: v for k, v in EXPERIMENTS.items()
                                  if a.only is None or k in a.only}
    root = os.path.join(HERE, a.root + ("_quick" if a.quick else ""))
    os.makedirs(root, exist_ok=True)
    env = environment_info(a.workers, a.runs)
    open(os.path.join(root, "environment.txt"), "w").write(env + "\n")
    print(env, flush=True)

    status = {}
    t_all = time.time()
    for name, extra in exps.items():
        out = os.path.join(root, name)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-u", BENCH, "--workers", str(a.workers), "--out", out]
        if "--runs" not in extra:
            cmd += ["--runs", str(a.runs)]
        cmd += extra
        print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd[2:])}\n{'=' * 70}", flush=True)
        t0 = time.time()
        with open(os.path.join(out, "log.txt"), "a") as log:
            log.write(f"\n### {datetime.datetime.now().isoformat(timespec='seconds')}  "
                      f"{' '.join(cmd[2:])}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd=HERE)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            rc = proc.wait()
        status[name] = ("OK" if rc == 0 else f"FAILED (exit {rc})") + \
                       f", {(time.time() - t0) / 60:.1f} min"
        print(f"[{name}] {status[name]}", flush=True)

    # ---------- one report + one zip ----------
    lines = ["DM-ABC experiment results", "=" * 26, env, "", "Status:"]
    lines += [f"  {k}: {v}" for k, v in status.items()]
    for name in exps:
        log = os.path.join(root, name, "log.txt")
        if os.path.exists(log):
            txt = open(log).read()
            i = txt.rfind("Wilcoxon")
            lines += ["", "-" * 70, f"[{name}]", txt[i:].strip() if i >= 0 else "(no summary)"]
    report = "\n".join(lines) + "\n"
    open(os.path.join(root, "REPORT.txt"), "w").write(report)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zpath = os.path.join(HERE, f"dmabc_results_{'quick_' if a.quick else ''}{stamp}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(BENCH, "dmabc_benchmark.py")
        z.write(REALPROB, "realprob.py")
        z.write(os.path.abspath(__file__), "run_all.py")
        for dirpath, _, files in os.walk(root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(root)))
    print("\n" + report)
    print(f"Total time: {(time.time() - t_all) / 60:.1f} min")
    print(f"\n>>> Send this file back: {zpath}")


if __name__ == "__main__":
    main()
