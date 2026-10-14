# DM-ABC: Direction-Memory Artificial Bee Colony

Code and results for the paper *A Direction-Memory Artificial Bee Colony Algorithm with Phase-Separated Guidance for Continuous Optimization* (under review).

DM-ABC learns, for every food source and dimension, whether increasing or decreasing that dimension has improved the source. The memory is updated from every move, successful or not, at no extra cost in function evaluations. Onlooker bees use it to choose step directions and to move confident dimensions together; employed bees keep a gbest-guided bidirectional search.

## Algorithms

| Name in the paper | Name accepted by the scripts | Description |
|---|---|---|
| DM-ABC | `DM-ABC` (or `DM-Onl-MR-G`) | full algorithm |
| DM-ABC-NG | `DM-ABC-NG` (or `DM-Onl-MR`) | without gbest guidance |
| DM-ABC-NGM | `DM-ABC-NGM` (or `DM-Onl`) | without gbest guidance and multi-dimensional moves |
| ABC, GABC, dABC, PSO, BA | `ABC`, `GABC`, `dABC`, `PSO`, `BA` | baselines |
| DDABC | `DDABC` | Zhou et al. (2024), with the settings of the original paper (SN = 50, limit = 100) |
| LL-ABC | `LL-ABC` | Gao et al. (2019): direction learning and elite learning, settings of the original paper (SN = 100, limit = 200) |
| L-ABC | `L-ABC` | LL-ABC with direction learning only (as in Gao et al., 2019, Table I) |

The merged result files in `results/` use the names of the paper; files written by the scripts use the internal names (`DM-Onl-MR-G`, `DM-Onl-MR`, `DM-Onl`).

## Requirements

Python 3.10 or later. Install the dependencies with

```
pip install -r requirements.txt
```

The CEC2017 functions come from `opfunu`; the results in the paper were produced with version 1.0.4. Function labels follow the official CEC2017 numbering (F1, F3–F30). opfunu drops the official F2 and renumbers the remaining functions, so official F*k* (k ≥ 3) is opfunu F*k*−1; `dmabc_benchmark.py` performs this mapping.

## Quick start

```
python run_all.py --quick
```

runs a short check (about one minute) of the classical, CEC2017 and real-world suites.

## Reproducing the experiments

```
python run_all.py --workers 16                      # all experiments
python run_all.py --only real                       # PV and CEC2011 problems
python run_all.py --only classic_D30 classic_D50    # classical functions
python run_all.py --only cec_D30 cec_D50            # CEC2017
```

Each experiment writes to `dmabc_runs/<name>/` and can be interrupted and resumed with the same command. When all experiments finish, `run_all.py` writes a report and packs everything into one zip file.

Single experiments can also be run directly, for example

```
python dmabc_benchmark.py --suite cec2017 --dim 30 --runs 30 --algos DM-ABC ABC GABC --out my_run
python dmabc_benchmark.py --from-raw my_run/raw_final_errors.csv --out my_stats   # statistics only
```

All algorithms receive the same budget of function evaluations and the same random seed in each run. Results are identical across runs on the same platform.

## Files

| File | Content |
|---|---|
| `dmabc_benchmark.py` | algorithms, classical and CEC2017 suites, statistics and plots |
| `realprob.py` | photovoltaic parameter extraction (RTC France, Photowatt-PWP201) and CEC2011 problems |
| `run_all.py` | runs all experiments of the paper and packs the results |
| `results/` | merged results used in the paper, one pair of files per suite (see below) |

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## Results

`results/` contains the results reported in the paper, merged over all experiments, one pair of files per suite:

| Suite | Files |
|---|---|
| Real-world problems (PV and CEC2011) | `real_final_errors.csv`, `real_summary.csv` |
| Classical functions, D = 30 and 50 | `classic_D30_*.csv`, `classic_D50_*.csv` |
| CEC2017, D = 30 and 50 | `cec_D30_*.csv`, `cec_D50_*.csv` |

`*_final_errors.csv` lists the final error of every run (problem, algorithm, run, error); `*_summary.csv` gives the mean, standard deviation, best and median error and the Wilcoxon rank-sum result against DM-ABC (+ better, = no significant difference, − worse, α = 0.05). CEC2017 problems are labeled with the official numbers (F1, F3–F30), and all 11 algorithms of the paper are included in every suite they were run on. `environment_machine*.txt` records the software and hardware of the two machines used.
