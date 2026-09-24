# APSP-kNN Benchmark

Reproduction repository for the manuscript **“Comparative Benchmarking of All-Pairs Shortest-Path Algorithms on k-Nearest Neighbor Graphs.”**

The study compares the SciPy execution paths for Floyd-Warshall, Johnson, and repeated Dijkstra on positive-weight undirected k-nearest-neighbor (kNN) graphs derived from four public datasets.

## Main reproduced findings

- Repeated Dijkstra was fastest in **9 of 12** original graph conditions.
- Floyd-Warshall was fastest on the three Wine graphs.
- Median repeated-Dijkstra speedup vs Floyd-Warshall: **2.29×**.
- Maximum speedup: **8.66×** on Digits at `k=5`.
- Memory winner counts: Floyd-Warshall **9**, repeated Dijkstra **2**, Johnson **1**.
- Controlled Digits ordering reversal:
  - `k=5`: 150–200 vertices
  - `k=10`: 150–200 vertices
  - `k=20`: 200–250 vertices
- Float64 cross-method checks: **24/24 passed**.
- Independent NetworkX checks: **3/3 passed**.
- Exploratory dataset-level Friedman test: **χ²(2)=3.500, p=0.174**.

> Absolute runtime and process-level RSS measurements are environment dependent. The crossover intervals are empirical properties of the tested graphs and execution environment, not universal vertex-count thresholds.

## Colab notebooks

Run the notebooks in order:

1. [`01_build_graphs.ipynb`](notebooks/01_build_graphs.ipynb) — rebuild and verify the original and controlled-scaling kNN graphs.
2. [`02_original_benchmark.ipynb`](notebooks/02_original_benchmark.ipynb) — rerun the 12 original graph runtime, memory, and numerical-equivalence benchmarks.
3. [`03_controlled_scaling.ipynb`](notebooks/03_controlled_scaling.ipynb) — rerun the within-Digits scaling experiment.
4. [`04_results_and_figures.ipynb`](notebooks/04_results_and_figures.ipynb) — reproduce tables, key claims, and manuscript figures from the archived results.

Colab links become active after this repository is uploaded to:

`https://github.com/gokhanturan/apsp-knn-benchmark`

## Repository structure

```text
apsp-knn-benchmark/
├── data/                 # CSV snapshots used in the experiments
├── graphs/               # 12 original kNN graphs
├── scaling_graphs/       # controlled Digits scaling graphs
├── results/              # raw and summarized benchmark outputs
├── figures/              # manuscript figures
├── notebooks/            # Colab-ready reproduction notebooks
├── src/                  # benchmark and graph-construction scripts
├── docs/                 # reproduction summary
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Environment used for the reported benchmark

The archived experiment metadata reports:

- Python 3.12.13
- NumPy 2.0.2
- SciPy 1.16.3
- scikit-learn 1.6.1
- pandas 2.2.2
- psutil 5.9.5
- NetworkX 3.6.1
- Google Colab hosted Linux runtime
- Intel Xeon CPU @ 2.00 GHz exposed through KVM
- benchmark worker pinned to one logical CPU
- common numerical-library thread counts fixed to 1

The exact metadata is stored in `results/system_metadata.json`.

## Quick start

```bash
git clone https://github.com/gokhanturan/apsp-knn-benchmark.git
cd apsp-knn-benchmark
python -m pip install -r requirements.txt
python src/check_current_manuscript_claims.py
```

The final command should report **12/12** current-manuscript claim checks as passed.

## Benchmark protocol

The original-condition benchmark uses:

- `k = 5, 10, 20`
- three warm-up calls per method
- 30 wall-clock/CPU measurements per algorithm-condition pair
- calibrated batching for calls shorter than 10 ms
- randomized method order with fixed seed `20260804`
- 30 fresh child processes per algorithm-condition pair for process-level memory measurement
- float64 APSP outputs
- `np.allclose` numerical validation with `rtol=1e-10`, `atol=1e-12`

The controlled Digits experiment uses sample sizes:

`100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 1797`

and sampling seeds:

`11, 23, 37, 53, 71`

For the full 1,797-sample dataset only one graph is needed because all observations are retained.

## Data and graph construction

The bundled CSV files are snapshots of the four datasets used in the experiment. The feature columns are standardized to zero mean and unit variance before neighbor search. Euclidean distance is used for kNN construction and edge weighting. Directed kNN neighborhoods are converted to an undirected **union graph**: an edge is retained if either endpoint selects the other.

Dataset targets are excluded from graph construction and APSP benchmarking.

## Notes on memory measurements

Incremental peak RSS is a **process-level** measure. It can include the dense APSP output matrix, temporary arrays, Python interpreter state, and library allocations. It should not be interpreted as the theoretical auxiliary-space complexity of the abstract algorithm.

## License

No software license is assigned in this prepared package. Add the license you want before making the repository public.
