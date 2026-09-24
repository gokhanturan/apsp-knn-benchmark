from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

runtime = pd.read_csv(R / "runtime_summary_30.csv")
memory = pd.read_csv(R / "memory_summary_30.csv")
correct = pd.read_csv(R / "correctness_float64.csv")
nx = pd.read_csv(R / "networkx_validation.csv")
reversal = pd.read_csv(R / "scaling_order_reversal.csv")

pivot = runtime.pivot_table(index=["dataset", "k"], columns="algorithm", values="wall_median_s")
pivot["fastest"] = pivot.idxmin(axis=1)
pivot["dijkstra_speedup"] = pivot["floyd_warshall"] / pivot["dijkstra"]

fast_counts = pivot["fastest"].value_counts().to_dict()
median_speedup = float(pivot["dijkstra_speedup"].median())
max_speedup = float(pivot["dijkstra_speedup"].max())

mp = memory.pivot_table(index=["dataset", "k"], columns="algorithm", values="incremental_peak_rss_median_mib")
mem_counts = mp.idxmin(axis=1).value_counts().to_dict()

expected_reversal = [
    {"k": 5, "last_tested_n_with_floyd_not_slower": 150, "first_tested_n_with_dijkstra_faster": 200},
    {"k": 10, "last_tested_n_with_floyd_not_slower": 150, "first_tested_n_with_dijkstra_faster": 200},
    {"k": 20, "last_tested_n_with_floyd_not_slower": 200, "first_tested_n_with_dijkstra_faster": 250},
]
obs_reversal = reversal.astype(int).to_dict("records")

# Friedman values are stored in key_results.json and originate from the dataset-level analysis.
import json
with open(R / "key_results.json", encoding="utf-8") as f:
    key = json.load(f)

checks = [
    ("Original fastest counts", fast_counts, {"dijkstra": 9, "floyd_warshall": 3}, fast_counts == {"dijkstra": 9, "floyd_warshall": 3}),
    ("Wine fastest algorithm", pivot.loc["wine", "fastest"].tolist(), ["floyd_warshall"] * 3, pivot.loc["wine", "fastest"].tolist() == ["floyd_warshall"] * 3),
    ("Dijkstra fastest on Diabetes/Breast Cancer/Digits", pivot.loc[["diabetes", "breast_cancer", "digits"], "fastest"].tolist(), ["dijkstra"] * 9, pivot.loc[["diabetes", "breast_cancer", "digits"], "fastest"].tolist() == ["dijkstra"] * 9),
    ("Median Dijkstra speedup rounded", round(median_speedup, 2), 2.29, round(median_speedup, 2) == 2.29),
    ("Maximum Dijkstra speedup rounded", round(max_speedup, 2), 8.66, round(max_speedup, 2) == 8.66),
    ("Memory winner counts", mem_counts, {"floyd_warshall": 9, "dijkstra": 2, "johnson": 1}, mem_counts == {"floyd_warshall": 9, "dijkstra": 2, "johnson": 1}),
    ("Cross-method float64 checks", int(len(correct)), 24, len(correct) == 24),
    ("All float64 checks pass", bool(correct["passed"].all()), True, bool(correct["passed"].all())),
    ("NetworkX checks", int(nx["passed"].sum()), 3, int(nx["passed"].sum()) == 3),
    ("Friedman chi-square rounded", round(float(key["clustered_friedman_chi2"]), 3), 3.5, round(float(key["clustered_friedman_chi2"]), 3) == 3.5),
    ("Friedman p rounded", round(float(key["clustered_friedman_p"]), 3), 0.174, round(float(key["clustered_friedman_p"]), 3) == 0.174),
    ("Controlled scaling ordering reversal", obs_reversal, expected_reversal, obs_reversal == expected_reversal),
]

df = pd.DataFrame(checks, columns=["claim", "observed", "manuscript_expected", "pass"])
df.to_csv(R / "current_manuscript_claim_checks.csv", index=False)
print(df.to_string(index=False))
print(f"\nPassed: {int(df['pass'].sum())}/{len(df)}")
