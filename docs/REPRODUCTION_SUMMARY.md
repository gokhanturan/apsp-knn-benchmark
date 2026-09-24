# APSP–kNN Reproduction Summary

This repository accompanies the manuscript **Comparative Benchmarking of All-Pairs Shortest-Path Algorithms on k-Nearest Neighbor Graphs**.

## Key reproduced findings

- Original fastest counts: repeated Dijkstra = 9, Floyd-Warshall = 3, Johnson = 0.
- Median repeated-Dijkstra speedup vs Floyd-Warshall: **2.29×**.
- Maximum repeated-Dijkstra speedup vs Floyd-Warshall: **8.66×** (Digits, k=5).
- Memory winner counts: Floyd-Warshall = 9, repeated Dijkstra = 2, Johnson = 1.
- Dataset-level exploratory Friedman test: **χ²(2)=3.500, p=0.174**.
- Float64 cross-method checks: **24/24 passed**.
- Independent NetworkX checks: **3/3 passed**.
- Controlled Digits ordering reversal:
  - k=5: 150–200 vertices
  - k=10: 150–200 vertices
  - k=20: 200–250 vertices
- Current-manuscript claim checks: **12/12 passed**.

## Reproducibility note

Absolute runtime and process-level RSS values depend on the execution environment. Structural and ordinal findings are the primary cross-hardware reproducibility targets. The crossover intervals above are specific to the tested graphs and software/hardware environment; they are not universal vertex-count thresholds.
