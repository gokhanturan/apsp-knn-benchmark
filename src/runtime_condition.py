
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ.setdefault(_v,'1')
import argparse, json, random, time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import shortest_path
from common import set_single_thread_env, pin_first_cpu, clean
set_single_thread_env()
ALGORITHMS={'floyd_warshall':'FW','johnson':'J','dijkstra':'D'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--seed-id',type=int,default=0)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--random-seed',type=int,default=20260804)
    p.add_argument('--batch-target-seconds',type=float,default=0.05)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--correctness-output',type=Path)
    a=p.parse_args()
    cpu=pin_first_cpu()
    graph=load_npz(a.graph).tocsr().astype(np.float64)

    for method in ALGORITHMS.values():
        for _ in range(a.warmups):
            d=shortest_path(graph,method=method,directed=False,return_predecessors=False)
            del d; clean()

    batch_counts={}
    for algorithm,method in ALGORITHMS.items():
        clean(); w0=time.perf_counter_ns()
        d=shortest_path(graph,method=method,directed=False,return_predecessors=False)
        elapsed=(time.perf_counter_ns()-w0)/1e9; del d
        if elapsed < 0.010:
            batch_counts[algorithm]=max(2,min(200,int(np.ceil(a.batch_target_seconds/max(elapsed,1e-6)))))
        else:
            batch_counts[algorithm]=1

    rng=random.Random(a.random_seed + sum(map(ord,a.dataset)) + 1009*a.k + 7919*a.seed_id)
    rows=[]; first={}
    for rep in range(1,a.repeats+1):
        order=list(ALGORITHMS.items()); rng.shuffle(order)
        for algorithm,method in order:
            clean(); batch=batch_counts[algorithm]
            c0=time.process_time_ns(); w0=time.perf_counter_ns(); d=None
            for _ in range(batch):
                if d is not None: del d
                d=shortest_path(graph,method=method,directed=False,return_predecessors=False)
            wall=((time.perf_counter_ns()-w0)/1e9)/batch
            cpu_s=((time.process_time_ns()-c0)/1e9)/batch
            rows.append({'dataset':a.dataset,'k':a.k,'seed_id':a.seed_id,'algorithm':algorithm,'method':method,'repeat':rep,
                         'batch_iterations':batch,'wall_time_s':wall,'cpu_time_s':cpu_s,
                         'cpu_wall_ratio':cpu_s/wall if wall else np.nan,'affinity_cpu':cpu,
                         'nodes':graph.shape[0],'edges':graph.nnz//2})
            if rep==1: first[algorithm]=np.array(d,dtype=np.float64,copy=True)
            del d

    a.output.parent.mkdir(parents=True,exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(a.output,index=False)

    if a.correctness_output:
        ref=first['dijkstra']; checks=[]; tiny=np.finfo(np.float64).tiny
        for alg in ('floyd_warshall','johnson'):
            arr=first[alg]
            finite=np.array_equal(np.isfinite(ref),np.isfinite(arr))
            mask=np.isfinite(ref)&np.isfinite(arr)
            diff=np.abs(ref[mask]-arr[mask])
            denom=np.maximum(np.abs(ref[mask]),tiny)
            max_abs=float(diff.max()) if diff.size else 0.0
            max_rel=float((diff/denom).max()) if diff.size else 0.0
            close=bool(np.allclose(ref,arr,rtol=1e-10,atol=1e-12,equal_nan=True))
            checks.append({'dataset':a.dataset,'k':a.k,'comparison':f'{alg}_vs_dijkstra',
                           'finite_pattern_match':finite,'max_abs_error':max_abs,'max_rel_error':max_rel,
                           'rtol':1e-10,'atol':1e-12,'passed':bool(finite and close)})
        pd.DataFrame(checks).to_csv(a.correctness_output,index=False)
    print(json.dumps({'rows':len(rows),'dataset':a.dataset,'k':a.k,'seed_id':a.seed_id,'cpu':cpu,'batch_counts':batch_counts}))

if __name__=='__main__': main()
