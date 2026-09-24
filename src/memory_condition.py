
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ.setdefault(_v,'1')
import argparse, json, resource
from pathlib import Path
import numpy as np, psutil
from scipy.sparse import load_npz
from scipy.sparse.csgraph import shortest_path
from common import set_single_thread_env, pin_first_cpu, clean
set_single_thread_env()
METHODS={'floyd_warshall':'FW','johnson':'J','dijkstra':'D'}

def child_measure(graph, method, write_fd):
    try:
        cpu=pin_first_cpu(); clean()
        proc=psutil.Process(os.getpid())
        baseline_rss=proc.memory_info().rss
        baseline_max=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        d=shortest_path(graph,method=method,directed=False,return_predecessors=False)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        current=proc.memory_info().rss
        payload={
            'baseline_rss_mib':baseline_rss/2**20,
            'baseline_maxrss_mib':baseline_max/2**20,
            'peak_maxrss_mib':peak/2**20,
            'incremental_peak_rss_mib':max(0,peak-baseline_rss)/2**20,
            'end_rss_mib':current/2**20,
            'output_matrix_mib':d.nbytes/2**20,
            'temporary_over_output_mib':max(0,(peak-baseline_rss)-d.nbytes)/2**20,
            'affinity_cpu':cpu,
        }
        os.write(write_fd,(json.dumps(payload)+'\n').encode())
    except BaseException as e:
        os.write(write_fd,(json.dumps({'error':repr(e)})+'\n').encode())
    finally:
        os.close(write_fd); os._exit(0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--algorithm',choices=METHODS,required=True)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not hasattr(os,'fork'):
        raise RuntimeError('This publication memory protocol requires Linux/Unix os.fork().')
    pin_first_cpu(); graph=load_npz(a.graph).tocsr().astype(np.float64); clean(); rows=[]
    for rep in range(1,a.repeats+1):
        r,w=os.pipe(); pid=os.fork()
        if pid==0:
            os.close(r); child_measure(graph,METHODS[a.algorithm],w)
        os.close(w); data=b''
        while True:
            chunk=os.read(r,65536)
            if not chunk: break
            data+=chunk
        os.close(r); _,status=os.waitpid(pid,0)
        payload=json.loads(data.decode().strip())
        if 'error' in payload: raise RuntimeError(payload['error'])
        rows.append({'dataset':a.dataset,'k':a.k,'algorithm':a.algorithm,'method':METHODS[a.algorithm],
                     'repeat':rep,'nodes':graph.shape[0],'edges':graph.nnz//2,**payload})
    import pandas as pd
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print(json.dumps({'rows':len(rows),'dataset':a.dataset,'k':a.k,'algorithm':a.algorithm}))

if __name__=='__main__': main()
