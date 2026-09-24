
from __future__ import annotations
import ctypes, gc, os

def set_single_thread_env():
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[k]='1'

def pin_first_cpu():
    try:
        cpus=sorted(os.sched_getaffinity(0))
        if cpus:
            os.sched_setaffinity(0,{cpus[0]})
            return cpus[0]
    except Exception:
        pass
    return None

def malloc_trim():
    try: ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception: pass

def clean():
    gc.collect(); malloc_trim()
