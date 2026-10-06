"""Isolate SGLang's memory-saver preload/spawn path with one tiny tensor."""

import json
import multiprocessing as mp
import argparse
import os
from pathlib import Path


def child():
    print('EVA_MEMORY_SAVER_SPAWN child_entered', flush=True)
    import torch
    from torch_memory_saver import torch_memory_saver
    print('EVA_MEMORY_SAVER_SPAWN torch_imported', flush=True)
    with torch_memory_saver.region(tag='eva-preflight', enable_cpu_backup=True):
        value = torch.ones(1024, device='cuda')
    torch_memory_saver.pause(tag='eva-preflight')
    torch_memory_saver.resume(tag='eva-preflight')
    torch.cuda.synchronize()
    assert value.sum().item() == 1024
    print('EVA_MEMORY_SAVER_SPAWN ' + json.dumps({
        'status': 'complete', 'pause_resume_value_preserved': True,
        'synthetic_infrastructure_only': True,
    }), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-cuda-library-path', action='store_true')
    args = parser.parse_args()
    if args.with_cuda_library_path:
        from run_full_parameter import cuda_runtime_environment
        runtime = cuda_runtime_environment()
        assert (Path(runtime['CUDA_HOME']) / 'lib/libcudart.so.13').is_file()
        os.environ.update(runtime)
    from torch_memory_saver import configure_subprocess
    process = mp.get_context('spawn').Process(target=child)
    with configure_subprocess():
        process.start()
    process.join(timeout=45)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            # Only this directly created tiny probe child is in scope here.
            process.kill()
            process.join(timeout=10)
        raise RuntimeError('Tiny owned memory-saver spawn probe timed out')
    print('EVA_MEMORY_SAVER_SPAWN child_exit_code=' + str(process.exitcode), flush=True)
    if process.exitcode != 0:
        raise RuntimeError('Memory-saver spawn child failed')
