"""Restore the authorized final-SFT service with fresh, verified PID receipts.

The established RSI serving context owns the child process and its cleanup.
This detached driver keeps it alive until explicitly signalled; no training or
benchmark actor is launched here. Startup includes one fixed public-image check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from threading import Event

from training.eva_rsi.serving import serving_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    training = repo / 'runs/qwen35-9b-sft-epoch-20260910.v1'
    image = repo.parent / ('datasets/automedbench-lite-8928073-cds.v1/public-release/benchmarks/'
        'AutoMedBench-classification/data/SkinLesionISIC/public/ISIC_00000001/image.jpg')
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with serving_session(training / 'hf/iter_0000170', training / 'checkpoints',
                         repo.parent / 'Qwen3.5-9B', args.output, image, port=30910) as service:
        print(json.dumps({'status': 'ready', 'server_pid': service.pid,
            'identity_path': str(service.identity_path), 'canary_path': str(service.canary_path),
            'training_updates': 0}), flush=True)
        while not stop.wait(10):
            if not Path(f'/proc/{service.pid}').exists():
                raise RuntimeError('Owned final checkpoint server exited')


if __name__ == '__main__':
    main()
