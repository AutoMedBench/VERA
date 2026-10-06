from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_campaign_script_exposes_high_width_and_lease_controls() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--worker-width" in completed.stdout
    assert "--queue-capacity" in completed.stdout
    assert "--heartbeat-seconds" in completed.stdout
    assert "--progress-seconds" in completed.stdout
