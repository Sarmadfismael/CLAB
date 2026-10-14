"""Re-run every CPU experiment from the cached features, then rebuild the tables.

    python run_all.py                # about 30-60 min on a laptop CPU
    python run_all.py --skip e03     # skip the slow adapter sweep
    python run_all.py --data-root /path/to/datasets   # also rebuild Table 1
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["e01_main_grid", "e02_loto", "e03_adapters", "e04_robustness", "e05_fusion",
         "e06_perclass_invariance", "e07_source_budget", "e08_alpha_estimators",
         "e09_geometry"]

ag = argparse.ArgumentParser()
ag.add_argument("--skip", default="")
ag.add_argument("--data-root", default="")
a = ag.parse_args()
skip = {s.strip() for s in a.skip.split(",") if s.strip()}
for s in STEPS:
    if any(s.startswith(k) for k in skip):
        continue
    t = time.time()
    print(f"=== {s}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "experiments" / f"{s}.py")], check=True)
    print(f"    {time.time() - t:.0f} s", flush=True)
if a.data_root:
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e10_dataset_stats.py"),
                    "--data-root", a.data_root], check=True)
subprocess.run([sys.executable, str(ROOT / "tools" / "make_tables.py")], check=True)
