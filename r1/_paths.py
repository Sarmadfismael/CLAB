"""Repository paths for the revision (R1) analyses. Every script imports this first.

Defaults: features from ../features, per-seed results from ../results, outputs to ./out.
Override with the environment variables CLAB_FEATURES, CLAB_RESULTS and R1_OUT."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('CLAB_FEATURES', str(ROOT / 'features'))
os.environ.setdefault('CLAB_RESULTS', str(ROOT / 'results'))
os.environ.setdefault('R1_OUT', str(HERE / 'out'))
FEAT, RESULTS, OUT = (Path(os.environ[k]) for k in ('CLAB_FEATURES', 'CLAB_RESULTS', 'R1_OUT'))
OUT.mkdir(parents=True, exist_ok=True)
