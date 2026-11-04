#!/usr/bin/env python3
"""Launch a focused diagnostic: what does the model actually see at decision time.

Reads one seed from the eval log, prints:
- per-step observation snapshot (state) for the first few steps (or a specific step)
- per-step action probabilities / logits for the chosen actions
- per-step mask structure
- how many drivers could actually dispatch in each step
- how many steps had pending orders
"""

from pathlib import Path
import json, numpy as np, numpy.typing as npt
from collections import defaultdict
import sys

RUN = Path(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1002
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 0

log = list(RUN.glob('*.jsonl'))
log = [p for p in log if p.name.startswith('eval')]
if not log:
    print('no eval log found in', RUN)
    sys.exit(1)
log = log[0]

rows = [json.loads(l) for l in Path(log).read_text().splitlines() if l.strip()]
print(f'read {len(rows)} eval rows from {log}')

# separate per-step obs
obs_rows = [r for r in rows if 'step' in r and 'obs' in r]

# keys
if obs_rows:
    print()
    print('obs keys:', sorted(obs_rows[0].keys()))
    sample = obs_rows[0]['obs']
    if isinstance(sample, dict):
        print('obs sub-keys:', sorted(sample.keys()))
        for k2,v2 in sample.items():
            print(' ', k2, '->', type(v2).__name__, (_shape(v2) if isinstance(v2,(list,tuple,np.ndarray)) else v2))
    else:
        print('obs value type:', type(sample).__name__, list(sample.shape) if hasattr(sample,'shape') else sample)

def _shape(v):
    if isinstance(v, np.ndarray):
        return list(v.shape)
    if isinstance(v, (list, tuple)):
        try:
            return [len(v)] + _shape(v[0])
        except Exception:
            return [len(v)]
    return v
PY