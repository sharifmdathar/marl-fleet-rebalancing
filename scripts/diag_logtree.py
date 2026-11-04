#!/usr/bin/env python3
"""Inspect the eval log directory tree under one run to learn what files exist."""

from pathlib import Path
import sys

RUN = Path(sys.argv[1])
print(RUN, '->', RUN.stat().st_size, 'bytes')
for p in sorted(RUN.rglob('*')):
    if p.is_file():
        print(f'  {p.relative_to(RUN)}  {p.stat().st_size}')