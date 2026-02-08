#!/usr/bin/env python3
"""Organize project structure"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Create directories
for d in ['examples', 'docs', 'visualization/images']:
    dir_path = ROOT / d
    dir_path.mkdir(exist_ok=True)
    print(f"Created: {d}")

# Move .py files to examples/
py_files = list(ROOT.glob('*.py'))
for py_file in py_files:
    if py_file.name in ['demo.py', 'test_n1000.py', 'visualize_heatmap.py']:
        dest = ROOT / 'examples' / py_file.name
        if not dest.exists():
            shutil.move(str(py_file), str(dest))
            print(f"Moved: {py_file.name} -> examples/")
        else:
            print(f"Exists: {examples}/{py_file.name}")

# Move .md files to docs/ (keep only key papers in root)
key_papers = ['CICADA_PAPER.md', 'README.md']
md_files = list(ROOT.glob('*.md'))
for md_file in md_files:
    if md_file.name not in key_papers:
        dest = ROOT / 'docs' / md_file.name
        if not dest.exists():
            shutil.move(str(md_file), str(dest))
            print(f"Moved: {md_file.name} -> docs/")

print("\nDone!")
