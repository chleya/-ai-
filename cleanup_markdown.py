#!/usr/bin/env python3
"""
Markdown Document Cleanup Script
================================

Identifies and archives redundant .md files while keeping core documents.

Keep pattern:
- README.md
- CICADA_PAPER.md
- papers/*.md
- *_THEORY.md
- *_REPORT.md (unique ones)

Archive pattern:
- FINAL_*.md
- COMPLETE_*.md
- GOLDEN_*.md
- DEEP_*.md
- *_FINAL.md
- *_COMPLETE.md
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Keep these patterns
KEEP_PATTERNS = [
    'README',           # README.md
    'THEORY',          # theories
    'ROADMAP',         # roadmap
    'PIVOT',           # pivot theory
    'PHASE_TRANSITION', # phase transition
    'EVENT_TRIGGERED',  # event triggered
    'STRESS_THRESHOLD', # stress threshold
    'SUBMISSION',      # submission plan
]

# Archive these patterns
ARCHIVE_PATTERNS = [
    'FINAL_',          # FINAL_*.md
    'COMPLETE_',       # COMPLETE_*.md
    'GOLDEN_',         # GOLDEN_*.md
    'DEEP_',           # DEEP_*.md
    '_FINAL',          # *_FINAL.md
    '_COMPLETE',        # *_COMPLETE.md
    'VERIFICATION',     # verification (keep only one)
    'SYNTHESIS',       # synthesis (keep only one)
]

# Valid papers directory
PAPERS_DIR = ROOT / 'papers'
PAPERS_DIR.mkdir(exist_ok=True)


def should_keep(filename: str) -> bool:
    """Check if file should be kept."""
    # Always keep README.md
    if filename == 'README.md':
        return True
    
    # Check keep patterns
    for pattern in KEEP_PATTERNS:
        if pattern in filename:
            return True
    
    return False


def should_archive(filename: str) -> bool:
    """Check if file should be archived."""
    # Check archive patterns
    for pattern in ARCHIVE_PATTERNS:
        if pattern in filename:
            return True
    return False


def cleanup():
    """Execute cleanup."""
    print("=" * 60)
    print("Markdown Cleanup Script")
    print("=" * 60)
    
    # Create archive directory
    archive_dir = ROOT / 'docs_archive'
    archive_dir.mkdir(exist_ok=True)
    
    md_files = list(ROOT.glob('*.md'))
    
    keep_count = 0
    archive_count = 0
    skip_count = 0
    
    for md_file in md_files:
        filename = md_file.name
        
        if should_keep(filename):
            print(f"  KEEP: {filename}")
            keep_count += 1
        elif should_archive(filename):
            # Move to archive
            new_path = archive_dir / filename
            if new_path.exists():
                # Rename if exists
                counter = 1
                while new_path.exists():
                    new_path = archive_dir / f"{md_file.stem}_{counter}{md_file.suffix}"
                    counter += 1
            
            shutil.move(str(md_file), str(new_path))
            print(f"  ARCHIVE: {filename}")
            archive_count += 1
        else:
            print(f"  SKIP: {filename}")
            skip_count += 1
    
    print()
    print("-" * 60)
    print(f"Summary:")
    print(f"  Kept: {keep_count}")
    print(f"  Archived: {archive_count}")
    print(f"  Skipped: {skip_count}")
    print()
    print(f"Archived files moved to: {archive_dir}")
    print()
    print("Next steps:")
    print("  1. Check docs_archive/ for archived files")
    print("  2. Review kept files")
    print("  3. git add -A && git commit -m 'Cleanup: archive redundant docs'")
    print("=" * 60)


if __name__ == '__main__':
    cleanup()
