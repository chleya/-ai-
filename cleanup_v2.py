#!/usr/bin/env python3
"""
Markdown Cleanup - Simplified Version
===================================
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Files to KEEP (exact names)
KEEP_FILES = {
    'README.md',
    'CICADA_PAPER.md',
    'DYNAMICS_THEORY.md',
    'EVENT_TRIGGERED_RESET.md',
    'N1000_PHASE_TRANSITION.md',
    'N1000_THEORY.md',
    'PIVOT_THEORY.md',
    'PHASE_TRANSITION_REPORT.md',
    'STRESS_THRESHOLD_HEATMAP.md',
    'STRESS_THRESHOLD_SUMMARY.md',
    'STRESS_THRESHOLD_HEATMAP_ANALYSIS.md',
    'SUBMISSION_PLAN.md',
    'QUICKSTART.md',
}

# Directories to keep
KEEP_DIRS = {'papers'}

# Patterns that indicate duplicates (to archive)
DUPLICATE_PATTERNS = [
    'FINAL_',       # FINAL_*.md
    'COMPLETE_',    # COMPLETE_*.md  
    'GOLDEN_',      # GOLDEN_*.md
    'DEEP_',        # DEEP_*.md
    'VERIFICATION',  # *VERIFICATION*.md
    'SYNTHESIS',    # *SYNTHESIS*.md
    '_FINAL',       # *_FINAL.md
    '_COMPLETE',    # *_COMPLETE.md
    '_FINAL',       # *_FINAL.md
]

# Files that are NOT research papers (skip)
NON_PAPERS = {
    'AGENTS.md',
    'SOUL.md',
    'USER.md',
    'PROJECT_MANIFESTO.md',
    'PROJECT_SUMMARY.md',
    'PROJECT_OPTIMIZATION.md',
    'RESEARCH_HISTORY.md',
    'RESEARCH_PLANS.md',
    'DOCUMENTATION_PLAN.md',
    'CONSTRAINT_REVISION.md',
    'CORRECTIONS.md',
    'THEORETICAL_DEEPENING.md',
    'THEORETICAL_DERIVATION.md',
}


def is_duplicate(filename: str) -> bool:
    """Check if filename indicates a duplicate."""
    for pattern in DUPLICATE_PATTERNS:
        if pattern in filename:
            return True
    return False


def cleanup():
    """Execute cleanup."""
    print("=" * 60)
    print("Markdown Cleanup - Simplified")
    print("=" * 60)
    
    # Create archive directory
    archive_dir = ROOT / 'docs_archive'
    archive_dir.mkdir(exist_ok=True)
    
    md_files = list(ROOT.glob('*.md'))
    
    keep = 0
    archive = 0
    skip = 0
    
    for md_file in md_files:
        name = md_file.name
        
        # Check if should keep
        if name in KEEP_FILES:
            print(f"  KEEP: {name}")
            keep += 1
            continue
        
        # Check if non-paper (internal config, skip)
        if name in NON_PAPERS:
            print(f"  SKIP: {name} (internal)")
            skip += 1
            continue
        
        # Check if duplicate pattern
        if is_duplicate(name):
            dest = archive_dir / name
            if dest.exists():
                # Rename with counter
                i = 1
                while (archive_dir / f"{md_file.stem}_{i}{md_file.suffix}").exists():
                    i += 1
                dest = archive_dir / f"{md_file.stem}_{i}{md_file.suffix}"
            shutil.move(str(md_file), str(dest))
            print(f"  ARCHIVE: {name}")
            archive += 1
            continue
        
        # Default: archive most other .md files in root
        dest = archive_dir / name
        if dest.exists():
            i = 1
            while (archive_dir / f"{md_file.stem}_{i}{md_file.suffix}").exists():
                i += 1
            dest = archive_dir / f"{md_file.stem}_{i}{md_file.suffix}"
        shutil.move(str(md_file), str(dest))
        print(f"  ARCHIVE: {name}")
        archive += 1
    
    print()
    print("-" * 60)
    print(f"Summary:")
    print(f"  Kept: {keep}")
    print(f"  Archived: {archive}")
    print(f"  Skipped: {skip}")
    print()
    print(f"Archive location: {archive_dir}")
    print()
    print("To sync to GitHub:")
    print("  git add -A && git commit -m 'Cleanup: archive redundant docs' && git push")


if __name__ == '__main__':
    cleanup()
