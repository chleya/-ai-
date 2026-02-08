#!/usr/bin/env python3
"""
Final Cleanup - Move internal config docs to archive
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Internal/non-research docs to archive
INTERNAL_DOCS = {
    'AGENTS.md',
    'CONSTRAINT_REVISION.md',
    'CORRECTIONS.md',
    'DOCUMENTATION_PLAN.md',
    'PROJECT_MANIFESTO.md',
    'PROJECT_OPTIMIZATION.md',
    'PROJECT_SUMMARY.md',
    'RESEARCH_HISTORY.md',
    'RESEARCH_PLANS.md',
    'SOUL.md',
    'THEORETICAL_DEEPENING.md',
    'THEORETICAL_DERIVATION.md',
    'USER.md',
}

# Keep only these research papers
KEEP_RESEARCH = {
    'README.md',
    'CICADA_PAPER.md',
    'QUICKSTART.md',
    'DYNAMICS_THEORY.md',
    'EVENT_TRIGGERED_RESET.md',
    'N1000_PHASE_TRANSITION.md',
    'N1000_THEORY.md',
    'PHASE_TRANSITION_REPORT.md',
    'PIVOT_THEORY.md',
    'STRESS_THRESHOLD_HEATMAP.md',
    'STRESS_THRESHOLD_HEATMAP_ANALYSIS.md',
    'STRESS_THRESHOLD_SUMMARY.md',
    'SUBMISSION_PLAN.md',
}


def final_cleanup():
    """Final cleanup - keep only essential research docs."""
    print("=" * 60)
    print("Final Cleanup - Keep Only Research Papers")
    print("=" * 60)
    
    archive_dir = ROOT / 'docs_archive'
    archive_dir.mkdir(exist_ok=True)
    
    md_files = list(ROOT.glob('*.md'))
    
    kept = 0
    archived = 0
    
    for md_file in md_files:
        name = md_file.name
        
        if name in KEEP_RESEARCH:
            print(f"  KEEP: {name}")
            kept += 1
        else:
            dest = archive_dir / name
            if dest.exists():
                i = 1
                while (archive_dir / f"{md_file.stem}_{i}{md_file.suffix}").exists():
                    i += 1
                dest = archive_dir / f"{md_file.stem}_{i}{md_file.suffix}"
            shutil.move(str(md_file), str(dest))
            print(f"  ARCHIVE: {name}")
            archived += 1
    
    print()
    print("-" * 60)
    print(f"Final Result:")
    print(f"  Kept: {kept}")
    print(f"  Archived: {archived}")
    print(f"  Archive: {archive_dir}")
    print()
    print("Remaining in root:")
    for f in sorted(ROOT.glob('*.md')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    final_cleanup()
