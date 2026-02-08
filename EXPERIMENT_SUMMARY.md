# Cicada Protocol - Experiment Summary
====================================

Generated: 2026-02-08T16:30:49.454615

## Core Experiments (examples/)

| File | Status |
|------|--------|
| demo.py | DONE |
| parameter_sweep.py | DONE |
| event_triggered.py | DONE |
| task_switching.py | DONE |
| consensus_quick.py | DONE |
| large_scale_fast.py | DONE |
| phase_transition_scan.py | DONE |

## Results (results/)

- large_scale_scan.json (KEY DATA)
- consensus_scalability.json
- 30+ legacy result files

## Visualizations (visualization/)

- consensus_quick.png
- event_triggered_comparison.png
- task_switching_comparison.png
- large_scale_scan.png
- phase_transition.png

## Documentation (docs/)

- THEORETICAL_FOUNDATION.md (KEY)
- PARAMETER_CLARIFICATION.md
- RESULTS.md
- 14+ other docs

## Key Findings

### Large Scale N Scan (N=50-2000)

| N | No Reset λ | Reset λ | Reduction |
|---|------------|---------|-----------|
| 50 | 0.228 | 0.228 | 0% |
| 100 | 0.339 | 0.339 | 0% |
| 200 | 0.659 | 0.659 | 0% |
| 400 | 1.187 | 0.399 | 66.4% |
| 600 | 1.532 | 0.488 | 68.1% |
| 800 | 1.815 | 0.565 | 68.9% |
| 1000 | 2.133 | 0.635 | 70.2% |
| 1500 | 2.838 | 0.776 | 72.7% |
| 2000 | 3.534 | 0.894 | 74.7% |

**Phase Transition**: N ≈ 400-800

### Consensus Experiment

- No reset: λ_max = 0.620
- With reset: λ_max = 0.433
- Reduction: 30%

## Reproducibility

```bash
# Run experiments
python examples/demo.py
python examples/large_scale_fast.py
python examples/event_triggered.py
python examples/task_switching.py
```

---
Generated: 2026-02-08T16:30:49.454615
