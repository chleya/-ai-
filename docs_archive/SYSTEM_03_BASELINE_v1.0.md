# System 03 Baseline v1.0 - GOLDEN REFERENCE

## Status: OFFICIAL BASELINE FOR FUTURE COMPARISONS

---

## Configuration (Best Performing)

```python
System03_Baseline = {
    # Core parameters
    'N': 20,                    # State dimension
    'T': 20000,                 # Evolution steps
    'seed': 42,                 # Standard seed for reproducibility
    
    # Dynamic alpha mechanism
    'alpha_init': 0.45,         # START in death zone (intentional)
    'alpha_min': 0.3,           # Minimum constraint
    'alpha_max': 2.0,           # Maximum constraint
    
    # Feedback regulation
    'desired_var': 0.015,       # Target variance
    'window': 500,             # Variance averaging window
    'gain': 0.01,              # Adaptation rate
    'beta': 0.9,               # EMA factor for variance
    
    # Perturbation test config
    'shock_time': 5000,        # When perturbation is applied
    'shock_type': 'state_shrink',
    'shock_factor': 0.2,       # State *= 0.2
}
```

---

## Golden Results (v1.0)

### Core Performance
| Metric | Value | Notes |
|--------|-------|-------|
| Final Variance | **0.0220** | 2.2x survival threshold |
| Converged Alpha | **0.685** | Stable across runs |
| Survival Rate | **100%** | From death zone rescue |
| Variance Improvement | **+130%** | vs Fixed α=0.45 |

### Convergence Properties
| Property | Value | Notes |
|----------|-------|-------|
| Convergence Alpha | 0.685 ± 0.002 | Very stable |
| Convergence Time | ~2000 steps | Fast adaptation |
| Alpha Range | 0.450 → 0.685 | Monotonic increase |

### Perturbation Recovery
| Test | Result | Standard |
|------|--------|----------|
| Recovery Ratio | 100.3% ± 0.4% | Target: >80% |
| Recovery Time | 1 step | Instantaneous |
| Alpha Boost | +0.004 ± 0.001 | Minimal adjustment |
| Success Rate | 10/10 (100%) | All seeds passed |

### Multi-Sigma Validation
| σ | Rescue | Converged α | Notes |
|---|--------|-------------|-------|
| 0.5 | ✅ | 0.685 | Standard |
| 1.0 | ✅ | 0.686 | Noise-independent |
| 2.0 | ✅ | 0.686 | Noise-independent |
| 3.0 | ✅ | 0.686 | Noise-independent |

---

## Mechanism Summary

### The Rule
```
Every step:
1. Evolve state with current alpha
2. Compute recent variance (window=500)
3. Calculate target_alpha = desired_var / recent_var
4. Update alpha += gain * (target_alpha - alpha)
5. Clip alpha to [alpha_min, alpha_max]
```

### Why It Works
1. **Feedback alignment**: Alpha tracks what variance level the system "needs"
2. **Stability through regulation**: System self-corrects when variance drops
3. **Optimal convergence**: ~0.685 emerges as the "natural" constraint strength

### Emergent Property
> **HOMEOSTASIS**: The system maintains its own existence through internal regulation.

---

## Usage Examples

### Basic Run
```python
from system_03_dynamic_alpha import SystemDynamicAlpha

system = SystemDynamicAlpha(
    sigma=0.5,
    alpha_init=0.45,  # Start in death zone
    desired_var=0.015,
    seed=42
)
result = system.run()
```

### Perturbation Test
```python
phases = [
    (0, 5000, {}),                                    # Normal
    (5000, 5100, {'perturbation': 'state_shrink'}), # Shock
    (5100, 20000, {}),                                # Recovery
]
result = system.run(phases=phases)
```

---

## Validation Checklist

- [x] Single run works
- [x] Multiple seeds (10+) all succeed
- [x] Multi-sigma validation (σ=0.5-3.0)
- [x] Perturbation recovery (100%+)
- [x] Cross-seed stability confirmed

---

## Known Limitations

1. **Dimension**: Only tested at N=20
2. **Perturbation range**: Tested up to state *= 0.2
3. **Noise range**: σ=0.5-3.0 tested

---

## For Comparison

When testing new mechanisms, compare against:

| Metric | System 03 v1.0 | Your Mechanism |
|--------|-----------------|----------------|
| Rescue from α=0.45 | 100% | ? |
| Final Variance | 0.0220 | ? |
| Converged Alpha | 0.685 | ? |
| Perturbation Recovery | 100% | ? |

---

*Date: 2026-02-07*
*Status: GOLDEN REFERENCE*
*Motto: First prove the system can live, then discuss what it can do.*
