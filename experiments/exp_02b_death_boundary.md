# Experiment 02b: Death Boundary Scan with STDP

## Research Question
Can STDP expand the survival boundary or delay death in critical parameter regions?

## Core Hypothesis
> In "easy survival" zones (σ ≤ 0.7), weak STDP has negligible effect.
> At the TRUE DEATH BOUNDARY (σ = 0.75-1.0), STDP might show:
> - Delayed death (live longer)
> - Expanded survival range (rescue dying points)
> - Changed death mode (sudden crash → slow decay)
> - Variance boost near criticality

---

## Experimental Design

### Parameter Grid
```
Sigma (σ): 0.75 ~ 1.0, step = 0.025 (11 points)
Alpha (α): 0.8, 1.0, 1.2 (3 values)
Total: 11 × 3 = 33 points
```

### Configuration Matrix

| Test ID | STDP | Learning Rate | Purpose |
|---------|------|---------------|---------|
| Baseline | ❌ | N/A | True death boundary |
| STDP-weak | ✅ | 1e-4 | Current conservative |
| STDP-mid | ✅ | 3e-4 | Mild enhancement |
| STDP-strong | ✅ | 1e-3 | Strong enhancement |

### Protocol per Point
```
1. Run System 01 (No STDP) - 3 repeats
2. Run System 02 (STDP-weak) - 3 repeats  
3. Run System 02 (STDP-mid) - 3 repeats
4. Run System 02 (STDP-strong) - 3 repeats
```

### Metrics
- **Survival Rate**: % of runs where variance > 0.01 at T=10000
- **Final Variance**: Mean variance in last 1000 steps
- **Death Time**: First step where variance < 0.005 (if applicable)
- **Stability**: std(final_variance) - lower = more stable

---

## Expected Outcomes

### If STDP is Useful
| Observation | Interpretation |
|-------------|----------------|
| Survival rate ↑ at same σ | STDP rescues dying systems |
| Death time ↑ (later crash) | STDP delays death |
| Variance ↑ near boundary | STDP maintains structure |
| Multiple survival modes | New attractor emerges |

### If STDP is Useless
| Observation | Interpretation |
|-------------|----------------|
| Identical curves across tests | STDP doesn't affect existence |
| Stronger STDP = same effect | Mechanism not suitable |
| No rescue at boundary | Need different mechanism |

---

## Execution Command

```bash
cd F:/system_stability
python core/system_02_death_boundary.py \
    --sigma-min 0.75 \
    --sigma-max 1.0 \
    --sigma-step 0.025 \
    --alphas 0.8,1.0,1.2 \
    --repeats 3
```

---

## Success Criteria

**Breakthrough**: STDP expands survival boundary by >5% in σ space
**Positive Signal**: Any measurable improvement in survival metrics
**Null Result**: No difference → move to alternative mechanisms

---

## Related Experiments
- **Predecessor**: `exp_02_stdp_design.md` (initial STDP test)
- **Alternative**: `exp_02c_classic_stdp.md` (if this fails)
- **Fallback**: `exp_02d_dynamic_norm.md` (if STDP useless)

---

*Date: 2026-02-07*
*Motto: First prove the system can live, then discuss what it can do.*
