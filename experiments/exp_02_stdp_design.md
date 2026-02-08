# Experiment 02: Weak STDP Constraint

## Research Question
Can the system develop self-adjustment capability under minimal STDP constraint?

## Core Hypothesis
> Current system: passively alive (norm constraint "holds it up")
> With weak STDP: system develops "self-optimization" ability
> If STDP expands survival region or increases variance stability → Qualitative breakthrough

## Design Philosophy
**"Minimal viable addition"**: Test "does it work" before "make it perfect"

---

## Implementation: WeakSTDP_v1

### Configuration
```python
class System02_STDP:
    """
    System 01 + minimal STDP: self-adjusting connectivity
    
    Core mechanism:
    - Every 50-100 steps, identify active connections
    - Apply weak positive strengthening to active connections
    - NO decay (very conservative first version)
    """
    
    # STDP Parameters
    STDP_INTERVAL: int = 50       # Apply every 50 steps
    LEARNING_RATE: float = 1e-4   # Very small (conservative)
    ACTIVITY_THRESHOLD: float = 0.1  # Connections above this get boosted
    
    # Constraint preservation
    NORM_STRENGTH: float = 1.0    # Keep original constraint
    NOISE_LEVEL: float = 0.11     # Use moderate noise
```

### Mechanism Detail
```
1. Run 50 steps of standard evolution
2. Compute activity: activity_j = mean(|state_i|) over time
3. Identify active connections: W_ij where activity_i * activity_j > threshold
4. Apply update: W_ij += lr * activity_i * activity_j
5. Renormalize W to preserve spectral properties
6. Continue evolution
```

### Why This Design?
| Feature | Rationale |
|---------|----------|
| Interval=50 | Balance: frequent enough to matter, rare enough to not disrupt |
| lr=1e-4 | Very conservative: "almost invisible" effect |
| No decay | First prove positive effect exists |
| Activity-based | Natural selection: strengthen what works |

---

## Comparison Protocol

### Baseline: System 01 (No STDP)
```
σ=0.11, α=1.0 → variance ~0.048, stable
```

### Test: System 02 (With STDP)
```
σ=0.11, α=1.0, STDP enabled → ?
```

### Expected Outcomes
| Result | Interpretation |
|--------|----------------|
| variance ↑, survival region ↑ | ✅ STDP works: self-organization observed |
| variance ↑, same survival | ⚠️ STDP helps but limited |
| No change | ❌ Activity-based strengthening not sufficient |
| variance ↓, survival ↓ | ⚠️ STDP destabilizes (too strong) |

---

## Experiment Protocol

### Phase 1: Single Point Test
```bash
cd F:/system_stability
python core/system_02_stdp.py --mode single --sigma 0.11 --alpha 1.0
```
- Compare: System 01 vs System 02 at same parameters
- Metrics: variance trajectory, final survival, attractor shape

### Phase 2: Survival Region Scan
```bash
python core/system_02_stdp.py --mode phase --sigma-range 0.01-1.0
```
- Map survival region with STDP enabled
- Compare: STDP-off vs STDP-on region size

### Phase 3: Boundary Zoom
```bash
python core/system_02_stdp.py --mode zoom --sigma 0.4-0.7 --step 0.01
```
- Focus on transition zone
- Test: Does STDP expand the "living" region?

---

## Success Criteria

**Minimal Success**: 
- System 02 variance > System 01 variance at same (σ, α)
- No destabilization observed

**Breakthrough Success**:
- Survival region expands by >10%
- New attractor morphology observed
- "Self-optimization" behavior confirmed

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| STDP too strong → destabilization | Use lr=1e-6 as fallback |
| No observable effect | Try lr=1e-3 as upper bound |
| Activity threshold wrong | Run hyperparameter sweep |

---

## Related Experiments
- **Predecessor**: `system_01_constrained.py` (no STDP)
- **Follow-up**: `system_03_multiscale.py` (if STDP works)

---

*Date: 2026-02-07*
*Motto: First prove the system can live, then discuss what it can do.*
