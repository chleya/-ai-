# 2026-02-07 - STRESS TEST RESULTS

## Experiment 03c: How Fragile is Homeostasis?

---

## Stress Test Suite Results

### Quick Tests (5 conditions)

| Test | Recovery | Robustness |
|------|----------|------------|
| Baseline | 100.5% | ROBUST ✅ |
| State=0 (total annihilation) | 100.5% | ROBUST ✅ |
| State*=0.01 | 100.5% | ROBUST ✅ |
| Alpha=0.3 (500 steps) | 100.5% | ROBUST ✅ |
| Sigma 5x (σ=0.5→5.0) | 100.4% | ROBUST ✅ |

**Summary**: All quick tests ROBUST

---

## Finding the LIMIT

### Ultimate Tests

| Condition | Result | Notes |
|-----------|--------|-------|
| State=0 from t=0 | **ALIVED** | System regenerated from zero |
| State*=0.001 | ALIVED | Near-zero recovery |
| Sigma 10x (σ=0.5→10.0) | **ALIVED** | Extreme noise |
| Sigma 20x (σ=0.5→20.0) | **ALIVED** | Survived! |

**Finding**: System is EXTREMELY robust to transient perturbations

---

## The TRUE Limit

### Fixed Alpha Survival Boundary

| Alpha | Variance | Status |
|-------|----------|--------|
| 0.25 | 0.0029 | DEAD |
| 0.28 | 0.0037 | DEAD |
| 0.30 | 0.0042 | DEAD |
| 0.32 | 0.0048 | DEAD |
| 0.35 | 0.0058 | DEAD |

**Critical Finding**: Even with dynamic alpha, the system CANNOT survive below α ≈ 0.35

### Why?

Because our alpha_min = 0.3, and when variance drops:
- target_alpha = desired_var / recent_var → goes very high
- But alpha is CLIPPED at 0.3 (alpha_min)
- So the system STARVES and dies

---

## Key Insight: The Floor is the Limit

```
Homeostasis is robust to TRANSIENT shocks...
...but COLLAPSES when the constraint floor is reached.

          ┌─────────────────────────────────────┐
          │           TRANSIENT SHOCKS          │
          │  (state=0, state*=0.01, σ jump)   │
          │                                     │
          │  Recovery: 100%                     │
          │  Alpha: Recovers quickly            │
          │  Robustness: EXTREME                │
          └─────────────────────────────────────┘
                         │
                         ▼
          ┌─────────────────────────────────────┐
          │         SUSTAINED DEPRESSION         │
          │  (alpha stuck < 0.35)               │
          │                                     │
          │  Recovery: 0%                       │
          │  Alpha: Floor hit                   │
          │  Robustness: FRAGILE                │
          └─────────────────────────────────────┘
```

---

## Interpretation

### What's NOT a Problem

1. **State annihilation**: System regenerates from zero
2. **Noise spikes**: Even σ=20 is fine
3. **Transient damage**: State*=0.01 recovers instantly

### What IS a Problem

1. **Sustained constraint weakness**: If α is forced < 0.35 for too long
2. **Alpha floor hitting**: The clipping at α_min is the kill switch
3. **Starvation**: System can't maintain variance with weak constraint

---

## Implications

### For System Design

1. **Increase alpha_min**: To make system more robust (but limits exploration)
2. **Remove alpha clipping**: But this might cause instability
3. **Multi-timescale regulation**: Fast response + slow recovery

### For Understanding Homeostasis

1. **Homeostasis is bounded**: Has physical limits
2. **Feedback can fail**: When the regulation mechanism is saturated
3. **The floor matters**: Clipping determines survival boundary

---

## The Deep Question

> "If the system can recover from total annihilation (state=0), 
> why can't it survive a sustained weak constraint?"

Answer:
- From zero, noise immediately generates variance
- Alpha can then grow from 0.45 upward
- But if alpha is FORCED to stay low, NOISE CANNOT GENERATE variance fast enough
- The system STARVES

This is a fundamental principle of dynamical systems.

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: Homeostasis is robust to shocks, fragile to sustained deprivation.*
