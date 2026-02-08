# 2026-02-07 - CEILING DISCOVERY REPORT

## Research Question
How high can positive feedback climb? Is there a ceiling?

---

## EXPERIMENT 04c: Climbing Limit Test

### Method
- Test initial targets: 0.020, 0.030, 0.040, 0.050, 0.060, 0.100
- Measure: Can the system climb above initial target?
- Positive feedback: variance ↑ → target ↑ (1% increase)

---

## RESULTS

| Initial Target | Climbed | Final Target | Drift | Variance |
|----------------|---------|--------------|--------|-----------|
| 0.020 | ✅ YES | 0.0278 | +39% | 0.0266 |
| 0.030 | ✅ YES | 0.0396 | +32% | 0.0358 |
| 0.040 | ✅ YES | 0.0464 | +16% | 0.0401 |
| 0.050 | ❌ NO | 0.0500 | 0% | 0.0475 |
| 0.060 | ❌ NO | 0.0600 | 0% | 0.0552 |
| 0.100 | ❌ NO | 0.1000 | 0% | 0.0781 |

---

## KEY FINDING: CEILING at ~0.05

### Evidence
1. **Targets below 0.05**: All climb (+16% to +39%)
2. **Targets at/above 0.05**: No climbing (0% drift)

### Pattern
```
Initial 0.020 → Final 0.0278 (+39%)
Initial 0.030 → Final 0.0396 (+32%)
Initial 0.040 → Final 0.0464 (+16%)
Initial 0.050 → Final 0.0500 (0%)
```

The percentage improvement DECREASES as target increases, suggesting a saturation point.

---

## INTERPRETATION

### Why Does Climbing Stop at ~0.05?

**Hypothesis**: The ceiling is where variance ≈ target

| Target | Variance | Ratio (Var/Target) | Feedback |
|--------|----------|-------------------|----------|
| 0.020 | 0.0266 | 1.33 | ↑ (var > target) |
| 0.030 | 0.0358 | 1.19 | ↑ (var > target) |
| 0.040 | 0.0401 | 1.00 | ~ (var ≈ target) |
| 0.050 | 0.0475 | 0.95 | ~ (var ≈ target) |
| 0.060 | 0.0552 | 0.92 | ~ (var < target) |

**Mechanism**:
- Positive feedback triggers ONLY when variance > target
- As target approaches ~0.05, variance cannot exceed it
- When variance ≈ target, feedback stops
- The system reaches equilibrium at the ceiling

### What This Means

1. **Positive feedback has limits**: Cannot drive infinite growth
2. **Natural equilibrium exists**: Around target=0.05, variance=0.05
3. **The ceiling is dynamic, not fixed**: It's where the feedback loop's gain equals the system's capacity

---

## THEORETICAL IMPLICATIONS

### The Feedback Loop

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    variance ──> compare to target ──> if var > target: target ↑
         │                                      │
         │                                      │
         └──────────────────────────────────────┘
```

At equilibrium: variance ≈ target
This is the ceiling.

### Why ~0.05?

Possible explanations:
1. **System capacity**: The dynamics can only sustain ~0.05 variance
2. **Regulation limit**: Alpha regulation balances at ~0.05
3. **Scale invariance**: The N=20 system has a characteristic scale

---

## SIGNIFICANCE

### What We Proved

1. **Positive feedback works** (from System 04b)
2. **Positive feedback has limits** (from this experiment)

### The Discovery

> "Positive feedback creates climbing behavior, but only until the system reaches its natural equilibrium point. At that point, variance ≈ target, and the feedback loop closes."

This is a fundamental result:
- The system can optimize (System 04)
- The optimization has limits (this experiment)
- The limit is where feedback equilibrium is reached

---

## NEXT QUESTIONS

1. **What determines the ceiling value (~0.05)?**
   - Is it N-dependent? (Test N=50, 100)
   - Is it σ-dependent? (Test higher noise)
   - Is it gain-dependent? (Test different feedback gains)

2. **Can the ceiling be raised?**
   - What if we increase feedback gain?
   - What if we add a secondary mechanism?

3. **What happens at the ceiling?**
   - Oscillation?
   - Stable equilibrium?
   - Bifurcation?

---

## FILES

- `results/ceiling_test.json`: Raw data
- `RESULTS_04B_COMPLETE.md`: System 04b results

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: Positive feedback has a natural ceiling at ~0.05.*
