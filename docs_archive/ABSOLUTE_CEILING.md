# 2026-02-07 - ABSOLUTE CEILING DISCOVERY

## Research Question
Is there a hard ceiling that even high gain cannot push beyond?

---

## EXPERIMENT 04f: Extreme Gain Test

### Method
- Gain: 10x, 15x, 20x
- Starting targets: 0.040, 0.050
- Protections: target clip [0.01, 0.25], variance brake >0.8
- Duration: T=20000

---

## RESULTS

### From target=0.040

| Gain | Final Target | Drift | Std |
|------|--------------|-------|-----|
| 10x | 0.0549 | +0.0149 | 0.000025 |
| 15x | 0.0552 | +0.0152 | 0.000009 |
| 20x | 0.0554 | +0.0154 | 0.000000 |

### From target=0.050

| Gain | Final Target | Drift | Std |
|------|--------------|-------|-----|
| 10x | 0.0550 | +0.0050 | 0.000020 |
| 15x | 0.0552 | +0.0052 | 0.000005 |
| 20x | 0.0555 | +0.0055 | 0.000000 |

---

## KEY FINDING: ABSOLUTE CEILING at ~0.055

### Evidence

1. **Convergence to same value**:
   - 10x: ~0.0549
   - 15x: ~0.0552
   - 20x: ~0.0554
   - **Difference: only ~0.0005** (negligible)

2. **All runs perfectly stable**:
   - std ≈ 0.00002-0.00000
   - No oscillation
   - Perfect convergence

3. **Different starting points converge**:
   - 0.040 → ~0.055
   - 0.050 → ~0.055

---

## INTERPRETATION

### Revised Ceiling Model

```
Previous: ceiling = f(gain) [higher gain = higher ceiling]
Discovery: ceiling = min(f(gain), ABSOLUTE_LIMIT~0.055)

When gain is high enough, other factors limit the ceiling.
```

### Limiting Factors (Speculation)

1. **System dynamics**: N=20 state space has intrinsic limits
2. **Alpha regulation**: α approaching upper bound (1.0-1.03)
3. **Variance-target balance**: Cannot sustain var > target indefinitely

---

## THEORETICAL IMPLICATIONS

### What This Means

| Question | Answer |
|----------|--------|
| Can positive feedback climb? | Yes (+33% from baseline) |
| Is there a ceiling? | Yes (~0.055) |
| Is ceiling adjustable? | Partially (gain affects 1x-8x) |
| Is there an absolute limit? | **Yes (~0.055)** |

### The Ceiling Hierarchy

| Regime | Gain | Ceiling | Behavior |
|--------|------|---------|----------|
| Low | 1x-3x | 0.047-0.052 | Gain ↑ → Ceiling ↑ |
| High | 10x-20x | ~0.055 | Gain ↑ → No effect |

---

## NEXT STEPS

### 1. Characterize the Absolute Limit
- Test N=50, N=100: Does absolute ceiling scale?
- Test σ=1.5: Is ceiling σ-dependent?

### 2. Theoretical Explanation
- Why ~0.055?
- Is it related to N, σ, or system eigenvalues?

### 3. System Behavior at Ceiling
- What happens at exactly 0.055?
- Is it a bifurcation point?

---

## FILES

- `results/extreme_gain_test.json`: Raw data

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: An absolute ceiling exists at ~0.055, even high gain cannot push beyond.*
