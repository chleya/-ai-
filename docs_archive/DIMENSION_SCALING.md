# 2026-02-07 - DIMENSION SCALING DISCOVERY

## Research Question
Does the absolute ceiling (~0.055) scale with dimension N?

---

## EXPERIMENT 04g: Dimension Scaling Test

### Method
- N: 20, 50, 100
- Gain: 3x, 10x
- σ: 0.5
- T: 15000

---

## INITIAL RESULTS (Confusing)

| N | Gain | Final Target | Drift |
|---|------|--------------|-------|
| 20 | 3.0 | 0.0528 | +0.0128 |
| 20 | 10.0 | 0.0561 | +0.0161 |
| 50 | 3.0 | 0.0400 | 0.0 |
| 50 | 10.0 | 0.0400 | 0.0 |
| 100 | 3.0 | 0.0400 | 0.0 |
| 100 | 10.0 | 0.0400 | 0.0 |

**Puzzle**: Why no climbing at N=50/100?

---

## DIAGNOSIS: Variance Decreases with N

| N | Variance | Target | Ratio | Feedback |
|---|----------|--------|-------|----------|
| 20 | 0.0459 | 0.0453 | 1.01 | TRIGGERS |
| 50 | 0.0315 | 0.0400 | 0.79 | NO TRIGGER |
| 100 | 0.0251 | 0.0400 | 0.63 | NO TRIGGER |

---

## KEY FINDING: DIMENSION-DEPENDENT VARIANCE

### Evidence
1. **Higher N = Lower variance**
   - N=20: variance ~0.046
   - N=50: variance ~0.032 (-30%)
   - N=100: variance ~0.025 (-45%)

2. **Feedback only triggers when variance > target**
   - At N=20: variance > target → feedback triggers → ceiling climbs
   - At N=50/100: variance < target → no feedback → no climbing

### Interpretation

```
Previous hypothesis: ceiling = fixed value (~0.055)
Discovery: ceiling = f(N), and at N=50/100, system doesn't even try to climb

The "ceiling" at N=20 is actually the SYSTEM'S EQUILIBRIUM POINT.
At higher N, the equilibrium is LOWER, so no climbing occurs.
```

---

## THEORETICAL IMPLICATIONS

### What We Learned

| Question | Answer |
|----------|--------|
| Does ceiling scale with N? | **No, equilibrium point changes with N** |
| Why no climbing at N=50/100? | **variance < target at equilibrium** |
| What determines the equilibrium? | **System dynamics (W eigenvalues, tanh)** |

### New Model

```
At N=20: equilibrium variance ~0.046 → above initial target → climbs to ~0.055
At N=50: equilibrium variance ~0.032 → below initial target → no climb
At N=100: equilibrium variance ~0.025 → below initial target → no climb
```

The ceiling is not a "limit" but the **natural equilibrium** of the system.

---

## DIMENSION SCALING LAW

### Observed Pattern

```
N    equilibrium_variance
20   ~0.046
50   ~0.032
100  ~0.025
```

**Hypothesis**: equilibrium_variance ∝ 1/N or log(N)

### Implication

The system self-regulates to different equilibrium points depending on dimension.
Higher dimensions = more stable = lower variance.

---

## NEXT STEPS

### 1. Characterize the N-variance Relationship
- Test N=30, 40, 60, 80
- Fit functional form: variance = f(N)

### 2. Theoretical Derivation
- Use mean-field theory to predict variance vs N
- Relate to W eigenvalue distribution

### 3. Implications for System Design
- Higher N = more stable system
- Lower N = more dynamic/variable system

---

## FILES

- `results/dimension_test.json`: Raw data
- `ABSOLUTE_CEILING.md`: Previous ceiling discovery

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: The "ceiling" is actually the system's natural equilibrium, which decreases with dimension.*
