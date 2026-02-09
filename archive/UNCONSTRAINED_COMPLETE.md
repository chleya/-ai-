# 2026-02-07 - UNCONSTRAINED DYNAMICS: COMPLETE REPORT

## Research Questions

1. What happens without the ||x|| = α constraint?
2. Is the 1/N law a constraint artifact?
3. What is the true equilibrium?

---

## FINDING 1: Variance is INDEPENDENT of N (Without Constraint)

### Results (σ = 0.5)

| N | Variance | Std |
|---|----------|-----|
| 10 | 0.2565 | 0.0862 |
| 20 | 0.2792 | 0.0633 |
| 50 | 0.2769 | 0.0388 |
| 100 | 0.2835 | 0.0273 |
| 200 | 0.2834 | 0.0196 |
| 300 | 0.2839 | 0.0161 |

### Scaling Analysis
```
log-log slope = 0.024 ≈ 0
```

**Conclusion**: Variance is INDEPENDENT of N!

### Comparison: Constraint vs Unconstrained

| Condition | N=50 | Ratio |
|-----------|-------|-------|
| Constrained (||x||=α) | 0.0040 | 1x |
| Unconstrained | 0.2769 | **69x** |

**The constraint was suppressing variance by ~69x!**

---

## FINDING 2: σ Dependence is Non-Linear

### Results (N=100)

| σ | Variance | σ²/2 | Error |
|---|----------|-------|-------|
| 0.1 | 0.071 | 0.005 | +1329% |
| 0.3 | 0.183 | 0.045 | +306% |
| 0.5 | 0.280 | 0.125 | +124% |
| 1.0 | 0.459 | 0.500 | -8% |
| 1.5 | 0.573 | 1.125 | -49% |
| 2.0 | 0.651 | 2.000 | -67% |

### Regime Analysis

1. **Low σ regime (σ < 0.5)**
   - tanh saturation dominates
   - E[tanh(z)²] >> σ²/2
   - Variance much higher than σ²/2 prediction

2. **High σ regime (σ > 1.0)**
   - Noise dominates
   - Variance approaches σ²/2
   - tanh saturates at 1

3. **Transition region (σ ≈ 1.0)**
   - Balanced regime
   - v ≈ 0.45, σ²/2 ≈ 0.50

---

## THEORETICAL IMPLICATIONS

### Self-Consistent Equation

Without constraint, the equilibrium satisfies:

```
v = E[tanh(z)²]    where z ~ N(0, v + σ²)
```

### Analytical Approximation

For small v, tanh(z) ≈ z - z³/3:

```
v ≈ (v + σ²) - (2/3)(v + σ²)²
```

Solving gives v ≈ σ²/2 for large σ.

But for small σ, the tanh nonlinearity creates effective amplification.

---

## REVISED UNDERSTANDING

### What We Misinterpreted

| Finding | With Constraint | Without Constraint |
|---------|-----------------|-------------------|
| 1/N law | v = α² / N (real) | **No 1/N scaling** |
| "Ceiling" | ~0.05 (constraint) | ~0.28 (natural) |
| "Climbing" | Compensation | May not apply |
| Stability | Forced | Natural |

### The Constraint Hierarchy

```
Unconstrained:     v ≈ 0.28 (constant, N-independent)
Weak constraint:   v ≈ 0.28 (minor correction)
Strong constraint: v = α²/N ≈ 0.004

Constraint can suppress variance by 30-100x!
```

---

## KEY EQUATIONS

### With Constraint (Original)
```
x_{t+1} = normalize(tanh(W·x_t + ξ_t), α)
variance = α² / N
```

### Without Constraint (New)
```
x_{t+1} = tanh(W·x_t + ξ_t)
v = E[tanh(z)²], z ~ N(0, v + σ²)
v ≈ 0.28 for σ = 0.5
```

---

## WHAT THIS MEANS

### For Previous Findings

1. **1/N law**: Was a constraint artifact, not natural dynamics
2. "Ceiling": Was the constraint ceiling, not dynamical limit
3. Positive feedback: Was compensation, not optimization

### For Future Work

1. **Study unconstrained dynamics**
   - Natural equilibrium is ~0.28
   - N-independent for N≥10

2. **Characterize σ dependence**
   - Low σ: tanh saturation effect
   - High σ: noise dominance

3. **Add feedback to unconstrained system**
   - May not work the same way
   - New equilibrium expected

---

## CONCLUSION

### What We Proved

| Question | Answer |
|----------|--------|
| Does system explode without constraint? | **No** |
| Is there a natural equilibrium? | **Yes, v ≈ 0.28** |
| Does variance scale with N? | **No** |
| Was 1/N a constraint artifact? | **Yes** |
| How much did constraint suppress? | **~69x** |

### The Big Picture

```
Before: v = α² / N (constraint-dominated)
After:  v = E[tanh(z)²] (dynamics-dominated)

The constraint was masking the true dynamics.
Without it, we see natural equilibrium at ~0.28.
```

---

## FILES

- `results/unconstrained_N_scan.json`: N scan data
- `results/sigma_dependence.json`: σ dependence data

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: Without constraint, the system has a natural equilibrium of ~0.28, independent of N.*
