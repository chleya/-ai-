# 2026-02-07 - NONLINEARITY & MEMORY ANALYSIS

## Research Question
Can different nonlinearities create attractors with memory properties?

---

## FINDING: All Systems Quickly Forget Perturbations

### Experiment Design
```
Uncontrolled: x_{t+1} = φ(W x_t + noise) (with spectral normalization)

Nonlinearities tested:
    - tanh: symmetric, saturated
    - ReLU: asymmetric, unbounded
    - LeakyReLU: asymmetric, slight negative slope
    - Swish: non-monotonic, smooth
```

### Results (ρ = 0.9)

| Nonlinearity | Variance | Stability | Perturbation Effect |
|--------------|----------|-----------|-------------------|
| tanh | 0.1896 | 0.15 | -0.3% (forgot) |
| ReLU | 0.1024 | 0.28 | +0.2% (forgot) |
| LeakyReLU | 0.1116 | 0.27 | +0.2% (forgot) |
| Swish | 0.0739 | 0.28 | -0.0% (forgot) |

### Key Observations

1. **Different attractors confirmed**
   - tanh: v ≈ 0.19 (highest)
   - LeakyReLU: v ≈ 0.11
   - ReLU: v ≈ 0.10
   - Swish: v ≈ 0.07 (lowest)

2. **No persistent memory**
   - All systems forget perturbations within ~1000 steps
   - Perturbation effect: -0.3% to +0.2%
   - The attractor dominates all initial conditions

3. **Stability varies**
   - tanh: most stable (0.15)
   - Others: similar stability (~0.27)

---

## THEORETICAL IMPLICATIONS

### What This Means

1. **Attractors dominate dynamics**
   - Initial conditions don't matter
   - Perturbations decay
   - System always returns to attractor

2. **Nonlinearity shapes the attractor**
   - tanh → high-variance attractor
   - Swish → low-variance attractor
   - But none create "memory"

3. **To get memory, need structure**
   - Hopfield networks (energy minima)
   - Recurrent connections with slow dynamics
   - External memory mechanisms
   - Structural changes (connection growth/death)

### The Memory Problem

```
Current systems: No memory
- Perturbations decay within ~1000 steps
- System returns to attractor
- No information storage

Needed for memory:
    - Multiple attractors
    - Slow dynamics between attractors
    - Structural persistence
```

---

## THE BIG PICTURE

### What We Proved

| Question | Answer |
|----------|--------|
| Can we escape 0.28? | **Yes** (change φ) |
| Do we get memory? | **No** (all forget) |
| Can attractors be shaped? | **Yes** (by φ) |

### The Hierarchy

```
tanh: High variance (~0.19) → simple attractor
ReLU: Low variance (~0.10) → simple attractor
Swish: Lowest variance (~0.07) → simple attractor

All: Single attractor, no memory, quick forgetting
```

---

## NEXT DIRECTIONS

### To Get Memory, We Need:

1. **Multiple Attractors**
   - Hopfield-style energy landscape
   - Hebbian learning to create minima
   - Structural organization

2. **Slow Dynamics**
   - Multiple timescales
   - Slow variable adaptation
   - History-dependent updates

3. **Structural Persistence**
   - Connection growth/death
   - Hebbian plasticity
   - Synaptic scaling

---

## CONCLUSION

### What We Learned

| Finding | Evidence |
|---------|----------|
| Nonlineraity shapes attractor | Different φ → Different v |
| No natural memory | Perturbations decay in ~1000 steps |
| Attractor dominates | Initial conditions irrelevant |

### The Path Forward

```
To create systems with memory:
    1. Add Hebbian learning
    2. Create multiple attractors
    3. Add structural plasticity
    4. Combine with slow variables
```

---

## FILES

- `results/nonlinearity_controlled.json`: Raw data

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: Nonlinearaity changes the attractor, but no nonlinearity creates natural memory.*
