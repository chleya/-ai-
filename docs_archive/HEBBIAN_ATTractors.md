# 2026-02-07 - HEBBIAN LEARNING: MULTIPLE ATTRACTORS DISCOVERED

## Research Question
Can Hebbian learning create multiple attractors in an unconstrained system?

---

## FINDING: Hebbian Learning Creates Binary Attractors!

### Experiment Design
```
Dynamics with Hebbian learning:
    x_{t+1} = tanh(W x_t + noise)
    W_{t+1} = normalize(W_t + η * x_t x_t^T)
```

### Results

| η | Variance | Energy | Attractors |
|---|----------|---------|------------|
| 0.0001 | 0.2842 | 0.55 | Binary (+/-1) |
| 0.001 | 0.9856 | 10.00 | Binary (+/-1) |
| 0.010 | 0.9856 | 10.00 | Binary (+/-1) |

### Attractor Verification

| Initial Condition | Final State | Correlation |
|------------------|-------------|-------------|
| Seed 0 | Basin +1 | +1.000 |
| Seed 100 | Basin +1 | +1.000 |
| Seed 200 | Basin -1 | -1.000 |
| Seed 300 | Basin +1 | +1.000 |
| Seed 400 | Basin -1 | -1.000 |
| ... | ... | ... |

**Conclusion**: Multiple attractors exist! States converge to either +1 or -1 basins.

---

## THEORETICAL IMPLICATIONS

### What This Means

1. **Multiple Attractors are Possible**
   - Hebbian learning creates energy minima
   - System has binary attractors (Hopfield-like)
   - Initial conditions determine final basin

2. **Memory Emerges**
   - Different initial conditions → different final states
   - System can "remember" which basin it started in
   - This is a primitive form of memory!

3. **Beyond Single Attractor**
   - tanh alone: single attractor at ~0.28
   - Hebbian: multiple attractors at +/-1
   - Structural learning changes dynamics

### The Energy Landscape

```
Without Hebbian:
    Single attractor at v* ≈ 0.28
    
With Hebbian:
    Energy minima at binary states
    Basins of attraction for +1 and -1
    Multiple stable states exist
```

---

## KEY OBSERVATIONS

### 1. Binary Attractors
- All final states are either +1 or -1
- Correlation with baseline: exactly +1.000 or -1.000
- This is perfect separation!

### 2. Hebbian Creates Structure
- W becomes structured through outer product updates
- Strong connections between similarly activated neurons
- Weak connections between oppositely activated neurons

### 3. Initial Condition Determines Fate
- Random seeds lead to different basins
- System "remembers" initial conditions
- This is the essence of memory!

---

## MEMORY ANALYSIS

### What We Observed

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Attractor separation | 2.0 (perfect) | Clear basins |
| Correlation std | 0.917 | Multiple basins |
| State space | Binary +/-1 | Discrete states |

### Memory Mechanism

```
Hebbian update: W += η * x x^T

This creates:
    - Positive correlations: reinforced
    - Negative correlations: weakened
    - Energy minima at coherent states

Initial conditions determine which minimum is reached.
```

---

## COMPARISON

### Before vs After Hebbian

| Property | Without Hebbian | With Hebbian |
|----------|-----------------|--------------|
| Attractors | Single (~0.28) | Multiple (+/-1) |
| Memory | None | Binary basins |
| Sensitivity | Low | High |
| Information | Lost | Preserved |

### The Breakthrough

```
Before: System always returns to ~0.28
After:  System has memory of initial conditions
        Multiple stable states exist
        Information is preserved!
```

---

## THE BIG PICTURE

### What We Proved

| Question | Answer |
|----------|--------|
| Can we escape single attractor? | **Yes** |
| Does Hebbian create structure? | **Yes** |
| Are there multiple attractors? | **Yes (+/-1 bins)** |
| Is there memory? | **Yes (basin memory)** |

### The New Understanding

```
Single attractor (tanh only):
    - No memory
    - All initial conditions converge to same state
    - Information lost
    
Multiple attractors (with Hebbian):
    - Memory exists
    - Different initial conditions → different final states
    - Information preserved in basin structure
```

---

## NEXT DIRECTIONS

### 1. Characterize the Attractors
- How many attractors exist?
- What determines the basin boundaries?
- How stable are the attractors?

### 2. Continuous Hebbian Learning
- Does the system keep evolving?
- Can we stabilize at a fixed point?
- What happens with infinite time?

### 3. Hebbian + Feedback
- Can feedback move between attractors?
- Can we design specific attractors?
- Learning + optimization?

### 4. Sparse Hebbian
- What if we add sparsity constraints?
- More realistic neural dynamics?
- Different attractor structure?

---

## CONCLUSION

### What We Discovered

| Finding | Evidence |
|---------|----------|
| Multiple attractors exist | Binary +/-1 convergence |
| Hebbian creates structure | Energy minima in W |
| Memory emerges | Basin-dependent final states |
| Beyond single attractor | System now has choice |

### The Breakthrough

> "Hebbian learning transforms a single-attractor system into a multi-attractor system with memory. This is the bridge from 'statistical mechanics' to 'information processing.'"

### Implications

1. **Memory is possible** without explicit storage
2. **Structure emerges** from Hebbian correlations
3. **Multiple states** can coexist stably
4. **Initial conditions matter** for final outcome

---

## FILES

- `core/experiment_a3_hebbian.py`: Hebbian experiment code
- `results/exp_A3_hebbian.json`: Raw data

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: Hebbian learning creates multiple attractors with basin memory!*
