# 2026-02-07 - Sparse Hebbian & Bridge Experiment Reports

## Part 1: Sparse Hebbian - Negative Result

### Experiment: 80% Sparsity vs Interference Rate

**Hypothesis**: Cutting 80% of weak connections will reduce interference from 71% to <40%.

### Results

| Correlation | Full Connection | 80% Sparse | Change |
|-------------|-----------------|------------|--------|
| 0.0 | 40% | 46% | +6% |
| 0.3 | 42% | 48% | +6% |
| 0.6 | 44% | 50% | +6% |
| 0.9 | 48% | 52% | +4% |

### Key Finding

**Sparsity INCREASES interference, not decreases.**

### Physical Interpretation

1. **Signal-to-Noise Ratio Decline**:
   - Cutting 80% of weights removes structural redundancy
   - Remaining 20% cannot form stable basin boundaries
   - System becomes more susceptible to drift

2. **Basin Boundary Blurring**:
   - Hebbian learning creates GLOBAL attractors
   - Random pruning cannot split basins
   - All patterns still compete for the same weight space

### Conclusion

**Negative Result**: Simple pruning cannot achieve "logical isolation."

**Insight**: Hebbian attractors are fundamentally global. To reduce interference, we need structural separation (multiple nodes) rather than local pruning.

---

## Part 2: Bridge Communication - Positive Result

### Experiment: Can Node A Rescue Node B?

**Design**:
- Node A: Trained on pattern P (confident attractor)
- Node B: Reset to 90% noise (lost)
- Coupling: x_B(t+1) = tanh(W_B @ x_B + α · x_A)

### Alpha Sweep Results

| Alpha (Coupling) | Success Rate |
|------------------|-------------|
| 0.05 | 60% |
| 0.10 | 70% |
| **0.20** | **80%** |
| 0.30 | 60% |
| 0.50 | 80% |
| 1.00 | 80% |

### Optimal Configuration

- **Alpha**: 0.20
- **Success Rate**: 80%
- **Noise Level**: 90%

### Key Finding

**Bridge mechanism works! Node A can rescue Node B from 90% noise with 80% success.**

### Mechanism Analysis

1. **Coupling Injection**: Node A's confident state provides a "signal anchor"
2. **Basin Reinforcement**: Coupled input strengthens Node B's attractor
3. **Information Transfer**: Pattern P is communicated through coupling

### Theoretical Implication

```
Before: Single node stores 7 patterns → 71% interference
After:  Multiple nodes communicate → 80% rescue success

This is the path to "Evolutionary Intelligence Swarm"
```

---

## Part 3: Strategic Summary

### What We Learned

| Experiment | Result | Implication |
|------------|--------|-------------|
| Sparse Hebbian | Negative (+6% interference) |单体优化失败 |
| Bridge Communication | Positive (80% success) |群体协作成功 |

### Path Forward

**From**: "Make single node smarter"
**To**: "Connect multiple nodes to share intelligence"

### Next Steps

1. **Multi-node rescue**: Test 3+ nodes rescuing each other
2. **Bidirectional coupling**: A↔B mutual rescue
3. **Node competition**: Which attractor dominates?
4. **Swarm intelligence**: Collective decision-making

---

## Files

- `results/sparsity_sweep.json`: Sparsity test data
- `results/bridge_systematic.json`: Bridge test data
- `results/alpha_sweep.json`: Optimal coupling data

---

*Report generated from experimental results*
