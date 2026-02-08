# 2026-02-07 - Triple Node Democratic Consensus Experiment

## Experiment Design

### Configuration
- **Node A**: 30% noise (clear, holds truth)
- **Node B**: 80% noise (fuzzy, confused)
- **Node C**: 80% noise + WRONG pattern (stubborn, opposing)

### Topology
Fully connected: A↔B↔C (triangular coupling)

### Hypothesis
Can the correct node (A) sway the group despite B being confused and C opposing?

---

## Results Summary

### Alpha Comparison

| Alpha | Consensus Rate | Correct Rate | Interpretation |
|-------|---------------|--------------|---------------|
| 0.1 | 100% | 20% | Always agree, usually wrong |
| 0.2 | 100% | 20% | Same |
| 0.3 | 50% | 30% | Less consensus, more correct |

### Key Finding

**Lower alpha (0.1-0.2)**: 
- 100% consensus rate
- But 80% consensus is WRONG
- Stubborn node C dominates

**Higher alpha (0.3)**:
- Only 50% consensus
- But 30% of those are CORRECT
- Group sometimes resists wrong influence

---

## Detailed Analysis

### Observation 1: Stubborn Node Dominance

When alpha is low (0.1-0.2), the system always reaches consensus, but:
- A (30% noise) provides weak signal
- B (80% noise) is easily swayed
- C (wrong attractor) provides strong counter-signal

Result: Group collapses to C's wrong attractor.

### Observation 2: Alpha Trade-off

| Alpha | Pro | Con |
|-------|-----|-----|
| Low (0.1) | Fast consensus | Wrong consensus |
| High (0.3) | Sometimes correct | No consensus |

### Observation 3: Civil War State

At alpha=0.3, we sometimes observe:
```
[+1.0, +1.0, -1.0]  (A and B agree, C opposes)
[+1.0, -1.0, +1.0]  (A and C agree, B confused)
[-1.0, +1.0, -1.0]  (B and C agree against A)
```

This is "civil war" - the group cannot unify.

---

## Physical Interpretation

### Why Stubborn Node Wins

1. **Attractor Strength**: C learned -P strongly
2. **Signal Imbalance**: A's correct signal is diluted by B's noise
3. **Majority Effect**: 2vs1 situation favors C

### Why Higher Alpha Helps Sometimes

1. **Autonomy Preservation**: Each node maintains its own attractor
2. **Selective Agreement**: Nodes only agree when truly aligned
3. **Truth Resistance**: Correct attractor can resist wrong influence

---

## Theoretical Implications

### For Swarm Intelligence

| Discovery | Meaning |
|------------|---------|
| Stubborn node dominates | Majority doesn't mean correct |
| Alpha determines behavior | Coupling strength is critical |
| Civil war is possible | Consensus is not guaranteed |

### For Democratic Systems

```
Democratic Problem: How to reach correct consensus?
Our Finding: Pure democracy fails with stubborn minority.

Solution needed:
- Weighted voting (A's vote counts more)
- Reputation systems (A has higher trust)
- Iterated interaction (A can convince C over time)
```

---

## Next Steps

### Experiment 1: Weighted Voting
Give A's signal higher weight:
```
x_A' = x_A + beta * x_A (beta > 0)
```

### Experiment 2: Reputation System
A's coupling to others is stronger:
```
x_B = W_B @ x_B + alpha_correct * x_A + alpha_wrong * x_C
```

### Experiment 3: Iterated Debate
Multiple rounds of coupling with memory:
```
Round 1: Initial coupling
Round 2: Update based on Round 1 results
Round 3: Final consensus
```

---

## Files

- `core/triple_node_consensus.py`: Full experiment code
- `results/`: Raw data files

---

## Conclusion

### Key Insight

> "Democratic consensus is not guaranteed to be correct. Stubborn minorities can dominate when the majority is uncertain."

### Implications for AI

1. **Majority voting is insufficient**: Need weighted or reputation-based systems
2. **Coupling strength matters**: Alpha is a critical hyperparameter
3. **Civil war is a state**: Systems can get stuck in disagreement

### Path Forward

To achieve correct consensus, we need:
1. **Trust calibration**: Identify and trust correct nodes
2. **Iterated interaction**: Allow time for truth to emerge
3. **Structural bias**: Favor truth-attractors over noise

---

*Experiment completed 2026-02-07*
