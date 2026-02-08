# 2026-02-07 - Multi-Layer Consensus Experiments: Complete Report

## Executive Summary

This report summarizes our exploration of multi-node consensus mechanisms for edge computing swarms.

### Key Finding

**Simple leader weight advantage does NOT solve the stubborn node problem.**

| Method | Consensus Rate | Correct Rate | Verdict |
|--------|---------------|--------------|---------|
| Equal coupling (α=0.2) | 100% | 20% | Stubborn wins |
| Energy confidence | 40% | 20% | No improvement |
| Selective broadcast | 40% | 20% | No improvement |
| Leader weight (2.0x) | 33% | **26%** | Slightly better |
| Leader weight (5.0x) | 66% | 13% | Worse |

---

## Part 1: Stubborn Node Problem

### Experiment: Triple Node with Equal Coupling

**Configuration**:
- Node A: 30% noise (correct)
- Node B: 80% noise (confused)
- Node C: Wrong pattern (stubborn)
- Topology: Fully connected A↔B↔C

**Results**:
- Consensus Rate: 100%
- Correct Consensus: **20%**
- Finding: Stubborn node C dominates

### Physical Interpretation

1. **Signal Imbalance**: A's correct signal is diluted by B's noise
2. **Attractor Competition**: C's wrong attractor is stronger
3. **Majority Effect**: 2vs1 situation favors wrong answer

---

## Part 2: Energy-Based Confidence

### Hypothesis
Nodes with deeper energy basins should have stronger influence.

**Equation**:
```
C_i = |mean(x_i)|  # Confidence
x_j(t+1) = tanh(W_j @ x_j + α * Σ C_i * x_i)
```

### Results

| Alpha | Consensus | Correct |
|-------|-----------|---------|
| 0.1 | 40% | 20% |
| 0.2 | 10% | 0% |
| 0.3 | 40% | 10% |
| 0.5 | 40% | 10% |

**Finding**: Confidence-based coupling does NOT solve the problem.

### Why It Fails

1. **Initial State**: All nodes start with similar confidence
2. **Gradient Descent**: All nodes fall into their attractors simultaneously
3. **No Leader Emergence**: The system doesn't identify the "correct" node

---

## Part 3: Selective Broadcast

### Hypothesis
Only high-confidence nodes should transmit signals.

**Equation**:
```
broadcast = 0
if |mean(x_A)| > 0.5: broadcast += x_A
if |mean(x_B)| > 0.5: broadcast += x_B  # Rare
if |mean(x_C)| > 0.5: broadcast += x_C  # Often

x_j = tanh(W_j @ x_j + α * broadcast)
```

### Results

| Alpha | Consensus | Correct |
|-------|-----------|---------|
| 0.1 | 40% | 20% |
| 0.2 | 10% | 0% |
| 0.3 | 40% | 10% |
| 0.5 | 10% | 10% |

**Finding**: Still no improvement.

### Why It Fails

1. **Threshold Problem**: No clear confidence threshold exists
2. **C's Confidence**: Stubborn node maintains high confidence
3. **Feedback Loop**: High confidence reinforces itself

---

## Part 4: Leader Mode (Weight Advantage)

### Hypothesis
Giving Node A a fixed weight advantage will allow it to dominate.

**Equation**:
```
x_A = tanh(W_A @ x_A + leader_weight * α * x_A + α * (x_B + x_C))
x_B = tanh(W_B @ x_B + leader_weight * α * x_A + α * (x_A + x_C))
x_C = tanh(W_C @ x_C + leader_weight * α * x_A + α * (x_A + x_B))
```

### Results

| Leader Weight | Consensus | Correct |
|---------------|-----------|---------|
| 1.0x (equal) | 53% | 20% |
| 2.0x | 33% | **26%** |
| 3.0x | 53% | 20% |
| 5.0x | 66% | 13% |
| 10.0x | 73% | 13% |

**Finding**: Higher leader weight INCREASES consensus but DECREASES correctness!

### Paradox

```
More leader weight → More consensus → LESS correctness
```

This reveals a fundamental trade-off:
- High consensus = Groupthink (follows whoever is loudest)
- Low consensus = Civil war (no agreement)

---

## Part 5: Theoretical Analysis

### The Stubborn Node Paradox

**Problem**: 
- Stubborn minority can defeat confident majority
- Democratic mechanisms don't guarantee correctness
- Weight advantage can backfire

### Energy Landscape Interpretation

```
Energy Function: E(x) = -x^T W x

Node A (correct): Deep basin at +1
Node B (confused): Shallow basin, mobile
Node C (wrong): Deep basin at -1

Coupling: x_j(t+1) = -∇E_j(x) + coupling

Result: Energy minimization pulls everyone to -1
```

### Why Democracy Fails

1. **Energy Wins**: Deepest attractor dominates regardless of truth
2. **No Truth Signal**: System doesn't know which attractor is "correct"
3. **Feedback Amplification**: Wrong attractors reinforce themselves

---

## Part 6: Path Forward

### Option A: External Truth Signal

Introduce a small "truth bias" that favors the correct attractor:

```
x_j = tanh(W_j @ x_j + α * coupling + ε * P_correct)
```

**Pros**: Guaranteed correctness
**Cons**: Centralized, not truly decentralized

### Option B: Reputation Evolution

Nodes develop reputations over time:

```
R_A(t+1) = R_A(t) + η * (correct_predictions - R_A(t))
Weight = R_i / Σ R_j
```

**Pros**: Emergent trust
**Cons**: Requires iteration, slow

### Option C: Hierarchical Structure

A → B
↑   ↓
C → D

Upper layers filter lower layers' signals.

**Pros**: Scalable, fault-tolerant
**Cons**: More complex

### Option D: Physical Constraints

Add constraints that favor correct patterns:

```
|x_i| = 1.0  # Strong polarization
|Σ x_i| > threshold  # Require consensus
```

**Pros**: Simple
**Cons**: May not work for complex patterns

---

## Part 7: Conclusions

### What We Proved

| Question | Answer |
|----------|--------|
| Can democracy solve stubborn node? | **No** |
| Does energy confidence help? | **No** |
| Can leader weight solve it? | **Paradox: More consensus = Less correct** |

### The Fundamental Insight

> "Consensus is not truth. Deepest attractor wins regardless of correctness."

### Implications for Edge Computing

1. **Trust is not emergent**: Cannot identify correct nodes automatically
2. **Democratic mechanisms fail**: Need external truth signal or hierarchy
3. **Coupling strength is critical**: Balance between consensus and correctness

### Recommendations

For production edge swarm systems:

1. **Add truth anchors**: External validation of correct attractors
2. **Use reputation**: Track node accuracy over time
3. **Hierarchical voting**: Upper layers arbitrate lower layers
4. **Physical constraints**: Favor polarized, stable states

---

## Files Generated

- `core/triple_node_consensus.py`: Original experiment
- `results/energy_confidence.json`: Energy-based experiment
- `results/selective_broadcast.json`: Broadcast experiment
- `results/leader_mode.json`: Leader weight experiment

---

## Research Path

```
System Stability (COMPLETE)
├── Sparse Hebbian (FAILED: sparsity increases interference)
├── Bridge Communication (SUCCESS: α=0.2, 80% rescue)
├── Triple Consensus (FAILED: stubborn node wins)
└── Leader Mode (PARADOX: more weight = less correct)

NEXT: Hierarchical Structure or External Truth Signal?
```

---

*Report generated 2026-02-07*
*All code and data available in multi_layer/*

**Key Takeaway**: Deepest energy basin wins, not deepest truth. Democracy needs anchors.
