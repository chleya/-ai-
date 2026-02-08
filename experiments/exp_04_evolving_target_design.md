# System 04: Evolving Target Variable - EXPERIMENT DESIGN

## Research Question
Can the system evolve its own target (target_var) to find better operating points?

## Key Hypothesis
If target_var can slowly adapt:
- System might discover better equilibrium than human-set value
- Target should correlate with achievable variance
- Hierarchical homeostasis emerges (fast + slow regulation)

---

## Experiment 1: Random Drift with Exploration

### Design
```
Level 1 (Fast): Alpha regulation (per step)
Level 2 (Slow): target_var drift + exploration (every N steps)
```

### Mechanism
```python
# Every drift_interval steps:
target_var += random_normal(0, drift_sigma)
target_var = clip(target_var, 0.005, 0.30)

# Exploration: if recent variance > target, nudge target up
if recent_var > target_var * 1.2:
    target_var += exploration_rate * (recent_var - target_var)
```

---

## Experiment 2: Target Follows Variance

### Design
```
target_var slowly follows achievable variance level
```

### Mechanism
```python
# Every adaptation_interval steps:
achievable_var = mean(variance[-window])

# Target approaches achievable (but with smoothing)
target_var += learning_rate * (achievable_var - target_var)

# This lets the system "discover" its natural variance level
```

---

## Expected Outcomes

### If Evolution Works
| Metric | Expected |
|--------|----------|
| target_var drift | Sustained from initial |
| Final variance | Higher than fixed target |
| Hierarchical behavior | Two timescales visible |

### If Evolution Fails
| Metric | Expected |
|--------|----------|
| target_var | Drifts to boundary and stays |
| Final variance | Same as fixed target |
| No hierarchy | Single regulation mode |

---

## Success Criteria

**Minimal Success**: 
- target_var shows sustained drift (>10% from initial)

**Breakthrough Success**:
- Final variance > fixed target baseline
- target_var correlates with natural variance level
- Two-timescale regulation visible in analysis

---

## Execution Plan

### Phase 1: Simple Random Drift
```bash
python core/system_04_evolving_target.py --mode explore --sigma 0.5
```

### Phase 2: Exploration Mechanism
```bash
python core/system_04_evolving_target.py --mode explore --sigma 0.5 --exploration
```

### Phase 3: Adaptive Target Following
```bash
python core/system_04_evolving_target.py --mode adaptive --sigma 0.5
```

---

## Analysis Metrics

1. **Target drift**: final_target - initial_target
2. **Variance improvement**: (evolving - fixed) / fixed
3. **Hierarchy detection**: ACF analysis of target vs alpha
4. **Correlation**: target_var vs variance over time

---

*Date: 2026-02-07*
*Goal: Can system define its own optimal operating point?*
