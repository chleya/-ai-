# Cicada Protocol - Readable Code Guide

## Design Principles

1. **Simple is better** - Avoid clever tricks
2. **Documented** - Every function has a docstring
3. **Consistent** - Same style throughout
4. **Testable** - Easy to run and debug

## Code Structure

```
cicada/
├── core.py           # Main module (simple, readable)
├── __init__.py      # Package exports
└── __main__.py      # CLI entry point

demo.py              # Standalone demo
```

## Core API (core.py)

### One Function = One Concept

```python
# Protocol: Run simulation
W, s, history = cicada_protocol(N=200, reset_interval=300)

# Analysis: Analyze results
spectrum = analyze_spectrum(W)
# {'max': 1.73, 'ratio': 1.48, ...}

# Statistics: Calculate metrics
survival = calculate_survival_rate(history)
# {'survival_rate': 0.85, ...}

# Experiment: Run multiple trials
result = run_experiment(N=200, reset_interval=300)
# {'success_rate': 0.85, ...}
```

### Function Template

```python
def function_name(param1: Type1, param2: Type2 = default) -> ReturnType:
    """
    One-line description.
    
    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2, optional
        Description of param2
    
    Returns
    -------
    ReturnType
        Description of return value
    
    Example
    -------
    >>> result = function_name(value)
    >>> print(result)
    """
    # Code here
    return result
```

## Demo Structure (demo.py)

```
CONFIG          # Configuration at top
├─ simulate()   # Core simulation
├─ plot()       # Visualization
└─ main()       # Orchestration
```

### Demo Pattern

```python
# 1. Print banner
print("=" * 60)
print("Demo Title")
print("=" * 60)

# 2. Run simulation
print("Running...")
result = simulate()

# 3. Show results
print("Results:")
print("  Value: {:.2f}".format(result))

# 4. Key insight
print()
print("=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
```

## Debugging Tips

### 1. Run the demo
```bash
python demo.py
```

### 2. Quick test
```bash
python -c "from cicada.core import cicada_protocol; W, s, h = cicada_protocol(); print('OK')"
```

### 3. Check configuration
```python
from cicada.core import cicada_protocol
W, s, h = cicada_protocol(N=100, steps=100, seed=42)  # Small is fast
```

### 4. Visualize
```python
from cicada.core import cicada_protocol
import matplotlib.pyplot as plt

W, s, h = cicada_protocol(N=200, reset_interval=300)
plt.plot(h)
plt.savefig('debug.png')
```

## Common Patterns

### Pattern 1: Basic Usage
```python
from cicada.core import cicada_protocol, analyze_spectrum

W, s, history = cicada_protocol(N=200, reset_interval=300)
spectrum = analyze_spectrum(W)
```

### Pattern 2: Compare Strategies
```python
from cicada.core import run_experiment

fixed = run_experiment(reset_interval=100, protocol='fixed')
event = run_experiment(alpha=1.6, protocol='event')

print(f"Fixed: {fixed['success_rate']:.1%}")
print(f"Event: {event['success_rate']:.1%}")
```

### Pattern 3: Find Best Parameters
```python
from cicada.core import run_experiment

for interval in [100, 200, 300, 400]:
    result = run_experiment(reset_interval=interval)
    print(f"Interval {interval}: {result['success_rate']:.1%}")
```

## Key Metrics

| Metric | Healthy | Warning | Unstable |
|--------|---------|---------|----------|
| lambda_max | < 1.8 | 1.8 - 2.0 | > 2.0 |
| survival_rate | > 80% | 50% - 80% | < 50% |

## File Organization

```
F:/system_stability/
├── cicada/
│   ├── core.py         # 9.5KB - Simple, well-documented
│   ├── __init__.py      # Package exports
│   └── __main__.py     # CLI
├── demo.py              # 5.6KB - Standalone demo
└── README.md            # Project documentation
```

## Quick Reference

| Task | Code |
|------|------|
| Run simulation | `W, s, h = cicada_protocol(N=200, reset_interval=300)` |
| Analyze | `spectrum = analyze_spectrum(W)` |
| Statistics | `survival = calculate_survival_rate(h)` |
| Experiment | `result = run_experiment(N=200, trials=10)` |
| Event-triggered | `W, s, h, resets = event_triggered_protocol(alpha=1.6)` |

---

**Goal**: Any new contributor can understand and modify the code in < 10 minutes.

**Principle**: "Readability over cleverness."
