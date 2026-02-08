# Cicada Protocol - Core Code Summary

## Project Status: Research Prototype ✅

### Verified Working Components

| Component | Status | File | Lines | Notes |
|-----------|--------|------|-------|-------|
| **Core Protocol** | ✅ | `cicada/core.py` | 14KB | Fixed + Event-triggered |
| **Package Entry** | ✅ | `cicada/__init__.py` | 12KB | Full API |
| **CLI Entry** | ✅ | `cicada/__main__.py` | 2KB | Command line |
| **Demo Script** | ✅ | `demo.py` | 4.8KB | Standalone |
| **Visualization** | ✅ | `visualization/visualize_attractors.py` | 24KB | 6 plots |

### Core API (cicada/core.py)

```python
# Basic protocol
W, s, history = cicada_protocol(
    N=200,              # System size
    reset_interval=300, # Reset every N steps
    total_steps=800,    # Total evolution steps
    seed=42             # Random seed
)

# Analysis
spectrum = analyze_spectrum(W)
# {'max': 1.73, 'ratio': 1.48, ...}

survival = calculate_survival_rate(history)
# {'survival_rate': 0.85, ...}

# Event-triggered protocol
W, s, history, resets = event_triggered_protocol(
    N=200,
    alpha=1.6,  # Optimal threshold
    window=20
)

# Experiment
result = run_experiment(N=200, reset_interval=300, trials=10)
# ExperimentResult(success_rate=0.85, ...)
```

### Usage Examples

#### Basic Usage
```python
from cicada.core import cicada_protocol, analyze_spectrum

W, s, history = cicada_protocol(N=200, reset_interval=300, total_steps=800)
spectrum = analyze_spectrum(W)
print(f"λ_max: {spectrum['max']:.4f}")
```

#### Compare Strategies
```python
from cicada.core import run_experiment

# Fixed interval
fixed = run_experiment(N=200, reset_interval=300, protocol='fixed')

# Event-triggered
event = run_experiment(N=200, alpha=1.6, protocol='event')

print(f"Fixed: {fixed.success_rate:.1%}")
print(f"Event: {event.success_rate:.1%}")
```

#### Find Optimal Parameters
```python
from cicada.core import find_optimal_reset_interval

results = find_optimal_reset_interval(
    N=200,
    intervals=[100, 200, 300, 400, 500]
)
```

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|---------------|
| λ_max (Peak) | 1.73 | Healthy |
| λ_max (Random) | 2.15 | Warning |
| Critical Threshold | 1.8 | Boundary |
| Optimal Reset Interval | 300 | Steps |
| Optimal α (event) | 1.6 | Threshold multiplier |

### Documentation Files

| Category | Files |
|----------|-------|
| **Core Papers** | CICADA_PAPER.md, PIVOT_THEORY.md |
| **Theory** | DYNAMICS_THEORY.md, PHASE_TRANSITION_REPORT.md |
| **Experiments** | EVENT_TRIGGERED_RESET.md, N1000_THEORY.md |
| **Submission** | SUBMISSION_PLAN.md, STRESS_THRESHOLD_*.md |

### Temporary Files (to be cleaned)

| File | Purpose | Action |
|------|---------|--------|
| cicada_demo.py | Old demo | Keep as reference |
| cicada_v2.py | Old version | Keep as reference |
| cleanup_docs.py | Cleanup script | Delete |
| quick_test.py | Quick test | Delete |
| complete_analysis.py | Analysis | Keep |
| run_complete_scan.py | Scan script | Keep |
| convert_to_html.py | HTML conversion | Keep |

### Recommendations

#### Immediate Actions

1. **Delete temporary files**
   ```bash
   del cleanup_docs.py quick_test.py
   ```

2. **Verify core API**
   ```bash
   python -c "from cicada import cicada_protocol, event_triggered_protocol, run_experiment; print('API OK')"
   ```

3. **Run demo**
   ```bash
   python demo.py
   ```

#### Short-term Improvements

1. Add unit tests in `tests/`
2. Add type hints throughout
3. Add docstring examples
4. Create Jupyter notebook demos

#### Long-term Goals

1. Achieve pip-installable package
2. Add GPU support
3. Implement distributed version
4. Submit to OSDI/SOSP

### Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/chleya/-ai-.git
cd -ai-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run demo
python demo.py

# 4. Use as library
from cicada.core import cicada_protocol
W, s, history = cicada_protocol(N=200, reset_interval=300)
```

### File Statistics

| Metric | Count | Notes |
|--------|-------|-------|
| Total .py files | 64 | Including archived |
| Core .py files | 4 | In cicada/ |
| Demo .py files | 3 | Standalone demos |
| Visualization .py files | 2 | Main + heatmap |
| .md files | 13 | Research papers |

### Version History

| Version | Date | Changes |
|---------|------|--------|
| 1.0 | 2026-02-07 | Initial research |
| 1.5 | 2026-02-08 | Added event-triggered |
| 2.0 | 2026-02-08 | Full API, docs, demos |

---

**Last Updated**: 2026-02-08
**Status**: Research Prototype v2.0
