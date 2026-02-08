# Cicada Protocol: Long-term Stability for Edge Computing Consensus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxx)

## 🌟 Overview

The **Cicada Protocol** is a novel approach to maintaining long-term stability in edge computing consensus systems through **periodic system reset**. This repository contains the complete research codebase, paper, and visualizations.

### Key Discoveries

| Discovery | Finding | Impact |
|-----------|---------|--------|
| **Critical Time Point** | Tc ≈ 400 steps | Long-term stability achieved |
| **Spectral Stability** | λmax: 2.15 → 1.73 | Healthier spectrum |
| **Phase Transition** | Nc ≈ 900 | Scale-dependent behavior |
| **Task Switching** | Rand +11.9% faster | Forgetting is better |
| **Event-triggered** | α = 1.6 optimal | Adaptive resilience |

### Core Insight

> "Perfect immortality is unattainable, but robust renewal is achievable."

---

## 📁 Repository Structure

```
cicada_protocol/
├── cicada/
│   ├── __init__.py
│   ├── core.py           # Core protocol implementation
│   ├── experiments.py     # Experiment scripts
│   └── analysis.py        # Analysis tools
├── visualization/
│   ├── visualize_cicada.py  # Animation & plotting
│   ├── phase_transition_heatmap.png
│   ├── efficiency_heatmap.png
│   └── comprehensive_dashboard.png
├── papers/
│   ├── CICADA_PAPER.md   # Main paper (v2.0)
│   └── SUBMISSION_PLAN.md # Submission strategy
├── data/
│   └── experiments/       # Experimental data
├── tests/
│   └── test_*.py         # Unit tests
├── docs/
│   └── *.md              # Documentation
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourname/cicada_protocol.git
cd cicada_protocol
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from cicada import CicadaProtocol

# Create protocol instance
protocol = CicadaProtocol(N=500, reset_interval=300)

# Run evolution
W, s = protocol.evolve(total_steps=800)

# Analyze results
stats = protocol.analyze()
print(f"Survival rate: {stats['survival']:.1%}")
print(f"Spectral radius: {stats['lambda_max']:.3f}")
```

### Running Experiments

```bash
# Basic experiment
python -m cicada.experiments.basic

# Phase transition analysis
python -m cicada.experiments.phase_transition

# Task switching test
python -m cicada.experiments.task_switching

# Stress threshold heatmap
python -m cicada.experiments.stress_heatmap
```

---

## 📊 Key Results

### Phase Transition (N vs Survival)

```
Survival Rate (%)
    │
100 ┤███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │███████████████░░░░░░░░░░░░░░░░░░░░░░░
    │███████████████████████████░░░░░░░░░░░
    │███████████████████████████████░░░░░░
 60 ┤████████████████████████████████████
    │████████████████████████████████████
    └──────────────────────────────────── N
     200   400   600   800  1000
     
     █ Peak Init    □ Random Init
```

### Optimal Trigger Sensitivity

```
Efficiency (×10³)
    │
318 ┤█████████████████░░░░░░░░░░░░░░░░░░░░
    │███████████████████████████████░░░░░
    │███████████████████████████████████
315 ┤██████████████████████████████████████
    │██████████████████████████████████████
    │██████████████████████████████████████
312 ┤██████████████████████████████████████
    └──────────────────────────────────── α
    1.1  1.3  1.5  1.6  1.8  2.0  3.0
    
    ★ Optimal: α = 1.6
```

---

## 🎬 Visualizations

### Spectral Radius Dynamics

![Spectral Dynamics](visualization/spectrum_demo.gif)

Shows how the eigenvalue spectrum evolves under attack and triggers stress rebirth.

### Phase Transition Heatmap

![Phase Transition](visualization/phase_transition_heatmap.png)

Complete phase diagram showing survival rate vs system scale.

### Comprehensive Dashboard

![Dashboard](visualization/comprehensive_dashboard.png)

Four-panel visualization of all key findings.

---

## 📖 Paper

**Latest Version**: [CICADA_PAPER.md](papers/CICADA_PAPER.md)

### Abstract

Edge computing consensus mechanisms face long-term stability challenges. Traditional evolutionary learning methods suffer from "basin collapse" over time. This paper proposes the "Cicada Protocol" - a mechanism that maintains long-term stability through periodic system reset. Through systematic experiments, we discover a critical phase transition at scale Nc≈900 and establish an isotropy theory framework explaining the geometric advantages of random initialization in high-dimensional spaces.

**Keywords**: Edge Computing, Consensus Mechanism, Cicada Protocol, Spectral Stability, Phase Transition, Isotropy, Long-term Evolution

---

## 🎯 Use Cases

| Scenario | Scale | Dynamics | Recommended Strategy |
|----------|-------|----------|----------------------|
| Industrial Sensors | N < 400 | Low | Peak Inheritance |
| Drone Swarms | N > 900 | High | Random Reset |
| Disaster Recovery | Any | Extreme | Random Reset |
| Adaptive | Any | Mixed | Event-triggered Reset |

---

## 📈 Performance

### Long-term Survival Rate

| Strategy | 300 steps | 500 steps | 800 steps |
|----------|-----------|-----------|-----------|
| No Reset | 100% | 40% | 0% |
| Fixed-300 | 100% | 75% | 70% |
| Event-1.6 | 100% | 78% | 75% |

### Task Switching Agility

| Metric | Peak | Rand | Advantage |
|--------|------|------|-----------|
| t=50 Correlation | 0.0190 | 0.0213 | **+11.9%** |
| Eigen Fluctuation | 0.0150 | 0.0093 | **-38%** |

---

## 🔬 Theory

### Spectral Stability

The weight matrix evolves as:

$$W(t+1) = W(t) + \eta \cdot s(t) \cdot s(t)^T$$

Healthy spectral radius: $\lambda_{max} < 1.8$

### Phase Transition

Critical point where system behavior fundamentally changes:

$$N_c \approx 900$$

### Isotropy Theory

Random initialization provides:
- **Directional uniformity**: $\rho(\theta) = \text{const}$
- **High-dimensional advantage**: $\text{rank}(W_{rand}) = N$
- **No manifold locking**: $\text{supp}(W_{rand}) = \mathbb{R}^N$

---

## 🛠️ Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_core.py -v

# With coverage
pytest tests/ --cov=cicada
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

### Style Guide

- Python: PEP 8
- Documentation: Google style
- Commits: Conventional Commits

---

## 📚 Citation

If you use Cicada Protocol in your research, please cite:

```bibtex
@article{cicada_protocol_2026,
  title={Cicada Protocol: Long-term Stability for Edge Computing Consensus},
  author={[Author Name]},
  journal={},
  year={2026},
  note={Available at: https://github.com/yourname/cicada_protocol}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contact

- **Main Author**: OpenClaw AI Assistant
- **Email**: [email protected]
- **Project Link**: [https://github.com/yourname/cicada_protocol](https://github.com/yourname/cicada_protocol)

---

## 🙏 Acknowledgments

- Statistical Physics community for isotropy theory
- Distributed Systems community for consensus mechanisms
- Machine Learning community for evolutionary learning

---

**Last Updated**: 2026-02-08
**Version**: 2.0.0
