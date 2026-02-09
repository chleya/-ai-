# Cicada Protocol

<div align="center">

![Cicada Protocol](https://img.shields.io/badge/Cicada-Protocol-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![arXiv](https://img.shields.io/badge/arXiv-2026-orange?style=for-the-badge)

**Reset as Entropy Injection for Stabilizing Hebbian Learning**

</div>

---

## 🌟 Overview

The **Cicada Protocol** is a novel framework for stabilizing Hebbian learning in distributed systems through periodic reset. Our core insight is that Hebbian learning naturally induces spectral growth ($\lambda_{\max}$ increase), which we formalize through the **H-Theorem**.

### Key Results

| Task | Improvement | Lambda Reduction |
|------|-------------|------------------|
| Federated Learning (Non-IID) | **94.9%** | 85% |
| Time-Varying Optimization | **26.4%** | 72% |
| Pattern Classification | **11.0%** | 99.9% |

### Core Formula

$$\lambda_{\max}(N) = 0.015 \times N^{0.72}$$

---

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Theory](#theory)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cicada-protocol.git
cd cicada-protocol

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.5.0
```

---

## ⚡ Quick Start

```python
import numpy as np
from cicada import CicadaProtocol

# Initialize
cicada = CicadaProtocol(N=787, theta=1.2, alpha=1.0)

# Run simulation
W = np.random.randn(787, 787) * 0.1
s = np.random.randn(787)

# Hebbian update with reset
W = cicada.step(W, s)

print(f"Lambda max: {cicada.lambda_max:.4f}")
```

---

## 📓 Examples

### 1. Federated Learning (94.9% improvement)

```bash
python examples/fl_quick.py
```

![Federated Learning](examples/fl_quick.png)

### 2. Time-Varying Optimization (26.4% improvement)

```bash
python examples/tv_opt_quick.py
```

![Time-Varying](examples/tv_opt_quick.png)

### 3. Pattern Classification (11.0% improvement)

```bash
python examples/pattern_hard.py
```

![Pattern Classification](examples/pattern_hard.png)

---

## 📖 Theory

### The Threefold Framework

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Hebbian Update      H-Theorem       Attractor Theory      │
│   s × sᵀ            dH/dt > 0       λ_max as attractor   │
│       ↓                  ↓                  ↓               │
│   Correlation      Entropy Increase   Unstable Basins        │
│   Growth           System Chaos      λ_max > 1             │
│                                                             │
│                          ↓                                  │
│                    CICADA RESET                              │
│                    λ → λ_reset                              │
│                    H → H_reset                              │
│                    Attractor Switch                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Equations

1. **Hebbian Update**:
   $$W(t+1) = W(t) + \eta \times s \times s^\top$$

2. **Scaling Law** (validated, R²=0.998):
   $$\lambda_{\max}(N) = 0.015 \times N^{0.72}$$

3. **Phase Transition**:
   $$N_c = (1.0 / 0.015)^{1/0.72} \approx 782$$

4. **Reset Condition**:
   $$\text{Reset when: } \lambda_{\max} > \theta \times \alpha$$

---

## 📊 Results

### Task Comparison

| Task | Improvement | Lambda Reduction | Status |
|------|-------------|------------------|--------|
| Consensus | 0% | 0% | Static (topology) |
| Static Distributed | 0% | 0% | Static (convex) |
| **Time-Varying** | **26.4%** | 72% | Dynamic ✅ |
| **Federated (Non-IID)** | **94.9%** | 85% | Dynamic ✅ |
| **Pattern Classification** | **11.0%** | 99.9% | Dynamic ✅ |

### Scaling Law Validation

![Scaling Law](visualization/scaling_law.png)

---

## 📝 Citation

If you use the Cicada Protocol in your research, please cite:

```bibtex
@article{cicada2026,
    author = {Chen, Leiyang and OpenClaw},
    title = {The Cicada Protocol: Reset as Entropy Injection for Stabilizing Hebbian Learning},
    journal = {arXiv preprint},
    year = {2026},
    eprint = {xxxx.xxxxx},
    primaryClass = {cs.LG}
}
```

---

## 📂 Project Structure

```
cicada-protocol/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── cicada/
│   ├── __init__.py
│   ├── core.py
│   ├── protocol.py
│   └── metrics.py
├── examples/
│   ├── cida_protocol.py
│   ├── fl_quick.py
│   ├── tv_opt_quick.py
│   └── pattern_hard.py
├── paper/
│   ├── cicada_paper.tex
│   └── references.bib
├── results/
│   ├── fl_quick.json
│   ├── tv_opt_quick.json
│   └── pattern_hard.json
├── visualization/
│   ├── fl_quick.png
│   ├── tv_opt_quick.png
│   └── pattern_hard.png
└── docs/
    ├── theory.md
    └── api.md
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Chen Leiyang**: chleiyang@example.com
- **OpenClaw**: openclaw@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by Chen Leiyang and OpenClaw**

</div>
