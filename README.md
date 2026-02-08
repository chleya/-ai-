# Cicada Protocol

A periodic reset mechanism for maintaining long-term stability in distributed consensus systems.

## Quick Start

```bash
# Clone
git clone https://github.com/chleya/-ai-.git
cd -ai-

# Install
pip install -e .

# Run demo
python examples/demo.py

# Or use CLI
python -m cicada --demo
```

## Project Structure

```
├── cicada/                  # Python package
│   ├── core.py            # Core protocol implementation
│   ├── __init__.py
│   ├── __main__.py        # CLI: python -m cicada --demo
│   └── experiments.py
├── examples/                # Demo scripts
│   ├── demo.py            # Main demo
│   ├── test_n1000.py      # N=1000 experiment
│   └── visualize_heatmap.py
├── papers/                  # Research papers
│   └── CICADA_PAPER.md
├── docs/                   # Documentation
│   ├── QUICKSTART.md
│   ├── DYNAMICS_THEORY.md
│   ├── EVENT_TRIGGERED_RESET.md
│   └── *.md
├── visualization/          # Visualizations
│   ├── images/            # Generated images
│   └── visualize_*.py
├── LICENSE
├── README.md
├── setup.py
└── requirements.txt
```

## Usage

### Python API

```python
from cicada.core import cicada_protocol, analyze_spectrum

# Run protocol
W, s, history = cicada_protocol(N=200, reset_interval=300)

# Analyze results
spectrum = analyze_spectrum(W)
print(f"λ_max: {spectrum['max']:.4f}")
```

### Command Line

```bash
# Show help
python -m cicada --help

# Run demo
python -m cicada --demo
```

### Demo Script

```bash
python examples/demo.py
```

## Key Results

| Metric | Value |
|--------|-------|
| λ_max (healthy) | < 1.8 |
| Optimal reset interval | 300 steps |
| Optimal α (event) | 1.6 |

## Demo Output

```
No Reset     -> λ = 0.56
Reset 100    -> λ = 0.28 (↓49%)
Reset 200    -> λ = 0.27 (↓52%)
```

## Papers

- [papers/CICADA_PAPER.md](papers/CICADA_PAPER.md) - Main research paper
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [docs/DYNAMICS_THEORY.md](docs/DYNAMICS_THEORY.md) - Theoretical framework

## License

MIT
