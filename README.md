# Cicada Protocol

A periodic reset mechanism for maintaining long-term stability in distributed consensus systems.

## Quick Start

```bash
# Clone
git clone https://github.com/chleya/-ai-.git
cd -ai-

# Install
pip install -r requirements.txt

# Run demo
python examples/demo.py
```

## Project Structure

```
├── cicada/                 # Core Python package
│   ├── core.py            # Main protocol implementation
│   ├── __init__.py       # Package exports
│   └── __main__.py       # CLI entry point
├── examples/               # Demo scripts
│   ├── demo.py           # Main demo (spectral radius comparison)
│   ├── test_n1000.py     # N=1000 experiment
│   └── visualize_heatmap.py # Heatmap visualization
├── docs/                  # Documentation
│   ├── CICADA_PAPER.md    # Research paper
│   ├── QUICKSTART.md     # Quick start guide
│   ├── THEORY.md         # Theoretical framework
│   └── *.md              # Other papers
├── visualization/         # Visualizations
│   └── images/           # Generated images
├── requirements.txt        # Dependencies
└── setup.py              # Package setup
```

## Usage

### Basic Protocol

```python
from cicada.core import cicada_protocol, analyze_spectrum

W, s, history = cicada_protocol(N=200, reset_interval=300)
spectrum = analyze_spectrum(W)
print(f"λ_max: {spectrum['max']:.4f}")
```

### Command Line

```bash
python -m cicada --help
```

### Run Demo

```bash
python examples/demo.py
```

## Key Results

| Metric | Value |
|--------|-------|
| λ_max (healthy) | < 1.8 |
| Optimal reset interval | 300 steps |
| Optimal α (event) | 1.6 |

## Papers

- [CICADA_PAPER.md](docs/CICADA_PAPER.md) - Main research paper
- [QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [THEORY.md](docs/DYNAMICS_THEORY.md) - Theoretical framework

## License

MIT
