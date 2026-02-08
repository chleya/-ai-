# Cicada Protocol

A Python library for maintaining long-term stability in distributed
consensus systems through periodic system reset.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cicada import CicadaProtocol, compare_strategies

# Simple demo
protocol = CicadaProtocol(N=200, reset_interval=300)
W, s = protocol.evolve(steps=800)
stats = protocol.analyze()
print(f"Survival: {stats['survival_rate']:.1%}")

# Compare strategies
results = compare_strategies(N=200, steps=500, trials=3)
```

## Documentation

- See `docs/` directory for full documentation.
- See `examples/` for usage examples.

## Citation

See `papers/CICADA_PAPER.md`.
