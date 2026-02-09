# API Reference

## `CicadaProtocol`

The main class for the Cicada Protocol.

### Constructor

```python
CicadaProtocol(N=787, theta=1.2, alpha=1.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| N | int | 787 | System size (phase transition point) |
| theta | float | 1.2 | Stability threshold |
| alpha | float | 1.0 | Sensitivity coefficient |

### Methods

#### `step(W, s, eta=0.01)`

Execute one step of the Cicada Protocol.

```python
W_new = cicada.step(W, s, eta=0.01)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| W | np.ndarray | - | Current weight matrix |
| s | np.ndarray | - | Input pattern |
| eta | float | 0.01 | Learning rate |

**Returns:** `np.ndarray` - Updated weight matrix

---

#### `hebbian_update(W, s, eta=0.01)`

Perform Hebbian update.

```python
W_new = cicada.hebbian_update(W, s, eta=0.01)
```

---

#### `compute_lambda_max(W, iterations=10)`

Compute maximum eigenvalue using power iteration.

```python
lam = cicada.compute_lambda_max(W, iterations=10)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| W | np.ndarray | - | Weight matrix |
| iterations | int | 10 | Number of power iterations |

**Returns:** `float` - Maximum eigenvalue

---

#### `reset(W)`

Reset weight matrix (inject negative entropy).

```python
W_reset = cicada.reset(W)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| W | np.ndarray | Weight matrix |

**Returns:** `np.ndarray` - Reset weight matrix

---

#### `get_stats()`

Get protocol statistics.

```python
stats = cicada.get_stats()
```

**Returns:** `dict` - Dictionary containing:
- `lambda_max`: Average lambda max over last 10 steps
- `entropy`: Average entropy over last 10 steps
- `resets`: Total number of resets
- `lambda_growth`: Total lambda growth

---

#### `summary()`

Print protocol summary to console.

```python
cicada.summary()
```

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from cicada import CicadaProtocol

# Initialize
cicada = CicadaProtocol(N=787, theta=1.2, alpha=1.0)

# Initialize weight matrix
W = np.random.randn(787, 787) * 0.1

# Run simulation
for t in range(100):
    s = np.random.randn(787)
    W = cicada.step(W, s)

# Get statistics
stats = cicada.get_stats()
print(f"Resets: {stats['resets']}")
print(f"Lambda: {stats['lambda_max']:.4f}")
```

### Federated Learning

```python
import numpy as np
from cicada import CicadaProtocol

# Initialize
cicada = CicadaProtocol(N=500, theta=1.8, alpha=1.2)

# K clients
K = 5
W_global = np.random.randn(K, 500) * 0.1

for round_idx in range(50):
    # Local updates
    for k in range(K):
        s = np.random.randn(500)  # Local data
        W_global[k] = cicada.step(W_global[k], s)
    
    # Global aggregation
    W_global = np.mean(W_global, axis=0)
```

---

## Metrics

### Lambda Max ($\lambda_{\max}$)

The maximum eigenvalue of the weight matrix, representing the system's spectral radius. When $\lambda_{\max} > 1$, the system enters chaotic dynamics.

### Entropy ($H$)

Approximate von Neumann entropy, measuring the system's disorder. Higher entropy correlates with less stable dynamics.

### Reset Trigger

When $\lambda_{\max} > \theta \times \alpha$, a reset is triggered to inject negative entropy.
