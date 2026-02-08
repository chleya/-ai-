# EXTREME GAIN TEST - PROTECTIONS & METRICS

## Protections (Active)

### 1. Target Clip
```python
target = np.clip(target, 0.01, 0.25)
```

### 2. Variance Emergency Brake
```python
if window_var > 0.8:
    target *= 0.5  # Halve to brake
```

### 3. Early Stopping
```python
if np.std(variance[-500:]) > 0.05:
    early_stop("high_variance_std")
```

---

## Improved Instability Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| target_std | std(target[-2000:]) | > 0.005 = unstable |
| var_fluctuation | std(var)/mean(var) | > 0.20 = high |
| max_step_change | max(|target[t] - target[t-1]|) | > 0.005 = unstable |
| overshoot_count | count(var/target > 1.3) | > 100 = overshooting |
| alpha_volatility | std(alpha)/mean(alpha) | > 0.1 = volatile |

---

## Test Grid

| Target | Gain |
|--------|------|
| 0.040 | 10x |
| 0.040 | 15x |
| 0.040 | 20x |
| 0.050 | 10x |
| 0.050 | 15x |
| 0.050 | 20x |

---

*Created: 2026-02-07*
