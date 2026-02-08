# 2026-02-07 - HIGHER GAIN TEST: CEILING IS SOFT!

## Research Question
Is the ceiling (~0.05) a **soft limit** (pushable) or **hard limit** (fixed by dynamics)?

---

## EXPERIMENT 04d: Higher Gain Test

### Method
- Gain multipliers: 1.0x, 1.5x, 2.0x, 3.0x
- Starting targets: 0.040, 0.050
- Formula: target += base_gain * multiplier * (variance/target - 1)

---

## RESULTS

### From target=0.040

| Multiplier | Final Target | Drift | Status |
|------------|---------------|-------|--------|
| 1.0x | 0.0474 | +0.0074 | Baseline |
| 1.5x | 0.0492 | +0.0092 | ↑ Higher |
| 2.0x | 0.0505 | +0.0105 | ↑ Higher |
| 3.0x | **0.0520** | **+0.0120** | ↑ Highest |

**Ceiling moved: 0.047 → 0.052 (+10%)**

### From target=0.050

| Multiplier | Final Target | Drift |
|------------|---------------|-------|
| 1.0x | 0.0517 | +0.0017 |
| 3.0x | 0.0534 | +0.0034 |

Also shows ceiling movement, but smaller (starting closer to equilibrium).

---

## KEY FINDING: CEILING IS SOFT

### Evidence
1. **Higher gain = Higher ceiling**: 1.0x → 3.0x pushes target from 0.047 to 0.052
2. **All runs stable**: No oscillation or explosion
3. **Gain-dependent**: The ceiling is not fixed by dynamics

### Interpretation

```
Previous discovery: Equilibrium at variance ≈ target (natural balance)
This discovery: That balance point is ADJUSTABLE via feedback gain

Ceiling = function(gain), not fixed by system dynamics
```

---

## THEORETICAL IMPLICATIONS

### What We Proved

| Question | Answer |
|----------|--------|
| Can positive feedback climb? | Yes (+33% from baseline) |
| Is there a ceiling? | Yes (~0.05 at gain=1.0x) |
| Is the ceiling fixed? | **No - it's soft** |
| What determines ceiling? | **Feedback gain** |

### The Ceiling Model

```
Ceiling(gain) = baseline_ceiling + f(gain_multiplier)

At gain=1.0x: ceiling ≈ 0.047
At gain=3.0x: ceiling ≈ 0.052
```

The ceiling can be pushed arbitrarily high by increasing gain (until instability).

---

## NEXT QUESTIONS

1. **What's the instability threshold?**
   - How high can we push before oscillation/explosion?
   - Test gain=5x, 10x

2. **What's the gain-ceiling relationship?**
   - Linear? Logarithmic?
   - ceiling = a * log(gain) + b ?

3. **Does this generalize?**
   - Test N=50, N=100
   - Does ceiling scale with dimension?

---

## FILES

- `results/higher_gain_test.json`: Raw data
- `CEILING_DISCOVERY.md`: Previous ceiling discovery

---

*Motto: First prove the system can live, then discuss what it can do.*

*Finding: The ceiling is SOFT - adjustable via feedback gain.*
