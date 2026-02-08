"""
SIMPLE LARGE-SCALE TEST
Fast and reliable
"""

import numpy as np, json, sys

def nl(x): return np.tanh(x)

def test(N, steps, tests):
    correct = 0
    for t in range(tests):
        np.random.seed(t * 100 + N + steps)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_good = np.random.randn(N, N) / np.sqrt(N)
        W_bad = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(steps):
            x_good = nl(W_good @ P + np.random.randn(N) * 0.3)
            W_good += 0.001 * np.outer(x_good, x_good)
            x_bad = nl(W_bad @ (-P) + np.random.randn(N) * 0.5)
            W_bad += 0.001 * np.outer(x_bad, x_bad)
            
            if np.linalg.norm(W_good) > 10: W_good = W_good / np.linalg.norm(W_good) * 10
            if np.linalg.norm(W_bad) > 10: W_bad = W_bad / np.linalg.norm(W_bad) * 10
        
        # LTD
        x_bad_final = nl(W_bad @ (-P) + np.random.randn(N) * 0.2)
        W_bad -= 0.5 * np.outer(x_bad_final, x_bad_final)
        W_bad = W_bad / np.linalg.norm(W_bad) * 0.1
        
        # Test
        s_good = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s_bad = (1 - 0.5) * (-P) + 0.5 * np.random.randn(N)
        
        for _ in range(200):
            s_good = nl(W_good @ s_good + 0.2 * (s_good + s_bad))
            s_bad = nl(W_bad @ s_bad + 0.2 * (s_good + s_bad))
        
        if np.sign(np.mean(s_good)) == 1: correct += 1
    
    return correct * 100 // tests

print("="*60)
print("LARGE-SCALE TEST")
print("="*60)

configs = [
    (100, 1500, "N=100"),
    (200, 2000, "N=200"),
    (500, 2500, "N=500"),
]

results = {}

for N, steps, name in configs:
    print(f"\nRunning {name}...")
    sys.stdout.flush()
    result = test(N, steps, 10)
    results[name] = result
    print(f"  {name}: LTD = {result}%")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

for name, result in results.items():
    marker = "[BEST]" if result >= 70 else ("[GOOD]" if result >= 50 else "[LOW]")
    print(f"{marker} {name}: {result}%")

print("\nKey Finding:")
if results.get("N=500", 0) >= 50:
    print("LTD SCALES to large systems!")
else:
    print("LTD decays with scale.")

# Save
with open('results/simple_large_scale.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/simple_large_scale.json")
