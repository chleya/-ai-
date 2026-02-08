"""
ULTRA-FAST MINIMAL TEST
只测试最关键的配置
"""

import numpy as np

def nl(x): return np.tanh(x)

def minimal_test(N, n_tests=5):
    """最小测试"""
    
    correct_ltd = 0
    
    for test in range(n_tests):
        np.random.seed(test * 100 + N)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_good = np.random.randn(N, N) / np.sqrt(N)
        W_bad = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(500):  # 减少步数
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
        
        for _ in range(100):  # 减少测试步数
            s_good = nl(W_good @ s_good + 0.2 * (s_good + s_bad))
            s_bad = nl(W_bad @ s_bad + 0.2 * (s_good + s_bad))
        
        if np.sign(np.mean(s_good)) == 1:
            correct_ltd += 1
    
    return correct_ltd * 100 // n_tests

print("="*60)
print("ULTRA-FAST MINIMAL TEST")
print("="*60)

tests = [
    (50, "N=50 (基线)"),
    (100, "N=100 (小)"),
    (200, "N=200 (中)"),
    (500, "N=500 (大)"),
]

results = {}

for N, desc in tests:
    print(f"\n{desc}...")
    result = minimal_test(N)
    results[desc] = result
    print(f"  LTD: {result}%")

print()
print("="*60)
print("RESULTS")
print("="*60)

for desc, result in results.items():
    marker = '⭐' if result >= 70 else ('✓' if result >= 50 else '❌')
    print(f"{marker} {desc}: {result}%")

print()
print("Analysis:")
baseline = results.get("N=50 (基线)", 0)
large = results.get("N=500 (大)", 0)
decay = baseline - large

if decay < 20:
    print("LTD SCALES WELL: 衰减 < 20%")
elif decay < 40:
    print("LTD PARTIAL: 适度衰减")
else:
    print("LTD FAILS: 严重衰减")
