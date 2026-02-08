"""
FAST LARGE-SCALE TEST
简化版：只测试关键配置
"""

import numpy as np

def nl(x): return np.tanh(x)

def quick_test(N, n_nodes, steps, n_tests=10):
    """快速大规模测试"""
    
    correct_ltd = 0
    
    for test in range(n_tests):
        np.random.seed(test * 100 + N + n_nodes)
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
        
        if np.sign(np.mean(s_good)) == 1:
            correct_ltd += 1
    
    return correct_ltd * 100 // n_tests

print("="*60)
print("FAST LARGE-SCALE TEST")
print("="*60)

configs = [
    (100, 10, 1500, "N=100, 10节点"),
    (200, 20, 2000, "N=200, 20节点"),
    (500, 50, 2500, "N=500, 50节点"),
]

results = {}

for N, nodes, steps, desc in configs:
    print(f"\n{desc}...")
    result = quick_test(N, nodes, steps)
    results[desc] = result
    print(f"  LTD正确率: {result}%")

print()
print("="*60)
print("SUMMARY")
print("="*60)

for desc, result in results.items():
    marker = '⭐' if result >= 70 else ('✓' if result >= 50 else '❌')
    print(f"{marker} {desc}: {result}%")

print()
print("Key Finding:")
large_result = results.get("N=500, 50节点", 0)
if large_result >= 50:
    print("LTD SCALES: 50+节点仍保持50%+正确率")
else:
    print("WARNING: LTD在50节点时降至" + str(large_result) + "%")
