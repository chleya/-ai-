"""
EXTREME SPARSITY EXPERIMENT
Test 30-50 modules to reach near-single-node precision
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def extreme_sparse_test(N, n_modules, coupling=0.1, steps=800, trials=8):
    """极度稀疏测试"""
    
    baseline = 0
    ltd = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 10000 + N + n_modules)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # 基线
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
        for _ in range(500):
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + coupling * s_m)
            s = new_s
        if np.sign(np.mean(s)) == 1: baseline += 1
        
        # LTD
        np.random.seed(t * 10000 + N + n_modules)
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        W_bad = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(steps):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # LTD破坏
        x_bad = nl(W_bad @ (-P) + np.random.randn(N) * 0.3)
        W_bad -= 0.5 * np.outer(x_bad, x_bad)
        W_bad = W_bad / np.linalg.norm(W_bad) * 0.1
        
        s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
        for _ in range(500):
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + coupling * s_m)
            s = new_s
        if np.sign(np.mean(s)) == 1: ltd += 1
    
    return {'baseline': baseline*100//trials, 'ltd': ltd*100//trials}

print("="*60)
print("EXTREME SPARSITY EXPERIMENT")
print("Testing 30-50 modules for near-single-node precision")
print("="*60)

configs = [
    (1000, 30, "N=1000, 30模块"),
    (1000, 40, "N=1000, 40模块"),
    (1000, 50, "N=1000, 50模块"),
]

all_results = {}

for N, modules, name in configs:
    print(f"\n{name}...")
    result = extreme_sparse_test(N, modules)
    all_results[name] = result
    print(f"  基线: {result['baseline']}%, LTD: {result['ltd']}%")

print("\n" + "="*60)
print("EXTREME SPARSITY RESULTS")
print("="*60)

for name, r in all_results.items():
    delta = r['ltd'] - r['baseline']
    marker = "[BEST]" if r['ltd'] >= 60 else ("[GOOD]" if r['ltd'] >= 50 else "[OK]")
    print(f"{marker} {name}: 基线{r['baseline']}% -> LTD{r['ltd']}% ({delta:+d}%)")

print("\nKey Finding:")
best = max(all_results.values(), key=lambda x: x['ltd'])
if best['ltd'] >= 70:
    print("SUCCESS: 接近单节点精度(~70%)！")
elif best['ltd'] >= 60:
    print("PROGRESS: 显著提升，但未达单节点精度")
elif best['ltd'] >= 50:
    print("PARTIAL: 极度稀疏有效，但有限")
else:
    print("LIMITED: 极度稀疏未带来显著改善")

# Save
with open('results/extreme_sparsity.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/extreme_sparsity.json")
