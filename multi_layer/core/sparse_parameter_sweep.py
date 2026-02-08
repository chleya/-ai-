"""
SPARSE BLOCK COUPLING - PARAMETER SWEEP
测试不同模块数和耦合强度
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def sparse_test(N, n_modules, coupling, steps=800, trials=8):
    """稀疏块状耦合测试"""
    
    baseline = 0
    ltd = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 10000 + N + n_modules * 10)
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
        np.random.seed(t * 10000 + N + n_modules * 10)
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
print("SPARSE BLOCK PARAMETER SWEEP")
print("="*60)

# 测试不同配置
configs = [
    (500, 10, 0.1, "N=500, 10模块, 弱耦合"),
    (500, 20, 0.1, "N=500, 20模块, 弱耦合"),
    (1000, 10, 0.1, "N=1000, 10模块, 弱耦合"),
    (1000, 20, 0.1, "N=1000, 20模块, 弱耦合"),
    (1000, 20, 0.2, "N=1000, 20模块, 强耦合"),
]

all_results = {}

for N, modules, coupling, name in configs:
    print(f"\n{name}...")
    result = sparse_test(N, modules, coupling)
    all_results[name] = result
    print(f"  基线: {result['baseline']}%, LTD: {result['ltd']}%")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for name, r in all_results.items():
    marker = "[BEST]" if r['ltd'] >= r['baseline'] + 10 else ("[OK]" if r['ltd'] >= r['baseline'] else "[FAIL]")
    delta = r['ltd'] - r['baseline']
    print(f"{marker} {name}: 基线{r['baseline']}% -> LTD{r['ltd']}% ({delta:+d}%)")

print("\nKey Finding:")
positives = sum(1 for r in all_results.values() if r['ltd'] >= r['baseline'])
if positives >= 3:
    print("LTD VIABLE: Works in some sparse configurations")
elif positives >= 1:
    print("LTD PARTIAL: Works only with specific parameters")
else:
    print("LTD FAILS: Sparse coupling may not benefit from LTD")

# Save
with open('results/sparse_parameter_sweep.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/sparse_parameter_sweep.json")
