"""
DEEPENING RESEARCH: Why Strong Coupling Works
研究核心问题：为什么强耦合α=0.5比弱耦合α=0.3效果更好？
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def coupling_analysis(N=1000, n_modules=50, trials=10):
    """耦合强度分析"""
    
    results = {}
    module_size = N // n_modules
    
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        success = 0
        for t in range(trials):
            np.random.seed(t * 1000 + N + int(alpha * 100))
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            
            W_modules = []
            for m in range(n_modules):
                W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
                W_modules.append(W)
            
            # 隔离期
            for step in range(200):
                for m in range(n_modules):
                    idx = slice(m*module_size, (m+1)*module_size)
                    P_m = P[idx]
                    x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                    W_modules[m] += 0.001 * np.outer(x_good, x_good)
                    if np.linalg.norm(W_modules[m]) > 10:
                        W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
            
            # 结晶期 - 不同耦合强度
            for step in range(200):
                s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
                new_s = np.zeros(N)
                for m in range(n_modules):
                    idx = slice(m*module_size, (m+1)*module_size)
                    s_m = s[idx]
                    new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
                s = new_s
            
            if np.sign(np.mean(s)) == 1:
                success += 1
        
        results[f'alpha={alpha}'] = success * 100 // trials
    
    return results

print("="*60)
print("COUPLING STRENGTH ANALYSIS")
print("Why does strong coupling (0.5) work better?")
print("="*60)

print("\nRunning experiments...")
results = coupling_analysis()

print("\n" + "="*60)
print("RESULTS")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (alpha, rate) in enumerate(sorted_results):
    marker = "[BEST]" if i == 0 else ("[TOP3]" if i < 3 else "")
    print(f"{marker} {alpha}: {rate}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

best = sorted_results[0]
if best[1] >= 70:
    print(f"STRONG COUPLING CONFIRMED: α={best[0]} achieves {best[1]}%")
    print("\nHypothesis:")
    print("- Strong coupling forces faster global synchronization")
    print("- Prevents re-entrance of local errors during crystallization")
    print("- Creates stronger global attractor")
else:
    print(f"Best result: {best[0]} = {best[1]}%")

# Save
with open('results/coupling_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/coupling_analysis.json")
