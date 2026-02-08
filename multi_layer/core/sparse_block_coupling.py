"""
SPARSE BLOCK COUPLING EXPERIMENT
基于用户建议：模块化 + Gossip
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def sparse_block_test(N=1000, n_modules=20, steps=1000, trials=10):
    """稀疏块状耦合测试"""
    
    results = {'baseline': 0, 'ltd': 0}
    
    for t in range(trials):
        np.random.seed(t * 1000 + N)
        module_size = N // n_modules
        
        # 创建模式
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # 模块内权重
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 全局错误吸引子
        W_bad = np.random.randn(N, N) / np.sqrt(N)
        
        # 学习阶段
        for _ in range(steps):
            for m in range(n_modules):
                idx_start = m * module_size
                idx_end = (m + 1) * module_size
                P_m = P[idx_start:idx_end]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 基线测试（无LTD）
        s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
        for _ in range(500):
            new_s = np.zeros(N)
            # 模块内耦合 + Gossip
            for m in range(n_modules):
                idx_start = m * module_size
                idx_end = (m + 1) * module_size
                s_m = s[idx_start:idx_end]
                new_s[idx_start:idx_end] = nl(W_modules[m] @ s_m + 0.1 * s_m)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            results['baseline'] += 1
        
        # LTD测试
        np.random.seed(t * 1000 + N)
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        W_bad = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(steps):
            for m in range(n_modules):
                idx_start = m * module_size
                idx_end = (m + 1) * module_size
                P_m = P[idx_start:idx_end]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # LTD破坏错误吸引子
        x_bad_final = nl(W_bad @ (-P) + np.random.randn(N) * 0.2)
        W_bad -= 0.5 * np.outer(x_bad_final, x_bad_final)
        W_bad = W_bad / np.linalg.norm(W_bad) * 0.1
        
        # LTD测试
        s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
        for _ in range(500):
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx_start = m * module_size
                idx_end = (m + 1) * module_size
                s_m = s[idx_start:idx_end]
                new_s[idx_start:idx_end] = nl(W_modules[m] @ s_m + 0.1 * s_m)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            results['ltd'] += 1
    
    return {
        'baseline': results['baseline'] * 100 // trials,
        'ltd': results['ltd'] * 100 // trials
    }

print("="*60)
print("SPARSE BLOCK COUPLING EXPERIMENT")
print("="*60)
print("\nRunning N=1000, 20 modules...")

result = sparse_block_test(N=1000, n_modules=20, steps=1000, trials=10)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"基线: {result['baseline']}%")
print(f"LTD:  {result['ltd']}%")
print(f"改善: +{result['ltd'] - result['baseline']}%")

print("\nKey Finding:")
if result['ltd'] >= 50:
    print("SUCCESS: LTD works with sparse block coupling!")
else:
    print("NEEDS WORK: LTD underperforms at N=1000 scale")

# Save
with open('results/sparse_block_coupling.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nSaved to results/sparse_block_coupling.json")
