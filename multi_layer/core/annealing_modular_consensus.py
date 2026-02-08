"""
ANNEALING MODULAR CONSENSUS
终极实验：退火耦合

阶段设计：
1. 隔离期 (Step 0-100): α=0, 独立演化
2. 渗透期 (Step 101-300): α → 0.05
3. 结晶期 (Step 301-Final): α → 0.2

目标：冲击80%正确率
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def annealing_test(N=1000, n_modules=50, trials=10):
    """退火耦合测试"""
    
    results = {'baseline': 0, 'annealing': 0}
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 10000 + N + n_modules)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # ==================== 基线 ====================
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 静态弱耦合
        s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
        for _ in range(500):
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + 0.1 * s_m)
            s = new_s
        if np.sign(np.mean(s)) == 1:
            results['baseline'] += 1
        
        # ==================== 退火耦合 ====================
        np.random.seed(t * 10000 + N + n_modules)
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 阶段1: 隔离期 (α=0, 独立演化)
        for step in range(100):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 阶段2: 渗透期 (α: 0 → 0.05)
        for step in range(200):
            alpha = 0.05 * (step / 200)  # 线性增加
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
            s = new_s
        
        # 阶段3: 结晶期 (α: 0.05 → 0.2)
        for step in range(200):
            alpha = 0.05 + 0.15 * (step / 200)  # 增加到0.2
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            results['annealing'] += 1
    
    return {
        'baseline': results['baseline'] * 100 // trials,
        'annealing': results['annealing'] * 100 // trials
    }

print("="*60)
print("ANNEALING MODULAR CONSENSUS")
print("Ultimate Experiment: Breaking the 70% Ceiling")
print("="*60)

print("\nRunning N=1000, M=50 with Annealing Coupling...")

result = annealing_test(N=1000, n_modules=50, trials=10)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"基线 (静态耦合 0.1):  {result['baseline']}%")
print(f"退火耦合:            {result['annealing']}%")
print(f"改善:                 +{result['annealing'] - result['baseline']}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if result['annealing'] >= 80:
    print("MILESTONE: 突破70%天花板！达到80%！")
elif result['annealing'] >= 70:
    print("SUCCESS: 达到理论天花板(~70%)")
elif result['annealing'] >= 60:
    print("PROGRESS: 显著提升，但未达天花板")
elif result['annealing'] >= result['baseline']:
    print("VALIDATED: 退火耦合有效")
else:
    print("NEEDS WORK: 需要调整参数")

# 保存结果
with open('results/annealing_consensus.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nSaved to results/annealing_consensus.json")
