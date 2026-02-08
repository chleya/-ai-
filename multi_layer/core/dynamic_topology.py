"""
DYNAMIC TOPOLOGY (Adaptive Modules)
基于互信息的动态模块重组

模拟生物大脑皮层的功能柱演化
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def compute_correlation(signal1, signal2):
    """计算两个信号的相关性"""
    return np.corrcoef(signal1, signal2)[0, 1]

def dynamic_topology(N=1000, n_modules=50, n_super_modules=10, phases=(100,200,200), trials=8):
    """动态拓扑实验"""
    
    success = 0
    module_size = N // n_modules
    super_size = n_modules // n_super_modules  # 每个超模块包含的模块数
    
    for t in range(trials):
        np.random.seed(t * 10000 + N + n_modules)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # 初始化模块
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 阶段1: 独立演化
        for step in range(phases[0]):
            signals = []
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
                signals.append(x_good)
        
        # 阶段2: 基于互信息的动态重组
        # 计算模块间的相关性矩阵
        correlation_matrix = np.zeros((n_modules, n_modules))
        for i in range(n_modules):
            for j in range(i+1, n_modules):
                corr = compute_correlation(signals[i], signals[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # 根据相关性重组为超模块
        # 高相关的模块组成超模块
        super_module_assignments = []
        used = set()
        for i in range(n_modules):
            if i in used: continue
            group = [i]
            used.add(i)
            for j in range(i+1, n_modules):
                if j in used: continue
                if correlation_matrix[i, j] > 0.5:  # 高相关阈值
                    group.append(j)
                    used.add(j)
            super_module_assignments.append(group)
        
        # 阶段3: 超模块级退火
        # 在超模块内部进行强耦合，超模块之间进行弱耦合
        for step in range(phases[1] + phases[2]):
            alpha_within = 0.3 if step > phases[1] else 0.1
            alpha_between = 0.05
            
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            
            # 超模块内部强耦合
            for group in super_module_assignments:
                group_signals = [s[m*module_size:(m+1)*module_size] for m in group]
                group_mean = np.mean(group_signals, axis=0)
                for m in group:
                    idx = slice(m*module_size, (m+1)*module_size)
                    s_m = s[idx]
                    new_s[idx] = nl(W_modules[m] @ s_m + alpha_within * group_mean)
            
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

print("="*60)
print("DYNAMIC TOPOLOGY (Adaptive Modules)")
print("Based on Mutual Information")
print("="*60)

print("\nRunning experiment...")
result = dynamic_topology()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Dynamic Topology: {result}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if result >= 80:
    print("BREAKTHROUGH: Dynamic topology achieves 80%+!")
elif result >= 70:
    print("SUCCESS: Dynamic topology effective")
elif result >= 60:
    print("PROGRESS: Partial improvement")
else:
    print("LIMITED: Needs more optimization")

# Save
with open('results/dynamic_topology.json', 'w') as f:
    json.dump({'dynamic_topology': result}, f, indent=2)
print("\nSaved to results/dynamic_topology.json")
