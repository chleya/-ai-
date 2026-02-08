"""
ANNEALING PARAMETER SWEEP
多变量退火耦合优化
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def annealing_variation(N, n_modules, phases, growth_func, trials=8):
    """退火变体测试"""
    
    success = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 10000 + N + n_modules + phases[0])
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 阶段1: 隔离期
        for step in range(phases[0]):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 阶段2: 渗透期 + 结晶期
        total_remaining = phases[1] + phases[2]
        for step in range(total_remaining):
            alpha = growth_func(step, total_remaining)
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

print("="*60)
print("ANNEALING PARAMETER SWEEP")
print("Optimizing for 80% Target")
print("="*60)

# 退火函数定义
def linear(step, total): return 0.15 * (step / total)
def sigmoid(step, total): return 0.2 / (1 + np.exp(-10 * (step/total - 0.5)))
def exponential(step, total): return 0.2 * (1 - np.exp(-3 * step / total))

configs = [
    ((100, 200, 300), linear, "标准线性 (100-200-300)"),
    ((150, 150, 300), linear, "长隔离期 (150-150-300)"),
    ((100, 300, 200), sigmoid, "缓慢增长 (100-300-200)"),
    ((50, 300, 300), exponential, "指数增长 (50-300-300)"),
    ((200, 200, 200), sigmoid, "均衡配置 (200-200-200)"),
]

all_results = {}

for phases, func, name in configs:
    print(f"\n{name}...")
    result = annealing_variation(1000, 50, phases, func)
    all_results[name] = result
    print(f"  正确率: {result}%")

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)

sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
for i, (name, result) in enumerate(sorted_results):
    marker = "[BEST]" if i == 0 else ("[TOP3]" if i < 3 else "")
    print(f"{marker} {name}: {result}%")

print("\n" + "="*60)
print("TARGET ANALYSIS")
print("="*60)

best = sorted_results[0][1]
if best >= 80:
    print("MILESTONE: 突破80%天花板！")
elif best >= 70:
    print("SUCCESS: 达到理论极限(~70%)")
elif best >= 60:
    print("PROGRESS: 显著提升，需继续优化")
else:
    print("LIMITED: 需要重新设计")

# Save
with open('results/annealing_sweep.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/annealing_sweep.json")
