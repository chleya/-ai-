"""
NONLINEAR ANNEALING CURVES
探索指数、Sigmoid、分段等非线性退火曲线
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def nonlinear_annealing(N=1000, n_modules=50, phases=(100,200,200), trials=8):
    """非线性退火曲线测试"""
    
    results = {}
    module_size = N // n_modules
    
    # 退火曲线定义
    curves = {
        'linear': lambda step, total: 0.3 * (step / total),
        'exponential': lambda step, total: 0.3 * (1 - np.exp(-3 * step / total)),
        'sigmoid_fast': lambda step, total: 0.3 / (1 + np.exp(-10 * (step/total - 0.3))),
        'sigmoid_slow': lambda step, total: 0.3 / (1 + np.exp(-15 * (step/total - 0.5))),
        'step_50': lambda step, total: 0.3 if step > total * 0.5 else 0.1,
        'step_70': lambda step, total: 0.3 if step > total * 0.7 else 0.1,
        'warmup': lambda step, total: 0.1 + 0.2 * np.sin(np.pi * step / total),
    }
    
    for curve_name, curve_func in curves.items():
        success = 0
        for t in range(trials):
            np.random.seed(t * 10000 + N + n_modules + len(curves))
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            
            W_modules = []
            for m in range(n_modules):
                W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
                W_modules.append(W)
            
            # 隔离期
            for step in range(phases[0]):
                for m in range(n_modules):
                    idx = slice(m*module_size, (m+1)*module_size)
                    P_m = P[idx]
                    x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                    W_modules[m] += 0.001 * np.outer(x_good, x_good)
                    if np.linalg.norm(W_modules[m]) > 10:
                        W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
            
            # 非线性退火期
            total_annealing = phases[1] + phases[2]
            for step in range(total_annealing):
                alpha = curve_func(step, total_annealing)
                s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
                new_s = np.zeros(N)
                for m in range(n_modules):
                    idx = slice(m*module_size, (m+1)*module_size)
                    s_m = s[idx]
                    new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
                s = new_s
            
            if np.sign(np.mean(s)) == 1:
                success += 1
        
        results[curve_name] = success * 100 // trials
    
    return results

print("="*60)
print("NONLINEAR ANNEALING CURVES")
print("Testing: Exponential, Sigmoid, Step, Warmup")
print("="*60)

print("\nRunning experiments...")
results = nonlinear_annealing()

print("\n" + "="*60)
print("RESULTS")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (curve, rate) in enumerate(sorted_results):
    marker = "[BEST]" if i == 0 else ("[TOP3]" if i < 3 else "")
    print(f"{marker} {curve}: {rate}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

best = sorted_results[0]
if best[1] >= 80:
    print(f"BREAKTHROUGH: {best[0]} achieves {best[1]}%!")
elif best[1] >= 70:
    print(f"SUCCESS: {best[0]} achieves {best[1]}%")
elif best[1] >= 60:
    print(f"PROGRESS: {best[0]} achieves {best[1]}%")
else:
    print(f"LIMITED: Best result {best[1]}%")

# Save
with open('results/nonlinear_annealing.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/nonlinear_annealing.json")
