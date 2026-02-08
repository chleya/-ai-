"""
REDUCING ENTROPY: Momentum + Median Gating
终极方案A：惯性项（Momentum）
终极方案B：中位数门控（Median Gating）
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def momentum_test(N=1000, n_modules=50, mu=0.5, trials=30):
    """方案A：惯性项测试"""
    
    success = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 惯性项初始化
        delta_s = np.zeros(N)
        
        # 隔离期
        for step in range(100):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 结晶期（带惯性）
        alpha = 0.2
        for step in range(200):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
            
            # 惯性项：delta_s = mu * delta_s + (new_s - s)
            delta_s = mu * delta_s + (new_s - s)
            s = new_s + delta_s  # 应用惯性
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

def median_gating_test(N=1000, n_modules=50, trials=30):
    """方案B：中位数门控测试"""
    
    success = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        # 隔离期
        for step in range(100):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 结晶期（中位数门控）
        alpha = 0.2
        for step in range(200):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            
            # 计算每个模块的输出
            module_outputs = []
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                output = nl(W_modules[m] @ s_m + alpha * s_m)
                new_s[idx] = output
                module_outputs.append(np.mean(output))
            
            # 中位数门控：识别离群模块
            median = np.median(module_outputs)
            mad = np.median(np.abs(np.array(module_outputs) - median))  # MAD
            
            # 对离群模块应用更强的抑制
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                deviation = np.abs(module_outputs[m] - median)
                if deviation > 2 * mad:  # 超过2 MAD
                    # 抑制离群模块
                    new_s[idx] = new_s[idx] * 0.5
            
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

def combined_test(N=1000, n_modules=50, mu=0.5, trials=30):
    """组合方案：惯性项 + 中位数门控"""
    
    success = 0
    module_size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
            W_modules.append(W)
        
        delta_s = np.zeros(N)
        
        # 隔离期
        for step in range(100):
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # 结晶期（组合方案）
        alpha = 0.2
        for step in range(200):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            
            # 模块输出 + 中位数门控
            module_outputs = []
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                s_m = s[idx]
                output = nl(W_modules[m] @ s_m + alpha * s_m)
                new_s[idx] = output
                module_outputs.append(np.mean(output))
            
            # 中位数门控
            median = np.median(module_outputs)
            mad = np.median(np.abs(np.array(module_outputs) - median))
            for m in range(n_modules):
                idx = slice(m*module_size, (m+1)*module_size)
                if np.abs(module_outputs[m] - median) > 2 * mad:
                    new_s[idx] = new_s[idx] * 0.5
            
            # 惯性项
            delta_s = mu * delta_s + (new_s - s)
            s = new_s + delta_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

def momentum_sweep():
    """惯性项参数扫描"""
    
    print("="*60)
    print("MOMENTUM PARAMETER SWEEP")
    print("="*60)
    
    results = {}
    
    for mu in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        print(f"\nTesting mu={mu}...")
        rate = momentum_test(mu=mu, trials=20)
        results[f'mu={mu}'] = rate
        print(f"mu={mu}: {rate}%")
    
    return results

def main():
    """主实验"""
    
    print("="*60)
    print("REDUCING ENTROPY: Final Experiments")
    print("Target: Compress the distribution, eliminate outliers")
    print("="*60)
    
    # 基线
    print("\n[1] Baseline (no optimization):")
    baseline = momentum_test(mu=0, trials=30)
    print(f"    Baseline: {baseline}%")
    
    # 方案A：惯性项
    print("\n[2] Scheme A - Momentum:")
    momentum_results = momentum_sweep()
    
    # 方案B：中位数门控
    print("\n[3] Scheme B - Median Gating:")
    median_rate = median_gating_test(trials=30)
    print(f"    Median Gating: {median_rate}%")
    
    # 组合方案
    print("\n[4] Combined (Momentum + Median):")
    combined_rate = combined_test(mu=0.5, trials=30)
    print(f"    Combined: {combined_rate}%")
    
    # 汇总
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_results = {
        'baseline': baseline,
        'momentum': momentum_results,
        'median_gating': median_rate,
        'combined': combined_rate
    }
    
    print(f"Baseline:       {baseline}%")
    print(f"Best Momentum:  {max(momentum_results.values())}% (mu={max(momentum_results, key=momentum_results.get)})")
    print(f"Median Gating:  {median_rate}%")
    print(f"Combined:       {combined_rate}%")
    
    # 分析
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    best_mu = max(momentum_results, key=momentum_results.get)
    best_momentum = momentum_results[best_mu]
    
    if best_momentum > baseline:
        print(f"Momentum SUCCESS: mu={best_mu} improves {baseline}% -> {best_momentum}%")
    else:
        print("Momentum FAILED: No improvement over baseline")
    
    if median_rate > baseline:
        print(f"Median Gating SUCCESS: {baseline}% -> {median_rate}%")
    else:
        print("Median Gating FAILED: No improvement over baseline")
    
    if combined_rate > baseline:
        print(f"Combined SUCCESS: {baseline}% -> {combined_rate}%")
    else:
        print("Combined FAILED: No improvement over baseline")
    
    # 保存
    with open('results/reducing_entropy.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/reducing_entropy.json")

if __name__ == "__main__":
    main()
