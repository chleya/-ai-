"""
FINAL SANITIZATION: Three Devil's Checks
三项"魔鬼核查"：信息泄露、过拟合初期、景观深度
"""

import numpy as np, json

def nl(x): return np.tanh(x)

# ==================== 核查一：信息泄露检查 ====================
def check_information_leakage(N=1000, trials=20):
    """
    核查：噪声是否与正确模式A正交？
    """
    
    print("="*60)
    print("CHECK 1: Information Leakage")
    print("核查一：噪声是否与A正交？")
    print("="*60)
    
    # 生成正确模式A
    P = np.random.randn(N)
    P = P / np.linalg.norm(P)
    
    # 检查各种噪声与A的相关性
    correlations = {}
    
    for trial in range(trials):
        np.random.seed(trial * 100 + 42)
        
        # 1. 高斯白噪声
        gaussian = np.random.randn(N)
        gaussian = gaussian / np.linalg.norm(gaussian)
        corr = np.abs(np.dot(P, gaussian))
        correlations.setdefault('gaussian', []).append(corr)
        
        # 2. 正弦噪声
        t = np.arange(N)
        sine = np.sin(2 * np.pi * 0.1 * t / 100)
        sine = sine / np.linalg.norm(sine)
        corr = np.abs(np.dot(P, sine))
        correlations.setdefault('sine', []).append(corr)
        
        # 3. 随机共振脉冲
        for step in range(200):
            sr_pulse = 0.05 * np.sin(2 * np.pi * 0.1 * step / 100)
            if step == 100:  # 采样
                pulse = np.ones(N) * sr_pulse
                pulse = pulse / np.linalg.norm(pulse)
                corr = np.abs(np.dot(P, pulse))
                correlations.setdefault('sr_pulse', []).append(corr)
    
    # 统计
    print("\n噪声与A的相关性（0=完全正交，1=完全相同）：")
    for noise_type, corrs in correlations.items():
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs)
        print(f"  {noise_type}: {mean_corr:.6f} ± {std_corr:.6f}")
        
        if mean_corr < 0.01:
            print(f"    ✅ 结论：{noise_type}与A几乎完全正交，无信息泄露")
        elif mean_corr < 0.1:
            print(f"    ⚠️ 结论：{noise_type}与A有微弱相关性")
        else:
            print(f"    ❌ 警告：{noise_type}与A存在显著相关性！")
    
    return correlations

# ==================== 核查二：过拟合初期检查 ====================
def check_initial_bias(N=1000, trials=10):
    """
    核查：延长步数后，100%是否还能保持？
    """
    
    print("\n" + "="*60)
    print("CHECK 2: Initial Bias (Long-term Stability)")
    print("核查二：长期演化稳定性")
    print("="*60)
    
    results = {}
    
    for steps in [200, 1000, 5000, 10000]:
        successes = 0
        
        for trial in range(trials):
            np.random.seed(trial * 1000 + 42)
            M, alpha = 50, 0.2
            size = N // M
            
            # 生成模式
            P = np.random.randn(N)
            P = P / np.linalg.norm(P)
            
            # 初始化权重
            Ws = [np.random.randn(size, size) / np.sqrt(size) for _ in range(M)]
            
            # 隔离期
            for step in range(100):
                for m in range(M):
                    idx = slice(m*size, (m+1)*size)
                    xg = nl(Ws[m] @ P[idx] + np.random.randn(size) * 0.3)
                    Ws[m] += 0.001 * np.outer(xg, xg)
                    if np.linalg.norm(Ws[m]) > 10:
                        Ws[m] = Ws[m] / np.linalg.norm(Ws[m]) * 10
            
            # 结晶期（不同长度）
            for step in range(steps):
                s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
                new_s = np.zeros(N)
                
                for m in range(M):
                    idx = slice(m*size, (m+1)*size)
                    s_m = s[idx]
                    noise_pulse = 0.05 * np.sin(2 * np.pi * 0.1 * step / 100)
                    new_s[idx] = nl(Ws[m] @ s_m + alpha * s_m + noise_pulse)
                
                s = new_s
            
            if np.sign(np.mean(s)) == 1:
                successes += 1
        
        rate = successes * 100 // trials
        results[steps] = rate
        print(f"  Steps={steps:5d}: {rate}%")
    
    return results

# ==================== 核查三：景观深度分析 ====================
def analyze_landscape_depth(N=1000, trials=20):
    """
    核查：A、B、C三个模式的能量深度对比
    """
    
    print("\n" + "="*60)
    print("CHECK 3: Landscape Profiling")
    print("核查三：能量景观深度分析")
    print("="*60)
    
    depths = {'A': [], 'B': [], 'C': []}
    
    for trial in range(trials):
        np.random.seed(trial * 100 + 42)
        M, size = 50, N // 50
        
        # 生成三个模式
        P_A = np.random.randn(N)
        P_A = P_A / np.linalg.norm(P_A)
        P_B = np.random.randn(N)
        P_B = P_B / np.linalg.norm(P_B)
        P_C = -P_A  # 错误模式
        
        # 初始化权重
        Ws = [np.random.randn(size, size) / np.sqrt(size) for _ in range(M)]
        
        # 模拟学习过程
        for step in range(100):
            # A模式被正向学习
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                x_A = nl(Ws[m] @ P_A[idx] + np.random.randn(size) * 0.3)
                Ws[m] += 0.001 * np.outer(x_A, x_A)
                if np.linalg.norm(Ws[m]) > 10:
                    Ws[m] = Ws[m] / np.linalg.norm(Ws[m]) * 10
        
        # 计算每个模式的能量深度
        for mode_name, P in [('A', P_A), ('B', P_B), ('C', P_C)]:
            energy = 0
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                x = nl(Ws[m] @ P[idx] + np.random.randn(size) * 0.3)
                energy += -np.mean(x)  # 能量 = -激活（负值表示稳定）
            depths[mode_name].append(energy)
    
    # 统计
    print("\n能量深度（越负越深=越稳定）：")
    for mode_name, energy_list in depths.items():
        mean_e = np.mean(energy_list)
        std_e = np.std(energy_list)
        print(f"  模式{mode_name}: {mean_e:.4f} ± {std_e:.4f}")
    
    # 分析
    A_depth = np.mean(depths['A'])
    B_depth = np.mean(depths['B'])
    C_depth = np.mean(depths['C'])
    
    print("\n深度对比：")
    print(f"  A vs B: {(A_depth - B_depth):.4f}")
    print(f"  A vs C: {(A_depth - C_depth):.4f}")
    print(f"  B vs C: {(B_depth - C_depth):.4f}")
    
    if A_depth < B_depth and A_depth < C_depth:
        print("  ✅ A是最深的模式，正确引导成立")
    else:
        print("  ❌ 警告：A不是最深的！")
    
    return depths

# ==================== 绝户网测试 ====================
def final_stress_test(N=1000, trials=10):
    """
    绝户网测试：反向注入、零信号、结构扰动
    """
    
    print("\n" + "="*60)
    print("FINAL STRESS TESTS")
    print("绝户网测试")
    print("="*60)
    
    results = {}
    
    # 测试1：反向注入
    print("\n[1] Reverse Injection (noise=0.5):")
    successes = 0
    for trial in range(trials):
        np.random.seed(trial * 1000 + 42)
        M, alpha, size = 50, 0.2, N // 50
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        Ws = [np.random.randn(size, size) / np.sqrt(size) for _ in range(M)]
        
        for _ in range(100):
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                xg = nl(Ws[m] @ P[idx] + np.random.randn(size) * 0.3)
                Ws[m] += 0.001 * np.outer(xg, xg)
                if np.linalg.norm(Ws[m]) > 10:
                    Ws[m] = Ws[m] / np.linalg.norm(Ws[m]) * 10
        
        for step in range(1000):  # 延长步数
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                s_m = s[idx]
                # 反向噪声
                noise = 0.5 * np.sin(2 * np.pi * 0.1 * step / 100)
                new_s[idx] = nl(Ws[m] @ s_m + alpha * s_m - noise)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            successes += 1
    
    results['reverse'] = successes * 100 // trials
    print(f"  Result: {results['reverse']}%")
    
    # 测试2：零信号（只留50% C）
    print("\n[2] Zero Signal (Only C, no A):")
    successes = 0
    for trial in range(trials):
        np.random.seed(trial * 1000 + 42)
        M, alpha, size = 50, 0.2, N // 50
        # 只生成C模式
        P_C = np.ones(N)  # 纯错误
        
        Ws = [np.random.randn(size, size) / np.sqrt(size) for _ in range(M)]
        
        for _ in range(100):
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                xc = nl(Ws[m] @ P_C[idx] + np.random.randn(size) * 0.3)
                Ws[m] += 0.001 * np.outer(xc, xc)
                if np.linalg.norm(Ws[m]) > 10:
                    Ws[m] = Ws[m] / np.linalg.norm(Ws[m]) * 10
        
        for step in range(1000):
            s = np.random.randn(N)  # 纯噪声初始
            new_s = np.zeros(N)
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                s_m = s[idx]
                noise = 0.05 * np.sin(2 * np.pi * 0.1 * step / 100)
                new_s[idx] = nl(Ws[m] @ s_m + alpha * s_m + noise)
            s = new_s
        
        if np.sign(np.mean(s)) == -1:  # 正确收敛到-C
            successes += 1
    
    results['zero_signal'] = successes * 100 // trials
    print(f"  Result: {results['zero_signal']}%")
    
    # 测试3：结构扰动
    print("\n[3] Structural Perturbation (flip 10% weights at step 500):")
    successes = 0
    for trial in range(trials):
        np.random.seed(trial * 1000 + 42)
        M, alpha, size = 50, 0.2, N // 50
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        Ws = [np.random.randn(size, size) / np.sqrt(size) for _ in range(M)]
        
        for _ in range(100):
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                xg = nl(Ws[m] @ P[idx] + np.random.randn(size) * 0.3)
                Ws[m] += 0.001 * np.outer(xg, xg)
                if np.linalg.norm(Ws[m]) > 10:
                    Ws[m] = Ws[m] / np.linalg.norm(Ws[m]) * 10
        
        # 演化500步后扰动
        for step in range(1000):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            
            for m in range(M):
                idx = slice(m*size, (m+1)*size)
                s_m = s[idx]
                noise = 0.05 * np.sin(2 * np.pi * 0.1 * step / 100)
                new_s[idx] = nl(Ws[m] @ s_m + alpha * s_m + noise)
            
            # 在第500步进行扰动
            if step == 500:
                for m in range(M):
                    if np.random.rand() < 0.5:  # 50%模块被扰动
                        # 反转10%权重
                        n_flip = int(0.1 * size * size)
                        flip_indices = np.random.choice(size * size, n_flip, replace=False)
                        Ws[m].flat[flip_indices] *= -1
            
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            successes += 1
    
    results['perturbation'] = successes * 100 // trials
    print(f"  Result: {results['perturbation']}%")
    
    return results

def main():
    print("\n" + "="*60)
    print("FINAL SANITIZATION: Three Devil's Checks")
    print("三项\"魔鬼核查\"")
    print("="*60)
    
    # 核查一：信息泄露
    correlations = check_information_leakage()
    
    # 核查二：过拟合初期
    long_term = check_initial_bias()
    
    # 核查三：景观深度
    depths = analyze_landscape_depth()
    
    # 绝户网测试
    stress = final_stress_test()
    
    # 汇总
    print("\n" + "="*60)
    print("FINAL VERIFICATION SUMMARY")
    print("最终核查汇总")
    print("="*60)
    
    print("\n1. Information Leakage:")
    leak_free = all(np.mean(c) < 0.01 for c in correlations.values())
    print(f"   Status: {'✅ CLEAN' if leak_free else '❌ SUSPICIOUS'}")
    
    print("\n2. Long-term Stability:")
    stable = all(r >= 80 for r in long_term.values())
    print(f"   Status: {'✅ STABLE' if stable else '❌ UNSTABLE'}")
    
    print("\n3. Landscape Depth:")
    A_deepest = np.mean(depths['A']) < np.mean(depths['B']) and np.mean(depths['A']) < np.mean(depths['C'])
    print(f"   Status: {'✅ A DEEPEST' if A_deepest else '❌ A NOT DEEPEST'}")
    
    print("\n4. Stress Tests:")
    all_pass = all(v >= 80 for v in stress.values())
    print(f"   Status: {'✅ ROBUST' if all_pass else '❌ FRAGILE'}")
    
    # 最终判断
    print("\n" + "="*60)
    print("FINAL JUDGMENT")
    print("最终判断")
    print("="*60)
    
    if leak_free and stable and A_deepest and all_pass:
        print("✅ 100% RESULT IS AUTHENTIC")
        print("   随机共振的100%结果是真实的！")
    elif leak_free and stable and A_deepest:
        print("⚠️ 100% RESULT IS MOSTLY AUTHENTIC")
        print("   但需要更多stress test验证")
    else:
        print("❌ 100% RESULT NEEDS MORE VERIFICATION")
        print("   存在可疑之处，需要进一步调查")
    
    return {
        'correlations': correlations,
        'long_term': long_term,
        'depths': depths,
        'stress': stress
    }

if __name__ == "__main__":
    results = main()
    
    # 保存
    with open('results/final_sanitization.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/final_sanitization.json")
