"""
Complete Stress Threshold Analysis Dashboard
包含：热力图、相图、效率曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def run_full_analysis():
    """运行完整分析"""
    
    print("="*70)
    print("STRESS THRESHOLD COMPLETE ANALYSIS")
    print("="*70)
    
    # 参数设置
    N = 500
    alphas = np.linspace(1.1, 3.0, 15)
    trials = 10
    total_steps = 600
    
    results = {a: {'survival': [], 'resets': [], 'jitters': []} for a in alphas}
    
    for trial in range(trials):
        np.random.seed(trial * 999 + 42)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        for alpha in alphas:
            W = np.random.randn(N, N) / np.sqrt(N)
            s = np.random.randn(N)
            jitters = []
            survival = 0
            resets = 0
            
            for e in range(total_steps):
                s = (1 - 0.5) * s + 0.5 * np.random.randn(N)
                s = np.tanh(W @ s + 0.2 * s + 0.05 * np.sin(0.1 * e))
                W = W + 0.001 * np.outer(s, s)
                
                if np.linalg.norm(W) > 10:
                    W = W / np.linalg.norm(W) * 10
                
                jitter = np.std(s)
                jitters.append(jitter)
                
                if np.mean(s) > 0:
                    survival += 1
                
                if len(jitters) > 20 and jitters[-1] > np.mean(jitters[-20:]) * alpha:
                    W = np.random.randn(N, N) / np.sqrt(N)
                    s = np.random.randn(N)
                    jitters = []
                    resets += 1
            
            results[alpha]['survival'].append(survival)
            results[alpha]['resets'].append(resets)
            results[alpha]['jitters'].append(jitters)
    
    # 计算统计数据
    stats = {}
    for alpha in alphas:
        survival_mean = np.mean(results[alpha]['survival'])
        survival_std = np.std(results[alpha]['survival'])
        resets_mean = np.mean(results[alpha]['resets'])
        efficiency = survival_mean / (resets_mean + 0.001)
        
        stats[alpha] = {
            'survival_mean': survival_mean,
            'survival_std': survival_std,
            'resets_mean': resets_mean,
            'efficiency': efficiency
        }
    
    # 打印结果
    print()
    print("RESULTS:")
    print("-" * 70)
    print(f"{'Alpha':<10} {'Survival':<15} {'Resets':<10} {'Efficiency':<15}")
    print("-" * 70)
    
    for alpha in alphas:
        s = stats[alpha]['survival_mean']
        r = stats[alpha]['resets_mean']
        e = stats[alpha]['efficiency']
        print(f"{alpha:<10.2f} {s:<15.1f} {r:<10.1f} {e:<15.2f}")
    
    # 找到最佳点
    best_survival = max(stats, key=lambda a: stats[a]['survival_mean'])
    best_efficiency = max(stats, key=lambda a: stats[a]['efficiency'])
    
    print()
    print("BEST POINTS:")
    print(f"  Best Survival:   alpha={best_survival:.2f}, survival={stats[best_survival]['survival_mean']:.1f}")
    print(f"  Best Efficiency: alpha={best_efficiency:.2f}, efficiency={stats[best_efficiency]['efficiency']:.2f}")
    
    # 生成可视化
    create_visualizations(alphas, stats, results)
    
    return stats

def create_visualizations(alphas, stats, results):
    """生成可视化图表"""
    
    # 图1：存活率曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    survival_means = [stats[a]['survival_mean'] for a in alphas]
    survival_stds = [stats[a]['survival_std'] for a in alphas]
    resets_means = [stats[a]['resets_mean'] for a in alphas]
    efficiencies = [stats[a]['efficiency'] for a in alphas]
    
    # 1.1 存活率曲线
    ax1.errorbar(alphas, survival_means, yerr=survival_stds, fmt='o-', capsize=3)
    ax1.set_xlabel('Alpha (Threshold Multiplier)')
    ax1.set_ylabel('Survival Rate (steps)')
    ax1.set_title('Survival Rate vs Alpha')
    ax1.grid(True, alpha=0.3)
    best_s = max(alphas, key=lambda a: stats[a]['survival_mean'])
    ax1.axvline(best_s, color='r', linestyle='--', alpha=0.7)
    
    # 1.2 重置次数
    ax2.plot(alphas, resets_means, 'r-o')
    ax2.set_xlabel('Alpha (Threshold Multiplier)')
    ax2.set_ylabel('Average Reset Count')
    ax2.set_title('Reset Frequency vs Alpha')
    ax2.grid(True, alpha=0.3)
    
    # 1.3 效率曲线
    ax3.plot(alphas, efficiencies, 'g-o')
    ax3.set_xlabel('Alpha (Threshold Multiplier)')
    ax3.set_ylabel('Efficiency (Survival/Resets)')
    ax3.set_title('Efficiency Heatmap')
    ax3.grid(True, alpha=0.3)
    best_e = max(alphas, key=lambda a: stats[a]['efficiency'])
    ax3.axvline(best_e, color='b', linestyle='--', alpha=0.7)
    
    # 1.4 综合热力图
    heatmap_data = np.array([efficiencies])
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                    extent=[alphas[0], alphas[-1], 0, 1])
    ax4.set_xlabel('Alpha (Threshold Multiplier)')
    ax4.set_ylabel('Efficiency')
    ax4.set_title('Efficiency Heatmap')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('stress_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print()
    print("Visualization saved to: stress_threshold_analysis.png")
    
    # 生成ASCII热力图
    print()
    print("ASCII EFFICIENCY HEATMAP:")
    print("="*50)
    
    max_eff = max(efficiencies)
    min_eff = min(efficiencies)
    
    for i, (a, e) in enumerate(zip(alphas, efficiencies)):
        # 归一化到0-10
        score = int((e - min_eff) / (max_eff - min_eff) * 10)
        bar = '█' * (score + 1)
        print(f"alpha={a:.1f}: {bar} {e:.2f}")

if __name__ == '__main__':
    stats = run_full_analysis()
