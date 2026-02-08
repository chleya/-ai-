"""
LARGE-SCALE EXPANSION EXPERIMENTS
Target: 10+ nodes, N=100-500
验证小规模结论在大规模系统中的适用性
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def run_large_scale_test(N, n_nodes, steps, n_tests=15):
    """大规模LTD测试"""
    
    correct_baseline = 0
    correct_ltd = 0
    
    for test in range(n_tests):
        np.random.seed(test * 100 + N + n_nodes)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        # 创建节点
        W_correct = np.random.randn(N, N) / np.sqrt(N)
        W_wrong = np.random.randn(N, N) / np.sqrt(N)
        
        # 学习
        for _ in range(steps):
            # 正确模式
            x_good = nl(W_correct @ P + np.random.randn(N) * 0.3)
            W_correct += 0.001 * np.outer(x_good, x_good)
            
            # 错误模式
            x_bad = nl(W_wrong @ (-P) + np.random.randn(N) * 0.5)
            W_wrong += 0.001 * np.outer(x_bad, x_bad)
            
            # 归一化
            if np.linalg.norm(W_correct) > 10:
                W_correct = W_correct / np.linalg.norm(W_correct) * 10
            if np.linalg.norm(W_wrong) > 10:
                W_wrong = W_wrong / np.linalg.norm(W_wrong) * 10
        
        # 基线测试
        s_good = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s_bad = (1 - 0.5) * (-P) + 0.5 * np.random.randn(N)
        
        # 多个模糊节点
        fuzzy_nodes = []
        for i in range(n_nodes - 2):
            s_fuzzy = (1 - 0.8) * P + 0.8 * np.random.randn(N)
            fuzzy_nodes.append(s_fuzzy)
        
        # 演化
        for _ in range(300):
            s_good = nl(W_correct @ s_good + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
            s_bad = nl(W_wrong @ s_bad + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
            for i, s_f in enumerate(fuzzy_nodes):
                fuzzy_nodes[i] = nl(W_correct @ s_f + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
        
        if np.sign(np.mean(s_good)) == 1:
            correct_baseline += 1
        
        # LTD测试
        np.random.seed(test * 100 + N + n_nodes)
        W_correct = np.random.randn(N, N) / np.sqrt(N)
        W_wrong = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(steps):
            x_good = nl(W_correct @ P + np.random.randn(N) * 0.3)
            W_correct += 0.001 * np.outer(x_good, x_good)
            x_bad = nl(W_wrong @ (-P) + np.random.randn(N) * 0.5)
            W_wrong += 0.001 * np.outer(x_bad, x_bad)
            
            if np.linalg.norm(W_correct) > 10:
                W_correct = W_correct / np.linalg.norm(W_correct) * 10
            if np.linalg.norm(W_wrong) > 10:
                W_wrong = W_wrong / np.linalg.norm(W_wrong) * 10
        
        # LTD: 破坏错误吸引子
        x_bad_final = nl(W_wrong @ (-P) + np.random.randn(N) * 0.2)
        W_wrong -= 0.5 * np.outer(x_bad_final, x_bad_final)
        W_wrong = W_wrong / np.linalg.norm(W_wrong) * 0.1
        
        # LTD测试
        s_good = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s_bad = (1 - 0.5) * (-P) + 0.5 * np.random.randn(N)
        
        fuzzy_nodes = []
        for i in range(n_nodes - 2):
            s_fuzzy = (1 - 0.8) * P + 0.8 * np.random.randn(N)
            fuzzy_nodes.append(s_fuzzy)
        
        for _ in range(300):
            s_good = nl(W_correct @ s_good + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
            s_bad = nl(W_wrong @ s_bad + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
            for i, s_f in enumerate(fuzzy_nodes):
                fuzzy_nodes[i] = nl(W_correct @ s_f + 0.2 * (s_good + s_bad + sum(fuzzy_nodes)))
        
        if np.sign(np.mean(s_good)) == 1:
            correct_ltd += 1
    
    return {
        'baseline': correct_baseline / n_tests * 100,
        'ltd': correct_ltd / n_tests * 100,
        'improvement': correct_ltd / n_tests * 100 - correct_baseline / n_tests * 100
    }

print("="*70)
print("LARGE-SCALE EXPANSION EXPERIMENTS")
print("="*70)

# 实验配置
configs = [
    {'N': 100, 'nodes': 5, 'steps': 2000, 'desc': 'N=100, 5节点'},
    {'N': 100, 'nodes': 10, 'steps': 2000, 'desc': 'N=100, 10节点'},
    {'N': 100, 'nodes': 20, 'steps': 2000, 'desc': 'N=100, 20节点'},
    {'N': 200, 'nodes': 10, 'steps': 3000, 'desc': 'N=200, 10节点'},
    {'N': 500, 'nodes': 10, 'steps': 3000, 'desc': 'N=500, 10节点'},
    {'N': 500, 'nodes': 50, 'steps': 3000, 'desc': 'N=500, 50节点'},
]

results = {}

for config in configs:
    print(f"\nRunning: {config['desc']}...")
    result = run_large_scale_test(config['N'], config['nodes'], config['steps'])
    results[config['desc']] = result
    print(f"  基线: {result['baseline']:.0f}%, LTD: {result['ltd']:.0f}%, 改善: +{result['improvement']:.0f}%")

print()
print("="*70)
print("LARGE-SCALE RESULTS SUMMARY")
print("="*70)

for desc, result in results.items():
    marker = '⭐' if result['ltd'] >= 70 else ('✓' if result['ltd'] >= 50 else ' ')
    print(f"{marker} {desc}: 基线{result['baseline']:.0f}% → LTD{result['ltd']:.0f}% (+{result['improvement']:.0f}%)")

print()
print("="*70)
print("ANALYSIS: Does LTD Scale to Large Systems?")
print("="*70)

# 分析趋势
small_scale = results.get('N=100, 5节点', {'ltd': 0})
large_scale_10 = results.get('N=100, 10节点', {'ltd': 0})
large_scale_50 = results.get('N=500, 50节点', {'ltd': 0})

print()
print("LTD效果随规模变化:")
print(f"  小规模 (5节点): {small_scale['ltd']:.0f}%")
print(f"  中规模 (10节点): {large_scale_10['ltd']:.0f}%")
print(f"  大规模 (50节点): {large_scale_50['ltd']:.0f}%")

if large_scale_50['ltd'] >= 50:
    print()
    print("✅ SUCCESS: LTD scales to large systems!")
    print("   大规模系统中LTD依然有效。")
elif large_scale_50['ltd'] >= 30:
    print()
    print("⚠️  PARTIAL: LTD效果随规模下降，但依然有效。")
else:
    print()
    print("❌ FAILURE: LTD在规模化后失效。")

# 保存结果
with open('results/large_scale_expansion.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print("Saved to results/large_scale_expansion.json")
