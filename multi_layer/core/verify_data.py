"""
SYSTEMATIC DATA VERIFICATION
Purpose: 确保LTD数据前后一致
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def run_ltd_test(N, n_nodes, steps, n_tests=20):
    """Test LTD效果在不同配置下"""
    
    # 创建节点配置
    # Node A: 清醒节点
    # Node B: 模糊节点  
    # Node C: 错误节点
    
    correct_baseline = 0
    correct_ltd = 0
    
    for test in range(n_tests):
        np.random.seed(test * 100 + steps + n_nodes)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        # 创建权重
        W_A = np.random.randn(N, N) / np.sqrt(N)
        W_B = np.random.randn(N, N) / np.sqrt(N)
        W_C = np.random.randn(N, N) / np.sqrt(N)
        
        # 学习
        for _ in range(steps):
            # A: 70%正确
            x_A = nl(W_A @ P + np.random.randn(N) * 0.3)
            W_A += 0.001 * np.outer(x_A, x_A)
            
            # B: 80%噪声
            x_B = nl(W_B @ P + np.random.randn(N) * 0.8)
            W_B += 0.001 * np.outer(x_B, x_B)
            
            # C: 错误
            x_C = nl(W_C @ (-P) + np.random.randn(N) * 0.8)
            W_C += 0.001 * np.outer(x_C, x_C)
            
            # 归一化
            for W in [W_A, W_B, W_C]:
                if np.linalg.norm(W) > 10:
                    W[:] = W / np.linalg.norm(W) * 10
        
        # 基线测试
        s_A = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s_B = (1 - 0.8) * P + 0.8 * np.random.randn(N)
        s_C = (1 - 0.8) * (-P) + 0.8 * np.random.randn(N)
        
        for _ in range(300):
            s_A = nl(W_A @ s_A + 0.2 * (s_A + s_B + s_C))
            s_B = nl(W_B @ s_B + 0.2 * (s_A + s_B + s_C))
            s_C = nl(W_C @ s_C + 0.2 * (s_A + s_B + s_C))
        
        if np.sign(np.mean(s_A)) == 1:
            correct_baseline += 1
        
        # LTD测试
        np.random.seed(test * 100 + steps + n_nodes)
        W_A = np.random.randn(N, N) / np.sqrt(N)
        W_B = np.random.randn(N, N) / np.sqrt(N)
        W_C = np.random.randn(N, N) / np.sqrt(N)
        
        for _ in range(steps):
            x_A = nl(W_A @ P + np.random.randn(N) * 0.3)
            W_A += 0.001 * np.outer(x_A, x_A)
            
            x_B = nl(W_B @ P + np.random.randn(N) * 0.8)
            W_B += 0.001 * np.outer(x_B, x_B)
            
            x_C = nl(W_C @ (-P) + np.random.randn(N) * 0.8)
            W_C += 0.001 * np.outer(x_C, x_C)
            
            for W in [W_A, W_B, W_C]:
                if np.linalg.norm(W) > 10:
                    W[:] = W / np.linalg.norm(W) * 10
        
        # LTD: 破坏C
        x_C_final = nl(W_C @ (-P) + np.random.randn(N) * 0.2)
        W_C -= 0.5 * np.outer(x_C_final, x_C_final)
        W_C[:] = W_C / np.linalg.norm(W_C) * 0.1  # 扁平化
        
        # LTD测试
        s_A = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s_B = (1 - 0.8) * P + 0.8 * np.random.randn(N)
        s_C = (1 - 0.8) * (-P) + 0.8 * np.random.randn(N)
        
        for _ in range(300):
            s_A = nl(W_A @ s_A + 0.2 * (s_A + s_B + s_C))
            s_B = nl(W_B @ s_B + 0.2 * (s_A + s_B + s_C))
            s_C = nl(W_C @ s_C + 0.2 * (s_A + s_B + s_C))
        
        if np.sign(np.mean(s_A)) == 1:
            correct_ltd += 1
    
    return {
        'baseline': correct_baseline / n_tests * 100,
        'ltd': correct_ltd / n_tests * 100,
        'improvement': correct_ltd / n_tests * 100 - correct_baseline / n_tests * 100
    }

print("="*70)
print("SYSTEMATIC LTD DATA VERIFICATION")
print("="*70)

# 测试不同配置
configs = [
    {'N': 50, 'steps': 1000, 'desc': 'N=50, 1000步'},
    {'N': 50, 'steps': 2000, 'desc': 'N=50, 2000步'},
    {'N': 50, 'steps': 5000, 'desc': 'N=50, 5000步'},
    {'N': 100, 'steps': 2000, 'desc': 'N=100, 2000步'},
    {'N': 200, 'steps': 2000, 'desc': 'N=200, 2000步'},
]

results = {}

for config in configs:
    result = run_ltd_test(config['N'], 3, config['steps'])
    results[config['desc']] = result
    print(f"{config['desc']}: 基线={result['baseline']:.0f}%, LTD={result['ltd']:.0f}%, 改善=+{result['improvement']:.0f}%")

print()
print("="*70)
print("KEY FINDINGS")
print("="*70)

# 分析：为何有差异？
print()
print("1. LTD效果随演化步数增加:")
for desc in ['N=50, 1000步', 'N=50, 2000步', 'N=50, 5000步']:
    r = results[desc]
    print(f"   {desc}: +{r['improvement']:.0f}%")

print()
print("2. 维度影响 (2000步):")
for desc in ['N=50, 2000步', 'N=100, 2000步', 'N=200, 2000步']:
    r = results[desc]
    print(f"   {desc}: +{r['improvement']:.0f}%")

print()
print("="*70)
print("DATA CONSISTENCY ANALYSIS")
print("="*70)

# 原始报告中提到的数据点
report_data = {
    "第7章LTD(标准)": 70.0,
    "第10章多A实验": 28.0,
    "Aggressive LTD文件": 70.0,
}

print()
print("报告数据 vs 实测数据:")
print("-"*50)

# 现在运行与原始报告相近的配置测试
test_result = run_ltd_test(N=50, n_nodes=3, steps=2000)
print(f"复现第7章配置(N=50, 3节点, 2000步):")
print(f"  预期: ~70%")
print(f"  实测: {test_result['ltd']:.0f}%")

# 保存结果
with open('results/systematic_verification.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print("Saved to results/systematic_verification.json")
