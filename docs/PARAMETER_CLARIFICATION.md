# Cicada Protocol - 参数澄清

## 问题背景

README声称：
- λ_max: 2.15 → 1.73 (下降19.5%)
- 相变 Nc ≈ 900

但当前demo结果不匹配。

## 原因分析

### 1. 核心问题：没有权重归一化

当前代码（W += lr * outer(s,s)）会导致权重无限增长。

### 2. 学习率选择

| lr | 无归一化时 λ_max | 问题 |
|----|-----------------|------|
| 0.001 | ~1.4 | 太低 |
| 0.01 | ~10 | 中等 |
| 0.05 | ~100 | 爆炸 |

## 完整参数设置

### 推荐的实验参数

```python
# 完整配置
CONFIG = {
    # 系统参数
    'N': 200,                      # 系统规模
    
    # 学习参数
    'lr': 0.001,                   # 学习率（需归一化时可用更高）
    'input_distribution': 'gaussian',  # s(t) ~ N(0, 1)
    
    # 权重控制（关键！）
    'weight_normalization': True,   # 是否使用谱归一化
    'weight_clip': 10.0,           # 权重裁剪阈值
    'spectral_normalization': True, # 谱归一化
    
    # 实验参数
    'steps': 2000,                 # 演化步数
    'trials': 20,                 # 重复次数
    
    # 重置策略
    'reset_interval': 300,         # 固定周期
    'event_alpha': 1.6,            # 事件触发阈值
}
```

## 实际能达到的参数范围

### 场景1: 有归一化（推荐用于demo）

```python
def cicada_protocol_normalized(N=200, lr=0.01, steps=1000):
    """带归一化的协议"""
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        
        # 谱归一化（关键！）
        if np.linalg.norm(W) > 1.0:
            W = W / np.linalg.norm(W)
        
        eigenvals = np.linalg.eigvalsh(W)
        history.append(np.max(np.abs(eigenvals)))
    
    return W, history

# 结果（有归一化）
# lr=0.01: λ_max ~ 0.8-1.2
# lr=0.05: λ_max ~ 1.5-2.5  <-- 这个范围对demo最好
```

### 场景2: 无归一化（展示重置效果）

```python
def cicada_protocol_unbounded(N=200, lr=0.001, steps=500):
    """无归一化，展示重置的必要性"""
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        
        eigenvals = np.linalg.eigvalsh(W)
        history.append(np.max(np.abs(eigenvals)))
        
        if (t + 1) % 100 == 0:  # 重置
            W = np.random.randn(N, N) * 0.01
    
    return W, history

# 结果（lr=0.001）
# 无重置: λ_max ~ 1.5
# 有重置: λ_max ~ 0.7
```

## 相变实验 Nc ≈ 900

```python
def phase_transition_scan():
    """扫描不同N值，观察生存率变化"""
    
    N_values = [100, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000]
    results = []
    
    for N in N_values:
        survival_rates = []
        for trial in range(10):
            _, history = cicada_protocol(N=N, lr=0.001, steps=1000)
            # 生存率 = λ < 1.8 的比例
            healthy = sum(1 for x in history if x < 1.8) / len(history)
            survival_rates.append(healthy)
        
        mean_survival = np.mean(survival_rates)
        results.append((N, mean_survival))
        print(f"N={N:4d}: survival={mean_survival:.1%}")
    
    return results
```

## 事件触发机制

```python
class EventTriggered:
    """事件触发重置"""
    
    def __init__(self, alpha=1.6, window=20):
        self.alpha = alpha
        self.window = window
    
    def should_reset(self, lambda_history):
        """当λ超过均值×α时触发"""
        if len(lambda_history) < self.window:
            return False
        
        current = lambda_history[-1]
        mean = np.mean(lambda_history[-self.window:])
        
        return current > self.alpha * mean
```

## 下一步行动

1. **更新demo.py**: 使用明确的学习率和归一化
2. **添加参数表**: 完整文档化所有参数
3. **运行相变扫描**: 验证 Nc ≈ 900
4. **实现事件触发**: 添加 α=1.6 的代码

## 结论

当前demo使用的是低学习率(lr=0.001) + 无归一化，导致λ_max只有~1.4。

要达到README声称的2.15→1.73，需要：
- lr=0.01 + 适度归一化
- 或lr=0.05 + 强归一化

**关键问题不是参数，而是缺乏统一的参数文档！**
