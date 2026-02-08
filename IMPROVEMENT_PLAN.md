# Cicada Protocol - 改进计划

## 当前问题总结

| 问题 | 严重程度 | 现状 | 目标 |
|------|----------|------|------|
| λ_max偏低 | 🔴 高 | 1.4 | 2.0+ |
| 相变证据不足 | 🔴 高 | 无数据 | N=100~2000扫描 |
| 事件触发缺失 | 🟡 中 | 只有固定周期 | α=1.6实现 |
| 任务切换缺失 | 🟡 中 | 无 | +11.9%对比 |
| 参数不透明 | 🟡 中 | 未文档化 | 完整参数表 |

---

## 1. 参数统一问题

### 当前demo参数（core.py）
```python
learning_rate: 0.001  # η
noise_level: 0.5
N: 200
steps: 800
```

### 问题
- η=0.001 太小 → λ_max 只有 ~1.4
- README声称 2.15 → 1.73，但demo达不到

### 需要明确
1. 什么条件下λ_max会达到2.0+？
2. N、η、输入分布、正则化等参数的具体值

---

## 2. 实验改进计划

### Phase 1: 基准实验（当前最缺）

#### 2.1 λ_max vs η 扫描
```python
for eta in [0.001, 0.01, 0.05, 0.1]:
    for N in [50, 100, 200, 500, 1000]:
        run_experiment(eta=eta, N=N, trials=10)
```

#### 2.2 相变扫描 (N_c ≈ 900)
```python
N_values = [100, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000]
survival_rates = []
for N in N_values:
    rate = run_experiment(N=N, trials=20)
    survival_rates.append(rate)
```

#### 2.3 固定参数集（必须统一！）
```python
# 推荐的基准参数
CONFIG = {
    'N': 200,                    # 系统规模
    'eta': 0.01,                 # 学习率（需要验证）
    'input_distribution': 'gaussian',  # s(t) ~ N(0,1)
    'normalization': 'spectral',  # 谱归一化
    'weight_clip': 10.0,          # 权重裁剪阈值
    'steps': 2000,               # 演化步数
    'trials': 20,                # 重复次数
}
```

---

## 3. 事件触发机制实现

### 当前缺失
- README提到α=1.6，但没有代码实现
- 没有threshold-based触发逻辑

### 需要实现
```python
class EventTriggeredStrategy:
    """事件触发重置策略"""
    
    def __init__(self, alpha: float = 1.6, window: int = 20):
        self.alpha = alpha
        self.window = window
    
    def should_reset(self, lambda_history: List[float]) -> bool:
        """
        当 λ_max(t) > α × mean(λ_max recent) 时触发
        """
        if len(lambda_history) < self.window:
            return False
        
        current = lambda_history[-1]
        mean_recent = np.mean(lambda_history[-self.window:])
        
        return current > self.alpha * mean_recent
```

---

## 4. 任务切换实验

### 当前缺失
- 没有多任务实验
- 没有"峰值继承"vs"随机重置"对比

### 需要实现
```python
class TaskSwitchingExperiment:
    """任务切换实验"""
    
    def __init__(self, task_a_config, task_b_config):
        self.task_a = task_a_config
        self.task_b = task_b_config
    
    def run(self, strategy: str = 'reset'):
        """
        比较:
        - 随机重置 (reset): 每个任务用新的随机初始化
        - 峰值继承 (inherit): 保留上个任务的权重
        """
        # 实现对比实验
        pass
```

---

## 5. 理论改进

### 当前问题
- Isotropy Theory 缺乏推导
- 没有引用相关文献

### 需要添加
1. 完整的数学推导（附录）
2. 相关工作引用：
   - Amari (1977) - 神经网络的自然梯度
   - Saxe et al. (2013) - 深度学习中的相变
   - Sompolinsky et al. (1988) - 随机神经网络动力学

---

## 6. 改进进度

### ✅ 已完成

| 日期 | 优先级 | 任务 | 状态 |
|------|--------|------|------|
| 2026-02-08 | P0 | 相变扫描 N=50~2000 | ✅ 完成 |
| 2026-02-08 | P1 | 事件触发 α=1.6 | ✅ 完成 |
| 2026-02-08 | P1 | 任务切换实验 | ✅ 完成 |

### ⏳ 待改进

| 优先级 | 任务 | 目标 | 工作量 |
|--------|------|------|--------|
| P1 | 任务切换实验 | 验证 +11.9% claim | 需调参 |
| P2 | 理论推导补充 | 完整数学推导 | 5天 |

---

## 7. 快速验证命令

```bash
# 1. 检查当前demo
python examples/demo.py

# 2. 运行N=1000相变实验
python examples/test_n1000.py

# 3. CLI对比实验
python -m cicada --compare --N 200 --trials 10
```

---

## 8. 下一步行动

### 立即做（今天）
1. [ ] 在README中添加完整参数表
2. [ ] 统一demo.py和core.py的参数
3. [ ] 验证λ_max能达到2.0+

### 本周内
4. [ ] 实现事件触发策略（CicadaProtocol中添加）
5. [ ] 运行N=100~2000相变扫描
6. [ ] 生成相变图

### 长期
7. [ ] 任务切换实验
8. [ ] 理论推导补充
9. [ ] 论文投稿（OSDI/SOSP）

---

## 关键问题：为什么随机重置比峰值继承更好？

### 几何解释（需要补充）
1. **权重空间搜索**：随机初始化 = 从新区域开始搜索
2. **逃逸局部最优**：避免困在不良吸引子
3. **各向同性**：随机W具有等方性，不偏向特定方向

### 实验验证（需要做）
```python
# 伪代码
def compare_inheritance_vs_reset():
    results = {}
    
    # 方法1: 随机重置
    results['reset'] = run_with_random_reset()
    
    # 方法2: 峰值继承
    results['inherit'] = run_with_peak_inheritance()
    
    # 比较
    return results
```

---

## 参考：相关工作

- **Hebbian学习不稳定性**: Miller & MacKay (1994)
- **谱半径控制**: Sussillo & Abbott (2009) - FORCE learning
- **相变理论**: Sompolinsky et al. (1988)
- **Isotropy**: Amari (1977), Yang & Schoenemann (2022)

---

*最后更新: 2026-02-08*
