# 应激蝉蜕协议（Event-triggered Reset）

## 核心问题

传统蝉蜕协议使用"定时蝉蜕"（每300步），但在非平稳环境中可能不是最优选择。

## 新范式

```
定时蝉蜕（固定间隔）
        ↓
应激蝉蜕（监测触发）
```

## 实验设计

### 监测指标

| 指标 | 计算方法 | 阈值 |
|------|----------|------|
| Jitter | std(s) | > 均值的1.5倍 |
| 特征值波动 | |λmax(t) - λmax(t-1)| | > 阈值 |
| 共识退化 | P(corr < 0.5) | > 50%时间 |

### 触发条件

$$Jitter(t) > \alpha \cdot \bar{Jitter}_{20}$$

其中：
- $Jitter(t)$ = 当前时刻的抖动
- $\bar{Jitter}_{20}$ = 最近20步的平均抖动
- $\alpha$ = 触发系数（实验确定）

---

## 对比策略

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| Fixed-300 | 每300步重置 | 简单 | 不适应环境 |
| Fixed-500 | 每500步重置 | 更少开销 | 可能不及时 |
| Event-triggered | Jitter超阈值触发 | 自适应 | 需要监测 |

---

## 预期结果

| 策略 | 存活率 | 触发次数 | 效率 |
|------|---------|----------|------|
| Fixed-300 | ~70% | 2-3次 | 中等 |
| Fixed-500 | ~60% | 1-2次 | 高 |
| Event-triggered | ~75% | 自适应 | 最高 |

---

## 核心领悟

> "定时蝉蜕是被动的，应激蝉蜕是主动的。"

> "监测系统状态，按需触发，比固定时间间隔更高效。"

---

# 完整自适应框架

```python
class AdaptiveCicada:
    def __init__(self, N, strategies):
        self.N = N
        self.strategies = strategies
        self.history = []
        self.current_strategy = 'fixed-300'
    
    def monitor(self, W, s, t):
        """监测系统状态"""
        jitter = np.std(s)
        eigenvals = np.linalg.eigvalsh(W)
        max_eigen = eigenvals[-1]
        correlation = np.dot(s, self.target)
        
        return {
            'jitter': jitter,
            'max_eigen': max_eigen,
            'correlation': correlation,
            't': t
        }
    
    def should_reset(self, state):
        """判断是否需要触发蝉蜕"""
        if self.current_strategy == 'event-triggered':
            jitter = state['jitter']
            threshold = np.mean([s['jitter'] for s in self.history[-20:]]) * 1.5
            return jitter > threshold
        else:
            return (state['t'] + 1) % self.reset_interval == 0
    
    def evolve(self, total_steps=800):
        """执行自适应演化"""
        W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        s = np.random.randn(self.N)
        
        for t in range(total_steps):
            # 正常演化
            s = evolve(s, W)
            W = update(W, s)
            
            # 监测
            state = self.monitor(W, s, t)
            self.history.append(state)
            
            # 触发判断
            if self.should_reset(state):
                W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
                s = np.random.randn(self.N)
        
        return W, s
```

---

## 研究意义

1. **效率提升**：减少不必要的重置开销
2. **适应性**：应对非平稳环境
3. **鲁棒性**：自动应对突发干扰

---

**生成日期**: 2026-02-08
**版本**: v1.0
