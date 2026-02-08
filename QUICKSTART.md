# Cicada Protocol - Quick Start Guide

**5分钟上手 Cicada 协议**

## 安装

```bash
# 克隆仓库
git clone https://github.com/chleya/-ai-.git
cd -ai-

# 安装依赖
pip install -r requirements.txt

# 可选：安装为可编辑包
pip install -e .
```

## 快速开始

### 方式1：命令行

```bash
# 运行演示
python -m cicada --demo

# 比较策略
python -m cicada --compare --N 200 --steps 500

# 自定义实验
python -m cicada --N 300 --steps 800 --interval 200
```

### 方式2：Python API

```python
from cicada import CicadaProtocol

# 创建协议实例
protocol = CicadaProtocol(
    N=200,              # 系统规模
    reset_interval=300, # 重置间隔
    seed=42             # 随机种子（可重复）
)

# 运行演化
W, s = protocol.evolve(steps=800)

# 分析结果
stats = protocol.analyze()
print(f"存活率: {stats['survival_rate']:.1%}")
print(f"最终谱半径: {stats['final_lambda']:.4f}")
print(f"重置次数: {stats['reset_count']}")
```

### 方式3：策略对比

```python
from cicada import compare_strategies

# 比较不同策略
results = compare_strategies(
    N=200,         # 系统规模
    steps=500,     # 演化步数
    trials=3       # 重复次数
)

for name, result in results.items():
    print(f"{name}: 存活率={result.survival_rate:.1%}")
```

## 核心概念

### 谱半径 (Spectral Radius)

谱半径 $\lambda_{max}$ 衡量权重矩阵的稳定性：
- $\lambda_{max} < 1.8$: 系统稳定
- $\lambda_{max} > 1.8$: 系统不稳定

### 重置策略

| 策略 | 说明 | 参数 |
|------|------|------|
| FixedInterval | 固定周期重置 | `reset_interval` |
| EventTriggered | 事件触发重置 | `alpha`, `window` |
| Adaptive | 自适应重置 | `threshold` |

## 示例代码

参见 `examples/` 目录：

- `01_basic_demo.py` - 基础演示
- `02_strategy_comparison.py` - 策略对比
- `03_phase_transition.py` - 相变实验

## 下一步

- 阅读 [完整文档](docs/)
- 查看 [API参考](docs/API.md)
- 运行示例：`python -m cicada --demo`
