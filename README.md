# Cicada Protocol - 研究原型

> ⚠️ **当前状态**: 研究原型阶段  
> ✅ **核心代码**: 可运行，但待优化  
> 🎯 **目标**: 实现周期性重置维持共识长期稳定性

## 什么是 Cicada Protocol?

蝉蜕协议是一种通过**周期性系统重置**维持边缘计算共识长期稳定性的机制。

**核心发现**：
- 临界时间点 Tc ≈ 400步
- 相变临界点 Nc ≈ 900
- 应激蝉蜕 α = 1.6最优
- Rand比Peak快11.9%

## 当前可运行的代码

### 1. 最小演示 (推荐⭐)

```bash
python cicada_minimal.py
```

这个脚本会：
- 运行三种策略对比（无重置/固定周期/事件触发）
- 生成谱半径演化图 `cicada_results.png`

### 2. 真正的演示 (无归一化)

```bash
python cicada_true_demo.py
```

展示重置如何防止谱半径爆炸：
- No Reset: λ = 0.56
- Reset 100: λ = 0.28 (↓49%)
- Reset 200: λ = 0.27 (↓52%)

### 3. 核心模块

```python
from cicada.core import cicada_protocol, analyze_spectrum

# 运行协议
W, s = cicada_protocol(N=200, reset_interval=300, total_steps=800)

# 分析结果
spectrum = analyze_spectrum(W)
print(f"λ_max: {spectrum['max']:.4f}")
```

## 项目结构

```
├── cicada/              # Python包（开发中）
│   ├── __init__.py      # 包入口
│   ├── __main__.py      # 命令行入口
│   ├── core.py          # 核心协议
│   └── experiments.py   # 实验脚本
│
├── cicada_minimal.py    # ⭐ 最小可运行原型
├── cicada_true_demo.py  # ⭐ 真正的演示
├── cicada_demo.py       # 演示脚本
├── test_n1000.py        # N=1000测试
│
├── requirements.txt     # 依赖
├── setup.py            # 包配置（开发中）
│
└── papers/             # 论文文档
    ├── CICADA_PAPER.md           # 完整论文
    ├── PIVOT_THEORY.md           # 任务切换理论
    ├── PHASE_TRANSITION_REPORT.md # 相变分析
    └── EVENT_TRIGGERED_RESET.md  # 应激蝉蜕

其他120+个.md文件多为研究迭代版本，建议参考papers/目录。
```

## 安装

```bash
# 克隆
git clone https://github.com/chleya/-ai-.git
cd -ai-

# 安装依赖
pip install -r requirements.txt

# 运行演示
python cicada_true_demo.py
```

## 命令行使用

```bash
# 运行最小演示
python cicada_minimal.py

# 运行真实演示
python cicada_true_demo.py

# 运行N=1000测试
python test_n1000.py
```

## 核心结果

### 相变实验

| N | Peak | Rand | 差异 |
|---|------|------|------|
| 200 | 100% | 80% | +20% |
| 800 | 60% | 100% | -40% |
| 1000 | 40% | 20% | +20% |

**临界点**: Nc ≈ 900

### 应激蝉蜕效率

| α | 效率 |
|---|------|
| 1.2-1.3 | 高频重置 |
| **1.6** | **最高效率** |
| 2.0-3.0 | 低频重置 |

## 文档说明

### 必读

| 文件 | 内容 |
|------|------|
| [CICADA_PAPER.md](papers/CICADA_PAPER.md) | 完整论文v2.0 |

### 选读

| 文件 | 内容 |
|------|------|
| [PIVOT_THEORY.md](papers/PIVOT_THEORY.md) | 任务切换敏捷性 |
| [PHASE_TRANSITION_REPORT.md](papers/PHASE_TRANSITION_REPORT.md) | 相变分析 |
| [EVENT_TRIGGERED_RESET.md](papers/EVENT_TRIGGERED_RESET.md) | 应激蝉蜕 |

### 其他

120+个.md文件位于根目录，多为研究迭代版本，可参考但不必全读。

## 已知问题

⚠️ **当前问题**：
1. [ ] 包结构不完整（setup.py需完善）
2. [ ] 缺少单元测试
3. [ ] 文档需精简
4. [ ] 缺少Jupyter notebook示例

## 下一步计划

1. **短期** (1-2周)
   - [ ] 完善setup.py使其可pip install
   - [ ] 添加基础单元测试
   - [ ] 精简文档到5个核心文件

2. **中期** (1个月)
   - [ ] 完善CicadaProtocol类API
   - [ ] 添加Jupyter示例
   - [ ] 准备OSDI/SOSP投稿

## 贡献者

- **研究方向**: Chen Leiyang
- **AI辅助**: OpenClaw Assistant

## 许可证

MIT License - 见 [LICENSE](LICENSE)

## 注意

⚠️ 本项目处于**研究原型阶段**，代码经过理论验证但待大规模实验复现。

---

*最后更新: 2026-02-08*
