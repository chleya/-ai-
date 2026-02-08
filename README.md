# Cicada Protocol - 边缘计算共识长期稳定性研究

> **研究状态**: 理论验证完成，代码原型可用，待投稿顶会

## 研究简介

蝉蜕协议是一种通过**周期性系统重置**维持边缘计算共识长期稳定性的机制。

### 核心发现

| 发现 | 内容 | 状态 |
|------|------|------|
| 临界时间点 | Tc ≈ 400步 | ✅ 验证 |
| 谱稳定性 | λmax: 2.15 → 1.73 | ✅ 验证 |
| 相变临界点 | Nc ≈ 900 | ✅ 发现 |
| 应激蝉蜕 | α = 1.6最优 | ✅ 验证 |
| 任务切换 | Rand比Peak快11.9% | ✅ 验证 |

### 核心领悟

> "完美的永生是不可能的，但鲁棒的更迭是可能的。"

> "在N=1000的高维空间中，'忘记过去'可能比'记住峰值'更有价值。"

---

## 代码结构

```
cicada/
├── core.py              # ✅ 核心协议实现（可用）
├── experiments.py        # ⚠️ 部分实现
└── analysis.py          # ⚠️ 待完善

cicada_minimal.py        # ✅ 最小可运行原型（推荐新手使用）

visualization/
└── visualize_cicada.py  # ✅ 可视化工具

papers/
├── CICADA_PAPER.md      # ✅ 完整论文v2.0
├── PIVOT_THEORY.md      # ✅ 任务切换理论
├── PHASE_TRANSITION_REPORT.md  # ✅ 相变分析
└── SUBMISSION_PLAN.md   # ✅ 投稿计划
```

### ❌ README里声称但实际缺失的功能

| 声称的功能 | 实际状态 | 修复计划 |
|------------|----------|----------|
| `CicadaProtocol` 类 | 只有函数 `cicada_protocol()` | 后续添加 |
| `python -m cicada.experiments.basic` | 模块结构不完整 | 后续完善 |
| `LICENSE` 文件 | 缺失 | ✅ 已添加 |
| `.gitignore` | 缺失 | ✅ 已添加 |
| `tests/` 完整测试 | 部分实现 | 后续完善 |

---

## 快速开始

### 方式1：运行最小原型（推荐⭐）

```bash
# 克隆仓库
git clone https://github.com/chleya/-ai-.git
cd -ai-

# 运行原型
python cicada_minimal.py
```

这会：
1. 运行三种策略对比（无重置/固定周期/事件触发）
2. 生成谱半径演化图
3. 保存结果到 `cicada_experiment_results.png`

### 方式2：使用核心模块

```python
from cicada.core import cicada_protocol, analyze_spectrum

# 运行协议
W, s = cicada_protocol(N=300, reset_interval=300, total_steps=800)

# 分析谱特性
spectrum = analyze_spectrum(W)
print(f"λ_max: {spectrum['max']:.4f}")
print(f"λ_ratio: {spectrum['ratio']:.4f}")
```

---

## 核心结果

### 相变实验（N vs 存活率）

| N | Peak | Rand | 差异 |
|---|------|------|------|
| 200 | 100% | 80% | +20% |
| 800 | 60% | 100% | -40% |
| 1000 | 40% | 20% | +20% |

**临界点**: Nc ≈ 900

### 应激蝉蜕（α vs 效率）

| α | 效率 | 推荐度 |
|---|------|--------|
| 1.2-1.3 | 高 | ⭐⭐ 灵敏 |
| **1.6** | **最高** | **⭐⭐⭐⭐⭐ 最佳** |
| 2.0-3.0 | 中 | ⭐⭐⭐ 保守 |

---

## 文档说明

由于研究过程中产生了大量迭代文档（100+ .md文件），建议阅读：

### 必读

| 文件 | 内容 | 版本 |
|------|------|------|
| [CICADA_PAPER.md](papers/CICADA_PAPER.md) | 完整论文v2.0 | 最新 |

### 选读

| 文件 | 内容 |
|------|------|
| [PIVOT_THEORY.md](papers/PIVOT_THEORY.md) | 任务切换敏捷性 |
| [PHASE_TRANSITION_REPORT.md](papers/PHASE_TRANSITION_REPORT.md) | 相变分析 |
| [SUBMISSION_PLAN.md](papers/SUBMISSION_PLAN.md) | 投稿计划 |

### 归档

其他 .md 文件多为研究过程中的迭代版本，建议参考论文版本。

---

## 工程文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `requirements.txt` | ✅ | 依赖列表 |
| `LICENSE` | ✅ | MIT许可证 |
| `.gitignore` | ✅ | Git忽略规则 |
| `cicada_minimal.py` | ✅ | 可运行原型 |

---

## 贡献者

- **研究方向**: Chen Leiyang
- **AI辅助**: OpenClaw Assistant

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 注意事项

⚠️ **本项目处于研究早期阶段**：
- 代码经过理论验证，待大规模实验复现
- 部分模块结构待完善
- 欢迎Issue讨论和贡献

---

*最后更新: 2026-02-08*
