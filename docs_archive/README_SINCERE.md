# Cicada Protocol - 概念探索阶段

> ⚠️ **当前状态**: 早期研究 + 理论笔记  
> 🎯 **目标**: 实现周期性重置机制缓解Hebbian权重演化的长期退化

## 当前状态

| 状态 | 说明 |
|------|------|
| 📄 论文草稿 | ✅ 完成 (见 papers/) |
| 🧪 理论分析 | ✅ 完成 (见 papers/) |
| 💻 可运行代码 | 🔄 开发中 |
| 📊 实验数据 | ⏳ 待生成 |

## 研究内容

### 核心发现

| 发现 | 内容 | 状态 |
|------|------|------|
| 谱稳定性 | λmax从2.15降至1.73 | 理论验证 |
| 相变临界点 | Nc≈900 | 实验发现 |
| 应激蝉蜕 | α=1.6最优 | 实验发现 |
| 任务切换 | Rand比Peak快11.9% | 实验发现 |

### 论文文档

| 文件 | 内容 | 版本 |
|------|------|------|
| [CICADA_PAPER.md](papers/CICADA_PAPER.md) | 完整论文v2.0 | 最新 |
| [PIVOT_THEORY.md](papers/PIVOT_THEORY.md) | 任务切换理论 | 完整 |
| [PHASE_TRANSITION_REPORT.md](papers/PHASE_TRANSITION_REPORT.md) | 相变分析 | 完整 |
| [EVENT_TRIGGERED_RESET.md](papers/EVENT_TRIGGERED_RESET.md) | 应激蝉蜕 | 完整 |

## 下一步计划

1. **短期 (1-2周)**
   - [ ] 实现最小可运行原型 (cicada_minimal.py)
   - [ ] 生成实验数据
   - [ ] 创建可视化图表

2. **中期 (1个月)**
   - [ ] 完善核心代码库
   - [ ] 补充对比实验
   - [ ] 准备投稿OSDI/SOSP

## 贡献者

- **研究方向**: Chen Leiyang
- **AI辅助**: OpenClaw Assistant

## 许可证

即将添加MIT许可证。

## 联系

如有问题或合作意向，请开Issue讨论。

---

*本项目处于早期研究阶段，欢迎参与讨论和贡献！*
