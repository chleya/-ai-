# 文档精简计划

## 当前问题
- 90+ .md文件严重冗余
- 大量重复内容
- 读者无法识别权威版本

## 目标结构

```
papers/
├── README.md              ← 项目总览（当前已删除冗余描述）
├── CICADA_PAPER.md        ← 核心论文（唯一权威版本）
├── THEORY.md              ← 理论框架（公式推导）
├── RESULTS.md             ← 实验结果（数据+图表）
├── SUBMISSION.md          ← 投稿计划
└── ROADMAP.md             ← 下一步计划

docs/
├── INSTALL.md             ← 安装指南
├── USAGE.md               ← 使用说明
└── API.md                 ← API文档

scripts/
├── run_experiment.py       ← 实验脚本
└── visualize.py           ← 可视化脚本

tests/
└── test_*.py              ← 单元测试
```

## 文件分类建议

### 保留（5个核心文档）

| 文件 | 优先级 | 理由 |
|------|--------|------|
| README.md | P0 | 项目入口 |
| CICADA_PAPER.md | P1 | 核心论文 |
| THEORY.md | P2 | 理论推导 |
| RESULTS.md | P3 | 实验数据 |
| SUBMISSION.md | P4 | 投稿计划 |

### 可归档（移到archive/）

| 原文件名 | 内容摘要 |
|----------|----------|
| FINAL_REPORT.md | 旧版本报告 |
| COMPLETE_REPORT.md | 旧版本报告 |
| DEEP_*.md | 过程记录 |
| *_FINAL.md | 过程记录 |

### 删除（过时草稿）

| 文件类型 | 示例 |
|----------|------|
| 临时笔记 | NOTE_*.md |
| 草稿版本 | DRAFT_*.md |
| 重复内容 | *COPY*.md |

## 执行命令（Git操作）

```bash
# 1. 创建目录结构
mkdir -p papers/ docs/ scripts/ tests/ archive/

# 2. 移动核心文档
cp CICADA_PAPER.md papers/
cp PIVOT_THEORY.md papers/
cp PHASE_TRANSITION_REPORT.md papers/
cp EVENT_TRIGGERED_RESET.md papers/
cp SUBMISSION_PLAN.md papers/

# 3. 归档非核心文档
mkdir -p archive/
git mv FINAL_*.md archive/ 2>/dev/null || mv FINAL_*.md archive/
git mv *FINAL*.md archive/ 2>/dev/null || mv *FINAL*.md archive/
git mv DEEP_*.md archive/ 2>/dev/null || mv DEEP_*.md archive/

# 4. 删除冗余
rm -f *_COPY*.md
rm -f DRAFT_*.md
rm -f NOTE_*.md
```

## 验证清单

- [ ] README正确描述项目状态
- [ ] CICADA_PAPER.md是唯一论文版本
- [ ] 没有重复内容文件
- [ ] 新贡献者能快速理解项目结构
