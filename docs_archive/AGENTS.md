# AGENTS.md - 系统研究工作空间

## 研究目录

```
F:/system_stability/
├── core/              # 核心系统实现
├── experiments/        # 实验记录
├── analysis/          # 相图与稳定性分析
├── docs/              # 文档
├── SOUL.md            # 身份定义
├── USER.md            # 用户说明
├── PROJECT_MANIFESTO.md # 项目宣言
└── MEMORY.md          # 研究记忆
```

## 研究方法

### 每轮工作

1. **定义最小系统**
   - 状态空间 X
   - 演化规则 Φ
   - 约束 C

2. **运行长期演化**
   - 至少10万步（理想100万步）
   - 记录轨迹

3. **分析稳定性**
   - 绘制相图
   - 检查吸引子类型
   - 记录结构变化

### 禁用词检查

| 禁止 | 替代 |
|------|------|
| 模型 model | 系统 system / 状态空间 state space |
| 学习 learning | 演化 evolution |
| 任务 task | 约束 constraint |
| loss | 时间 time |
| reward | 稳定性 stability |
| 泛化 generalization | 吸引子 attractor |
| 性能 performance | 结构 structure |
| 准确率 accuracy | 相稳定性 phase stability |

## 文件命名

- 系统实现: `core/system_XX.py` (XX是序号)
- 实验记录: `experiments/exp_XX_YYYYMMDD.md`
- 分析报告: `analysis/ana_XX_YYYYMMDD.md`

## 记忆文件

- `memory/YYYY-MM-DD.md` - 每日研究日志
- `MEMORY.md` - 长期研究记忆

## 安全

- 不要运行未经验证的长期演化
- 资源使用不超过50%（保留缓冲）
- 每次实验前保存checkpoint

## 第一项工作

```
1. 创建 core/system_01_minimal.py
   - 64维状态空间
   - 简单线性演化
   - 归一化约束

2. 运行10万步演化

3. 分析相图与稳定性
```

---

_记住：先证明系统能活，再讨论它能做什么。_
