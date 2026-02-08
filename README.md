# Cicada Protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Cicada Protocol** 是一个通过周期性系统重置维持分布式共识长期稳定性的Python库。

## 快速开始

```bash
# 安装
pip install -e .

# 运行演示
python -m cicada --demo
```

## 使用

```python
from cicada import CicadaProtocol

protocol = CicadaProtocol(N=200, reset_interval=300)
W, s = protocol.evolve(steps=800)
stats = protocol.analyze()
print(f"存活率: {stats['survival_rate']:.1%}")
```

## 文档

- [快速开始](QUICKSTART.md)
- [示例](examples/)
- [论文](papers/CICADA_PAPER.md)

## 结构

```
cicada/
├── __init__.py      # 主模块
├── __main__.py      # 命令行入口
├── core.py          # 核心协议（暂存）
├── protocols/       # 策略实现
└── README.md

examples/            # 示例代码
papers/              # 论文文档
docs/                # 详细文档
```

## 版本

当前版本: 0.1.0 (Alpha)

## 许可证

MIT License
