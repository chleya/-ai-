#!/usr/bin/env python3
"""
文档清理脚本
将冗余的.md文件归档，精简项目结构
"""

import os
import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent

# 要保留的核心文档（保持不变）
KEEP_FILES = {
    # 工程文件
    'README_HONEST.md',  # 使用诚实版README
    'requirements.txt',
    'LICENSE',
    '.gitignore',
    
    # 核心代码
    'cicada_minimal.py',
    'cicada/core.py',
    'cicada/experiments.py',
    
    # 可视化
    'visualization/visualize_cicada.py',
}

# 要保留的papers目录文件
KEEP_PAPERS = {
    'papers/CICADA_PAPER.md',
    'papers/PIVOT_THEORY.md',
    'papers/PHASE_TRANSITION_REPORT.md',
    'papers/EVENT_TRIGGERED_RESET.md',
    'papers/SUBMISSION_PLAN.md',
}

# 要归档的重复文件名模式
ARCHIVE_PATTERNS = [
    'FINAL_',      # FINAL_xxx.md
    'COMPLETE_',   # COMPLETE_xxx.md
    'GOLDEN_',     # GOLDEN_xxx.md
    '_FINAL',      # xxx_FINAL.md
    '_COMPLETE',   # xxx_COMPLETE.md
    'DEEP_',       # DEEP_xxx.md
    'VERIFICATION_', # VERIFICATION_xxx.md
    'DRAFT_',      # DRAFT_xxx.md
]

# 已知的有效papers文件（保留）
VALID_PAPERS = {
    'CICADA_PAPER.md',
    'PIVOT_THEORY.md', 
    'PHASE_TRANSITION_REPORT.md',
    'EVENT_TRIGGERED_RESET.md',
    'SUBMISSION_PLAN.md',
    'THEORY.md',
    'RESULTS.md',
    'ROADMAP.md',
}


def should_archive(filename: str) -> bool:
    """判断文件是否应该归档"""
    # 保留目录
    if filename in ['papers', 'cicada', 'visualization', 'core', 'experiments', 
                    'memory', 'multi_layer', 'results', 'sparse_hebbian']:
        return False
    
    # 保留核心文件
    if filename in KEEP_FILES:
        return False
    
    # 保留有效的papers文件
    if filename.startswith('papers/'):
        basename = filename.replace('papers/', '')
        if basename in VALID_PAPERS or basename.endswith('.py'):
            return False
    
    # 检查归档模式
    for pattern in ARCHIVE_PATTERNS:
        if pattern in filename:
            return True
    
    return False


def cleanup():
    """执行清理"""
    print("=" * 60)
    print("文档清理脚本")
    print("=" * 60)
    
    # 创建归档目录
    archive_dir = ROOT / 'archive'
    archive_dir.mkdir(exist_ok=True)
    print(f"归档目录: {archive_dir}")
    
    # 统计
    archived_count = 0
    kept_count = 0
    
    # 遍历根目录下的所有文件
    for item in ROOT.iterdir():
        if item.is_file():
            if should_archive(item.name):
                # 移动到archive
                new_path = archive_dir / item.name
                if new_path.exists():
                    # 如果已存在，添加序号
                    counter = 1
                    while new_path.exists():
                        new_path = archive_dir / f"{item.stem}_{counter}{item.suffix}"
                        counter += 1
                shutil.move(str(item), str(new_path))
                print(f"  归档: {item.name}")
                archived_count += 1
            else:
                print(f"  保留: {item.name}")
                kept_count += 1
    
    print()
    print(f"统计:")
    print(f"  已归档: {archived_count} 个文件")
    print(f"  已保留: {kept_count} 个文件")
    print()
    print("清理完成！")
    print()
    print("建议下一步:")
    print("  1. 检查 archive/ 目录确认无误")
    print("  2. 运行: git add -A && git commit -m 'Cleanup: archive redundant docs'")
    print("  3. 运行: git push")


def generate_readme_structure():
    """生成推荐的目录结构说明"""
    structure = """
## 推荐的项目结构

```
├── cicada/
│   ├── __init__.py
│   ├── core.py              # ✅ 核心协议
│   └── experiments.py       # ⚠️ 部分实现
├── visualization/
│   └── visualize_cicada.py  # ✅ 可视化工具
├── papers/
│   ├── CICADA_PAPER.md      # ✅ 核心论文
│   ├── THEORY.md            # ⏳ 理论框架（待整理）
│   └── RESULTS.md           # ⏳ 实验结果（待整理）
├── archive/                  # ⬅️ 新增：归档目录
│   └── *.md                  # 冗余文档已移至此
├── cicada_minimal.py        # ✅ 最小原型
├── requirements.txt         # ✅ 依赖
├── LICENSE                  # ✅ 许可证
└── README_HONEST.md         # ✅ 诚实版README
```

## 文件说明

| 状态 | 标记 | 含义 |
|------|------|------|
| ✅ | 完整可用 | 功能完整，可直接使用 |
| ⚠️ | 部分实现 | 有内容但待完善 |
| ⏳ | 待整理 | 有内容但需整理 |

"""
    print(structure)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        # 仅显示预览
        print("预览模式（不实际执行）:")
        print()
        for item in ROOT.iterdir():
            if item.is_file():
                status = "归档" if should_archive(item.name) else "保留"
                print(f"  [{status}] {item.name}")
    else:
        cleanup()
        generate_readme_structure()
