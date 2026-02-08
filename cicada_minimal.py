#!/usr/bin/env python3
"""
Cicada Protocol - Minimal Working Prototype
==========================================
展示周期性重置如何抑制Hebbian权重演化中的谱半径爆炸

核心机制:
- 权重矩阵W通过Hebbian学习演化: W += η·s·s^T
- 定期重置W以防止谱半径过度增长
- 演示"蝉蜕"如何维持长期稳定性

Author: Chen Leiyang
Date: 2026-02-08
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    strategy: str
    lambda_max_history: List[float]
    survival_rate: float
    reset_count: int
    final_lambda: float


class SimpleCicada:
    """
    简化版蝉蜕协议实现
    
    核心参数:
        N: 系统规模（节点数）
        reset_interval: 重置间隔（步数）
        learning_rate: 学习率
        noise_level: 噪声强度
    """
    
    def __init__(self, N: int = 300, reset_interval: int = 200,
                 learning_rate: float = 0.01, noise_level: float = 0.5):
        self.N = N
        self.reset_interval = reset_interval
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        
        # 初始化权重矩阵（Xavier初始化）
        self.W = np.random.randn(N, N) * np.sqrt(2.0 / N)
        
        # 历史记录
        self.lambda_max_history: List[float] = []
        self.state_history: List[np.ndarray] = []
        
    def step(self) -> float:
        """
        执行一步演化
        
        Returns:
            当前谱半径
        """
        # 生成输入（带噪声）
        s = np.random.randn(self.N)
        s = s / np.linalg.norm(s)  # 归一化
        
        # Hebbian权重更新
        self.W += self.learning_rate * np.outer(s, s)
        
        # 谱半径计算
        eigenvals = np.linalg.eigvalsh(self.W)
        lambda_max = np.max(np.abs(eigenvals))
        
        # 记录
        self.lambda_max_history.append(lambda_max)
        self.state_history.append(s.copy())
        
        return lambda_max
    
    def reset(self):
        """蝉蜕：重置权重矩阵"""
        self.W = np.random.randn(self.N, self.N) * np.sqrt(2.0 / self.N)
    
    def run(self, steps: int, strategy: str = 'fixed') -> ExperimentResult:
        """
        运行实验
        
        Args:
            steps: 总步数
            strategy: 重置策略 ('fixed', 'event', 'none')
        
        Returns:
            实验结果
        """
        reset_count = 0
        jitters: List[float] = []
        
        for t in range(steps):
            # 执行一步
            lambda_max = self.step()
            
            # 计算抖动（用于事件触发策略）
            if len(self.state_history) > 10:
                jitter = np.std(self.state_history[-10:])
                jitters.append(jitter)
            
            # 重置决策
            if strategy == 'fixed':
                # 固定周期重置
                if (t + 1) % self.reset_interval == 0:
                    self.reset()
                    reset_count += 1
                    
            elif strategy == 'event':
                # 事件触发重置（基于抖动）
                if len(jitters) > 20:
                    current_jitter = jitters[-1]
                    mean_jitter = np.mean(jitters[-20:])
                    if current_jitter > 1.5 * mean_jitter:
                        self.reset()
                        reset_count += 1
                        jitters = []
        
        # 计算存活率（谱半径<1.8的比例）
        healthy_threshold = 1.8
        healthy_steps = sum(1 for l in self.lambda_max_history if l < healthy_threshold)
        survival_rate = healthy_steps / steps
        
        return ExperimentResult(
            strategy=strategy,
            lambda_max_history=self.lambda_max_history,
            survival_rate=survival_rate,
            reset_count=reset_count,
            final_lambda=self.lambda_max_history[-1]
        )


def run_comparison实验():
    """
    运行策略对比实验
    
    对比三种策略:
    1. 'none': 永不复位（基线）
    2. 'fixed': 固定周期重置
    3. 'event': 事件触发重置
    """
    print("=" * 60)
    print("蝉蜕协议实验：策略对比")
    print("=" * 60)
    
    # 实验配置
    N = 300
    steps = 800
    reset_interval = 300
    
    strategies = ['none', 'fixed', 'event']
    results = {}
    
    for strategy in strategies:
        print(f"\n运行策略: {strategy}...")
        
        # 多次实验取平均
        trials = 3
        all_results = []
        
        for trial in range(trials):
            np.random.seed(trial * 42)
            cicada = SimpleCicada(N=N, reset_interval=reset_interval)
            result = cicada.run(steps, strategy)
            all_results.append(result)
        
        # 聚合结果
        results[strategy] = {
            'survival_rate': np.mean([r.survival_rate for r in all_results]),
            'reset_count': np.mean([r.reset_count for r in all_results]),
            'final_lambda': np.mean([r.final_lambda for r in all_results]),
            'lambda_history': all_results[0].lambda_max_history
        }
        
        print(f"  存活率: {results[strategy]['survival_rate']:.1%}")
        print(f"  重置次数: {results[strategy]['reset_count']:.1f}")
        print(f"  最终谱半径: {results[strategy]['final_lambda']:.3f}")
    
    # 打印对比表格
    print("\n" + "=" * 60)
    print("实验结果对比")
    print("=" * 60)
    print(f"{'策略':<10} {'存活率':<12} {'重置次数':<10} {'最终λmax':<10}")
    print("-" * 60)
    
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<10} {r['survival_rate']:<12.1%} {r['reset_count']:<10.1f} {r['final_lambda']:<10.3f}")
    
    return results


def plot_results(results: dict, save_path: str = None):
    """
    可视化实验结果
    
    Args:
        results: 实验结果字典
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：谱半径演化
    ax1 = axes[0]
    colors = {'none': 'red', 'fixed': 'blue', 'event': 'green'}
    labels = {'none': 'No Reset', 'fixed': 'Fixed (300步)', 'event': 'Event-triggered'}
    
    for strategy in ['none', 'fixed', 'event']:
        history = results[strategy]['lambda_history']
        ax1.plot(history, color=colors[strategy], label=labels[strategy], linewidth=1.5)
    
    ax1.axhline(y=1.8, color='orange', linestyle='--', linewidth=2, label='Critical (1.8)')
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Spectral Radius (λmax)', fontsize=12)
    ax1.set_title('Spectral Radius Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)
    
    # 图2：策略对比柱状图
    ax2 = axes[1]
    strategies = ['none', 'fixed', 'event']
    survival_rates = [results[s]['survival_rate'] for s in strategies]
    colors_bar = [colors[s] for s in strategies]
    
    bars = ax2.bar(strategies, survival_rates, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Survival Rate', fontsize=12)
    ax2.set_title('Survival Rate Comparison', fontsize=14)
    ax2.set_ylim(0, 1.1)
    
    # 添加数值标签
    for bar, rate in zip(bars, survival_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    # 运行实验
    results = run_comparison实验()
    
    # 可视化
    plot_results(results, 'cicada_experiment_results.png')
    
    # 保存结果
    with open('experiment_results.json', 'w') as f:
        json.dump({
            s: {
                'survival_rate': r['survival_rate'],
                'reset_count': r['reset_count'],
                'final_lambda': r['final_lambda']
            }
            for s, r in results.items()
        }, f, indent=2)
    
    print("\n实验完成！")
    print("结果已保存至:")
    print("  - cicada_experiment_results.png")
    print("  - experiment_results.json")


if __name__ == "__main__":
    main()
