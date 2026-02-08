"""
System 01: 极简约束动力系统 - 存在性与持久性研究

核心问题: 什么约束参数组合能让系统从噪声中自组织出持久结构？

系统定义:
    S = (X, Φ, C, T)
    
    X: N维状态空间 (~N(0,1) 初始化)
    Φ: 演化规则 = tanh(W · x + noise)
    C: 归一化约束 = norm_strength * x / ||x||
    T: 长期迭代 (T >> N)

研究目标:
    1. 探索"存在区域": 系统保持非平凡结构
    2. 识别"死亡区域": 退化为均衡或混沌
    3. 产出: 行为相图 (x=sigma, y=norm_strength, z=variance)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import json
import os
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SystemParams:
    """系统参数"""
    N: int = 20          # 状态维度
    T: int = 10000       # 演化步数
    sigma: float = 0.1   # 噪声水平
    norm_strength: float = 1.0  # 归一化强度
    seed: int = 42       # 随机种子
    
    def __post_init__(self):
        self.W_scale = 1.0 / np.sqrt(self.N)


@dataclass
class SystemState:
    """系统状态"""
    state: np.ndarray
    step: int
    variance: float
    norm: float


class ConstrainedDynamicalSystem:
    """
    极简约束动力系统
    
    设计原理:
    - 状态: N维随机向量
    - 演化: tanh(W·x + noise) - 有界非线性
    - 约束: 归一化 - 防止状态爆炸
    - 噪声: 环境扰动
    
    观测指标:
    - variance: 状态多样性 (活=稳定非零, 死=趋零)
    - norm: 状态范数 (检查混沌)
    - entropy: 近似熵
    """
    
    def __init__(self, params: SystemParams):
        self.params = params
        self.rng = np.random.RandomState(params.seed)
        
        # 初始化权重矩阵 (固定, 无先验结构)
        self.W = self.rng.randn(params.N, params.N) / params.W_scale
        
        # 初始化状态 (~N(0,1))
        self.state = self.rng.randn(params.N)
        
        # 历史记录
        self.history: List[SystemState] = []
        self.variance_history: List[float] = []
        self.norm_history: List[float] = []
        self.entropy_history: List[float] = []
        
    def evolve_once(self) -> SystemState:
        """单步演化"""
        # 1. 噪声注入
        noise = self.params.sigma * self.rng.randn(self.params.N)
        
        # 2. 演化规则
        pre_activation = np.dot(self.W, self.state) + noise
        next_state = np.tanh(pre_activation)  # 有界
        
        # 3. 归一化约束
        norm = np.linalg.norm(next_state)
        if norm > 0:
            next_state = self.params.norm_strength * next_state / norm
        
        # 4. 更新状态
        self.state = next_state
        
        # 5. 记录统计
        variance = float(np.var(self.state))
        state_norm = float(np.linalg.norm(self.state))
        entropy = float(-np.sum(self.state * np.log(np.abs(self.state) + 1e-10)))
        
        system_state = SystemState(
            state=self.state.copy(),
            step=len(self.variance_history),
            variance=variance,
            norm=state_norm
        )
        
        return system_state
    
    def evolve(self, record_interval: int = 10) -> Dict:
        """
        长期演化
        
        Args:
            record_interval: 记录间隔
            
        Returns:
            演化统计字典
        """
        print(f"开始演化: N={self.params.N}, T={self.params.T}, "
              f"σ={self.params.sigma}, α={self.params.norm_strength}")
        
        start_time = datetime.now()
        
        for step in range(self.params.T):
            system_state = self.evolve_once()
            
            # 定期记录
            if step % record_interval == 0:
                self.history.append(system_state)
                self.variance_history.append(system_state.variance)
                self.norm_history.append(system_state.norm)
                
                # 进度报告
                if step % 1000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    remaining = elapsed / (step + 1) * (self.params.T - step)
                    
                    print(f"  Step {step:>6}/{self.params.T}: "
                          f"var={system_state.variance:.4f}, "
                          f"norm={system_state.norm:.4f}, "
                          f"ETA={remaining:.0f}s")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 计算稳态指标
        last_10pct = int(self.params.T * 0.1)
        final_variance = np.mean(self.variance_history[-last_10pct:])
        final_norm = np.mean(self.norm_history[-last_10pct:])
        final_entropy = np.mean(self.entropy_history[-last_10pct:]) if self.entropy_history else 0
        
        variance_stability = np.std(self.variance_history[-last_10pct:])
        
        stats = {
            'total_steps': self.params.T,
            'duration_seconds': duration,
            'final_variance': final_variance,
            'final_norm': final_norm,
            'variance_stability': variance_stability,
            'alive': final_variance > 0.01,  # 存活阈值
            'dead': final_variance < 0.001,   # 死亡阈值
            'chaotic': final_norm > 10.0,     # 混沌检测
        }
        
        print(f"\n演化完成: {duration:.1f}秒")
        print(f"  最终variance: {final_variance:.6f} ({'活' if stats['alive'] else '死'})")
        print(f"  最终norm: {final_norm:.4f} ({'混沌' if stats['chaotic'] else '正常'})")
        print(f"  稳定性: {variance_stability:.6f}")
        
        return stats
    
    def analyze_phase(self) -> Dict:
        """
        相分析
        
        判断系统处于哪个相:
        - 存活相: variance稳定非零
        - 死亡相: variance趋零
        - 混沌相: norm爆炸
        """
        last_10pct = int(len(self.variance_history) * 0.1)
        
        final_var = np.mean(self.variance_history[-last_10pct:])
        var_std = np.std(self.variance_history[-last_10pct:])
        final_norm = np.mean(self.norm_history[-last_10pct:])
        
        # 相判断
        if final_norm > 10.0:
            phase = 'chaotic'
            description = '混沌死亡: norm爆炸'
        elif final_var < 0.001:
            phase = 'dead'
            description = '结构死亡: variance趋零'
        elif final_var > 0.01 and var_std < 0.1 * final_var:
            phase = 'alive'
            description = '存活: variance稳定非零'
        else:
            phase = 'transitional'
            description = '过渡态: variance波动'
        
        return {
            'phase': phase,
            'description': description,
            'final_variance': final_var,
            'variance_std': var_std,
            'final_norm': final_norm,
            'survival_ratio': final_var / (final_var + var_std + 1e-10)
        }


class PhaseSpaceExplorer:
    """
    相空间探索器
    
    功能:
    - 参数网格扫描
    - 相图生成
    - 阈值分析
    """
    
    def __init__(self, N: int = 20, T: int = 10000, seeds: int = 3):
        self.N = N
        self.T = T
        self.seeds = seeds
        
    def run_grid(self,
                 sigma_range: Tuple[float, float, float],
                 norm_range: Tuple[float, float, float]) -> Dict:
        """
        运行参数网格扫描
        
        Args:
            sigma_range: (min, max, step)
            norm_range: (min, max, step)
            
        Returns:
            网格结果字典
        """
        sigma_min, sigma_max, sigma_step = sigma_range
        norm_min, norm_max, norm_step = norm_range
        
        sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
        norms = np.arange(norm_min, norm_max + norm_step, norm_step)
        
        n_sigma = len(sigmas)
        n_norm = len(norms)
        
        print(f"运行参数网格: {n_sigma} x {n_norm} = {n_sigma * n_norm} 点")
        print(f"每次实验重复 {self.seeds} 次取平均")
        
        # 结果矩阵
        mean_variance = np.zeros((n_sigma, n_norm))
        survival_rate = np.zeros((n_sigma, n_norm))
        chaos_rate = np.zeros((n_sigma, n_norm))
        
        for i, sigma in enumerate(sigmas):
            for j, norm in enumerate(norms):
                print(f"  扫描: σ={sigma:.2f}, α={norm:.2f}")
                
                variances = []
                survivals = []
                chaos_count = 0
                
                for seed in range(self.seeds):
                    params = SystemParams(
                        N=self.N, T=self.T,
                        sigma=sigma, norm_strength=norm,
                        seed=42 + seed
                    )
                    
                    system = ConstrainedDynamicalSystem(params)
                    stats = system.evolve()
                    
                    variances.append(stats['final_variance'])
                    survivals.append(1 if stats['alive'] else 0)
                    chaos_count += 1 if stats['chaotic'] else 0
                
                mean_variance[i, j] = np.mean(variances)
                survival_rate[i, j] = np.mean(survivals)
                chaos_rate[i, j] = chaos_count / self.seeds
        
        return {
            'sigmas': sigmas,
            'norms': norms,
            'mean_variance': mean_variance,
            'survival_rate': survival_rate,
            'chaos_rate': chaos_rate,
            'params': {'N': self.N, 'T': self.T, 'seeds': self.seeds}
        }
    
    def generate_phase_diagram(self, results: Dict, save_path: str = None):
        """
        生成相图
        
        输出:
        - 热图: mean_variance
        - 叠加: 存活边界线
        """
        sigmas = results['sigmas']
        norms = results['norms']
        variance = results['mean_variance']
        survival = results['survival_rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Variance热图
        ax1 = axes[0]
        im1 = ax1.pcolormesh(sigmas, norms, variance.T, cmap='viridis')
        ax1.set_xlabel('Noise Level (σ)')
        ax1.set_ylabel('Normalization (α)')
        ax1.set_title('Mean Variance')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Survival Rate热图
        ax2 = axes[1]
        im2 = ax2.pcolormesh(sigmas, norms, survival.T, cmap='RdYlGn')
        ax2.set_xlabel('Noise Level (σ)')
        ax2.set_ylabel('Normalization (α)')
        ax2.set_title('Survival Rate')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Chaos Rate热图
        ax3 = axes[2]
        im3 = ax3.pcolormesh(sigmas, norms, results['chaos_rate'].T, cmap='Reds')
        ax3.set_xlabel('Noise Level (σ)')
        ax3.set_ylabel('Normalization (α)')
        ax3.set_title('Chaos Rate')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"相图已保存: {save_path}")
        
        return fig
    
    def find_phase_boundary(self, results: Dict) -> List[Dict]:
        """
        寻找相变边界
        
        Returns:
            边界线列表
        """
        boundaries = []
        survival = results['survival_rate']
        sigmas = results['sigmas']
        norms = results['norms']
        
        # 对每个norm_strength找存活边界
        for j, norm in enumerate(norms):
            for i in range(len(sigmas) - 1):
                if survival[i, j] >= 0.5 and survival[i+1, j] < 0.5:
                    boundaries.append({
                        'type': 'survival_boundary',
                        'norm': norm,
                        'sigma_low': sigmas[i],
                        'sigma_high': sigmas[i+1]
                    })
                    break
        
        return boundaries


def run_single_experiment(N: int = 20, T: int = 10000,
                         sigma: float = 0.1, norm_strength: float = 1.0,
                         seed: int = 42) -> Dict:
    """运行单个实验"""
    print("=" * 60)
    print(f"System 01: 极简约束动力系统")
    print(f"N={N}, T={T}, σ={sigma}, α={norm_strength}")
    print("=" * 60)
    
    params = SystemParams(N=N, T=T, sigma=sigma,
                         norm_strength=norm_strength, seed=seed)
    
    system = ConstrainedDynamicalSystem(params)
    stats = system.evolve()
    phase = system.analyze_phase()
    
    # 绘制轨迹
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(system.variance_history)
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Variance Over Time')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(system.norm_history)
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.title('Norm Over Time')
    
    plt.tight_layout()
    plt.savefig('system_01_trajectory.png', dpi=150)
    print(f"轨迹图已保存: system_01_trajectory.png")
    
    return {
        'params': {'N': N, 'T': T, 'sigma': sigma,
                   'norm_strength': norm_strength, 'seed': seed},
        'stats': stats,
        'phase': phase
    }


def run_phase_diagram(N: int = 20, T: int = 10000,
                      sigma_range: Tuple[float, float, float] = (0.01, 1.0, 0.05),
                      norm_range: Tuple[float, float, float] = (0.5, 2.0, 0.1)) -> Dict:
    """运行相图扫描"""
    print("=" * 60)
    print("System 01: 相空间扫描")
    print("=" * 60)
    
    explorer = PhaseSpaceExplorer(N=N, T=T, seeds=3)
    results = explorer.run_grid(sigma_range, norm_range)
    
    # 生成相图
    explorer.generate_phase_diagram(
        results,
        save_path='system_01_phase_diagram.png'
    )
    
    # 找边界
    boundaries = explorer.find_phase_boundary(results)
    
    print("\n相变边界:")
    for b in boundaries:
        print(f"  α={b['norm']:.2f}: σ < {b['sigma_high']:.2f} 存活")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='System 01: 极简约束动力系统实验')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'phase'],
                       help='运行模式: single=单实验, phase=相图扫描')
    parser.add_argument('--N', type=int, default=20,
                       help='状态维度')
    parser.add_argument('--T', type=int, default=10000,
                       help='演化步数')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='噪声水平')
    parser.add_argument('--norm', type=float, default=1.0,
                       help='归一化强度')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_experiment(
            N=args.N, T=args.T,
            sigma=args.sigma, norm_strength=args.norm,
            seed=args.seed
        )
    else:
        run_phase_diagram(N=args.N, T=args.T)
