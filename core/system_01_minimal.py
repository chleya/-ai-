"""
System 01: 最小稳定系统

研究问题: 什么条件下，系统可以在长期演化中保持非平凡结构？

系统定义:
    S = (X, Φ, C, T)
    
    X: 64维连续状态空间
    Φ: 归一化约束下的线性演化
    C: L2归一化 + 相位约束
    T: 离散时间步

运行方法:
    python core/system_01_minimal.py --steps 100000
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from datetime import datetime
import json
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SystemState:
    """系统状态"""
    x: np.ndarray  # 64维状态向量
    step: int
    timestamp: str
    
    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.x))
    
    @property
    def energy(self) -> float:
        return float(np.dot(self.x, self.x))
    
    @property
    def entropy_approx(self) -> float:
        """近似熵 (状态分散度)"""
        return float(np.std(self.x))


class MinimalSystem:
    """
    最小稳定系统
    
    设计原则:
    1. 状态空间: 64维连续
    2. 演化规则: 线性变换 + 归一化
    3. 约束: L2范数固定
    """
    
    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
        # 初始状态: 均匀分布在球面上
        self.x = self.rng.randn(dim)
        self.x = self.x / np.linalg.norm(self.x)  # L2归一化 = 约束C
        
        # 演化参数
        self.alpha = 0.99  # 衰减系数
        self.noise_scale = 0.01  # 噪声强度
        
        # 统计
        self.trajectory = []
        self.energy_history = []
        self.norm_history = []
        self.entropy_history = []
        
    def evolve_once(self) -> Dict:
        """单步演化: Φ(x)"""
        # 线性变换 + 噪声
        x_new = self.alpha * self.x + self.noise_scale * self.rng.randn(self.dim)
        
        # 归一化约束: C(x)
        norm = np.linalg.norm(x_new)
        if norm > 0:
            x_new = x_new / norm
        
        self.x = x_new
        
        # 记录统计
        stats = {
            'step': len(self.trajectory),
            'energy': self.energy,
            'norm': self.norm,
            'entropy': self.entropy_approx
        }
        
        return stats
    
    def evolve(self, steps: int, record_interval: int = 100) -> Dict:
        """
        长期演化
        
        Args:
            steps: 演化步数
            record_interval: 记录间隔
        """
        print(f"开始演化: {steps} 步")
        print(f"状态维度: {self.dim}")
        print(f"衰减系数: {self.alpha}")
        print(f"噪声强度: {self.noise_scale}")
        print("-" * 50)
        
        start_time = datetime.now()
        
        for step in range(steps):
            stats = self.evolve_once()
            
            # 定期记录
            if step % record_interval == 0:
                self.trajectory.append(self.x.copy())
                self.energy_history.append(stats['energy'])
                self.norm_history.append(stats['norm'])
                self.entropy_history.append(stats['entropy'])
                
                # 进度报告
                if step % 10000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = step / elapsed if elapsed > 0 else 0
                    remaining = (steps - step) / rate if rate > 0 else 0
                    
                    print(f"步 {step:>6}/{steps}: "
                          f"E={stats['energy']:.4f}, "
                          f"H={stats['entropy']:.4f}, "
                          f"ETA={remaining:.0f}s")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 最终统计
        final_stats = {
            'total_steps': steps,
            'duration_seconds': duration,
            'final_energy': self.energy,
            'final_norm': self.norm,
            'final_entropy': self.entropy_approx,
            'energy_mean': float(np.mean(self.energy_history)),
            'energy_std': float(np.std(self.energy_history)),
            'entropy_mean': float(np.mean(self.entropy_history)),
            'entropy_std': float(np.std(self.entropy_history)),
        }
        
        print("-" * 50)
        print(f"演化完成")
        print(f"耗时: {duration:.1f}秒")
        print(f"最终能量: {final_stats['final_energy']:.6f}")
        print(f"最终熵: {final_stats['final_entropy']:.6f}")
        
        return final_stats
    
    @property
    def energy(self) -> float:
        return float(np.dot(self.x, self.x))
    
    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.x))
    
    @property
    def entropy_approx(self) -> float:
        return float(np.std(self.x))
    
    def analyze_attractor(self) -> Dict:
        """
        吸引子分析
        
        问题: 系统是否进入了某种稳定相？
        
        分析维度:
        1. 能量是否收敛？
        2. 熵是否稳定？
        3. 轨迹是否停留在某个区域？
        """
        analysis = {
            'attractor_type': 'unknown',
            'energy_stability': 'unknown',
            'entropy_stability': 'unknown',
            'convergence_rate': 0.0,
            'evidence': []
        }
        
        # 能量稳定性分析
        if len(self.energy_history) > 100:
            early = self.energy_history[:len(self.energy_history)//4]
            late = self.energy_history[-len(self.energy_history)//4:]
            
            early_mean = np.mean(early)
            late_mean = np.mean(late)
            late_std = np.std(late)
            
            # 检查是否收敛到某个值
            if late_std < 0.01 and abs(late_mean - 1.0) < 0.1:
                analysis['energy_stability'] = 'stable'
                analysis['evidence'].append('能量收敛到1.0附近')
            elif late_std < 0.01:
                analysis['energy_stability'] = 'stable_at_other_value'
                analysis['evidence'].append(f'能量收敛到{late_mean:.4f}')
            else:
                analysis['energy_stability'] = 'unstable'
                analysis['evidence'].append(f'能量波动: std={late_std:.4f}')
        
        # 熵稳定性分析
        if len(self.entropy_history) > 100:
            late_entropy = self.entropy_history[-len(self.entropy_history)//4:]
            entropy_std = np.std(late_entropy)
            
            if entropy_std < 0.001:
                analysis['entropy_stability'] = 'stable'
                analysis['evidence'].append('熵稳定')
            else:
                analysis['entropy_stability'] = 'fluctuating'
                analysis['evidence'].append(f'熵波动: std={entropy_std:.4f}')
        
        # 吸引子类型判断
        if (analysis['energy_stability'] == 'stable' and 
            analysis['entropy_stability'] == 'stable'):
            analysis['attractor_type'] = 'fixed_point'
            analysis['evidence'].append('进入固定点吸引子')
        elif analysis['energy_stability'] == 'stable':
            analysis['attractor_type'] = 'limit_cycle_candidate'
            analysis['evidence.append('可能是极限环']
        else:
            analysis['attractor_type'] = 'chaotic_or_unstable'
            analysis['evidence'].append('未发现稳定吸引子')
        
        return analysis
    
    def save_results(self, stats: Dict, experiment_id: str = "01"):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存统计
        results_dir = os.path.join(PROJECT_ROOT, 'experiments')
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(results_dir, f'exp_{experiment_id}_{timestamp}.json')
        
        results = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'system_params': {
                'dim': self.dim,
                'alpha': self.alpha,
                'noise_scale': self.noise_scale,
                'seed': self.rng.seed
            },
            'evolution_stats': stats,
            'attractor_analysis': self.analyze_attractor()
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {result_file}")
        return result_file


def run_experiment(steps: int = 100000, experiment_id: str = "01"):
    """运行实验"""
    print("=" * 60)
    print(f"System 01 最小稳定系统实验")
    print(f"实验ID: {experiment_id}")
    print("=" * 60)
    
    # 创建系统
    system = MinimalSystem(dim=64, seed=42)
    
    # 运行演化
    stats = system.evolve(steps=steps, record_interval=100)
    
    # 分析吸引子
    analysis = system.analyze_attractor()
    
    print("\n" + "=" * 60)
    print("吸引子分析结果")
    print("=" * 60)
    for key, value in analysis.items():
        if key != 'evidence':
            print(f"  {key}: {value}")
    
    # 保存结果
    result_file = system.save_results(stats, experiment_id)
    
    return system, stats, analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='System 01: 最小稳定系统')
    parser.add_argument('--steps', type=int, default=100000, help='演化步数')
    parser.add_argument('--exp', type=str, default='01', help='实验ID')
    args = parser.parse_args()
    
    system, stats, analysis = run_experiment(steps=args.steps, experiment_id=args.exp)
