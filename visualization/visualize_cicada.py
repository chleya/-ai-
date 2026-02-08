#!/usr/bin/env python3
"""
Manim Animation: Cicada Protocol Stress Rebirth
展示系统受到攻击时，特征值谱半径如何触碰红线并触发"应激重生"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
N = 100  # 系统规模
frames = 200  # 动画帧数
fps = 15  # 帧率
crit_lambda = 1.8  # 临界谱半径（红线）

# ==================== 模拟数据 ====================
def simulate_system():
    """模拟系统演化过程"""
    np.random.seed(42)
    
    # 初始化
    W = np.random.randn(N, N) / np.sqrt(N)
    eigenvalues = []
    states = []
    resets = []
    attack_phases = []
    
    current_lambda = 1.0
    under_attack = False
    attack_start = 50
    attack_end = 120
    reset_time = None
    
    for t in range(frames):
        # 正常演化
        s = np.random.randn(N)
        s = np.tanh(W @ s + 0.2 * s)
        W = W + 0.001 * np.outer(s, s)
        
        # 谱半径演化
        eigenvals = np.linalg.eigvalsh(W)
        current_lambda = eigenvals[-1]
        
        # 攻击模拟
        if attack_start <= t <= attack_end:
            under_attack = True
            current_lambda *= 1.15  # 攻击导致谱半径膨胀
        else:
            under_attack = False
        
        # 归一化
        if current_lambda > 3.0:
            current_lambda = 3.0
        
        eigenvalues.append(current_lambda)
        states.append(s.copy())
        attack_phases.append(under_attack)
        
        # 检测是否需要重置
        if current_lambda > crit_lambda and reset_time is None:
            reset_time = t
        
        resets.append(reset_time)
    
    return eigenvalues, attack_phases, resets

# ==================== 动画类 ====================
class CicadaProtocolAnimation:
    def __init__(self):
        self.eigenvalues, self.attack_phases, self.resets = simulate_system()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('蝉蜕协议：应激重生动态演示\nCicada Protocol: Stress Rebirth Dynamics', 
                          fontsize=16, fontweight='bold')
        
        # 设置颜色
        self.colors = {
            'normal': '#2ecc71',      # 绿色 - 正常
            'warning': '#f39c12',     # 橙色 - 警告
            'danger': '#e74c3c',      # 红色 - 危险
            'safe': '#3498db',        # 蓝色 - 安全区
            'reset': '#9b59b6',       # 紫色 - 重置
            'attack': '#c0392b'       # 深红 - 攻击
        }
    
    def init_animation(self):
        """初始化动画"""
        # 1. 左上：谱半径演化
        self.ax1 = self.axes[0, 0]
        self.ax1.set_xlim(0, frames)
        self.ax1.set_ylim(0, 3.5)
        self.ax1.axhline(y=crit_lambda, color=self.colors['danger'], 
                         linestyle='--', linewidth=2, label='Critical Threshold')
        self.ax1.fill_between([0, frames], crit_lambda, 3.5, 
                             alpha=0.2, color=self.colors['danger'])
        self.ax1.set_xlabel('Time (steps)')
        self.ax1.set_ylabel('Spectral Radius λmax')
        self.ax1.set_title('Eigenvalue Spectrum Evolution', fontsize=12)
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        
        # 谱半径曲线
        self.line1, = self.ax1.plot([], [], color=self.colors['normal'], 
                                    linewidth=2, label='λmax')
        self.point1, = self.ax1.plot([], [], 'o', color=self.colors['danger'], 
                                     markersize=10)
        
        # 攻击区域标记
        self.ax1.axvspan(50, 120, alpha=0.3, color=self.colors['attack'], 
                        label='Attack Phase')
        
        # 2. 右上：谱分布圆盘
        self.ax2 = self.axes[0, 1]
        self.ax2.set_xlim(-2.5, 2.5)
        self.ax2.set_ylim(-2.5, 2.5)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('Real Part')
        self.ax2.set_ylabel('Imaginary Part')
        self.ax2.set_title('Eigenvalue Distribution', fontsize=12)
        
        # 临界圆
        circle = plt.Circle((0, 0), crit_lambda, fill=False, 
                           color=self.colors['danger'], linestyle='--', linewidth=2)
        self.ax2.add_patch(circle)
        
        # 特征值散点
        self.scatter = self.ax2.scatter([], [], alpha=0.6, s=20)
        
        # 标注
        self.ax2.text(0, 2.7, 'Critical Radius', ha='center', fontsize=10,
                     color=self.colors['danger'])
        
        # 3. 左下：系统状态
        self.ax3 = self.axes[1, 0]
        self.ax3.set_xlim(-3, 3)
        self.ax3.set_ylim(-3, 3)
        self.ax3.set_aspect('equal')
        self.ax3.set_xlabel('State Dimension 1')
        self.ax3.set_ylabel('State Dimension 2')
        self.ax3.set_title('System State Evolution', fontsize=12)
        
        # 状态轨迹
        self.line3, = self.ax3.plot([], [], color=self.colors['normal'], 
                                    linewidth=1, alpha=0.7)
        self.point3, = self.ax3.plot([], [], 'o', color=self.colors['normal'], 
                                      markersize=8)
        
        # 目标方向
        self.ax3.arrow(0, 0, 2, 2, head_width=0.1, head_length=0.1, 
                       fc=self.colors['safe'], ec=self.colors['safe'])
        
        # 4. 右下：重置事件时间线
        self.ax4 = self.axes[1, 1]
        self.ax4.set_xlim(0, frames)
        self.ax4.set_ylim(-0.5, 1.5)
        self.ax4.set_xlabel('Time (steps)')
        self.ax4.set_yticks([])
        self.ax4.set_title('Reset Event Timeline', fontsize=12)
        
        # 时间线
        self.ax4.axhline(y=0.5, color='gray', linewidth=2)
        
        # 重置标记
        self.reset_marker, = self.ax4.plot([], 'v', color=self.colors['reset'], 
                                           markersize=15, label='Reset Event')
        
        # 攻击区域
        self.ax4.axvspan(50, 120, alpha=0.3, color=self.colors['attack'])
        self.ax4.text(85, 1.2, 'ATTACK', ha='center', fontsize=10, 
                     fontweight='bold', color=self.colors['attack'])
        
        # 状态标签
        self.status_text = self.ax4.text(frames/2, -0.3, '', ha='center', 
                                         fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    def animate(self, i):
        """动画帧更新"""
        # 更新谱半径曲线
        self.line1.set_data(range(i+1), self.eigenvalues[:i+1])
        
        # 检测当前状态
        current_lambda = self.eigenvalues[i]
        is_under_attack = self.attack_phases[i]
        reset_time = self.resets[i]
        
        # 颜色变化
        if current_lambda > crit_lambda:
            color = self.colors['danger']
            status = '⚠️ WARNING: λmax Exceeds Critical!'
        elif is_under_attack:
            color = self.colors['warning']
            status = '🔴 Under Attack!'
        else:
            color = self.colors['normal']
            status = '✅ Stable'
        
        self.line1.set_color(color)
        self.point1.set_data([i], [current_lambda])
        
        # 更新谱分布
        if i % 5 == 0:  # 每5帧更新一次
            np.random.seed(i)
            eigenvals = np.random.randn(N) + 1j * np.random.randn(N)
            eigenvals = eigenvals * (current_lambda / 2)
            self.scatter.set_offsets(np.column_stack([eigenvals.real, eigenvals.imag]))
        
        # 更新状态轨迹
        state = self.states[i] if i < len(self.states) else np.random.randn(N)
        if i == 0:
            self.state_history = [state[:2]]
        else:
            self.state_history.append(state[:2])
        
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
        
        history = np.array(self.state_history)
        self.line3.set_data(history[:, 0], history[:, 1])
        self.point3.set_data([history[-1, 0]], [history[-1, 1]])
        
        # 更新重置标记
        if reset_time is not None and i >= reset_time:
            self.ax4.plot([reset_time], [0.5], 'v', color=self.colors['reset'], 
                          markersize=15)
            self.ax4.annotate('RESET', (reset_time, 0.5), 
                             textcoords="offset points", 
                             xytext=(0, 10), ha='center',
                             fontsize=9, color=self.colors['reset'])
        
        # 更新状态文本
        if current_lambda > crit_lambda:
            self.status_text.set_text('🚨 STRESS REBIRTH TRIGGERED!')
            self.status_text.set_color(self.colors['danger'])
        else:
            self.status_text.set_text(status)
            self.status_text.set_color(color)
        
        return self.line1, self.point1, self.scatter, self.line3, self.point3
    
    def save(self, filename='cicada_protocol_demo.mp4'):
        """保存动画"""
        self.init_animation()
        self.states = [np.random.randn(N) for _ in range(frames)]
        self.state_history = []
        
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=frames,
            interval=1000/fps, blit=False
        )
        
        ani.save(filename, writer='ffmpeg', fps=fps)
        print(f"动画已保存: {filename}")
        plt.close()

# ==================== 静态图表 ====================
def create_static_figures():
    """生成静态图表"""
    
    # 图1：相变热力图
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    N_values = [200, 300, 400, 500, 600, 800, 1000]
    peak_rates = [100, 100, 100, 100, 100, 60, 40]
    rand_rates = [80, 80, 100, 100, 100, 100, 20]
    
    x = np.arange(len(N_values))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, peak_rates, width, label='Peak', 
                     color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rand_rates, width, label='Rand', 
                     color='#e74c3c', alpha=0.8)
    
    ax1.axvline(x=4.5, color='green', linestyle='--', linewidth=2, 
               label='Critical Point (Nc≈900)')
    ax1.fill_between([-0.5, 4.5], 0, 110, alpha=0.1, color='blue')
    ax1.fill_between([4.5, 6.5], 0, 110, alpha=0.1, color='red')
    
    ax1.set_xlabel('System Scale (N)')
    ax1.set_ylabel('Survival Rate (%)')
    ax1.set_title('Phase Transition: Survival Rate vs System Scale', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(N_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('phase_transition_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图1已保存: phase_transition_heatmap.png")
    
    # 图2：效率曲线
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
    efficiencies = [108, 313100, 312800, 314900, 314300, 318300, 315200, 310700, 315000, 310100]
    
    # 归一化显示（排除异常值）
    eff_normalized = [e / 1000 for e in efficiencies]
    
    colors = ['#e74c3c' if a < 1.4 else '#f39c12' if a < 1.7 else '#2ecc71' 
              for a in alphas]
    
    bars = ax2.bar(alphas, eff_normalized, color=colors, alpha=0.8, edgecolor='black')
    
    # 标注最佳点
    best_idx = efficiencies.index(max(efficiencies))
    ax2.annotate(f'Optimal\nα=1.6', 
                xy=(alphas[best_idx], eff_normalized[best_idx]),
                xytext=(alphas[best_idx] + 0.3, eff_normalized[best_idx] + 20),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='left')
    
    ax2.set_xlabel('Alpha (Threshold Multiplier)')
    ax2.set_ylabel('Efficiency (×1000)')
    ax2.set_title('Efficiency Heatmap: Optimal Trigger Sensitivity', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加区域标签
    ax2.axvspan(1.0, 1.35, alpha=0.1, color='red')
    ax2.axvspan(1.35, 1.7, alpha=0.1, color='green')
    ax2.axvspan(1.7, 3.2, alpha=0.1, color='blue')
    ax2.text(1.15, max(eff_normalized) * 0.95, 'Sensitive', ha='center', fontsize=9)
    ax2.text(1.5, max(eff_normalized) * 0.95, 'OPTIMAL', ha='center', fontsize=9, 
             fontweight='bold', color='green')
    ax2.text(2.2, max(eff_normalized) * 0.95, 'Conservative', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('efficiency_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图2已保存: efficiency_heatmap.png")
    
    # 图3：四合一综合图
    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 3.1 谱半径演化
    t = np.arange(frames)
    lambda_normal = 1.0 + 0.005 * t
    lambda_with_attack = lambda_normal.copy()
    lambda_with_attack[50:120] = lambda_with_attack[50:120] * 1.4
    
    ax1.plot(t, lambda_normal, 'g-', linewidth=2, label='Normal')
    ax1.plot(t, lambda_with_attack, 'r-', linewidth=2, label='Under Attack')
    ax1.axhline(y=crit_lambda, color='red', linestyle='--', linewidth=2)
    ax1.fill_between(t, crit_lambda, 3.5, alpha=0.2, color='red')
    ax1.axvspan(50, 120, alpha=0.2, color='orange')
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('λmax')
    ax1.set_title('Spectral Radius Evolution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3.2 特征值分布
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(crit_lambda * np.cos(theta), crit_lambda * np.sin(theta), 
             'r--', linewidth=2, label='Critical Circle')
    
    # 正常分布
    for i in range(5):
        np.random.seed(i)
        eigenvals = np.random.randn(N) * 0.5
        ax2.scatter(eigenvals, np.random.randn(N) * 0.5, alpha=0.5, s=20)
    
    # 攻击分布
    np.random.seed(99)
    eigenvals_attack = np.random.randn(N) * 1.2
    ax2.scatter(eigenvals_attack, np.random.randn(N) * 1.2, 
                color='red', alpha=0.5, s=20, label='Attack')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imaginary')
    ax2.set_title('Eigenvalue Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3.3 相变曲线
    N_values = [200, 400, 600, 800, 1000]
    peak = [100, 100, 100, 60, 40]
    rand = [80, 100, 100, 100, 20]
    
    ax3.plot(N_values, peak, 'b-o', linewidth=2, markersize=8, label='Peak')
    ax3.plot(N_values, rand, 'r-s', linewidth=2, markersize=8, label='Rand')
    ax3.axvline(x=900, color='green', linestyle='--', linewidth=2)
    ax3.fill_between([800, 1000], 0, 110, alpha=0.1, color='red')
    ax3.set_xlabel('System Scale (N)')
    ax3.set_ylabel('Survival Rate (%)')
    ax3.set_title('Phase Transition Curve', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 110)
    
    # 3.4 重置策略对比
    strategies = ['Fixed-300', 'Fixed-500', 'Event-1.5', 'Event-1.6', 'Event-1.8']
    survival = [400, 380, 415, 408, 402]
    
    colors = ['#3498db', '#3498db', '#9b59b6', '#9b59b6', '#9b59b6']
    bars = ax4.bar(strategies, survival, color=colors, alpha=0.8, edgecolor='black')
    
    # 标注最佳
    ax4.annotate('Optimal', xy=(2, 415), xytext=(2.5, 430),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='left')
    
    ax4.set_ylabel('Survival Rate')
    ax4.set_title('Reset Strategy Comparison', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图3已保存: comprehensive_dashboard.png")

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("蝉蜕协议可视化生成器")
    print("Cicada Protocol Visualization Generator")
    print("="*60)
    
    print("\n1. 生成静态图表...")
    create_static_figures()
    
    print("\n2. 生成动画（需要ffmpeg）...")
    try:
        ani = CicadaProtocolAnimation()
        ani.save('cicada_protocol_demo.mp4')
    except Exception as e:
        print(f"动画生成跳过（需要ffmpeg）: {e}")
    
    print("\n" + "="*60)
    print("生成完成！")
    print("="*60)
