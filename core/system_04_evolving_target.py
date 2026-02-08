"""
System 04: Evolving Target Variable - Self-Defining Stability

A cleaner implementation testing whether systems can evolve their own targets.

Research Questions:
1. Can target_var drift away from initial value?
2. Can the system discover better operating points than human-set values?
3. Does hierarchical regulation emerge (fast + slow timescales)?

Three Modes:
1. 'drift': Simple random drift
2. 'exploration': Drift + exploration (target chases achievable variance)
3. 'adaptive': Target follows long-term variance level

Usage:
    python core/system_04_evolving_target.py --mode drift --sigma 0.5
    python core/system_04_evolving_target.py --mode exploration --sigma 0.5
    python core/system_04_evolving_target.py --mode compare --sigma 0.5
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemBase:
    """Fixed target baseline."""
    
    N = 20
    T = 20000
    
    def __init__(self, sigma=0.5, target_var=0.015, seed=42):
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.alpha = 0.45
        self.target_var = target_var
        self.history = {'variance': [], 'alpha': []}
        
    def step(self, t):
        """Single step."""
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.alpha
        
        var = np.var(self.state)
        self.history['variance'].append(var)
        self.history['alpha'].append(self.alpha)
        
        # Alpha regulation
        recent = np.mean(self.history['variance'][-500:]) if len(self.history['variance']) > 500 else var
        target_alpha = self.target_var / (recent + 1e-8)
        self.alpha += 0.01 * (target_alpha - self.alpha)
        self.alpha = np.clip(self.alpha, 0.3, 2.0)
        
        return var
    
    def run(self):
        for t in range(self.T):
            self.step(t)
        
        return {
            'final_variance': np.mean(self.history['variance'][-1000:]),
            'final_alpha': np.mean(self.history['alpha'][-100:]),
            'variance_trajectory': self.history['variance'],
            'alpha_trajectory': self.history['alpha']
        }


class SystemDrift(SystemBase):
    """Random drift on target_var."""
    
    def __init__(self, sigma=0.5, target_var=0.015, drift_interval=2000, 
                 drift_sigma=0.005, target_min=0.005, target_max=0.30, seed=42):
        super().__init__(sigma, target_var, seed)
        self.drift_interval = drift_interval
        self.drift_sigma = drift_sigma
        self.target_min = target_min
        self.target_max = target_max
        self.history['target_var'] = [target_var]
        
    def step(self, t):
        var = super().step(t)
        
        # Drift target
        if (t + 1) % self.drift_interval == 0:
            self.target_var += np.random.normal(0, self.drift_sigma)
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        self.history['target_var'].append(self.target_var)
        return var
    
    def run(self):
        for t in range(self.T):
            self.step(t)
        
        return {
            'final_variance': np.mean(self.history['variance'][-1000:]),
            'final_alpha': np.mean(self.history['alpha'][-100:]),
            'final_target_var': np.mean(self.history['target_var'][-500:]),
            'initial_target_var': self.history['target_var'][0],
            'target_drift': np.mean(self.history['target_var'][-500:]) - self.history['target_var'][0],
            'target_trajectory': self.history['target_var'],
            'variance_trajectory': self.history['variance'],
            'alpha_trajectory': self.history['alpha']
        }


class SystemExploration(SystemBase):
    """Target follows achievable variance with exploration."""
    
    def __init__(self, sigma=0.5, target_var=0.015, exploration_interval=1000,
                 exploration_rate=0.01, target_min=0.005, target_max=0.30, seed=42):
        super().__init__(sigma, target_var, seed)
        self.exploration_interval = exploration_interval
        self.exploration_rate = exploration_rate
        self.target_min = target_min
        self.target_max = target_max
        self.history['target_var'] = [target_var]
        
    def step(self, t):
        var = super().step(t)
        
        # Exploration: nudge target toward achievable variance
        if (t + 1) % self.exploration_interval == 0 and t > 2000:
            recent_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            
            # Target chases achievable, but with lag
            if recent_var > self.target_var:
                self.target_var += self.exploration_rate * (recent_var - self.target_var)
            
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        self.history['target_var'].append(self.target_var)
        return var
    
    def run(self):
        for t in range(self.T):
            self.step(t)
        
        return {
            'final_variance': np.mean(self.history['variance'][-1000:]),
            'final_alpha': np.mean(self.history['alpha'][-100:]),
            'final_target_var': np.mean(self.history['target_var'][-500:]),
            'initial_target_var': self.history['target_var'][0],
            'target_drift': np.mean(self.history['target_var'][-500:]) - self.history['target_var'][0],
            'target_trajectory': self.history['target_var'],
            'variance_trajectory': self.history['variance'],
            'alpha_trajectory': self.history['alpha']
        }


class SystemAdaptive(SystemBase):
    """Target slowly follows long-term variance level."""
    
    def __init__(self, sigma=0.5, target_var=0.015, adaptation_interval=500,
                 adaptation_rate=0.001, target_min=0.005, target_max=0.30, seed=42):
        super().__init__(sigma, target_var, seed)
        self.adaptation_interval = adaptation_interval
        self.adaptation_rate = adaptation_rate
        self.target_min = target_min
        self.target_max = target_max
        self.history['target_var'] = [target_var]
        
    def step(self, t):
        var = super().step(t)
        
        # Slow adaptation
        if (t + 1) % self.adaptation_interval == 0 and t > 3000:
            long_var = np.mean(self.history['variance'][-3000:]) if len(self.history['variance']) > 3000 else var
            
            # Target moves toward natural variance level
            self.target_var += self.adaptation_rate * (long_var - self.target_var)
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        self.history['target_var'].append(self.target_var)
        return var
    
    def run(self):
        for t in range(self.T):
            self.step(t)
        
        return {
            'final_variance': np.mean(self.history['variance'][-1000:]),
            'final_alpha': np.mean(self.history['alpha'][-100:]),
            'final_target_var': np.mean(self.history['target_var'][-500:]),
            'initial_target_var': self.history['target_var'][0],
            'target_drift': np.mean(self.history['target_var'][-500:]) - self.history['target_var'][0],
            'target_trajectory': self.history['target_var'],
            'variance_trajectory': self.history['variance'],
            'alpha_trajectory': self.history['alpha']
        }


def run_comparison(mode='compare', sigma=0.5, verbose=True):
    """Run comparison between fixed and evolving targets."""
    
    results = {}
    
    # Baseline
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPARISON: Fixed vs Evolving Target (mode={mode})")
        print(f"{'='*70}")
    
    baseline = SystemBase(sigma=sigma, target_var=0.015, seed=42)
    rb = baseline.run()
    results['fixed'] = {
        'final_variance': rb['final_variance'],
        'final_alpha': rb['final_alpha']
    }
    
    if verbose:
        print(f"\n--- Fixed target (0.015) ---")
        print(f"Final variance: {rb['final_variance']:.4f}")
        print(f"Final alpha: {rb['final_alpha']:.3f}")
    
    # Evolving
    if mode == 'drift':
        evolving = SystemDrift(sigma=sigma, target_var=0.015, 
                               drift_interval=2000, drift_sigma=0.005, seed=42)
    elif mode == 'exploration':
        evolving = SystemExploration(sigma=sigma, target_var=0.015,
                                    exploration_interval=1000, exploration_rate=0.01, seed=42)
    elif mode == 'adaptive':
        evolving = SystemAdaptive(sigma=sigma, target_var=0.015,
                                adaptation_interval=500, adaptation_rate=0.001, seed=42)
    else:
        evolving = SystemDrift(sigma=sigma, target_var=0.015, seed=42)
    
    re = evolving.run()
    results['evolving'] = {
        'final_variance': re['final_variance'],
        'final_alpha': re['final_alpha'],
        'final_target_var': re['final_target_var'],
        'target_drift': re['target_drift'],
        'target_trajectory': re['target_trajectory'],
        'variance_trajectory': re['variance_trajectory']
    }
    
    if verbose:
        print(f"\n--- Evolving target ({mode}) ---")
        print(f"Final variance: {re['final_variance']:.4f}")
        print(f"Final alpha: {re['final_alpha']:.3f}")
        print(f"Final target: {re['final_target_var']:.4f}")
        print(f"Target drift: {re['target_drift']:+.4f}")
    
    # Analysis
    variance_improvement = (re['final_variance'] - rb['final_variance']) / max(rb['final_variance'], 1e-6) * 100
    
    if verbose:
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print(f"{'='*70}")
        print(f"Variance improvement: {variance_improvement:+.1f}%")
        print(f"Target drift: {re['target_drift']:+.4f}")
        
        if abs(re['target_drift']) > 0.001:
            print(f"\n*** Target evolved {'up' if re['target_drift'] > 0 else 'down'}! ***")
        else:
            print(f"\n*** Target remained near initial value ***")
    
    # Trajectory check
    if verbose:
        print(f"\n--- Trajectory samples ---")
        target = re['target_trajectory']
        variance = re['variance_trajectory']
        for t in [5000, 10000, 15000, 19999]:
            if t < len(target):
                print(f"t={t}: target={target[t]:.4f}, variance={np.mean(variance[max(0,t-200):t]):.4f}")
    
    return results


def run_multi_sigma(mode='drift', sigma_list=[0.5, 1.0, 2.0], verbose=True):
    """Test across different noise levels."""
    
    results = {}
    
    for sigma in sigma_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"SIGMA = {sigma}")
            print(f"{'='*60}")
        
        # Fixed
        fixed = SystemBase(sigma=sigma, target_var=0.015, seed=42)
        rf = fixed.run()
        
        # Evolving
        if mode == 'drift':
            evo = SystemDrift(sigma=sigma, target_var=0.015, drift_interval=2000, drift_sigma=0.005, seed=42)
        elif mode == 'exploration':
            evo = SystemExploration(sigma=sigma, target_var=0.015, exploration_interval=1000, seed=42)
        else:
            evo = SystemAdaptive(sigma=sigma, target_var=0.015, adaptation_interval=500, seed=42)
        
        re = evo.run()
        
        improvement = (re['final_variance'] - rf['final_variance']) / rf['final_variance'] * 100
        
        results[str(sigma)] = {
            'fixed_variance': rf['final_variance'],
            'evolving_variance': re['final_variance'],
            'improvement': improvement,
            'target_drift': re['target_drift'],
            'final_target': re['final_target_var']
        }
        
        if verbose:
            print(f"Fixed: {rf['final_variance']:.4f} | Evolving: {re['final_variance']:.4f} | "
                  f"Improv: {improvement:+.1f}% | Target drift: {re['target_drift']:+.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 04: Evolving Target Variable')
    parser.add_argument('--mode', choices=['drift', 'exploration', 'adaptive', 'compare', 'multi-sigma'], 
                       default='compare')
    parser.add_argument('--sigma', type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04: Evolving Target Variable")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'compare':
        results = run_comparison(mode='drift', sigma=args.sigma)
    elif args.mode == 'drift':
        results = run_comparison(mode='drift', sigma=args.sigma)
    elif args.mode == 'exploration':
        results = run_comparison(mode='exploration', sigma=args.sigma)
    elif args.mode == 'adaptive':
        results = run_comparison(mode='adaptive', sigma=args.sigma)
    elif args.mode == 'multi-sigma':
        results = run_multi_sigma(mode='drift', sigma_list=[0.5, 1.0, 2.0])
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'sigma': args.sigma,
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04_evolving_target.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
