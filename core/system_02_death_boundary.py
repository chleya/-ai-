"""
Experiment 02b: Death Boundary Scan with STDP

Research Question: Can STDP expand survival boundary in critical parameter regions?

Usage:
    python core/system_02_death_boundary.py --sigma-min 0.75 --sigma-max 1.0 --step 0.025
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemBase:
    """Base system (No STDP)"""
    N = 20
    T = 10000
    
    def __init__(self, sigma, norm_strength, seed=42):
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.norm_strength = norm_strength
        self.history = []
        
    def run(self, verbose=False):
        for t in range(self.T):
            noise = np.random.randn(self.N) * self.sigma
            self.state = np.tanh(self.W @ self.state + noise)
            norm = np.linalg.norm(self.state)
            if norm > 1e-8:
                self.state = (self.state / norm) * self.norm_strength
            self.history.append(np.var(self.state))
        
        final_var = np.mean(self.history[-1000:])
        final_std = np.std(self.history[-1000:])
        
        # Death detection: when variance drops below threshold
        death_threshold = 0.005
        death_time = None
        for i, v in enumerate(self.history):
            if v < death_threshold:
                death_time = i
                break
        
        return {
            'final_variance': final_var,
            'variance_stability': final_std,
            'survived': final_var > 0.01,
            'death_time': death_time,
            'variance_trajectory': self.history
        }


class SystemSTDP(SystemBase):
    """System with STDP self-adjustment"""
    
    def __init__(self, sigma, norm_strength, stdp_lr=1e-4, seed=42):
        super().__init__(sigma, norm_strength, seed)
        self.stdp_lr = stdp_lr
        self.activity_history = np.zeros((self.N,))
        self.W_evolution = []
        
    def apply_stdp(self):
        """Apply weak STDP: strengthen active connections"""
        connection_activity = np.outer(self.activity_history, self.activity_history)
        active_mask = connection_activity > 0.1
        
        W_update = np.zeros_like(self.W)
        W_update[active_mask] = self.stdp_lr * connection_activity[active_mask]
        
        self.W += W_update
        
        # Renormalize to preserve spectral radius ~1
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        if spectral_radius > 1e-8:
            self.W = self.W / spectral_radius
        
        self.W_evolution.append(self.W.copy())
        
    def run(self, verbose=False):
        for t in range(self.T):
            noise = np.random.randn(self.N) * self.sigma
            self.state = np.tanh(self.W @ self.state + noise)
            norm = np.linalg.norm(self.state)
            if norm > 1e-8:
                self.state = (self.state / norm) * self.norm_strength
            
            # Update activity
            activity = np.abs(self.state)
            self.activity_history = 0.9 * self.activity_history + 0.1 * activity
            
            self.history.append(np.var(self.state))
            
            # Apply STDP every 50 steps
            if (t + 1) % 50 == 0:
                self.apply_stdp()
        
        final_var = np.mean(self.history[-1000:])
        final_std = np.std(self.history[-1000:])
        
        death_threshold = 0.005
        death_time = None
        for i, v in enumerate(self.history):
            if v < death_threshold:
                death_time = i
                break
        
        return {
            'final_variance': final_var,
            'variance_stability': final_std,
            'survived': final_var > 0.01,
            'death_time': death_time,
            'variance_trajectory': self.history
        }


def run_death_boundary_scan(sigma_min, sigma_max, sigma_step, alphas, repeats=3, verbose=True):
    """Run comprehensive death boundary scan with multiple STDP configurations"""
    
    sigma_values = np.arange(sigma_min, sigma_max + sigma_step/2, sigma_step)
    
    # Configuration matrix
    configs = [
        {'name': 'Baseline', 'class': SystemBase, 'lr': None},
        {'name': 'STDP-weak', 'class': SystemSTDP, 'lr': 1e-4},
        {'name': 'STDP-mid', 'class': SystemSTDP, 'lr': 3e-4},
        {'name': 'STDP-strong', 'class': SystemSTDP, 'lr': 1e-3},
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'sigma_range': [float(sigma_min), float(sigma_max)],
        'alphas': [float(a) for a in alphas],
        'configurations': {}
    }
    
    for config in configs:
        config_name = config['name']
        results['configurations'][config_name] = {
            'learning_rate': config['lr'],
            'survival_matrix': {},
            'variance_matrix': {}
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name} (lr={config['lr']})")
            print(f"{'='*60}")
        
        for alpha in alphas:
            survival_rates = []
            final_variances = []
            
            for sigma in sigma_values:
                survived_count = 0
                variances = []
                
                for r in range(repeats):
                    if config['lr'] is None:
                        sys = SystemBase(sigma=sigma, norm_strength=alpha, seed=42+r)
                    else:
                        sys = SystemSTDP(sigma=sigma, norm_strength=alpha, 
                                         stdp_lr=config['lr'], seed=42+r)
                    
                    result = sys.run(verbose=False)
                    survived_count += 1 if result['survived'] else 0
                    variances.append(result['final_variance'])
                
                survival_rate = survived_count / repeats
                survival_rates.append(survival_rate)
                final_variances.append(np.mean(variances))
                
                status = "YES" if survival_rate > 0.5 else "NO"
                if verbose:
                    print(f"  sigma={sigma:.3f}, alpha={alpha:.1f}: {status} (survive={survival_rate:.0%}, var={np.mean(variances):.4f})")
            
            results['configurations'][config_name]['survival_matrix'][str(alpha)] = survival_rates
            results['configurations'][config_name]['variance_matrix'][str(alpha)] = final_variances
    
    return sigma_values, results


def analyze_boundary_effects(results, sigma_values):
    """Analyze if STDP expands survival boundary"""
    
    analysis = {
        'boundary_shift': {},
        'variance_improvement': {}
    }
    
    for alpha in results['alphas']:
        baseline_survival = np.array(results['configurations']['Baseline']['survival_matrix'][str(alpha)])
        
        # Find boundary: first sigma where survival < 50%
        boundary_idx = np.where(baseline_survival < 0.5)[0]
        if len(boundary_idx) > 0:
            baseline_boundary = float(sigma_values[boundary_idx[0]])
        else:
            baseline_boundary = float(sigma_values[-1])
        
        analysis['boundary_shift'][alpha] = {
            'baseline': baseline_boundary
        }
        
        for config_name in ['STDP-weak', 'STDP-mid', 'STDP-strong']:
            config_survival = np.array(results['configurations'][config_name]['survival_matrix'][str(alpha)])
            
            # Find boundary for this config
            boundary_idx = np.where(config_survival < 0.5)[0]
            if len(boundary_idx) > 0:
                config_boundary = float(sigma_values[boundary_idx[0]])
            else:
                config_boundary = float(sigma_values[-1])
            
            shift = config_boundary - baseline_boundary
            analysis['boundary_shift'][alpha][config_name] = shift
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Death Boundary Scan with STDP')
    parser.add_argument('--sigma-min', type=float, default=0.75)
    parser.add_argument('--sigma-max', type=float, default=1.0)
    parser.add_argument('--step', type=float, default=0.025)
    parser.add_argument('--alphas', type=str, default='0.8,1.0,1.2')
    parser.add_argument('--repeats', type=int, default=3)
    
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(',')]
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 02b: Death Boundary Scan with STDP")
    print(f"Sigma Range: [{args.sigma_min}, {args.sigma_max}], step={args.step}")
    print(f"Alphas: {alphas}")
    print(f"{'='*70}\n")
    
    sigma_values, results = run_death_boundary_scan(
        args.sigma_min, args.sigma_max, args.step, alphas, args.repeats
    )
    
    # Analyze effects
    analysis = analyze_boundary_effects(results, sigma_values)
    results['analysis'] = analysis
    
    # Save results
    output_path = Path(__file__).parent / 'results' / 'exp_02b_death_boundary.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    for alpha in results['alphas']:
        print(f"\nAlpha = {alpha}:")
        baseline_boundary = analysis['boundary_shift'][alpha]['baseline']
        print(f"  Baseline boundary: sigma = {baseline_boundary:.3f}")
        
        for config_name in ['STDP-weak', 'STDP-mid', 'STDP-strong']:
            shift = analysis['boundary_shift'][alpha].get(config_name, 0)
            sign = "+" if shift > 0 else ""
            print(f"  {config_name}: sigma shift = {sign}{shift:.3f}")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")
    
    # Key insight
    max_shift = max([
        analysis['boundary_shift'][a].get('STDP-mid', 0) 
        for a in results['alphas']
    ])
    
    if max_shift > 0.02:
        print(f"\n*** STDP EFFECT DETECTED: Max boundary shift = {max_shift:.3f} ***")
    else:
        print(f"\n*** No significant STDP effect at boundary (max shift = {max_shift:.3f}) ***")


if __name__ == '__main__':
    main()
