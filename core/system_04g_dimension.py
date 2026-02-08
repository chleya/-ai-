"""
System 04g: Dimension Scaling Test

RESEARCH QUESTION:
Does the absolute ceiling (~0.055) scale with dimension N?

HYPOTHESIS:
- If ceiling scales with N: N↑ → ceiling↑ (dimensional effect)
- If ceiling constant: ceiling = f(tanh, W_eigenvalues) not N-dependent

EXPERIMENT:
- N: 20 (baseline), 50, 100
- Gain: 3x (low), 10x (high)
- σ: 0.5, 1.5
- T: 20000

Usage:
    python core/system_04g_dimension.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemDimensionTest:
    """
    System with adjustable dimension for scaling test.
    """
    
    T = 20000
    
    def __init__(self, N=20, sigma=0.5, target_init=0.040, 
                 gain_multiplier=3.0, base_gain=0.003, seed=42):
        
        np.random.seed(seed)
        self.N = N
        self.state = np.random.randn(N)
        self.W = np.random.randn(N, N) / np.sqrt(N)
        self.sigma = sigma
        
        # Gain
        self.target_var = target_init
        self.target_init = target_init
        self.gain_multiplier = gain_multiplier
        self.base_gain = base_gain
        
        # Clips
        self.target_min = 0.01
        self.target_max = 0.25
        
        # State
        self.alpha = 0.45
        
        # Tracking
        self.history = {'variance': [], 'alpha': [], 'target': []}
        
    def step(self, t):
        """Single step."""
        # Evolution
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.alpha
        
        var = np.var(self.state)
        
        # Track
        self.history['variance'].append(var)
        self.history['alpha'].append(self.alpha)
        self.history['target'].append(self.target_var)
        
        # Fast alpha regulation
        recent = np.mean(self.history['variance'][-500:]) if len(self.history['variance']) > 500 else var
        target_alpha = self.target_var / (recent + 1e-8)
        self.alpha += 0.01 * (target_alpha - self.alpha)
        self.alpha = np.clip(self.alpha, 0.3, 2.0)
        
        # Slow target feedback
        if (t + 1) % 200 == 0 and t > 3000:
            window_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            window_target = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
            
            ratio = window_var / max(window_target, 1e-6)
            
            if ratio > 1.0:
                gain_effect = self.base_gain * self.gain_multiplier * (ratio - 1.0)
                self.target_var += gain_effect
            
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 2500 == 0:
                print(f"  [N={self.N:3d}, {t:5d}/{self.T}] var={var:.4f}, target={self.target_var:.4f}")
        
        return self.analyze()
    
    def analyze(self):
        """Analyze results."""
        variance = np.array(self.history['variance'])
        target = np.array(self.history['target'])
        alpha = np.array(self.history['alpha'])
        
        final_var = np.mean(variance[-2000:]) if len(variance) > 2000 else np.mean(variance)
        final_target = np.mean(target[-2000:]) if len(target) > 2000 else np.mean(target)
        final_alpha = np.mean(alpha[-2000:]) if len(alpha) > 2000 else np.mean(alpha)
        
        # Metrics
        if len(target) > 2000:
            target_std = np.std(target[-2000:])
            var_fluctuation = np.std(variance[-2000:]) / max(np.mean(variance[-2000:]), 1e-6)
        else:
            target_std = 0
            var_fluctuation = 0
        
        # Convergence
        if len(target) > 10000:
            late = np.mean(target[-5000:])
            mid = np.mean(target[-10000:-5000])
            convergence = abs(late - mid) / max(mid, 1e-6)
        else:
            convergence = 0
        
        return {
            'N': self.N,
            'sigma': self.sigma,
            'gain': self.gain_multiplier,
            'target_init': self.target_init,
            'final_variance': float(final_var),
            'final_target': float(final_target),
            'target_drift': float(final_target - self.target_init),
            'target_std': float(target_std),
            'var_fluctuation': float(var_fluctuation),
            'convergence': float(convergence),
            'stable': target_std < 0.005 and convergence < 0.05,
            'trajectories': {
                'variance': self.history['variance'][:5000],
                'target': self.history['target'][:5000]
            }
        }


def run_dimension_scan(N_values=[20, 50, 100], gains=[3.0, 10.0],
                       sigmas=[0.5, 1.5], targets=[0.040], verbose=True):
    """Scan dimensions."""
    
    results = {}
    
    for N in N_values:
        for sigma in sigmas:
            for gain in gains:
                for target_init in targets:
                    key = f"N{N}_s{sigma}_g{gain}_t{target_init}"
                    
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"TEST: N={N}, σ={sigma}, gain={gain}x, target={target_init}")
                        print(f"{'='*60}")
                    
                    system = SystemDimensionTest(N=N, sigma=sigma, 
                                          target_init=target_init,
                                          gain_multiplier=gain, seed=42)
                    r = system.run(verbose=verbose)
                    
                    results[key] = r
                    
                    if verbose:
                        print(f"\n  Final variance: {r['final_variance']:.4f}")
                        print(f"  Final target:  {r['final_target']:.4f}")
                        print(f"  Drift:         {r['target_drift']:+.4f}")
                        print(f"  Stable:        {r['stable']}")
    
    return results


def analyze_dimension_scaling(results, verbose=True):
    """Analyze if ceiling scales with N."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("DIMENSION SCALING ANALYSIS")
        print(f"{'='*70}")
        
        # Group by N
        for N in [20, 50, 100]:
            n_results = {k: v for k, v in results.items() if v['N'] == N}
            if not n_results:
                continue
                
            print(f"\nN = {N}:")
            print(f"  {'Config':<25} {'Final Target':<15} {'Drift':<12} {'Stable'}")
            print("  " + "-" * 65)
            
            for k, r in sorted(n_results.items()):
                config = f"σ={r['sigma']}, gain={r['gain']}x"
                print(f"  {config:<25} {r['final_target']:<15.4f} {r['target_drift']:<+12.4f} {r['stable']}")
    
    # Ceiling comparison
    if verbose:
        print(f"\n{'='*70}")
        print("CEILING COMPARISON BY N")
        print(f"{'='*70}")
        
        ceiling_by_N = {}
        for N in [20, 50, 100]:
            n_results = [r['final_target'] for k, r in results.items() if r['N'] == N]
            if n_results:
                ceiling_by_N[N] = np.mean(n_results)
        
        for N in sorted(ceiling_by_N.keys()):
            ceiling = ceiling_by_N[N]
            change = ceiling - ceiling_by_N.get(20, ceiling)
            pct = (change / ceiling_by_N[20]) * 100 if 20 in ceiling_by_N else 0
            arrow = '↑' if change > 0.001 else ('↓' if change < -0.001 else '→')
            print(f"N={N:3d}: ceiling = {ceiling:.4f} ({arrow} {change:+.4f}, {pct:+.1f}%)")
        
        # Conclusion
        if 20 in ceiling_by_N and 50 in ceiling_by_N:
            if ceiling_by_N[50] - ceiling_by_N[20] > 0.005:
                print(f"\n*** CEILING SCALES WITH N ***")
            elif ceiling_by_N[50] - ceiling_by_N[20] < -0.005:
                print(f"\n*** CEILING DECREASES WITH N ***")
            else:
                print(f"\n*** CEILING CONSTANT ACROSS N ***")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 04g: Dimension Scaling')
    parser.add_argument('--mode', choices=['single', 'scan'], default='scan')
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--gain', type=float, default=3.0)
    parser.add_argument('--target', type=float, default=0.040)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04g: DIMENSION SCALING TEST")
    print(f"Mode: {args.mode} | N: {args.N}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemDimensionTest(N=args.N, sigma=args.sigma,
                              target_init=args.target,
                              gain_multiplier=args.gain, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal target: {result['final_target']:.4f}")
        print(f"Stable: {result['stable']}")
        
        results = {f"N{args.N}": result}
    
    elif args.mode == 'scan':
        results = run_dimension_scan(
            N_values=[20, 50, 100],
            gains=[3.0, 10.0],
            sigmas=[0.5, 1.5],
            targets=[0.040],
            verbose=True
        )
    
    # Analysis
    analyze_dimension_scaling(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04g_dimension.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
