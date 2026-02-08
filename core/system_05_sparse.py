"""
System 05: Structured Matrix Test

RESEARCH QUESTION:
Does sparse W change the 1/N scaling law?

HYPOTHESIS:
- Sparse W may increase variance (more independent degrees of freedom)
- May shift equilibrium point upward
- May enhance climbing ability

EXPERIMENT:
- N = 50
- Dense: p=1.0, W ~ N(0, 1/√N)
- Sparse: p=0.1, W non-zero ~ N(0, 1/√(pN))
- Sparse: p=0.01, same scaling
- Gain: 3x, 10x
- 3 seeds average

Usage:
    python core/system_05_sparse.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


def create_W(N, p, seed=42):
    """
    Create sparse weight matrix.
    
    Dense: p = 1.0, W ~ N(0, 1/√N)
    Sparse: p < 1.0, W non-zero ~ N(0, 1/√(pN))
    
    This scaling keeps spectral radius ≈ 1.
    """
    np.random.seed(seed)
    
    if p >= 1.0:
        # Dense matrix
        W = np.random.randn(N, N) / np.sqrt(N)
    else:
        # Sparse matrix
        # Create mask
        mask = (np.random.rand(N, N) < p).astype(float)
        
        # Scale non-zero elements to maintain variance
        scale = 1.0 / np.sqrt(p * N)
        W = np.random.randn(N, N) * mask * scale
    
    return W


class SystemSparse:
    """
    System with adjustable matrix sparsity.
    """
    
    T = 15000
    
    def __init__(self, N=50, p=1.0, sigma=0.5, target_init=0.040,
                 gain_multiplier=3.0, base_gain=0.003, seed=42):
        
        np.random.seed(seed)
        self.N = N
        self.state = np.random.randn(N)
        self.W = create_W(N, p, seed)
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
        self.history = {'variance': [], 'target': []}
        
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
            
            if verbose and t % 2000 == 0:
                print(f"  [t={t:5d}] var={var:.4f}, alpha={self.alpha:.3f}, target={self.target_var:.4f}")
        
        return self.analyze()
    
    def analyze(self):
        """Analyze results."""
        variance = np.array(self.history['variance'])
        target = np.array(self.history['target'])
        
        final_var = np.mean(variance[-2000:]) if len(variance) > 2000 else np.mean(variance)
        final_target = np.mean(target[-2000:]) if len(target) > 2000 else np.mean(target)
        final_alpha = np.mean(self.history['alpha'][-2000:]) if 'alpha' in self.history else 0.45
        
        # Metrics
        if len(variance) > 2000:
            target_std = np.std(target[-2000:])
            var_std = np.std(variance[-2000:])
            var_fluctuation = var_std / max(np.mean(variance[-2000:]), 1e-6)
        else:
            target_std = 0
            var_fluctuation = 0
        
        # Ratio
        ratio = final_var / max(final_target, 1e-6)
        
        return {
            'N': self.N,
            'p': getattr(self, 'N', 50) / getattr(self, 'N', 50),  # Will be set by create_W
            'gain': self.gain_multiplier,
            'final_variance': float(final_var),
            'final_target': float(final_target),
            'final_alpha': float(final_alpha),
            'target_drift': float(final_target - self.target_init),
            'target_std': float(target_std),
            'var_fluctuation': float(var_fluctuation),
            'ratio': float(ratio),
            'stable': target_std < 0.005,
            'trajectories': {
                'variance': self.history['variance'][:3000],
                'target': self.history['target'][:3000]
            }
        }


def run_sparse_scan(N=50, p_values=[1.0, 0.1, 0.01], gains=[3.0, 10.0],
                   sigmas=[0.5], seeds=[42, 123, 456], verbose=True):
    """
    Scan different sparsity levels.
    """
    
    results = {}
    
    for p in p_values:
        for sigma in sigmas:
            for gain in gains:
                for seed in seeds:
                    key = f"p{p}_g{gain}_s{seed}"
                    
                    if verbose:
                        print(f"\n{'='*50}")
                        print(f"TEST: p={p}, gain={gain}x, seed={seed}")
                        print(f"{'='*50}")
                    
                    system = SystemSparse(N=N, p=p, sigma=sigma,
                                        gain_multiplier=gain, seed=seed)
                    r = system.run(verbose=verbose)
                    r['p'] = p
                    r['sigma'] = sigma
                    r['seed'] = seed
                    
                    results[key] = r
                    
                    if verbose:
                        print(f"\n  Final variance: {r['final_variance']:.4f}")
                        print(f"  Final target:  {r['final_target']:.4f}")
                        print(f"  Drift:         {r['target_drift']:+.4f}")
                        print(f"  Ratio:         {r['ratio']:.2f}")
    
    return results


def analyze_sparse(results, verbose=True):
    """Analyze sparsity effects."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("SPARSE MATRIX ANALYSIS")
        print(f"{'='*70}")
        
        # Group by p
        for p in [1.0, 0.1, 0.01]:
            p_results = {k: v for k, v in results.items() if abs(v.get('p', p) - p) < 0.01}
            if not p_results:
                continue
                
            print(f"\nSparsity p = {p}:")
            print(f"  {'Gain':<8} {'Variance':<12} {'Target':<12} {'Drift':<10} {'Ratio':<8}")
            print("  " + "-" * 55)
            
            for k, r in sorted(p_results.items()):
                gain = r.get('gain', 0)
                print(f"  {gain:<8.1f} {r['final_variance']:<12.4f} {r['final_target']:<12.4f} "
                      f"{r['target_drift']:<+10.4f} {r['ratio']:<8.2f}")
    
    # Compare densities
    if verbose:
        print(f"\n{'='*70}")
        print("DENSITY COMPARISON")
        print(f"{'='*70}")
        
        # Average over seeds
        avg_by_p = {}
        for p in [1.0, 0.1, 0.01]:
            p_data = [r['final_variance'] for k, r in results.items() 
                     if abs(r.get('p', p) - p) < 0.01]
            if p_data:
                avg_by_p[p] = np.mean(p_data)
        
        for p in sorted(avg_by_p.keys()):
            v = avg_by_p[p]
            baseline = avg_by_p.get(1.0, v)
            change = (v - baseline) / baseline * 100
            arrow = '↑' if change > 1 else ('↓' if change < -1 else '→')
            print(f"p={p:.2f}: avg variance = {v:.4f} ({arrow} {change:+.1f}%)")
        
        # Conclusion
        if 1.0 in avg_by_p and 0.1 in avg_by_p:
            if avg_by_p[0.1] > avg_by_p[1.0] * 1.1:
                print(f"\n*** SPARSE W INCREASES VARIANCE ***")
            elif avg_by_p[0.1] < avg_by_p[1.0] * 0.9:
                print(f"\n*** SPARSE W DECREASES VARIANCE ***")
            else:
                print(f"\n*** SPARSE W HAS LITTLE EFFECT ***")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 05: Sparse Matrix Test')
    parser.add_argument('--mode', choices=['single', 'scan'], default='scan')
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--gain', type=float, default=3.0)
    parser.add_argument('--sigma', type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 05: SPARSE MATRIX TEST")
    print(f"Mode: {args.mode} | N: {args.N}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemSparse(N=args.N, p=args.p, sigma=args.sigma,
                             gain_multiplier=args.gain, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal variance: {result['final_variance']:.4f}")
        print(f"Final target: {result['final_target']:.4f}")
        print(f"Ratio: {result['ratio']:.2f}")
        
        results = {f"p{args.p}": result}
    
    elif args.mode == 'scan':
        results = run_sparse_scan(
            N=args.N,
            p_values=[1.0, 0.1, 0.01],
            gains=[3.0, 10.0],
            sigmas=[0.5],
            seeds=[42, 123, 456],
            verbose=True
        )
    
    # Analysis
    analyze_sparse(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'N': args.N
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_05_sparse.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
