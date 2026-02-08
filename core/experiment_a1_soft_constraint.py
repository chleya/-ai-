"""
Experiment A1: Soft Energy Constraint + Slow Variable

Goal: Test if soft energy penalty allows variance escape from ~0.28

Hypothesis:
- Hard constraint (||x||=α) suppressed dynamics
- Soft penalty (λ||x||²) might allow more flexibility

Setup:
- x_{t+1} = tanh(W x_t + noise)
- Energy penalty: E += λ * ||x||²
- But no hard normalization

Usage:
    python core/experiment_a1.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SoftConstraintSystem:
    """
    System with soft energy constraint.
    
    x_{t+1} = tanh(W x_t + noise)
    Energy: E = λ * ||x||² (penalty, not constraint)
    """
    
    T = 30000
    
    def __init__(self, N=50, sigma=0.5, lambda_penalty=0.001, 
                 seed=42):
        
        np.random.seed(seed)
        self.N = N
        self.x = np.random.randn(N) * 0.1
        self.W = np.random.randn(N, N) / np.sqrt(N)
        self.sigma = sigma
        self.lambda_penalty = lambda_penalty
        
        # Tracking
        self.history = {'variance': [], 'energy': [], 'x': []}
        
    def step(self, t):
        """Single step."""
        # Evolution
        noise = np.random.randn(self.N) * self.sigma
        self.x = np.tanh(self.W @ self.x + noise)
        
        # Soft energy penalty
        norm_sq = np.sum(self.x ** 2)
        energy = self.lambda_penalty * norm_sq
        
        # Apply penalty (push toward lower energy)
        if energy > 0:
            self.x *= np.exp(-energy)  # Soft push
        
        # Track
        var = np.var(self.x)
        self.history['variance'].append(var)
        self.history['energy'].append(energy)
        self.history['x'].append(self.x.copy())
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 5000 == 0:
                print(f"  [t={t:5d}] var={var:.4f}, energy={self.history['energy'][-1]:.6f}")
        
        return self.analyze()
    
    def analyze(self):
        """Analyze results."""
        variance = np.array(self.history['variance'])
        energy = np.array(self.history['energy'])
        
        final_var = np.mean(variance[-5000:])
        final_energy = np.mean(energy[-1000:])
        
        # Stability
        late_var = np.mean(variance[-10000:])
        early_var = np.mean(variance[:5000])
        trend = (late_var - early_var) / max(early_var, 1e-6)
        
        # Check if escaped 0.28
        escaped = late_var > 0.28 * 1.1  # 10% above 0.28
        
        return {
            'N': self.N,
            'sigma': self.sigma,
            'lambda': self.lambda_penalty,
            'final_variance': float(final_var),
            'final_energy': float(final_energy),
            'trend': float(trend),
            'escaped': bool(escaped),
            'stable': abs(trend) < 0.1,
            'trajectories': {
                'variance': self.history['variance'],
                'energy': self.history['energy']
            }
        }


def run_lambda_scan(N=50, sigma=0.5, lambdas=[0.0001, 0.001, 0.01, 0.1], verbose=True):
    """Scan different lambda values."""
    
    results = {}
    
    for lam in lambdas:
        key = f"lambda_{lam}"
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Lambda = {lam}")
            print(f"{'='*50}")
        
        system = SoftConstraintSystem(N=N, sigma=sigma, lambda_penalty=lam, seed=42)
        r = system.run(verbose=verbose)
        
        results[key] = {
            'lambda': lam,
            'final_variance': r['final_variance'],
            'final_energy': r['final_energy'],
            'trend': r['trend'],
            'escaped': r['escaped'],
            'stable': r['stable']
        }
        
        if verbose:
            print(f"\n  Final variance: {r['final_variance']:.4f}")
            print(f"  Final energy:  {r['final_energy']:.6f}")
            print(f"  Trend:         {r['trend']:+.4f}")
            print(f"  Escaped 0.28:  {r['escaped']}")
            print(f"  Stable:        {r['stable']}")
    
    return results


def analyze_results(results, verbose=True):
    """Analyze soft constraint results."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("SOFT CONSTRAINT ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Lambda':<10} {'Variance':<12} {'Energy':<12} {'Trend':<10} {'Escaped'}")
        print("-" * 60)
        
        for key, r in sorted(results.items(), key=lambda x: float(x[1]['lambda'])):
            print(f"{r['lambda']:<10.4f} {r['final_variance']:<12.4f} {r['final_energy']:<12.6f} {r['trend']:<+10.4f} {r['escaped']}")
        
        # Check if any lambda allowed escape
        escaped_any = any(r['escaped'] for r in results.values())
        
        print()
        if escaped_any:
            print("*** SOME LAMBDA VALUES ALLOWED ESCAPE FROM 0.28! ***")
        else:
            print("*** NO ESCAPE: All variances returned to ~0.28 ***")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment A1: Soft Energy Constraint')
    parser.add_argument('--mode', choices=['single', 'scan'], default='scan')
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--lambda', type=float, default=0.001)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT A1: Soft Energy Constraint")
    print(f"Mode: {args.mode} | N: {args.N}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SoftConstraintSystem(N=args.N, sigma=args.sigma, 
                                    lambda_penalty=args.lambda, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal variance: {result['final_variance']:.4f}")
        print(f"Escaped 0.28: {result['escaped']}")
        
        results = {f"lambda_{args.lambda}": result}
    
    elif args.mode == 'scan':
        results = run_lambda_scan(
            N=args.N,
            sigma=args.sigma,
            lambdas=[0.0001, 0.001, 0.01, 0.1],
            verbose=True
        )
    
    # Analysis
    analyze_results(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'N': args.N,
            'sigma': args.sigma
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_A1_soft_constraint.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
