"""
System 03: Dynamic Alpha - Self-Regulating Constraint

Research Question: Can the system develop self-stabilization by adapting its constraint strength?

Core Hypothesis:
- System dies when normalization is too weak (alpha low) -> state collapses -> variance -> 0
- If alpha can adapt based on recent variance, system might "pull itself up"
- This represents "self-organization for survival"

Design:
- Track recent variance (window=500)
- Adjust alpha up/down based on threshold
- Initial alpha in death zone (0.4-0.5)
- Test if dynamic regulation can rescue dying systems

Usage:
    python core/system_03_dynamic_alpha.py --mode compare
    python core/system_03_dynamic_alpha.py --mode scan --sigma 0.5
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemDynamicAlpha:
    """
    Dynamical system with self-regulating constraint (alpha).
    
    Evolution rule:
        state_{t+1} = tanh(W · state_t + noise)
        state_{t+1} = alpha_t · state_{t+1} / ||state_{t+1}||
        
    Alpha regulation rule:
        alpha_t+1 = alpha_t + gain · (target_alpha - alpha_t)
        target_alpha = alpha_base · (desired_var / recent_var)
    """
    
    N = 20
    T = 10000
    
    def __init__(self, sigma, alpha_init=0.45, desired_var=0.015,
                 window=500, gain=0.01, alpha_min=0.3, alpha_max=2.0,
                 seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        
        # Dynamic alpha parameters
        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.desired_var = desired_var
        self.window = window
        self.gain = gain
        
        # Tracking
        self.history = {'variance': [], 'alpha': [], 'norm': []}
        self.ema_var = None
        self.beta = 0.9  # EMA factor for variance
        
    def step(self, t):
        """Single evolution step with dynamic alpha."""
        # Evolution
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        
        # Current alpha normalization
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.alpha
        
        # Track
        var = np.var(self.state)
        self.history['variance'].append(var)
        self.history['alpha'].append(self.alpha)
        self.history['norm'].append(norm)
        
        # Update EMA variance
        if self.ema_var is None:
            self.ema_var = var
        else:
            self.ema_var = self.beta * self.ema_var + (1 - self.beta) * var
        
        # Dynamic alpha regulation
        if t >= self.window:
            recent_var = np.mean(self.history['variance'][-self.window:])
            
            # Target alpha based on desired variance
            target_alpha = self.desired_var / (recent_var + 1e-8)
            target_alpha = np.clip(target_alpha, self.alpha_min, self.alpha_max * 2)
            
            # Smooth adjustment
            self.alpha += self.gain * (target_alpha - self.alpha)
            self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 1000 == 0:
                print(f"  [{t:5d}/{self.T}] var={var:.4f}, alpha={self.alpha:.3f}")
        
        final_var = np.mean(self.history['variance'][-1000:])
        final_alpha = np.mean(self.history['alpha'][-100:])
        
        return {
            'final_variance': final_var,
            'final_alpha': final_alpha,
            'alpha_trajectory': self.history['alpha'],
            'variance_trajectory': self.history['variance'],
            'survived': final_var > 0.01,
            'alpha_range': [min(self.history['alpha']), max(self.history['alpha'])]
        }


class SystemFixedAlpha:
    """Baseline: Fixed alpha (for comparison)"""
    
    N = 20
    T = 10000
    
    def __init__(self, sigma, alpha, seed=42):
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.alpha = alpha
        self.history = []
        
    def run(self, verbose=True):
        for t in range(self.T):
            noise = np.random.randn(self.N) * self.sigma
            self.state = np.tanh(self.W @ self.state + noise)
            norm = np.linalg.norm(self.state)
            if norm > 1e-8:
                self.state = (self.state / norm) * self.alpha
            self.history.append(np.var(self.state))
            
            if verbose and t % 1000 == 0:
                print(f"  [{t:5d}/{self.T}] var={self.history[-1]:.4f}")
        
        final_var = np.mean(self.history[-1000:])
        return {
            'final_variance': final_var,
            'survived': final_var > 0.01
        }


def run_comparison(sigma=0.5, verbose=True):
    """
    Compare three conditions:
    1. Fixed alpha=0.45 (known to die)
    2. Fixed alpha=0.50 (barely survives)
    3. Dynamic alpha (starts at 0.45)
    """
    
    results = {}
    
    # Condition 1: Fixed alpha=0.45 (death zone)
    if verbose:
        print(f"\n{'='*60}")
        print("CONDITION 1: Fixed alpha=0.45 (Death Zone)")
        print(f"{'='*60}")
    
    sys1 = SystemFixedAlpha(sigma=sigma, alpha=0.45, seed=42)
    r1 = sys1.run(verbose)
    results['fixed_045'] = r1
    status1 = "DEAD" if not r1['survived'] else "ALIVE"
    if verbose:
        print(f"  Result: {status1} | variance={r1['final_variance']:.4f}")
    
    # Condition 2: Fixed alpha=0.50 (survival zone)
    if verbose:
        print(f"\n{'='*60}")
        print("CONDITION 2: Fixed alpha=0.50 (Survival Zone)")
        print(f"{'='*60}")
    
    sys2 = SystemFixedAlpha(sigma=sigma, alpha=0.50, seed=42)
    r2 = sys2.run(verbose)
    results['fixed_050'] = r2
    status2 = "DEAD" if not r2['survived'] else "ALIVE"
    if verbose:
        print(f"  Result: {status2} | variance={r2['final_variance']:.4f}")
    
    # Condition 3: Dynamic alpha
    if verbose:
        print(f"\n{'='*60}")
        print("CONDITION 3: Dynamic Alpha (Starts at 0.45)")
        print(f"{'='*60}")
    
    sys3 = SystemDynamicAlpha(sigma=sigma, alpha_init=0.45, 
                               desired_var=0.015, window=500,
                               gain=0.01, seed=42)
    r3 = sys3.run(verbose)
    results['dynamic'] = r3
    status3 = "DEAD" if not r3['survived'] else "ALIVE"
    if verbose:
        print(f"  Result: {status3} | variance={r3['final_variance']:.4f}")
        print(f"  Alpha range: {r3['alpha_range'][0]:.3f} - {r3['alpha_range'][1]:.3f}")
        print(f"  Final alpha: {r3['final_alpha']:.3f}")
    
    return results


def run_scan(sigma=0.5, verbose=True):
    """Scan across multiple conditions."""
    
    conditions = [
        ('Fixed 0.45', {'type': 'fixed', 'alpha': 0.45}),
        ('Fixed 0.48', {'type': 'fixed', 'alpha': 0.48}),
        ('Fixed 0.50', {'type': 'fixed', 'alpha': 0.50}),
        ('Dynamic 0.45', {'type': 'dynamic', 'alpha_init': 0.45, 'desired_var': 0.015}),
        ('Dynamic 0.45-var02', {'type': 'dynamic', 'alpha_init': 0.45, 'desired_var': 0.020}),
    ]
    
    results = {}
    
    for name, config in conditions:
        if verbose:
            print(f"\n{name}:")
        
        if config['type'] == 'fixed':
            sys = SystemFixedAlpha(sigma=sigma, alpha=config['alpha'], seed=42)
            r = sys.run(verbose=False)
        else:
            sys = SystemDynamicAlpha(sigma=sigma, alpha_init=config.get('alpha_init', 0.45),
                                      desired_var=config.get('desired_var', 0.015),
                                      seed=42)
            r = sys.run(verbose=False)
        
        results[name] = r
        status = "YES" if r['survived'] else "NO"
        var = r['final_variance']
        
        if config['type'] == 'dynamic':
            print(f"  {status} | var={var:.4f} | alpha=[{r['alpha_range'][0]:.3f}-{r['alpha_range'][1]:.3f}]")
        else:
            print(f"  {status} | var={var:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 03: Dynamic Alpha Experiment')
    parser.add_argument('--mode', choices=['compare', 'scan'], default='compare')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--verbose', type=bool, default=True)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 03: Dynamic Alpha - Self-Regulating Constraint")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'compare':
        results = run_comparison(args.sigma, args.verbose)
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        r1 = results['fixed_045']
        r2 = results['fixed_050']
        r3 = results['dynamic']
        
        print(f"\nFixed alpha=0.45: {'DEAD' if not r1['survived'] else 'ALIVE'} (var={r1['final_variance']:.4f})")
        print(f"Fixed alpha=0.50: {'DEAD' if not r2['survived'] else 'ALIVE'} (var={r2['final_variance']:.4f})")
        print(f"Dynamic alpha:    {'DEAD' if not r3['survived'] else 'ALIVE'} (var={r3['final_variance']:.4f})")
        
        # Key insight
        if r3['survived'] and not r1['survived']:
            print(f"\n*** BREAKTHROUGH: Dynamic alpha rescued dying system! ***")
            print(f"    Alpha adapted from 0.45 to {r3['final_alpha']:.3f}")
        elif r3['survived']:
            print(f"\n*** Dynamic alpha maintains survival ***")
        else:
            print(f"\n*** No rescue effect observed ***")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'sigma': args.sigma,
            'results': {
                'fixed_045': r1,
                'fixed_050': r2,
                'dynamic': r3
            }
        }
        
    else:
        results = run_scan(args.sigma, args.verbose)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'sigma': args.sigma,
            'results': results
        }
    
    # Save
    output_path = Path(__file__).parent / 'results' / 'exp_03_dynamic_alpha.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
