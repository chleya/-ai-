"""
System 04b: Multi-Sigma Systematic Scan & Feedback Boundary Test

RESEARCH QUESTIONS:
1. What's the "working window" of positive feedback across different noise levels?
2. What's the "climbing limit" - how high can variance go?
3. Does negative feedback break or just converge differently?

THREE FEEDBACK MODES:
1. POSITIVE: variance ↑ → target ↑ (current working mode)
2. NEGATIVE: variance ↑ → target ↓ (opposite, to prove positive feedback effect)
3. FIXED: target constant (baseline)

MULTI-SIGMA SCAN:
- Sigma: 0.5, 1.0, 1.5, 2.0, 3.0
- Initial Target: 0.010, 0.015, 0.020
- Repeats: 2-3 per combination

OUTPUT:
1. target_var(t) curves
2. variance(t) curves
3. target vs variance phase diagram

WARNING:
Keep feedback strength moderate (0.5%-2%) to prevent instability.

Usage:
    python core/system_04b_feedback_scan.py --mode scan --sigma 0.5
    python core/system_04b_feedback_scan.py --mode compare --sigma 0.5
    python core/system_04b_feedback_scan.py --mode full
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemFeedback:
    """
    System with controllable feedback mode for systematic study.
    
    Modes:
    - 'positive': variance ↑ → target ↑ (climbing)
    - 'negative': variance ↑ → target ↓ (opposite)
    - 'fixed': target constant (baseline)
    """
    
    N = 20
    T = 30000
    
    def __init__(self, sigma=0.5, target_init=0.015, feedback_mode='positive',
                 feedback_strength=0.01,  # 1% adjustment
                 target_min=0.005, target_max=0.30,
                 seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        
        # Feedback parameters
        self.feedback_mode = feedback_mode
        self.feedback_strength = feedback_strength
        self.target_min = target_min
        self.target_max = target_max
        
        # State
        self.alpha = 0.45
        self.target_var = target_init
        self.target_init = target_init
        
        # Tracking
        self.history = {'variance': [], 'alpha': [], 'target': []}
        
    def step(self, t):
        """Single step with feedback regulation."""
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
        
        # Slow target feedback (every 500 steps)
        if (t + 1) % 500 == 0 and t > 3000:
            window_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            window_target = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
            
            # Feedback based on mode
            if self.feedback_mode == 'positive':
                # If variance is higher than what target would achieve, raise target
                if window_var > window_target * 1.05:
                    self.target_var *= (1 + self.feedback_strength)
                # Small exploration kick
                elif window_var > 0.015:
                    self.target_var += np.random.normal(0, 0.001)
                    
            elif self.feedback_mode == 'negative':
                # If variance is high, LOWER target (opposite)
                if window_var > window_target * 1.1:
                    self.target_var *= (1 - self.feedback_strength)
                    
            elif self.feedback_mode == 'fixed':
                pass  # No change
            
            # Clip
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 3000 == 0:
                print(f"  [{t:5d}/{self.T}] var={var:.4f}, alpha={self.alpha:.3f}, target={self.target_var:.4f}")
        
        # Analysis
        variance = np.array(self.history['variance'])
        target = np.array(self.history['target'])
        alpha = np.array(self.history['alpha'])
        
        return {
            'final_variance': float(np.mean(variance[-2000:])),
            'final_alpha': float(np.mean(alpha[-500:])),
            'final_target': float(np.mean(target[-500:])),
            'initial_target': self.target_init,
            'target_drift': float(np.mean(target[-500:]) - self.target_init),
            'variance_max': float(np.max(variance)),
            'variance_mean': float(np.mean(variance)),
            'trajectories': {
                'variance': self.history['variance'],
                'target': self.history['target'],
                'alpha': self.history['alpha']
            },
            'phase_data': {
                'target': target.tolist(),
                'variance': variance.tolist()
            }
        }


def run_single_scan(sigma=0.5, target_init=0.015, modes=['positive', 'negative', 'fixed'], 
                    verbose=True):
    """Scan all three feedback modes for one sigma/target combination."""
    
    results = {}
    
    for mode in modes:
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Mode: {mode.upper()}")
            print(f"{'─'*50}")
        
        system = SystemFeedback(sigma=sigma, target_init=target_init, 
                               feedback_mode=mode, seed=42)
        r = system.run(verbose=verbose)
        r['mode'] = mode
        results[mode] = r
        
        if verbose:
            print(f"  Final variance: {r['final_variance']:.4f}")
            print(f"  Final target:  {r['final_target']:.4f}")
            print(f"  Target drift:  {r['target_drift']:+.4f}")
    
    return results


def run_multi_sigma_scan(sigma_list=[0.5, 1.0, 1.5], target_list=[0.010, 0.015, 0.020],
                       modes=['positive', 'negative', 'fixed'], verbose=True):
    """Systematic scan across sigma and target values."""
    
    all_results = {}
    
    for sigma in sigma_list:
        for target in target_list:
            key = f"sigma_{sigma}_target_{target}"
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"COMBINATION: sigma={sigma}, target={target}")
                print(f"{'='*70}")
            
            results = run_single_scan(sigma, target, modes, verbose)
            all_results[key] = {
                'sigma': sigma,
                'target_init': target,
                'results': {m: {
                    'final_variance': r['final_variance'],
                    'final_target': r['final_target'],
                    'target_drift': r['target_drift'],
                    'variance_max': r['variance_max']
                } for m, r in results.items()}
            }
    
    return all_results


def analyze_positive_feedback_boundary(results, verbose=True):
    """Analyze if positive feedback has limits or boundaries."""
    
    positive_results = []
    fixed_results = []
    
    for key, data in results.items():
        if 'results' in data:
            if 'positive' in data['results'] and 'fixed' in data['results']:
                p = data['results']['positive']
                f = data['results']['fixed']
                
                positive_results.append({
                    'sigma': data['sigma'],
                    'target': data['target_init'],
                    'positive_var': p['final_variance'],
                    'fixed_var': f['final_variance'],
                    'improvement': (p['final_variance'] - f['final_variance']) / max(f['final_variance'], 1e-6) * 100,
                    'positive_target': p['final_target'],
                    'fixed_target': f['final_target'],
                    'positive_drift': p['target_drift']
                })
    
    if verbose:
        print(f"\n{'='*70}")
        print("POSITIVE FEEDBACK ANALYSIS")
        print(f"{'='*70}")
        print(f"\n{'Sigma':<8} {'Target':<8} {'Fixed':<10} {'Positive':<10} {'Improv':<10} {'Drift':<10}")
        print("-" * 70)
        
        for r in positive_results:
            print(f"{r['sigma']:<8} {r['target']:<8} {r['fixed_var']:<10.4f} "
                  f"{r['positive_var']:<10.4f} {r['improvement']:>+8.1f}% {r['positive_drift']:>+8.4f}")
        
        print("-" * 70)
        
        # Key insights
        avg_improvement = np.mean([r['improvement'] for r in positive_results])
        max_improvement = max([r['improvement'] for r in positive_results])
        min_improvement = min([r['improvement'] for r in positive_results])
        
        print(f"\nAverage improvement: {avg_improvement:+.1f}%")
        print(f"Max improvement: {max_improvement:+.1f}%")
        print(f"Min improvement: {min_improvement:+.1f}%")
        
        # Check for instability
        unstable = [r for r in positive_results if r['positive_var'] > 0.10]
        if unstable:
            print(f"\n*** POTENTIAL INSTABILITY: {len(unstable)} runs with variance > 0.10 ***")
            for r in unstable:
                print(f"    sigma={r['sigma']}, target={r['target']}: var={r['positive_var']:.4f}")
        else:
            print(f"\n*** All runs STABLE (variance < 0.10) ***")
    
    return positive_results


def print_phase_comparison(results, verbose=True):
    """Compare positive vs negative vs fixed in phase space."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("FEEDBACK MODE COMPARISON (Final States)")
        print(f"{'='*70}")
        print(f"\n{'Key':<30} {'Mode':<12} {'Variance':<12} {'Target':<12} {'Drift':<12}")
        print("-" * 80)
    
    for key, data in results.items():
        if 'results' in data:
            for mode, r in data['results'].items():
                if verbose:
                    print(f"{key:<30} {mode:<12} {r['final_variance']:<12.4f} "
                          f"{r['final_target']:<12.4f} {r['target_drift']:<+12.4f}")


def main():
    parser = argparse.ArgumentParser(description='System 04b: Feedback Boundary Test')
    parser.add_argument('--mode', choices=['single', 'scan', 'compare', 'full'], default='single')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--target', type=float, default=0.015)
    parser.add_argument('--strength', type=float, default=0.01)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04b: FEEDBACK BOUNDARY TEST")
    print(f"Mode: {args.mode} | Sigma: {args.sigma} | Target: {args.target}")
    print(f"Feedback strength: {args.strength*100:.1f}%")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        results = run_single_scan(args.sigma, args.target, verbose=True)
        
    elif args.mode == 'scan':
        results = run_multi_sigma_scan([args.sigma], [args.target], verbose=True)
        
    elif args.mode == 'compare':
        results = run_single_scan(args.sigma, args.target, verbose=True)
        
    elif args.mode == 'full':
        # Full systematic scan
        results = run_multi_sigma_scan(
            sigma_list=[0.5, 1.0, 1.5],
            target_list=[0.010, 0.015, 0.020],
            verbose=True
        )
    
    # Analysis
    if args.mode in ['single', 'scan', 'full']:
        print_phase_comparison(results)
        
        if args.mode in ['scan', 'full']:
            analyze_positive_feedback_boundary(results)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'config': {
            'sigma': args.sigma,
            'target_init': args.target,
            'feedback_strength': args.strength
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04b_feedback_scan.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
