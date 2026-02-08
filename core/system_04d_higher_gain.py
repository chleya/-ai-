"""
System 04d: Higher Gain Test - Can We Push the Ceiling?

RESEARCH QUESTION:
Is the ceiling (~0.05) a soft limit (pushable) or hard limit (fixed by dynamics)?

HYPOTHESIS:
If ceiling is soft: Higher gain should push variance higher
If ceiling is hard: Higher gain = same ceiling (oscillation/limited by dynamics)

EXPERIMENT:
- Gain multipliers: 1x, 1.5x, 2x, 3x
- Starting targets: 0.040 (below ceiling), 0.050 (at ceiling)
- Measure: Final target, variance, oscillation, stability

SUCCESS CRITERIA:
- Ceiling moves up: Soft limit, gain-dependent
- Ceiling stays: Hard limit, dynamics-dependent
- Oscillation appears: System hitting regulation limit
- Instability: Gain too high

Usage:
    python core/system_04d_higher_gain.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemHigherGain:
    """
    System with adjustable feedback gain for ceiling push test.
    
    Gain formula:
        gain_effect = base_gain * multiplier * max(0, variance/target - 1)
        target += gain_effect
    """
    
    N = 20
    T = 30000
    
    def __init__(self, sigma=0.5, target_init=0.040, gain_multiplier=1.0, 
                 base_gain=0.005, seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        
        # Gain parameters
        self.target_var = target_init
        self.target_init = target_init
        self.gain_multiplier = gain_multiplier
        self.base_gain = base_gain
        
        # State
        self.alpha = 0.45
        
        # Tracking
        self.history = {'variance': [], 'alpha': [], 'target': [], 'gain_effect': []}
        
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
        
        # Slow target feedback (every 300 steps)
        if (t + 1) % 300 == 0 and t > 3000:
            window_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            window_target = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
            
            # Higher gain feedback formula
            # Only increase when variance exceeds target
            ratio = window_var / max(window_target, 1e-6)
            
            if ratio > 1.0:
                # Higher multiplier = stronger push
                gain_effect = self.base_gain * self.gain_multiplier * (ratio - 1.0)
                self.target_var += gain_effect
            
            # Soft clip to prevent explosion
            self.target_var = np.clip(self.target_var, 0.01, 0.15)
            
            self.history['gain_effect'].append(gain_effect if ratio > 1.0 else 0)
        
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
        
        # Calculate metrics
        final_var = np.mean(variance[-3000:])
        final_target = np.mean(target[-3000:])
        target_drift = final_target - self.target_init
        
        # Oscillation detection
        if len(variance) > 1000:
            var_diff = np.diff(variance[-5000:])
            oscillation = np.std(var_diff) / max(np.mean(np.abs(var_diff)), 1e-6)
        else:
            oscillation = 0
        
        # Stability check
        final_10pct = np.mean(variance[-3000:])
        mid_10pct = np.mean(variance[20000:23000])
        stability = abs(final_10pct - mid_10pct) / max(mid_10pct, 1e-6)
        
        return {
            'final_variance': float(final_var),
            'final_target': float(final_target),
            'target_drift': float(target_drift),
            'initial_target': self.target_init,
            'gain_multiplier': self.gain_multiplier,
            'oscillation': float(oscillation),
            'stability': float(stability),
            'stable': stability < 0.1,  # <10% change
            'status': self.categorize_status(oscillation, stability, final_target),
            'trajectories': {
                'variance': self.history['variance'],
                'target': self.history['target'],
                'alpha': self.history['alpha']
            }
        }
    
    def categorize_status(self, oscillation, stability, final_target):
        """Categorize the system's behavior."""
        if stability > 0.2:
            return 'UNSTABLE'
        elif oscillation > 0.5:
            return 'OSCILLATING'
        elif stability > 0.1:
            return 'FLUCTUATING'
        else:
            return 'STABLE'


def run_gain_scan(sigma=0.5, multipliers=[1.0, 1.5, 2.0, 3.0],
                targets=[0.040, 0.050], verbose=True):
    """
    Scan different gain multipliers.
    
    Returns:
        - Does higher gain push ceiling higher?
        - Does oscillation appear?
        - What is the stable ceiling?
    """
    
    results = {}
    
    for target in targets:
        for mult in multipliers:
            key = f"t{target}_m{mult}"
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TEST: target={target}, multiplier={mult}")
                print(f"{'='*60}")
            
            system = SystemHigherGain(sigma=sigma, target_init=target,
                                  gain_multiplier=mult, seed=42)
            r = system.run(verbose=verbose)
            results[key] = {
                'target_init': target,
                'multiplier': mult,
                'final_variance': r['final_variance'],
                'final_target': r['final_target'],
                'target_drift': r['target_drift'],
                'oscillation': r['oscillation'],
                'stable': r['stable'],
                'status': r['status']
            }
            
            if verbose:
                print(f"\n  Final variance: {r['final_variance']:.4f}")
                print(f"  Final target:  {r['final_target']:.4f}")
                print(f"  Target drift:  {r['target_drift']:+.4f}")
                print(f"  Oscillation:  {r['oscillation']:.4f}")
                print(f"  Status:       {r['status']}")
    
    return results


def analyze_gain_effect(results, verbose=True):
    """Analyze if higher gain pushes ceiling."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("GAIN EFFECT ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Target':<10} {'Multiplier':<12} {'Variance':<12} {'Target':<12} {'Drift':<10} {'Status'}")
        print("-" * 80)
    
    ceiling_analysis = {}
    
    for key, r in results.items():
        target = r['target_init']
        mult = r['multiplier']
        
        if verbose:
            print(f"{target:<10} {mult:<12.1f} {r['final_variance']:<12.4f} {r['final_target']:<12.4f} {r['target_drift']:<+10.4f} {r['status']}")
        
        # Group by target
        if target not in ceiling_analysis:
            ceiling_analysis[target] = []
        ceiling_analysis[target].append({
            'multiplier': mult,
            'final_target': r['final_target'],
            'final_variance': r['final_variance'],
            'status': r['status']
        })
    
    # Analyze ceiling movement
    if verbose:
        print(f"\n{'='*70}")
        print("CEILING MOVEMENT ANALYSIS")
        print(f"{'='*70}")
        
        for target, data in ceiling_analysis.items():
            sorted_data = sorted(data, key=lambda x: x['multiplier'])
            
            print(f"\nTarget {target}:")
            for d in sorted_data:
                move = d['final_target'] - target
                moved = '↑' if move > 0.002 else ('↓' if move < -0.002 else '→')
                print(f"  {d['multiplier']}x → {d['final_target']:.4f} ({moved} {move:+.4f})")
            
            # Check if ceiling moved
            min_m = min(d['multiplier'] for d in data)
            max_m = max(d['multiplier'] for d in data)
            min_final = min(d['final_target'] for d in data)
            max_final = max(d['final_target'] for d in data)
            
            if max_final - min_final > 0.005:
                print(f"  *** CEILING MOVED with higher gain! ***")
            else:
                print(f"  === CEILING STABLE across gains ===")
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}")
        
        # Check stability
        unstable = [k for k, r in results.items() if not r['stable']]
        if unstable:
            print(f"WARNING: {len(unstable)} runs unstable")
        else:
            print("All runs STABLE")
        
        # Check ceiling
        all_targets = [r['final_target'] for r in results.values()]
        if max(all_targets) - min(all_targets) > 0.01:
            print(f"Ceiling RANGE: {min(all_targets):.4f} - {max(all_targets):.4f}")
            print("→ Gain DOES affect ceiling")
        else:
            print(f"Ceiling STABLE: {np.mean(all_targets):.4f} ± {np.std(all_targets):.4f}")
            print("→ Gain does NOT significantly affect ceiling")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 04d: Higher Gain Test')
    parser.add_argument('--mode', choices=['single', 'scan', 'analyze'], default='scan')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--target', type=float, default=0.040)
    parser.add_argument('--multiplier', type=float, default=1.0)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04d: HIGHER GAIN TEST")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemHigherGain(sigma=args.sigma, target_init=args.target,
                               gain_multiplier=args.multiplier, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal variance: {result['final_variance']:.4f}")
        print(f"Final target: {result['final_target']:.4f}")
        print(f"Status: {result['status']}")
        
        results = {f"t{args.target}_m{args.multiplier}": result}
    
    elif args.mode == 'scan':
        results = run_gain_scan(
            sigma=args.sigma,
            multipliers=[1.0, 1.5, 2.0, 3.0],
            targets=[0.040, 0.050],
            verbose=True
        )
        
        analyze_gain_effect(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sigma': args.sigma
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04d_higher_gain.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
