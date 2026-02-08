"""
System 04c: Climbing Limit Test - Finding the Ceiling

RESEARCH QUESTION:
How high can positive feedback climb? Is there a saturation point?

EXPERIMENT DESIGN:
- Higher initial targets: 0.030, 0.040, 0.050
- Test if variance continues to grow or saturates
- Observe target drift behavior

HYPOTHESIS:
- If positive feedback continues: variance keeps growing
- If saturation: target stops drifting, variance stabilizes
- If instability: variance oscillates or diverges

SUCCESS CRITERIA:
- Continued growth: "No ceiling found"
- Saturation: "Ceiling discovered at target ≈ X"
- Oscillation: "System limits itself through dynamics"

Usage:
    python core/system_04c_climbing_limit.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemClimbingLimit:
    """
    System designed to test climbing limits.
    
    Key modifications:
    - Start from higher initial targets
    - Track target drift carefully
    - Detect saturation/oscillation
    """
    
    N = 20
    T = 30000
    
    def __init__(self, sigma=0.5, target_init=0.030, feedback_gain=1.01, 
                 seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.target_var = target_init
        self.target_init = target_init
        self.feedback_gain = feedback_gain
        self.alpha = 0.45
        
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
        
        # Positive feedback on target (every 500 steps)
        if (t + 1) % 500 == 0 and t > 3000:
            window_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            window_target = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
            
            # Positive feedback: if variance exceeds target, raise target
            if window_var > window_target * 1.05:
                self.target_var *= self.feedback_gain
            
            self.target_var = np.clip(self.target_var, 0.005, 0.50)  # Higher cap
        
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
        
        # Detect saturation
        late_variance = np.mean(variance[-5000:])
        early_variance = np.mean(variance[:5000])
        
        late_target = np.mean(target[-5000:])
        early_target = np.mean(target[:5000])
        
        growth_rate = (late_variance - early_variance) / max(early_variance, 1e-6)
        
        # Saturation detection: is growth slowing down?
        mid_variance = np.mean(variance[10000:15000])
        late_growth = (late_variance - mid_variance) / max(mid_variance, 1e-6)
        early_growth = (mid_variance - early_variance) / max(early_variance, 1e-6)
        
        saturation_ratio = late_growth / max(early_growth, 1e-6)
        
        # Oscillation detection
        if len(variance) > 1000:
            var_diff = np.diff(variance[-2000:])
            oscillation = np.std(var_diff) / max(np.mean(np.abs(var_diff)), 1e-6)
        else:
            oscillation = 0
        
        return {
            'final_variance': float(late_variance),
            'final_alpha': float(np.mean(alpha[-500:])),
            'final_target': float(late_target),
            'initial_target': self.target_init,
            'target_drift': float(late_target - self.target_init),
            'variance_growth': growth_rate * 100,
            'saturation_ratio': saturation_ratio,
            'oscillation': oscillation,
            'status': self.categorize_status(saturation_ratio, oscillation),
            'trajectories': {
                'variance': self.history['variance'],
                'target': self.history['target'],
                'alpha': self.history['alpha']
            }
        }
    
    def categorize_status(self, saturation_ratio, oscillation):
        """Categorize the system's behavior."""
        if oscillation > 0.5:
            return 'OSCILLATING'
        elif saturation_ratio < 0.1:
            return 'SATURATED'
        elif saturation_ratio < 0.5:
            return 'SLOWING'
        elif saturation_ratio < 1.0:
            return 'GROWING'
        else:
            return 'ACCELERATING'


def run_climbing_scan(sigma=0.5, targets=[0.030, 0.040, 0.050], 
                     feedback_gains=[1.01, 1.015, 1.02], verbose=True):
    """
    Scan higher targets and feedback gains.
    
    Returns:
        - Final variance for each combination
        - Saturation status
        - Target drift behavior
    """
    
    results = {}
    
    for target in targets:
        for gain in feedback_gains:
            key = f"target_{target}_gain_{gain}"
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TEST: target={target}, gain={gain}")
                print(f"{'='*60}")
            
            system = SystemClimbingLimit(sigma=sigma, target_init=target, 
                                        feedback_gain=gain, seed=42)
            r = system.run(verbose=verbose)
            results[key] = r
            
            if verbose:
                print(f"\n  Final variance: {r['final_variance']:.4f}")
                print(f"  Final target:  {r['final_target']:.4f}")
                print(f"  Target drift:  {r['target_drift']:+.4f}")
                print(f"  Growth rate:   {r['variance_growth']:.1f}%")
                print(f"  Saturation:    {r['saturation_ratio']:.3f}")
                print(f"  Status:       {r['status']}")
    
    return results


def analyze_limits(results, verbose=True):
    """Analyze the climbing limits."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("CLIMBING LIMIT ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Config':<35} {'Variance':<12} {'Drift':<12} {'Status':<15}")
        print("-" * 85)
    
    status_counts = {'SATURATED': 0, 'SLOWING': 0, 'GROWING': 0, 
                     'ACCELERATING': 0, 'OSCILLATING': 0}
    
    max_variance = 0
    max_config = None
    
    for key, r in results.items():
        if verbose:
            print(f"{key:<35} {r['final_variance']:<12.4f} {r['target_drift']:<+12.4f} {r['status']:<15}")
        
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1
        
        if r['final_variance'] > max_variance:
            max_variance = r['final_variance']
            max_config = key
    
    if verbose:
        print("-" * 85)
        print(f"\nStatus distribution:")
        for status, count in status_counts.items():
            if count > 0:
                print(f"  {status}: {count}")
        
        print(f"\nMaximum variance achieved: {max_variance:.4f}")
        print(f"Configuration: {max_config}")
        
        # Key insights
        print(f"\n{'='*70}")
        print("KEY INSIGHTS")
        print(f"{'='*70}")
        
        if status_counts.get('SATURATED', 0) > 0:
            print("*** CEILING DETECTED: Some runs saturated ***")
        elif status_counts.get('GROWING', 0) > 0 or status_counts.get('ACCELERATING', 0) > 0:
            print("*** NO CEILING YET: System continues to grow ***")
        else:
            print("*** MIXED RESULTS ***")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='System 04c: Climbing Limit Test')
    parser.add_argument('--mode', choices=['single', 'scan', 'analyze'], default='scan')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--target', type=float, default=0.030)
    parser.add_argument('--gain', type=float, default=1.01)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04c: CLIMBING LIMIT TEST")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemClimbingLimit(sigma=args.sigma, target_init=args.target,
                                   feedback_gain=args.gain, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal variance: {result['final_variance']:.4f}")
        print(f"Final target: {result['final_target']:.4f}")
        print(f"Status: {result['status']}")
        
        results = {f"target_{args.target}_gain_{args.gain}": result}
    
    elif args.mode == 'scan':
        results = run_climbing_scan(
            sigma=args.sigma,
            targets=[0.030, 0.040, 0.050],
            feedback_gains=[1.01, 1.015],
            verbose=True
        )
    
    # Analysis
    analyze_limits(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sigma': args.sigma
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04c_climbing_limit.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
