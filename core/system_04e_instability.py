"""
System 04e: Instability Threshold Test

RESEARCH QUESTION:
How high can gain go before the system destabilizes?

HYPOTHESIS:
- Low gain (1-3x): Stable climbing
- Medium gain (4-5x): Possible oscillation
- High gain (6x+): Instability or explosion

EXPERIMENT DESIGN:
- Gain: 3x (known stable) → 4x → 5x → 6x → 8x
- Starting target: 0.040
- Clip: target ∈ [0.01, 0.20]
- Duration: T=20000
- Metrics: variance, target, alpha stability

SUCCESS CRITERIA:
- Find FIRST unstable gain
- Characterize instability mode
- Measure Lyapunov-like exponent

Usage:
    python core/system_04e_instability.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemInstabilityTest:
    """
    System designed to test instability threshold.
    
    Key modifications:
    - Higher gain multipliers
    - Longer evolution (T=20000)
    - Detailed oscillation detection
    - Instability classification
    """
    
    N = 20
    T = 20000
    
    def __init__(self, sigma=0.5, target_init=0.040, gain_multiplier=3.0, 
                 base_gain=0.003, seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        
        # Gain parameters
        self.target_var = target_init
        self.target_init = target_init
        self.gain_multiplier = gain_multiplier
        self.base_gain = base_gain
        
        # Clip to prevent explosion
        self.target_min = 0.01
        self.target_max = 0.20
        
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
        
        # Slow target feedback (every 200 steps)
        if (t + 1) % 200 == 0 and t > 3000:
            window_var = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
            window_target = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
            
            # Higher gain feedback
            ratio = window_var / max(window_target, 1e-6)
            
            if ratio > 1.0:
                gain_effect = self.base_gain * self.gain_multiplier * (ratio - 1.0)
                self.target_var += gain_effect
            
            # Clip
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 2000 == 0:
                print(f"  [{t:5d}/{self.T}] var={var:.4f}, alpha={self.alpha:.3f}, target={self.target_var:.4f}")
        
        return self.analyze_stability()
    
    def analyze_stability(self):
        """Analyze system stability."""
        variance = np.array(self.history['variance'])
        target = np.array(self.history['alpha'])
        
        final_var = np.mean(variance[-2000:])
        final_target = np.mean(target[-2000:])
        final_alpha = np.mean(self.history['alpha'][-2000:])
        
        # Oscillation detection
        if len(variance) > 1000:
            var_diff = np.diff(variance[-5000:])
            oscillation = np.std(var_diff) / max(np.mean(np.abs(var_diff)), 1e-6)
        else:
            oscillation = 0
        
        # Convergence check
        late_target = np.mean(self.history['target'][-3000:])
        early_target = np.mean(self.history['target'][-10000:-5000])
        convergence = abs(late_target - early_target) / max(early_target, 1e-6)
        
        # Explosion detection
        early_var = np.mean(variance[:2000])
        late_var = np.mean(variance[-2000:])
        var_ratio = late_var / max(early_var, 1e-6)
        
        # Instability classification
        instability = self.classify_instability(oscillation, convergence, var_ratio, final_var)
        
        return {
            'final_variance': float(final_var),
            'final_target': float(late_target),
            'final_alpha': float(final_alpha),
            'oscillation': float(oscillation),
            'convergence': float(convergence),
            'variance_ratio': float(var_ratio),
            'target_drift': float(late_target - self.target_init),
            'instability': instability,
            'stable': instability == 'STABLE',
            'trajectories': {
                'variance': self.history['variance'],
                'target': self.history['target'],
                'alpha': self.history['alpha']
            }
        }
    
    def classify_instability(self, oscillation, convergence, var_ratio, final_var):
        """Classify system stability."""
        # Explosion criteria
        if var_ratio > 10 or final_var > 0.5:
            return 'EXPLODED'
        elif final_var < 0.001:
            return 'COLLAPSED'
        
        # Oscillation criteria
        elif oscillation > 0.5:
            return 'OSCILLATING'
        elif oscillation > 0.2:
            return 'FLUCTUATING'
        
        # Convergence criteria
        elif convergence > 0.1:
            return 'DIVERGING'
        elif convergence > 0.05:
            return 'SLOW_CONVERGENCE'
        
        else:
            return 'STABLE'


def run_instability_scan(sigma=0.5, gains=[3.0, 4.0, 5.0, 6.0, 8.0], verbose=True):
    """
    Scan gain levels to find instability threshold.
    
    Returns:
        - First unstable gain
        - Instability mode
        - Stability metrics for each gain
    """
    
    results = {}
    
    for gain in gains:
        key = f"gain_{gain}"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GAIN = {gain}x")
            print(f"{'='*60}")
        
        system = SystemInstabilityTest(sigma=sigma, target_init=0.040,
                                   gain_multiplier=gain, seed=42)
        r = system.run(verbose=verbose)
        
        results[key] = {
            'gain': gain,
            'sigma': sigma,
            'final_variance': r['final_variance'],
            'final_target': r['final_target'],
            'oscillation': r['oscillation'],
            'convergence': r['convergence'],
            'variance_ratio': r['variance_ratio'],
            'instability': r['instability'],
            'stable': r['stable']
        }
        
        if verbose:
            print(f"\n  Final variance: {r['final_variance']:.4f}")
            print(f"  Final target:  {r['final_target']:.4f}")
            print(f"  Oscillation:  {r['oscillation']:.4f}")
            print(f"  Convergence:  {r['convergence']:.4f}")
            print(f"  Status:       {r['instability']}")
    
    return results


def analyze_threshold(results, verbose=True):
    """Analyze instability threshold."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("INSTABILITY THRESHOLD ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Gain':<10} {'Variance':<12} {'Target':<12} {'Osc':<10} {'Status':<15}")
        print("-" * 70)
    
    threshold = None
    
    for key, r in results.items():
        if verbose:
            print(f"{r['gain']:<10.1f} {r['final_variance']:<12.4f} {r['final_target']:<12.4f} "
                  f"{r['oscillation']:<10.4f} {r['instability']:<15}")
        
        if not r['stable'] and threshold is None:
            threshold = r['gain']
    
    if verbose:
        print("-" * 70)
        
        if threshold:
            print(f"\n*** INSTABILITY THRESHOLD: gain = {threshold} ***")
        else:
            print(f"\n*** ALL GAINS STABLE up to {max(r['gain'] for r in results.values())} ***")
        
        # Summary
        stable = sum(1 for r in results.values() if r['stable'])
        unstable = sum(1 for r in results.values() if not r['stable'])
        
        print(f"\nStable: {stable}, Unstable: {unstable}")
        
        if unstable > 0:
            unstable_types = {}
            for r in results.values():
                if not r['stable']:
                    t = r['instability']
                    unstable_types[t] = unstable_types.get(t, 0) + 1
            
            print(f"Instability types: {unstable_types}")
    
    return threshold


def main():
    parser = argparse.ArgumentParser(description='System 04e: Instability Threshold Test')
    parser.add_argument('--mode', choices=['single', 'scan', 'analyze'], default='scan')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--gain', type=float, default=3.0)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04e: INSTABILITY THRESHOLD TEST")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemInstabilityTest(sigma=args.sigma, target_init=0.040,
                                    gain_multiplier=args.gain, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nInstability: {result['instability']}")
        print(f"Stable: {result['stable']}")
        
        results = {f"gain_{args.gain}": result}
    
    elif args.mode == 'scan':
        results = run_instability_scan(
            sigma=args.sigma,
            gains=[3.0, 4.0, 5.0, 6.0, 8.0],
            verbose=True
        )
    
    # Analysis
    threshold = analyze_threshold(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sigma': args.sigma
        },
        'results': results,
        'threshold': threshold
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04e_instability.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
