"""
System 04f: Extreme Gain Test

RESEARCH QUESTION:
How high can gain go before real instability appears?

ADDED PROTECTIONS:
1. target clip: [0.01, 0.25]
2. variance emergency brake: if var > 0.8, halve
3. early stopping: if variance std > 0.05 for 500 steps

IMPROVED METRICS:
1. target std (last 2000 steps)
2. variance relative fluctuation
3. max single-step change
4. overshoot ratio: variance/target > 1.3 or < 0.7
5. alpha volatility

Usage:
    python core/system_04f_extreme_gain.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemExtremeGain:
    """
    System with extreme gain testing and protections.
    """
    
    N = 20
    T = 25000
    
    def __init__(self, sigma=0.5, target_init=0.040, gain_multiplier=10.0, 
                 base_gain=0.003, seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
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
        
        # Early stopping flags
        self.instability_detected = False
        self.stop_reason = None
        
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
            
            # High gain feedback
            ratio = window_var / max(window_target, 1e-6)
            
            if ratio > 1.0:
                gain_effect = self.base_gain * self.gain_multiplier * (ratio - 1.0)
                self.target_var += gain_effect
            
            # Clip target
            self.target_var = np.clip(self.target_var, self.target_min, self.target_max)
            
            # Emergency brake: if variance too high
            if window_var > 0.8:
                self.target_var *= 0.5  # Halve to brake
                if not self.instability_detected:
                    self.instability_detected = True
                    self.stop_reason = 'variance_emergency_brake'
        
        return var
    
    def check_early_stop(self, t):
        """Check early stopping conditions."""
        if t < 5000:
            return False
            
        # Check last 500 steps variance std
        if len(self.history['variance']) > 500:
            recent_var = self.history['variance'][-500:]
            if np.std(recent_var) > 0.05:
                if not self.instability_detected:
                    self.instability_detected = True
                    self.stop_reason = 'high_variance_std'
                return True
        
        return False
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            # Early stopping check
            if self.check_early_stop(t):
                if verbose:
                    print(f"  [EARLY STOP at {t}] {self.stop_reason}")
                break
            
            if verbose and t % 2500 == 0:
                print(f"  [{t:5d}/{self.T}] var={var:.4f}, alpha={self.alpha:.3f}, target={self.target_var:.4f}")
        
        return self.analyze()
    
    def analyze(self):
        """Analyze system behavior."""
        variance = np.array(self.history['variance'])
        target = np.array(self.history['target'])
        alpha = np.array(self.history['alpha'])
        
        # Basic metrics
        final_var = np.mean(variance[-2000:]) if len(variance) > 2000 else np.mean(variance)
        final_target = np.mean(target[-2000:]) if len(target) > 2000 else np.mean(target)
        final_alpha = np.mean(alpha[-2000:]) if len(alpha) > 2000 else np.mean(alpha)
        
        # Improved metrics
        if len(variance) > 2000:
            last_var = variance[-2000:]
            last_target = target[-2000:]
            last_alpha = alpha[-2000:]
            
            target_std = np.std(last_target)
            var_fluctuation = np.std(last_var) / max(np.mean(last_var), 1e-6)
            
            # Max single step change
            target_diff = np.abs(np.diff(last_target))
            max_step_change = np.max(target_diff)
            
            # Overshoot ratio
            overshoot_count = np.sum((last_var / last_target) > 1.3)
            undershoot_count = np.sum((last_var / last_target) < 0.7)
            
            # Alpha volatility
            alpha_volatility = np.std(last_alpha) / max(np.mean(last_alpha), 1e-6)
        else:
            target_std = 0
            var_fluctuation = 0
            max_step_change = 0
            overshoot_count = 0
            undershoot_count = 0
            alpha_volatility = 0
        
        # Convergence
        if len(target) > 10000:
            late_target = np.mean(target[-5000:])
            mid_target = np.mean(target[-10000:-5000])
            convergence = abs(late_target - mid_target) / max(mid_target, 1e-6)
        else:
            convergence = 0
        
        # Classification
        instability = self.classify_instability(
            target_std, var_fluctuation, max_step_change,
            overshoot_count, undershoot_count, alpha_volatility,
            convergence, self.instability_detected
        )
        
        return {
            'final_variance': float(final_var),
            'final_target': float(final_target),
            'final_alpha': float(final_alpha),
            'target_drift': float(final_target - self.target_init),
            'target_std': float(target_std),
            'var_fluctuation': float(var_fluctuation),
            'max_step_change': float(max_step_change),
            'overshoot_count': int(overshoot_count),
            'undershoot_count': int(undershoot_count),
            'alpha_volatility': float(alpha_volatility),
            'convergence': float(convergence),
            'instability': instability,
            'stable': instability == 'STABLE',
            'early_stop': self.instability_detected,
            'stop_reason': self.stop_reason,
            'trajectories': {
                'variance': self.history['variance'][:5000],  # Save memory
                'target': self.history['target'][:5000],
                'alpha': self.history['alpha'][:5000]
            }
        }
    
    def classify_instability(self, target_std, var_fluctuation, max_step_change,
                            overshoot, undershoot, alpha_vol, convergence, early_stop):
        """Classify system instability level."""
        # Severe conditions
        if early_stop:
            return 'EARLY_STOP'
        
        # Extreme values
        if target_std > 0.01:
            return 'HIGHLY_UNSTABLE'
        elif target_std > 0.005:
            return 'UNSTABLE'
        
        # Moderate conditions
        if var_fluctuation > 0.20:
            return 'HIGH_FLUCTUATION'
        elif max_step_change > 0.005:
            return 'STEP_UNSTABLE'
        
        # Mild conditions
        if alpha_vol > 0.1:
            return 'ALPHA_VOLATILE'
        elif overshoot > 100 or undershoot > 100:
            return 'OVERSHOOTING'
        
        # Convergence
        if convergence > 0.1:
            return 'DIVERGING'
        elif convergence > 0.05:
            return 'SLOW_CONVERGENCE'
        
        return 'STABLE'


def run_extreme_gain_scan(sigma=0.5, gains=[10.0, 15.0, 20.0], 
                          targets=[0.040, 0.050], verbose=True):
    """Scan extreme gain levels."""
    
    results = {}
    
    for target in targets:
        for gain in gains:
            key = f"t{target}_g{gain}"
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TEST: target={target}, gain={gain}x")
                print(f"{'='*60}")
            
            system = SystemExtremeGain(sigma=sigma, target_init=target,
                                 gain_multiplier=gain, seed=42)
            r = system.run(verbose=verbose)
            
            results[key] = {
                'target_init': target,
                'gain': gain,
                'final_variance': r['final_variance'],
                'final_target': r['final_target'],
                'target_std': r['target_std'],
                'var_fluctuation': r['var_fluctuation'],
                'max_step_change': r['max_step_change'],
                'overshoot_count': r['overshoot_count'],
                'alpha_volatility': r['alpha_volatility'],
                'convergence': r['convergence'],
                'instability': r['instability'],
                'stable': r['stable'],
                'early_stop': r['early_stop'],
                'stop_reason': r.get('stop_reason')
            }
            
            if verbose:
                print(f"\n  Final variance: {r['final_variance']:.4f}")
                print(f"  Final target:  {r['final_target']:.4f}")
                print(f"  Target std:    {r['target_std']:.6f}")
                print(f"  Var fluctuation: {r['var_fluctuation']:.4f}")
                print(f"  Status:       {r['instability']}")
    
    return results


def analyze_extreme_gain(results, verbose=True):
    """Analyze extreme gain results."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("EXTREME GAIN ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Target':<10} {'Gain':<8} {'Final':<10} {'Drift':<10} {'Std':<10} {'Status':<15}")
        print("-" * 70)
    
    threshold = None
    max_gain_tested = 0
    
    for key, r in results.items():
        if verbose:
            print(f"{r['target_init']:<10} {r['gain']:<8.1f} {r['final_target']:<10.4f} "
                  f"{r['target_drift']:<+10.4f} {r['target_std']:<10.6f} {r['instability']:<15}")
        
        max_gain_tested = max(max_gain_tested, r['gain'])
        
        if r['instability'] in ['EARLY_STOP', 'HIGHLY_UNSTABLE', 'UNSTABLE']:
            if threshold is None:
                threshold = r['gain']
    
    if verbose:
        print("-" * 70)
        
        if threshold:
            print(f"\n*** INSTABILITY THRESHOLD: gain = {threshold} ***")
        else:
            print(f"\n*** ALL GAINS STABLE up to {max_gain_tested} ***")
            print(f"*** Real threshold > {max_gain_tested} ***")
        
        # Trend analysis
        print(f"\n{'='*70}")
        print("CEILING vs GAIN TREND")
        print(f"{'='*70}")
        
        for target in [0.040, 0.050]:
            target_data = [(r['gain'], r['final_target']) for k, r in results.items() 
                          if r['target_init'] == target]
            target_data.sort(key=lambda x: x[0])
            
            print(f"\nTarget {target}:")
            for g, t in target_data:
                move = t - target
                arrow = 'UP' if move > 0.001 else ('DOWN' if move < -0.001 else 'SAME')
                print(f"  {g:5.1f}x → {t:.4f} ({arrow} {move:+.4f})")
    
    return threshold


def main():
    parser = argparse.ArgumentParser(description='System 04f: Extreme Gain Test')
    parser.add_argument('--mode', choices=['single', 'scan'], default='scan')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--gain', type=float, default=10.0)
    parser.add_argument('--target', type=float, default=0.040)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 04f: EXTREME GAIN TEST")
    print(f"Mode: {args.mode} | Sigma: {args.sigma}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = SystemExtremeGain(sigma=args.sigma, target_init=args.target,
                              gain_multiplier=args.gain, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nInstability: {result['instability']}")
        print(f"Stable: {result['stable']}")
        
        results = {f"t{args.target}_g{args.gain}": result}
    
    elif args.mode == 'scan':
        results = run_extreme_gain_scan(
            sigma=args.sigma,
            gains=[10.0, 15.0, 20.0],
            targets=[0.040, 0.050],
            verbose=True
        )
    
    # Analysis
    threshold = analyze_extreme_gain(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sigma': args.sigma
        },
        'results': results,
        'threshold': threshold
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_04f_extreme_gain.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
