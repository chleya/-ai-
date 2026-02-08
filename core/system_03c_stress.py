"""
Experiment 03c: Stress Tests - How Fragile is Homeostasis?

RESEARCH QUESTION:
How robust is the dynamic alpha homeostasis? What are its limits?

STRESS TEST SUITE:
1. Extreme State Zero: state = state * 0 (total annihilation)
2. Extreme Shrink: state = state * 0.01 (near-zero)
3. Sustained Depression: alpha = 0.3 for 1000 steps, then release
4. Noise Jump: sigma from 0.5 → 5.0 (10x noise increase)

HYPOTHESIS:
- System should recover from mild perturbations
- Extreme perturbations may reveal limits
- Sustained stress tests "true" robustness

SUCCESS CRITERIA:
- Recovery ratio > 80%: Robust
- Recovery ratio 50-80%: Moderate
- Recovery ratio < 50%: Fragile (reveals limit)

Usage:
    python core/system_03c_stress.py --mode suite
    python core/system_03c_stress.py --mode single --test extreme_zero
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class SystemDynamicAlpha:
    """Dynamic alpha system with stress testing support."""
    
    N = 20
    T = 25000  # Extended for stress tests
    
    def __init__(self, sigma=0.5, alpha_init=0.45, desired_var=0.015,
                 window=500, gain=0.01, alpha_min=0.3, alpha_max=2.0,
                 seed=42):
        
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.desired_var = desired_var
        self.window = window
        self.gain = gain
        
        self.history = {'variance': [], 'alpha': [], 'norm': [], 'sigma': []}
        self.ema_var = None
        self.beta = 0.9
        
    def step(self, t):
        """Single step with dynamic alpha."""
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.alpha
        
        var = np.var(self.state)
        self.history['variance'].append(var)
        self.history['alpha'].append(self.alpha)
        self.history['norm'].append(norm)
        self.history['sigma'].append(self.sigma)
        
        if self.ema_var is None:
            self.ema_var = var
        else:
            self.ema_var = self.beta * self.ema_var + (1 - self.beta) * var
        
        if t >= self.window:
            recent_var = np.mean(self.history['variance'][-self.window:])
            target_alpha = self.desired_var / (recent_var + 1e-8)
            target_alpha = np.clip(target_alpha, self.alpha_min, self.alpha_max * 2)
            self.alpha += self.gain * (target_alpha - self.alpha)
            self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)
        
        return var
    
    def apply_perturbation(self, ptype, intensity=1.0):
        """Apply various perturbation types."""
        if ptype == 'zero':
            self.state = np.zeros(self.N)
        elif ptype == 'shrink_001':
            self.state = self.state * 0.01
        elif ptype == 'shrink_01':
            self.state = self.state * 0.1
        elif ptype == 'depress_alpha':
            self.alpha = 0.3
        elif ptype == 'noise_jump':
            self.sigma = 5.0
        return self.state
    
    def run_phases(self, phases, verbose=True):
        """Run with multiple phases for stress testing."""
        
        phase_info = {}
        
        for start, end, config in phases:
            ptype = config.get('type', 'normal')
            phase_info[ptype] = {'start': start, 'end': end, 'config': config}
            
            for t in range(start, min(end, self.T)):
                if t == start and 'perturbation' in config:
                    self.apply_perturbation(config['perturbation'])
                
                var = self.step(t)
                
                if verbose and t % 2000 == 0:
                    print(f"  [{t:5d}/{self.T}] var={var:.4f}, alpha={self.alpha:.3f}, sigma={self.sigma:.1f}")
        
        final_var = np.mean(self.history['variance'][-500:])
        final_alpha = np.mean(self.history['alpha'][-500:])
        
        return {
            'final_variance': final_var,
            'final_alpha': final_alpha,
            'alpha_trajectory': self.history['alpha'],
            'variance_trajectory': self.history['variance'],
            'sigma_trajectory': self.history['sigma'],
            'phase_info': phase_info,
            'survived': final_var > 0.01
        }


def analyze_stress(result, shock_time, recovery_window=2000):
    """Analyze stress test recovery."""
    
    variance = np.array(result['variance_trajectory'])
    alpha = np.array(result['alpha_trajectory'])
    
    # Pre-shock baseline
    pre_var = np.mean(variance[max(0, shock_time-500):shock_time])
    pre_alpha = np.mean(alpha[max(0, shock_time-500):shock_time])
    
    # Post-shock minimum
    post_shock = variance[shock_time:min(shock_time+200, len(variance))]
    min_var = np.min(post_shock)
    min_var_time = shock_time + np.argmin(post_shock)
    
    # Recovery (later portion)
    if len(variance) > 2000:
        final_var = np.mean(variance[-500:])
        final_alpha = np.mean(alpha[-500:])
    else:
        final_var = np.mean(variance[-100:])
        final_alpha = np.mean(alpha[-100:])
    
    # Recovery ratio
    recovery_ratio = final_var / pre_var if pre_var > 0 else 0
    
    # Recovery threshold time
    recovery_threshold = 0.8 * pre_var
    recovery_time = None
    for i, v in enumerate(variance[shock_time:]):
        if v >= recovery_threshold:
            recovery_time = i
            break
    
    # Alpha response
    max_alpha = np.max(alpha[shock_time:])
    alpha_boost = max_alpha - pre_alpha
    
    return {
        'pre_shock_variance': pre_var,
        'post_shock_min_variance': min_var,
        'post_shock_min_time': min_var_time,
        'final_variance': final_var,
        'recovery_ratio': recovery_ratio,
        'recovery_time': recovery_time,
        'pre_shock_alpha': pre_alpha,
        'max_alpha': max_alpha,
        'alpha_boost': alpha_boost,
        'survived': final_var > 0.01,
        'robustness': 'ROBUST' if recovery_ratio > 0.8 else ('MODERATE' if recovery_ratio > 0.5 else 'FRAGILE')
    }


def run_stress_suite(verbose=True):
    """Run complete stress test suite."""
    
    tests = [
        ('baseline', 'Baseline (no perturbation)', [(0, 20000, {})]),
        
        ('extreme_zero', 'Total Annihilation (state=0)', [
            (0, 5000, {}),
            (5000, 5001, {'perturbation': 'zero'}),
            (5001, 20000, {})
        ]),
        
        ('extreme_shrink_001', 'Extreme Shrink (state*=0.01)', [
            (0, 5000, {}),
            (5000, 5001, {'perturbation': 'shrink_001'}),
            (5001, 20000, {})
        ]),
        
        ('sustained_depress', 'Sustained Depression (α=0.3, 1000 steps)', [
            (0, 5000, {}),
            (5000, 6000, {'perturbation': 'depress_alpha'}),
            (6000, 20000, {})
        ]),
        
        ('noise_jump_5x', 'Noise Jump (σ: 0.5→5.0)', [
            (0, 5000, {}),
            (5000, 5001, {'perturbation': 'noise_jump'}),
            (5001, 20000, {})
        ]),
        
        ('noise_jump_10x', 'Extreme Noise Jump (σ: 0.5→10.0)', [
            (0, 5000, {}),
            (5000, 5001, {'perturbation': 'noise_jump', 'intensity': 10}),
            (5001, 20000, {})
        ]),
    ]
    
    results = {}
    
    for test_name, desc, phases in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {desc}")
        print(f"{'='*60}")
        
        system = SystemDynamicAlpha(sigma=0.5, alpha_init=0.45, desired_var=0.015, seed=42)
        
        # Modify sigma for noise jump tests
        if 'noise_jump' in test_name:
            if test_name == 'noise_jump_10x':
                # Set up extreme jump
                pass  # Will be handled in perturbation
            system.sigma = 0.5
        
        result = system.run_phases(phases, verbose=verbose)
        analysis = analyze_stress(result, shock_time=5000)
        
        results[test_name] = {
            'description': desc,
            'result': result,
            'analysis': analysis
        }
        
        # Print summary
        print(f"\n--- ANALYSIS ---")
        print(f"Pre-shock variance:  {analysis['pre_shock_variance']:.4f}")
        print(f"Post-shock minimum:  {analysis['post_shock_min_variance']:.4f} (t={analysis['post_shock_min_time']})")
        print(f"Final variance:      {analysis['final_variance']:.4f}")
        print(f"Recovery ratio:      {analysis['recovery_ratio']*100:.1f}%")
        print(f"Recovery time:       {analysis['recovery_time'] if analysis['recovery_time'] else 'N/A'} steps")
        print(f"Alpha boost:        {analysis['alpha_boost']:+.3f}")
        print(f"ROBUSTNESS:         {analysis['robustness']}")
    
    return results


def print_summary(results):
    """Print comprehensive summary."""
    
    print(f"\n{'='*70}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Test':<25} {'Recovery':<12} {'Status':<12} {'Alpha Boost':<12}")
    print("-" * 70)
    
    for test_name, data in results.items():
        a = data['analysis']
        recovery_pct = a['recovery_ratio'] * 100
        status = a['robustness'][:6]
        alpha_boost = f"{a['alpha_boost']:+.3f}"
        print(f"{test_name:<25} {recovery_pct:>6.1f}%     {status:<12} {alpha_boost:<12}")
    
    print("-" * 70)
    
    # Count
    robust = sum(1 for r in results.values() if r['analysis']['robustness'] == 'ROBUST')
    moderate = sum(1 for r in results.values() if r['analysis']['robustness'] == 'MODERATE')
    fragile = sum(1 for r in results.values() if r['analysis']['robustness'] == 'FRAGILE')
    
    print(f"\nRobust: {robust}, Moderate: {moderate}, Fragile: {fragile}")
    
    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    
    # Find most and least robust
    robust_tests = [(k, v['analysis']['recovery_ratio']) for k, v in results.items() 
                   if v['analysis']['recovery_ratio'] > 0]
    if robust_tests:
        most_robust = max(robust_tests, key=lambda x: x[1])
        least_robust = min(robust_tests, key=lambda x: x[1])
        
        print(f"Most robust:   {most_robust[0]} ({most_robust[1]*100:.1f}%)")
        print(f"Least robust:  {least_robust[0]} ({least_robust[1]*100:.1f}%)")
        
        # Find limit
        fragile_tests = [(k, v['analysis']) for k, v in results.items() 
                        if v['analysis']['robustness'] == 'FRAGILE']
        if fragile_tests:
            print(f"\nLIMIT REACHED: {fragile_tests[0][0]}")
            print(f"  {fragile_tests[0][1]['recovery_ratio']*100:.1f}% recovery")


def main():
    parser = argparse.ArgumentParser(description='Stress Tests for Homeostasis')
    parser.add_argument('--mode', choices=['suite', 'single'], default='suite')
    parser.add_argument('--test', choices=['baseline', 'extreme_zero', 'extreme_shrink_001', 
                                          'sustained_depress', 'noise_jump_5x', 'noise_jump_10x'],
                       default='suite')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 03c: STRESS TEST SUITE")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"{'='*70}")
    
    if args.mode == 'suite':
        results = run_stress_suite()
        print_summary(results)
        
        # Save
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': {
                k: {
                    'description': v['description'],
                    'analysis': v['analysis']
                } for k, v in results.items()
            }
        }
        
        output_path = Path(__file__).parent / 'results' / 'exp_03c_stress_tests.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"Results saved: {output_path}")
        print(f"{'='*70}")
    
    else:
        # Single test
        test_configs = {
            'baseline': [(0, 20000, {})],
            'extreme_zero': [(0, 5000, {}), (5000, 5001, {'perturbation': 'zero'}), (5001, 20000, {})],
            'extreme_shrink_001': [(0, 5000, {}), (5000, 5001, {'perturbation': 'shrink_001'}), (5001, 20000, {})],
            'sustained_depress': [(0, 5000, {}), (5000, 6000, {'perturbation': 'depress_alpha'}), (6000, 20000, {})],
            'noise_jump_5x': [(0, 5000, {}), (5000, 5001, {'perturbation': 'noise_jump'}), (5001, 20000, {})],
        }
        
        system = SystemDynamicAlpha(sigma=0.5, alpha_init=0.45, desired_var=0.015, seed=42)
        result = system.run_phases(test_configs[args.test])
        analysis = analyze_stress(result, shock_time=5000)
        
        print(f"\nTest: {args.test}")
        print(f"Recovery ratio: {analysis['recovery_ratio']*100:.1f}%")
        print(f"Robustness: {analysis['robustness']}")


if __name__ == '__main__':
    main()
