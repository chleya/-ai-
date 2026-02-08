"""
Experiment 03b: Perturbation Recovery Test

RESEARCH QUESTION:
Does dynamic alpha system demonstrate TRUE self-stabilization recovery?

HYPOTHESIS:
If system has active self-stabilization, it should recover from severe perturbations:
- After shock (state *= 0.2), variance drops then alpha increases to compensate
- After forced alpha depression (alpha=0.35), system should escape when released
- Recovery time should be proportional to perturbation severity

EXPERIMENT DESIGN:
Phase 1 (0-5000): Normal evolution with dynamic alpha
Phase 2 (5000-5500): Perturbation - state *= 0.2 (severe damage)
Phase 3 (5500-15000): Recovery - return to normal

METRICS:
- Recovery ratio: final_variance / pre_shock_variance
- Recovery time: steps to reach 90% of pre-shock variance
- Alpha response: max alpha during recovery

SUCCESS CRITERIA:
- Recovery ratio > 80%: System returns to original state
- Recovery ratio > 50%: Partial recovery, meaningful
- Recovery ratio < 50%: Perturbation too severe for mechanism

Usage:
    python core/system_03b_perturbation.py --mode single
    python core/system_03b_perturbation.py --mode batch
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SystemDynamicAlpha:
    """Dynamic alpha system with state tracking."""
    
    N = 20
    T = 20000  # Extended for perturbation test
    
    def __init__(self, sigma=0.5, alpha_init=0.45, desired_var=0.015,
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
        self.beta = 0.9
        
    def step(self, t):
        """Single evolution step."""
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.alpha
        
        var = np.var(self.state)
        self.history['variance'].append(var)
        self.history['alpha'].append(self.alpha)
        self.history['norm'].append(norm)
        
        # EMA variance
        if self.ema_var is None:
            self.ema_var = var
        else:
            self.ema_var = self.beta * self.ema_var + (1 - self.beta) * var
        
        # Dynamic alpha
        if t >= self.window:
            recent_var = np.mean(self.history['variance'][-self.window:])
            target_alpha = self.desired_var / (recent_var + 1e-8)
            target_alpha = np.clip(target_alpha, self.alpha_min, self.alpha_max * 2)
            self.alpha += self.gain * (target_alpha - self.alpha)
            self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)
        
        return var
    
    def apply_perturbation(self, perturbation_type='state_shrink'):
        """Apply perturbation at current state."""
        if perturbation_type == 'state_shrink':
            self.state = self.state * 0.2
        elif perturbation_type == 'alpha_depress':
            self.alpha = 0.35
        return self.state
    
    def run(self, phases=None, verbose=True):
        """
        Run with phases for perturbation test.
        
        phases: list of (start, end, config) tuples
        Example: [(0, 5000, {}), (5000, 5500, {'type': 'state_shrink'}), (5500, 20000, {})]
        """
        if phases is None:
            phases = [(0, self.T, {})]
        
        phase_info = {}
        
        for start, end, config in phases:
            phase_type = config.get('type', 'normal')
            phase_info[phase_type] = {'start': start, 'end': end}
            
            for t in range(start, min(end, self.T)):
                # Apply perturbation at start of phase if specified
                if t == start and 'perturbation' in config:
                    self.apply_perturbation(config['perturbation'])
                
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
            'phase_info': phase_info,
            'survived': final_var > 0.01
        }


def analyze_recovery(system_result, shock_time=5000, recovery_window=1000):
    """Analyze recovery after perturbation."""
    
    variance = np.array(system_result['variance_trajectory'])
    alpha = np.array(system_result['alpha_trajectory'])
    
    # Pre-shock baseline (last 500 steps before shock)
    pre_shock_var = np.mean(variance[max(0, shock_time-500):shock_time])
    pre_shock_alpha = np.mean(alpha[max(0, shock_time-500):shock_time])
    
    # Post-shock minimum
    post_shock_var = variance[shock_time:min(shock_time+100, len(variance))]
    min_var = np.min(post_shock_var)
    min_var_time = shock_time + np.argmin(post_shock_var)
    
    # Recovery (last 500 steps of run)
    final_var = np.mean(variance[-500:])
    final_alpha = np.mean(alpha[-500:])
    
    # Recovery time (when variance returns to 90% of pre-shock level)
    recovery_threshold = 0.9 * pre_shock_var
    recovery_time = None
    for i, v in enumerate(variance[shock_time:]):
        if v >= recovery_threshold:
            recovery_time = i
            break
    
    # Recovery ratio
    recovery_ratio = final_var / pre_shock_var if pre_shock_var > 0 else 0
    
    # Alpha response
    max_alpha = np.max(alpha[shock_time:])
    alpha_boost = max_alpha - pre_shock_alpha
    
    return {
        'pre_shock_variance': pre_shock_var,
        'post_shock_min_variance': min_var,
        'post_shock_min_time': min_var_time,
        'final_variance': final_var,
        'recovery_ratio': recovery_ratio,
        'recovery_time_steps': recovery_time,
        'pre_shock_alpha': pre_shock_alpha,
        'max_alpha_during_recovery': max_alpha,
        'alpha_boost': alpha_boost,
        'recovery_success': recovery_ratio > 0.8
    }


def run_single_perturbation_test(sigma=0.5, verbose=True):
    """Run single perturbation test."""
    
    print(f"\n{'='*70}")
    print(f"PERTURBATION RECOVERY TEST")
    print(f"Sigma: {sigma}")
    print(f"{'='*70}")
    
    # Three phases:
    # 1. Normal (0-5000) - reach stability
    # 2. Shock (5000-5100) - severe damage (state *= 0.2)
    # 3. Recovery (5100-15000) - return to normal
    
    phases = [
        (0, 5000, {}),                          # Normal
        (5000, 5100, {'perturbation': 'state_shrink'}),  # Shock
        (5100, 15000, {}),                      # Recovery
    ]
    
    if verbose:
        print(f"\nPhase 1 (0-5000): Normal evolution")
        print(f"Phase 2 (5000-5100): State *= 0.2 (severe damage)")
        print(f"Phase 3 (5100-15000): Recovery")
    
    system = SystemDynamicAlpha(sigma=sigma, alpha_init=0.45, 
                                desired_var=0.015, seed=42)
    result = system.run(phases=phases, verbose=False)
    
    # Analyze
    analysis = analyze_recovery(result, shock_time=5000)
    
    # Print results
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"Pre-shock variance:    {analysis['pre_shock_variance']:.4f}")
    print(f"Post-shock min:        {analysis['post_shock_min_variance']:.4f} (at t={analysis['post_shock_min_time']})")
    print(f"Final variance:        {analysis['final_variance']:.4f}")
    print(f"Recovery ratio:        {analysis['recovery_ratio']*100:.1f}%")
    print(f"Recovery time:          {analysis['recovery_time_steps'] if analysis['recovery_time_steps'] else '>9000'} steps")
    print(f"Pre-shock alpha:        {analysis['pre_shock_alpha']:.3f}")
    print(f"Max alpha during rec:   {analysis['max_alpha_during_recovery']:.3f}")
    print(f"Alpha boost:            {analysis['alpha_boost']:+.3f}")
    
    success = analysis['recovery_success']
    print(f"\nRecovery success:        {'YES' if success else 'NO'} (target: >80%)")
    
    if success:
        print(f"\n*** BREAKTHROUGH: System demonstrates TRUE self-stabilization! ***")
    else:
        print(f"\n*** Partial recovery - perturbation may be too severe ***")
    
    return result, analysis


def run_batch_perturbation_tests(sigma=0.5, n_runs=10, perturbation_type='state_shrink'):
    """Run multiple perturbation tests for robustness."""
    
    print(f"\n{'='*70}")
    print(f"BATCH PERTURBATION TESTS ({n_runs} runs)")
    print(f"Sigma: {sigma} | Perturbation: {perturbation_type}")
    print(f"{'='*70}")
    
    analyses = []
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        
        phases = [
            (0, 5000, {}),
            (5000, 5100, {'perturbation': perturbation_type}),
            (5100, 15000, {}),
        ]
        
        system = SystemDynamicAlpha(sigma=sigma, alpha_init=0.45,
                                    desired_var=0.015, seed=42+i)
        result = system.run(phases=phases, verbose=False)
        
        analysis = analyze_recovery(result, shock_time=5000)
        analysis['run'] = i + 1
        analyses.append(analysis)
        
        print(f"  Recovery: {analysis['recovery_ratio']*100:.1f}% | "
              f"Alpha boost: {analysis['alpha_boost']:+.3f}")
    
    # Summary statistics
    recovery_ratios = [a['recovery_ratio'] for a in analyses]
    alpha_boosts = [a['alpha_boost'] for a in analyses]
    recovery_times = [a['recovery_time_steps'] for a in analyses if a['recovery_time_steps']]
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Recovery ratio:  {np.mean(recovery_ratios)*100:.1f}% +/- {np.std(recovery_ratios)*100:.1f}%")
    print(f"Alpha boost:     {np.mean(alpha_boosts):.3f} +/- {np.std(alpha_boosts):.3f}")
    print(f"Recovery time:   {np.mean(recovery_times) if recovery_times else 'N/A':.0f} steps (avg)")
    print(f"Success rate:    {sum(1 for a in analyses if a['recovery_success'])}/{n_runs}")
    
    return analyses


def main():
    parser = argparse.ArgumentParser(description='Perturbation Recovery Test')
    parser.add_argument('--mode', choices=['single', 'batch', 'compare'], default='single')
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--perturbation', choices=['state_shrink', 'alpha_depress'], 
                       default='state_shrink')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYSTEM 03b: PERTURBATION RECOVERY TEST")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        result, analysis = run_single_perturbation_test(args.sigma)
        
        # Save individual result
        output = {
            'timestamp': datetime.now().isoformat(),
            'sigma': args.sigma,
            'result': result,
            'analysis': analysis
        }
        
    elif args.mode == 'batch':
        analyses = run_batch_perturbation_tests(args.sigma, args.runs, args.perturbation)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'sigma': args.sigma,
            'perturbation': args.perturbation,
            'analyses': analyses,
            'summary': {
                'recovery_ratio_mean': float(np.mean([a['recovery_ratio'] for a in analyses])),
                'recovery_ratio_std': float(np.std([a['recovery_ratio'] for a in analyses])),
                'alpha_boost_mean': float(np.mean([a['alpha_boost'] for a in analyses])),
                'success_rate': sum(1 for a in analyses if a['recovery_success']) / len(analyses)
            }
        }
    
    # Save results
    output_path = Path(__file__).parent / 'results' / 'exp_03b_perturbation.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
