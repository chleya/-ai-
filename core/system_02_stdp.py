"""
System 02: Weak STDP Constraint for Self-Organization

Research Question: Can minimal STDP give system self-adjustment capability?

Core Mechanism:
- Every 50 steps, strengthen active connections (activity-based)
- Very conservative learning rate (1e-4)
- No decay (first prove positive effect exists)

Usage:
    python core/system_02_stdp.py --mode single --sigma 0.11 --alpha 1.0
    python core/system_02_stdp.py --mode phase --sigma-min 0.01 --sigma-max 1.0
    python core/system_02_stdp.py --mode compare --sigma 0.11 --alpha 1.0
"""

import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import json


class System02_STDP:
    """
    Constrained dynamical system with weak STDP self-adjustment.
    
    Evolution rule:
        state_{t+1} = tanh(W · state_t + noise)
        state_{t+1} = norm_strength * state_{t+1} / ||state_{t+1}||
        
    STDP rule (weak, activity-based):
        Every STDP_INTERVAL steps:
        W[active] += lr * activity[active]
        Then renormalize W
    """
    
    # === CONFIGURATION ===
    N: int = 20              # State dimension
    T: int = 10000           # Evolution steps
    STDP_INTERVAL: int = 50  # Apply STDP every 50 steps
    LR: float = 1e-4         # Very conservative learning rate
    ACTIVITY_THRESHOLD: float = 0.1  # Connections above this get boosted
    
    def __init__(self, sigma: float, norm_strength: float, seed: int = 42):
        self.sigma = sigma          # Noise level
        self.norm_strength = norm_strength  # Constraint strength
        self.seed = seed
        
        # Initialize state X ~ N(0, 1)
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        
        # Initialize weight matrix W (random, spectral radius ~1)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        
        # Track history for analysis
        self.history = {
            'states': [],
            'variance': [],
            'norm': [],
            'W_evolution': [],
            'activity': []
        }
        
        # Activity tracking
        self.activity_history = np.zeros((self.N,))
        
    def step(self, t: int) -> float:
        """
        Single evolution step.
        Returns current variance.
        """
        # Evolution: state_{t+1} = tanh(W · state_t + noise)
        noise = np.random.randn(self.N) * self.sigma
        self.state = np.tanh(self.W @ self.state + noise)
        
        # Constraint: normalize to fixed norm
        norm = np.linalg.norm(self.state)
        if norm > 1e-8:
            self.state = (self.state / norm) * self.norm_strength
        
        # Track variance (proxy for "alive")
        var = np.var(self.state)
        
        # Update activity (exponential moving average)
        activity = np.abs(self.state)
        self.activity_history = 0.9 * self.activity_history + 0.1 * activity
        
        return var
    
    def apply_stdp(self, t: int):
        """
        Apply weak STDP: strengthen active connections.
        
        Mechanism:
        - Identify connections where both pre and post are active
        - Apply weak positive strengthening
        - Renormalize W to preserve spectral properties
        """
        # Compute connection activity: outer product of state
        connection_activity = np.outer(self.activity_history, self.activity_history)
        
        # Find active connections (above threshold)
        active_mask = connection_activity > self.ACTIVITY_THRESHOLD
        
        # Apply STDP update (very weak)
        W_update = np.zeros_like(self.W)
        W_update[active_mask] = self.LR * connection_activity[active_mask]
        
        self.W += W_update
        
        # Renormalize W to preserve spectral radius ~1
        # This prevents explosion while allowing structure evolution
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        if spectral_radius > 1e-8:
            self.W = self.W / spectral_radius
        
        # Track W evolution
        self.history['W_evolution'].append(self.W.copy())
    
    def run(self, verbose: bool = True) -> dict:
        """
        Run full evolution.
        Returns statistics.
        """
        np.random.seed(self.seed)
        
        if verbose:
            print(f"=== System 02: σ={self.sigma}, α={self.norm_strength} ═══")
        
        for t in range(self.T):
            var = self.step(t)
            self.history['variance'].append(var)
            self.history['norm'].append(np.linalg.norm(self.state))
            
            # Progress output
            if verbose and t % 1000 == 0:
                print(f"  [{t:5d}/{self.T}] var={var:.4f}, norm={self.norm_strength:.2f}")
            
            # Apply STDP at intervals
            if (t + 1) % self.STDP_INTERVAL == 0:
                self.apply_stdp(t)
                if verbose and (t + 1) % 1000 == 0:
                    W_activity = np.mean(self.activity_history)
                    print(f"  [STDP @ t={t+1}] mean activity={W_activity:.4f}")
        
        # Final statistics
        final_var = np.mean(self.history['variance'][-1000:])
        final_std = np.std(self.history['variance'][-1000:])
        
        result = {
            'sigma': self.sigma,
            'norm_strength': self.norm_strength,
            'final_variance': final_var,
            'variance_stability': final_std,
            'survived': final_var > 0.01,
            'activity_mean': np.mean(self.activity_history),
            'W_frobenius': np.linalg.norm(self.W, 'fro'),
            'W_spectral': np.max(np.abs(np.linalg.eigvals(self.W)))
        }
        
        if verbose:
            print(f"  Survived: {'YES' if result['survived'] else 'NO'} | "
                  f"variance: {final_var:.4f} +- {final_std:.4f}")
            print(f"  W spectral radius: {result['W_spectral']:.4f}")
        
        return result


def run_single_point(sigma: float, alpha: float, verbose: bool = True):
    """Run single point comparison."""
    sys02 = System02_STDP(sigma=sigma, norm_strength=alpha, seed=42)
    result = sys02.run(verbose=verbose)
    return result


def run_phase_scan(sigma_min: float, sigma_max: float, alpha: float = 1.0, 
                   num_points: int = 20, repeats: int = 3, verbose: bool = False):
    """
    Run phase space scan with STDP.
    
    Returns:
        survival_map: 2D array (sigma vs repeat)
        variance_map: 2D array (sigma vs repeat)
    """
    sigma_values = np.linspace(sigma_min, sigma_max, num_points)
    
    survival_map = np.zeros((num_points, repeats))
    variance_map = np.zeros((num_points, repeats))
    
    for i, sigma in enumerate(sigma_values):
        for r in range(repeats):
            sys02 = System02_STDP(sigma=sigma, norm_strength=alpha, seed=42 + r)
            result = sys02.run(verbose=False)
            
            survival_map[i, r] = 1 if result['survived'] else 0
            variance_map[i, r] = result['final_variance']
            
            if verbose:
                print(f"sigma={sigma:.3f} [{r+1}/{repeats}]: "
                      f"{'YES' if result['survived'] else 'NO'} var={result['final_variance']:.4f}")
    
    return sigma_values, survival_map, variance_map


def compare_with_baseline(sigma: float, alpha: float, verbose: bool = True):
    """
    Compare System 01 (no STDP) vs System 02 (with STDP).
    Inline baseline to avoid import issues.
    """
    # Baseline: System 01 (no STDP) - inline implementation
    class BaselineSystem:
        N = 20
        T = 10000
        
        def __init__(self, sigma, norm_strength, seed=42):
            np.random.seed(seed)
            self.state = np.random.randn(self.N)
            self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
            self.sigma = sigma
            self.norm_strength = norm_strength
            self.history = []
            
        def run(self):
            for t in range(self.T):
                noise = np.random.randn(self.N) * self.sigma
                self.state = np.tanh(self.W @ self.state + noise)
                norm = np.linalg.norm(self.state)
                if norm > 1e-8:
                    self.state = (self.state / norm) * self.norm_strength
                self.history.append(np.var(self.state))
            
            final_var = np.mean(self.history[-1000:])
            final_std = np.std(self.history[-1000:])
            return {'final_variance': final_var, 'survived': final_var > 0.01}
    
    if verbose:
        print("-" * 50)
        print("BASELINE: System 01 (No STDP)")
        print("-" * 50)
    
    baseline_sys = BaselineSystem(sigma=sigma, norm_strength=alpha, seed=42)
    baseline = baseline_sys.run()
    
    if verbose:
        survived_mark = "YES" if baseline['survived'] else "NO"
        print(f"  Survived: {survived_mark} | variance={baseline['final_variance']:.4f}")
    
    if verbose:
        print("\n" + "━" * 50)
        print("TEST: System 02 (With STDP)")
        print("━" * 50)
    
    sys02 = System02_STDP(sigma=sigma, norm_strength=alpha, seed=42)
    test = sys02.run(verbose=verbose)
    
    # Comparison
    improvement = (test['final_variance'] - baseline['final_variance']) / max(baseline['final_variance'], 1e-6) * 100
    
    comparison = {
        'baseline_survived': baseline['survived'],
        'test_survived': test['survived'],
        'baseline_variance': baseline['final_variance'],
        'test_variance': test['final_variance'],
        'variance_improvement_%': improvement,
        'success': test['survived'] and (improvement > 0)
    }
    
    if verbose:
        print("\n" + "═" * 50)
        print("COMPARISON RESULT")
        print("═" * 50)
        print(f"Baseline (No STDP):  variance={baseline['final_variance']:.4f}")
        print(f"Test (With STDP):    variance={test['final_variance']:.4f}")
        print(f"Improvement:         {improvement:+.2f}%")
        print(f"STDP Effect:         {'YES POSITIVE' if comparison['success'] else 'NO No/Negative'}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='System 02: Weak STDP Experiment')
    parser.add_argument('--mode', choices=['single', 'phase', 'compare'], 
                       default='single', help='Run mode')
    parser.add_argument('--sigma', type=float, default=0.11, help='Noise level')
    parser.add_argument('--alpha', type=float, default=1.0, help='Norm strength')
    parser.add_argument('--sigma-min', type=float, default=0.01, help='Min sigma for phase scan')
    parser.add_argument('--sigma-max', type=float, default=1.0, help='Max sigma for phase scan')
    parser.add_argument('--num-points', type=int, default=20, help='Number of sigma points')
    parser.add_argument('--repeats', type=int, default=3, help='Repeats per point')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"System 02: Weak STDP Self-Organization Experiment")
    print(f"Mode: {args.mode} | Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    if args.mode == 'single':
        result = run_single_point(args.sigma, args.alpha)
        print(f"\nResult: variance={result['final_variance']:.4f}")
        
    elif args.mode == 'phase':
        print(f"Phase Scan: σ ∈ [{args.sigma_min}, {args.sigma_max}] × {args.num_points} points")
        print(f"STDP enabled: interval={System02_STDP.STDP_INTERVAL}, lr={System02_STDP.LR}")
        sigma_values, survival, variance = run_phase_scan(
            args.sigma_min, args.sigma_max, args.alpha, 
            args.num_points, args.repeats
        )
        
        # Save results
        results = {
            'sigma_values': sigma_values.tolist(),
            'survival_rate': np.mean(survival, axis=1).tolist(),
            'variance_mean': np.mean(variance, axis=1).tolist(),
            'stability_std': np.std(variance, axis=1).tolist()
        }
        
        output_path = Path(__file__).parent / 'results' / 'system_02_phase_results.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved: {output_path}")
        print(f"Survival rate range: {np.min(np.mean(survival, axis=1)):.2%} - {np.max(np.mean(survival, axis=1)):.2%}")
        
    elif args.mode == 'compare':
        comparison = compare_with_baseline(args.sigma, args.alpha)
        print(f"\nConclusion: {'YES STDP helps!' if comparison['success'] else 'NO No improvement'}")


if __name__ == '__main__':
    main()
