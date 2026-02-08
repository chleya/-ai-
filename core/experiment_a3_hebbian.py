"""
Experiment A3: Hebbian Learning for Multiple Attractors

Goal: Create multiple attractors through Hebbian learning

Setup:
- x_{t+1} = φ(W x_t + noise)
- Hebbian update: W += η * x_t x_t^T (normalized)
- Target: Create energy minima at specific patterns

Usage:
    python core/experiment_a3_hebbian.py --mode scan
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path


class HebbianSystem:
    """
    System with Hebbian learning.
    
    Dynamics:
        x_{t+1} = φ(W x_t + noise)
        W_{t+1} = normalize(W_t + η * x_t x_t^T)
    
    Goal: Create attractors at Hebbian-assocated patterns.
    """
    
    T = 30000
    
    def __init__(self, N=50, phi='tanh', eta=0.001, sigma=0.5,
                 seed=42, memory_patterns=3):
        
        np.random.seed(seed)
        self.N = N
        self.x = np.random.randn(N) * 0.1
        self.W = np.random.randn(N, N) / np.sqrt(N)
        self.sigma = sigma
        self.eta = eta
        self.phi = phi
        self.memory_patterns = memory_patterns
        
        # Tracking
        self.history = {'variance': [], 'energy': [], 'W_norm': [], 'x_history': []}
        
    def nonlinearity(self, x):
        if self.phi == 'tanh':
            return np.tanh(x)
        elif self.phi == 'relu':
            return np.maximum(0, x)
        elif self.phi == 'swish':
            return x / (1 + np.exp(-np.clip(x, -500, 500)))
        return np.tanh(x)
    
    def normalize_W(self):
        """Normalize W to prevent explosion"""
        norm = np.linalg.norm(self.W)
        if norm > 10:
            self.W = self.W / norm * 10
    
    def step(self, t):
        """Single step with Hebbian update."""
        # Evolution
        noise = np.random.randn(self.N) * self.sigma
        x_new = self.W @ self.x + noise
        self.x = self nonlinearity(x_new)
        
        # Track before update
        var = np.var(self.x)
        energy = np.linalg.norm(self.W @ self.x) / np.sqrt(self.N)
        W_norm = np.linalg.norm(self.W)
        
        self.history['variance'].append(var)
        self.history['energy'].append(energy)
        self.history['W_norm'].append(W_norm)
        
        # Hebbian update every 10 steps
        if t % 10 == 0 and t > 1000:
            # Outer product update
            self.W += self.eta * np.outer(self.x, self.x)
            self.normalize_W()
        
        return var
    
    def run(self, verbose=True):
        """Full evolution."""
        for t in range(self.T):
            var = self.step(t)
            
            if verbose and t % 5000 == 0:
                print(f"  [t={t:5d}] var={var:.4f}, energy={self.history['energy'][-1]:.4f}")
        
        return self.analyze()
    
    def analyze(self):
        """Analyze results."""
        variance = np.array(self.history['variance'])
        energy = np.array(self.history['energy'])
        W_norm = np.array(self.history['W_norm'])
        
        # Stability
        late_var = np.mean(variance[-5000:])
        early_var = np.mean(variance[:5000])
        trend = (late_var - early_var) / max(early_var, 1e-6)
        
        # Energy landscape
        late_energy = np.mean(energy[-5000:])
        energy_change = late_energy - np.mean(energy[:5000])
        
        # Check for multiple attractors
        # Run from different initial conditions
        test_starts = [np.random.randn(self.N) for _ in range(5)]
        final_states = []
        
        for x0 in test_starts:
            x = x0.copy()
            for _ in range(1000):
                x = self.nonlinearity(self.W @ x)
            final_states.append(np.sign(x))
        
        # Measure separation of final states
        if len(final_states) > 1:
            separations = []
            for i in range(len(final_states)):
                for j in range(i+1, len(final_states)):
                    sep = np.mean(np.abs(final_states[i] - final_states[j]))
                    separations.append(sep)
            mean_separation = np.mean(separations)
        else:
            mean_separation = 0
        
        return {
            'N': self.N,
            'phi': self.phi,
            'eta': self.eta,
            'final_variance': float(late_var),
            'trend': float(trend),
            'final_energy': float(late_energy),
            'energy_change': float(energy_change),
            'W_norm': float(np.mean(W_norm[-1000:])),
            'mean_separation': float(mean_separation),
            'multiple_attractors': mean_separation > 0.1,
            'trajectories': {
                'variance': self.history['variance'],
                'energy': self.history['energy'],
                'W_norm': self.history['W_norm']
            }
        }


def run_eta_scan(N=50, phi='tanh', etas=[0.0001, 0.001, 0.01], verbose=True):
    """Scan different learning rates."""
    
    results = {}
    
    for eta in etas:
        key = f"eta_{eta}"
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Learning rate = {eta}")
            print(f"{'='*50}")
        
        system = HebbianSystem(N=N, phi=phi, eta=eta, seed=42)
        r = system.run(verbose=verbose)
        
        results[key] = {
            'eta': eta,
            'final_variance': r['final_variance'],
            'trend': r['trend'],
            'final_energy': r['final_energy'],
            'energy_change': r['energy_change'],
            'mean_separation': r['mean_separation'],
            'multiple_attractors': r['multiple_attractors']
        }
        
        if verbose:
            print(f"\n  Final variance: {r['final_variance']:.4f}")
            print(f"  Final energy:  {r['final_energy']:.4f}")
            print(f"  Separation:   {r['mean_separation']:.4f}")
            print(f"  Multi-attractor: {r['multiple_attractors']}")
    
    return results


def analyze_hebbian(results, verbose=True):
    """Analyze Hebbian learning results."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("HEBBIAN LEARNING ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n{'Eta':<10} {'Variance':<12} {'Energy':<12} {'Separation':<12} {'Multi-A'}")
        print("-" * 60)
        
        for key, r in sorted(results.items(), key=lambda x: float(x[1]['eta'])):
            print(f"{r['eta']:<10.4f} {r['final_variance']:<12.4f} {r['final_energy']:<12.4f} "
                  f"{r['mean_separation']:<12.4f} {r['multiple_attractors']}")
        
        # Check if any created multiple attractors
        multi = any(r['multiple_attractors'] for r in results.values())
        
        print()
        if multi:
            print("*** HEBBIAN LEARNING CREATED MULTIPLE ATTRACTORS! ***")
        else:
            print("*** NO MULTIPLE ATTRACTORS: All converged to single state ***")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment A3: Hebbian Learning')
    parser.add_argument('--mode', choices=['single', 'scan'], default='scan')
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--phi', choices=['tanh', 'relu', 'swish'], default='tanh')
    parser.add_argument('--eta', type=float, default=0.001)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT A3: HEBBIAN LEARNING FOR MULTIPLE ATTRACTORS")
    print(f"Mode: {args.mode} | N: {args.N} | φ: {args.phi}")
    print(f"{'='*70}")
    
    if args.mode == 'single':
        system = HebbianSystem(N=args.N, phi=args.phi, eta=args.eta, seed=42)
        result = system.run(verbose=True)
        
        print(f"\nFinal variance: {result['final_variance']:.4f}")
        print(f"Multiple attractors: {result['multiple_attractors']}")
        
        results = {f"eta_{args.eta}": result}
    
    elif args.mode == 'scan':
        results = run_eta_scan(
            N=args.N,
            phi=args.phi,
            etas=[0.0001, 0.001, 0.01],
            verbose=True
        )
    
    # Analysis
    analyze_hebbian(results, verbose=True)
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'N': args.N,
            'phi': args.phi
        },
        'results': results
    }
    
    output_path = Path(__file__).parent / 'results' / 'exp_A3_hebbian.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
