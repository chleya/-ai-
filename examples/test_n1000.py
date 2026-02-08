"""
N=1000 Scalability Test
========================

This script tests the Cicada Protocol at N=1000 scale.
"""

import numpy as np


def nl(x):
    return np.tanh(x)


def quick_test(N=1000, time_points=[100, 200, 300, 400], trials=5):
    """
    Quick scalability test for given N.
    """
    results = {tp: {'peak': 0, 'random': 0} for tp in time_points}
    
    for t in range(trials):
        np.random.seed(t * 999 + 42)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        W = np.random.randn(N, N) / np.sqrt(N)
        peak = None
        
        # Collect peak state
        for e in range(200):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            s = nl(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
            W = W + 0.001 * np.outer(s, s)
            if np.linalg.norm(W) > 10:
                W = W / np.linalg.norm(W) * 10
            if np.mean(s) > 0 and peak is None:
                peak = s.copy()
        
        if peak is None:
            continue
        
        # Test at each time point
        for tp in time_points:
            # Peak initialization
            s_peak = peak.copy()
            W_peak = W.copy()
            for e in range(tp):
                s_peak = (1 - 0.5) * s_peak + 0.5 * np.random.randn(N)
                s_peak = nl(W_peak @ s_peak + 0.2 * s_peak + 
                           0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_peak = W_peak + 0.001 * np.outer(s_peak, s_peak)
                if np.linalg.norm(W_peak) > 10:
                    W_peak = W_peak / np.linalg.norm(W_peak) * 10
            if np.mean(s_peak) > 0:
                results[tp]['peak'] += 1
            
            # Random initialization
            W_rand = np.random.randn(N, N) / np.sqrt(N)
            s_rand = np.random.randn(N)
            for e in range(tp):
                s_rand = (1 - 0.5) * s_rand + 0.5 * np.random.randn(N)
                s_rand = nl(W_rand @ s_rand + 0.2 * s_rand +
                           0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_rand = W_rand + 0.001 * np.outer(s_rand, s_rand)
                if np.linalg.norm(W_rand) > 10:
                    W_rand = W_rand / np.linalg.norm(W_rand) * 10
            if np.mean(s_rand) > 0:
                results[tp]['random'] += 1
    
    # Calculate rates
    rates = {}
    for tp in time_points:
        rates[tp] = {
            'peak': results[tp]['peak'] * 100 // trials,
            'random': results[tp]['random'] * 100 // trials
        }
    
    return rates


def spectral_test(N=1000, trials=3):
    """
    Test spectral properties at N=1000.
    """
    eigenvals = []
    
    for t in range(trials):
        np.random.seed(t * 777 + 42)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        W = np.random.randn(N, N) / np.sqrt(N)
        
        # Evolve
        for e in range(400):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            s = nl(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
            W = W + 0.001 * np.outer(s, s)
            if np.linalg.norm(W) > 10:
                W = W / np.linalg.norm(W) * 10
        
        # Analyze spectrum
        eigvals = np.linalg.eigvalsh(W)
        eigenvals.append({
            'max': eigvals[-1],
            'ratio': eigvals[-1] / (abs(eigvals[0]) + 1e-6)
        })
    
    return {
        'max': np.mean([e['max'] for e in eigenvals]),
        'ratio': np.mean([e['ratio'] for e in eigenvals])
    }


if __name__ == "__main__":
    print("N=1000 Scalability Test")
    print("=" * 60)
    
    # Quick survival test
    print("\n1. Survival Rate Test")
    rates = quick_test(N=1000, time_points=[100, 200, 300, 400], trials=3)
    for tp, r in rates.items():
        print(f"  t={tp}: Peak={r['peak']}% Random={r['random']}%")
    
    # Spectral test
    print("\n2. Spectral Properties")
    spectrum = spectral_test(N=1000, trials=3)
    print(f"  λ_max: {spectrum['max']:.4f}")
    print(f"  λ_ratio: {spectrum['ratio']:.4f}")
    
    # Comparison
    print("\n3. Comparison with N=200")
    print("  N=200: λ_max=1.73, λ_ratio=1.48")
    print(f"  N=1000: λ_max={spectrum['max']:.4f}, λ_ratio={spectrum['ratio']:.4f}")
