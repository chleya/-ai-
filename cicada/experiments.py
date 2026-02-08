"""
Cicada Protocol Experiment Suite
"""

import numpy as np
from cicada.core import cicada_protocol, analyze_spectrum


def long_term_convergence_test(N=200, time_points=None, trials=20):
    """
    Test survival rate at different time points.
    
    Args:
        N: System size
        time_points: List of time points to test
        trials: Number of trials per setting
    
    Returns:
        dict: Survival rates at each time point
    """
    if time_points is None:
        time_points = [100, 200, 300, 400, 500, 600, 800]
    
    results = {t: {'peak': [], 'random': []} for t in time_points}
    
    for t in range(trials):
        np.random.seed(t * 999 + 42)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        W = np.random.randn(N, N) / np.sqrt(N)
        peak = None
        
        # Evolve to collect peak state
        for e in range(200):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            s = np.tanh(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
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
                s_peak = np.tanh(W_peak @ s_peak + 0.2 * s_peak + 
                               0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_peak = W_peak + 0.001 * np.outer(s_peak, s_peak)
                if np.linalg.norm(W_peak) > 10:
                    W_peak = W_peak / np.linalg.norm(W_peak) * 10
            results[tp]['peak'].append(1 if np.mean(s_peak) > 0 else 0)
            
            # Random initialization
            W_rand = np.random.randn(N, N) / np.sqrt(N)
            s_rand = np.random.randn(N)
            for e in range(tp):
                s_rand = (1 - 0.5) * s_rand + 0.5 * np.random.randn(N)
                s_rand = np.tanh(W_rand @ s_rand + 0.2 * s_rand + 
                               0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_rand = W_rand + 0.001 * np.outer(s_rand, s_rand)
                if np.linalg.norm(W_rand) > 10:
                    W_rand = W_rand / np.linalg.norm(W_rand) * 10
            results[tp]['random'].append(1 if np.mean(s_rand) > 0 else 0)
    
    # Calculate survival rates
    rates = {}
    for tp in time_points:
        rates[tp] = {
            'peak': np.mean(results[tp]['peak']) * 100,
            'random': np.mean(results[tp]['random']) * 100
        }
    
    return rates


def scalability_test(sizes=None, time_point=400, trials=10):
    """
    Test performance across different system sizes.
    
    Args:
        sizes: List of system sizes to test
        time_point: Time point to evaluate
        trials: Number of trials
    
    Returns:
        dict: Results for each size
    """
    if sizes is None:
        sizes = [200, 400, 600]
    
    results = {}
    
    for N in sizes:
        peak_succ = 0
        rand_succ = 0
        
        for t in range(trials):
            np.random.seed(t * 999 + 42)
            P = np.random.randn(N)
            P = P / np.linalg.norm(P)
            
            W = np.random.randn(N, N) / np.sqrt(N)
            peak = None
            
            # Collect peak
            for e in range(200):
                s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
                s = np.tanh(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W = W + 0.001 * np.outer(s, s)
                if np.linalg.norm(W) > 10:
                    W = W / np.linalg.norm(W) * 10
                if np.mean(s) > 0 and peak is None:
                    peak = s.copy()
            
            if peak is None:
                continue
            
            # Test
            s = peak.copy()
            for e in range(time_point):
                s = (1 - 0.5) * s + 0.5 * np.random.randn(N)
                s = np.tanh(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W = W + 0.001 * np.outer(s, s)
                if np.linalg.norm(W) > 10:
                    W = W / np.linalg.norm(W) * 10
            if np.mean(s) > 0:
                peak_succ += 1
            
            W_rand = np.random.randn(N, N) / np.sqrt(N)
            s = np.random.randn(N)
            for e in range(time_point):
                s = (1 - 0.5) * s + 0.5 * np.random.randn(N)
                s = np.tanh(W_rand @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_rand = W_rand + 0.001 * np.outer(s, s)
                if np.linalg.norm(W_rand) > 10:
                    W_rand = W_rand / np.linalg.norm(W_rand) * 10
            if np.mean(s) > 0:
                rand_succ += 1
        
        results[N] = {
            'peak': peak_succ * 100 // trials,
            'random': rand_succ * 100 // trials
        }
    
    return results


def byzantine_test(attack_rates=None, N=200, trials=15):
    """
    Test robustness against Byzantine attacks.
    
    Args:
        attack_rates: List of attack ratios to test
        N: System size
        trials: Number of trials
    
    Returns:
        dict: Survival rates at each attack level
    """
    if attack_rates is None:
        attack_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    results = {rate: [] for rate in attack_rates}
    
    for t in range(trials):
        np.random.seed(t * 555 + 42)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        W = np.random.randn(N, N) / np.sqrt(N)
        
        # Warm up
        for i in range(30):
            xg = np.tanh(W @ P + np.random.randn(N) * 0.3)
            W = W + 0.001 * np.outer(xg, xg)
            if np.linalg.norm(W) > 10:
                W = W / np.linalg.norm(W) * 10
        
        for rate in attack_rates:
            W_attack = W.copy()
            n_malicious = int(N * rate)
            
            # Evolution with attacks
            for e in range(400):
                s = np.random.randn(N)
                indices = list(range(N))
                np.random.shuffle(indices)
                malicious = set(indices[:n_malicious])
                
                for i in range(N):
                    if i in malicious:
                        s[i] = -P[i] * 2  # Byzantine signal
                
                s = np.tanh(W_attack @ s + 0.2 * s + 
                           0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_attack = W_attack + 0.001 * np.outer(s, s)
                if np.linalg.norm(W_attack) > 10:
                    W_attack = W_attack / np.linalg.norm(W_attack) * 10
            
            # Test survival
            success = 0
            for e in range(200):
                s = np.random.randn(N)
                s = np.tanh(W_attack @ s + 0.2 * s + 
                           0.05 * np.sin(2 * np.pi * 0.1 * e / 100))
                W_attack = W_attack + 0.001 * np.outer(s, s)
                if np.linalg.norm(W_attack) > 10:
                    W_attack = W_attack / np.linalg.norm(W_attack) * 10
                if np.mean(s) > 0:
                    success += 1
            
            results[rate].append(success)
    
    # Calculate rates
    rates = {}
    for rate in attack_rates:
        rates[rate] = np.mean(results[rate])
    
    return rates


if __name__ == "__main__":
    print("Cicada Protocol Experiments")
    print("=" * 60)
    
    # Long-term test
    print("\n1. Long-term Convergence Test")
    print("-" * 40)
    rates = long_term_convergence_test(N=200, trials=10)
    for tp, r in rates.items():
        print(f"  t={tp}: Peak={r['peak']:.0f}% Random={r['random']:.0f}%")
    
    # Scalability test
    print("\n2. Scalability Test")
    print("-" * 40)
    results = scalability_test(sizes=[200, 400, 600], trials=5)
    for N, r in results.items():
        print(f"  N={N}: Peak={r['peak']}% Random={r['random']}%")
    
    # Byzantine test
    print("\n3. Byzantine Attack Test")
    print("-" * 40)
    rates = byzantine_test(N=200, trials=10)
    for rate, r in rates.items():
        print(f"  {rate*100:.0f}% attack: {r:.1f}%")
