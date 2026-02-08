"""
Cicada Protocol Core Implementation
"""

import numpy as np


def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)


def cicada_protocol(N=200, reset_interval=300, total_steps=800, seed=None):
    """
    Cicada Protocol Main Loop
    
    A periodic reset mechanism for maintaining long-term stability
    in edge computing consensus systems.
    
    Args:
        N: System size (number of nodes)
        reset_interval: Reset interval in steps
        total_steps: Total evolution steps
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (W, s) Final weight matrix and system state
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
    W = np.random.randn(N, N) / np.sqrt(N)
    s = np.random.randn(N)
    
    for t in range(total_steps):
        # Normal evolution
        s = (1 - 0.5) * s + 0.5 * np.random.randn(N)
        s = tanh(W @ s + 0.2 * s + 0.05 * np.sin(2 * np.pi * 0.1 * t / 100))
        W = W + 0.001 * np.outer(s, s)
        
        # Normalize to prevent numerical explosion
        if np.linalg.norm(W) > 10:
            W = W / np.linalg.norm(W) * 10
        
        # Periodic reset (Cicada moment)
        if (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) / np.sqrt(N)
            s = np.random.randn(N)
    
    return W, s


def analyze_spectrum(W):
    """
    Analyze the spectral properties of a weight matrix.
    
    Args:
        W: Weight matrix
    
    Returns:
        dict: Spectral analysis results
    """
    eigenvalues = np.linalg.eigvalsh(W)
    
    return {
        'max': eigenvalues[-1],  # Largest eigenvalue
        'min': eigenvalues[0],   # Smallest eigenvalue
        'ratio': eigenvalues[-1] / (abs(eigenvalues[0]) + 1e-6),
        'spread': eigenvalues[-1] - eigenvalues[0],
        'mean': np.mean(eigenvalues),
        'std': np.std(eigenvalues)
    }


def run_experiment(N=200, reset_interval=300, total_steps=800, trials=10):
    """
    Run multiple trials of the Cicada Protocol.
    
    Args:
        N: System size
        reset_interval: Reset interval
        total_steps: Total steps
        trials: Number of trials
    
    Returns:
        dict: Experiment results
    """
    successes = 0
    
    for _ in range(trials):
        W, s = cicada_protocol(N, reset_interval, total_steps)
        if np.mean(s) > 0:  # Consensus achieved
            successes += 1
    
    return {
        'success_rate': successes / trials,
        'successes': successes,
        'trials': trials
    }


def reset_frequency_experiment(N=200, total_steps=800, trials=10):
    """
    Experiment to find optimal reset frequency.
    
    Args:
        N: System size
        total_steps: Total steps
        trials: Number of trials per setting
    
    Returns:
        dict: Results for each reset interval
    """
    intervals = [100, 200, 300, 400, 500]
    results = {}
    
    for interval in intervals:
        interval_results = run_experiment(
            N=N,
            reset_interval=interval,
            total_steps=total_steps,
            trials=trials
        )
        results[interval] = interval_results['success_rate']
    
    return results


if __name__ == "__main__":
    # Quick demo
    print("Cicada Protocol Demo")
    print("=" * 50)
    
    # Run protocol
    W, s = cicada_protocol(N=200, reset_interval=300, total_steps=800)
    
    # Analyze spectrum
    spectrum = analyze_spectrum(W)
    print(f"Spectral Analysis:")
    print(f"  λ_max: {spectrum['max']:.4f}")
    print(f"  λ_ratio: {spectrum['ratio']:.4f}")
    
    # Check consensus
    consensus = np.mean(s) > 0
    print(f"\nConsensus Achieved: {consensus}")
    print(f"Mean State Value: {np.mean(s):.4f}")
