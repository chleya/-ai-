"""
Core Cicada Protocol implementation
"""

import numpy as np


class CicadaProtocol:
    """
    The Cicada Protocol: Reset as Entropy Injection for Stabilizing Hebbian Learning
    
    Attributes:
        N: System size
        theta: Stability threshold (default: 1.2)
        alpha: Sensitivity coefficient (default: 1.0)
        lambda_max: Current maximum eigenvalue
        reset_count: Number of resets triggered
    """
    
    def __init__(self, N=787, theta=1.2, alpha=1.0):
        """
        Initialize Cicada Protocol
        
        Args:
            N: System size (default: 787, phase transition point)
            theta: Stability threshold (default: 1.2)
            alpha: Sensitivity coefficient (default: 1.0)
        """
        self.N = N
        self.theta = theta
        self.alpha = alpha
        self.lambda_max = 0.0
        self.reset_count = 0
        self.history = []
    
    def hebbian_update(self, W, s, eta=0.01):
        """
        Hebbian update: W += eta * s * s^T
        
        Args:
            W: Weight matrix
            s: Input pattern
            eta: Learning rate
            
        Returns:
            Updated weight matrix
        """
        return W + eta * np.outer(s, s)
    
    def compute_lambda_max(self, W, iterations=10):
        """
        Compute maximum eigenvalue using power iteration
        
        Args:
            W: Weight matrix
            iterations: Number of power iterations
            
        Returns:
            Maximum eigenvalue
        """
        v = np.random.rand(W.shape[0])
        for _ in range(iterations):
            v = W @ v
            v /= np.linalg.norm(v)
        return np.abs(v @ (W @ v))
    
    def approximate_entropy(self, W, epsilon=1e-6):
        """
        Approximate von Neumann entropy using log det
        
        Args:
            W: Weight matrix
            epsilon: Small value for numerical stability
            
        Returns:
            Approximate entropy
        """
        try:
            return np.log(np.abs(np.linalg.det(W + np.eye(W.shape[0]) * epsilon))) / W.shape[0]
        except:
            return 0.0
    
    def step(self, W, s, eta=0.01):
        """
        One step of Cicada Protocol
        
        Args:
            W: Current weight matrix
            s: Input pattern
            eta: Learning rate
            
        Returns:
            Updated weight matrix
        """
        # Hebbian update
        W_new = self.hebbian_update(W, s, eta)
        
        # Compute lambda max
        self.lambda_max = self.compute_lambda_max(W_new)
        
        # Check reset condition
        threshold = self.theta * self.alpha
        if self.lambda_max > threshold:
            W_new = self.reset(W_new)
        
        # Record history
        self.history.append({
            'lambda_max': self.lambda_max,
            'entropy': self.approximate_entropy(W_new),
            'reset': self.reset_count
        })
        
        return W_new
    
    def reset(self, W):
        """
        Reset weight matrix (inject negative entropy)
        
        Args:
            W: Current weight matrix
            
        Returns:
            Reset weight matrix
        """
        self.reset_count += 1
        # Partial reset (5% of weights)
        mask = np.random.rand(W.shape[0], W.shape[1]) < 0.05
        W[mask] = np.random.randn(W.shape[0], W.shape[1])[mask] / np.sqrt(W.shape[0])
        return W
    
    def get_stats(self):
        """
        Get protocol statistics
        
        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {'lambda_max': 0, 'entropy': 0, 'resets': 0}
        
        return {
            'lambda_max': np.mean([h['lambda_max'] for h in self.history[-10:]]),
            'entropy': np.mean([h['entropy'] for h in self.history[-10:]]),
            'resets': self.reset_count,
            'lambda_growth': self.history[-1]['lambda_max'] - self.history[0]['lambda_max'] if len(self.history) > 1 else 0
        }
    
    def summary(self):
        """Print protocol summary"""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"Cicada Protocol Summary")
        print(f"{'='*50}")
        print(f"System Size (N): {self.N}")
        print(f"Threshold (theta): {self.theta}")
        print(f"Sensitivity (alpha): {self.alpha}")
        print(f"Resets Triggered: {stats['resets']}")
        print(f"Avg Lambda Max: {stats['lambda_max']:.4f}")
        print(f"Lambda Growth: {stats['lambda_growth']:.4f}")
        print(f"{'='*50}\n")
