"""
Sparse Hebbian Experiment: 80% Sparsity vs Interference Rate
Based on FINAL_REPORT.md (System Stability)
"""

import numpy as np, json

def nonlinearity(x):
    return np.tanh(x)

def create_sparse_W(W, sparsity):
    """Create sparse version by zeroing small connections"""
    W_sparse = W.copy()
    threshold = np.percentile(np.abs(W), sparsity * 100)
    W_sparsity_mask = np.abs(W) >= threshold
    W_sparse[~W_sparsity_mask] = 0
    return W_sparse

def create_correlated_patterns(p1, corr_target):
    """Create P2 with given correlation to P1"""
    N = len(p1)
    p2 = corr_target * p1 + np.sqrt(1-corr_target**2) * np.random.randn(N)
    p2 = p2 / np.linalg.norm(p2)
    return p2

def train_W_on_patterns(N, patterns, steps=15000):
    """Train W on list of patterns"""
    W = np.random.randn(N, N) / np.sqrt(N)
    eta = 0.001
    
    for _ in range(steps):
        for p in patterns:
            noise = np.random.randn(N) * 0.2
            x = nonlinearity(W @ p + noise)
            W += eta * np.outer(x, x)
            if np.linalg.norm(W) > 10:
                W = W / np.linalg.norm(W) * 10
    
    return W

def test_interference(W, p1, p2, n_trials=20):
    """Test if patterns interfere"""
    interfere_count = 0
    
    for seed in range(n_trials):
        np.random.seed(seed)
        # Test P1
        x1 = p1.copy() + np.random.randn(len(p1)) * 0.3
        for _ in range(300):
            x1 = nonlinearity(W @ x1)
        r1 = np.sign(np.mean(x1))
        
        # Test P2
        x2 = p2.copy() + np.random.randn(len(p2)) * 0.3
        for _ in range(300):
            x2 = nonlinearity(W @ x2)
        r2 = np.sign(np.mean(x2))
        
        if r1 == r2:
            interfere_count += 1
    
    return interfere_count / n_trials

print('='*70)
print('SPARSE HEBBIAN EXPERIMENT: 80% SPARSITY')
print('Target: Reduce 71% interference rate')
print('='*70)

N = 50
sparsity_levels = [0.0, 0.5, 0.8, 0.9]
n_pattern_pairs = 5

print()
print('PART 1: Baseline (Full Connection)')
print('-'*50)

# Create random pattern pairs
pattern_pairs = []
for i in range(n_pattern_pairs):
    np.random.seed(i * 100)
    p1 = np.random.randn(N)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.random.randn(N)  # ~orthogonal
    p2 = p2 / np.linalg.norm(p2)
    pattern_pairs.append((p1, p2))

# Test full connection
full_interference = []
for p1, p2 in pattern_pairs:
    W = train_W_on_patterns(N, [p1, p2])
    rate = test_interference(W, p1, p2)
    full_interference.append(rate)

print('Pattern pairs: %d' % n_pattern_pairs)
print('Full connection interference rate: %.0f%%' % (np.mean(full_interference) * 100))

print()
print('PART 2: Sparse Connection Tests')
print('-'*50)

results = {}

for sparsity in sparsity_levels:
    sparse_interference = []
    
    for p1, p2 in pattern_pairs:
        W = train_W_on_patterns(N, [p1, p2])
        W_sparse = create_sparse_W(W, sparsity)
        
        rate = test_interference(W_sparse, p1, p2)
        sparse_interference.append(rate)
    
    results[sparsity] = np.mean(sparse_interference)
    
    reduction = (np.mean(full_interference) - np.mean(sparse_interference)) / np.mean(full_interference) * 100
    
    print('Sparsity %.0f%%: Interference %.0f%% (Reduction: %.0f%%)' % 
          (sparsity * 100, np.mean(sparse_interference) * 100, reduction))

print()
print('='*70)
print('ANALYSIS')
print('='*70)

baseline = np.mean(full_interference)
best_sparsity = min(results.keys(), key=lambda s: results[s])
best_rate = results[best_sparsity]

print()
print('Results Summary:')
print('-'*40)
print('Baseline (0%): %.0f%%' % (baseline * 100))
for sparsity, rate in sorted(results.items()):
    print('Sparsity %.0f%%: %.0f%%' % (sparsity * 100, rate * 100))

print()
print('Best Sparsity: %.0f%% (%.0f%% interference)' % (best_sparsity * 100, best_rate * 100))

improvement = (baseline - best_rate) / baseline * 100
print('Improvement: %.0f%%' % improvement)

if improvement > 20:
    print()
    print('🎉 SUCCESS: 80% sparsity significantly reduces interference!')
elif improvement > 0:
    print()
    print('⚠️ PARTIAL: Sparsity helps but not dramatically')
else:
    print()
    print('❌ FAILURE: Sparsity does not help')

# Save
output = {
    'baseline_interference': baseline,
    'results': {str(k): v for k, v in results.items()},
    'best_sparsity': best_sparsity,
    'best_rate': best_rate,
    'improvement_percent': improvement,
    'target_met': improvement > 20
}

with open('results/sparsity_80percent.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print('Saved to results/sparsity_80percent.json')
