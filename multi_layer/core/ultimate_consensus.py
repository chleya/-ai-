"""
ULTIMATE CONSENSUS: LTD + Soft Weighting Combination
Challenge: Can we achieve 80%+ correctness?

Strategy:
1. LTD: Destroy C's attractor first
2. Soft Weighting: B contributes proportionally to confidence
3. A is amplified: Strong correct signal
"""

import numpy as np, json

def nl(x):
    return np.tanh(x)

def confidence(x):
    return abs(np.mean(x))

def energy(W, pattern):
    return -np.mean(pattern @ W @ pattern)

print('='*70)
print('ULTIMATE CONSENSUS: LTD + Soft Weighting')
print('Phase 1: LTD (destroy C)')
print('Phase 2: Soft Weighting (confidence-based contribution)')
print('Phase 3: Measure correctness')
print('='*70)

N = 50
n_tests = 40

# Test different configurations
configs = [
    {'ltd': True, 'soft': True, 'amp_A': 2.0, 'factor_B': 2.0, 'desc': 'LTD + Soft + Amp'},
    {'ltd': True, 'soft': True, 'amp_A': 3.0, 'factor_B': 2.5, 'desc': 'LTD + Soft + Strong Amp'},
    {'ltd': True, 'soft': True, 'amp_A': 2.5, 'factor_B': 2.0, 'desc': 'LTD + Soft + Med Amp'},
    {'ltd': True, 'soft': False, 'amp_A': 2.0, 'factor_B': 1.0, 'desc': 'LTD only + Amp'},
    {'ltd': False, 'soft': True, 'amp_A': 1.0, 'factor_B': 2.0, 'desc': 'Soft only'},
    {'ltd': False, 'soft': False, 'amp_A': 1.0, 'factor_B': 1.0, 'desc': 'Baseline'},
]

results = {}

for config in configs:
    correct_count = 0
    energy_before_C = []
    energy_after_C = []
    
    for test in range(n_tests):
        np.random.seed(test * 100 + hash(config['desc']) % 1000)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        # Create nodes
        W1 = np.random.randn(N, N) / np.sqrt(N)
        W2 = np.random.randn(N, N) / np.sqrt(N)
        W3 = np.random.randn(N, N) / np.sqrt(N)
        
        # Learn patterns
        for _ in range(2000):
            x1 = nl(W1 @ P + np.random.randn(N) * 0.2)
            x2 = nl(W2 @ P + np.random.randn(N) * 0.2)
            x3 = nl(W3 @ (-P) + np.random.randn(N) * 0.2)
            
            W1 += 0.001 * np.outer(x1, x1)
            W2 += 0.001 * np.outer(x2, x2)
            W3 += 0.001 * np.outer(x3, x3)
            
            if np.linalg.norm(W1) > 10:
                W1 = W1 / np.linalg.norm(W1) * 10
            if np.linalg.norm(W2) > 10:
                W2 = W2 / np.linalg.norm(W2) * 10
            if np.linalg.norm(W3) > 10:
                W3 = W3 / np.linalg.norm(W3) * 10
        
        # Track C's energy before LTD
        energy_before_C.append(energy(W3, -P))
        
        # LTD: Destroy C's attractor
        if config['ltd']:
            # Negative Hebbian
            x3_final = nl(W3 @ (-P) + np.random.randn(N) * 0.2)
            W3 -= 0.5 * np.outer(x3_final, x3_final)
            # Normalize to flatten
            W3 = W3 / np.linalg.norm(W3) * 0.1
        
        energy_after_C.append(energy(W3, -P))
        
        # Initial states
        s1 = (1 - 0.3) * P + 0.3 * np.random.randn(N)
        s2 = (1 - 0.8) * P + 0.8 * np.random.randn(N)
        s3 = (1 - 0.8) * (-P) + 0.8 * np.random.randn(N)
        
        # Evolution with configuration
        for _ in range(500):
            conf_A = confidence(s1)
            conf_B = confidence(s2)
            conf_C = confidence(s3)
            
            # Calculate contributions
            amp_A = config['amp_A']
            factor_B = config['factor_B']
            
            # A's contribution
            contrib_A = amp_A * conf_A
            
            # B's contribution (soft weighted)
            if config['soft']:
                contrib_B = factor_B * conf_B
            else:
                contrib_B = conf_B
            
            # C's contribution (minimal if LTD, normal otherwise)
            contrib_C = conf_C
            
            # Normalize total
            total = contrib_A + contrib_B + contrib_C + 0.01
            
            # Update states
            s1 = nl(W1 @ s1 + (contrib_A / total) * 2.0 * s1 + 
                           (contrib_B / total) * 2.0 * s2 + 
                           (contrib_C / total) * 2.0 * s3)
            
            s2 = nl(W2 @ s2 + (contrib_A / total) * 2.0 * s1 + 
                           (contrib_B / total) * 2.0 * s2 + 
                           (contrib_C / total) * 2.0 * s3)
            
            s3 = nl(W3 @ s3 + (contrib_A / total) * 2.0 * s1 + 
                           (contrib_B / total) * 2.0 * s2 + 
                           (contrib_C / total) * 2.0 * s3)
        
        # Check correctness
        f1 = np.sign(np.mean(s1))
        f2 = np.sign(np.mean(s2))
        f3 = np.sign(np.mean(s3))
        
        # Correct if all agree on +1
        if f1 == 1 and f2 == 1 and f3 == 1:
            correct_count += 1
    
    correct_rate = correct_count / n_tests * 100
    avg_energy_change = np.mean(energy_after_C) - np.mean(energy_before_C)
    
    results[config['desc']] = {
        'correct_rate': correct_rate,
        'config': config,
        'energy_change': avg_energy_change
    }
    
    print('%s: %.0f%% correct' % (config['desc'], correct_rate))
    print('  Energy change: %.3f' % avg_energy_change)

print()
print('='*70)
print('ANALYSIS: Can LTD + Soft Weighting achieve 80%+?')
print('='*70)

# Find best configuration
best_desc = max(results.keys(), key=lambda x: results[x]['correct_rate'])
best_result = results[best_desc]

print()
print('BEST CONFIGURATION:')
print('  %s' % best_desc)
print('  Correct Rate: %.0f%%' % best_result['correct_rate'])
print('  Energy Change: %.3f' % best_result['energy_change'])

print()
print('SUMMARY:')
for desc, res in sorted(results.items(), key=lambda x: -x[1]['correct_rate']):
    marker = '⭐' if res['correct_rate'] >= 80 else ('✓' if res['correct_rate'] >= 70 else ' ')
    print('%s %s: %.0f%%' % (marker, desc, res['correct_rate']))

if best_result['correct_rate'] >= 80:
    print()
    print('🎉 SUCCESS: Ultimate combination achieves 80%+!')
elif best_result['correct_rate'] >= 70:
    print()
    print('✅ GOOD: Achieves 70%+. Close to target.')
else:
    print()
    print('⚠️  Still below 80%. More work needed.')

# Save results
with open('results/ultimate_consensus.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print('Saved to results/ultimate_consensus.json')
