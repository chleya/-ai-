"""
MULTI-A REDUNDANCY: Breaking the 70% Ceiling
Test: Can 3 A nodes achieve 80%+ correctness?

Structure: A1, A2, A3 (each 70% correct), B (80% noise), C (wrong)
Hypothesis: Multiple correct nodes create deeper attractor basin
"""

import numpy as np, json

def nl(x):
    return np.tanh(x)

def confidence(x):
    return abs(np.mean(x))

def energy(W, pattern):
    return -np.mean(pattern @ W @ pattern)

print('='*70)
print('MULTI-A REDUNDANCY: Breaking the Ceiling')
print('3 A nodes vs 1 A node')
print('='*70)

N = 50
n_tests = 40

configs = [
    {'n_A': 1, 'use_ltd': True, 'desc': '1A + LTD'},
    {'n_A': 2, 'use_ltd': True, 'desc': '2A + LTD'},
    {'n_A': 3, 'use_ltd': True, 'desc': '3A + LTD'},
    {'n_A': 3, 'use_ltd': False, 'desc': '3A (no LTD)'},
]

results = {}

for config in configs:
    n_A = config['n_A']
    use_ltd = config['use_ltd']
    
    correct_count = 0
    energy_measurements = []
    
    for test in range(n_tests):
        np.random.seed(test * 100 + n_A * 10 + use_ltd * 100)
        P = np.random.randn(N)
        P = P / np.linalg.norm(P)
        
        # Create A nodes
        W_A = []
        for i in range(n_A):
            W = np.random.randn(N, N) / np.sqrt(N)
            W_A.append(W)
        
        # Create B and C
        W_B = np.random.randn(N, N) / np.sqrt(N)
        W_C = np.random.randn(N, N) / np.sqrt(N)
        
        # Learn patterns
        for _ in range(2000):
            # A nodes learn correct pattern
            for i, W in enumerate(W_A):
                x = nl(W @ P + np.random.randn(N) * 0.2)
                W_A[i] += 0.001 * np.outer(x, x)
                if np.linalg.norm(W_A[i]) > 10:
                    W_A[i] = W_A[i] / np.linalg.norm(W_A[i]) * 10
            
            # B learns correct but with noise
            x_B = nl(W_B @ P + np.random.randn(N) * 0.2)
            W_B += 0.001 * np.outer(x_B, x_B)
            if np.linalg.norm(W_B) > 10:
                W_B = W_B / np.linalg.norm(W_B) * 10
            
            # C learns wrong pattern
            x_C = nl(W_C @ (-P) + np.random.randn(N) * 0.2)
            W_C += 0.001 * np.outer(x_C, x_C)
            if np.linalg.norm(W_C) > 10:
                W_C = W_C / np.linalg.norm(W_C) * 10
        
        # Track energy before LTD
        energy_measurements.append(energy(W_C, -P))
        
        # LTD: Destroy C's attractor
        if use_ltd:
            x_C_final = nl(W_C @ (-P) + np.random.randn(N) * 0.2)
            W_C -= 0.5 * np.outer(x_C_final, x_C_final)
            W_C = W_C / np.linalg.norm(W_C) * 0.1
        
        energy_measurements.append(energy(W_C, -P))
        
        # Initial states
        states = []
        
        # A nodes: 70% correct
        for i in range(n_A):
            s = (1 - 0.3) * P + 0.3 * np.random.randn(N)
            states.append(s)
        
        # B: 80% noise
        s_B = (1 - 0.8) * P + 0.8 * np.random.randn(N)
        states.append(s_B)
        
        # C: Wrong pattern
        s_C = (1 - 0.8) * (-P) + 0.8 * np.random.randn(N)
        states.append(s_C)
        
        # Evolution
        for _ in range(500):
            # Calculate contributions
            contribs = []
            for s in states:
                contribs.append(confidence(s))
            
            # Update all states
            new_states = []
            for i, s in enumerate(states):
                # Calculate input from all nodes
                total_input = np.zeros(N)
                for j, sj in enumerate(states):
                    if i < n_A:
                        # A node: learns correct pattern
                        total_input += W_A[i] @ s
                    elif i == n_A:
                        # B node
                        total_input += W_B @ s
                    else:
                        # C node
                        total_input += W_C @ s
                
                # Add coupling from all nodes
                coupling = np.zeros(N)
                for j, sj in enumerate(states):
                    coupling += contribs[j] * sj
                
                new_s = nl(total_input + 0.2 * coupling / sum(contribs))
                new_states.append(new_s)
            
            states = new_states
        
        # Check correctness
        final_signs = [np.sign(np.mean(s)) for s in states]
        
        # Correct if all A nodes and B agree on +1
        all_correct = True
        for i in range(n_A):
            if final_signs[i] != 1:
                all_correct = False
        if final_signs[n_A] != 1:  # B should be +1
            all_correct = False
        
        if all_correct:
            correct_count += 1
    
    correct_rate = correct_count / n_tests * 100
    energy_change = energy_measurements[-1] - energy_measurements[0]
    
    results[config['desc']] = {
        'correct_rate': correct_rate,
        'n_A': n_A,
        'use_ltd': use_ltd,
        'energy_change': energy_change
    }
    
    print('%s: %.0f%% correct' % (config['desc'], correct_rate))
    print('  Energy change: %.3f' % energy_change)

print()
print('='*70)
print('ANALYSIS: Can Multi-A Break the 70% Ceiling?')
print('='*70)

# Find best
best_desc = max(results.keys(), key=lambda x: results[x]['correct_rate'])
best = results[best_desc]

print()
print('BEST: %s (%.0f%%)' % (best_desc, best['correct_rate']))

print()
print('COMPARISON:')
for desc, res in sorted(results.items(), key=lambda x: -x[1]['correct_rate']):
    marker = '⭐' if res['correct_rate'] >= 80 else ('✓' if res['correct_rate'] >= 70 else ' ')
    print('%s %s: %.0f%%' % (marker, desc, res['correct_rate']))

if best['correct_rate'] >= 80:
    print()
    print('🎉 BREAKTHROUGH: Multi-A achieves 80%+!')
elif best['correct_rate'] >= 70:
    print()
    print('✅ SUCCESS: Multi-A approaches ceiling.')
else:
    print()
    print('⚠️  Still below 70%. Ceiling confirmed.')

# Save
with open('results/multi_A.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print('Saved to results/multi_A.json')
