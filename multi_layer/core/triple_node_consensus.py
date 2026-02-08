"""
Triple Node Consensus: Democratic Agreement Experiment
Testing: Can 3 nodes with different noise levels reach consensus?

Node A: 30% noise (clear)
Node B: 80% noise (fuzzy)
Node C: 80% noise + WRONG attractor (stubborn)

Topology: Fully connected (A↔B↔C)
"""

import numpy as np, json

def nonlinearity(x):
    return np.tanh(x)

class HebbianNode:
    def __init__(self, N=50, name='Node'):
        self.N = N
        self.name = name
        self.W = np.random.randn(N, N) / np.sqrt(N)
        self.state = np.zeros(N)
        self.history = []
    
    def learn(self, pattern, steps=5000, eta=0.001):
        self.state = pattern.copy()
        for _ in range(steps):
            noise = np.random.randn(self.N) * 0.2
            x = nonlinearity(self.W @ self.state + noise)
            self.W += eta * np.outer(x, x)
            if np.linalg.norm(self.W) > 10:
                self.W = self.W / np.linalg.norm(self.W) * 10
            self.state = x
    
    def set_state(self, state):
        self.state = state / np.linalg.norm(state) * 0.5
        self.history = []
    
    def evolve_autonomous(self, steps=100):
        for _ in range(steps):
            self.state = nonlinearity(self.W @ self.state)
            self.history.append(np.sign(np.mean(self.state)))
    
    def coupled_evolve(self, neighbors, alpha=0.2):
        """Evolve with coupled input from neighbors"""
        coupled_input = self.W @ self.state
        for neighbor in neighbors:
            coupled_input += alpha * neighbor.state
        self.state = nonlinearity(coupled_input)
        self.history.append(np.sign(np.mean(self.state)))
    
    def get_sign(self):
        return np.sign(np.mean(self.state))


def create_noisy_pattern(pattern, noise_level, seed=42):
    """Create noisy version of pattern"""
    np.random.seed(seed)
    return (1 - noise_level) * pattern + noise_level * np.random.randn(len(pattern))


def run_consensus_experiment(N=50, alpha=0.2, test_name='Test'):
    """Run one consensus experiment"""
    
    # Create correct pattern P
    np.random.seed(42)
    P = np.random.randn(N)
    P = P / np.linalg.norm(P)
    
    # Node A: Clear (30% noise)
    NodeA = HebbianNode(N, 'NodeA')
    NodeA.learn(P, steps=5000)
    state_A = create_noisy_pattern(P, 0.3, seed=100)
    NodeA.set_state(state_A)
    
    # Node B: Fuzzy (80% noise)
    NodeB = HebbianNode(N, 'NodeB')
    NodeB.learn(P, steps=5000)
    state_B = create_noisy_pattern(P, 0.8, seed=200)
    NodeB.set_state(state_B)
    
    # Node C: Stubborn (80% noise + WRONG pattern)
    NodeC = HebbianNode(N, 'NodeC')
    NodeC.learn(-P, steps=5000)  # Learned wrong pattern!
    wrong_P = -P + np.random.randn(N) * 0.1  # Small noise
    state_C = create_noisy_pattern(wrong_P, 0.8, seed=300)
    NodeC.set_state(state_C)
    
    # Initial signs
    initial_signs = [NodeA.get_sign(), NodeB.get_sign(), NodeC.get_sign()]
    
    # Coupled evolution (fully connected)
    consensus_signs = []
    for t in range(500):
        # All evolve simultaneously
        A_input = NodeA.W @ NodeA.state + alpha * (NodeB.state + NodeC.state)
        B_input = NodeB.W @ NodeB.state + alpha * (NodeA.state + NodeC.state)
        C_input = NodeC.W @ NodeC.state + alpha * (NodeA.state + NodeB.state)
        
        NodeA.state = nonlinearity(A_input)
        NodeB.state = nonlinearity(B_input)
        NodeC.state = nonlinearity(C_input)
        
        consensus_signs.append([NodeA.get_sign(), NodeB.get_sign(), NodeC.get_sign()])
        
        # Check for early consensus
        if t > 50 and len(set([s[-1] for s in consensus_signs[-10:]])) == 1:
            break
    
    final_signs = [s[-1] for s in consensus_signs]
    
    # Did they reach consensus?
    consensus = len(set(final_signs)) == 1
    correct_consensus = consensus and (final_signs[0] == np.sign(np.mean(P)))
    
    return {
        'initial': initial_signs,
        'final': final_signs,
        'consensus': consensus,
        'correct': correct_consensus,
        'steps': len(consensus_signs),
        'timeline': consensus_signs
    }


print('='*70)
print('TRIPLE NODE CONSENSUS EXPERIMENT')
print('='*70)

N = 50
n_tests = 10
alphas = [0.1, 0.2, 0.3]

print()
print('Configuration:')
print('- Node A: 30% noise (clear)')
print('- Node B: 80% noise (fuzzy)')
print('- Node C: 80% noise + WRONG pattern')
print('- Topology: Fully connected (A↔B↔C)')
print()

all_results = {}

for alpha in alphas:
    print('='*50)
    print('Alpha = %.1f' % alpha)
    print('='*50)
    
    consensus_count = 0
    correct_count = 0
    avg_steps = 0
    
    for test in range(n_tests):
        np.random.seed(test * 100 + int(alpha * 100))
        result = run_consensus_experiment(N, alpha, 'Test%d' % test)
        
        if result['consensus']:
            consensus_count += 1
            avg_steps += result['steps']
            if result['correct']:
                correct_count += 1
        
        print('Test %d: Initial=%s -> Final=%s [%s]' % (
            test+1, 
            str(result['initial']), 
            str(result['final']),
            'CORRECT' if result['correct'] else 'WRONG'
        ))
    
    all_results[alpha] = {
        'consensus_rate': consensus_count / n_tests * 100,
        'correct_rate': correct_count / n_tests * 100,
        'avg_steps': avg_steps / consensus_count if consensus_count > 0 else 'N/A'
    }
    
    print()
    print('Consensus Rate: %.0f%%' % (consensus_count / n_tests * 100))
    print('Correct Consensus: %.0f%%' % (correct_count / n_tests * 100))
    print('Avg Steps: %s' % (avg_steps / consensus_count if consensus_count > 0 else 'N/A'))

print()
print('='*70)
print('ANALYSIS')
print('='*70)

print()
print('Alpha | Consensus | Correct | Avg Steps')
print('-'*50)
for alpha, r in sorted(all_results.items()):
    print('%.1f   | %.0f%%      | %.0f%%     | %s' % (
        alpha, 
        r['consensus_rate'], 
        r['correct_rate'],
        str(r['avg_steps'])
    ))

# Best alpha
best_alpha = max(all_results.keys(), key=lambda a: all_results[a]['correct_rate'])
best = all_results[best_alpha]

print()
if best['correct_rate'] >= 70:
    print('SUCCESS: Democracy works! Correct nodes can sway the group.')
elif best['correct_rate'] >= 40:
    print('PARTIAL: Sometimes consensus, sometimes civil war.')
else:
    print('FAILURE: Stubborn nodes dominate. Alpha=%.1f too weak.' % best_alpha)

# Save
with open('results/triple_consensus.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print()
print('Saved to results/triple_consensus.json')
