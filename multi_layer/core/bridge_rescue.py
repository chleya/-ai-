"""
Bridge Communication: Node A pulls Node B from noise
Testing: Can a confident Node A rescue a noisy Node B?
"""

import numpy as np, json

def nonlinearity(x, name='tanh'):
    if name == 'tanh':
        return np.tanh(x)
    elif name == 'relu':
        return np.maximum(0, x)
    return np.tanh(x)

class HebbianNode:
    def __init__(self, N=50, name='Node'):
        self.N = N
        self.name = name
        self.W = np.random.randn(N, N) / np.sqrt(N)
        self.state = np.zeros(N)
        
    def learn(self, pattern, steps=5000, eta=0.001):
        """Learn a pattern via Hebbian"""
        self.state = pattern.copy()
        for _ in range(steps):
            noise = np.random.randn(self.N) * 0.2
            x = nonlinearity(self.W @ self.state + noise)
            self.W += eta * np.outer(x, x)
            if np.linalg.norm(self.W) > 10:
                self.W = self.W / np.linalg.norm(self.W) * 10
            self.state = x
    
    def evolve(self, steps=300):
        """Self-evolve without input"""
        for _ in range(steps):
            self.state = nonlinearity(self.W @ self.state)
    
    def get_sign(self):
        return np.sign(np.mean(self.state))

class BridgeSystem:
    def __init__(self, N=50, alpha=0.2):
        self.N = N
        self.alpha = alpha  # Coupling strength
        self.NodeA = HebbianNode(N, 'NodeA')
        self.NodeB = HebbianNode(N, 'NodeB')
    
    def coupled_evolve(self, steps=300):
        """Both nodes evolve with coupling A->B"""
        for _ in range(steps):
            # Node B receives its own state +耦合 from A
            coupled_input = self.NodeB.W @ self.NodeB.state + self.alpha * self.NodeA.state
            self.NodeB.state = nonlinearity(coupled_input)
            
            # Node A evolves autonomously
            self.NodeA.state = nonlinearity(self.NodeA.W @ self.NodeA.state)
    
    def reset_b_to_noise(self, pattern, noise_level=0.9):
        """Reset Node B to noisy version of pattern"""
        np.random.seed(42)
        self.NodeB.state = (1 - noise_level) * pattern + noise_level * np.random.randn(self.N)
        # Normalize
        self.NodeB.state = self.NodeB.state / np.linalg.norm(self.NodeB.state) * 0.5

print('='*70)
print('BRIDGE EXPERIMENT: Can Node A Rescue Node B?')
print('='*70)

N = 50
alpha = 0.2  # Coupling strength

print()
print('PART 1: Train Node A on Pattern P')
print('-'*50)

# Create pattern P
np.random.seed(42)
P = np.random.randn(N)
P = np.linalg.norm(P)
P = P / np.linalg.norm(P)

# Train Node A
NodeA = HebbianNode(N, 'NodeA')
NodeA.learn(P, steps=10000)
NodeA.evolve()

print('Node A trained on pattern P')
print('Node A state: sign = %+.1f' % NodeA.get_sign())

print()
print('PART 2: Reset Node B to 90% noise')
print('-'*50)

# Create Node B (fresh, untrained)
NodeB = HebbianNode(N, 'NodeB')

# Reset B to noisy version of P
noisy_B = (1 - 0.9) * P + 0.9 * np.random.randn(N)
noisy_B = noisy_B / np.linalg.norm(noisy_B) * 0.5
NodeB.state = noisy_B

print('Node B noisy state: sign = %+.1f' % NodeB.get_sign())

print()
print('PART 3: Coupled Evolution (A->B, alpha=%.1f)' % alpha)
print('-'*50)

# Bridge system
bridge = BridgeSystem(N, alpha)
bridge.NodeA = NodeA
bridge.NodeB = NodeB

# Track evolution
signs = []
for t in range(100):
    bridge.coupled_evolve(steps=10)
    sign = bridge.NodeB.get_sign()
    signs.append(sign)
    if t % 20 == 0:
        print('Step %d: Node B sign = %+.1f' % (t * 10, sign))

print()
print('='*70)
print('ANALYSIS')
print('='*70)

# Count signs
n_positive = signs.count(1.0)
n_negative = signs.count(-1.0)

print()
print('Node B final sign: %+.1f' % signs[-1])
print('Positive steps: %d' % n_positive)
print('Negative steps: %d' % n_negative)

print()
if signs[-1] == NodeA.get_sign():
    print('🎉 SUCCESS: Node A rescued Node B!')
    print('Coupling worked!')
else:
    print('❌ FAILED: Node B resisted rescue.')
    print('Alpha=%.1f may be too weak.' % alpha)

# Save
result = {
    'alpha': alpha,
    'noise_level': 0.9,
    'node_a_sign': float(NodeA.get_sign()),
    'node_b_final_sign': float(signs[-1]),
    'rescue_success': signs[-1] == NodeA.get_sign(),
    'rescue_timeline': signs
}

with open('results/bridge_rescue.json', 'w') as f:
    json.dump(result, f, indent=2)

print()
print('Saved to results/bridge_rescue.json')
