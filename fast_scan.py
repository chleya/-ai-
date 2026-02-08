import numpy as np
import json

print('FAST SCAN: System 04b Feedback')

class System:
    N, T = 20, 15000
    
    def __init__(self, sigma=0.5, target=0.015, mode='positive', seed=42):
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.mode = mode
        self.target = target
        self.alpha = 0.45
        self.history = []
    
    def run(self):
        for t in range(self.T):
            noise = np.random.randn(self.N) * self.sigma
            self.state = np.tanh(self.W @ self.state + noise)
            norm = np.linalg.norm(self.state)
            if norm > 1e-8:
                self.state = (self.state / norm) * self.alpha
            
            var = np.var(self.state)
            self.history.append(var)
            
            # Alpha
            recent = np.mean(self.history[-500:]) if len(self.history) > 500 else var
            self.alpha += 0.01 * (self.target / (recent + 1e-8) - self.alpha)
            self.alpha = np.clip(self.alpha, 0.3, 2.0)
            
            # Target feedback
            if (t + 1) % 300 == 0 and t > 2000:
                window = np.mean(self.history[-1000:]) if len(self.history) > 1000 else var
                if self.mode == 'positive' and window > self.target * 1.05:
                    self.target *= 1.01
                elif self.mode == 'negative' and window > self.target * 1.1:
                    self.target *= 0.99
                self.target = np.clip(self.target, 0.005, 0.30)
        
        return np.mean(self.history[-500:]), self.target

# Scan
sigmas = [0.5, 1.0, 1.5]
targets = [0.010, 0.015, 0.020]
modes = ['positive', 'negative', 'fixed']

print()
print('Sigma  Target   PosVar   NegVar   FixVar   Improv%')
print('-'*60)

results = []

for s in sigmas:
    for t in targets:
        pos_var, pos_t = System(s, t, 'positive').run()
        neg_var, neg_t = System(s, t, 'negative').run()
        fix_var, fix_t = System(s, t, 'fixed').run()
        
        improv = (pos_var - fix_var) / fix_var * 100
        
        print('%.1f    %.3f   %.4f  %.4f  %.4f   %+.1f%%' % 
              (s, t, pos_var, neg_var, fix_var, improv))
        
        results.append({'sigma': s, 'target': t, 'improvement': improv})

avg = np.mean([r['improvement'] for r in results])
print()
print('Average improvement: %.1f%%' % avg)
print('All stable:', all(r['improvement'] < 100 for r in results))
