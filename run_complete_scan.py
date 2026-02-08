import numpy as np
import json
from datetime import datetime

print('='*70)
print('SYSTEM 04b: COMPLETE FEEDBACK SCAN')
print('='*70)

class SystemFeedback:
    N, T = 20, 25000
    
    def __init__(self, sigma=0.5, target_init=0.015, mode='positive', seed=42):
        np.random.seed(seed)
        self.state = np.random.randn(self.N)
        self.W = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        self.sigma = sigma
        self.mode = mode
        self.target_var = target_init
        self.target_init = target_init
        self.alpha = 0.45
        self.history = {'variance': [], 'alpha': [], 'target': []}
    
    def run(self):
        for t in range(self.T):
            noise = np.random.randn(self.N) * self.sigma
            self.state = np.tanh(self.W @ self.state + noise)
            norm = np.linalg.norm(self.state)
            if norm > 1e-8:
                self.state = (self.state / norm) * self.alpha
            
            var = np.var(self.state)
            self.history['variance'].append(var)
            self.history['alpha'].append(self.alpha)
            self.history['target'].append(self.target_var)
            
            recent = np.mean(self.history['variance'][-500:]) if len(self.history['variance']) > 500 else var
            target_alpha = self.target_var / (recent + 1e-8)
            self.alpha += 0.01 * (target_alpha - self.alpha)
            self.alpha = np.clip(self.alpha, 0.3, 2.0)
            
            if (t + 1) % 500 == 0 and t > 3000:
                window = np.mean(self.history['variance'][-2000:]) if len(self.history['variance']) > 2000 else var
                target_avg = np.mean(self.history['target'][-2000:]) if len(self.history['target']) > 2000 else self.target_var
                
                if self.mode == 'positive':
                    if window > target_avg * 1.05:
                        self.target_var *= 1.01
                elif self.mode == 'negative':
                    if window > target_avg * 1.1:
                        self.target_var *= 0.99
                
                self.target_var = np.clip(self.target_var, 0.005, 0.30)
        
        return {
            'final_variance': float(np.mean(self.history['variance'][-1000:])),
            'final_target': float(np.mean(self.history['target'][-500:])),
            'target_drift': float(np.mean(self.history['target'][-500:]) - self.target_init),
        }

# Full scan
sigmas = [0.5, 1.0, 1.5, 2.0]
targets = [0.010, 0.015, 0.020]
modes = ['positive', 'negative', 'fixed']

results = {}

for sigma in sigmas:
    for target in targets:
        key = 's%d_t%.3f' % (sigma, target)
        results[key] = {'sigma': sigma, 'target': target, 'modes': {}}
        
        for mode in modes:
            sys = SystemFeedback(sigma=sigma, target_init=target, mode=mode)
            r = sys.run()
            results[key]['modes'][mode] = {
                'final_variance': r['final_variance'],
                'final_target': r['final_target'],
                'target_drift': r['target_drift']
            }

# Print results
print()
print('COMPLETE RESULTS TABLE')
print('='*90)
print('Sigma  Target   PosVar   NegVar   FixVar   PosDrift  NegDrift')
print('-'*90)

positive_only = []

for key, data in results.items():
    s = data['sigma']
    t = data['target']
    pm = data['modes']['positive']
    nm = data['modes']['negative']
    fm = data['modes']['fixed']
    
    print('%.1f    %.3f   %.4f   %.4f   %.4f   %+.4f   %+.4f' % 
          (s, t, pm['final_variance'], nm['final_variance'], fm['final_variance'],
           pm['target_drift'], nm['target_drift']))
    
    improv = (pm['final_variance'] - fm['final_variance']) / fm['final_variance'] * 100
    positive_only.append({'sigma': s, 'target': t, 'improvement': improv})

print()
print('='*90)
print('POSITIVE FEEDBACK SUMMARY')
print('='*90)
print('Sigma  Target   Improvement  Status')
print('-'*50)

all_stable = True
for r in positive_only:
    status = 'STABLE' if r['improvement'] < 100 else 'CHECK'
    print('%.1f    %.3f   %+8.1f%%    %s' % (r['sigma'], r['target'], r['improvement'], status))
    if r['improvement'] > 100:
        all_stable = False

avg_imp = np.mean([r['improvement'] for r in positive_only])
print()
print('='*90)
if all_stable:
    print('*** ALL RUNS STABLE - Positive feedback works across all conditions ***')
print('Average improvement: %+.1f%%' % avg_imp)

# Save
with open('results/exp_04b_complete.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print('Results saved.')
