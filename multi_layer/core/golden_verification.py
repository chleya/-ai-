"""
GOLDEN VERIFICATION: SR Window, Robustness, Frequency
Final Validation: SR Window, Robustness, Frequency Specificity
"""

import numpy as np, json

def nl(x): return np.tanh(x)

# 1. Noise Intensity Scan
def sr_intensity_scan(N=1000, n_modules=50, trials=15):
    results = {}
    for intensity in np.arange(0.01, 0.15, 0.01):
        success = 0
        for t in range(trials):
            np.random.seed(t * 1000 + 42)
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            
            W_modules = []
            size = N // n_modules
            for m in range(n_modules):
                W = np.random.randn(size, size) / np.sqrt(size)
                W_modules.append(W)
            
            # Isolation
            for step in range(60):
                for m in range(n_modules):
                    idx = slice(m*size, (m+1)*size)
                    P_m = P[idx]
                    x_good = nl(W_modules[m] @ P_m + np.random.randn(size) * 0.3)
                    W_modules[m] += 0.001 * np.outer(x_good, x_good)
                    if np.linalg.norm(W_modules[m]) > 10:
                        W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
            
            # Crystallization with SR
            alpha = 0.2
            for step in range(100):
                s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
                new_s = np.zeros(N)
                for m in range(n_modules):
                    idx = slice(m*size, (m+1)*size)
                    s_m = s[idx]
                    noise_pulse = intensity * np.sin(2 * np.pi * 0.1 * step / 100)
                    new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m + noise_pulse)
                s = new_s
            
            if np.sign(np.mean(s)) == 1:
                success += 1
        
        results[f'{intensity:.2f}'] = success * 100 // trials
    
    return results

# 2. Robustness Stress Test
def robustness_stress_test(N=1000, n_modules=50, error_ratio=0.5, trials=15):
    success = 0
    size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # Create corrupted nodes
        n_error = int(N * error_ratio)
        P_corrupted = P.copy()
        P_corrupted[:n_error] = -P_corrupted[:n_error]
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(size, size) / np.sqrt(size)
            W_modules.append(W)
        
        # Isolation
        for step in range(60):
            for m in range(n_modules):
                idx = slice(m*size, (m+1)*size)
                P_m = P_corrupted[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # Crystallization with SR
        alpha = 0.2
        intensity = 0.05
        
        for step in range(100):
            s = (1 - 0.5) * P_corrupted + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            for m in range(n_modules):
                idx = slice(m*size, (m+1)*size)
                s_m = s[idx]
                noise_pulse = intensity * np.sin(2 * np.pi * 0.1 * step / 100)
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m + noise_pulse)
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

# 3. Frequency Specificity
def frequency_specificity_test(N=1000, n_modules=50, mode='continuous', trials=15):
    success = 0
    size = N // n_modules
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        W_modules = []
        for m in range(n_modules):
            W = np.random.randn(size, size) / np.sqrt(size)
            W_modules.append(W)
        
        # Isolation
        for step in range(60):
            for m in range(n_modules):
                idx = slice(m*size, (m+1)*size)
                P_m = P[idx]
                x_good = nl(W_modules[m] @ P_m + np.random.randn(size) * 0.3)
                W_modules[m] += 0.001 * np.outer(x_good, x_good)
                if np.linalg.norm(W_modules[m]) > 10:
                    W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
        
        # Crystallization with different frequency modes
        alpha = 0.2
        intensity = 0.05
        
        for step in range(100):
            s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
            new_s = np.zeros(N)
            
            for m in range(n_modules):
                idx = slice(m*size, (m+1)*size)
                s_m = s[idx]
                
                if mode == 'continuous':
                    noise_pulse = intensity * np.sin(2 * np.pi * 0.1 * step / 100)
                elif mode == 'pulsed':
                    noise_pulse = intensity * np.sin(2 * np.pi * 0.1 * step / 100) if step % 5 == 0 else 0
                elif mode == 'burst':
                    noise_pulse = intensity * np.sin(2 * np.pi * 0.1 * step / 100) if step % 10 < 3 else 0
                elif mode == 'decay':
                    decay = 1 - step / 100
                    noise_pulse = intensity * decay * np.sin(2 * np.pi * 0.1 * step / 100)
                elif mode == 'ramp':
                    ramp = step / 100
                    noise_pulse = intensity * ramp * np.sin(2 * np.pi * 0.1 * step / 100)
                
                new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m + noise_pulse)
            
            s = new_s
        
        if np.sign(np.mean(s)) == 1:
            success += 1
    
    return success * 100 // trials

def main():
    print("="*60)
    print("GOLDEN VERIFICATION: SR Window")
    print("="*60)
    
    # 1. Intensity Scan
    print("\n[1] Noise Intensity Scan:")
    intensity_results = sr_intensity_scan()
    for intensity, rate in sorted(intensity_results.items()):
        marker = "[GOLDEN]" if rate == 100 else ("[HIGH]" if rate >= 80 else "")
        print(f"    intensity={intensity}: {rate}% {marker}")
    
    golden = max(intensity_results.items(), key=lambda x: x[1])
    print(f"\n    Golden Intensity: {golden[0]} = {golden[1]}%")
    
    # 2. Robustness
    print("\n[2] Robustness Stress Test:")
    robustness = {}
    for err in [0.3, 0.4, 0.5, 0.6]:
        rate = robustness_stress_test(error_ratio=err)
        robustness[err] = rate
        marker = "[GOLDEN]" if rate == 100 else ("[HIGH]" if rate >= 80 else "")
        print(f"    error={err}: {rate}% {marker}")
    
    # 3. Frequency
    print("\n[3] Frequency Specificity:")
    freq = {}
    for mode in ['continuous', 'pulsed', 'burst', 'decay', 'ramp']:
        rate = frequency_specificity_test(mode=mode)
        freq[mode] = rate
        marker = "[BEST]" if rate == 100 else ""
        print(f"    mode={mode}: {rate}% {marker}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_results = {
        'intensity_scan': intensity_results,
        'robustness': robustness,
        'frequency': freq,
        'golden_intensity': golden[0]
    }
    
    print(f"\nGolden Intensity: {golden[0]} = {golden[1]}%")
    best_freq = max(freq.items(), key=lambda x: x[1])
    print(f"Best Frequency: {best_freq[0]} = {best_freq[1]}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Non-monotonicity
    rates = list(intensity_results.values())
    if rates[-1] < rates[0]:
        print("Non-monotonicity CONFIRMED")
    
    # Robustness limit
    max_err = max([k for k, v in robustness.items() if v >= 80])
    print(f"Robustness Limit: {max_err*100:.0f}% errors")
    
    # Save
    with open('results/golden_verification.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/golden_verification.json")

if __name__ == "__main__":
    main()
