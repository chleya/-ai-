"""
HIGH VARIANCE ANALYSIS
研究高方差的根本原因
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def analyze_variance(trials=50):
    """分析高方差来源"""
    
    results = []
    N, M, alpha = 1000, 50, 0.2
    size = N // M
    
    for t in range(trials):
        np.random.seed(t * 1000 + 42)
        P = np.random.randn(N); P = P / np.linalg.norm(P)
        
        # 记录关键指标
        initial_noise = []
        W_evolution = []
        final_signals = []
        
        Ws = [np.random.randn(size,size)/np.sqrt(size) for _ in range(M)]
        
        # 隔离期
        for step in range(100):
            step_signals = []
            for m in range(M):
                sl = slice(m*size, (m+1)*size)
                xg = nl(Ws[m]@P[sl]+np.random.randn(size)*0.3)
                Ws[m] += 0.001*np.outer(xg,xg)
                if np.linalg.norm(Ws[m])>10: Ws[m]=Ws[m]/np.linalg.norm(Ws[m])*10
                step_signals.append(np.mean(xg))
            W_evolution.append(np.std(step_signals))
        
        # 结晶期
        s = (1-0.5)*P+0.5*np.random.randn(N)
        for step in range(200):
            ns = np.zeros(N)
            for m in range(M):
                sl = slice(m*size, (m+1)*size)
                ns[sl] = nl(Ws[m]@s[sl]+alpha*s[sl])
            s = ns
            final_signals.append(np.mean(s))
        
        final_mean = np.mean(s)
        final_std = np.std(s)
        results.append({
            'seed': t,
            'final_mean': final_mean,
            'final_std': final_std,
            'correct': 1 if np.sign(final_mean)==1 else 0,
            'evolution_std': np.mean(W_evolution),
            'convergence': np.std(final_signals[-50:]) if len(final_signals) > 50 else 0
        })
    
    return results

def seed_sensitivity():
    """种子敏感性分析"""
    
    print("="*60)
    print("SEED SENSITIVITY ANALYSIS")
    print("="*60)
    
    # 使用不同种子区间
    seed_ranges = [
        (0, 1000, "Small seeds (0-1000)"),
        (10000, 11000, "Medium seeds (10000-11000)"),
        (99999, 100999, "Large seeds (99999-100999)"),
    ]
    
    all_results = {}
    
    for start, end, name in seed_ranges:
        results = []
        N, M, alpha = 1000, 50, 0.2
        size = N // M
        
        for seed in range(start, min(end, start+10)):  # 每区间10个种子
            np.random.seed(seed)
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            Ws = [np.random.randn(size,size)/np.sqrt(size) for _ in range(M)]
            
            for _ in range(100):
                for m in range(M):
                    sl = slice(m*size, (m+1)*size)
                    xg = nl(Ws[m]@P[sl]+np.random.randn(size)*0.3)
                    Ws[m] += 0.001*np.outer(xg,xg)
                    if np.linalg.norm(Ws[m])>10: Ws[m]=Ws[m]/np.linalg.norm(Ws[m])*10
            
            for _ in range(200):
                s = (1-0.5)*P+0.5*np.random.randn(N)
                ns = np.zeros(N)
                for m in range(M):
                    sl = slice(m*size, (m+1)*size)
                    ns[sl] = nl(Ws[m]@s[sl]+alpha*s[sl])
                s = ns
            
            results.append(1 if np.sign(np.mean(s))==1 else 0)
        
        rate = np.mean(results) * 100
        all_results[name] = rate
        print(f"{name}: {rate:.1f}%")
    
    return all_results

def stability_factors():
    """稳定性因素分析"""
    
    print("\n" + "="*60)
    print("STABILITY FACTORS")
    print("="*60)
    
    factors = {
        'longer_isolation': [],
        'stronger_coupling': [],
        'larger_modules': [],
        'smaller_modules': [],
    }
    
    N, M_base = 1000, 50
    size_base = N // M_base
    
    # 1. 更长隔离期
    for isolation in [100, 200, 300, 500]:
        correct = 0
        for t in range(10):
            np.random.seed(t*100 + isolation)
            size = N // M_base
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            Ws = [np.random.randn(size,size)/np.sqrt(size) for _ in range(M_base)]
            for _ in range(isolation):
                for m in range(M_base):
                    sl = slice(m*size, (m+1)*size)
                    xg = nl(Ws[m]@P[sl]+np.random.randn(size)*0.3)
                    Ws[m] += 0.001*np.outer(xg,xg)
                    if np.linalg.norm(Ws[m])>10: Ws[m]=Ws[m]/np.linalg.norm(Ws[m])*10
            for _ in range(200):
                s = (1-0.5)*P+0.5*np.random.randn(N)
                ns = np.zeros(N)
                for m in range(M_base):
                    sl = slice(m*size, (m+1)*size)
                    ns[sl] = nl(Ws[m]@s[sl]+0.2*s[sl])
                s = ns
            if np.sign(np.mean(s))==1: correct += 1
        factors['longer_isolation'].append((isolation, correct*10))
    
    # 2. 更强耦合
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.8]:
        correct = 0
        for t in range(10):
            np.random.seed(t*100 + int(alpha*100))
            size = N // M_base
            P = np.random.randn(N); P = P / np.linalg.norm(P)
            Ws = [np.random.randn(size,size)/np.sqrt(size) for _ in range(M_base)]
            for _ in range(100):
                for m in range(M_base):
                    sl = slice(m*size, (m+1)*size)
                    xg = nl(Ws[m]@P[sl]+np.random.randn(size)*0.3)
                    Ws[m] += 0.001*np.outer(xg,xg)
                    if np.linalg.norm(Ws[m])>10: Ws[m]=Ws[m]/np.linalg.norm(Ws[m])*10
            for _ in range(200):
                s = (1-0.5)*P+0.5*np.random.randn(N)
                ns = np.zeros(N)
                for m in range(M_base):
                    sl = slice(m*size, (m+1)*size)
                    ns[sl] = nl(Ws[m]@s[sl]+alpha*s[sl])
                s = ns
            if np.sign(np.mean(s))==1: correct += 1
        factors['stronger_coupling'].append((alpha, correct*10))
    
    print("\n1. Isolation Length:")
    for iso, rate in factors['longer_isolation']:
        print(f"   {iso} steps: {rate:.1f}%")
    
    print("\n2. Coupling Strength:")
    for alpha, rate in factors['stronger_coupling']:
        print(f"   α={alpha}: {rate:.1f}%")
    
    return factors

if __name__ == "__main__":
    # 种子敏感性
    seed_sensitivity()
    
    # 稳定性因素
    stability_factors()
