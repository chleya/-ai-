"""
DISSECTING FAILURE: Energy Landscape Analysis
解剖失败：提取成功与失败案例的演化曲线对比
"""

import numpy as np, json

def nl(x): return np.tanh(x)

def dissect_case(seed, N=1000, n_modules=50, alpha=0.2):
    """解剖单个案例，记录完整演化过程"""
    
    np.random.seed(seed)
    module_size = N // n_modules
    P = np.random.randn(N); P = P / np.linalg.norm(P)
    
    # 记录演化过程
    history = {
        'seed': seed,
        'phases': {
            'isolation': [],
            'crystallization': []
        },
        'energy': {
            'isolation': [],
            'crystallization': []
        },
        'module_signals': {
            'isolation': [],
            'crystallization': []
        }
    }
    
    # 初始化
    W_modules = []
    for m in range(n_modules):
        W = np.random.randn(module_size, module_size) / np.sqrt(module_size)
        W_modules.append(W)
    
    # ==================== 隔离期 ====================
    for step in range(100):
        step_data = {
            'step': step,
            'signals': [],
            'energy': 0
        }
        
        for m in range(n_modules):
            idx = slice(m*module_size, (m+1)*module_size)
            P_m = P[idx]
            x_good = nl(W_modules[m] @ P_m + np.random.randn(module_size) * 0.3)
            W_modules[m] += 0.001 * np.outer(x_good, x_good)
            if np.linalg.norm(W_modules[m]) > 10:
                W_modules[m] = W_modules[m] / np.linalg.norm(W_modules[m]) * 10
            
            step_data['signals'].append(np.mean(x_good))
            step_data['energy'] += np.linalg.norm(W_modules[m])
        
        # 计算模块间同步度
        signals = step_data['signals']
        sync = 1 - np.std(signals) / (np.abs(np.mean(signals)) + 1e-6)
        step_data['sync'] = sync
        
        history['phases']['isolation'].append(step_data)
        history['energy']['isolation'].append(step_data['energy'])
        history['module_signals']['isolation'].append(step_data['signals'])
    
    # ==================== 结晶期 ====================
    s = (1 - 0.5) * P + 0.5 * np.random.randn(N)
    
    for step in range(200):
        step_data = {
            'step': step,
            'global_mean': np.mean(s),
            'global_std': np.std(s),
            'sync': 0
        }
        
        new_s = np.zeros(N)
        module_means = []
        
        for m in range(n_modules):
            idx = slice(m*module_size, (m+1)*module_size)
            s_m = s[idx]
            new_s[idx] = nl(W_modules[m] @ s_m + alpha * s_m)
            module_means.append(np.mean(new_s[idx]))
        
        s = new_s
        
        # 计算全局同步度
        step_data['sync'] = 1 - np.std(module_means) / (np.abs(np.mean(module_means)) + 1e-6)
        
        history['phases']['crystallization'].append(step_data)
        history['energy']['crystallization'].append(np.linalg.norm(s))
        history['module_signals']['crystallization'].append(module_means)
    
    # 最终结果
    final_mean = np.mean(s)
    final_sign = np.sign(final_mean)
    history['final_mean'] = final_mean
    history['final_sign'] = int(final_sign)
    history['correct'] = 1 if final_sign == 1 else 0
    
    # 关键转折点检测
    history['critical_point'] = detect_critical_point(history)
    
    return history

def detect_critical_point(history):
    """检测关键转折点"""
    
    # 隔离期结束时的状态
    iso_end = history['phases']['isolation'][-1]
    iso_sync = iso_end['sync']
    
    # 结晶期前50步的同步度变化
    cryo_start = history['phases']['crystallization'][:50]
    sync_changes = [s['sync'] for s in cryo_start]
    
    # 检测是否有"相位脱节"
    if iso_sync < 0.5 and np.mean(sync_changes) < 0.5:
        return 'phase_desync'
    
    # 检测是否有"错误锁定"
    if history['final_sign'] == -1:
        return 'wrong_attractor'
    
    # 检测是否成功
    if history['correct'] == 1:
        return 'success'
    
    return 'unknown'

def run_dissection(trials=20):
    """运行案例解剖"""
    
    print("="*60)
    print("DISSECTING FAILURE: Energy Landscape Analysis")
    print("="*60)
    
    cases = {
        'success': [],
        'failure': []
    }
    
    # 收集成功和失败案例
    for seed in range(trials * 10):  # 尝试更多种子
        case = dissect_case(seed)
        
        if case['correct'] == 1:
            cases['success'].append(case)
        else:
            cases['failure'].append(case)
        
        if len(cases['success']) >= 5 and len(cases['failure']) >= 5:
            break
    
    # 分析
    print(f"\n收集到 {len(cases['success'])} 个成功案例, {len(cases['failure'])} 个失败案例")
    
    # 计算统计
    results = {
        'success_stats': compute_stats(cases['success']),
        'failure_stats': compute_stats(cases['failure']),
        'cases': {
            'success': [c['seed'] for c in cases['success']],
            'failure': [c['seed'] for c in cases['failure']]
        }
    }
    
    return results, cases

def compute_stats(cases):
    """计算案例统计"""
    
    if not cases:
        return None
    
    # 平均同步度
    iso_syncs = [c['phases']['isolation'][-1]['sync'] for c in cases]
    cryo_syncs = [np.mean([s['sync'] for s in c['phases']['crystallization'][:50]]) for c in cases]
    
    return {
        'count': len(cases),
        'iso_end_sync_mean': np.mean(iso_syncs),
        'iso_end_sync_std': np.std(iso_syncs),
        'cryo_early_sync_mean': np.mean(cryo_syncs),
        'cryo_early_sync_std': np.std(cryo_syncs),
    }

def compare_cases(cases):
    """对比成功与失败案例"""
    
    print("\n" + "="*60)
    print("CASE COMPARISON")
    print("="*60)
    
    success = cases['success']
    failure = cases['failure']
    
    # 关键指标对比
    print("\n1. 隔离期结束同步度:")
    print(f"   成功案例: {np.mean([c['phases']['isolation'][-1]['sync'] for c in success]):.3f} ± {np.std([c['phases']['isolation'][-1]['sync'] for c in success]):.3f}")
    print(f"   失败案例: {np.mean([c['phases']['isolation'][-1]['sync'] for c in failure]):.3f} ± {np.std([c['phases']['isolation'][-1]['sync'] for c in failure]):.3f}")
    
    print("\n2. 结晶期前50步同步度:")
    success_cryo = [np.mean([s['sync'] for s in c['phases']['crystallization'][:50]]) for c in success]
    failure_cryo = [np.mean([s['sync'] for s in c['phases']['crystallization'][:50]]) for c in failure]
    print(f"   成功案例: {np.mean(success_cryo):.3f} ± {np.std(success_cryo):.3f}")
    print(f"   失败案例: {np.mean(failure_cryo):.3f} ± {np.std(failure_cryo):.3f}")
    
    print("\n3. 关键转折点分布:")
    success_types = [c['critical_point'] for c in success]
    failure_types = [c['critical_point'] for c in failure]
    print(f"   成功: {success_types}")
    print(f"   失败: {failure_types}")
    
    # 生成对比报告
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if np.mean([c['phases']['isolation'][-1]['sync'] for c in failure]) < np.mean([c['phases']['isolation'][-1]['sync'] for c in success]):
        print("发现: 失败案例在隔离期结束时同步度更低")
        print("诊断: 相位脱节")
    
    if np.mean(failure_cryo) < np.mean(success_cryo):
        print("发现: 失败案例在结晶期无法保持同步")
        print("诊断: 错误吸引子锁定")
    
    return {
        'success': success,
        'failure': failure,
        'iso_sync_diff': np.mean([c['phases']['isolation'][-1]['sync'] for c in failure]) - np.mean([c['phases']['isolation'][-1]['sync'] for c in success]),
        'cryo_sync_diff': np.mean(failure_cryo) - np.mean(success_cryo)
    }

def generate_report(results, comparison):
    """生成分析报告"""
    
    report = {
        'summary': {
            'total_trials': results['success_stats']['count'] + results['failure_stats']['count'],
            'success_rate': results['success_stats']['count'] / (results['success_stats']['count'] + results['failure_stats']['count']) * 100
        },
        'success_analysis': results['success_stats'],
        'failure_analysis': results['failure_stats'],
        'comparison': comparison,
        'diagnosis': {
            'phase_desync': bool(comparison['iso_sync_diff'] < 0),
            'attractor_lock': bool(comparison['cryo_sync_diff'] < 0)
        }
    }
    
    return report

if __name__ == "__main__":
    # 运行案例解剖
    results, cases = run_dissection(trials=20)
    
    # 对比分析
    comparison = compare_cases(cases)
    
    # 生成报告
    report = generate_report(results, comparison)
    
    # 保存
    with open('results/dissection_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nSaved to results/dissection_report.json")
    
    # 打印诊断
    print("\n" + "="*60)
    print("FINAL DIAGNOSIS")
    print("="*60)
    
    if report['diagnosis']['phase_desync']:
        print("诊断: 相位脱节 (Phase Desynchronization)")
        print("     失败案例在隔离期结束时同步度显著低于成功案例")
        print("     建议: 增加隔离期或使用更强的LTD")
    
    if report['diagnosis']['attractor_lock']:
        print("诊断: 错误吸引子锁定 (Wrong Attractor Lock)")
        print("     失败案例在结晶期无法保持同步")
        print("     建议: 增加LTD强度或使用多次LTD迭代")
