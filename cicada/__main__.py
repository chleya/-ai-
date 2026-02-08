#!/usr/bin/env python3
"""
Cicada Protocol - Command Line Interface
"""

import argparse
from cicada import CicadaProtocol, compare_strategies, quick_demo


def main():
    parser = argparse.ArgumentParser(
        description="Cicada Protocol: Periodic reset for consensus stability"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different strategies"
    )
    parser.add_argument(
        "--N", "-N",
        type=int,
        default=200,
        help="System size (default: 200)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=800,
        help="Evolution steps (default: 800)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="Reset interval (default: 300)"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=5,
        help="Number of trials (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    elif args.compare:
        print("Comparing strategies...")
        results = compare_strategies(
            N=args.N,
            steps=args.steps,
            trials=args.trials
        )
        print("\nComparison complete!")
    else:
        # Default: run simple experiment
        print(f"Running experiment: N={args.N}, steps={args.steps}, interval={args.interval}")
        protocol = CicadaProtocol(
            N=args.N,
            reset_interval=args.interval,
            seed=42
        )
        W, s = protocol.evolve(steps=args.steps)
        stats = protocol.analyze()
        
        print(f"\nResults:")
        print(f"  Survival rate: {stats['survival_rate']:.1%}")
        print(f"  Final λ_max: {stats['final_lambda']:.4f}")
        print(f"  Reset count: {stats['reset_count']}")


if __name__ == "__main__":
    main()
