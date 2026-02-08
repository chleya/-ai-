#!/usr/bin/env python3
"""Quick test of cicada_protocol"""

import numpy as np
from cicada.core import cicada_protocol, analyze_spectrum

np.random.seed(42)
W, s = cicada_protocol(N=100, reset_interval=100, total_steps=200)
spec = analyze_spectrum(W)

print('SUCCESS!')
print('lambda_max:', round(spec['max'], 4))
print('mean state:', round(float(np.mean(s)), 4))
