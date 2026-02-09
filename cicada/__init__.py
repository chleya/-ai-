"""
Cicada Protocol
===============

A framework for stabilizing Hebbian learning through periodic reset.

The core insight: Hebbian learning induces spectral growth (lambda_max increase),
which we formalize through the H-Theorem. The Cicada Protocol combats this by
injecting "negative entropy" through discrete reset operations.

Key Formula:
    lambda_max(N) = 0.015 * N^0.72

Example:
    >>> from cicada import CicadaProtocol
    >>> cicada = CicadaProtocol(N=787, theta=1.2, alpha=1.0)
    >>> W = cicada.step(W, s)

Author: Chen Leiyang, OpenClaw
License: MIT
"""

from cicada.core import CicadaProtocol

__version__ = "1.0.0"
__author__ = "Chen Leiyang, OpenClaw"
__license__ = "MIT"

__all__ = ["CicadaProtocol"]
