#!/usr/bin/env python3
"""
Cicada Protocol Setup
=====================

A Python library for maintaining long-term stability in distributed
consensus systems through periodic system reset.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cicada-protocol",
    version="0.1.0",
    author="Chen Leiyang",
    author_email="chleiyang@example.com",
    description="Periodic reset for consensus stability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chleya/-ai-",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cicada-run=cicada.__main__:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/chleya/-ai-/issues",
        "Source": "https://github.com/chleya/-ai-",
        "Documentation": "https://github.com/chleya/-ai-/tree/main/docs",
    },
)
