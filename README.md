# 🧮 MCMC-Explorer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> *Navigating the complex landscapes of multi-modal probability distributions with Metropolis-Hastings MCMC.*

## Overview

This repository contains from-scratch implementations of MCMC samplers:
- **Metropolis-Hastings** with adaptive proposal tuning
- **Gibbs sampler** with Metropolis-within-Gibbs
- **Diagnostic tools** for convergence assessment
- **Benchmarking** against Hamiltonian Monte Carlo via Mici

## ✨ Key Features

- **Pure NumPy Implementation**: No black-box libraries for the core algorithm
- **Adaptive Proposal Tuning**: Robbins-Monro style adaptation for optimal acceptance rates
- **Comprehensive Visualizations**: Trace plots, acceptance rates, mode-switching behavior
- **Convergence Diagnostics**: Effective Sample Size (ESS), Gelman-Rubin R-hat
- **Multi-modal Test Cases**: Mixture of Gaussians, Banana-shaped distributions, Ring distributions
- **Benchmark Against HMC**: Comparison with [Mici](https://github.com/matt-graham/mici) (state-of-the-art HMC)
- **Gibbs Sampler**: Implementation of Gibbs sampling with Metropolis-within-Gibbs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcmc-explorer.git
cd mcmc-explorer

# Create virtual environment and install dependencies
uv venv
uv sync --all-extras
source .venv/bin/activate  # On Windows: .venv\Scripts\activate