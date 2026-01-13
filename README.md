# H-EFT-VA: An Effective-Field-Theory Variational Ansatz

[cite_start]Official implementation of the **H-EFT Variational Ansatz (H-EFT-VA)**, a quantum circuit architecture designed to avoid Barren Plateaus (BPs) using principles from Effective Field Theory (EFT)[cite: 39].

## Abstract
Variational Quantum Algorithms (VQAs) are critically threatened by the Barren Plateau phenomenon. [cite_start]In this work, we introduce H-EFT-VA, which enforces a hierarchical "UV-cutoff" on initialization to restrict state exploration and prevent the formation of unitary 2-designs[cite: 39, 40]. [cite_start]We provide rigorous proof of an inverse-polynomial lower bound on gradient variance while maintaining volume-law entanglement[cite: 41, 42].

## Benchmarks
The code includes 16 comprehensive experiments (T1-T16) comparing H-EFT-VA against Hardware-Efficient Ansätze (HEA). Key results include:
- [cite_start]**109x** lower energy convergence error[cite: 43].
- [cite_start]**10.7x** higher ground-state fidelity[cite: 43].
- [cite_start]**p-value** of $< 10^{-88}$[cite: 43].

## Installation
1. Clone the repository: `git clone https://github.com/your-username/H-EFT-VA.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
To run the full suite of 16 tests and generate all figures:
```bash
python src/main.py