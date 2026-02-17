# Exact Combinatorial Multi-Class Graph Cuts for Semi-Supervised Learning (AAAI 2026)

[Paper](https://sharif.edu/~aamini/Papers/ExactGraphCut.pdf) 

This repository is the official implementation of "Exact Combinatorial Multi-Class Graph Cuts for Semi-Supervised Learning".


## Overview

**CombCut** is the first **exact combinatorial optimization framework** for multi-class graph-based semi-supervised learning (SSL). It is specifically designed for extreme label sparsity and class imbalance — regimes where existing graph-SSL methods often collapse to trivial or unstable solutions.

Unlike prior approaches that rely on convex relaxations, spectral approximations, or heuristic balancing constraints, CombCut operates **directly on discrete one-hot label assignments**. The method formulates multi-class SSL as a true combinatorial optimization problem and solves it without relaxing the discrete structure.

At the core of CombCut is a **minorization–maximization (MM) scheme** that transforms each optimization step into a structured linear assignment problem. Each subproblem is solved efficiently using **network-flow algorithms**, and total unimodularity guarantees that all iterates remain integral — eliminating the need for rounding or projection steps.

We provide rigorous theoretical guarantees:

- Monotonic ascent of the original discrete objective  
- Integrality of all intermediate solutions  
- Convergence of every limit point to a **Karush–Kuhn–Tucker (KKT) stationary solution** of the original combinatorial formulation  

Importantly, CombCut:

- Requires **no hyperparameter tuning**
- Scales near-linearly in the number of graph vertices
- Remains stable under severe supervision constraints

Empirically, on **MNIST**, **Fashion-MNIST**, and **CIFAR-10**, with as few as **1–5 labeled samples per class**, CombCut significantly outperforms state-of-the-art graph-based SSL baselines, particularly in worst-case labeling scenarios.



## Installation
Our codebase is largely built on [GraphLearning](https://github.com/jwcalder/GraphLearning/tree/master), so at first we need to install it.
Follow its instruction or simply do as follows:

### 1. GraphLearning Installation
```bash
sudo apt install build-essential python3-dev
conda create -n combcut python=3.10
conda activate combcut
conda install -c conda-forge gcc gxx        
git clone https://github.com/jwcalder/GraphLearning
cd GraphLearning
pip install -r requirements.txt            
pip install .
```
after running last command there should be a file named "cextensions.cpython-310-x86_64-linux-gnu.so" at graphlearning directory, if it was not try:
```bash
pip install -e .
```
in order to make sure, you installed it right, run this command:
```bash
python -c "import graphlearning; print('OK')"
```

### 2. cumbcut files
```bash
git clone https://github.com/yasin-sala/CombCut.git
cp -r CombCut/CombCut_graphlearning/* graphlearning/
cp -r CombCut/Results .
cp -r CombCut/RunTime .
```

### 3. run the code
In order to get the results of the paper and trying the **CombCut** algorithm
run any of the [.py](Results/) or to get the run time of the algorithm see [RunTime Folder](RunTime/) 

 ## Citation
@inproceedings{yourname2026exact,
  title={Exact Combinatorial Multi-Class Graph Cuts for Semi-Supervised Learning},
  author={Your Name},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
} 
