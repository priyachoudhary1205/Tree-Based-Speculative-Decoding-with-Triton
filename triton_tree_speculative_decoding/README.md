# Triton Tree-Based Speculative Decoding

Research-oriented project scaffold for EAGLE/Medusa-style candidate trees and parallel verification.

## What is implemented
- Candidate token-tree representation
- Tree topology flattening
- Acceptance-rate controller that adapts draft depth
- PyTorch reference verifier
- Triton kernel skeleton for topology-aware verification

## Requirements
A CUDA-capable NVIDIA GPU is required for the Triton benchmark path.

```bash
pip install -r requirements.txt
python -m tests.test_tree
```

## Important
The Triton kernel is intentionally a correctness-oriented starting point. Production integration
requires model-specific KV-cache/tree-attention wiring and extensive numerical benchmarking.
