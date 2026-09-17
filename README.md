# Tree-Based Speculative Decoding with Triton

**Adaptive token-tree drafting and parallel verification for LLM inference — with a custom Triton kernel for the verification step.**

<p>
<img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.2-ee4c2c">
<img alt="Triton" src="https://img.shields.io/badge/Triton-%E2%89%A52.2-76b900">
<img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
</p>

---

## Why this exists

Autoregressive decoding is bandwidth-bound: every generated token requires a full forward pass over billions of weights to produce a *single* token. Speculative decoding breaks that 1:1 ratio — a cheap draft model proposes several tokens, and the expensive target model verifies them **in one batched pass**.

Linear speculation (a single chain of guesses) has a hard ceiling: one wrong token invalidates the entire remaining chain. **Tree speculation** — the idea behind Medusa and EAGLE — proposes a *branching* set of continuations, so the verifier picks the longest accepted path through the tree instead of the longest accepted prefix of one guess. More candidate coverage, same number of target-model passes.

This repository implements the pieces that make that work:

1. a **candidate-tree data structure** and a flattening pass that turns it into GPU-friendly parallel arrays,
2. a **Triton kernel** that verifies all candidate nodes in parallel,
3. a **closed-loop controller** that tunes draft depth at runtime from measured acceptance rate and verification latency,
4. a **PyTorch reference implementation** that the kernel is checked against.

---

## Core idea

A draft produces a tree of candidate continuations rather than a chain:

```
              [the]                     depth 0
             /     \
        [cat]       [dog]               depth 1
        /    \          \
    [sat]   [ran]      [barked]         depth 2
```

`flatten()` linearizes this with an explicit iterative DFS into three parallel arrays — no recursion, no Python-side tree walk in the hot loop:

| array       | meaning                                         |
|-------------|-------------------------------------------------|
| `token_ids` | candidate token at each node, in flattened order |
| `parents`   | index of each node's parent (`-1` for the root)  |
| `depths`    | distance from root, for depth-wise masking       |

`parents` is the whole trick. It's exactly the information a tree-attention mask needs (a node may attend only to its ancestors), and it's what lets the verifier reconstruct accepted paths from a flat per-node accept/reject vector.

---

## What's implemented

| Component | File | Status |
|---|---|---|
| Candidate token-tree representation | `specdecode/tree.py` | ✅ Complete |
| Iterative DFS flattening → `(token_ids, parents, depths)` | `specdecode/tree.py` | ✅ Complete |
| Triton parallel verification kernel | `specdecode/kernel.py` | ✅ Runs, correctness-oriented |
| PyTorch reference verifier (ground truth for tests) | `specdecode/reference.py` | ✅ Complete |
| Adaptive draft-depth controller | `specdecode/controller.py` | ✅ Complete |
| Unit tests for all of the above | `tests/test_tree.py` | ✅ Passing |
| Fused tree-attention + KV-cache verification | — | 🚧 Roadmap |
| End-to-end integration with a real draft/target pair | — | 🚧 Roadmap |

> **Read this before citing throughput numbers.** This is a research prototype of the *tree and verification machinery*, not a drop-in inference server. The Triton kernel is written for clarity and correctness first: it does a blocked argmax scan over the vocabulary per candidate, which is a legitimate GPU verification kernel but not yet the fused tree-attention path a production engine needs. No speedup figures are claimed here because none have been measured end-to-end against a real target model — see [Roadmap](#roadmap).

---

## The Triton kernel

`tree_verify_kernel` answers one question for every candidate node at once: *did the target model's argmax at this position match the drafted token?*

```python
pid     = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)     # one candidate per lane
```

Each program handles a `BLOCK` of candidates. Because the vocabulary (30k–150k entries) will not fit in registers, the kernel **tiles the vocabulary dimension** and carries a running maximum:

```python
for start in range(0, vocab, BLOCK):
    values    = tl.load(logits_ptr + offsets[:, None] * vocab + v[None, :], ...)
    local_val = tl.max(values, axis=1)
    local_idx = tl.argmax(values, axis=1) + start
    better    = local_val > best_val
    best_val  = tl.where(better, local_val, best_val)   # branchless select
    best_idx  = tl.where(better, local_idx, best_idx)
```

Design notes worth pointing at:

- **Branchless updates.** `tl.where` instead of control flow keeps every lane in a warp on the same instruction path — divergence is the classic way to lose throughput in a kernel like this.
- **`-inf` masking.** Out-of-range vocabulary and out-of-range candidate lanes load `-inf` so they can never win the argmax, which removes the need for a separate bounds branch.
- **Boolean output.** The kernel writes a per-node accept mask; path reconstruction stays on the host, where the `parents` array makes it a cheap walk. Keeping policy out of the kernel means the acceptance rule can change (greedy → typical sampling → rejection sampling) without touching GPU code.

The kernel is guarded by a `try: import triton` so the package imports cleanly on CPU-only machines; `triton_verify()` raises a clear error instead of failing at import time.

---

## Adaptive draft controller

Draft depth is a live trade-off, not a constant. Deep trees win more tokens per target pass when the draft model agrees with the target, and waste compute when it doesn't. `DraftController` closes the loop on two measured signals:

```python
if acceptance_rate > 0.75 and verify_ms <= target_ms:
    depth += 1                       # drafting is cheap and paying off → go deeper
elif acceptance_rate < 0.45 or verify_ms > target_ms * 1.25:
    depth -= 1                       # wasted draft work or blown latency budget → pull back
```

Clamped to `[min_depth, max_depth]`. The asymmetry is deliberate: growth requires *both* high acceptance and latency headroom, while shrinking is triggered by *either* failure — an AIMD-style bias toward protecting the latency SLO, which is what actually matters in a serving path.

---

## Quickstart

Requires an NVIDIA GPU with CUDA for the Triton path. The tree, controller, and reference verifier run fine on CPU.

```bash
git clone https://github.com/<you>/triton-tree-speculative-decoding.git
cd triton-tree-speculative-decoding
pip install -r requirements.txt
pytest tests/ -v
```

Build and flatten a tree:

```python
from specdecode.tree import Node, flatten

tree = Node(1, [Node(2, [Node(4)]), Node(3)])
flat = flatten(tree)

flat.token_ids   # [1, 2, 4, 3]
flat.parents     # [-1, 0, 1, 0]
flat.depths      # [0, 1, 2, 1]
```

Verify candidates on GPU and compare against the reference:

```python
import torch
from specdecode.kernel import triton_verify
from specdecode.reference import verify_candidates

logits = torch.randn(len(flat.token_ids), 32000, device="cuda")
tokens = torch.tensor(flat.token_ids, device="cuda")

assert torch.equal(triton_verify(logits, tokens), verify_candidates(logits, tokens))
```

Drive the controller from live metrics:

```python
from specdecode.controller import DraftController

ctrl = DraftController(min_depth=2, max_depth=12, initial=4)
depth = ctrl.update(acceptance_rate=0.82, verify_ms=6.1)   # → 5
```

---

## Layout

```
triton_tree_speculative_decoding/
├── specdecode/
│   ├── tree.py          Node / FlatTree + iterative DFS flattening
│   ├── kernel.py        Triton verification kernel + Python wrapper
│   ├── reference.py     PyTorch ground-truth verifier
│   └── controller.py    adaptive draft-depth controller
├── tests/test_tree.py   flattening, controller, and verifier tests
└── requirements.txt
```

---

## Roadmap

- [ ] **Fused tree-attention kernel** — build the attention mask directly from `parents` so verification is a single fused pass instead of argmax-over-precomputed-logits
- [ ] **KV-cache integration** — tree-aware cache layout with rollback of rejected branches
- [ ] **End-to-end harness** — a real draft/target pair (e.g. a 1B draft against a 7B target), measuring tokens/sec, acceptance rate, and wall-clock speedup against vanilla decoding
- [ ] **Sampling-based acceptance** — rejection sampling for temperature > 0, replacing the current greedy argmax rule
- [ ] **Autotuning** — `triton.autotune` over `BLOCK` and `num_warps` across vocabulary sizes
- [ ] **Learned tree topology** — EAGLE-style draft head that shapes the tree instead of using a fixed fan-out

---

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2022)
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (2024)
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* (2024)
- Miao et al., *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference* (2023)

## License

MIT
