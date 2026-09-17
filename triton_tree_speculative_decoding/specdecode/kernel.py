import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton:
    @triton.jit
    def tree_verify_kernel(
        logits_ptr, token_ptr, out_ptr,
        stride_n: tl.constexpr, vocab: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < stride_n
        tokens = tl.load(token_ptr + offsets, mask=mask, other=-1)
        # Model-specific production version should perform fused tree-attention/KV verification.
        # This baseline computes argmax by scanning vocab for each candidate.
        best_val = tl.full([BLOCK], -float("inf"), tl.float32)
        best_idx = tl.zeros([BLOCK], tl.int32)
        for start in range(0, vocab, BLOCK):
            v = start + tl.arange(0, BLOCK)
            vm = v < vocab
            values = tl.load(logits_ptr + offsets[:, None] * vocab + v[None, :],
                             mask=mask[:, None] & vm[None, :], other=-float("inf"))
            local_val = tl.max(values, axis=1)
            local_idx = tl.argmax(values, axis=1) + start
            better = local_val > best_val
            best_val = tl.where(better, local_val, best_val)
            best_idx = tl.where(better, local_idx, best_idx)
        tl.store(out_ptr + offsets, best_idx == tokens, mask=mask)

def triton_verify(logits: torch.Tensor, tokens: torch.Tensor):
    if triton is None:
        raise RuntimeError("Triton is not installed")
    out = torch.empty(tokens.numel(), device=logits.device, dtype=torch.bool)
    n, vocab = logits.shape
    grid = (triton.cdiv(n, 32),)
    tree_verify_kernel[grid](logits, tokens, out, stride_n=n, vocab=vocab, BLOCK=32)
    return out
