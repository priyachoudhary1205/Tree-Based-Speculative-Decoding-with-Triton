import torch

def verify_candidates(logits: torch.Tensor, candidate_tokens: torch.Tensor) -> torch.Tensor:
    # logits: [N, vocab], candidate_tokens: [N]
    predicted = logits.argmax(dim=-1)
    return predicted.eq(candidate_tokens)
