"""
FiRe-MPO loss helpers matching paper Eq. 3–5 and Appendix B.

  L = L_rank + α · L_KL
  L_rank = -log σ(Σ r(v,y+) - Σ r(v,y-))
         + γ · -log σ(Σ r(v',y-) - Σ r(v',y+))
  L_KL   = Σ_t [ λ KL(π_ref ‖ π_θ) + (1-λ) KL(π_θ ‖ π_ref) ]  (token-wise, phrase-masked)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def token_wise_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample sum of token-wise KL(p ‖ q) over vocabulary, masked by ``loss_mask``.

    Args:
        logits_p: [B, S, V] distribution P (numerator / target of KL)
        logits_q: [B, S, V] distribution Q
        loss_mask: [B, S] boolean/float mask over completion positions
    Returns:
        [B] summed KL over masked positions
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    per_position = (p * (log_p - log_q)).sum(dim=-1)
    m = loss_mask.to(per_position.device).bool()
    return (per_position * m.float()).sum(dim=-1)


def compute_bidirectional_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    policy_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    lambda_: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bidirectional token-wise KL (Eq. 4).

    Returns:
        (forward_kl, reverse_kl, mixed_kl) each [B]
        mixed = λ · KL(ref‖π) + (1-λ) · KL(π‖ref)
    """
    # FKL: KL(ref ‖ π) — mean-seeking
    fkl = token_wise_kl(ref_logits, policy_logits, policy_mask)
    # RKL: KL(π ‖ ref) — mode-seeking
    rkl = token_wise_kl(policy_logits, ref_logits, ref_mask)
    mixed = float(lambda_) * fkl + (1.0 - float(lambda_)) * rkl
    return fkl, rkl, mixed


def combine_fire_mpo_losses(
    base_rank_loss: torch.Tensor,
    *,
    visual_rank_loss: Optional[torch.Tensor] = None,
    visual_weight: Optional[torch.Tensor] = None,
    gamma: Optional[float] = None,
    mixed_kl: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Compose total FiRe-MPO loss (Eq. 5): L_rank + α L_KL.

    ``visual_rank_loss`` is the (v', y-) ≻ (v', y+) term; scaled by γ and
    optionally masked by ``visual_weight`` (has_rejected_image).
    """
    losses = base_rank_loss
    if (
        visual_rank_loss is not None
        and gamma is not None
        and gamma > 0
    ):
        term = gamma * visual_rank_loss
        if visual_weight is not None:
            term = term * visual_weight
        losses = losses + term
    if mixed_kl is not None and alpha is not None and alpha > 0:
        losses = losses + alpha * mixed_kl
    return losses
