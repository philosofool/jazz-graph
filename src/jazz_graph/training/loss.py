# NT_Xent loss. Used in SimCLR.
import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss for SimCLR.

    For each sample i:
    - Positive pair: (z1[i], z2[i]) - same image, different augmentations
    - Negative pairs: All other samples in the batch

    Goal: Pull positives together, push negatives apart
    """
    batch_size = z1.size(0)

    # Concatenate both views: [2N, dim]
    z = torch.cat([z1, z2], dim=0)

    # We expect normalized embeddings in z, so dot product and cosine similarity are same:
    # sim[i,j] = cosine_similarity(z[i], z[j]) / temperature
    sim_matrix = torch.mm(z, z.t()) / temperature

    labels = torch.cat([
        torch.arange(batch_size) + batch_size,   # z1[i] matches z2[i]
        torch.arange(batch_size)                 # z2[i] matches z1[i]
    ])

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    # NOTE: internally, cross-entropy loss is in log space: entropy(-inf) == 0
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # Cross-entropy: treat it as a classification problem
    # "Which of the 2N-1 other samples is the positive pair?"
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def nt_xent_loss_with_masking(z: torch.Tensor, positives_mask, temperature: float = 0.5) -> torch.Tensor:
    # Credit for vectorize, logsumexp approach to Claude.ai
    batch_size = z.size(0)
    logits = torch.mm(z, z.t()) / temperature
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -float('inf'))
    positives_mask = positives_mask & ~self_mask
    if torch.sum(positives_mask.to(torch.float32)) == 0.:
        return torch.tensor(0.)

    pos_logits = logits.clone()
    pos_logits[~positives_mask] = -float('inf')
    log_pos = torch.logsumexp(pos_logits, dim=1)

    log_all = torch.logsumexp(logits, dim=1)
    loss_per_sample = -(log_pos - log_all)
    has_positives = positives_mask.sum(dim=1) > 0
    return loss_per_sample[has_positives].mean()

def nt_xent_loss_with_masking_(z, positive_mask, temperature=0.5):
    """
    Fully vectorized multi-positive NT-Xent loss.

    Args:
        z: [batch_size, dim] - normalized embeddings
        positive_mask: [batch_size, batch_size] - boolean mask of positives
        temperature: scalar

    Returns:
        Scalar loss
    """
    batch_size = z.size(0)
    device = z.device

    sim_matrix = torch.mm(z, z.t()) / temperature

    mask_diag = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim_matrix_no_diag = sim_matrix.masked_fill(mask_diag, -float('inf'))

    positive_mask_no_diag = positive_mask & ~mask_diag
    if torch.sum(positive_mask_no_diag.to(torch.float32)) == 0.:
        return torch.tensor(0.)

    # For numerical stability: subtract max before exp
    sim_max, _ = sim_matrix_no_diag.max(dim=1, keepdim=True)
    sim_matrix_stable = sim_matrix_no_diag - sim_max

    exp_sim = torch.exp(sim_matrix_stable)

    denom = exp_sim.sum(dim=1, keepdim=True)

    numer = (exp_sim * positive_mask_no_diag).sum(dim=1, keepdim=True)

    has_positives = positive_mask_no_diag.sum(dim=1) > 0

    # Loss: -log(numer / denom) = log(denom) - log(numer)
    loss_per_sample = -torch.log(numer / denom)

    # Average only over samples that have positives
    loss = loss_per_sample[has_positives].mean()

    return loss