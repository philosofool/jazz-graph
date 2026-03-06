import torch
from ignite.metrics import Metric


class UniformityLoss(Metric):
    def __init__(self, t=2, output_transform=lambda x: x, device="cpu"):
        self.uniformity_sum = 0.0
        self.count = 0
        self.t = t
        super().__init__(output_transform, device)

    def reset(self):
        self.uniformity_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, output):
        z1, z2 = output

        # Compute uniformity for both views
        for z in [z1, z2]:
            sq_dist = torch.pdist(z, p=2).pow(2)
            uniformity = sq_dist.mul(-self.t).exp().mean().log()
            self.uniformity_sum += uniformity.item()
            self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.uniformity_sum / self.count


class EmbeddingStd(Metric):
    """
    Metric to track average standard deviation per dimension of embeddings.
    Low values indicate potential collapse.
    """
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.sum_embeddings = None
        self.sum_squared_embeddings = None
        self.count = 0
        super().__init__(output_transform, device)

    @torch.no_grad()
    def reset(self):
        self.sum_embeddings = None
        self.sum_squared_embeddings = None
        self.count = 0

    @torch.no_grad()
    def update(self, output) -> None:
        """
        Args:
            output: Embeddings tensor of shape [batch_size, embedding_dim]
        """
        embeddings = torch.concat(output)
        batch_size = embeddings.size(0)

        # Initialize on first batch
        if self.sum_embeddings is None:
            embed_dim = embeddings.size(1)
            self.sum_embeddings = torch.zeros(embed_dim, device=embeddings.device)
            self.sum_squared_embeddings = torch.zeros(embed_dim, device=embeddings.device)

        # Accumulate statistics
        self.sum_embeddings += embeddings.sum(dim=0)
        self.sum_squared_embeddings += (embeddings ** 2).sum(dim=0)  # pyright: ignore [reportOperatorIssue]
        self.count += batch_size

    @torch.no_grad()
    def compute(self) -> float:
        """
        Compute average standard deviation per dimension.

        Using: std = sqrt(E[X^2] - E[X]^2)
        """
        if self.count == 0:
            return 0.0

        # Compute mean per dimension
        mean = self.sum_embeddings / self.count   # pyright: ignore [reportOptionalOperand]

        # Compute variance per dimension
        mean_squared = self.sum_squared_embeddings / self.count   # pyright: ignore [reportOptionalOperand]
        variance = mean_squared - (mean ** 2)

        # Standard deviation per dimension
        std_per_dim = torch.sqrt(variance.clamp(min=1e-8))  # Clamp to avoid NaN

        # Average across dimensions
        avg_std = std_per_dim.mean()

        return avg_std.item()


class AlignmentLoss(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.alignment_sum = 0.0
        self.count = 0
        super().__init__(output_transform, device)

    @torch.no_grad()
    def update(self, output) -> None:
        z1, z2 = output
        # Sum of squared distances
        self.alignment_sum += (z1 - z2).pow(2).sum(dim=1).mean().item()
        self.count += z1.size(0)

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.alignment_sum / self.count

    def reset(self):
        self.alignment_sum = 0.0
        self.count = 0