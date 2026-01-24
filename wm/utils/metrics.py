"""Evaluation metrics for image reconstruction and latent space."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSMetric(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.

    Measures perceptual similarity between images using deep features.
    Lower values indicate more similar images.
    """

    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Initialize LPIPS metric.

        Args:
            net: Network to use for feature extraction ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        super().__init__()
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package is required for LPIPS metric. "
                "Install it with: pip install lpips"
            )

        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance.

        Args:
            pred: Predicted images [B, C, H, W] in range [0, 1]
            target: Target images [B, C, H, W] in range [0, 1]

        Returns:
            LPIPS distance (lower is better)
        """
        # LPIPS expects images in range [-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1

        return self.model(pred, target).mean()


def compute_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: LPIPSMetric | None = None
) -> dict:
    """
    Compute multiple reconstruction metrics.

    Args:
        pred: Predicted images [B, T, C, H, W] in range [0, 1]
        target: Target images [B, T, C, H, W] in range [0, 1]
        lpips_model: Optional LPIPS model for perceptual metric

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # MSE
    mse = torch.nn.functional.mse_loss(pred, target)
    metrics['mse'] = mse.item()

    # PSNR
    psnr = -10 * torch.log10(mse)
    metrics['psnr'] = psnr.item()

    # LPIPS (if model provided)
    if lpips_model is not None:
        B, T, C, H, W = pred.shape

        # Flatten batch and time dimensions
        pred_flat = pred.view(B * T, C, H, W)
        target_flat = target.view(B * T, C, H, W)

        lpips_score = lpips_model(pred_flat, target_flat)
        metrics['lpips'] = lpips_score.item()

    return metrics


def compute_latent_metrics(z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
    """
    Compute detailed metrics between predicted and target latents.

    Args:
        z_pred: (B, N, D) or (B, 1, N, D) predicted latent
        z_target: (B, N, D) or (B, 1, N, D) target latent

    Returns:
        Dictionary with keys: mse, cosine_sim, per_token_mse, relative_error
    """
    # Ensure consistent shape
    if z_pred.dim() == 4:
        z_pred = z_pred.squeeze(1)
    if z_target.dim() == 4:
        z_target = z_target.squeeze(1)

    # MSE
    mse = F.mse_loss(z_pred, z_target).item()

    # Cosine similarity (flatten spatial dims)
    z_pred_flat = z_pred.flatten(start_dim=1)  # (B, N*D)
    z_target_flat = z_target.flatten(start_dim=1)  # (B, N*D)
    cos_sim = F.cosine_similarity(z_pred_flat, z_target_flat, dim=-1).mean().item()

    # Per-token MSE: average across batch and latent dim, keep spatial
    per_token_mse = ((z_pred - z_target) ** 2).mean(dim=(0, 2))  # (N,)

    # Relative error (RMSE normalized by target magnitude)
    target_magnitude = (z_target ** 2).mean().item()
    rel_error = (mse / (target_magnitude + 1e-8)) ** 0.5

    return {
        "mse": mse,
        "cosine_sim": cos_sim,
        "per_token_mse": per_token_mse.tolist(),
        "relative_error": rel_error,
    }
