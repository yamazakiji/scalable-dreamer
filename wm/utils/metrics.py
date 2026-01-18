"""Evaluation metrics for image reconstruction."""
import torch
import torch.nn as nn

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
