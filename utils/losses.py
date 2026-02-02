import torch
import torch.nn as nn
import torch.nn.functional as F


class CycleConsistencyLoss(nn.Module):
    """
    Penalizes hallucination by enforcing SR → LR consistency.
    If model invents fake structures, they won't survive downsampling.
    """
    def __init__(self, scale=4):
        super(CycleConsistencyLoss, self).__init__()
        self.scale = scale

    def forward(self, sr, lr):
        """
        Args:
            sr: Super-resolved image  [B, C, H*scale, W*scale]
            lr: Original low-res      [B, C, H, W]
        """
        sr_downsampled = F.interpolate(
            sr,
            size=(lr.shape[2], lr.shape[3]),
            mode='bicubic',
            align_corners=False
        )
        return F.l1_loss(sr_downsampled, lr)


class CombinedLoss(nn.Module):
    """
    L_total = L1(SR, HR) + λ * CycleConsistency(SR, LR)
    """
    def __init__(self, scale=4, lambda_cycle=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.cycle_loss = CycleConsistencyLoss(scale=scale)
        self.lambda_cycle = lambda_cycle

    def forward(self, sr, hr, lr):
        l1 = self.l1_loss(sr, hr)
        cycle = self.cycle_loss(sr, lr)
        total = l1 + self.lambda_cycle * cycle
        return total, l1, cycle


# --- Quick smoke test ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = CombinedLoss(scale=4, lambda_cycle=0.1)
    sr = torch.rand(2, 3, 512, 512).to(device)
    hr = torch.rand(2, 3, 512, 512).to(device)
    lr = torch.rand(2, 3, 128, 128).to(device)

    loss, l1, cycle = criterion(sr, hr, lr)
    print(f"✅ Total: {loss:.4f}, L1: {l1:.4f}, Cycle: {cycle:.4f}")
