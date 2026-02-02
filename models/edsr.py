import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + res


class EDSR(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=64):
        super(EDSR, self).__init__()

        # Shallow feature extraction
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)

        # Residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats) for _ in range(n_resblocks)]
        )
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Upsampling
        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2)
            )

        # HR reconstruction
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)

        res = x
        x = self.body(x)
        x = self.body_tail(x)
        x = x + res

        x = self.upscale(x)
        x = self.tail(x)

        return x


# --- Quick smoke test ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EDSR(scale=4, n_resblocks=16, n_feats=64).to(device)
    lr = torch.rand(1, 3, 128, 128).to(device)
    sr = model(lr)
    print(f"✅ EDSR output shape: {sr.shape}")  # expect [1, 3, 512, 512]
