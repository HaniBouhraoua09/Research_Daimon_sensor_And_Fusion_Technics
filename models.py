"""
models.py - Single-modality baselines + 3 fusion architectures.

Architectures:
    SingleModalityCNN  - Baseline (one CNN per modality)
    EarlyFusionCNN     - Concatenate all modalities at input (6 channels)
    LateFusionEnsemble - Average logits from 4 baselines
    HybridFusionNet    - Per-modality encoders, fuse at feature level
"""

import torch
import torch.nn as nn


# ============================================================
# SHARED CNN ENCODER
# ============================================================
class SmallCNN(nn.Module):
    """Compact CNN. Input: (B, C_in, H, W). Output: (B, feature_dim)."""
    
    def __init__(self, in_channels, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


# ============================================================
# 1. SINGLE-MODALITY BASELINE
# ============================================================
class SingleModalityCNN(nn.Module):
    """One CNN for one modality. Used for the 4 baselines."""
    
    MODALITY_CHANNELS = {'rawimg': 1, 'depth': 1, 'deformation': 2, 'shear': 2}
    
    def __init__(self, modality, num_classes=3):
        super().__init__()
        in_channels = self.MODALITY_CHANNELS[modality]
        self.modality = modality
        self.encoder = SmallCNN(in_channels)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, batch):
        x = batch[self.modality]
        feat = self.encoder(x)
        return self.classifier(feat)


# ============================================================
# 2. EARLY FUSION — concatenate at input
# ============================================================
class EarlyFusionCNN(nn.Module):
    """Stack all modalities as 6 channels and run one CNN."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        # 1 (rawimg) + 1 (depth) + 2 (deformation) + 2 (shear) = 6
        self.encoder = SmallCNN(in_channels=6)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, batch):
        x = torch.cat([batch['rawimg'], batch['depth'],
                       batch['deformation'], batch['shear']], dim=1)
        feat = self.encoder(x)
        return self.classifier(feat)


# ============================================================
# 3. LATE FUSION — average logits
# ============================================================
class LateFusionEnsemble(nn.Module):
    """Average logits from 4 modality-specific models."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.models = nn.ModuleDict({
            mod: SingleModalityCNN(mod, num_classes)
            for mod in ['rawimg', 'depth', 'deformation', 'shear']
        })
    
    def forward(self, batch):
        logits = [m(batch) for m in self.models.values()]
        return torch.stack(logits).mean(dim=0)


# ============================================================
# 4. HYBRID FUSION — concatenate features at bottleneck
# ============================================================
class HybridFusionNet(nn.Module):
    """Per-modality encoders → concatenate features → MLP head."""
    
    def __init__(self, num_classes=3, feature_dim=128):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'rawimg':      SmallCNN(1, feature_dim),
            'depth':       SmallCNN(1, feature_dim),
            'deformation': SmallCNN(2, feature_dim),
            'shear':       SmallCNN(2, feature_dim),
        })
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, batch):
        feats = [enc(batch[mod]) for mod, enc in self.encoders.items()]
        fused = torch.cat(feats, dim=1)
        return self.fusion(fused)


# ============================================================
# MODEL FACTORY
# ============================================================
def build_model(name, num_classes=3):
    if name in ['rawimg', 'depth', 'deformation', 'shear']:
        return SingleModalityCNN(name, num_classes)
    elif name == 'early_fusion':
        return EarlyFusionCNN(num_classes)
    elif name == 'late_fusion':
        return LateFusionEnsemble(num_classes)
    elif name == 'hybrid_fusion':
        return HybridFusionNet(num_classes)
    raise ValueError(f"Unknown model: {name}")


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    # Create a fake batch to test forward passes
    fake_batch = {
        'rawimg':      torch.randn(2, 1, 240, 320),
        'depth':       torch.randn(2, 1, 240, 320),
        'deformation': torch.randn(2, 2, 240, 320),
        'shear':       torch.randn(2, 2, 240, 320),
        'label':       torch.tensor([0, 1]),
    }
    
    for name in ['rawimg', 'depth', 'deformation', 'shear',
                 'early_fusion', 'late_fusion', 'hybrid_fusion']:
        model = build_model(name)
        out = model(fake_batch)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s} → output {tuple(out.shape)}, params: {n_params:,}")