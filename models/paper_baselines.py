"""Official-code SCOPE and an explicitly assumed Zhao MOT17 adaptation.

Zhao Figure 2: RGB temporal channel stacking, 16 features, spatial resblock,
conditional fusion and RGB residual. Four-frame context includes future frames
within each independently processed clip. This is not an authors' checkpoint.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SCOPE_REVISION = "453a40bebd73d4a7c3ad4436370e96f412a0a546"
SCOPE_HASHES = {
    "model.py": "d419cf9bde23302f6c8ffd5c5ab423278fd56d026601daa534d06b92b14c3317",
    "losses.py": "f8e98bd000da8cffcf68a21d9bb1065d28bf1353a73f56e97e19d47077eeca73",
}


def official_parts(root):
    """Verify LF-normalized upstream source before importing executable code."""
    modules = []
    for filename, expected in SCOPE_HASHES.items():
        path = Path(root) / "scope" / filename
        if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Official SCOPE source hash mismatch: {filename}")
        spec = importlib.util.spec_from_file_location("official_scope_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


class OfficialScopeClip(nn.Module):
    def __init__(self, root):
        super().__init__()
        source, self.losses = official_parts(root)
        self.network = source.SCOPE(num_codec_families=1, base_ch=48,
            attention_bias=-4.0, bg_mode="blur_replace", bg_blur_kernel=51)

    def forward(self, video, severity):
        b, t, c, h, w = video.shape
        family = torch.zeros(b, dtype=torch.long, device=video.device)
        frames = []
        for i in range(t):
            if self.training and torch.is_grad_enabled():
                values = checkpoint(self.network, video[:, i], severity, family, use_reentrant=False)
            else:
                values = self.network(video[:, i], severity, family)
            frames.append(values)
        return {name: torch.stack([v[j] for v in frames], dim=1)
                for j, name in enumerate(("edited", "delta_obj", "delta_bg", "attention", "effective_delta"))}


class ZhaoVideoPreprocessor(nn.Module):
    def __init__(self, clip_len=4):
        super().__init__()
        self.clip_len = clip_len
        self.spatial = nn.Conv2d(3, 16, 3, padding=1)
        self.resblock = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 16, 3, padding=1))
        self.temporal = nn.Conv2d(3 * clip_len, 16, 3, padding=1)
        self.attention = nn.Conv2d(32, 16, 1)
        self.output = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(16, 3, 3, padding=1))

    def forward(self, video, severity=None):
        b, t, c, h, w = video.shape
        if t != self.clip_len or c != 3:
            raise ValueError("Zhao reproduction expects four RGB frames")
        temporal = F.relu(self.temporal(video.reshape(b, t * c, h, w)))
        images = video.reshape(b * t, c, h, w)
        spatial = F.relu(self.spatial(images))
        spatial = spatial + self.resblock(spatial)
        temporal = temporal[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, 16, h, w)
        gate = torch.sigmoid(self.attention(torch.cat((spatial, temporal), dim=1)))
        # Attention conditions the temporal contribution; spatial path remains residual.
        residual = self.output(spatial + gate * temporal)
        return {"edited": (images + residual).clamp(0, 1).reshape_as(video)}


def zhao_objective(original, reconstructed, bpp, task_loss):
    return 10.0 * (F.mse_loss(reconstructed.float(), original.float()) + 0.001 * bpp) + task_loss


def feature_consistency(teacher, original, decoded, mask=None):
    """Native SCOPE normalized smooth-L1, p2-p5; Zhao uses full-frame mean."""
    with torch.no_grad():
        target = teacher.extract_backbone_fpn_features(original)
    student = teacher.extract_backbone_fpn_features(decoded)
    values = []
    for key in ("p2", "p3", "p4", "p5"):
        a, b = student[key].float(), target[key].float()
        scale = b.abs().mean().clamp_min(1e-3)
        loss = F.smooth_l1_loss(a / scale, b / scale, beta=0.5, reduction="none")
        if mask is None:
            values.append(loss.mean())
        else:
            roi = F.interpolate(mask.float(), size=a.shape[-2:], mode="bilinear", align_corners=False)
            local = (loss * roi).sum() / (roi.sum() * a.shape[1]).clamp_min(1e-6)
            values.append(0.70 * local + 0.30 * loss.mean())
    return torch.stack(values).mean()


def scope_objective(model, result, original, reconstructed, bpp, masks, teacher):
    """Native full-launcher weights; common teacher and video proxy adaptations."""
    helper = model.losses
    flatten = lambda x: x.flatten(0, 1)
    image, edited, decoded = map(flatten, (original, result["edited"], reconstructed))
    hard = flatten(masks)
    tight = helper.soften_mask(hard, 3)
    context = helper.soften_mask(hard, 41)
    background = 1.0 - context
    attention, delta = map(flatten, (result["attention"], result["effective_delta"]))
    target = helper.background_simplified_target(image, "blur", 51, 16).detach()
    task = feature_consistency(teacher, image, decoded, context)
    # Native non-CompressAI families use TV for the rate term. The common
    # entropy estimate remains diagnostic for SCOPE and is Zhao's rate loss.
    rate = helper.tv_loss(decoded, background)
    return (0.25 * helper.weighted_l1(edited, target, background)
            + 0.35 * helper.tv_loss(edited, background) + 0.50 * rate
            + helper.attention_loss(attention, tight) + 0.15 * attention.mean()
            + 0.50 * F.relu(attention.mean() - tight.mean().detach())
            + 0.02 * delta.abs().mean() + 0.05 * helper.tv_loss(delta) + 4.0 * task)
