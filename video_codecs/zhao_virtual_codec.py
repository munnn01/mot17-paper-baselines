"""Assumed implementation of Zhao Figure 3; shared by both adapted methods.

8x8 RGB DCT; intra candidates are fixed DC levels 0, 0.5, 1; P candidates use
the previous reconstructed frame and 16x16 motion search from the project.
Mode selection is detached block SAD; reconstruction retains gradients.
Separate I/P monotonic factorized CDFs model each RGB/frequency channel.
No claim is made that estimated bpp equals actual x265 bitstream size.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from video_codecs.hybrid_x265_ste import create_dct_matrix, estimate_motion_compensated_pred


class FactorizedCDF(nn.Module):
    """Ballé-style monotonic neural CDF, independently parameterized per channel."""
    def __init__(self, channels=192):
        super().__init__()
        widths = (1, 3, 3, 3, 1)
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            initial = math.log(math.expm1(1.0 / (2.0 * b)))
            self.matrices.append(nn.Parameter(torch.full((channels, b, a), initial)))
            self.biases.append(nn.Parameter(torch.zeros(channels, b, 1)))
            if i < 3:
                self.factors.append(nn.Parameter(torch.zeros(channels, b, 1)))

    def logits(self, values):
        for i, (matrix, bias) in enumerate(zip(self.matrices, self.biases)):
            values = torch.bmm(F.softplus(matrix), values) + bias
            if i < len(self.factors):
                values = values + torch.tanh(self.factors[i]) * torch.tanh(values)
        return values

    def cdf(self, values):
        return torch.sigmoid(self.logits(values))

    def forward(self, symbols):
        low, high = self.logits(symbols - 0.5), self.logits(symbols + 0.5)
        sign = -torch.sign(low + high).detach()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return (torch.sigmoid(sign * high) - torch.sigmoid(sign * low)).abs().clamp_min(1e-9)


class ZhaoVirtualCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dct_matrix", create_dct_matrix(8))
        self.entropy = nn.ModuleList([FactorizedCDF(), FactorizedCDF()])

    def freeze_entropy(self):
        self.entropy.requires_grad_(False)
        return self

    def dct(self, image):
        b, c, h, w = image.shape
        blocks = image.reshape(b, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)
        matrix = self.dct_matrix.to(image)
        return matrix @ blocks @ matrix.t()

    def idct(self, coefficients, h, w):
        matrix = self.dct_matrix.to(coefficients)
        blocks = matrix.t() @ coefficients @ matrix
        return blocks.permute(0, 1, 2, 4, 3, 5).reshape(coefficients.shape[0], 3, h, w)

    @staticmethod
    def symbols(coefficients):
        return coefficients.permute(1, 4, 5, 0, 2, 3).reshape(192, 1, -1)

    @staticmethod
    def intra_candidates(frame):
        # Decoder-known DC palette: no current source pixels enter prediction.
        # Two mode bits per 8x8 block are charged below. This inexpensive
        # assumed intra search is explicitly narrower than standard HEVC.
        return [torch.full_like(frame, level) for level in (0.0, 0.5, 1.0)]

    def forward(self, video, severity, return_symbols=False):
        b, t, c, h, w = video.shape
        if c != 3 or h % 16 or w % 16 or min(h, w) < 16:
            raise ValueError("Virtual codec requires RGB and dimensions divisible by16")
        with torch.amp.autocast(device_type=video.device.type, enabled=False):
            video = video.float()
            qp = 22.0 + 25.0 * severity.float()
            q = (2.0 ** ((qp - 4.0) / 6.0) / 255.0).reshape(b, 1, 1, 1, 1, 1)
            reconstructions, rates, calibration = [], [], []
            previous = None
            for i in range(t):
                frame = video[:, i]
                candidates = self.intra_candidates(frame)
                if previous is not None:
                    candidates.append(estimate_motion_compensated_pred(frame, previous))
                stacked = torch.stack(candidates, dim=1)
                with torch.no_grad():
                    error = (stacked - frame[:, None]).abs().mean(2)
                    error = F.avg_pool2d(error, 8, 8)
                    index = error.argmin(1, keepdim=True).repeat_interleave(8, -2).repeat_interleave(8, -1)
                prediction = stacked.gather(1, index[:, :, None].expand(-1, 1, 3, -1, -1)).squeeze(1)
                normalized = self.dct(frame - prediction) / q
                quantized = normalized + (normalized.round() - normalized).detach()
                reconstructed = (prediction + self.idct(quantized * q, h, w)).clamp(0, 1)
                symbols = self.symbols(quantized)
                prior_id = 0 if i == 0 else 1
                likelihood = self.entropy[prior_id](symbols)
                # Fixed-length mode IDs and conservative 2x7-bit motion
                # vectors per16x16 block, including blocks choosing intra.
                side_bpp = 2.0 / 64.0 + (14.0 / 256.0 if i else 0.0)
                rates.append(-likelihood.log2().sum() / (b * h * w) + side_bpp)
                if return_symbols:
                    calibration.append((prior_id, symbols.detach()))
                reconstructions.append(reconstructed)
                previous = reconstructed
            return {"video_recon": torch.stack(reconstructions, 1),
                    "rate_bpp": torch.stack(rates).mean(), "q_step": q.flatten(),
                    "calibration_symbols": calibration}
