# logic/pixelation_logic/Pixelation_detection.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class PixelationAnalysis:
    is_pixelated: bool
    mean_block_variance: float
    gd_ratio: float
    details: Dict[str, Any]


class PixelationDetectorV2:
    def __init__(
        self,
        block_size: int = 8,
        var_thr: float = 20.0,
        gd_low_thr: float = 8.0,
        gd_high_thr: float = 25.0,
        gd_ratio_thr: float = 0.6,
    ):
        self.block_size = block_size
        self.var_thr = var_thr
        self.gd_low_thr = gd_low_thr
        self.gd_high_thr = gd_high_thr
        self.gd_ratio_thr = gd_ratio_thr

    def analyze(self, bgr_img: np.ndarray) -> PixelationAnalysis:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        mean_var, _ = self._block_variance(gray)
        gd_ratio, gd_counts = self._gradient_discontinuity_ratio(gray)

        is_pixelated = (mean_var < self.var_thr) or (gd_ratio < self.gd_ratio_thr)

        details = {
            "mean_block_variance": float(mean_var),
            "variance_threshold": float(self.var_thr),
            "gd_ratio": float(gd_ratio),
            "gd_ratio_threshold": float(self.gd_ratio_thr),
            "gd_counts": gd_counts,
            "block_size": self.block_size,
        }

        return PixelationAnalysis(
            is_pixelated=is_pixelated,
            mean_block_variance=mean_var,
            gd_ratio=gd_ratio,
            details=details,
        )

    def _block_variance(self, gray: np.ndarray) -> Tuple[float, np.ndarray]:
        h, w = gray.shape
        bs = self.block_size
        h_trim = (h // bs) * bs
        w_trim = (w // bs) * bs
        gray_trim = gray[:h_trim, :w_trim]

        if gray_trim.size == 0:
            return 0.0, np.zeros((0, 0), dtype=np.float32)

        blocks = gray_trim.reshape(h_trim // bs, bs, w_trim // bs, bs)
        blocks = blocks.transpose(0, 2, 1, 3)  # (by, bx, bs, bs)
        var_map = blocks.reshape(blocks.shape[0], blocks.shape[1], -1).var(axis=2)
        mean_var = float(var_map.mean())
        return mean_var, var_map

    def _gradient_discontinuity_ratio(self, gray: np.ndarray):
        gray = gray.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag_flat = mag.reshape(-1)

        nonzero = mag_flat > 0.5
        mag_nz = mag_flat[nonzero]
        if mag_nz.size == 0:
            return 0.0, {"num_weak": 0, "num_mid": 0, "num_strong": 0}

        num_weak = int(((mag_nz > 0) & (mag_nz < self.gd_low_thr)).sum())
        num_mid = int(((mag_nz >= self.gd_low_thr) & (mag_nz < self.gd_high_thr)).sum())
        num_strong = int((mag_nz >= self.gd_high_thr).sum())

        gd_ratio = (num_weak + 1) / (num_strong + 1)

        counts = {
            "num_weak": num_weak,
            "num_mid": num_mid,
            "num_strong": num_strong,
            "total_nonzero": int(mag_nz.size),
        }
        return float(gd_ratio), counts
