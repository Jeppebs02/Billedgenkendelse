import cv2
import numpy as np
from ..types import CheckResult, Requirement, Severity


class exposure_check:
    """
    Vurderer eksponering og belysning i et ansigt baseret på MediaPipe-landmarks.
    Giver et CheckResult (Requirement.LIGHTING_OK) med detaljerede lysmetrikker.
    """

    def __init__(self):
        # Starttærskler (kan kalibreres med egne data)
        self.thr_clip_hi = 0.06
        self.thr_clip_lo = 0.12
        self.thr_dark_p05 = 8
        self.thr_bright_p95 = 252
        self.thr_dynamic_range = 28
        self.thr_std = 16
        self.thr_side_diff_pct = 0.22
        self.thr_side_ratio = 1.45
        self.thr_p50_dark = 60
        self.thr_p50_bright = 190
        self.min_mask_pixels = 800

    # =====================================================
    # Public API
    # =====================================================
    def analyze(self, bgr: np.ndarray, landmarks) -> CheckResult:
        """
        Input:
            bgr: np.ndarray (OpenCV BGR billede)
            landmarks: MediaPipe landmark-liste for ét ansigt
        Output:
            CheckResult for Requirement.LIGHTING_OK
        """
        L = self._luminance_lab(bgr)
        face_mask = self._landmarks_to_mask(bgr.shape, landmarks)
        face_mask = self._refine_face_mask(face_mask)

        if int(face_mask.sum()) < self.min_mask_pixels:
            return CheckResult(
                requirement=Requirement.LIGHTING_OK,
                passed=False,
                severity=Severity.ERROR,
                message="Ingen eller usikker ansigtsmaske til lysvurdering.",
                details={"mask_pixels": int(face_mask.sum())}
            )

        # Mild blur for at fjerne små spejlinger
        L_smooth = cv2.GaussianBlur(L, (5, 5), 0)
        L_face = L_smooth[face_mask == 1]

        # Histogram-metrikker
        p05 = float(np.percentile(L_face, 5))
        p50 = float(np.percentile(L_face, 50))
        p95 = float(np.percentile(L_face, 95))
        stdL = float(np.std(L_face))
        clip_hi_ratio = float(np.count_nonzero(L_face >= 252)) / L_face.size
        clip_lo_ratio = float(np.count_nonzero(L_face <= 3)) / L_face.size
        dynamic_range = float(p95 - p05)

        # Venstre/højre side
        ys, xs = np.where(face_mask == 1)
        x_mid = (int(xs.min()) + int(xs.max())) // 2
        left_mask = np.zeros_like(face_mask, np.uint8); left_mask[:, :x_mid] = 1
        right_mask = np.zeros_like(face_mask, np.uint8); right_mask[:, x_mid:] = 1
        left_mask &= face_mask; right_mask &= face_mask
        L_left = L_smooth[left_mask == 1]
        L_right = L_smooth[right_mask == 1]
        med_left = float(np.median(L_left)) if L_left.size > 0 else p50
        med_right = float(np.median(L_right)) if L_right.size > 0 else p50
        side_diff_abs = abs(med_left - med_right)
        side_diff_pct = side_diff_abs / max(1.0, p50)
        side_ratio = max(med_left, med_right) / max(1.0, min(med_left, med_right))

        details = {
            "p05": p05, "p50": p50, "p95": p95, "stdL": stdL,
            "clip_hi_ratio": clip_hi_ratio, "clip_lo_ratio": clip_lo_ratio,
            "dynamic_range": dynamic_range,
            "side_diff_abs": side_diff_abs, "side_diff_pct": side_diff_pct,
            "side_ratio": side_ratio,
            "mask_pixels": int(face_mask.sum())
        }

        # =====================================================
        # Decision logic (robust v2)
        # =====================================================

        # Overeksponeret kræver både høj p95 og mange klippede pixels
        if (p95 >= self.thr_bright_p95 and clip_hi_ratio >= self.thr_clip_hi):
            return self._fail("Overeksponeret/udbrændt i ansigtet.", details)

        # Under-eksponering lidt mere konservativ
        if (p05 <= self.thr_dark_p05 and clip_lo_ratio >= self.thr_clip_lo):
            return self._fail("For mørkt i ansigtet.", details)

        # Lav kontrast kun hvis begge parametre er lave
        if dynamic_range < self.thr_dynamic_range and stdL < self.thr_std:
            return self._fail("For lav kontrast/dynamik i ansigtet.", details)

        # Ujævn belysning relativt (20-25 % forskel)
        if side_diff_pct >= self.thr_side_diff_pct or side_ratio >= self.thr_side_ratio:
            return self._fail("Ujævn belysning mellem ansigtshalvdele.", details)

        # p50 failsafes kun hvis dynamikken også er lav
        if p50 <= self.thr_p50_dark and dynamic_range < 35:
            return self._fail("Generelt for lav ansigtsluminans.", details)
        if p50 >= self.thr_p50_bright and dynamic_range < 35:
            return self._fail("Generelt for høj ansigtsluminans.", details)

        return CheckResult(
            requirement=Requirement.LIGHTING_OK,
            passed=True,
            severity=Severity.INFO,
            message="Eksponering OK.",
            details=details
        )

    # =====================================================
    # Internal helpers
    # =====================================================

    def _fail(self, msg: str, details: dict) -> CheckResult:
        return CheckResult(
            requirement=Requirement.LIGHTING_OK,
            passed=False,
            severity=Severity.ERROR,
            message=msg,
            details=details
        )

    def _luminance_lab(self, bgr: np.ndarray) -> np.ndarray:
        """Brug LAB L-kanal (0..255), ofte mere stabil end YCrCb."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    def _landmarks_to_mask(self, img_shape, landmarks) -> np.ndarray:
        """Byg binær maske (0/1) ud fra ansigtslandmarks."""
        h, w = img_shape[:2]
        if not landmarks or len(landmarks) < 3:
            return np.zeros((h, w), np.uint8)
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in landmarks], np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask

    def _refine_face_mask(self, mask: np.ndarray) -> np.ndarray:
        """Fjern hår/baggrund med erosion og morfologisk åbning."""
        k = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask
