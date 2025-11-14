import cv2
import numpy as np
from pathlib import Path

from logic.types import CheckResult, Requirement, Severity
from .help_code import picture_modefication


def bytes_to_rgb_np(image_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PixelationDetector(picture_modefication):
    """
    Checks if an image (and especially the face) is clear or pixelated/blurry.

    Uses variance of Laplacian — low variance = blurry / low detail.
    We can:
      - require a MINIMUM variance for the FACE (sharp enough),
      - require a MAXIMUM variance for the BACKGROUND (nice smooth/simple background).
    """

    def __init__(
        self,
        face_threshold: float = 80.0,
        background_threshold: float = 40.0,
        min_face_pixels: int = 800,
    ):
        # legacy "threshold" for backwards compatibility (whole image)
        self.threshold = float(face_threshold)

        # separate thresholds
        self.face_threshold = float(face_threshold)
        self.background_threshold = float(background_threshold)
        self.min_face_pixels = int(min_face_pixels)

    def _variance_of_laplacian(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None
    ) -> float:
        """
        Compute variance of Laplacian, optionally only inside a binary mask (1 = keep).
        image is expected to be RGB.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        if mask is not None:
            # ensure mask is boolean and same HxW as image
            m = mask.astype(bool)
            values = laplacian[m]
        else:
            values = laplacian

        if values.size == 0:
            return 0.0

        return float(values.var())

    def analyze_bytes(
        self,
        image_bytes: bytes,
        landmarker_result=None
    ) -> CheckResult:
        """
        Analyze an image given as bytes.

        If landmarker_result is provided and usable, we:
          - build a face mask via picture_modefication,
          - compute Laplacian variance for face and background separately,
          - apply separate thresholds.
        If no usable face, we fall back to whole-image variance.
        """
        # Decode image to BGR (for masks) and RGB (for Laplacian)
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image from bytes.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --- Try to build a face mask ---
        face_mask = None
        face_pixels = 0
        used_face_mask = False

        if landmarker_result is not None and getattr(landmarker_result, "face_landmarks", None):
            raw_mask = self._landmarks_to_mask(bgr.shape, landmarker_result)
            refined_mask = self._refine_face_mask(raw_mask)

            face_pixels = int(refined_mask.sum())
            if face_pixels >= self.min_face_pixels:
                face_mask = refined_mask
                used_face_mask = True
            else:
                # mask too small / unreliable → ignore
                face_mask = None

        # --- Compute variances ---
        variance_full = self._variance_of_laplacian(rgb)

        variance_face = None
        variance_background = None

        if face_mask is not None:
            # face variance
            variance_face = self._variance_of_laplacian(rgb, mask=face_mask)

            # background = everything that is NOT face
            background_mask = (face_mask == 0).astype(np.uint8)
            variance_background = self._variance_of_laplacian(rgb, mask=background_mask)

        # --- Decide pass/fail ---

        if face_mask is None:
            # No usable face mask → fall back to full image logic (old behavior)
            clear = bool(variance_full >= self.threshold)
            message = (
                "Image is clear (no reliable face mask available)."
                if clear
                else "Image appears pixelated or blurry (no reliable face mask available)."
            )
        else:
            # Use separate thresholds
            face_ok = bool(variance_face >= self.face_threshold)
            # For background we want it "simple" → low variance is good,
            # so we treat background_threshold as a MAXIMUM.
            background_ok = bool(variance_background <= self.background_threshold)

            clear = face_ok and background_ok

            if clear:
                message = "Image is clear: face is sharp and background is simple."
            elif not face_ok and background_ok:
                message = "Face appears blurry or pixelated."
            elif face_ok and not background_ok:
                message = "Background has too much texture/variation (not clean enough)."
            else:
                message = "Face appears blurry and background has too much texture/variation."

        # --- Build details dict ---
        details = {
            "variance_full": variance_full,
            "variance_face": variance_face,
            "variance_background": variance_background,
            "face_threshold": self.face_threshold,
            "background_threshold": self.background_threshold,
            "used_face_mask": bool(used_face_mask),
            "face_pixels": face_pixels,
        }

        return CheckResult(
            requirement=Requirement.IMAGE_CLEAR,
            passed=clear,
            severity=Severity.ERROR if not clear else Severity.INFO,
            message=message,
            details=details,
        )

    def analyze_image(self, image_path: str) -> CheckResult:
        """
        File-based version without landmarks.
        For now this just uses whole-image variance (legacy behavior).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        variance_full = float(self._variance_of_laplacian(rgb))

        details = {
            "variance_full": variance_full,
            "variance_face": None,
            "variance_background": None,
            "face_threshold": self.face_threshold,
            "background_threshold": self.background_threshold,
            "used_face_mask": False,
            "face_pixels": 0,
        }

        clear = bool(variance_full >= self.threshold)

        return CheckResult(
            requirement=Requirement.IMAGE_CLEAR,
            passed=clear,
            severity=Severity.ERROR if not clear else Severity.INFO,
            message=("Image is clear." if clear else "Image appears pixelated or blurry."),
            details=details,
        )
