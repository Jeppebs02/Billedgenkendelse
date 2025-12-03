import cv2
import numpy as np

from utils.types import CheckResult, Requirement, Severity
from utils.picture_modefication import picture_modefication


class BackgroundDetector(picture_modefication):
    """
    Checks whether background color is uniform using Lab RMS distance.
    NO Laplacian. NO Tenengrad.
    """

    def __init__(
        self,
        max_background_color_rms: float = 50.0,
        min_background_pixels: int = 200,
    ):
        self.max_background_color_rms = float(max_background_color_rms)
        self.min_background_pixels = int(min_background_pixels)

    def _color_uniformity_lab(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """
        Compute RMS color difference from mean Lab color.
        Lower value means more uniform background.
        """
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        if mask is not None:
            pixels = lab[mask.astype(bool)]
        else:
            pixels = lab.reshape(-1, 3)

        if pixels.size == 0:
            return float("inf")

        mean_color = pixels.mean(axis=0)
        diffs = pixels - mean_color
        dist2 = np.sum(diffs ** 2, axis=1)
        rms = float(np.sqrt(dist2.mean()))
        return rms

    def analyze_bytes(
            self,
            image_bytes: bytes,
            landmarker_result=None,
    ) -> CheckResult:
        # decode
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image from bytes.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]

        # --- Build background mask (area ABOVE bottom of face, excluding face) ---
        background_mask = None
        used_face_mask = False
        background_pixels = 0

        if landmarker_result is not None and getattr(landmarker_result, "face_landmarks", None):
            raw_face_mask = self._landmarks_to_mask(bgr.shape, landmarker_result)
            face_mask = self._refine_face_mask(raw_face_mask)

            # find bottom of face in image coords
            ys, xs = np.where(face_mask > 0)
            if ys.size > 0:
                bottom_y = ys.max()

                # start with everything above bottom_y
                candidate_bg = np.zeros_like(face_mask, dtype=np.uint8)
                candidate_bg[:bottom_y, :] = 1  # alt OVER bunden af ansigtet

                # fjern selve ansigtet
                candidate_bg[face_mask > 0] = 0

                background_pixels = int(candidate_bg.sum())

                if background_pixels >= self.min_background_pixels:
                    background_mask = candidate_bg
                    used_face_mask = True
                else:
                    # for lidt baggrund → faldbak til hele billedet
                    background_mask = None
                    background_pixels = h * w
            else:
                # ingen pixels i facemasken → som ingen landmarks
                background_pixels = h * w
        else:
            # no landmarks → entire image is background
            background_pixels = h * w

        # --- compute uniformity ---
        color_rms = self._color_uniformity_lab(rgb, mask=background_mask)

        passed = bool(color_rms <= self.max_background_color_rms)

        if passed:
            if used_face_mask:
                message = "Background above the face is uniform."
            else:
                message = "Background is uniform."
        else:
            if used_face_mask:
                message = (
                    "Background above the face shows too much color variation."
                )
            else:
                message = "Background shows too much color variation."

        details = {
            "background_color_rms_lab": color_rms,
            "max_background_color_rms": self.max_background_color_rms,
            "background_pixels": background_pixels,
            "used_face_mask": used_face_mask,
        }

        return CheckResult(
            requirement=Requirement.IMAGE_BACKGROUND_UNIFORM,
            passed=passed,
            severity=Severity.ERROR if not passed else Severity.INFO,
            message=message,
            details=details,
        )
