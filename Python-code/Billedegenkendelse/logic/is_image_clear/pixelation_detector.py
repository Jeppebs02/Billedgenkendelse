import cv2
import numpy as np
from logic.types import CheckResult, Requirement, Severity  # <-- ADDED IMPORTS
from pathlib import Path


def bytes_to_rgb_np(image_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PixelationDetector:
    """
    Checks if an image is clear or pixelated.
    Uses variance of Laplacian — low variance = blurry/pixelated.
    """

    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def _variance_of_laplacian(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def analyze_bytes(self, image_bytes: bytes) -> CheckResult:  # <-- CHANGED return type
        image = bytes_to_rgb_np(image_bytes)
        variance = float(self._variance_of_laplacian(image))

        # Create the details dictionary first
        details = {
            "clear": bool(variance >= float(self.threshold)),
            "variance": variance,
            "threshold": float(self.threshold),
        }

        # Use the details to create and return the CheckResult
        clear = details["clear"]
        return CheckResult(
            requirement=Requirement.IMAGE_CLEAR,
            passed=clear,
            severity=Severity.ERROR if not clear else Severity.INFO,
            message=("Image is clear." if clear else "Image appears pixelated or blurry."),
            details=details
        )

    def analyze_image(self, image_path: str) -> CheckResult:  # <-- CHANGED return type
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {path}")

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        variance = float(self._variance_of_laplacian(image))

        details = {
            "clear": bool(variance >= float(self.threshold)),
            "variance": variance,
            "threshold": float(self.threshold),
        }

        clear = details["clear"]
        return CheckResult(
            requirement=Requirement.IMAGE_CLEAR,
            passed=clear,
            severity=Severity.ERROR if not clear else Severity.INFO,
            message=("Image is clear." if clear else "Image appears pixelated or blurry."),
            details=details
        )

    # THIS METHOD IS NO LONGER NEEDED AND CAN BE REMOVED
    # def _is_picture_clear(self, result: dict) -> CheckResult:
    #     ...