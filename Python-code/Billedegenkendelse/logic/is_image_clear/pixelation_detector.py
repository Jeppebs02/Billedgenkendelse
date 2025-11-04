import cv2
import numpy as np
from logic.utils.image_io import bytes_to_rgb_np

class PixelationDetector:
    """
    Checks if an image is clear or pixelated.
    Uses variance of Laplacian — low variance = blurry/pixelated.
    """

    def __init__(self, threshold: float = 100.0):
        """
        threshold: minimum variance of Laplacian to consider image as 'clear'
        (higher threshold = stricter clarity requirement)
        """
        self.threshold = threshold

    def _variance_of_laplacian(self, image: np.ndarray) -> float:
        """Calculate the Laplacian variance, which measures image sharpness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def analyze_bytes(self, image_bytes: bytes) -> dict:
        """Check clarity from bytes input."""
        image = bytes_to_rgb_np(image_bytes)
        variance = self._variance_of_laplacian(image)
        clear = variance >= self.threshold
        return {"clear": clear, "variance": variance, "threshold": self.threshold}

    def analyze_image(self, image_path: str) -> dict:
        """Check clarity from an image file."""
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        variance = self._variance_of_laplacian(image)
        clear = variance >= self.threshold
        return {"clear": clear, "variance": variance, "threshold": self.threshold}
