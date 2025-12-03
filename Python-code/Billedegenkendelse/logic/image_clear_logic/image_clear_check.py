from utils.types import CheckResult, Requirement, Severity
from utils.picture_modefication import picture_modefication

from .laplacian_detector import LaplacienDetector
from .tenengrad_detector import TenengradDetector
from .background_detector import BackgroundDetector


class ImageClearCheck(picture_modefication):
    """
    High-level checker der bruger:
      - LaplacienDetector (ansigtsskarphed)
      - TenengradDetector (ansigtsskarphed)
      - BackgroundDetector (baggrundsimplicitet) (commentede ud)
    og returnerer ét samlet CheckResult for IMAGE_CLEAR.
    """

    def __init__(
        self,
        laplacian_detector: LaplacienDetector | None = None,
        tenengrad_detector: TenengradDetector | None = None,
        #background_detector: BackgroundDetector | None = None,
    ):
        # Hvis der ikke gives instanser udefra, laver vi bare nye
        self.laplacian_detector = laplacian_detector or LaplacienDetector()
        self.tenengrad_detector = tenengrad_detector or TenengradDetector()
        #self.background_detector = background_detector or BackgroundDetector()

    def analyze_bytes(self, image_bytes: bytes, landmarker_result=None) -> CheckResult:
        lap_res = self.laplacian_detector.analyze_bytes(image_bytes, landmarker_result)
        ten_res = self.tenengrad_detector.analyze_bytes(image_bytes, landmarker_result)
        #bg_res = self.background_detector.analyze_bytes(image_bytes, landmarker_result)

        passed = bool(lap_res.passed and ten_res.passed)

        if passed:
            message = (
                "Image is clear: face is sharp (Laplacian + Tenengrad) "
                #"and background is simple/uniform."
            )
        else:
            failed_parts = []
            if not lap_res.passed:
                failed_parts.append("Laplacian sharpness check failed")
            if not ten_res.passed:
                failed_parts.append("Tenengrad sharpness check failed")
            #if not bg_res.passed:
                #failed_parts.append("Background check failed")

            message = (
                "Image did not meet all requirements: "
                + ", ".join(failed_parts)
                + "."
            )

        details = {
            "laplacian": lap_res.details,
            "tenengrad": ten_res.details,
            #"background": bg_res.details,
        }

        return CheckResult(
            requirement=Requirement.IMAGE_CLEAR,
            passed=passed,
            severity=Severity.ERROR if not passed else Severity.INFO,
            message=message,
            details=details,
        )

    def analyze_image(self, image_path: str) -> CheckResult:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.analyze_bytes(image_bytes, landmarker_result=None)
