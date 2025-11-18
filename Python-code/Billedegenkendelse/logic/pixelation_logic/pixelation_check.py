# logic/pixelation_logic/pixelation_check.py

from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


from logic.pixelation_logic.pixelation_detection import PixelationDetectorV2
from utils.types import CheckResult, Requirement, Severity


class PixelationCheck:
    def __init__(self):
        self.detector = PixelationDetectorV2()

    def check_pixelation_bytes(
        self,
        image_bytes: bytes,
        requirement: Optional[Requirement] = None,
    ) -> CheckResult:
        # bytes -> RGB -> BGR
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        analysis = self.detector.analyze(bgr)

        if analysis.is_pixelated:
            status = False
            msg = "Billedet er for pixeleret"
            severity = Severity.ERROR
        else:
            status = True
            msg = "Ingen tydelig pixelering detekteret."
            severity = Severity.INFO

        return CheckResult(
            requirement=Requirement.PIXELATION,
            passed=status,
            severity=severity,
            message=msg,
            details=analysis.details,
        )
