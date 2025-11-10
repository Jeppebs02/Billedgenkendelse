from .glasses_logic.glasses_logic import GlassesLogic
from .head_placement.head_centering_validator import HeadCenteringValidator
from .is_face_present.face_detector import DetectionVisualizer
from .is_hat_glasses.hat_glasses_detector import HatGlassesDetector
from .exposure.exposure_check import exposure_check
from .is_image_clear.pixelation_detector import PixelationDetector
from .types import AnalysisReport, CheckResult, Requirement, Severity
from mediapipe.tasks.python import vision
from typing import Tuple, Union, Optional, List
from logic.face_direction.face_looking_at_camera import FaceLookingAtCamera
import math



class LogicController:
    def __init__(self):
        self.face_detector = DetectionVisualizer()
        self.hat_glasses_detector = HatGlassesDetector()
        self.glasses_logic = GlassesLogic()
        self.exposure_check = exposure_check()
        # Here we can set a float which is the threshold for pixelation detection
        self.pixelation_detector = PixelationDetector(60)
        self.face_looking_at_camera = FaceLookingAtCamera()
        self.head_centering_validator = HeadCenteringValidator()


    # Utility functions




    def run_analysis(self, image_path: str, threshold: float = 0.5) -> AnalysisReport:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.run_analysis_bytes(image_bytes, threshold)

    def run_analysis_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> AnalysisReport:
        """
        In-memory analysis (no filed needed ;) ). Mirrors run_analysis(), but calls *bytes* APIs from face detector and landmarker AND hat_glasses.
        """
        # Face + landmarks (bytes-based)
        face_detector_result = self.face_detector.analyze_bytes(image_bytes)
        face_landmarker_result = self.face_detector.analyze_landmarks_bytes(image_bytes)

        checks = []

        # 1) Face present
        checks.append(self.face_detector.is_face_in_image(face_detector_result))

        # 2) Single face
        checks.append(self.face_detector.is_single_face(face_detector_result))

        # 3) Landmarks present
        checks.append(self.face_detector.are_landmarks_present(face_landmarker_result))

        # 4) Eyes visible
        checks.append(self.face_detector.eyes_visible_check(face_landmarker_result))

        # 5) Mouth closed
        checks.append(self.face_detector.mouth_closed_check(face_landmarker_result))

        # 6) No hat and no glasses
        checks.extend(self.hat_glasses_detector.check_hats_and_glasses_bytes(image_bytes, threshold=threshold))

        # 7) sunglasses / glare check
        checks.extend(
            self.glasses_logic.run_all(
                image_bytes=image_bytes,
                face_detector_result=face_detector_result,
                face_landmarker_result=face_landmarker_result
            )
        )
        # 8) image clear check
        checks.append(self.pixelation_detector.analyze_bytes(image_bytes))

        # 9) exposure / lighting check
        checks.append(self.exposure_check.analyze(image_bytes, face_landmarker_result))

        # 10) face looking straight
        checks.append(self.face_looking_at_camera.face_detector(result=face_landmarker_result))

        # 11) face centered
        checks.append(self.head_centering_validator.check_from_detection_bytes(det_res=face_detector_result, image_bytes=image_bytes))

        overall_pass = all(c.passed for c in checks)
        return AnalysisReport(
            image="<bytes>",
            passed=overall_pass,
            checks=checks
        )


