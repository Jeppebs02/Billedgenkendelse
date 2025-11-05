from .glasses_logic.glasses_logic import GlassesLogic
from .is_face_present.face_detector import DetectionVisualizer
from .is_hat_glasses.hat_glasses_detector import HatGlassesDetector
from .exposure.exposure_check import exposure_check
from .is_image_clear.pixelation_detector import PixelationDetector
from .types import AnalysisReport, CheckResult, Requirement, Severity
from mediapipe.tasks.python import vision
from typing import Tuple, Union, Optional, List
import math



class LogicController:
    def __init__(self):
        self.face_detector = DetectionVisualizer()
        self.hat_glasses_detector = HatGlassesDetector()
        self.glasses_logic = GlassesLogic()
        self.exposure_check = exposure_check()
        self.pixelation_detector = PixelationDetector()



    # Utility functions

    def _dist(self, p, q) -> float:
        '''Finds the distance between two points.'''
        dx, dy = (p.x - q.x), (p.y - q.y)
        return math.hypot(dx, dy)

    def _mean(self, xs):
        '''Returns the average of numbers in an interable'''
        return sum(xs) / max(1, len(xs))

    def _calculate_ear(self, eye_landmarks: List) -> float:
        """
        Calculates EAR for a single eye.
        """
        # Vertical distances
        p2_p6 = self._dist(eye_landmarks[1], eye_landmarks[5])
        p3_p5 = self._dist(eye_landmarks[2], eye_landmarks[4])

        # Horizontal distance
        p1_p4 = self._dist(eye_landmarks[0], eye_landmarks[3])

        # EAR calculation
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear


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
        checks.append(self._is_face_in_image(face_detector_result))

        # 2) Single face
        checks.append(self._is_single_face(face_detector_result))

        # 3) Landmarks present
        checks.append(self._are_landmarks_present(face_landmarker_result))

        # 4) Eyes visible
        checks.append(self._eyes_visible_check(face_landmarker_result))

        # 5) Mouth closed
        checks.append(self._mouth_closed_check(face_landmarker_result))

        # 6) No hat and no glasses
        checks.extend(self._check_hats_and_glasses_bytes(image_bytes, threshold=threshold))

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


        overall_pass = all(c.passed for c in checks)
        return AnalysisReport(
            image="<bytes>",
            passed=overall_pass,
            checks=checks
        )




    def _is_face_in_image(self, result: vision.FaceDetectorResult) -> CheckResult:
        count_faces = len(result.detections)

        if count_faces > 0:
            return CheckResult(
                requirement=Requirement.FACE_PRESENT,
                passed=True,
                severity=Severity.INFO,
                message=f"Detected {count_faces} face(s).",
                details={"count": count_faces}
            )
        else:
            return CheckResult(
                requirement=Requirement.FACE_PRESENT,
                passed=False,
                severity=Severity.ERROR,
                message="No face detected.",
                details={"count": 0}
            )

    def _is_single_face(self, result: vision.FaceDetectorResult) -> CheckResult:
        count = len(result.detections)
        if count == 1:
            return CheckResult(
                requirement=Requirement.SINGLE_FACE,
                passed=True,
                severity=Severity.INFO,
                message="One face detected.",
                details={"count": 1}
            )
        return CheckResult(
            requirement=Requirement.SINGLE_FACE,
            passed=False,
            severity=Severity.ERROR,
            message=f"Expected exactly 1 face, found {count}.",
            details={"count": count}
        )

    def _are_landmarks_present(self, result: vision.FaceLandmarkerResult) -> CheckResult:

        has_any_landmarks = bool(
            result is not None and
            getattr(result, "face_landmarks", None) and
            len(result.face_landmarks) > 0 and
            len(result.face_landmarks[0]) > 0
        )

        if has_any_landmarks:
            count_of_faces = [len(face) for face in result.face_landmarks]
            return CheckResult(
                requirement=Requirement.LANDMARKS_PRESENT,
                passed=True,
                severity=Severity.INFO,
                message=f"Detected landmarks for {len(result.face_landmarks)} face(s).",
                details={"faces_with_landmarks": len(result.face_landmarks),
                         "landmark_counts": count_of_faces}
            )
        else:
            return CheckResult(
                requirement=Requirement.LANDMARKS_PRESENT,
                passed=False,
                severity=Severity.ERROR,
                message="Face not fully visible in image",
                details={"faces_with_landmarks": 0}
            )

    def _eyes_visible_check(self, result: vision.FaceLandmarkerResult, ear_threshold: float = 0.2) -> CheckResult:

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=False,
                severity=Severity.ERROR,
                message="No landmarks detected – cannot check if eyes are visible.",
                details={"landmarks_detected": False}
            )

        # Use first detected face
        landmarks = result.face_landmarks[0]

        # Indices for eye corners
        # Check this link for details
        # https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/#Landmark-Detection-Using-Mediapipe-Face-Mesh-In-Python
        LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # left eye outer & inner
        RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # right eye outer & inner


        try:
            left_eye = [landmarks[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [landmarks[i] for i in RIGHT_EYE_LANDMARKS]
        except IndexError:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=False,
                severity=Severity.ERROR,
                message="Could not find expected eye landmarks.",
                details={"landmarks_count": len(landmarks)}
            )


        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)

        # Average EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # The EAR is typically around 0.25-0.3 for open eyes and drops towards 0 for closed eyes.
        eyes_visible = avg_ear > ear_threshold

        if eyes_visible:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=True,
                severity=Severity.INFO,
                message="Both eyes are visible.",
                details={
                    "left_eye_width": left_ear,
                    "right_eye_width": right_ear
                }
            )
        else:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=False,
                severity=Severity.ERROR,
                message=f"Eyes closed or not visible.",
                details={
                    "left_eye_width": left_ear,
                    "right_eye_width": right_ear
                }
            )

    def _mouth_closed_check(self, result: vision.FaceLandmarkerResult, max_gap_ratio: float = 0.03) -> CheckResult:
        # Check
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return CheckResult(
                requirement=Requirement.MOUTH_CLOSED,
                passed=False,
                severity=Severity.ERROR,
                message="No landmarks detected, cannot check if mouth is closed.",
                details={"landmarks_detected": False}
            )

        lmk = result.face_landmarks[0]

        try:
            # Inner-lip vertical gap
            # see https://github.com/google-ai-edge/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
            # for indices
            upper_inner = lmk[13]
            lower_inner = lmk[14]
            gap = self._dist(upper_inner, lower_inner)

            # Corner widths (outer + inner)
            left_outer = lmk[61]
            right_outer = lmk[291]

            left_inner = lmk[78]
            right_inner = lmk[308]
            width_outer = self._dist(left_outer, right_outer)
            width_inner = self._dist(left_inner, right_inner)
            # return whichever is larger
            width = max(width_outer, width_inner)

        except IndexError:
            return CheckResult(
                requirement=Requirement.MOUTH_CLOSED,
                passed=False,
                severity=Severity.ERROR,
                message="Could not find mouth in image, make sure your mouth is fully visible.",
                details={"landmarks_count": len(lmk)}
            )

        eps = 1e-6
        # eps is to avoid division by zero, it is a tiny number, so even if width is 0, ratio will be very large.
        # 1e-6 = 0.000001
        ratio = gap / (width + eps)
        # Why do we divide by width? To normalize the gap size relative to face size.
        final_result = ratio <= max_gap_ratio

        if final_result:
            return CheckResult(
                requirement=Requirement.MOUTH_CLOSED,
                passed=True,
                severity=Severity.INFO,
                message=f"Mouth closed (gap ratio {ratio:.3f} ≤ {max_gap_ratio}).",
                details={
                    "gap": gap,
                    "width_outer": width_outer,
                    "width_inner": width_inner,
                    "norm_width": width,
                    "gap_ratio": ratio
                }
            )
        else:
            return CheckResult(
                requirement=Requirement.MOUTH_CLOSED,
                passed=False,
                severity=Severity.ERROR,
                message=f"Open mouth detected.",
                details={
                    "gap": gap,
                    "width_outer": width_outer,
                    "width_inner": width_inner,
                    "norm_width": width,
                    "gap_ratio": ratio
                }
            )


    def _check_hats_and_glasses(self, image_path: str, threshold: float = 0.5) -> list[CheckResult]:
        """
        Run the YOLO hat and glasses detection.
        Returns two CheckResult objects: NO_HAT and NO_GLASSES. So to add it to the report, we need to use .extend()
        """
        yolo_confs = self.hat_glasses_detector.analyze_image(image_path)
        hat_conf = yolo_confs.get("Hat", 0.0)
        glasses_conf = yolo_confs.get("Glasses", 0.0)

        no_hat_pass = hat_conf < threshold
        no_glasses_pass = glasses_conf < threshold

        results = [
            CheckResult(
                requirement=Requirement.NO_HAT,
                passed=no_hat_pass,
                severity=Severity.ERROR if not no_hat_pass else Severity.INFO,
                message=("No hat detected." if no_hat_pass
                         else f"Hat detected."),
                details={"hat_confidence": hat_conf, "threshold": threshold}
            ),
            CheckResult(
                requirement=Requirement.NO_GLASSES,
                passed=no_glasses_pass,
                severity=Severity.ERROR if not no_glasses_pass else Severity.INFO,
                message=("No glasses detected." if no_glasses_pass
                         else f"Glasses detected."),
                details={"glasses_confidence": glasses_conf, "threshold": threshold}
            )
        ]

        return results

    def _check_hats_and_glasses_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> list[CheckResult]:
        """
        YOLO hat/glasses on in-memory bytes.
        Returns two CheckResult objects: NO_HAT and NO_GLASSES. So to add it to the report, we need to use .extend()
        """
        yolo_confs = self.hat_glasses_detector.analyze_bytes(image_bytes)
        hat_conf = yolo_confs.get("Hat", 0.0)
        glasses_conf = yolo_confs.get("Glasses", 0.0)

        no_hat_pass = hat_conf < threshold
        no_glasses_pass = glasses_conf < threshold

        return [
            CheckResult(
                requirement=Requirement.NO_HAT,
                passed=no_hat_pass,
                severity=Severity.ERROR if not no_hat_pass else Severity.INFO,
                message=("No hat detected." if no_hat_pass
                         else f"Hat detected."),
                details={"hat_confidence": hat_conf, "threshold": threshold}
            ),
            CheckResult(
                requirement=Requirement.NO_GLASSES,
                passed=no_glasses_pass,
                severity=Severity.ERROR if not no_glasses_pass else Severity.INFO,
                message=("No glasses detected." if no_glasses_pass
                         else f"Glasses detected."),
                details={"glasses_confidence": glasses_conf, "threshold": threshold}
            ),
        ]