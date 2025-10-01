from .is_face_present.face_detector import DetectionVisualizer
from .types import AnalysisReport, CheckResult, Requirement, Severity
from mediapipe.tasks.python import vision
from typing import Tuple, Union, Optional
import math



class LogicController:
    def __init__(self):
        self.face_detector = DetectionVisualizer()


    # Utility functions

    def _dist(self, p, q) -> float:
        '''Finds the distance between two points.'''
        dx, dy = (p.x - q.x), (p.y - q.y)
        return math.hypot(dx, dy)

    def _mean(self, xs):
        '''Returns the average of numbers in an interable'''
        return sum(xs) / max(1, len(xs))


    def run_analysis(self, image_path: str) -> AnalysisReport:
        # run each method and alert the user somehow
        face_detector_result = self.face_detector.analyze_image(image_path)
        face_landmarker_result = self.face_detector.analyze_landmarks(image_path)

        checks = []

        # 1) Face present
        face_present = self._is_face_in_image(face_detector_result)
        checks.append(face_present)

        # 2) Single face
        single_face = self._is_single_face(face_detector_result)
        checks.append(single_face)

        # 3 Landmarks present
        landmarks_present = self._are_landmarks_present(face_landmarker_result)
        checks.append(landmarks_present)

        # 4 Eyes open - TODO



        overall_pass = all(c.passed for c in checks)

        return AnalysisReport(
            image=f"{image_path}",
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
                message="No faces detected.",
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
                message="No face landmarks detected (face not fully visible).",
                details={"faces_with_landmarks": 0}
            )

    def eyes_visible_check(self, result: vision.FaceLandmarkerResult) -> CheckResult:

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
        LEFT_EYE_LANDMARKS = [33, 133]  # left eye outer & inner
        RIGHT_EYE_LANDMARKS = [362, 263]  # right eye outer & inner

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


        left_eye_width = self._dist(left_eye[0], left_eye[1])

        right_eye_width = self._dist(right_eye[0], right_eye[1])

        # Heuristic: eyes should have a nontrivial width (> ~0.02 in normalized coords)
        eyes_visible = left_eye_width > 0.02 and right_eye_width > 0.02

        if eyes_visible:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=True,
                severity=Severity.INFO,
                message="Both eyes are visible.",
                details={
                    "left_eye_width": left_eye_width,
                    "right_eye_width": right_eye_width
                }
            )
        else:
            return CheckResult(
                requirement=Requirement.EYES_VISIBLE,
                passed=False,
                severity=Severity.ERROR,
                message="Eyes not visible or too small (may be closed/obstructed).",
                details={
                    "left_eye_width": left_eye_width,
                    "right_eye_width": right_eye_width
                }
            )
