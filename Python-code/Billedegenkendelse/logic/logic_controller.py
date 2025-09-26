from logic.is_face_present.face_detector import DetectionVisualizer
from logic.types import AnalysisReport, CheckResult, Requirement, Severity
from mediapipe.tasks.python import vision



class LogicController:
    def __init__(self):
        self.face_detector = DetectionVisualizer()


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
