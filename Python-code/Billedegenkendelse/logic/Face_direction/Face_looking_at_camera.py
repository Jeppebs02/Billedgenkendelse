from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.tasks.python.vision.face_detector import FaceDetector
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

import logic.types
from logic.types import CheckResult



class FaceLookingAtCamera:
    def __init__(self, tolerance_degrees=10):
        self.tolerance = tolerance_degrees  # allowed angle deviation

    def face_detector(self, face_detector_result_from_Logic: FaceLandmarkerResult) -> CheckResult:
        # If no faces found → fail
        if len(face_detector_result_from_Logic.face_blendshapes) == 0:
            return CheckResult(success=False, message="No face detected")

        # Extract head pose angles
        # Many pipelines store this in face_blendshapes, or another attribute
        # Adjust according to your exact data structure
        try:
            yaw = face_detector_result_from_Logic.head_rotation_yaw
            pitch = face_detector_result_from_Logic.head_rotation_pitch
        except:
            return CheckResult(success=False, message="Head rotation data missing")

        # Determine if looking straight
        if abs(yaw) <= self.tolerance and abs(pitch) <= self.tolerance:
            return CheckResult(success=True, message="Looking straight at camera")
        else:
            return CheckResult(
                success=False,
                message=f"Not looking straight (yaw={yaw:.2f}, pitch={pitch:.2f})"
            )