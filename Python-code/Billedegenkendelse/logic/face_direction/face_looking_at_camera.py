from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.tasks.python.vision.face_detector import FaceDetector
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from logic.types import *
import math

import logic.types
from logic.types import CheckResult

class FaceLookingAtCamera:
    def __init__(self, tolerance_yaw_degrees=7, tolerance_pitch_degrees=15):
        self.yaw_tolerance = tolerance_yaw_degrees
        self.pitch_tolerance = tolerance_pitch_degrees
    def get_yaw_pitch(face_result: FaceLandmarkerResult) -> tuple[float, float]:
        """
        Computes head yaw & pitch (in degrees) using MediaPipe 3D landmarks,
        normalized so forward = 0°.
        """


    def face_detector(self, result: FaceLandmarkerResult) -> CheckResult:

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            raise ValueError("No landmarks found.")

        landmarks = result.face_landmarks[0]

        LEFT_EAR = 234
        RIGHT_EAR = 454
        CHIN = 152
        FOREHEAD = 10

        left_ear = landmarks[LEFT_EAR]
        right_ear = landmarks[RIGHT_EAR]
        chin = landmarks[CHIN]
        forehead = landmarks[FOREHEAD]

        # --- YAW ---
        ear_dx = left_ear.x - right_ear.x
        ear_dz = left_ear.z - right_ear.z
        yaw = math.degrees(math.atan2(ear_dz, ear_dx))

        # --- PITCH ---
        dy = forehead.y - chin.y
        dz = forehead.z - chin.z
        pitch = math.degrees(math.atan2(dz, dy))

        # --- NORMALIZE angles to [-90°, +90°] ---
        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        details: Dict[str, Any]={
            "Yaw": round(yaw,4),
            "Pitch": round(pitch,4)
        }


        if abs(yaw) <= self.yaw_tolerance and abs(pitch) <= self.pitch_tolerance:
            return CheckResult(
                requirement=Requirement.FACE_LOOKING_AT_CAMERA,
                passed=True,
                severity=Severity.INFO,
                message=f"Looking straight at camera",
                details = details
            )
        else:
            return CheckResult(
                requirement=Requirement.FACE_LOOKING_AT_CAMERA,
                passed=False,
                severity=Severity.ERROR,
                message=f"Not looking straight",
                details = details
            )


