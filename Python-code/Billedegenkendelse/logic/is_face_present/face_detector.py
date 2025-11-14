from typing import Tuple, Union, Optional, List
import math
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

from logic.types import CheckResult, Requirement, Severity
from utils.image_io import bytes_to_rgb_np


class DetectionVisualizer:
    def __init__(self,
                 margin: int = 10,
                 row_size: int = 10,
                 font_size: int = 1,
                 font_thickness: int = 1,
                 text_color: Tuple[int, int, int] = (255, 0, 0),
                 DETECTOR_MODEL_NAME: str = "blaze_face_short_range.tflite",
                 LANDMARKER_MODEL_NAME: str = "face_landmarker.task",

                 ):

        self.DETECTOR_MODEL_FILE = os.path.join("models", DETECTOR_MODEL_NAME)
        if not os.path.isfile(self.DETECTOR_MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found: {self.DETECTOR_MODEL_FILE}\n"
                "Place a MediaPipe face detection .tflite model at this path."
            )

        self.LANDMARKER_MODEL_FILE = os.path.join("models", LANDMARKER_MODEL_NAME)
        if not os.path.isfile(self.LANDMARKER_MODEL_FILE):
            raise FileNotFoundError(
                f"Landmarker model not found: {self.LANDMARKER_MODEL_FILE}\n"
                "Place the face landmarker .task model at this path."
            )

        self.OUT_DIR = "out"
        self.MODEL_NAME = DETECTOR_MODEL_NAME
        self.margin = margin
        self.row_size = row_size
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.text_color = text_color

    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)


    #hjælper
    def _dist(self, p, q) -> float:
        '''Finds the distance between two points.'''
        dx, dy = (p.x - q.x), (p.y - q.y)
        return math.hypot(dx, dy)
    #hjælper
    def _mean(self, xs):
        '''Returns the average of numbers in an interable'''
        return sum(xs) / max(1, len(xs))
    #Øjne logik
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

    #hjælper
    def _normalized_to_pixel_coordinates(
            self,
            normalized_x: float,
            normalized_y: float,
            image_width: int,
            image_height: int
    ) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (
                    value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            return None

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    #visualize
    def visualize(self, image_rgb: np.ndarray, detection_result) -> np.ndarray:
        """
        Draws bounding boxes/keypoints/labels on an RGB image and returns RGB.
        """
        annotated_image = image_rgb.copy()
        height, width, _ = annotated_image.shape

        for detection in detection_result.detections:
            # Bounding box
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(annotated_image, start_point, end_point, self.text_color, 3)

            # Keypoints (normalized → pixels)
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
                if keypoint_px:
                    cv2.circle(annotated_image, keypoint_px, 2, (0, 255, 0), 2)

            # Label + score
            category = detection.categories[0]
            category_name = category.category_name or ""
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (self.margin + bbox.origin_x, self.margin + self.row_size + bbox.origin_y)
            cv2.putText(
                annotated_image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                self.font_size,
                self.text_color,
                self.font_thickness,
            )

        return annotated_image
    #visualize
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in
                                  face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()

    # Detection functions

    def _build_face_detector_options(self, options: Optional[vision.FaceDetectorOptions] = None):
        if options is not None:
            return vision.FaceDetector.create_from_options(options)

        base_options = python.BaseOptions(model_asset_path=self.DETECTOR_MODEL_FILE)

        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
                min_suppression_threshold=0.3,
            )
        )

    def analyze_image(self, IMAGE_FILE_NAME: str, options: Optional[vision.FaceDetectorOptions] = None) -> vision.FaceDetectorResult:
        """
        Analyzes an image to detect faces and returns the raw result.
        """
        IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)
        if not os.path.isfile(IMAGE_FILE):
            raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

        if not os.path.isfile(self.DETECTOR_MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found: {self.DETECTOR_MODEL_FILE}\n"
                "Place a MediaPipe face detection .tflite model at this path."
            )

        self.ensure_dir(self.OUT_DIR)

        # Create detector and config

        detector = self._build_face_detector_options(options)


        # Load into mediapipe
        mp_image = mp.Image.create_from_file(IMAGE_FILE)

        # Run detection
        detection_result = detector.detect(mp_image)

        return detection_result

    def analyze_bytes(self, image_bytes: bytes, options: Optional[vision.FaceDetectorOptions] = None) -> vision.FaceDetectorResult:
        """
        Bytes → RGB ndarray → MediaPipe Image → FaceDetector.detect()
        Mirrors analyze_image(), but without touching the filesystem.
        """
        if not os.path.isfile(self.DETECTOR_MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found: {self.DETECTOR_MODEL_FILE}\n"
                "Place a MediaPipe face detection .tflite model at this path."
            )

        # Build detector
        detector = self._build_face_detector_options(options)

        # Convert bytes to RGB ndarray then to MediaPipe Image
        rgb = bytes_to_rgb_np(image_bytes)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        detection_result = detector.detect(mp_image)
        return detection_result

    def analyze_and_annotate_image(self, IMAGE_FILE_NAME: str, OUT_FILE_NAME: str, options: Optional[vision.FaceDetectorOptions] = None) -> vision.FaceDetectorResult:
        """
        Analyzes, annotates, and plots face detection results.
        """


        detection_result = self.analyze_image(IMAGE_FILE_NAME, options)

        print(f"Found {len(detection_result.detections)} faces")
        if len(detection_result.detections) == 0:
            print("No faces found, exiting")
            return detection_result

        IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)

        # Visualize detection result.
        image_rgb = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
        annotated_image = self.visualize(image_rgb, detection_result)

        # Save the result
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image written to {OUT_FILE}")

        return detection_result


    # Landmark functions

    def _build_face_landmarker_options(self, options: Optional[vision.FaceLandmarkerOptions] = None):
        if options is not None:
            return vision.FaceLandmarker.create_from_options(options)

        base_options = python.BaseOptions(model_asset_path=self.LANDMARKER_MODEL_FILE)
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
        )

    def analyze_landmarks(self, IMAGE_FILE_NAME: str, options: Optional[vision.FaceLandmarkerOptions] = None) -> vision.FaceLandmarkerResult:
        """
        Analyzes an image to detect face landmarks and returns the raw result.
        Defaults to detecting one face. Please use custom options to detect more.
        """
        IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)
        if not os.path.isfile(IMAGE_FILE):
            raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

        if not os.path.isfile(self.LANDMARKER_MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found: {self.LANDMARKER_MODEL_FILE}\n"
                "Place the MediaPipe face landmarker .task model at this path."
            )

        self.ensure_dir(self.OUT_DIR)

        # Create landmarker and config
        if options is None:
            # Create default options if none are provided.
            base_options = python.BaseOptions(model_asset_path=self.LANDMARKER_MODEL_FILE)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            landmarker = vision.FaceLandmarker.create_from_options(options)

        # Load into mediapipe
        mp_image = mp.Image.create_from_file(IMAGE_FILE)

        # Run detection
        detection_result = landmarker.detect(mp_image)

        return detection_result

    def analyze_landmarks_bytes(self, image_bytes: bytes, options: Optional[vision.FaceLandmarkerOptions] = None) -> vision.FaceLandmarkerResult:
        """
        Bytes → RGB ndarray → MediaPipe Image into FaceLandmarker.detect() (just like analyze_landmarks :) )
        Mirrors analyze_landmarks(), but without saving files
        """
        if not os.path.isfile(self.LANDMARKER_MODEL_FILE):
            raise FileNotFoundError(
                f"Landmarker model not found: {self.LANDMARKER_MODEL_FILE}\n"
                "Place the face landmarker .task model at this path."
            )

        # Build landmarker
        landmarker = self._build_face_landmarker_options(options)

        # Convert bytes → MediaPipe Image (RGB)
        rgb = bytes_to_rgb_np(image_bytes)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        detection_result = landmarker.detect(mp_image)
        return detection_result

    def analyze_and_annotate_landmarks(self, IMAGE_FILE_NAME: str, OUT_FILE_NAME: str, options: Optional[vision.FaceLandmarkerOptions] = None) -> vision.FaceLandmarkerResult:
        """
        Analyzes, annotates, and plots face landmark results.
        Defaults to detecting and plotting one face. Please use custom options to detect and plot more.
        """
        # Analyze the image to get landmark data
        detection_result = self.analyze_landmarks(IMAGE_FILE_NAME, options)

        # Check for landmarks
        print(f"Found {len(detection_result.face_landmarks)} face(s) and landmarks")
        if len(detection_result.face_landmarks) == 0:
            print("No face landmarks found, exiting.")
            return detection_result

        IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)

        # draw detection on image
        image_rgb = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
        annotated_image = self.draw_landmarks_on_image(image_rgb, detection_result)

        # Save image
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated landmark image written to {OUT_FILE}")

        return detection_result


    # Check Results functions

    def is_face_in_image(self, result: vision.FaceDetectorResult) -> CheckResult:
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


    def is_single_face(self, result: vision.FaceDetectorResult) -> CheckResult:
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


    def are_landmarks_present(self, result: vision.FaceLandmarkerResult) -> CheckResult:

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


    def eyes_visible_check(self, result: vision.FaceLandmarkerResult, ear_threshold: float = 0.2) -> CheckResult:

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


    def mouth_closed_check(self, result: vision.FaceLandmarkerResult, max_gap_ratio: float = 0.03) -> CheckResult:
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

    #visualize
    def annotate_center_and_size(self,
                                       image_rgb: np.ndarray,
                                       detection_result,
                                       tol_x: float = 0.08,
                                       tol_y: float = 0.50,
                                       min_height_ratio: float = 0.40,
                                       max_height_ratio: float = 0.55) -> np.ndarray:
        """
        Minimal version uden tekst — kun visual guides:
          - Grøn toleranceboks for centreringskrav
          - Blå ansigtsboks og rødt centerpunkt
          - Gul/magenta guidebokse for min/max hovedstørrelse
        """
        annot = image_rgb.copy()
        H, W = annot.shape[:2]

        # --- 1) billedcenter + toleranceboks ---
        cx_img, cy_img = int(W * 0.5), int(H * 0.5)
        tol_w, tol_h = int(tol_x * W), int(tol_y * H)
        cv2.drawMarker(annot, (cx_img, cy_img), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=40, thickness=2)
        cv2.rectangle(annot, (cx_img - tol_w, cy_img - tol_h),
                      (cx_img + tol_w, cy_img + tol_h), (0, 255, 0), 2)

        # --- 2) primær detektion ---
        if not getattr(detection_result, "detections", None):
            return annot

        det = max(detection_result.detections,
                  key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = det.bounding_box
        x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height

        # --- 3) ansigtsboks + centerpunkt ---
        cv2.rectangle(annot, (x, y), (x + w, y + h), (255, 0, 0), 2)
        head_cx_px = int(x + w / 2.0)
        head_cy_px = int(y + h / 2.0)
        cv2.circle(annot, (head_cx_px, head_cy_px), 6, (0, 0, 255), -1)

        # --- 4) tegn min/max hovedstørrelsesguider ---
        aspect = w / float(h) if h > 0 else 1.0

        def draw_size_guide(height_ratio_val: float, color):
            gh = int(height_ratio_val * H)
            gw = int(aspect * gh)
            gx = int(head_cx_px - gw / 2)
            gy = int(head_cy_px - gh / 2)
            cv2.rectangle(annot, (gx, gy), (gx + gw, gy + gh), color, 2)

        draw_size_guide(max_height_ratio, (255, 0, 255))  # magenta = max
        draw_size_guide(min_height_ratio, (0, 255, 255))  # gul = min

        return annot