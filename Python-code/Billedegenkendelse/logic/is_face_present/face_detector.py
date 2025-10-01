from typing import Tuple, Union, Optional
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

        if options is None:
            base_options = python.BaseOptions(model_asset_path=self.DETECTOR_MODEL_FILE)
            # "The minimum confidence score for the face detection to be considered successful."
            # Default er 0.5
            detection_confidence = 0.5

            # Default er image men vi kan også bruge VIDEO eller LIVE_STREAM
            running_mode = vision.RunningMode.IMAGE

            # "The minimum non-maximum-suppression threshold for face detection to be considered overlapped."
            # Default er 0.3
            min_suppression_threshold = 0.3

            options = vision.FaceDetectorOptions(base_options=base_options,
                                                 min_detection_confidence=detection_confidence,
                                                 running_mode=running_mode,
                                                 min_suppression_threshold=min_suppression_threshold)

            detector = vision.FaceDetector.create_from_options(options)


        else:
            detector = vision.FaceDetector.create_from_options(options)

        # Load into mediapipe
        mp_image = mp.Image.create_from_file(IMAGE_FILE)

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

