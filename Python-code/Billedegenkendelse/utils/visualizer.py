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
from utils import image_io
from utils.image_io import bytes_to_rgb_np

class VisualizerHelper:
    def __init__(self, IMAGE_FILE_NAME):
        self.IMAGE_FILE_NAME = IMAGE_FILE_NAME
        self.OUT_DIR = self.create_out_dir(IMAGE_FILE_NAME)

        os.makedirs(self.OUT_DIR, exist_ok=True)

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

    def create_out_dir(self, IMAGE_FILE_NAME):
        name_without_ext,_= os.path.splitext(IMAGE_FILE_NAME)
        return os.path.join("out", name_without_ext)

    def create_out_name(self, IMAGE_FILE_NAME):
        name_without_ext,_= os.path.splitext(IMAGE_FILE_NAME)
        return name_without_ext

    def create_out_ext(self, IMAGE_FILE_NAME):
        name_without_ext, file_ext = os.path.splitext(IMAGE_FILE_NAME)
        return file_ext

    def visualize(self, image_rgb: np.ndarray, detection_result) -> np.ndarray:
        """
        Draws bounding boxes/keypoints/labels on an RGB image and returns RGB.
        """
        annotated_image = image_rgb.copy()
        height, width, _ = annotated_image.shape
        margin = 10
        row_size = 10
        font_size = 1
        font_thickness= 1
        text_color = (255, 0, 0)

        for detection in detection_result.detections:
            # Bounding box
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(annotated_image, start_point, end_point, text_color, 3)

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
            text_location = (margin + bbox.origin_x, margin + row_size + bbox.origin_y)
            cv2.putText(
                annotated_image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                font_size,
                text_color,
                font_thickness,
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

    def annotate_facedetector(self, image_bytes, detection_result):

        """Annotates, and plots face detection results."""

        OUT_FILE_NAME = self.create_out_name(self.IMAGE_FILE_NAME) + "_annotate_facedetector" + self.create_out_ext(self.IMAGE_FILE_NAME)
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)

        # Visualize detection result.
        image_rgb = image_io.bytes_to_rgb_np(image_bytes)
        annotated_image = self.visualize(image_rgb, detection_result)

        # Save the result
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image written to {OUT_FILE}")

        return True

    def annotate_landmarks(self,image_bytes, landmark_result):
        """
        Annotates, and plots face landmark results.
        """

        OUT_FILE_NAME = self.create_out_name(self.IMAGE_FILE_NAME) + "_annotate_landmarks" + self.create_out_ext(
            self.IMAGE_FILE_NAME)
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)

        # draw detection on image
        image_rgb = image_io.bytes_to_rgb_np(image_bytes)
        annotated_image = self.draw_landmarks_on_image(image_rgb, landmark_result)

        # Save image
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated landmark image written to {OUT_FILE}")

        return True

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