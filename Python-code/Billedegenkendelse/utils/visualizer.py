from typing import Tuple, Union, Optional, List
import math
import cv2
import numpy as np
import os
import mediapipe as mp
from PIL.ImageChops import overlay
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

from logic.face_direction.face_looking_at_camera import FaceLookingAtCamera
from logic.head_placement.head_centering_validator import HeadCenteringConfig
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

        #Annotations

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

    def annotate_center_and_size(self, image_bytes, detection_result, cfg: Optional[HeadCenteringConfig] = None) -> np.ndarray:
        """
        Visualiserer centrerings- og størrelseskrav:
          - Grøn toleranceboks for centreringskrav
          - Blå ansigtsboks og rødt centerpunkt
          - Gul/magenta guidebokse for min/max hovedstørrelse
        """
        # Brug delt config – eller defaults hvis ingen er givet
        cfg = cfg or HeadCenteringConfig()

        tol_x = cfg.tol_x
        tol_y = cfg.tol_y
        min_height_ratio = cfg.min_height_ratio
        max_height_ratio = cfg.max_height_ratio

        OUT_FILE_NAME = self.create_out_name(self.IMAGE_FILE_NAME) + "_center_and_size" + self.create_out_ext(
            self.IMAGE_FILE_NAME)
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)

        # draw detection on image
        image_rgb = image_io.bytes_to_rgb_np(image_bytes)

        annot = image_rgb.copy()
        H, W = annot.shape[:2]

        # --- 1) billedcenter + toleranceboks ---
        cx_img, cy_img = int(W * 0.5), int(H * 0.5)
        tol_w, tol_h = int(tol_x * W), int(tol_y * H)
        cv2.drawMarker(
            annot, (cx_img, cy_img), (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=40, thickness=2
        )
        cv2.rectangle(
            annot,
            (cx_img - tol_w, cy_img - tol_h),
            (cx_img + tol_w, cy_img + tol_h),
            (0, 255, 0),
            2,
        )

        # --- 2) primær detektion ---
        if not getattr(detection_result, "detections", None):
            return annot

        det = max(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height
        )
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


        # Save image
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annot, cv2.COLOR_RGB2BGR))
        print(f"Annotated image written to {OUT_FILE}")

        return annot

    def annotate_looking_straight(
            self,
            image_bytes,
            landmark_result,
            yaw_tolerance: float,
            pitch_tolerance: float,
    ) -> np.ndarray:
        """
        Visualisering af om personen kigger mod kameraet:
          - Blå linjer: øre-til-øre og hage-til-pande (faktisk hoved-akse)
          - Blå pil: yaw-retningen
          - Yaw-bar under hovedet, der viser hvor langt man er fra grænsen
          - Info-boks med yaw/pitch + status (OK / Not straight)
        """
        if not landmark_result.face_landmarks or len(landmark_result.face_landmarks) == 0:
            raise ValueError("No landmarks found.")

        # Filnavn og mappe
        OUT_FILE_NAME = (
                self.create_out_name(self.IMAGE_FILE_NAME)
                + "_annotate_looking_straight"
                + self.create_out_ext(self.IMAGE_FILE_NAME)
        )
        OUT_FILE = os.path.join(self.OUT_DIR, OUT_FILE_NAME)
        os.makedirs(self.OUT_DIR, exist_ok=True)

        # 1) Beregn yaw/pitch
        yaw, pitch = FaceLookingAtCamera.get_yaw_pitch(landmark_result)

        # 2) Dekod billede
        image_rgb = image_io.bytes_to_rgb_np(image_bytes)
        annot = image_rgb.copy()
        H, W = annot.shape[:2]

        landmarks = landmark_result.face_landmarks[0]

        LEFT_EAR = 234
        RIGHT_EAR = 454
        CHIN = 152
        FOREHEAD = 10

        def to_px(lm):
            return int(lm.x * W), int(lm.y * H)

        left_ear_px = to_px(landmarks[LEFT_EAR])
        right_ear_px = to_px(landmarks[RIGHT_EAR])
        chin_px = to_px(landmarks[CHIN])
        forehead_px = to_px(landmarks[FOREHEAD])

        # Er vi indenfor tolerance?
        inside_tolerance = (
                abs(yaw) <= yaw_tolerance and
                abs(pitch) <= pitch_tolerance
        )
        main_color = (0, 255, 0) if inside_tolerance else (0, 0, 255)  # grøn / rød
        aux_color = (255, 0, 0)  # blå-ish til hjælpegrafik

        # 3) Akser: øre-til-øre + hage-til-pande
        cv2.line(annot, left_ear_px, right_ear_px, main_color, 3)
        cv2.line(annot, chin_px, forehead_px, main_color, 3)

        # Midtpunkt i hovedet (bruges til pil + bar)
        mid_x = (left_ear_px[0] + right_ear_px[0]) // 2
        mid_y = (left_ear_px[1] + right_ear_px[1]) // 2

        # 4) Pil der viser yaw-retningen
        arrow_len = int(0.18 * W)
        angle_rad = math.radians(yaw)
        end_x = int(mid_x + arrow_len * math.cos(angle_rad))
        end_y = int(mid_y - arrow_len * math.sin(angle_rad))
        cv2.arrowedLine(
            annot,
            (mid_x, mid_y),
            (end_x, end_y),
            main_color,
            3,
            tipLength=0.25
        )

        # 5) Yaw-bar under hovedet (visuelt "meter")
        bar_y = min(H - 40, mid_y + int(0.20 * H))
        bar_len = int(0.30 * W)
        bar_x1 = mid_x - bar_len
        bar_x2 = mid_x + bar_len

        # Baggrundslinje
        cv2.line(annot, (bar_x1, bar_y), (bar_x2, bar_y), (200, 200, 200), 2)

        # Marker nulpunkt i midten
        cv2.line(annot, (mid_x, bar_y - 6), (mid_x, bar_y + 6), (180, 180, 180), 2)

        # Beregn pointer-position ift. tolerance (clamp, så den ikke løber ud af billedet)
        if yaw_tolerance > 0:
            norm = max(-2.0, min(2.0, yaw / yaw_tolerance))  # -2..2
        else:
            norm = 0.0
        pointer_x = int(mid_x + norm * bar_len)
        cv2.circle(annot, (pointer_x, bar_y), 7, main_color, -1)

        # Tegn toleranceområde som lysere segment (± yaw_tolerance)
        if yaw_tolerance > 0:
            tol_norm = min(1.0, yaw_tolerance / yaw_tolerance)  # = 1.0, men bevarer logikken
            tol_x1 = int(mid_x - tol_norm * bar_len)
            tol_x2 = int(mid_x + tol_norm * bar_len)
            cv2.line(annot, (tol_x1, bar_y), (tol_x2, bar_y), aux_color, 4)

        # 6) Semi-transparent infoboks med tekst
        panel_w, panel_h = 420, 80
        x0, y0 = 10, 10
        x1, y1 = x0 + panel_w, y0 + panel_h

        overlay = annot.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), main_color, -1)
        alpha = 0.25
        annot = cv2.addWeighted(overlay, alpha, annot, 1 - alpha, 0)

        status_text = "OK" if inside_tolerance else "Not straight"
        text1 = f"Yaw: {yaw:.1f}°  (±{yaw_tolerance}°)"
        text2 = f"Pitch: {pitch:.1f}°  (±{pitch_tolerance}°)"
        text3 = f"Status: {status_text}"

        txt_color = (255, 255, 255)
        cv2.putText(annot, text1, (x0 + 10, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)
        cv2.putText(annot, text2, (x0 + 10, y0 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)
        cv2.putText(annot, text3, (x0 + 10, y0 + 71),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)

        # 7) Gem resultatet
        cv2.imwrite(OUT_FILE, cv2.cvtColor(annot, cv2.COLOR_RGB2BGR))
        print(f"Annotated image written to {OUT_FILE}")

        return annot

