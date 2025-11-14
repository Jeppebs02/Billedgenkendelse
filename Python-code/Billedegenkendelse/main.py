from tkinter.tix import IMAGE

import cv2
import os
import numpy as np



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logic.logic_controller import LogicController


# Syntaxen er from package.file import class/function
from logic.is_face_present.face_detector import DetectionVisualizer


# Denne funktion opretter en mappe, hvis den ikke findes
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)




if __name__ == '__main__':

    # Configuration

    #IMAGE_FILE_NAME = "1708197244569.jpg"
    #IMAGE_FILE_NAME = "ikke_centreret.png"
    #IMAGE_FILE_NAME = "centreret.png"
    #IMAGE_FILE_NAME = "closedeyes.jpg"
    #IMAGE_FILE_NAME = "to_close.png"
    IMAGE_FILE_NAME = "man_with_hat.jpg"
    #IMAGE_FILE_NAME = "open_mouth.jpg"
    #IMAGE_FILE_NAME = "MatiasNice.jpg"
    #IMAGE_FILE_NAME = "oscarimg.jpg"

    IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)

    FACE_DETECTOR_MODEL_FILE = os.path.join("models", "blaze_face_short_range.tflite")

    FACE_LANDMARK_MODEL_FILE = os.path.join("models", "face_landmarker.task")

    DETECTOR_OUT_FILE = IMAGE_FILE_NAME + "_annotated.jpg"

    LANDMARK_OUT_FILE = IMAGE_FILE_NAME + "_landmark_annotated.jpg"

    if not os.path.isfile(IMAGE_FILE):
        raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

    if not os.path.isfile(FACE_DETECTOR_MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {FACE_DETECTOR_MODEL_FILE}\n"
            "Place a MediaPipe face detection .tflite model at this path."
        )


    logic_controller = LogicController()

    report = logic_controller.run_analysis(IMAGE_FILE)

    report.print_console()

    vis = DetectionVisualizer()
    det_res = vis.analyze_image(IMAGE_FILE_NAME)

    bgr = cv2.imread(os.path.join("images", IMAGE_FILE_NAME))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    overlay = vis.annotate_center_and_size(
        rgb, det_res,
        tol_x=0.08, tol_y=0.50,
        min_height_ratio=0.40,  # “for langt fra” hvis under
        max_height_ratio=0.55  # “for tæt på” hvis over
    )

    cv2.imwrite(os.path.join("out", f"{IMAGE_FILE_NAME}_center_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

