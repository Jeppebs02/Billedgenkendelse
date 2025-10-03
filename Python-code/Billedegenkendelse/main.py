import cv2
import os
import numpy as np


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logic.logic_controller import LogicController


# Syntaxen er from package.file import class/function
from logic.is_face_present.face_detector import DetectionVisualizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Denne funktion opretter en mappe, hvis den ikke findes
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    print_hi('PyCharm')

    # Configuration
    IMAGE_FILE_NAME = "asian.jpg"

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

    report = logic_controller.run_analysis(IMAGE_FILE_NAME)

    report.print_console()


