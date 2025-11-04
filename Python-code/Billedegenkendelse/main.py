import cv2
import os
import numpy as np
from pathlib import Path

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



    #IMAGE_FILE_NAME = "closedeyes.jpg"
    #IMAGE_FILE_NAME = "man_with_hat.jpg"
    #IMAGE_FILE_NAME = "open_mouth.jpg"
    BASE_DIR = Path(__file__).resolve().parent  # folder of main.py

    IMAGE_FILE_NAME = "person_sunglasses.jpg"
    IMAGE_FILE = BASE_DIR / "images" / IMAGE_FILE_NAME
    FACE_DETECTOR_MODEL_FILE = BASE_DIR / "models" / "blaze_face_short_range.tflite"
    FACE_LANDMARK_MODEL_FILE = BASE_DIR / "models" / "face_landmarker.task"

    if not IMAGE_FILE.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

    logic_controller = LogicController()
    report = logic_controller.run_analysis(str(IMAGE_FILE))  # pass full path
    report.print_console()
