from tkinter.tix import IMAGE

import cv2
import os
import numpy as np



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logic.logic_controller import LogicController


# Syntaxen er from package.file import class/function.
from logic.face_present_logic.face_detector import DetectionVisualizer


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
    #IMAGE_FILE_NAME = "man_with_hat.jpg"
    #IMAGE_FILE_NAME = "open_mouth.jpg"
    #IMAGE_FILE_NAME = "MatiasNice.jpg"
    #IMAGE_FILE_NAME = "oscarimg.jpg"
    #IMAGE_FILE_NAME = "expoMan.png"
    #IMAGE_FILE_NAME = "blackman.png"
    #IMAGE_FILE_NAME = "pixelWoman.png"
    #IMAGE_FILE_NAME = "manshouldfaileyes.jpg"
    #IMAGE_FILE_NAME = "man_with_shadow.png"

    IMAGE_FILE_NAME = "man_with_shadow.png"


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

    report = logic_controller.run_analysis(IMAGE_FILE, IMAGE_FILE_NAME)

    report.print_console()


