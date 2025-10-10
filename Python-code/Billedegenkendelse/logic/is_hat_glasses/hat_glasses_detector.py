# is_hat_glasses/hat_glasses_detector.py
from typing import Dict, Set
from ultralytics import YOLO
from PIL import Image
import os

class HatGlassesDetector:

    def __init__(self,
                 model_path: str = os.path.join("models", "best.pt"),
                 targets: Set[str] = {"Hat", "Glasses"}):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
        #This loads the model
        self.model = YOLO(model_path)
        # class ids are numbered from 0, so here we map id to name like Hat and Glasses
        self.class_names = self.model.names
        # Targets we want to detect, so Hat and Glasses
        self.targets = targets

    def analyze_image(self, image_file_name: str) -> Dict[str, float]:
        """
        image_file_name: Image in /images folder
        Returns dict with max confidence per target, like: {"Hat": 0.72, "Glasses": 0.00}
        """
        image_path = os.path.join("images", image_file_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image (RGB), ligesom i face_detector
        img = Image.open(image_path).convert("RGB")
        # returns a list of results, we take the first [0] since we have only one image :)
        result = self.model(img, verbose=False)[0]

        # Initialize confidence, de skal være 0 til at starte med
        confs = {name: 0.0 for name in self.targets}

        # Do detections and keep the highest confidence :)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # Get class name from id
            cls_name = self.class_names.get(cls_id, str(cls_id))
            # Check if the new confidence is higher than the stored one, if so replace the stored confidence
            if cls_name in confs and conf > confs[cls_name]:
                confs[cls_name] = conf

        return confs
