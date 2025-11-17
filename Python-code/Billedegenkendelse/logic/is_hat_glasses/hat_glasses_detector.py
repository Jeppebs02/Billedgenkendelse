from typing import Dict, Set
from ultralytics import YOLO
from PIL import Image
import os

from logic.types import CheckResult, Requirement, Severity
from utils.image_io import bytes_to_pil

class HatGlassesDetector:

    # For yolo docs, look here https://docs.ultralytics.com/usage/python/

    def __init__(self,
                 model_path: str = os.path.join("models", "best.pt"),
                 targets: Set[str] = {"Hat", "Glasses"}):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
        #This loads the model from our path.
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



    def analyze_bytes(self, image_bytes: bytes) -> Dict[str, float]:
        """Return max confidence per target from in-memory bytes. Instead of an image"""
        img: Image.Image = bytes_to_pil(image_bytes)
        result = self.model(img, verbose=False)[0]

        confs = {name: 0.0 for name in self.targets}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            if cls_name in confs and conf > confs[cls_name]:
                confs[cls_name] = conf
        return confs



    def check_hats_and_glasses(self, image_path: str, threshold: float = 0.5) -> list[CheckResult]:
        """
        Run the YOLO hat and glasses detection.
        Returns two CheckResult objects: NO_HAT and NO_GLASSES. So to add it to the report, we need to use .extend()
        """
        yolo_confs = self.analyze_image(image_path)
        hat_conf = yolo_confs.get("Hat", 0.0)
        glasses_conf = yolo_confs.get("Glasses", 0.0)

        no_hat_pass = hat_conf < threshold
        no_glasses_pass = glasses_conf < threshold

        results = [
            CheckResult(
                requirement=Requirement.NO_HAT,
                passed=no_hat_pass,
                severity=Severity.ERROR if not no_hat_pass else Severity.INFO,
                message=("No hat detected." if no_hat_pass
                         else f"Hat detected."),
                details={"hat_confidence": hat_conf, "threshold": threshold}
            ),
            CheckResult(
                requirement=Requirement.NO_GLASSES,
                passed=no_glasses_pass,
                severity=Severity.ERROR if not no_glasses_pass else Severity.INFO,
                message=("No glasses detected." if no_glasses_pass
                         else f"Glasses detected."),
                details={"glasses_confidence": glasses_conf, "threshold": threshold}
            )
        ]

        return results



    def check_hats_and_glasses_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> list[CheckResult]:
        """
        YOLO hat/glasses on in-memory bytes.
        Returns two CheckResult objects: NO_HAT and NO_GLASSES. So to add it to the report, we need to use .extend()
        """
        yolo_confs = self.analyze_bytes(image_bytes)
        hat_conf = yolo_confs.get("Hat", 0.0)
        glasses_conf = yolo_confs.get("Glasses", 0.0)

        no_hat_pass = hat_conf < threshold
        no_glasses_pass = glasses_conf < threshold
        glasses_detected = glasses_conf >= threshold

        return [
            CheckResult(
                requirement=Requirement.NO_HAT,
                passed=no_hat_pass,
                severity=Severity.ERROR if not no_hat_pass else Severity.INFO,
                message=("No hat detected." if no_hat_pass
                         else f"Hat detected."),
                details={"hat_confidence": hat_conf, "threshold": threshold}
            ),
            # CheckResult(
            #     requirement=Requirement.NO_GLASSES,
            #     passed=no_glasses_pass,
            #     severity=Severity.ERROR if not no_glasses_pass else Severity.INFO,
            #     message=("No glasses detected." if no_glasses_pass
            #              else f"Glasses detected."),
            #     details={"glasses_confidence": glasses_conf, "threshold": threshold}
            # ),

            CheckResult(
                requirement=Requirement.NO_GLASSES,
                passed=True,  # <- altid true, så briller aldrig dumper
                severity=Severity.INFO,
                message=("Glasses detected (allowed)." if glasses_detected else "No glasses detected."),
                details={
                    "glasses_confidence": glasses_conf,
                    "threshold": threshold,
                    "detected": glasses_detected
                }
            ),
        ]
