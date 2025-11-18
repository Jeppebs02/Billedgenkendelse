# head_centering_check.py
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from mediapipe.tasks.python import vision
from utils.types import CheckResult, Requirement, Severity

@dataclass
class HeadCenteringConfig:
    # Center-tolerancer (andel af billedstørrelse)
    tol_x: float = 0.08
    tol_y: float   = 0.10
    # Størrelseskrav baseret på BBOX-højde / billedhøjde (mere stabil end areal)
    min_height_ratio: float = 0.40   # "for langt fra" hvis under dette (justér efter krav)
    max_height_ratio: float = 0.55   # "for tæt på" hvis over dette (justér efter krav)

class HeadCenteringValidator:
    def __init__(self, config: Optional[HeadCenteringConfig] = None):
        self.cfg = config or HeadCenteringConfig()

    @staticmethod
    def _primary_detection(result: vision.FaceDetectorResult):
        if not result or not getattr(result, "detections", None):
            return None
        return max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)

    def _evaluate(self, det_res: vision.FaceDetectorResult, w: int, h: int) -> CheckResult:
        det = self._primary_detection(det_res)
        if det is None:
            return CheckResult(
                requirement=Requirement.HEAD_CENTERED,
                passed=False,
                severity=Severity.ERROR,
                message="No face detected – cannot check centering.",
                details={}
            )

        bb = det.bounding_box
        x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height

        # center-normalisering
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        head_cx = cx / float(w)
        head_cy = cy / float(h)
        dx = head_cx - 0.5
        dy = head_cy - 0.5

        # Størrelseskrav via HØJDE-ratio (bh/h)
        height_ratio = bh / float(h)

        within_center = (abs(dx) <= self.cfg.tol_x) and (abs(dy) <= self.cfg.tol_y)
        within_size = (self.cfg.min_height_ratio <= height_ratio <= self.cfg.max_height_ratio)
        passed = within_center and within_size

        if not within_center and not within_size:
            msg = "Head not centered and size out of range"
        elif not within_center:
            msg = "Head not centered"
        elif not within_size:
            # Skeln 'for tæt på' vs 'for langt fra' for klar feedback
            if height_ratio > self.cfg.max_height_ratio:
                msg = "Head too large (too close)"
            else:
                msg = "Head too small (too far)"
        else:
            msg = "OK"
        #print(f"[DEBUG] dx={dx:.3f}, dy={dy:.3f}, area_ratio={area_ratio:.3f}, "
         #     f"center_ok={within_center}, size_ok={within_size}")
        return CheckResult(
            requirement=Requirement.HEAD_CENTERED,
            passed=passed,
            severity=Severity.INFO if passed else Severity.ERROR,
            message=msg,
            details={
                "head_center_norm": (round(head_cx, 4), round(head_cy, 4)),
                "delta_norm": (round(dx, 4), round(dy, 4)),
                "height_ratio": round(height_ratio, 4),  # ← NYTTIG TIL DEBUG
                "height_limits": {
                    "min_height_ratio": self.cfg.min_height_ratio,
                    "max_height_ratio": self.cfg.max_height_ratio
                },
                "tolerances": {"tol_x": self.cfg.tol_x, "tol_y": self.cfg.tol_y},
                "bbox_xywh": (x, y, bw, bh),
                "image_wh": (w, h),
            }
        )

    def check_from_detection_file(self, det_res: vision.FaceDetectorResult, image_file_name: str) -> CheckResult:
        image_path = os.path.join("images", image_file_name)
        bgr = cv2.imread(image_path)
        if bgr is None:
            return CheckResult(
                requirement=Requirement.HEAD_CENTERED,
                passed=False,
                severity=Severity.ERROR,
                message=f"Could not read image: {image_path}",
                details={}
            )
        h, w = bgr.shape[:2]
        return self._evaluate(det_res, w, h)

    def check_from_detection_bytes(self, det_res: vision.FaceDetectorResult, image_bytes: bytes) -> CheckResult:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return CheckResult(
                requirement=Requirement.HEAD_CENTERED,
                passed=False,
                severity=Severity.ERROR,
                message="Could not decode image bytes.",
                details={}
            )
        h, w = bgr.shape[:2]
        return self._evaluate(det_res, w, h)
