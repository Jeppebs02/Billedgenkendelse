# logic/glasses_logic/glasses_logic.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.tasks.python.vision.face_detector import FaceDetectorResult
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from logic.types import Requirement, Severity, CheckResult






@dataclass
class GlassesLogicConfig:
    # --- Sunglasses (tint) ---
    tint_darkness_ratio_k: float = 0.60     # eye_V < k * skin_V  => likely tinted. In other words Eyes must be at least 60% as bright as skin
    min_eye_texture: float = 15.0           # Laplacian variance threshold (too low => flat/tint)

    # --- Glare ---
    glare_area_pct: float = 0.015           # 1.5% of ROI area as bright specular area
    glare_max_blob_px: int = 200            # largest connected bright blob size
    glare_streak_ar_min: float = 3.0        # elongated “stripe” aspect ratio
    glare_streak_min_len_px: int = 40       # min major-axis length to count as streak
    require_iris_overlap_for_glare: bool = True

    # --- ROI handling ---
    expand_roi_pct: float = 0.12            # expand ROI by this fraction on each side

    # --- Preprocessing ---
    min_face_confidence: float = 0.5        # if you gate faces by score elsewhere, keep here for safety









class GlassesLogic:
    """
    Implements:
      - Requirement.NO_SUNGLASSES  (fail on tinted/dark lenses)
      - Requirement.NO_GLASSES_REFLECTION (fail on glare/shine obscuring eyes)

    Accepts:
      - image_bytes: bytes (optional but helpful for calculations)
      - face_detector_result: FaceDetectorResult (optional, used to sanity-check ROI)
      - face_landmarker_result: FaceLandmarkerResult (preferred for eye/iris landmarks)
      - yolo_glasses_boxes: Optional[List[Tuple[int,int,int,int]]]  # (x1,y1,x2,y2) in pixel coords

    Returns ready-to-append CheckResult objects so LogicController can just extend(checks).
    """

    def __init__(self, config: Optional[GlassesLogicConfig] = None):
        self.cfg = config or GlassesLogicConfig()

        # MediaPipe FaceMesh canonical indices (468 points model).
        # Eyes (outer contours) – common subset to bound ROIs robustly:
        self.left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]     # coarse hull-ish
        self.right_eye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362]

        # Iris centers (when using iris model variants): right: 468-472, left: 473-477
        # We'll use the central points 468 (right center) and 473 (left center) if present.
        self.right_iris_center_idx = 468
        self.left_iris_center_idx = 473

        # A few skin reference points around cheek/temple for brightness comparison
        # (picked as stable-ish external face points)
        self.skin_idx = [234, 93, 132, 361, 454, 323]

    # ------------------ Public API ------------------

    def has_glasses(self, face_detector_result: FaceDetectorResult) -> CheckResult:
        """
        Backwards-compat stub so your controller won't break.
        Presence of glasses should be determined by your YOLO 'NO_GLASSES' logic elsewhere.
        """
        return CheckResult(
            requirement=Requirement.NO_GLASSES,
            passed=True,
            severity=Severity.INFO,
            message="Glasses presence not evaluated in GlassesLogic (handled by YOLO).",
            details=None,
        )

    def run_all(
        self,
        image_bytes: Optional[bytes] = None,
        face_detector_result: Optional[FaceDetectorResult] = None,
        face_landmarker_result: Optional[FaceLandmarkerResult] = None,
        yolo_glasses_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[CheckResult]:
        checks: List[CheckResult] = []
        checks.extend(
            self.check_sunglasses(
                image_bytes=image_bytes,
                face_detector_result=face_detector_result,
                face_landmarker_result=face_landmarker_result,
                yolo_glasses_boxes=yolo_glasses_boxes,
            )
        )
        checks.extend(
            self.check_glare(
                image_bytes=image_bytes,
                face_detector_result=face_detector_result,
                face_landmarker_result=face_landmarker_result,
                yolo_glasses_boxes=yolo_glasses_boxes,
            )
        )
        return checks

    def check_sunglasses(
        self,
        image_bytes: Optional[bytes] = None,
        face_detector_result: Optional[FaceDetectorResult] = None,
        face_landmarker_result: Optional[FaceLandmarkerResult] = None,
        yolo_glasses_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[CheckResult]:
        """
        Detect tinted/dark lenses by comparing ROI brightness vs skin and checking texture.
        """
        if image_bytes is None:
            return [self._error_result(Requirement.NO_SUNGLASSES, "No image bytes provided.")]

        img_bgr = self._decode_image(image_bytes)
        if img_bgr is None:
            return [self._error_result(Requirement.NO_SUNGLASSES, "Could not decode image bytes.")]

        roi = self._get_eyes_roi(img_bgr, face_detector_result, face_landmarker_result, yolo_glasses_boxes)
        if roi is None:
            return [self._error_result(Requirement.NO_SUNGLASSES, "Could not localize eye region (ROI).")]

        x1, y1, x2, y2 = roi
        eye = img_bgr[y1:y2, x1:x2]
        if eye.size == 0:
            return [self._error_result(Requirement.NO_SUNGLASSES, "Empty eye ROI.")]

        hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
        V_eye = np.median(hsv[:, :, 2]) / 255.0

        # Sample skin patches near predefined face landmarks (if available)
        V_skin_vals: List[float] = []
        if face_landmarker_result is not None and len(face_landmarker_result.face_landmarks) > 0:
            V_skin_vals = self._sample_skin_brightness(img_bgr, face_landmarker_result)

        # Fallback: if no landmarks, use a ring around the ROI as pseudo-skin
        if not V_skin_vals:
            ring = self._roi_ring(img_bgr, roi, ring_px=10)
            if ring is not None and ring.size > 0:
                hsv_ring = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
                V_skin_vals = [np.median(hsv_ring[:, :, 2]) / 255.0]

        V_skin = float(np.median(V_skin_vals)) if V_skin_vals else 0.75  # conservative default

        # Texture score (Laplacian variance)
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_eye, cv2.CV_64F)
        texture = float(lap.var())

        # Decision
        is_dark_vs_skin = (V_eye < self.cfg.tint_darkness_ratio_k * V_skin)
        is_low_texture = (texture < self.cfg.min_eye_texture)

        passed = not (is_dark_vs_skin or is_low_texture)

        details: Dict[str, Any] = {
            "median_V_eye": round(V_eye, 4),
            "median_V_skin": round(V_skin, 4),
            "ratio_eye_to_skin": round((V_eye / V_skin) if V_skin > 1e-6 else 0.0, 4),
            "texture_var": round(texture, 2),
            "thresholds": {
                "tint_darkness_ratio_k": self.cfg.tint_darkness_ratio_k,
                "min_eye_texture": self.cfg.min_eye_texture,
            },
            "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

        message = "No tint detected." if passed else "Sunglasses / tinted lenses detected."
        return [CheckResult(
            requirement=Requirement.NO_SUNGLASSES,
            passed=passed,
            severity=Severity.INFO if passed else Severity.ERROR,
            message=message,
            details=details,
        )]

    def check_glare(
        self,
        image_bytes: Optional[bytes] = None,
        face_detector_result: Optional[FaceDetectorResult] = None,
        face_landmarker_result: Optional[FaceLandmarkerResult] = None,
        yolo_glasses_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[CheckResult]:
        """
        Detect glare/specular reflections in the lens ROI.
        """
        if image_bytes is None:
            return [self._error_result(Requirement.NO_GLASSES_REFLECTION, "No image bytes provided.")]

        img_bgr = self._decode_image(image_bytes)
        if img_bgr is None:
            return [self._error_result(Requirement.NO_GLASSES_REFLECTION, "Could not decode image bytes.")]

        roi = self._get_eyes_roi(img_bgr, face_detector_result, face_landmarker_result, yolo_glasses_boxes)
        if roi is None:
            return [self._error_result(Requirement.NO_GLASSES_REFLECTION, "Could not localize eye region (ROI).")]

        x1, y1, x2, y2 = roi
        eye = img_bgr[y1:y2, x1:x2]
        if eye.size == 0:
            return [self._error_result(Requirement.NO_GLASSES_REFLECTION, "Empty eye ROI.")]

        hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)

        # Bright near-white pixels (high V), low S (desaturated white-ish)
        V = hsv[:, :, 2].astype(np.float32) / 255.0
        S = hsv[:, :, 1].astype(np.float32) / 255.0
        bright_mask = (V > 0.90) & (S < 0.25)
        bright_mask = bright_mask.astype(np.uint8) * 255

        # Morphological opening to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
        roi_area = bright_mask.size
        bright_area = int((bright_mask > 0).sum())
        reflection_area_pct = float(bright_area) / float(max(roi_area, 1))

        largest_blob_px = 0
        max_aspect_ratio = 0.0
        max_len_px = 0.0

        # Compute simple aspect ratio and "length" for each blob
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            largest_blob_px = max(largest_blob_px, int(area))
            if w > 0 and h > 0:
                ar = max(w / h, h / w)
                max_aspect_ratio = max(max_aspect_ratio, float(ar))
                max_len_px = max(max_len_px, float(max(w, h)))

        # Iris overlap (if we have iris landmarks)
        overlaps_iris = False
        if face_landmarker_result is not None and len(face_landmarker_result.face_landmarks) > 0:
            ih, iw = img_bgr.shape[:2]
            iris_points_px = self._iris_points_pixels(face_landmarker_result, iw, ih)
            if iris_points_px:
                # Project to ROI-local coords and check any bright pixel within small radius
                for (cx, cy) in iris_points_px:
                    if x1 <= cx < x2 and y1 <= cy < y2:
                        rx = int(cx - x1)
                        ry = int(cy - y1)
                        r = 4  # 4px radius around iris center
                        y0, y1c = max(0, ry - r), min(bright_mask.shape[0], ry + r + 1)
                        x0, x1c = max(0, rx - r), min(bright_mask.shape[1], rx + r + 1)
                        patch = bright_mask[y0:y1c, x0:x1c]
                        if patch.size > 0 and (patch > 0).any():
                            overlaps_iris = True
                            break

        # Decision
        glare_by_area = (reflection_area_pct > self.cfg.glare_area_pct)
        glare_by_blob = (largest_blob_px > self.cfg.glare_max_blob_px)
        glare_by_streak = (max_aspect_ratio >= self.cfg.glare_streak_ar_min and
                           max_len_px >= self.cfg.glare_streak_min_len_px)

        if self.cfg.require_iris_overlap_for_glare:
            # Require iris overlap OR a stronger global signal (area 1.7x)
            passed = not ( (overlaps_iris and (glare_by_area or glare_by_blob or glare_by_streak)) or
                           (reflection_area_pct > (1.7 * self.cfg.glare_area_pct)) )
        else:
            passed = not (glare_by_area or glare_by_blob or glare_by_streak)

        details: Dict[str, Any] = {
            "reflection_area_pct": round(reflection_area_pct, 5),
            "largest_blob_px": int(largest_blob_px),
            "max_aspect_ratio": round(max_aspect_ratio, 3),
            "max_len_px": int(max_len_px),
            "overlaps_iris": overlaps_iris,
            "thresholds": {
                "glare_area_pct": self.cfg.glare_area_pct,
                "glare_max_blob_px": self.cfg.glare_max_blob_px,
                "glare_streak_ar_min": self.cfg.glare_streak_ar_min,
                "glare_streak_min_len_px": self.cfg.glare_streak_min_len_px,
                "require_iris_overlap_for_glare": self.cfg.require_iris_overlap_for_glare,
            },
            "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

        message = "No problematic glare detected." if passed else "Lens glare detected."
        return [CheckResult(
            requirement=Requirement.NO_GLASSES_REFLECTION,
            passed=passed,
            severity=Severity.INFO if passed else Severity.ERROR,
            message=message,
            details=details,
        )]

    # ------------------ Helpers ------------------

    def _error_result(self, req: Requirement, msg: str) -> CheckResult:
        return CheckResult(
            requirement=req,
            passed=False,
            severity=Severity.ERROR,
            message=msg,
            details=None,
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _get_eyes_roi(
        self,
        img_bgr: np.ndarray,
        face_detector_result: Optional[FaceDetectorResult],
        face_landmarker_result: Optional[FaceLandmarkerResult],
        yolo_glasses_boxes: Optional[List[Tuple[int, int, int, int]]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Priority:
          1) YOLO glasses box (if provided)
          2) FaceMesh eyes hull bbox
          3) FaceDetector face bbox (scaled to upper half)
        Always expands by cfg.expand_roi_pct and clips to image bounds.
        """
        ih, iw = img_bgr.shape[:2]

        # 1) YOLO glasses box
        if yolo_glasses_boxes:
            # If multiple, merge them (min/max)
            xs = [b[0] for b in yolo_glasses_boxes] + [b[2] for b in yolo_glasses_boxes]
            ys = [b[1] for b in yolo_glasses_boxes] + [b[3] for b in yolo_glasses_boxes]
            roi = (min(xs), min(ys), max(xs), max(ys))
            return self._pad_and_clip_roi(roi, iw, ih, self.cfg.expand_roi_pct)

        # 2) FaceMesh eyes hull
        if face_landmarker_result is not None and len(face_landmarker_result.face_landmarks) > 0:
            pts = []
            fl = face_landmarker_result.face_landmarks[0]
            for idx in (self.left_eye_idx + self.right_eye_idx):
                if 0 <= idx < len(fl):
                    x = int(fl[idx].x * iw)
                    y = int(fl[idx].y * ih)
                    pts.append((x, y))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                roi = (min(xs), min(ys), max(xs), max(ys))
                return self._pad_and_clip_roi(roi, iw, ih, self.cfg.expand_roi_pct)

        # 3) FaceDetector face bbox (upper half as proxy)
        if face_detector_result is not None and len(face_detector_result.detections) > 0:
            det = face_detector_result.detections[0]
            # MediaPipe gives relative bbox
            bbox = det.bounding_box
            x1 = max(0, int(bbox.origin_x))
            y1 = max(0, int(bbox.origin_y))
            x2 = min(iw, x1 + int(bbox.width))
            y2 = min(ih, y1 + int(bbox.height))
            # Focus on upper half (eyes zone)
            mid_y = y1 + (y2 - y1) // 2
            roi = (x1, y1, x2, mid_y)
            return self._pad_and_clip_roi(roi, iw, ih, self.cfg.expand_roi_pct)

        return None

    @staticmethod
    def _pad_and_clip_roi(roi: Tuple[int, int, int, int], iw: int, ih: int, expand_pct: float) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = roi
        w = x2 - x1
        h = y2 - y1
        dx = int(w * expand_pct)
        dy = int(h * expand_pct)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(iw, x2 + dx)
        y2 = min(ih, y2 + dy)
        return (x1, y1, x2, y2)

    def _sample_skin_brightness(self, img_bgr: np.ndarray, face_landmarker_result: FaceLandmarkerResult) -> List[float]:
        ih, iw = img_bgr.shape[:2]
        fl = face_landmarker_result.face_landmarks[0]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2].astype(np.float32) / 255.0

        vals: List[float] = []
        for idx in self.skin_idx:
            if 0 <= idx < len(fl):
                x = int(fl[idx].x * iw)
                y = int(fl[idx].y * ih)
                # small patch around the point
                r = 6
                y0, y1 = max(0, y - r), min(ih, y + r + 1)
                x0, x1 = max(0, x - r), min(iw, x + r + 1)
                patch = V[y0:y1, x0:x1]
                if patch.size > 0:
                    vals.append(float(np.median(patch)))
        return vals

    def _roi_ring(self, img_bgr: np.ndarray, roi: Tuple[int, int, int, int], ring_px: int = 10) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = roi
        ih, iw = img_bgr.shape[:2]

        # grow rectangle by ring_px and subtract inner ROI to make a ring
        gx1 = max(0, x1 - ring_px); gy1 = max(0, y1 - ring_px)
        gx2 = min(iw, x2 + ring_px); gy2 = min(ih, y2 + ring_px)

        outer = img_bgr[gy1:gy2, gx1:gx2]
        if outer.size == 0:
            return None

        mask = np.ones((gy2 - gy1, gx2 - gx1), dtype=np.uint8) * 255
        mask[(y1 - gy1):(y2 - gy1), (x1 - gx1):(x2 - gx1)] = 0

        # apply mask
        ring = cv2.bitwise_and(outer, outer, mask=mask)
        return ring

    def _iris_points_pixels(self, face_landmarker_result: FaceLandmarkerResult, iw: int, ih: int) -> List[Tuple[int, int]]:
        fl = face_landmarker_result.face_landmarks[0]
        pts: List[Tuple[int, int]] = []

        for idx in [self.right_iris_center_idx, self.left_iris_center_idx]:
            if 0 <= idx < len(fl):
                x = int(fl[idx].x * iw)
                y = int(fl[idx].y * ih)
                pts.append((x, y))
        return pts