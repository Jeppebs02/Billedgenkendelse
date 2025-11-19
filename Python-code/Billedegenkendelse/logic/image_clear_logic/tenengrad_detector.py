class tenengrad_detector(picture_modefication):
    """
    Checks sharpness using Tenengrad (Sobel-gradient baseret).
    Fokus: ansigt hvis muligt, ellers hele billedet.
    """

    def __init__(
        self,
        face_threshold: float = 300.0,
        min_face_pixels: int = 800,
    ):
        self.face_threshold = float(face_threshold)
        self.min_face_pixels = int(min_face_pixels)
        self.full_image_threshold = float(face_threshold)

    def _tenengrad(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        g2 = gx ** 2 + gy ** 2

        if mask is not None:
            m = mask.astype(bool)
            values = g2[m]
        else:
            values = g2

        if values.size == 0:
            return 0.0

        return float(values.mean())

    def analyze_bytes(
        self,
        image_bytes: bytes,
        landmarker_result=None,
    ) -> CheckResult:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image from bytes.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        face_mask = None
        face_pixels = 0
        used_face_mask = False

        if landmarker_result is not None and getattr(landmarker_result, "face_landmarks", None):
            raw_mask = self._landmarks_to_mask(bgr.shape, landmarker_result)
            refined_mask = self._refine_face_mask(raw_mask)
            face_pixels = int(refined_mask.sum())
            if face_pixels >= self.min_face_pixels:
                face_mask = refined_mask
                used_face_mask = True

        tenengrad_full = self._tenengrad(rgb)
        tenengrad_face = None
        if face_mask is not None:
            tenengrad_face = self._tenengrad(rgb, mask=face_mask)

        if face_mask is None:
            value = tenengrad_full
            threshold = self.full_image_threshold
            region = "whole image (no reliable face mask)"
        else:
            value = tenengrad_face
            threshold = self.face_threshold
            region = "face region"

        passed = bool(value >= threshold)

        if passed:
            message = f"Tenengrad sharpness is sufficient in {region}."
        else:
            message = f"Tenengrad sharpness is too low in {region}."

        details = {
            "tenengrad_full": tenengrad_full,
            "tenengrad_face": tenengrad_face,
            "face_threshold": self.face_threshold,
            "full_image_threshold": self.full_image_threshold,
            "used_face_mask": bool(used_face_mask),
            "face_pixels": face_pixels,
        }

        return CheckResult(
            requirement=Requirement.IMAGE_TENENGRAD_SHARPNESS,
            passed=passed,
            severity=Severity.ERROR if not passed else Severity.INFO,
            message=message,
            details=details,
        )

    def analyze_image(self, image_path: str) -> CheckResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tenengrad_full = self._tenengrad(rgb)

        passed = bool(tenengrad_full >= self.full_image_threshold)

        message = (
            "Tenengrad sharpness is sufficient."
            if passed
            else "Tenengrad sharpness is too low."
        )

        details = {
            "tenengrad_full": tenengrad_full,
            "tenengrad_face": None,
            "face_threshold": self.face_threshold,
            "full_image_threshold": self.full_image_threshold,
            "used_face_mask": False,
            "face_pixels": 0,
        }

        return CheckResult(
            requirement=Requirement.IMAGE_TENENGRAD_SHARPNESS,
            passed=passed,
            severity=Severity.ERROR if not passed else Severity.INFO,
            message=message,
            details=details,
        )
