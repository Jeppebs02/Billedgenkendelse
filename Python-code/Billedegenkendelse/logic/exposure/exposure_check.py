import cv2
import numpy as np
from logic.types import CheckResult, Requirement, Severity


class exposure_check:
    """
    Vurderer eksponering og belysning i et ansigt ud fra et billede (BGR)
    og MediaPipe-landmarks for ansigtet.

    Formålet er at afgøre, om ansigtsbelysningen er egnet til f.eks. kørekortbilleder:
    - Er billedet for lyst (overeksponeret)?
    - Er billedet for mørkt (underbelyst)?
    - Er kontrasten for lav?
    - Er lyset jævnt fordelt mellem venstre og højre side af ansigtet?

    Returnerer et CheckResult-objekt, som indeholder status og detaljerede målinger.
    """

    def __init__(self):
        # Tærskelværdier (kan justeres, afhængigt af dataset og ønsket følsomhed)
        self.thr_clip_hi = 0.06  # Hvor mange pixels må være helt hvide (maks lys)
        self.thr_clip_lo = 0.12  # Hvor mange pixels må være helt sorte (maks mørke)
        self.thr_dark_p05 = 8  # 5-percentilen skal mindst være over 8 (ellers for mørkt)
        self.thr_bright_p95 = 252  # 95-percentilen skal være under 252 (ellers for lyst)
        self.thr_dynamic_range = 28  # Minimum forskel mellem mørke og lyse områder
        self.thr_std = 16  # Minimum standardafvigelse i lysfordeling (for kontrast)
        self.thr_side_diff_pct = 0.22  # Maks forskel i lysstyrke mellem venstre/højre side (22 %)
        self.thr_side_ratio = 1.45  # Maks ratio mellem lys på de to sider (1.45 ≈ 45 % forskel)
        self.thr_p50_dark = 60  # Median under 60 → generelt for mørkt
        self.thr_p50_bright = 190  # Median over 190 → generelt for lyst
        self.min_mask_pixels = 800  # Minimum antal pixels i ansigtsmasken for gyldig analyse

    def analyze(self, image_bytes: bytes, landmarker_result) -> CheckResult:
        """
        Hovedfunktion til lysanalyse.

        Input:
            image_bytes: bytes — image file bytes
            landmarker_result: FaceLandmarkerResult object for a single face

        Output:
            CheckResult for Requirement.LIGHTING_OK
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Konverter til L-kanal (lysstyrke) fra LAB-farverum
        L = self._luminance_lab(bgr)

        # Opret en binær maske ud fra ansigtets landmarks
        face_mask = self._landmarks_to_mask(bgr.shape, landmarker_result)

        # Rens masken (fjerner støj og hår/baggrund)
        face_mask = self._refine_face_mask(face_mask)

        # Tjek om masken indeholder nok pixels til en meningsfuld analyse
        if int(face_mask.sum()) < self.min_mask_pixels:
            return CheckResult(
                requirement=Requirement.LIGHTING_OK,
                passed=False,
                severity=Severity.ERROR,
                message="Ingen eller usikker ansigtsmaske til lysvurdering.",
                details={"mask_pixels": int(face_mask.sum())}
            )

        # Let udglatning (fjerner små refleksioner og sensorstøj)
        L_smooth = cv2.GaussianBlur(L, (5, 5), 0)

        # Ekstraher kun lysværdier inden for ansigtsmasken
        L_face = L_smooth[face_mask == 1]

        # --- ADAPTIV HUD-BASERET P50-VINDUE ---
        skin_mask = self._skin_mask_lab(bgr, face_mask)
        use_adaptive = False
        LB_adapt = None
        UB_adapt = None

        if int(skin_mask.sum()) >= 800:  # kræv nok hud-pixels
            L_skin = L_smooth[skin_mask == 1]
            L_skin_mid = self._robust_middle_band(L_skin, 10, 90)  # strammere cut, mindre outliers
            if L_skin_mid.size >= 300:
                L_skin_p50 = float(np.median(L_skin_mid))
                # Evt. brug IQR til at sætte delta dynamisk
                q25 = float(np.percentile(L_skin_mid, 25))
                q75 = float(np.percentile(L_skin_mid, 75))
                iqr = max(1.0, q75 - q25)

                # Delta: enten fast fx 35, eller en funktion af IQR (mere robust på tværs af hudtoner/lys)
                delta = max(28.0, min(45.0, 0.9 * iqr + 25.0))

                LB_adapt = max(30.0, L_skin_p50 - delta)
                UB_adapt = min(230.0, L_skin_p50 + delta)
                use_adaptive = True

        # Beregn histogram-metrikker for ansigtets lysfordeling
        p05 = float(np.percentile(L_face, 5))  # Mørkeste områder
        p50 = float(np.percentile(L_face, 50))  # Median-lysniveau
        p95 = float(np.percentile(L_face, 95))  # Lyseste områder
        stdL = float(np.std(L_face))  # Kontrast / variation
        clip_hi_ratio = float(np.count_nonzero(L_face >= 252)) / L_face.size
        clip_lo_ratio = float(np.count_nonzero(L_face <= 3)) / L_face.size
        dynamic_range = float(p95 - p05)  # Lys-spænd (kontrastområde)

        # Del ansigtet op i venstre og højre halvdel for at vurdere ensartet belysning
        ys, xs = np.where(face_mask == 1)
        x_mid = (int(xs.min()) + int(xs.max())) // 2

        left_mask = np.zeros_like(face_mask, np.uint8)
        right_mask = np.zeros_like(face_mask, np.uint8)
        left_mask[:, :x_mid] = 1
        right_mask[:, x_mid:] = 1
        left_mask &= face_mask
        right_mask &= face_mask

        # Beregn median-lysniveau for hver side
        L_left = L_smooth[left_mask == 1]
        L_right = L_smooth[right_mask == 1]
        med_left = float(np.median(L_left)) if L_left.size > 0 else p50
        med_right = float(np.median(L_right)) if L_right.size > 0 else p50

        # Forskelle i lysstyrke mellem siderne
        side_diff_abs = abs(med_left - med_right)
        side_diff_pct = side_diff_abs / max(1.0, p50)
        side_ratio = max(med_left, med_right) / max(1.0, min(med_left, med_right))

        # Saml alle målinger i et dictionary til rapportering
        details = {
            "p05": p05, "p50": p50, "p95": p95, "stdL": stdL,
            "clip_hi_ratio": clip_hi_ratio, "clip_lo_ratio": clip_lo_ratio,
            "dynamic_range": dynamic_range,
            "side_diff_abs": side_diff_abs, "side_diff_pct": side_diff_pct,
            "side_ratio": side_ratio,
            "mask_pixels": int(face_mask.sum())
        }

        # Gem i details for sporbarhed
        details.update({
            "adaptive_used": bool(use_adaptive),
            "adaptive_LB": LB_adapt,
            "adaptive_UB": UB_adapt,
        })

        # ---- Evaluering mod tærskelværdier ----

        # Overeksponeret → mange meget lyse pixels + høj p95
        if (p95 >= self.thr_bright_p95 and clip_hi_ratio >= self.thr_clip_hi):
            return self._fail("Overeksponeret/udbrændt i ansigtet.", details)

        # Under-eksponeret → meget mørkt og lav p05
        if (p05 <= self.thr_dark_p05 and clip_lo_ratio >= self.thr_clip_lo):
            return self._fail("For mørkt i ansigtet.", details)

        # Lav kontrast → både lille dynamik og lav standardafvigelse
        if dynamic_range < self.thr_dynamic_range and stdL < self.thr_std:
            return self._fail("For lav kontrast/dynamik i ansigtet.", details)

        # Ujævn belysning → stor forskel mellem venstre og højre side
        if side_diff_pct >= self.thr_side_diff_pct or side_ratio >= self.thr_side_ratio:
            return self._fail("Ujævn belysning mellem ansigtshalvdele.", details)

        # Adaptiv p50-vurdering baseret på hud, hvis tilgængelig; ellers faste grænser
        if dynamic_range < 35:
            if use_adaptive:
                if p50 < LB_adapt:
                    return self._fail("Generelt for lav ansigtsluminans ift. hudniveau (adaptiv).", details)
                if p50 > UB_adapt:
                    return self._fail("Generelt for høj ansigtsluminans ift. hudniveau (adaptiv).", details)
            else:
                if p50 <= self.thr_p50_dark:
                    return self._fail("Generelt for lav ansigtsluminans.", details)
                if p50 >= self.thr_p50_bright:
                    return self._fail("Generelt for høj ansigtsluminans.", details)

        # Hvis ingen problemer blev fundet
        return CheckResult(
            requirement=Requirement.LIGHTING_OK,
            passed=True,
            severity=Severity.INFO,
            message="Eksponering OK.",
            details=details
        )

    # Hjælpefunktioner

    def _fail(self, msg: str, details: dict) -> CheckResult:
        """Returnér et fejlresultat med besked og detaljer."""
        return CheckResult(
            requirement=Requirement.LIGHTING_OK,
            passed=False,
            severity=Severity.ERROR,
            message=msg,
            details=details
        )

    def _luminance_lab(self, bgr: np.ndarray) -> np.ndarray:
        """
        Konverterer BGR-billede til LAB-farverum og returnerer L-kanalen (lyshed).
        LAB er mere robust mod farvevariationer end f.eks. RGB.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    def _landmarks_to_mask(self, img_shape, landmarker_result) -> np.ndarray:
        """
        Opretter en binær ansigtsmaske (0/1) ud fra ansigtslandmarks.
        Bruges til kun at analysere lys inden for ansigtets kontur.
        """
        h, w = img_shape[:2]
        if not landmarker_result or not landmarker_result.face_landmarks:
            return np.zeros((h, w), np.uint8)

        # Use landmarks from the first detected face
        landmarks = landmarker_result.face_landmarks[0]

        if len(landmarks) < 3:
            return np.zeros((h, w), np.uint8)

        # Konverter landmarks til pixelkoordinater
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in landmarks], np.int32)

        # Brug konveks-hull til at dække hele ansigtet (ingen huller)
        hull = cv2.convexHull(pts)

        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask

    def _refine_face_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Forbedrer masken:
        - Eroderer (trækker den lidt sammen) for at fjerne hårkanter.
        - Morfologisk 'åbning' for at fjerne små støjområder.
        """
        k = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    def _skin_mask_lab(self, bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """
        Grov huddetektor i LAB inde i ansigtsmasken.
        OpenCV-LAB har a*, b* ca. center 128. Vi bruger brede vinduer og forventer kalibrering på egne data.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Brede hudvinduer (skal kalibreres på dit kamera/lys)
        # a* (rød-grøn): lidt over midten → 130..175
        # b* (gul-blå): let gulligt → 120..185
        a_lo, a_hi = 130, 175
        b_lo, b_hi = 120, 185

        skin = (
                (A >= a_lo) & (A <= a_hi) &
                (B >= b_lo) & (B <= b_hi)
        ).astype(np.uint8)

        # Kun inde i ansigtet
        skin &= face_mask.astype(np.uint8)

        # Fjern små pletter
        k = np.ones((3, 3), np.uint8)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k, iterations=1)
        return skin

    def _robust_middle_band(self, arr: np.ndarray, low=5, high=95):
        """
        Skær top/bund fra (percentiler) for at undgå skygge/højlys outliers.
        Returner de værdier der ligger mellem low- og high-percentilen.
        """
        if arr.size == 0:
            return arr
        lo = np.percentile(arr, low)
        hi = np.percentile(arr, high)
        mid = arr[(arr >= lo) & (arr <= hi)]
        return mid