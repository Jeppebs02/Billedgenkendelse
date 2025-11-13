import cv2
import numpy as np
from logic.types import CheckResult, Requirement, Severity


class exposure_check(picture_modefication):
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
        self.thr_side_diff_pct = 0.30  # Maks forskel i lysstyrke mellem venstre/højre side (22 %)
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

        # Overexposed → many very bright pixels + high p95
        if (p95 >= self.thr_bright_p95 and clip_hi_ratio >= self.thr_clip_hi):
            return self._fail("Overexposed/burned out in the face.", details)

        # Underexposed → very dark and low p05
        if (p05 <= self.thr_dark_p05 and clip_lo_ratio >= self.thr_clip_lo):
            return self._fail("Too dark in the face.", details)

        # Low contrast → both low dynamic range and low standard deviation
        if dynamic_range < self.thr_dynamic_range and stdL < self.thr_std:
            return self._fail("Too low contrast/dynamic range in the face.", details)

        # Uneven lighting → large difference between left and right sides
        if side_diff_pct >= self.thr_side_diff_pct or side_ratio >= self.thr_side_ratio:
            return self._fail("Uneven lighting between the two halves of the face.", details)

        # Adaptiv p50-vurdering baseret på hud, hvis tilgængelig; ellers faste grænser
        if dynamic_range < 35:
            if use_adaptive:
                if p50 < LB_adapt:
                    return self._fail("Overall too low facial luminance compared to skin level (adaptive).", details)
                if p50 > UB_adapt:
                    return self._fail("Overall too high facial luminance compared to skin level (adaptive).", details)
            else:
                if p50 <= self.thr_p50_dark:
                    return self._fail("Overall too low facial luminance.", details)
                if p50 >= self.thr_p50_bright:
                    return self._fail("Overall too high facial luminance.", details)

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

