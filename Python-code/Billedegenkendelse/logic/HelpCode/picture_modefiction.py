import cv2
import numpy as np
import logic.types as types

class picture_modefiction:

def __init__(self):

#Exposure / Pixelation
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

#Exposure
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

#Exposure
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

#Exposure
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