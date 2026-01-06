import pytest
import math


# Importér de nødvendige funktioner og klasser fra koden
# Da vi ikke har de faktiske MediaPipe-klasser, skal vi mocke dem eller
# sikre, at de nødvendige dele er tilgængelige.
# I dette eksempel antager vi, at 'logic.types' er tilgængelig,
# og vi inkluderer 'get_yaw_pitch' og 'FaceLookingAtCamera' her for at gøre testen selvkørende.

# --- Simulerer nødvendige imports og definitioner ---
# Bemærk: I en reel testfil ville du importere disse fra dine kildemoduler.

# Mocking af en del af 'logic.types' for at køre testen
class Requirement:
    FACE_LOOKING_AT_CAMERA = "FACE_LOOKING_AT_CAMERA"


class Severity:
    INFO = "INFO"
    ERROR = "ERROR"


class CheckResult:
    def __init__(self, requirement, passed, severity, message):
        self.requirement = requirement
        self.passed = passed
        self.severity = severity
        self.message = message

    def __eq__(self, other):
        """Hjælper med at sammenligne CheckResult-objekter i tests."""
        if not isinstance(other, CheckResult):
            return NotImplemented
        return (self.requirement == other.requirement and
                self.passed == other.passed and
                self.severity == other.severity and
                self.message == other.message)

    def __repr__(self):
        return f"CheckResult(passed={self.passed}, severity='{self.severity}', message='{self.message}')"


class MockLandmark:
    """Mock-klasse for MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockFaceLandmarkerResult:
    """Mock-klasse for MediaPipe FaceLandmarkerResult."""

    def __init__(self, landmarks=None):
        # 'face_landmarks' er en liste af lister af NormalizedLandmark
        self.face_landmarks = [landmarks] if landmarks is not None else []


# --- get_yaw_pitch-funktion (kopieret fra din kode) ---
def get_yaw_pitch(face_result: MockFaceLandmarkerResult) -> tuple[float, float]:
    """
    Computes head yaw & pitch (in degrees) using MediaPipe 3D landmarks,
    normalized so forward = 0°.
    """
    if not face_result.face_landmarks or len(face_result.face_landmarks) == 0:
        raise ValueError("No landmarks found.")

    landmarks = face_result.face_landmarks[0]

    LEFT_EAR = 234
    RIGHT_EAR = 454
    CHIN = 152
    FOREHEAD = 10

    # Vi bruger MockLandmark i stedet for at indeksere
    left_ear = landmarks[LEFT_EAR]
    right_ear = landmarks[RIGHT_EAR]
    chin = landmarks[CHIN]
    forehead = landmarks[FOREHEAD]

    # --- YAW ---
    ear_dx = left_ear.x - right_ear.x
    ear_dz = left_ear.z - right_ear.z
    yaw = math.degrees(math.atan2(ear_dz, ear_dx))

    # --- PITCH ---
    dy = forehead.y - chin.y
    dz = forehead.z - chin.z
    pitch = math.degrees(math.atan2(dz, dy))

    # --- NORMALIZE angles to [-90°, +90°] ---
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180

    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return yaw, pitch


# --- FaceLookingAtCamera-klasse (kopieret fra din kode) ---
class FaceLookingAtCamera:
    def __init__(self, tolerance_yaw_degrees=7, tolerance_pitch_degrees=15):
        self.yaw_tolerance = tolerance_yaw_degrees
        self.pitch_tolerance = tolerance_pitch_degrees

    def face_detector(self, result: MockFaceLandmarkerResult) -> CheckResult:

        try:
            # Vi bruger mock-resultatet her
            yaw, pitch = get_yaw_pitch(result)
        except Exception as e:
            return CheckResult(
                requirement=Requirement.FACE_LOOKING_AT_CAMERA,
                passed=False,
                severity=Severity.ERROR,
                message=f"Failed to compute head rotation: {str(e)}"
            )

        if abs(yaw) <= self.yaw_tolerance and abs(pitch) <= self.pitch_tolerance:
            return CheckResult(
                requirement=Requirement.FACE_LOOKING_AT_CAMERA,
                passed=True,
                severity=Severity.INFO,
                message=f"Looking straight at camera"
            )
        else:
            return CheckResult(
                requirement=Requirement.FACE_LOOKING_AT_CAMERA,
                passed=False,
                severity=Severity.ERROR,
                message=f"Not looking straight"
            )


# --- PYTEST TESTS ---

# Opret et fixture for FaceLookingAtCamera med standard tolerance (10 grader)
@pytest.fixture
def face_checker_default():
    return FaceLookingAtCamera(tolerance_yaw_degrees=7, tolerance_pitch_degrees=15)


# Opret et fixture for FaceLookingAtCamera med en strammere tolerance
@pytest.fixture
def face_checker_tight():
    return FaceLookingAtCamera(tolerance_yaw_degrees=5, tolerance_pitch_degrees=10)


# Hjælpefunktion til at oprette et mock-resultat for "straight" ansigt
def create_straight_face_result():
    # Definerer de nødvendige 4 punkter (indekser: 234, 454, 152, 10)
    # Forward: ear_dx ≈ 0, ear_dz ≈ 0. pitch: dy > 0, dz ≈ 0.
    landmarks = [None] * 468  # Simulerer en liste af 468 landmarks

    # LEFT_EAR (234)
    landmarks[234] = MockLandmark(x=0.1, y=0.5, z=0.0)
    # RIGHT_EAR (454)
    landmarks[454] = MockLandmark(x=-0.1, y=0.5, z=0.0)
    # CHIN (152)
    landmarks[152] = MockLandmark(x=0.0, y=0.3, z=0.0)
    # FOREHEAD (10)
    landmarks[10] = MockLandmark(x=0.0, y=0.7, z=0.0)

    return MockFaceLandmarkerResult(landmarks)


# Hjælpefunktion til at oprette et mock-resultat for "yawed" ansigt
def create_yawed_face_result(angle_degrees):
    # Simulerer et hoved drejet om Y-aksen (yaw)
    landmarks = [None] * 468
    angle_rad = math.radians(angle_degrees)

    # Vi simulerer, at ørerne er i XY-plan, men roterer Z-koordinaten
    # Left ear: (r*sin(theta), y, r*cos(theta))
    # Right ear: (-r*sin(theta), y, r*cos(theta))
    # Da get_yaw_pitch bruger atan2(dz, dx) for ørerne, skal vi justere dx og dz

    # Simpel tilgang: Antag, at ørerne definerer en linje
    # Normalt ansigt: dx = 0.2, dz = 0.0 -> atan2(0, 0.2) = 0°

    # Roteret ansigt: Vi ønsker, at atan2(dz, dx) skal give 'angle_degrees'
    # dz = sin(angle_rad), dx = cos(angle_rad)

    # Skal justere dx og dz, så (dz / dx) ≈ tan(vinkel)

    # For nemhedens skyld antager vi bare Z-forskydning:
    # YAW: ear_dx = left_ear.x - right_ear.x, ear_dz = left_ear.z - right_ear.z
    # Hvis ansigtet er roteret -15 grader (til venstre), skal left_ear.z > right_ear.z
    # Hvis ansigtet er roteret +15 grader (til højre), skal left_ear.z < right_ear.z

    r = 0.1  # Radius fra center til øre

    # Ører (ca. center: x=0, z=0)
    landmarks[234] = MockLandmark(
        x=r * math.cos(angle_rad),
        y=0.5,
        z=r * math.sin(angle_rad)
    )
    landmarks[454] = MockLandmark(
        x=-r * math.cos(angle_rad),
        y=0.5,
        z=-r * math.sin(angle_rad)
    )

    # For at undgå division med nul og for at sikre en ren vinkel
    # yaw = math.degrees(math.atan2(ear_dz, ear_dx))
    # dx = 2*r*cos(angle_rad), dz = 2*r*sin(angle_rad)
    # atan2(dz, dx) = atan2(sin(angle), cos(angle)) = angle

    # Næse/hage (ingen pitch rotation)
    landmarks[152] = MockLandmark(x=0.0, y=0.3, z=0.0)  # CHIN
    landmarks[10] = MockLandmark(x=0.0, y=0.7, z=0.0)  # FOREHEAD

    return MockFaceLandmarkerResult(landmarks)


# Hjælpefunktion til at oprette et mock-resultat for "pitched" ansigt
def create_pitched_face_result(angle_degrees):
    # Simulerer et hoved drejet om X-aksen (pitch)
    landmarks = [None] * 468
    angle_rad = math.radians(angle_degrees)

    # PITCH: dy = forehead.y - chin.y, dz = forehead.z - chin.z
    # pitch = math.degrees(math.atan2(dz, dy))

    h = 0.2  # Halv højde

    # Kin og pande (ca. center: y=0.5, z=0)
    # Y-koordinaten er i billedplanet (op/ned), Z er dybde (frem/tilbage)

    landmarks[10] = MockLandmark(
        x=0.0,
        y=0.5 + h * math.cos(angle_rad),  # FOREHEAD Y
        z=h * math.sin(angle_rad)  # FOREHEAD Z
    )
    landmarks[152] = MockLandmark(
        x=0.0,
        y=0.5 - h * math.cos(angle_rad),  # CHIN Y
        z=-h * math.sin(angle_rad)  # CHIN Z
    )

    # Dy = 2*h*cos(angle_rad), Dz = 2*h*sin(angle_rad)
    # atan2(dz, dy) = atan2(sin(angle), cos(angle)) = angle

    # Ører (ingen yaw rotation)
    landmarks[234] = MockLandmark(x=0.1, y=0.5, z=0.0)  # LEFT_EAR
    landmarks[454] = MockLandmark(x=-0.1, y=0.5, z=0.0)  # RIGHT_EAR

    return MockFaceLandmarkerResult(landmarks)


### TESTS for get_yaw_pitch (for at sikre, at vinkelberegningen er korrekt)
@pytest.mark.parametrize("angle", [-15, 0, 15, 45, -45])
def test_get_yaw_pitch_yaw_correct(angle):
    result = create_yawed_face_result(angle)
    yaw, pitch = get_yaw_pitch(result)
    assert math.isclose(yaw, angle, abs_tol=0.1)
    assert math.isclose(pitch, 0.0, abs_tol=0.1)


@pytest.mark.parametrize("angle", [-15, 0, 15, 45, -45])
def test_get_yaw_pitch_pitch_correct(angle):
    result = create_pitched_face_result(angle)
    yaw, pitch = get_yaw_pitch(result)
    assert math.isclose(pitch, angle, abs_tol=0.1)
    assert math.isclose(yaw, 0.0, abs_tol=0.1)


def test_get_yaw_pitch_no_landmarks():
    result = MockFaceLandmarkerResult(landmarks=None)
    with pytest.raises(ValueError, match="No landmarks found."):
        get_yaw_pitch(result)


### TESTS for FaceLookingAtCamera.face_detector

# test: Ansigt ser lige frem (bør PASS)
def test_face_looking_at_camera_pass_straight(face_checker_default):
    result = create_straight_face_result()
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=True,
        severity=Severity.INFO,
        message="Looking straight at camera"
    )
    assert face_checker_default.face_detector(result) == expected


# test: Ansigt er tæt på grænsen (bør PASS)
def test_face_looking_at_camera_pass_at_tolerance_limit(face_checker_default):
    # YAW = 7 grader (yaw-tolerance=7) -> PASS
    result = create_yawed_face_result(7.0)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=True,
        severity=Severity.INFO,
        message="Looking straight at camera"
    )
    assert face_checker_default.face_detector(result) == expected



# test: Ansigt er lige over grænsen (bør FAIL)
def test_face_looking_at_camera_fail_just_over_tolerance(face_checker_default):
    # YAW = 7.1 grader (yaw-tolerance=7) -> FAIL
    result = create_yawed_face_result(7.1)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=False,
        severity=Severity.ERROR,
        message="Not looking straight"
    )
    assert face_checker_default.face_detector(result) == expected



# test: Ansigt med stor rotation i PITCH (bør FAIL)
def test_face_looking_at_camera_fail_high_pitch(face_checker_default):
    # PITCH = 20 grader (pitch-tolerance=15) -> FAIL
    result = create_pitched_face_result(20.0)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=False,
        severity=Severity.ERROR,
        message="Not looking straight"
    )
    assert face_checker_default.face_detector(result) == expected


def test_face_looking_at_camera_fail_just_over_pitch_tolerance(face_checker_default):
    # PITCH = 15.1 grader (pitch-tolerance=15) -> FAIL
    result = create_pitched_face_result(15.1)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=False,
        severity=Severity.ERROR,
        message="Not looking straight"
    )
    assert face_checker_default.face_detector(result) == expected


# test: Ansigt med rotation i begge akser, men inden for grænsen (bør PASS)
def test_face_looking_at_camera_pass_both_axes(face_checker_default):
    yaw_res = create_yawed_face_result(6.0)     # indenfor yaw<=7
    pitch_res = create_pitched_face_result(10.0) # indenfor pitch<=15

    # Kombinér landmarks: brug ører fra yaw og pande/hage fra pitch
    landmarks = [None] * 468
    landmarks[234] = yaw_res.face_landmarks[0][234]
    landmarks[454] = yaw_res.face_landmarks[0][454]
    landmarks[10]  = pitch_res.face_landmarks[0][10]
    landmarks[152] = pitch_res.face_landmarks[0][152]

    result = MockFaceLandmarkerResult(landmarks)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=True,
        severity=Severity.INFO,
        message="Looking straight at camera"
    )
    assert face_checker_default.face_detector(result) == expected



# test: test med strammere tolerance (5 grader)
def test_face_looking_at_camera_fail_tight_tolerance(face_checker_tight):
    # test med YAW = 6 grader (tolerance_degrees=5)
    result = create_yawed_face_result(6.0)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=False,
        severity=Severity.ERROR,
        message="Not looking straight"
    )
    assert face_checker_tight.face_detector(result) == expected


# test: Håndtering af manglende landmarks (ValueError)
def test_face_looking_at_camera_no_landmarks_error(face_checker_default):
    result = MockFaceLandmarkerResult(landmarks=None)
    expected = CheckResult(
        requirement=Requirement.FACE_LOOKING_AT_CAMERA,
        passed=False,
        severity=Severity.ERROR,
        message="Failed to compute head rotation: No landmarks found."
    )
    assert face_checker_default.face_detector(result).passed == expected.passed
    assert face_checker_default.face_detector(result).severity == expected.severity
    assert "No landmarks found" in face_checker_default.face_detector(result).message