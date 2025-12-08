from utils.visualizer import VisualizerHelper
from utils.types import AnalysisReport, Requirement

from .glare_sunglasses_logic.glare_sunglasses_check import GlassesLogic
from .head_placement_logic.head_centering_check import HeadCenteringValidator
from .face_present_logic.face_detector import DetectionVisualizer
from .hat_glasses_logic.hat_glasses_detector import HatGlassesDetector
from .exposure_logic.exposure_check import exposure_check
from .image_clear_logic.image_clear_check import ImageClearCheck
from .face_direction_logic.face_looking_at_camera_check import FaceLookingAtCamera
from .pixelation_logic.pixelation_check import PixelationCheck


class LogicController:
    def __init__(self):
        self.face_detector = DetectionVisualizer()
        self.hat_glasses_detector = HatGlassesDetector()
        self.glasses_logic = GlassesLogic()
        self.exposure_check = exposure_check()
        self.image_clear_check = ImageClearCheck()
        self.face_looking_at_camera = FaceLookingAtCamera()
        self.head_centering_validator = HeadCenteringValidator()
        self.visualizer_helper = None

        # Ekstra pixelation-check (blockiness osv.)
        self.pixelation_check = PixelationCheck()

    # Utility functions

    def find_check(self, checks, req: Requirement):
        for c in checks:
            if c.requirement == req:
                return c
        return None

    def run_analysis(self, image_path: str, image_name, threshold: float = 0.5) -> AnalysisReport:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        self.visualizer_helper = VisualizerHelper(image_name)

        face_detector_result = self.face_detector.analyze_bytes(image_bytes)
        face_landmarker_result = self.face_detector.analyze_landmarks_bytes(image_bytes)

        checks = []

        # 1) Face present
        checks.append(self.face_detector.is_face_in_image(face_detector_result))

        # 2) Single face
        checks.append(self.face_detector.is_single_face(face_detector_result))

        # 3) Landmarks present
        checks.append(self.face_detector.are_landmarks_present(face_landmarker_result))

        # 4) Eyes visible
        checks.append(self.face_detector.eyes_visible_check(face_landmarker_result))

        # 5) Mouth closed
        checks.append(self.face_detector.mouth_closed_check(face_landmarker_result))

        # 6) No hat and no glasses
        checks.extend(
            self.hat_glasses_detector.check_hats_and_glasses_bytes(
                image_bytes,
                threshold=threshold
            )
        )

        no_glasses_check = self.find_check(checks, Requirement.NO_GLASSES)

        det = False
        if no_glasses_check and no_glasses_check.details:
            det = no_glasses_check.details.get("detected", False)

        if det:
            # 7) Glare and sunglasses check
            checks.extend(
                self.glasses_logic.run_all(
                    image_bytes=image_bytes,
                    face_detector_result=face_detector_result,
                    face_landmarker_result=face_landmarker_result
                )
            )

        # 8) image clear check (new collective check: face sharpness + background)
        checks.append(
            self.image_clear_check.analyze_bytes(
            image_bytes=image_bytes,
            landmarker_result=face_landmarker_result,
            )
        )

        # 8.5) pixelation check (separat requirement, f.eks. blockiness / compression)
        #checks.append(
            #self.pixelation_check.check_pixelation_bytes(
                #image_bytes,
                #requirement=Requirement.PIXELATION
            #)
        #)

        # 9) exposure_logic / lighting check
        exposure_check_result = self.exposure_check.analyze(
            image_bytes,
            face_landmarker_result
        )
        checks.append(exposure_check_result)

        # 10) face looking straight
        checks.append(
            self.face_looking_at_camera.face_detector(result=face_landmarker_result)
        )

        # 11) face centered
        checks.append(
            self.head_centering_validator.check_from_detection_bytes(
                det_res=face_detector_result,
                image_bytes=image_bytes
            )
        )

        # --- Visualiseringer ---
        self.visualizer_helper.annotate_facedetector(
            image_bytes,
            face_detector_result
        )
        self.visualizer_helper.annotate_landmarks(
            image_bytes,
            face_landmarker_result
        )
        self.visualizer_helper.annotate_center_and_size(
            image_bytes,
            face_detector_result,
            self.head_centering_validator.cfg
        )
        self.visualizer_helper.annotate_looking_straight(
            image_bytes,
            face_landmarker_result,
            self.face_looking_at_camera.yaw_tolerance,
            self.face_looking_at_camera.pitch_tolerance,
        )
        self.visualizer_helper.visualize_exposure(
            image_bytes,
            exposure_result=exposure_check_result,
            checker_instance=self.exposure_check,
            detection_result=face_landmarker_result
        )
        self.visualizer_helper.visualize_mouth(
            image_bytes,
            face_landmarker_result,
        )



        self.visualizer_helper.visualize_pixelation(image_bytes, detection_result=face_detector_result)

        # Her bruger du stadig din eksisterende pixelations-visualisering
        self.visualizer_helper.visualize_pixelation(
            image_bytes,
            detection_result=face_detector_result
        )


        self.visualizer_helper.annotate_eyes_visible(
            image_bytes=image_bytes,
            landmark_result=face_landmarker_result,
            ear_threshold=0.2,  # eller hent den fra et config hvis du gør det dynamisk
        )

        sharpness_result = self.image_clear_check.analyze_bytes(
            image_bytes=image_bytes,
            landmarker_result=face_landmarker_result
        )

        self.visualizer_helper.visualize_sharpness(
            image_bytes=image_bytes, sharpness_result=sharpness_result)


        overall_pass = all(c.passed for c in checks)
        return AnalysisReport(
            image="<bytes>",
            passed=overall_pass,
            checks=checks
        )

    def run_analysis_bytes(self, image_bytes: bytes, threshold: float = 0.5) -> AnalysisReport:
        """
        In-memory analysis (no filed needed ;) ). Mirrors run_analysis(), but calls *bytes* APIs from face detector and landmarker AND hat_glasses.
        """
        # Face + landmarks (bytes-based)
        face_detector_result = self.face_detector.analyze_bytes(image_bytes)
        face_landmarker_result = self.face_detector.analyze_landmarks_bytes(image_bytes)

        checks = []

        # 1) Face present
        checks.append(self.face_detector.is_face_in_image(face_detector_result))

        # 2) Single face
        checks.append(self.face_detector.is_single_face(face_detector_result))

        # 3) Landmarks present
        checks.append(self.face_detector.are_landmarks_present(face_landmarker_result))

        # 4) Eyes visible
        checks.append(self.face_detector.eyes_visible_check(face_landmarker_result))

        # 5) Mouth closed
        checks.append(self.face_detector.mouth_closed_check(face_landmarker_result))

        # 6) No hat and no glasses
        checks.extend(
            self.hat_glasses_detector.check_hats_and_glasses_bytes(
                image_bytes,
                threshold=threshold
            )
        )

        no_glasses_check = self.find_check(checks, Requirement.NO_GLASSES)

        det = False
        if no_glasses_check and no_glasses_check.details:
            det = no_glasses_check.details.get("detected", False)

        if det:
            # 7) Glare and sunglasses check
            checks.extend(
                self.glasses_logic.run_all(
                    image_bytes=image_bytes,
                    face_detector_result=face_detector_result,
                    face_landmarker_result=face_landmarker_result
                )
            )

        # 8) image clear check (Laplacian + Tenengrad + Background)
        # checks.append(
        #     self.image_clear_check.analyze_bytes(
        #     image_bytes=image_bytes,
        #     landmarker_result=face_landmarker_result,
        #     )
        # )

        # 8.5) pixelation check (separat krav)
        #checks.append(
            #self.pixelation_check.check_pixelation_bytes(
                #image_bytes,
                #requirement=Requirement.PIXELATION
            #)
        #)

        # 9) exposure_logic / lighting check
        checks.append(
            self.exposure_check.analyze(image_bytes, face_landmarker_result)
        )

        # 10) face looking straight
        checks.append(
            self.face_looking_at_camera.face_detector(result=face_landmarker_result)
        )

        # 11) face centered
        checks.append(
            self.head_centering_validator.check_from_detection_bytes(
                det_res=face_detector_result,
                image_bytes=image_bytes
            )
        )

        overall_pass = all(c.passed for c in checks)
        return AnalysisReport(
            image="<bytes>",
            passed=overall_pass,
            checks=checks
        )


