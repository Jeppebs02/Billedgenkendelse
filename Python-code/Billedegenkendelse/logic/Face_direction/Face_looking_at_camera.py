from cvzone.PoseModule import PoseDetector
import cv2
from ultralytics import YOLO
import torch
import math
import numpy as np
import pickle
import imutils

cap = cv2.VideoCapture(1)

detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    org_img = img.copy()
    img_h, img_w, _ = org_img.shape
    # Find the human pose in the frame
    img = detector.findPose(img, draw=False)

    # Find the landmarks, bounding box, and center of the body in the frame
    # Set draw=True to draw the landmarks and bounding box on the image
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    # Check if any body landmarks are detected

    if lmList:
        body_2d = []
        body_3d = []
        for (idx, landmark) in enumerate(lmList):
            x, y, z = landmark
            if idx == 0 or idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 5 or idx == 6 or idx == 7 or idx == 8 or idx == 9 or idx == 10:
                # if idx==23 or idx==24:
                if idx == 0:
                    nose_2d = (x, y)
                    nose_3d = (x, y, z)

                body_2d.append([x, y])
                body_3d.append([x, y, z / 3000])

                cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        # point1 = body_3d[0]
        # point2 = body_3d[1]
        # point2 = body_3d[1]

        body_2d = np.array(body_2d, dtype=np.float64)
        body_3d = np.array(body_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cameraMatrix = camera_matrix = np.array([
            [focal_length, 0.0, img_h / 2],  # Principal point at the image center
            [0.0, focal_length, img_w / 2],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(body_3d, body_2d, cameraMatrix, dist_matrix)

        r_mat, jac = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(r_mat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        if y < -10:
            text = "Straight"
        elif y > 10:
            text = "Turn Right"
        # else:
        #     text = "Straight"

        hip_3d_projection, jacobian = cv2.projectPoints(hip_3d, rot_vec, trans_vec, cameraMatrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + (y * 10)), int(nose_2d[1] - (x * 10))
              cv2.line(img, p1, p2, (255, 0, 0), 3)

        cv2.putText(img, text, (20, 50), 1, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Y: {round(y, 2)}", (img_w - 100, 50), 1, 2, (0, 255, 0), 2)
        # Display the frame in a window
        cv2.imshow("Image", imutils.resize(img, height=480))

        # Wait for 1 millisecond between each frame
        key = cv2.waitKey(1)

        if key == 115:
            break