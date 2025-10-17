import cv2
import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logic.logic_controller import LogicController

from flask import Flask, request, jsonify, flash, redirect, url_for

app = Flask(__name__)
logic = LogicController()

# Configuration


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _to_jsonable_check(c):
    # Convert enums to strings. We need this because flasks jsonify cant handle enums (I LOVE FLASK)
    return {
        "requirement": str(getattr(c, "requirement", "")),
        "passed": bool(getattr(c, "passed", False)),
        "severity": str(getattr(c, "severity", "")),
        "message": getattr(c, "message", None),
        "details": getattr(c, "details", None),
    }


def _check_file_name(file):
    pass


@app.route("/")
def hello_world():
    return "<p>OK</p>"


@app.route("/analyze", methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        # TODO: dont forget to set the enctype="multipart/form-data" attribute on the frontend HTML form :)
        # See https://flask.palletsprojects.com/en/stable/patterns/fileuploads/
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # check file extension
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        if file.mimetype and not file.mimetype.startswith("image/"):
            return jsonify({"error": f"Unexpected content-type: {file.mimetype}"}), 400

        # Default to 0.5, get threshold from form or query param
        raw_thr = request.form.get("threshold") or request.args.get("threshold") or "0.5"
        try:
            threshold = float(raw_thr)
        except ValueError:
            return jsonify({"error": "threshold must be a float"}), 400

        # Read bytes
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        # Run analysis
        report = logic.run_analysis_bytes(image_bytes, threshold=threshold)

        # Build response
        resp = {
            "image": getattr(report, "image", "<bytes>"),
            "passed": bool(getattr(report, "passed", False)),
            "threshold": threshold,
            "checks": [_to_jsonable_check(c) for c in getattr(report, "checks", [])],
        }

        # Get an overall pass
        overall_pass = bool(resp.get("passed", False))
        decision = "APPROVED" if overall_pass else "REJECTED"
        resp["decision"] = decision

        # THIS IS OPTIONAL
        resp["summary"] = {
            "face": next(c for c in resp["checks"] if "FACE_PRESENT" in str(c["requirement"]))["passed"],
            "single_face": next(c for c in resp["checks"] if "SINGLE_FACE" in str(c["requirement"]))["passed"],
            "mouth_closed": next(c for c in resp["checks"] if "MOUTH_CLOSED" in str(c["requirement"]))["passed"],
            "no_hat": next(c for c in resp["checks"] if "NO_HAT" in str(c["requirement"]))["passed"],
            "no_glasses": next(c for c in resp["checks"] if "NO_GLASSES" in str(c["requirement"]))["passed"],
        }

        # Fail if not overall pass is false :)
        status = 200 if overall_pass else 422
        return jsonify(resp), status

    except Exception as e:
        # TODO: replace with proper logging
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)
