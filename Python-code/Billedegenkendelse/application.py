import cv2
import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchgen.gen_functionalization_type import wrap_propagate_mutations_and_return

from logic.logic_controller import LogicController

from flask import Flask, request, jsonify, flash, redirect, url_for
from functools import wraps
from flask_cors import CORS

app = Flask(__name__)
logic = LogicController()

# Configuration


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# hardcoded api key for simplicity

API_KEY = r"a541fe33-6c48-490c-b71a-eadab16594de"

# require api key decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('x-api-key')
        if key == API_KEY:
            print("API key valid")
            return f(*args, **kwargs)
        else:
            print("API key invalid")
            return jsonify({"error": "Unauthorized"}), 401
    return decorated_function


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
def health_endpoint():
    version = os.environ.get("APP_VERSION", "unknown")
    return f"<p>OK. App version: {version}</p>"


@app.route("/analyze", methods=['POST'])
@require_api_key
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        # See https://flask.palletsprojects.com/en/stable/patterns/fileuploads/
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # check file extension
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        if file.mimetype and not file.mimetype.startswith("image/"):
            return jsonify({"error": f"Unexpected content-type: {file.mimetype}"}), 400

        # Default to 0.5, get threshold from form or query param.
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

        # Fail if not overall pass is false :)
        status = 200 if overall_pass else 422
        return jsonify(resp), status

    except Exception as e:
        # TODO: replace with proper logging
        print(e)
        return jsonify({"error": str(e)}), 500


CORS(app, resources={r"/*": {"origins": "*"}})



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)
