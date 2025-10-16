import cv2
import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from logic.logic_controller import LogicController

from flask import Flask, request, jsonify, flash, redirect, url_for

app = Flask(__name__)

# Configuration

# TODO: change to actual image extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

        # check file extension
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400


        #TODO: Do actual logic with the file, ie. pass it to the LogicController

    except Exception as e:

        #TODO: Log the error properly PLEASE
        return jsonify({"error": str(e)}), 400


    # TODO: OBVIOUSLY change the return to the actual analysis result
    return "Image analyzed", 200



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)
