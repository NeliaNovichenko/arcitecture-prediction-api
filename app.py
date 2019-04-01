#!flask/bin/python
import json
from flask_cors import CORS, cross_origin
from config import Configuration
from prediction import predict
import sys
import os
import numpy
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config.from_object(Configuration)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/recognition', methods=['POST'])
def get_styles():
    # check if the post request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return abort(400)
    file = request.files['image']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return abort(400)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filePath)
        result = predict(filename)
        os.remove(filePath)
        return json.dumps(result)

