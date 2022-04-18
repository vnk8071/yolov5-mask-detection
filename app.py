from utils.general import scale_coords
from utils.plots import Annotator, colors
from utils.flask_utils import extract_file, get_object_predict, process_predict

import cv2
import copy
import json
import numpy as np
import os
import pandas as pd
import time
import tempfile
import torch

from flask import Flask, request, render_template, jsonify
from models.experimental import attempt_load

# Init flask
app = Flask(__name__)
app.config.from_file("config/flask_cfg.json", load=json.load)

# Params
imgsz = [640, 640]
dict_yolo_models = {'Yolov5s': 'models/best-yolov5s.pt'}


def read_video_cap(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    vid_cap = cv2.VideoCapture(tfile.name)
    num_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return vid_cap, num_frames


@app.route('/')
def upload_form():
    return render_template("index.html",
                           len_yolo=len(dict_yolo_models.keys()), list_yolo_models=list(dict_yolo_models.keys()))


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file, filename, filetype = extract_file(request)
        save_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Load object detection model
        model_object = attempt_load(
            dict_yolo_models["Yolov5s"], map_location='cpu')
        names = model_object.names

        # Process with image
        if filetype == "jpg" or filetype == "jpeg" or filetype == "png":
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(save_file + '.jpg', img0)

            # Predict object detection tbup
            img, object_predict = get_object_predict(
                img0, imgsz, model_object)

            # Process predictions
            img, count_class = process_predict(img, object_predict,
                                               img0, names)
            cv2.imwrite(save_file + '_predicted.jpg', img)
            return render_template("predict_image.html", image=save_file + '.jpg', image_predicted=save_file + '_predicted.jpg', count_class=count_class)

        # Process with video
        elif filetype == "mp4" or filetype == "avi":
            start = time.perf_counter()
            vid_cap, num_frames = read_video_cap(file)
            bs = 1  # batch_size
            vid_writer = [None] * bs
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            vid_writer = cv2.VideoWriter(
                save_file + '_predicted.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

            # Process with downsampling
            if vid_cap.isOpened():
                for i in range(num_frames):
                    ret, frame = vid_cap.read()
                    if ret and i % 3 == 0:
                        img, object_predict = get_object_predict(
                            frame, imgsz, model_object)
                        img, _ = process_predict(img, object_predict,
                                                 frame, names)
                        vid_writer.write(img)

                    elif ret and i % 3 != 0:
                        img, _ = process_predict(img, previous_pred_object,
                                                 frame, names)
                        vid_writer.write(img)
                    else:
                        pass
                    previous_pred_object = copy.deepcopy(object_predict)

            else:
                print("Cannot open ", filename)

            # Save inference
            vid_cap.release()
            vid_writer.release()
            end = time.perf_counter()

            total = round(end - start)
            # Print results
            print('Total frames: ', num_frames)
            print(
                f"Total time: {total//60} minute(s) {total%60} seconds with {round(total/num_frames, 3)} second/frame")
            return render_template("predict_video.html", video=save_file + '_predicted.avi')


@app.route('/predict', methods=['POST'])
def predict_json():
    if request.method == 'POST':
        file, filename, filetype = extract_file(request)
        save_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Load models
        model_object = attempt_load(
            'models/best-yolov5s.pt', map_location='cpu')
        names = model_object.names

        # Process with image
        if filetype == "jpg" or filetype == "jpeg" or filetype == "png":
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(save_file + '.jpg', img0)
            # Predict object detection tbup
            img, object_predict = get_object_predict(
                img0, imgsz, model_object)

            # Process predictions
            img, _ = process_predict(img, object_predict,
                                     img0, names)
            cv2.imwrite(save_file + '_predicted.jpg', img)
            return jsonify({"image": request.url_root + save_file + '_predicted.jpg'})

        # Process with video
        elif filetype == "mp4" or filetype == "avi":
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            vid_cap = cv2.VideoCapture(tfile.name)
            bs = 1  # batch_size
            vid_writer = [None] * bs
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_writer = cv2.VideoWriter(
                save_file + '_predicted.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

            # Process with downsampling
            if vid_cap.isOpened():
                for i in range(num_frames):
                    ret, frame = vid_cap.read()
                    if ret and i % 3 == 0:

                        img, object_predict = get_object_predict(
                            frame, imgsz, model_object)
                        img, _ = process_predict(img, object_predict,
                                                 frame, names)
                        vid_writer.write(img)

                    elif ret and i % 3 != 0:
                        img, _ = process_predict(img, previous_pred_object,
                                                 frame, names)
                        vid_writer.write(img)
                    else:
                        pass
                    previous_pred_object = copy.deepcopy(object_predict)

                # Save inference
                vid_cap.release()
                vid_writer.release()

            else:
                print("Cannot open ", filename)
    return jsonify({
        "video": request.url_root + save_file + '_predicted.avi'})


if __name__ == '__main__':
    if app.config['FLASK_ENV'] == 'development':
        app.run(host='0.0.0.0', port=app.config['PORT'])
    else:
        app.run(host=app.config['SERVER_IP'], port=app.config['PORT'])
