from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

import numpy as np
import pandas as pd
import time
import tempfile
import torch

from werkzeug.exceptions import BadRequest


def get_object_predict(img0, imgsz, model_object):
    """
    Return resized image (640, 640) and predict of object detection TBUP

    Args:
    img0<array> -- original image
    imgsz<list> -- Height and Width to resize (Default [640, 640])
    model_object<> -- load weight of object detection model
    """
    # Inference
    model_object(torch.zeros(
        1, 3, *imgsz).to('cpu').type_as(next(model_object.parameters())))
    stride = int(model_object.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Padded resize
    img = letterbox(img0, imgsz[0],
                    stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()/255

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Predict
    pred = model_object(img)[0]

    # NMS
    pred = non_max_suppression(
        pred, 0.25, 0.45, max_det=1000)
    return img, pred


def extract_file(request):
    """Checking if image uploaded is valid"""
    if 'file' not in request.files:
        raise BadRequest(
            "Missing file parameter! Key parameter is file for image and video")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    filename, filetype = file.filename.rsplit('.', 1)
    return file, filename, filetype.lower()


def process_predict(img, pred, frame, names):
    """
    Process object predictiton and update profile of each frame

    Args:
    idx<int> -- index of frame
    img<array> -- post-precessing image
    pred<dict> -- annotations of prediction
    pred_classes<list> -- list contain event tbup
    profiles<dict> -- series of encode object detection tbup
    frame<array> -- original image
    names<list> -- classes of TBUP
    vid_writer<function> cv2.VideoWriter -- save video
    """
    count_class = [0, 0, 0]
    # Process predictions
    for i, det in enumerate(pred):  # per image
        annotator = Annotator(
            frame, line_width=3, pil=not ascii)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], frame.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                count_class[c] += 1
                label = f'{names[c]}'
                annotator.box_label(
                    xyxy, label, color=colors(c, True))

        img = annotator.result()
        return img, count_class
