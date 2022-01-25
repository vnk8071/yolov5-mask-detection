# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                  | Example                   | `--include ...` argument
---                     | ---                       | ---
PyTorch                 | yolov5s.pt                | -
TorchScript             | yolov5s.torchscript       | `torchscript`
ONNX                    | yolov5s.onnx              | `onnx`
CoreML                  | yolov5s.mlmodel           | `coreml`
OpenVINO                | yolov5s_openvino_model/   | `openvino`
TensorFlow SavedModel   | yolov5s_saved_model/      | `saved_model`
TensorFlow GraphDef     | yolov5s.pb                | `pb`
TensorFlow Lite         | yolov5s.tflite            | `tflite`
TensorFlow.js           | yolov5s_web_model/        | `tfjs`
TensorRT                | yolov5s.engine            | `engine`

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx coreml openvino saved_model tflite tfjs

Inference:
    $ python path/to/detect.py --weights yolov5s.pt
                                         yolov5s.torchscript
                                         yolov5s.onnx
                                         yolov5s.mlmodel  (under development)
                                         yolov5s_openvino_model  (under development)
                                         yolov5s_saved_model
                                         yolov5s.pb
                                         yolov5s.tflite
                                         yolov5s.engine

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

from utils.torch_utils import select_device
from utils.general import (check_img_size, colorstr, file_size)
from utils.datasets import LoadImages
from utils.activations import SiLU
from models.yolo import Detect
from models.experimental import attempt_load
from models.common import Conv
import argparse
import json
import os
import urllib
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOGGER = logging.getLogger(__name__)


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        LOGGER.info(
            f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(
            max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        (optimize_for_mobile(ts) if optimize else ts).save(
            str(f), _extra_files=extra_files)

        LOGGER.info(
            f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    # '%2F' to '/', split https://url.com/file.txt?auth
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]
    return file


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
):
    t = time.time()
    include = [x.lower() for x in include]
    file = Path(url2file(weights) if str(weights).startswith(
        ('http:/', 'https:/')) else weights)

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    # Load PyTorch model
    device = select_device(device)
    assert not (
        device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device,
                         inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    # verify img_size are gs-multiples
    imgsz = [check_img_size(x, gs) for x in imgsz]
    # image size(1,3,320,192) BCHW iDetection
    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    # training mode = no Detect() layer grid construction
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(im)  # dry runs
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} ({file_size(file):.1f} MB)")

    # Exports
    if 'torchscript' in include:
        export_torchscript(model, im, file, optimize)

    # Finish
    LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true',
                        help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true',
                        help='model.train() mode')
    parser.add_argument('--optimize', action='store_true',
                        help='TorchScript: optimize for mobile')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, engine)')
    opt = parser.parse_args()
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
