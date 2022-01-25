# YOLOv5 - WEARING MASK - OBJECT DETECTION 

Data collected from FSOFT, the Organizing Commitee provides a dataset of 976 pictures labeled with bounding boxes in 3 classes: wearing masks, not wearing masks and wearing masks incorrectly; 1 pre-trained model; 1 public test dataset of 89 pictures for participants to make preliminary analysis of the model.

## Install requirements
```bash
conda create -n yolov5detector python=3.8
conda activate yolov5detector
pip install -r requirements.txt
```

## Download data (YOLO format)
```bash
bash setup_data.sh
```

The directory of dataset structured:

```bash
    ├────────── /data/ ──────────│
  images                       labels                   
    ├── train                    ├── train
    │    ├── 1.png               │    ├── 1.txt
    │    ├── ...                 │    ├── ...
    │    └── 792.png             │    └── 792.txt                    
    ├── val                      ├── val
    │    ├── 793.png             │    ├── 793.txt
    │    ├── ...                 │    ├── ...
    │    └── 976.png             │    └── 976.txt 
    └── test                     └── test
            ├── 977.png               ├── 977.txt
            ├── ...                   ├── ...
            └── 1064.png              └── 1064.txt

```

## Train
```bash
python train.py --batch-size 64 --name <version> --device 0
```
Note:
- <version>: To save weights and results
- Remember to check the config before train in config/train_cfg.yaml

## Evaluate
```bash
python val.py --weights ./results/train/<version>/weights/best.pt  --task val --batch-size 64 --name <version> --device 0
```

## Inference
```bash
python detect.py --weights results/train/<version>/weights/best.pt --source ./data/test --dir ./inference/<version>
```

## **YOLOv5 🚀 by Ultralytics, GPL-3.0 license**

Source: https://github.com/ultralytics/yolov5

