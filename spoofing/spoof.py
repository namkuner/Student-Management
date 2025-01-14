import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import cv2 as cv
from utils import read_py_config,build_model,load_checkpoint
import albumentations as A
import numpy as np

# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


spf_model ="weights/13_spoof/MobileNet3_0.75_small.pth.tar"
config  = "E:\AI\Student_Attendance_Update\spoofing\config_small_075.py"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = read_py_config(config)

def load_weight():
    model= build_model(config,device,mode="eval")
    model = load_checkpoint(spf_model,model,device)

    return model

def preprocessing(images):
    ''' making image preprocessing for pytorch pipeline '''
    mean = np.array(object=config.img_norm_cfg.mean).reshape((3, 1, 1))
    std = np.array(object=config.img_norm_cfg.std).reshape((3, 1, 1))
    height, width = list(config.resize.values())
    preprocessed_imges = []
    for img in images:
        img = cv.resize(img, (height, width), interpolation=cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = img / 255
        img = (img - mean) / std
        preprocessed_imges.append(img)
    return torch.tensor(preprocessed_imges, dtype=torch.float32)



