import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadCustomImages_from_np
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

class Config_opt():
    weights = "weights/yolov5x6.pt"
    img_size = 640
    conf_thres = 0.30
    iou_thres = 0.45
    device = 'cpu'
    classes = [2,5,7]
    line_thickness = 2
    augment = True
    hide_labels = True
    hide_conf = False
    agnostic_nms = True 
    stride=32

def load_model(opt):
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    if half:
        model.half()
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    #test run
    if device.type == 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
        print("Test run on CPU successful")
    
    return model,names,colors,half,device,imgsz,stride

def run_inf(model,names,colors,half,device,imgsz,stride,img0,opt):
    img = letterbox(img0,imgsz,stride=stride)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0 
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 
            opt.conf_thres, 
            opt.iou_thres, 
            classes=opt.classes, 
            agnostic=opt.agnostic_nms)
    
    cordinates = []
    scores = []

    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] 
        

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            s = ''
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            
            #print detection names
            print(s)
            for *xyxy, conf, cls in reversed(det):
                cordinates.append([
                                xyxy[0].item(),
                                xyxy[1].item(),
                                xyxy[2].item() - xyxy[0].item(),
                                xyxy[3].item()-xyxy[1].item()] 
                                )
                scores.append(conf.item())

    return cordinates,scores