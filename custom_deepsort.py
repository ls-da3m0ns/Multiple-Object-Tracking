import sys
sys.path.append('./yolov5/')
sys.path.append('./object_tracking')

from for_custom_tracking import *
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2,pickle,sys

from deepsort import *
from scipy.stats import multivariate_normal


def track(options):
    vid_filename = options.source

    tmp_cap = cv2.VideoCapture(vid_filename)
    tmp_shape = tmp_cap.read()[1].shape
    tmp_cap.release()

    #real capture
    cap = cv2.VideoCapture(vid_filename)

    #initialize deepsort and yolo
    #yolo
    print("Constructing YOLO V5")
    opt = Config_opt()
    opt.device = options.device
    opt.weights = options.yolo_weights
    model,names,colors,half,device,imgsz,stride = load_model(opt)
    print("YOLO V5 Constructed")

    #deepsort
    print("Initializing DeepSort with Siamese Network")
    deepsort = deepsort_rbc(wt_path = options.feature_extracter_weights)
    frame_id = 1

    #for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(options.output_file,fourcc, 15.0, (tmp_shape[1],tmp_shape[0]))
    
    while True:
        print(frame_id)
        ret,frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.uint8)

        try:
            detections,out_scores = run_inf(model,names,colors,half,device,imgsz,stride,frame,opt)
        except:
            frame_id+=1
            detections = None
            continue
        
        if detections is None:
            print("No dets")
            frame_id+=1
            continue

        detections = np.array(detections)
        out_scores = np.array(out_scores)

        tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                
            bbox = track.to_tlbr() 
            id_num = str(track.track_id) 
            features = track.features 
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            
            for det in detections_class:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
            
        out.write(frame)
        frame_id+=1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default="/gdrive/MyDrive/car_tracking/car_tracking/yolov5/weights/yolov5x6.pt", help='model.pt path(s)')
    parser.add_argument('--feature_extracter_weights', nargs='+', type=str, default="object_tracking/ckpts/model33.pt")
    parser.add_argument('--source', type=str, default='vid11.mp4', help='source vid file')
    parser.add_argument('--output_file', type=str, default="results_out.mp4", help='filename for output')
    parser.add_argument('--device', type=str, default="0", help='device to run on  0 1 etc')

    opt = parser.parse_args()
    print(f"Aurguments Provided are : {opt} ")
    track(opt)
