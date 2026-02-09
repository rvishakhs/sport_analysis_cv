import os
import shutil

import torch
from ultralytics import YOLO
from roboflow import Roboflow
from multiprocessing import freeze_support




rf = Roboflow(api_key="N9KNdHH9pUV7YKsbZUMO")
project = rf.workspace("visakh-jercs").project("basketball-players-fy4c2-vfsuv-gz8mj")
version = project.version(2)
dataset = version.download("yolo26")

#
# model = YOLO("yolo11x.pt")
#
# model.predict("input_video/video_1.mp4", save=True)

def main():
    # load pretrained model
    model = YOLO("yolo26x.pt")

    # load pretrained model
    results = model.train(
        data=dataset.location + "/data.yaml",
        epochs=250,
        imgsz=640,
        batch=8,
        workers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    os.makedirs("saved_models", exist_ok=True)
    shutil.copy("runs/detect/train/weights/best.pt", "best.pt")
    shutil.copy("runs/detect/train/weights/last.pt", "saved_models/last.pt")
    print("Weights saved to saved_models/")

if __name__ == '__main__':
    freeze_support()
    main()