"""
Used to train a m YOLOv8 model

Running this locally might actually suck lol
"""


import cv2
from ultralytics import YOLO, solutions

def main():
    # This is used to create a new model for YOLOv8
    model = YOLO("yolov8s.pt")
    model.train(
        data="baseball/data.yaml",
        epochs=100, # How many iterations of training the model is going to do
        imgsz=1280,
        batch = 2,
        patience=20,
        device=0,
        name="baseball_s_v1"
    )

if __name__ == "__main__":
    main()





