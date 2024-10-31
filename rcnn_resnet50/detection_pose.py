import torchvision
import numpy as np
import torch
import cv2
import detect_utils


cap = cv2.VideoCapture('/home/loylp/project/HAG/output003.mp4')
ret, frame = cap.read()

# Thay thế Faster R-CNN bằng Keypoint R-CNN
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, min_size=800)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)


while True:
    ret, frame = cap.read()
    boxes, classes, labels, keypoints = detect_utils.predict(frame, model, device, 0.8)
    frame = detect_utils.draw_boxes_and_keypoints(boxes, classes, labels, keypoints, frame)
    cv2.imshow('Image', frame)
    cv2.waitKey(1)
