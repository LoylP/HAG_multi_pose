import torchvision
import numpy as np
import torch
import cv2
import detect_utils

# Initialize all models
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, min_size=800)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)

# Initialize age and gender detection models
try:
    faceNet, ageNet, genderNet = detect_utils.initialize_age_gender_models()
except Exception as e:
    print(f"Error initializing age/gender models: {str(e)}")
    exit(1)

cap = cv2.VideoCapture('/home/loylp/project/HAG/output003.mp4')

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("End of video file")
            break
            
        # Kiểm tra frame có hợp lệ không
        if frame is None or frame.size == 0:
            print("Invalid frame")
            continue
            
        boxes, classes, labels, keypoints = detect_utils.predict(frame, model, device, 0.8)
        
        # Chỉ xử lý frame khi có detections
        if len(boxes) > 0:
            frame = detect_utils.draw_boxes_and_keypoints(boxes, classes, labels, keypoints, frame, 
                                                        faceNet, ageNet, genderNet)
        
        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()