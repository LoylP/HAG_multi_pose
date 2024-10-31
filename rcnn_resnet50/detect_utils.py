import torchvision.transforms as transforms
import cv2
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
transform = transforms.Compose([
    transforms.ToTensor(),
])

SKELETON_EDGES = {
    'face': [
        (0, 1),    # Mũi -> Mắt trái
        (0, 2),    # Mũi -> Mắt phải
        (1, 3),    # Mắt trái -> Tai trái
        (2, 4),    # Mắt phải -> Tai phải
    ],
    
    'body': [
        (0, 17),   # Mũi -> Điểm giữa vai (điểm tự tính)
        (17, 18),  # Điểm giữa vai -> Điểm giữa hông (điểm tự tính)
    ],
    
    'arms': [
        (5, 6),    # Nối 2 vai
        (5, 7),    # Vai trái -> Khuỷu tay trái
        (7, 9),    # Khuỷu tay trái -> Cổ tay trái
        (6, 8),    # Vai phải -> Khuỷu tay phải
        (8, 10),   # Khuỷu tay phải -> Cổ tay phải
    ],
    
    'legs': [
        (11, 13),  # Hông trái -> Đầu gối trái
        (13, 15),  # Đầu gối trái -> Mắt cá chân trái
        (12, 14),  # Hông phải -> Đầu gối phải
        (14, 16),  # Đầu gối phải -> Mắt cá chân phải
    ]
}

def predict(image, model, device, detection_threshold):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    pred_keypoints = outputs[0]['keypoints'].detach().cpu().numpy()  # Lấy keypoints từ model

    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    keypoints = pred_keypoints[pred_scores >= detection_threshold]  # Lọc keypoints theo threshold
    return boxes, pred_classes, outputs[0]['labels'], keypoints

def calculate_mid_point(p1, p2):
    return [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2, min(p1[2], p2[2])]

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid_connection(p1, p2, image_width, max_distance_ratio=0.5):
    distance = calculate_distance(p1, p2)
    max_allowed_distance = image_width * max_distance_ratio
    return distance <= max_allowed_distance

def draw_skeleton(image, keypoints, visibility_threshold=0.5):
    image_height, image_width = image.shape[:2]
    
    for person_kps in keypoints:
        if (person_kps[5][2] > visibility_threshold and 
            person_kps[6][2] > visibility_threshold and 
            is_valid_connection(person_kps[5], person_kps[6], image_width)):
            mid_shoulder = calculate_mid_point(person_kps[5], person_kps[6])
        else:
            mid_shoulder = [0, 0, 0]  # Invalid point

        if (person_kps[11][2] > visibility_threshold and 
            person_kps[12][2] > visibility_threshold and 
            is_valid_connection(person_kps[11], person_kps[12], image_width)):
            mid_hip = calculate_mid_point(person_kps[11], person_kps[12])
        else:
            mid_hip = [0, 0, 0]  

        extended_kps = np.vstack([
            person_kps,
            mid_shoulder,  # index 17
            mid_hip       # index 18
        ])

        for part, edges in SKELETON_EDGES.items():
            color = {
                'face': (176,224,230),    # Đỏ
                'body': (176,196,222),   # Vàng
                'arms': (135,206,235),     # Xanh lá
                'legs': (135,206,250)      # Xanh dương
            }[part]
            
            for edge in edges:
                p1, p2 = edge
                point1 = extended_kps[p1]
                point2 = extended_kps[p2]

                # Kiểm tra nhiều điều kiện trước khi vẽ
                if (point1[2] > visibility_threshold and 
                    point2[2] > visibility_threshold and 
                    is_valid_connection(point1, point2, image_width)):
                    
                    pt1 = (int(point1[0]), int(point1[1]))
                    pt2 = (int(point2[0]), int(point2[1]))

                    # Kiểm tra tọa độ có nằm trong ảnh không
                    if (0 <= pt1[0] < image_width and 0 <= pt1[1] < image_height and 
                        0 <= pt2[0] < image_width and 0 <= pt2[1] < image_height):
                        
                        # Thêm kiểm tra cho từng phần cụ thể
                        should_draw = True
                        if part == 'face':
                            # Chỉ vẽ khi khoảng cách các điểm trên mặt hợp lý
                            max_face_ratio = 0.2
                            should_draw = is_valid_connection(point1, point2, image_width, max_face_ratio)
                        elif part == 'body':
                            # Kiểm tra thân người
                            max_body_ratio = 0.4
                            should_draw = is_valid_connection(point1, point2, image_width, max_body_ratio)
                        elif part == 'arms' or part == 'legs':
                            # Kiểm tra tay và chân
                            max_limb_ratio = 0.3
                            should_draw = is_valid_connection(point1, point2, image_width, max_limb_ratio)

                        if should_draw:
                            cv2.line(image, pt1, pt2, color, 1)

def draw_boxes_and_keypoints(boxes, classes, labels, keypoints, image):
    for i, box in enumerate(boxes):
        if classes[i] == 'person':
            color = COLORS[labels[i]]
            
            # Vẽ bounding box chỉ khi box hợp lý
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            if box_width > 0 and box_height > 0 and box_width < image.shape[1] and box_height < image.shape[0]:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
                cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
            
            draw_skeleton(image, [keypoints[i]])
            
            # Chỉ vẽ keypoints có độ tin cậy cao và nằm trong box
            for kp in keypoints[i]:
                x, y, v = kp
                if (v > 0.5 and 
                    box[0] <= x <= box[2] and 
                    box[1] <= y <= box[3]):
                    cv2.circle(image, (int(x), int(y)), 3, (205,133,63), -1)

    return image

def person_boxes(boxes, labels, image):
    p_boxes = []
    for b, l in zip(boxes, labels):
        if l != 1:
            continue
        p_boxes.append(b)
        
    return image, p_boxes