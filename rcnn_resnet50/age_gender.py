import cv2 

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

def initialize_age_gender_models():
    faceProto = "config/opencv_face_detector.pbtxt"
    faceModel = "config/opencv_face_detector_uint8.pb"
    ageProto = "config/age_deploy.prototxt"
    ageModel = "config/age_net.caffemodel"
    genderProto = "config/gender_deploy.prototxt"
    genderModel = "config/gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    return faceNet, ageNet, genderNet

def detect_age_gender(image, box, faceNet, ageNet, genderNet, padding=20):
    face = image[
        max(0, int(box[1]) - padding):min(int(box[3]) + padding, image.shape[0] - 1),
        max(0, int(box[0]) - padding):min(int(box[2]) + padding, image.shape[1] - 1)
    ]
    
    if face.size == 0:
        return None, None
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    
    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    
    return gender, age