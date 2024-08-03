import cv2
import mediapipe as mp

import numpy as np
import joblib

mp_face_mesh = mp.solutions.face_mesh

model_file = "trained_XGBoost_model_learning_rate=0.2_max_depth=7_n_estimators=300.joblib"
model_folder = "model/"


model_filename =  model_folder+model_file
model = joblib.load(model_filename)

def get_unique(connections):
    indices = set()
    for connection in connections:
        indices.update(connection)
    return sorted(indices)
peta_wajah = mp_face_mesh.FACEMESH_FACE_OVAL
peta_alis = mp_face_mesh.FACEMESH_LEFT_EYEBROW | mp_face_mesh.FACEMESH_RIGHT_EYEBROW
peta_hidung = mp_face_mesh.FACEMESH_NOSE
peta_mulut = mp_face_mesh.FACEMESH_LIPS
peta_point4d = peta_wajah | peta_alis | peta_hidung | peta_mulut
peta_point4d = get_unique(peta_point4d)
_model = list(peta_point4d)

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
dataX = {}

def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    exclude_indices = ['13', '22', '23', '40', '41', '44', '49', '55', '57', '76', 
           '87', '89', '91', '93', '96', '98', '112', '113', '134', 
           '137', '155', '163', '169', '171', '199', '201', '203', 
           '205', '14', '23', '24', '41', '42', '45', '50', '56', 
           '58', '77', '88', '90', '92', '94', '97', '99', '113', 
           '114', '135', '138', '156', '164', '170', '172', '200', 
           '202', '204', '206']
    unique_indices = sorted(set(int(i) for i in exclude_indices))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            faced = face_landmarks.landmark
            landmarks = []
            base_landmark =[]
            for index in _model:
                x = int(faced[index].x * image.width)
                y = int(faced[index].y * image.height)
                base_landmark.extend([x, y])
                landmarks = [value for i, value in enumerate(base_landmark) if i not in unique_indices]
        return landmarks, face_landmarks
    else:
        return None, None

def detect_face_similarity(model, face_data):
    face_data = np.array(face_data).reshape(1, -1)  
    prediction = model.predict_proba(face_data)
    similarity_percentage = int(prediction[0][1] * 100) if prediction[0][1] > prediction[0][0] else int(prediction[0][0] * 100)
    if similarity_percentage <= 65 :
        detection =  f"Tidak ada indikasi Bell's Palsy dikarena persentase deteksi hanya: {similarity_percentage}%"
    else:
        detection = f"Terdeteksi Bell's Palsy dengan Persentase : {similarity_percentage}%"
    return detection

def draw_landmarks(image):
    img = np.array(image)
    for index in dataX:
        cv2.circle(img, (dataX[index][0], dataX[index][1]), 1, (0, 255, 0), -1)
    return img
