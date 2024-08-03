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
face_maps = mp_face_mesh.FACEMESH_FACE_OVAL | mp_face_mesh.FACEMESH_LEFT_EYEBROW | mp_face_mesh.FACEMESH_RIGHT_EYEBROW | mp_face_mesh.FACEMESH_NOSE | mp_face_mesh.FACEMESH_LIPS
face_model = get_unique(face_maps)


face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for index in face_model:
            x = int(index.x * image.width)
            y = int(index.y * image.height)
            landmarks.extend([x, y])
        return landmarks, face_landmarks
    else:
        return None, None

# Function to use the model for real-time face detection
def detect_face_similarity(model, face_data):
    face_data = np.array(face_data).reshape(1, -1)  # Ensure the data is in the correct shape
    prediction = model.predict_proba(face_data)
    similarity_percentage = prediction[0][1] * 100  # Assuming the positive class is at index 1
    return similarity_percentage

# Function to draw landmarks on the image
def draw_landmarks(image, face_landmarks):
    img = np.array(image)
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * img.shape[1])
        y = int(landmark.y * img.shape[0])
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    return img
