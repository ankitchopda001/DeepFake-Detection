import cv2 
from src.preprocessing import preprocess_image
from src.face_detector import extract_face

def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face = extract_face(frame)

        if face is not None:
            processed = preprocess_image(face)
            pred = model.predict(processed)[0][0]
            predictions.append(pred)

    cap.release()

    if len(predictions) == 0:
        return 0
    
    return sum(predictions) / len(predictions)

# 👉 Uses ALL above files