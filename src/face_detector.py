from mtcnn import MTCNN
import cv2

detector = MTCNN()

def extract_face(frame):
    results = detector.detect_faces(frame)

    if len(results) == 0:
        return None
    
    x, y, w, h = results[0]['box'] # w=width, h=height
    face = frame[y:y+h, x:x+w]

    return face

# 👉 Needed before prediction