import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# create folder if not exists
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# save model safely
def save_model(model, path="G:/DeepFack-Detection/models/model.h5"):
    create_dir(os.path.dirname(path))
    model.save(path)
    print(f"[INFO] model saved as: {path}")

# Load model
def load_model(path):
    from tensorflow.keras.models import load_model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    print(f"[INFO] Loading model from: {path}")
    return load_model(path)

# plot training history
def plot_training(history, save_path="G:/DeepFack-Detection/outputs/graphs/training_history.png"):
    create_dir(os.path.dirname(save_path))

    plt.figure()

    # Accuracy
    plt.plot(history.history['accuracy'], label = 'Train Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'validation Accuracy')
    
    # Loss
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loass'], label = 'validation loass')
    
    plt.legend()
    plt.title("Training Performance")

    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Training plot saved at: {save_path}")

# Convert predictions to label
def decode_prediction(pred):
    return "Fake" if pred >= 0.5 else "Real"



# Average predictions (for video)
def average_predictions(predictions):
    if len(predictions) == 0:
        return 0
    return float(np.mean(predictions))

# Save results log
def log_result(result, confidence, file_name="G:/DeepFack-Detection/outputs/results_log.txt"):
    create_dir(os.path.dirname(file_name))

    with open(file_name, "a") as f:
        f.write(f"{datetime.now()} | Result: {result} | confidence: {confidence}\n")


# Simple confidence formatter
def format_confidence(score):
    return f"{round(score * 100, 2)}%"


# Check file type
def is_image(file_type):
    return "image" in file_type

def is_video(file_type):
    return "video" in file_type