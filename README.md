#DeepFake Detection (Image & Video)

An end-to-end deep learning system to detect DeepFake media in both images and videos using computer vision and deep learning techniques.

🚀 Demo

👉 Upload an image or video and get:

✅ Real / Fake prediction
📊 Confidence score
🎥 Frame-by-frame video analysis
🧠 Problem Statement

DeepFakes are AI-generated media that manipulate faces, voices, or actions, posing serious risks such as:

Fake news & misinformation
Identity fraud
Cybersecurity threats

This project aims to build a robust detection system to identify manipulated content.

⚙️ Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy, Pandas
Streamlit (for UI)


🧠 Model Architecture
🔹 Image Detection
CNN-based model (EfficientNet)
Detects spatial inconsistencies in facial regions
🔹 Video Detection
Frame extraction using OpenCV
Face detection using MTCNN
Frame-level predictions using CNN
Aggregated results for final classification


📁 Project Structure
DeepShield-AI/
│
├── app/
│   └── app.py                # Streamlit UI
│
├── src/
│   ├── image_model.py       # CNN model
│   ├── video_model.py       # Video processing
│   ├── preprocessing.py     # Image preprocessing
│   ├── face_detector.py     # Face extraction
│   └── utils.py             # Helper functions
│
├── models/                  # Saved models
├── data/                    # Dataset (not included)
├── outputs/                 # Graphs & logs
├── notebooks/               # Experiments
│
├── requirements.txt
├── README.md
└── .gitignore


🖥️ Features

--> Image DeepFake Detection
--> Video DeepFake Detection
--> Face Extraction using MTCNN
--> Confidence Score Output
--> Fast and interactive UI


📈 Results

Achieves strong performance using transfer learning
Handles both spatial and temporal inconsistencies
Scalable architecture for real-world deployment

👨‍💻 Author

Ankit Chopda
Aspiring Data Scientist / ML Engineer
