# FAST-EYE: AI-Powered Brain Stroke Early Screening


FAST-EYE is an AI-powered stroke early-screening prototype that analyzes:

Facial asymmetry (possible facial droop)

Speech clarity (possible slurred speech)

User-typed symptoms (text-based inference of stroke type)

## The system provides:

Stroke risk level prediction

Possible stroke type (Ischemic, Hemorrhagic, TIA)

Immediate precautions and recommendations

Image, video, audio & text-based analysis

Clean Streamlit UI with wide layout

## ⚠️ This project is a prototype (MVP) created for an Idea Hackathon. It is NOT a medical diagnostic tool.

## ⭐ Features
## 🧠  1. Face-Based Stroke Screening

Upload neutral + smile images, OR

Upload a short video (frame is extracted automatically)

Facial landmark analysis using MediaPipe FaceMesh

Detects facial droop using asymmetry scoring

## 🎤 2. Speech-Based Stroke Screening

Upload a short 5–10 second audio clip

Audio processed using librosa + soundfile

Measures speech clarity → Detects slurred speech

## 💬 3. Symptom Chat Analysis

User can type symptoms in natural language

Rule-based medical keyword inference

## Predicts stroke type:

Ischemic

Hemorrhagic

TIA

Unclear/Other

Shows precautions for each type

## 🎯 4. Combined Recommendation

Merges face + speech + symptoms

Shows emergency guidance when needed

## 🎨 5. Modern Streamlit UI

Full-width layout

Column spacing

Custom CSS

Medium-sized preview of uploaded images

Color-coded alert boxes for clarity

## 🛠️ Tech Stack
Component	Technologies Used
Frontend/UI	Streamlit, HTML/CSS
Face Analysis	OpenCV, MediaPipe FaceMesh
Audio Analysis	librosa, soundfile
Symptom Classification	Rule-based NLP
Video Processing	OpenCV (middle-frame extraction)
Environment	Python 3.8+
📂 Project Structure
```bash
FAST_Eye/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── sample_data/          # Placeholder samples (audio, images, videos)
└── README.md             # Documentation
```

🚀 How to Run Locally


1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/FAST_Eye.git
cd FAST_Eye
```

2️⃣ **Create a Conda Environment**
```bash
conda create -n fasteye python=3.9
conda activate fasteye
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
4️⃣ **Run the App**
```bash
python -m streamlit run app.py
```
