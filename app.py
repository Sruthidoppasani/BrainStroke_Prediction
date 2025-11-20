import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
import tempfile
import os

# Full width
st.set_page_config(page_title="FAST-Eye Prototype", layout="wide")

# ---------------- CSS Styling (spacing, images, fonts) ----------------
st.markdown(
    """
    <style>
    /* More spacing between columns */
    [data-testid="column"] {
        padding-left: 18px;
        padding-right: 18px;
    }

    /* Slightly larger global font for readability */
    html, body, [class*="css"]  {
        font-size: 16px !important;
    }

    /* Limit uploaded image size (medium) */
    img {
        max-width: 380px !important;
        height: auto !important;
        border-radius: 6px;
    }

    /* Make Streamlit container a bit narrower vertically for large headings */
    .stApp .main {
        padding-top: 10px;
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Face mesh / heuristics ----------------
mp_face = mp.solutions.face_mesh

def extract_landmarks_from_image(image_bgr):
    """Returns list of (x,y) landmark positions or None"""
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image_bgr.shape[:2]
        pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
        return pts

def face_midline_and_width(pts):
    xs = [p[0] for p in pts]
    center_x = int(np.mean(xs))
    face_width = max(xs) - min(xs)
    return center_x, face_width

def asymmetry_score(pts):
    pairs = [
        (33, 263),
        (61, 291),
        (70, 300),
        (159, 386),
        (145, 374)
    ]
    center_x, face_w = face_midline_and_width(pts)
    diffs = []
    for l_idx, r_idx in pairs:
        try:
            lx, ly = pts[l_idx]
            rx, ry = pts[r_idx]
        except Exception:
            continue
        dl = abs(lx - center_x)
        dr = abs(rx - center_x)
        diffs.append(abs(dl - dr))
    if len(diffs) == 0:
        return 0.0
    raw = np.mean(diffs)
    score = raw / (face_w + 1e-6)
    asym_pct = min(100, score * 300)
    return asym_pct

# ---------------- Robust audio loader ----------------
def robust_load_audio_from_bytes(bytes_data, target_sr=16000, duration=10.0):
    """
    Save bytes to a temp file and try to load audio robustly:
    1) Try librosa.load
    2) Fallback to soundfile.read + resample
    Returns: (y, sr) or (None, None) on failure
    """
    if bytes_data is None:
        return None, None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(bytes_data)
        tmp.flush()
        tmp.close()
        # First attempt: librosa
        try:
            y, sr = librosa.load(tmp.name, sr=target_sr, mono=True, duration=duration)
            return y, sr
        except Exception:
            # Fallback: soundfile
            try:
                data, sr0 = sf.read(tmp.name, always_2d=False)
                if data is None:
                    return None, None
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr0 != target_sr:
                    try:
                        y = librosa.resample(data.astype(np.float32), orig_sr=sr0, target_sr=target_sr)
                        return y, target_sr
                    except Exception:
                        return data, sr0
                else:
                    return data, sr0
            except Exception:
                return None, None
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def load_audio_and_compute_score_from_bytes(bytes_data):
    """Return clarity score (0-100) or None"""
    y, sr = robust_load_audio_from_bytes(bytes_data, target_sr=16000, duration=10.0)
    if y is None:
        return None
    try:
        _ = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y, sr=sr))
        mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfcc, axis=1))
        rms = np.mean(librosa.feature.rms(y))
        score = 0.0
        score += np.tanh(rms * 10) * 40
        score += (1 - np.exp(-mfcc_var)) * 30
        score += (np.tanh(centroid / 2000)) * 30
        clarity = max(0, min(100, score))
        return clarity
    except Exception:
        return None

# ---------------- Video frame extraction using OpenCV ----------------
def extract_middle_frame_from_video_opencv(video_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            cap.release()
            try: os.unlink(tmp.name)
            except: pass
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            target_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                try: os.unlink(tmp.name)
                except: pass
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fallback: read early frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(0, 50):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                cap.release()
                try: os.unlink(tmp.name)
                except: pass
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
    except Exception:
        try: os.unlink(tmp.name)
        except: pass
    return None

# ---------------- Symptom-based rule classifier ----------------
def predict_stroke_from_symptoms(text):
    """
    Simple rule-based classifier:
    Returns (predicted_type, confidence_percent, matched_keywords)
    Types: 'Ischemic', 'Hemorrhagic', 'TIA', 'Unclear/Other'
    """
    if not text or text.strip() == "":
        return ("Unclear/Other", 0, [])

    t = text.lower()

    ischemic_kw = ["weakness","numb","numbness","face droop","face droops","arm weakness",
                   "leg weakness","one side","one-sided","slurred speech","slur","difficulty speaking",
                   "trouble speaking","speech slurred","confusion","difficulty understanding",
                   "vision loss","blurred vision","difficulty walking","loss of balance","droop"]
    hemorrhagic_kw = ["severe headache","worst headache","sudden severe headache","thunderclap","nausea","vomit","vomiting","loss of consciousness","seizure","very severe headache","headache and vomiting"]
    tia_kw = ["transient","temporary","minutes","came and went","brief","short-lived","resolved","fleeting"]

    score_isc = 0
    score_hem = 0
    score_tia = 0
    matched = []

    for kw in ischemic_kw:
        if kw in t:
            score_isc += 2
            matched.append(kw)
    for kw in hemorrhagic_kw:
        if kw in t:
            score_hem += 3
            matched.append(kw)
    for kw in tia_kw:
        if kw in t:
            score_tia += 3
            matched.append(kw)

    total = score_isc + score_hem + score_tia
    if total == 0:
        if "slur" in t or "droop" in t or "weak" in t:
            score_isc = 1
            total = 1
        elif "headache" in t and ("sudden" in t or "severe" in t):
            score_hem = 1
            total = 1
        else:
            return ("Unclear/Other", 10, matched)

    scores = {"Ischemic": score_isc, "Hemorrhagic": score_hem, "TIA": score_tia}
    pred = max(scores, key=scores.get)
    conf = int(round((scores[pred] / (total + 1e-6)) * 100))
    if total <= 2:
        conf = min(conf, 40)

    return (pred, conf, matched)

# ---------------- Precautions per stroke type ----------------
PRECAUTIONS = {
    "Ischemic": [
        "Call emergency services immediately and mention suspected stroke.",
        "Note the exact time symptoms began — time is critical for treatment.",
        "Do not give food, drink, or oral medications; keep the person comfortable and still.",
        "If trained, follow FAST steps (Face, Arms, Speech, Time) and prepare for transport.",
        "Keep airway clear and place the person in recovery position if vomiting."
    ],
    "Hemorrhagic": [
        "Call emergency services immediately — this may be life-threatening.",
        "Do NOT give aspirin, anticoagulants, or any blood-thinning medicine.",
        "Keep the person calm, lying flat with head slightly elevated if comfortable.",
        "Note the time of symptom onset and any loss of consciousness or seizures.",
        "Avoid moving the person unnecessarily; wait for professional medical help."
    ],
    "TIA": [
        "Even if symptoms resolved, seek urgent medical evaluation — TIA is a warning sign.",
        "Record the time symptoms started and how long they lasted.",
        "Avoid driving and get someone to accompany the person to medical care.",
        "Discuss with a physician about stroke prevention and possible medication.",
        "Follow-up tests (imaging and cardiology) are usually recommended."
    ],
    "Unclear/Other": [
        "If symptoms are sudden or severe, call emergency services immediately.",
        "Monitor the person closely for progression of weakness, speech trouble or loss of consciousness.",
        "Avoid giving food, drink or medications until assessed by a professional."
    ]
}

# ---------------- UI ----------------
st.title("FAST-Eye — Prototype + Symptom Chat")
st.caption("Face + Speech AI Screening + Symptom Chat")
st.write("---")

# 2-column layout: left inputs and analysis, right symptom chat
# Use wide layout and larger gap for spacing
left, right = st.columns([2, 1], gap="large")

# Left column: inputs and automated analysis
with left:
    st.header("Inputs (Face / Video / Audio)")
    st.info("Upload two images (Neutral + Smile) OR upload a short video (we extract a frame). Then upload a short audio (5–10s).")

    # Face input and video
    col1, col2 = st.columns(2)
    with col1:
        img_option = st.radio("Face input type:", ["Capture neutral + smile (camera)", "Upload images", "Upload video (we extract frame)"], index=1)
        if img_option == "Capture neutral + smile (camera)":
            neutral = st.camera_input("Capture neutral face (look straight, relaxed)")
            smile = st.camera_input("Capture smile face (big smile, show teeth)")
            video_file = None
        elif img_option == "Upload images":
            neutral = st.file_uploader("Upload neutral image", type=["png","jpg","jpeg"], key="neut")
            smile = st.file_uploader("Upload smile image", type=["png","jpg","jpeg"], key="smile")
            video_file = None
        else:
            video_file = st.file_uploader("Upload short video (mp4/mov/avi)", type=["mp4","mov","avi"], key="video")
            neutral = None
            smile = None

    # Audio uploader placed below video/images (in same left area)
    with col2:
        st.write("Speech Input")
        audio_file = st.file_uploader("Upload audio (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"], key="audio")

    st.markdown("### Tips for capturing good samples")
    st.write("- Neutral: straight face, mouth closed, frontal.  - Smile: big smile, show teeth.  - Audio: 5–10s clear sentence in a quiet room.")

    # Run analysis
    if st.button("Run Analysis (Face + Speech)", use_container_width=True):
        face_error = False
        face_scores = []

        # Face processing (video or images)
        if img_option == "Upload video (we extract frame)":
            if video_file:
                video_bytes = video_file.read()
                frame_rgb = extract_middle_frame_from_video_opencv(video_bytes)
                if frame_rgb is None:
                    st.error("Could not extract a usable frame from the video. Try a clearer/shorter video.")
                    face_error = True
                else:
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    pts = extract_landmarks_from_image(frame_bgr)
                    if pts is None:
                        st.error("No face detected in extracted video frame.")
                        face_error = True
                    else:
                        asym = asymmetry_score(pts)
                        face_scores.append(asym)
                        st.image(frame_rgb, caption=f"Extracted frame — asymmetry score: {asym:.1f}", width=360)
            else:
                st.error("Please upload a video file.")
                face_error = True
        else:
            if neutral is None or smile is None:
                st.error("Please provide both neutral and smile images (or upload a video).")
                face_error = True
            else:
                try:
                    n_bytes = neutral.read()
                    n_arr = np.frombuffer(n_bytes, np.uint8)
                    n_img = cv2.imdecode(n_arr, cv2.IMREAD_COLOR)
                except Exception:
                    n_img = None
                try:
                    s_bytes = smile.read()
                    s_arr = np.frombuffer(s_bytes, np.uint8)
                    s_img = cv2.imdecode(s_arr, cv2.IMREAD_COLOR)
                except Exception:
                    s_img = None
                if n_img is None or s_img is None:
                    st.error("Could not load one of the images.")
                    face_error = True
                else:
                    pts_n = extract_landmarks_from_image(n_img)
                    pts_s = extract_landmarks_from_image(s_img)
                    if pts_n is None or pts_s is None:
                        st.error("Face not detected in one of the images.")
                        face_error = True
                    else:
                        asym_n = asymmetry_score(pts_n)
                        asym_s = asymmetry_score(pts_s)
                        asym_diff = asym_s - asym_n
                        asym_final = max(asym_n, asym_s) + max(0, asym_diff)
                        face_scores.append(asym_final)
                        # Show uploaded images medium-size
                        try:
                            st.image(cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB), caption=f"Neutral — asym:{asym_n:.1f}", width=360)
                            st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), caption=f"Smile — asym:{asym_s:.1f}", width=360)
                        except Exception:
                            pass

        # Audio processing (robust) — silent on failure
        audio_clarity = None
        if audio_file:
            try:
                audio_bytes = audio_file.read()
                audio_clarity = load_audio_and_compute_score_from_bytes(audio_bytes)
            except Exception:
                audio_clarity = None

            if audio_clarity is not None:
                st.metric("Speech clarity (0-100)", f"{audio_clarity:.1f}")
            # NOTE: removed warning when audio cannot be processed (silent skip)
        else:
            st.info("No audio provided. Speech analysis skipped.")

        # Evaluate results
        if face_error and audio_clarity is None:
            st.error("No valid inputs to analyze. Provide at least face images/video or audio.")
        else:
            face_score = face_scores[0] if len(face_scores) > 0 else None
            speech_risk = 100 - audio_clarity if audio_clarity is not None else None

            if face_score is not None and speech_risk is not None:
                fused = 0.6 * face_score + 0.4 * speech_risk
            elif face_score is not None:
                fused = face_score
            else:
                fused = speech_risk

            if fused is None:
                st.error("No valid fused score.")
            else:
                if fused < 25:
                    risk = "Low"
                elif fused < 55:
                    risk = "Moderate"
                else:
                    risk = "High"

                st.markdown("## Automated Analysis Result (Face + Speech)")

                # colored box for risk
                if risk == "High":
                    st.markdown(
                        "<div style='background:#ffecec;border-left:6px solid #e23b3b;padding:12px;border-radius:8px;'>"
                        f"<strong style='color:#b30000;'>RISK LEVEL: {risk}</strong> — Combined risk score: <strong>{fused:.1f}</strong>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                elif risk == "Moderate":
                    st.markdown(
                        "<div style='background:#fff7e6;border-left:6px solid #ff8c00;padding:12px;border-radius:8px;'>"
                        f"<strong style='color:#b35900;'>RISK LEVEL: {risk}</strong> — Combined risk score: <strong>{fused:.1f}</strong>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='background:#ecffef;border-left:6px solid #2d9a2d;padding:12px;border-radius:8px;'>"
                        f"<strong style='color:#2d9a2d;'>RISK LEVEL: {risk}</strong> — Combined risk score: <strong>{fused:.1f}</strong>"
                        "</div>",
                        unsafe_allow_html=True
                    )

                findings = []
                if face_score is not None and face_score > 40:
                    findings.append("Facial asymmetry detected (possible facial droop)")
                if speech_risk is not None and speech_risk > 40:
                    findings.append("Speech clarity reduced (possible slurred speech)")

                if not findings:
                    st.success("No immediate signs detected in the provided inputs. (healthy)")
                else:
                    st.error("Potential problems found:")
                    for f in findings:
                        st.write("- " + f)

                st.markdown("### Recommendations & Precautions")
                if risk == "Low":
                    st.write("- No immediate red flags detected from the supplied input.")
                    st.write("- Continue to monitor symptoms. If sudden weakness, numbness, severe headache or confusion occurs, seek emergency care immediately.")
                elif risk == "Moderate":
                    st.write("- Some signs detected; consider contacting a medical professional or urgent care.")
                    st.write("- Re-test with clearer images/audio (quiet room, frontal face, big smile).")
                    st.write("- If face droop, arm weakness, or slurred speech occur, call emergency services.")
                else:
                    st.write("- **High risk detected — possible stroke signs.** Call emergency services now and follow FAST steps.")
                    st.write("- While waiting for help: note symptom onset time, keep airway clear, do NOT give medications orally.")
                st.markdown("---")
                st.caption("Prototype heuristic: this is NOT clinically validated. Use as pre-screening only.")

# Right column: Symptom chat & prediction
with right:
    st.header("Symptom Chat & Prediction")
    st.markdown("Type the symptoms the person is experiencing (e.g., 'sudden weakness on right side, slurred speech, severe headache').")
    symptom_text = st.text_area("Describe symptoms (5–50 words)", height=150)

    if st.button("Predict from Symptoms", use_container_width=True):
        pred_type, conf, matched = predict_stroke_from_symptoms(symptom_text)
        st.markdown("### Symptom-based Prediction")

        if pred_type == "Unclear/Other":
            st.warning("Could not determine a likely stroke type from the provided symptoms. Please provide more details or seek medical advice if symptoms are severe.")
        else:
            # Create HTML card for prediction with color depending on type
            if pred_type == "Hemorrhagic":
                box_html = (
                    "<div style='background:#ffecec;border-left:6px solid #e23b3b;padding:14px;border-radius:8px;'>"
                    f"<div style='font-size:18px;color:#b30000;font-weight:800;'>Type: {pred_type}</div>"
                    f"<div>Confidence: <strong>{conf}%</strong></div>"
                    "</div>"
                )
            elif pred_type == "Ischemic":
                box_html = (
                    "<div style='background:#fff7e6;border-left:6px solid #ff8c00;padding:14px;border-radius:8px;'>"
                    f"<div style='font-size:18px;color:#b35900;font-weight:800;'>Type: {pred_type}</div>"
                    f"<div>Confidence: <strong>{conf}%</strong></div>"
                    "</div>"
                )
            elif pred_type == "TIA":
                box_html = (
                    "<div style='background:#eaf5ff;border-left:6px solid #2b7bd8;padding:14px;border-radius:8px;'>"
                    f"<div style='font-size:18px;color:#1b5bb8;font-weight:800;'>Type: {pred_type}</div>"
                    f"<div>Confidence: <strong>{conf}%</strong></div>"
                    "</div>"
                )
            else:
                box_html = (
                    "<div style='background:#f0f0f0;border-left:6px solid #777;padding:14px;border-radius:8px;'>"
                    f"<div style='font-size:18px;color:#333;font-weight:800;'>Type: {pred_type}</div>"
                    f"<div>Confidence: <strong>{conf}%</strong></div>"
                    "</div>"
                )
            st.markdown(box_html, unsafe_allow_html=True)

        if matched:
            st.write("Matched keywords:", ", ".join(matched))

        st.markdown("### Precautions / Immediate Actions")
        precs = PRECAUTIONS.get(pred_type, PRECAUTIONS["Unclear/Other"])
        # show top 3-5 precautions
        for i, p in enumerate(precs[:5], start=1):
            st.write(f"{i}. {p}")
        st.markdown("---")
        st.caption("Reminder: This symptom-based classifier is heuristic and not a substitute for professional medical evaluation.")

    st.markdown("### Combined Recommendation")
    if st.button("Combined Suggestion (if you ran analysis above)", use_container_width=True):
        pred_type, conf, matched = predict_stroke_from_symptoms(symptom_text)
        st.write("This combined suggestion merges symptom-based inference with your earlier face/audio test (if run).")
        if pred_type == "Hemorrhagic":
            st.error("Combined Suggestion: Possible Hemorrhagic stroke signs. Call emergency services immediately.")
        elif pred_type == "Ischemic":
            st.warning("Combined Suggestion: Possible Ischemic stroke signs. Urgent medical evaluation needed.")
        elif pred_type == "TIA":
            st.info("Combined Suggestion: Symptoms suggest a TIA (mini-stroke). Seek urgent medical attention; this is a warning sign.")
        else:
            st.info("Combined Suggestion: Symptoms unclear. If you have any sudden or severe symptoms, call emergency services.")

st.write("---")
st.caption("FAST-Eye Prototype with symptom chat — NOT clinically validated. For demo/presentation use only. Always contact professional medical services for emergencies.")

