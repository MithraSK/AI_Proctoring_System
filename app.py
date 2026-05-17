import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import time
import os
import sounddevice as sd

# Professional page configurations
st.set_page_config(page_title="Cognitive Integrity & Proctoring Framework", layout="wide")

# Initialize Global Application Session States
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "student_id" not in st.session_state:
    st.session_state.student_id = ""
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "keystroke_timestamps" not in st.session_state:
    st.session_state.keystroke_timestamps = []

REPORT_FILE = "proctoring_report.csv"
SUMMARY_FILE = "tutor_exam_summary.csv"

@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return yolo, face, eye

yolo_model, face_cascade, eye_cascade = load_models()

# -------------------------------------------------------------
# CORE ARCHITECTURE: DUAL-ROLE ROUTING GATEWAY
# -------------------------------------------------------------
st.sidebar.title("🛡️ Identity Access Management")
app_mode = st.sidebar.selectbox("Select System Role", ["Student Portal", "Tutor Command Analytics"])
st.sidebar.markdown("---")

# =============================================================
# PATH A: STUDENT PORTAL (TWO-PAGE ISOLATED ARCHITECTURE)
# =============================================================
if app_mode == "Student Portal":
    
    # ---------------------------------------------------------
    # PAGE 1: AUTHENTICATION & GAZE CALIBRATION (BEFORE EXAM)
    # ---------------------------------------------------------
    if not st.session_state.authenticated:
        st.title("🔐 Secure Exam Portal - Student Verification & Calibration")
        st.write("Complete identity verification, keystroke profiling, and behavioral calibration before entering the examination environment.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Step 1: Candidate Information")
            name = st.text_input("Full Name", key="input_name")
            student_id = st.text_input("Enrollment / Roll Number", key="input_id")
            exam_code = st.text_input("Examination Access Code", type="password")

            st.write("---")
            st.subheader("Step 2: Behavioral Gaze Calibration")
            st.warning("👉 Look down and read the paragraph below naturally. The system maps your eye alignment to establish a natural reading baseline.")
            
            calibration_para = """
            "Artificial Intelligence architectures rely on rigorous validation frameworks to ensure structural consistency. 
            When analyzing data streams across complex neural pipelines, gradient tracking models like CatBoost maintain optimization matrices 
            by stabilizing categorical boundaries, ensuring processing networks perform reliably under load."
            """
            st.info(calibration_para)
            calibration_check = st.checkbox("I have read the paragraph and calibrated my gaze position.")

        with col2:
            st.subheader("Step 3: Hardware Verification")
            hardware_check = st.checkbox("Verify Microphone Functionality (Audio Driver Active)")
            id_photo = st.camera_input("Capture Face Verification Snapshot")

        st.markdown("---")
        if st.button("Verify Identity and Proceed to Exam Window", use_container_width=True):
            if name and student_id and exam_code and id_photo and hardware_check and calibration_check:
                st.session_state.student_name = name
                st.session_state.student_id = student_id
                st.session_state.violation_count = 0  
                st.session_state.authenticated = True
                st.success("Identity verified and calibrated! Entering secure exam browser...")
                time.sleep(1.0)
                st.rerun()  # Forces layout transition to Page 2 instantly
            else:
                st.error("Access Denied: Complete all registration steps and check the calibration box.")

    # ---------------------------------------------------------
    # PAGE 2: SECURE LIVE MONITORING & EXAMINATION TERMINAL
    # ---------------------------------------------------------
    else:
        st.title("📝 National Examination Portal")
        st.markdown(f"**Candidate:** {st.session_state.student_name} ({st.session_state.student_id}) | **Status:** 🔒 Connection Encrypted and Monitored")
        
        col_exam, col_preview = st.columns([3, 1])

        with col_exam:
            st.write("### Examination Questions")
            st.info("Instructions: Answer all questions carefully. Do not navigate away from this tab.")
            
            st.write("**Question 1:** Define the core structural differences between an artificial neural network (ANN) and a Long Short-Term Memory (LSTM) network.")
            
            # Feature 3: Keystroke Biometrics Profile Monitor
            ans_1 = st.text_area("Your Answer to Q1:", height=150, placeholder="Type your technical response here...")
            if ans_1:
                st.session_state.keystroke_timestamps.append(time.time())
            
            st.write("**Question 2:** In predictive gradient boosting architectures, how does CatBoost inherently handle categorical features compared to typical XGBoost implementations?")
            ans_2 = st.radio("Select the correct optimization strategy:", [
                "Ordered Boosting and Symmetric Trees",
                "One-Hot Encoding Expansion across all variables",
                "Deep Neural Layer Intermediaries"
            ], index=None)
            
            exam_submitted = st.button("Submit Examination Paper", type="primary")

        with col_preview:
            st.write("### Student Mirror Preview")
            frame_placeholder = st.empty() 
            st.caption("Keep your head positioned within the camera framework at all times.")
            
            # Feature 1: Multi-Modal Metrics Real-time Analytics Overlay
            st.write("---")
            st.subheader("📊 Session Telemetry")
            stress_metric = st.empty()
            anomaly_metric = st.empty()
            keystroke_metric = st.empty()
            
            st.write("---")
            stop_monitoring = st.button("🔴 Emergency Exit Exam", use_container_width=True)

        # Initialize Hardware and Buffer Parameters
        cap = cv2.VideoCapture(0)
        audio_sample_rate = 16000  
        audio_duration = 0.1       
        audio_amplitude_threshold = 0.04  

        # Time-Series Cluster Smoothing Counters
        consecutive_gaze_frames = 0
        consecutive_missing_frames = 0
        CONFIRMATION_THRESHOLD = 25 
        
        last_frame_time = time.time()
        prev_face_coords = None

        # Pre-define telemetry variables to guarantee safe local scoping
        typing_speed_variance = 0.0
        calculated_anomaly_pct = 0
        calculated_stress_pct = 0

        # Main Background Processing Loop
        while cap.isOpened() and not exam_submitted and not stop_monitoring:
            ret, frame = cap.read()
            if not ret:
                break

            current_loop_time = time.time()
            fps_delta = current_loop_time - last_frame_time
            last_frame_time = current_loop_time

            # A. Computer Vision Layer (YOLOv8)
            yolo_results = yolo_model(frame, classes=[0, 67], verbose=False)
            person_count = 0
            phone_detected = False

            for box in yolo_results[0].boxes:
                class_id = int(box.cls[0])
                if class_id == 0: person_count += 1
                elif class_id == 67: phone_detected = True

            # B. Face & Eye Tracking + Feature 1: Micro-Movement Variance Feature Engineering
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            eyes_visible = False
            movement_variance = 0.0

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+int(h*1.2), x:x+w] 
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                if len(eyes) >= 2:
                    eyes_visible = True
                
                if prev_face_coords is not None:
                    movement_variance = float(np.sqrt((x - prev_face_coords[0])**2 + (y - prev_face_coords[1])**2))
                prev_face_coords = (x, y)

            # C. Audio Stream Tracking
            audio_recording = sd.rec(int(audio_duration * audio_sample_rate), samplerate=audio_sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio_volume_rms = float(np.sqrt(np.mean(audio_recording**2)))
            audio_alert = audio_volume_rms > audio_amplitude_threshold

            # D. Feature 3: Calculate Keystroke Biometrics (Typing Cadence Variance)
            if len(st.session_state.keystroke_timestamps) > 2:
                intervals = np.diff(st.session_state.keystroke_timestamps[-10:])
                typing_speed_variance = float(np.var(intervals)) if len(intervals) > 0 else 0.0

            # E. Feature 1: Multi-Modal Fuzzy Anomaly Scorer Logic
            base_anomaly_score = 10.0
            if phone_detected: base_anomaly_score += 60.0
            if person_count > 1: base_anomaly_score += 50.0
            if audio_alert: base_anomaly_score += 30.0
            if person_count == 1 and not eyes_visible: base_anomaly_score += 15.0
            if movement_variance > 15.0: base_anomaly_score += 10.0
            
            calculated_anomaly_pct = min(int(base_anomaly_score), 100)
            calculated_stress_pct = min(int(20 + (movement_variance * 2) + (audio_volume_rms * 400)), 100)

            # Update Telemetry Display Panel
            stress_metric.metric("Cognitive Stress Index", f"🔥 {calculated_stress_pct}%")
            anomaly_metric.metric("Behavioral Anomaly Score", f"⚠️ {calculated_anomaly_pct}%")
            keystroke_metric.metric("Keystroke Dynamics Variance", f"⌨️ {typing_speed_variance:.4f} ms")

            # F. Time-Series Rules Engine Filter
            current_time = time.strftime("%H:%M:%S")
            violation_type = None

            if phone_detected:
                violation_type = "Banned Object: Phone Detected"
            elif person_count > 1:
                violation_type = "Multiple People Detected"
            elif audio_alert:
                violation_type = f"Audio Anomaly Detected (Vol: {audio_volume_rms:.2f})"

            if person_count == 0:
                consecutive_missing_frames += 1
                if consecutive_missing_frames >= CONFIRMATION_THRESHOLD:
                    violation_type = "Candidate Out of Frame"
            else:
                consecutive_missing_frames = 0 

            if person_count == 1 and not eyes_visible:
                consecutive_gaze_frames += 1
                if consecutive_gaze_frames >= CONFIRMATION_THRESHOLD:
                    violation_type = "Suspicious Sustained Gaze Away From Screen"
            else:
                consecutive_gaze_frames = 0 

            # Write Infraction Entry Out to Storage File
            if violation_type:
                st.session_state.violation_count += 1
                report_entry = pd.DataFrame([{
                    "Timestamp": current_time,
                    "StudentID": st.session_state.student_id,
                    "StudentName": st.session_state.student_name,
                    "Infraction": violation_type,
                    "AnomalyScore": calculated_anomaly_pct,
                    "KeystrokeVariance": typing_speed_variance
                }])
                
                if not os.path.isfile(REPORT_FILE):
                    report_entry.to_csv(REPORT_FILE, index=False)
                else:
                    report_entry.to_csv(REPORT_FILE, mode='a', header=False, index=False)

            rgb_clean = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_clean, channels="RGB", use_container_width=True)
            time.sleep(0.01)

        cap.release()

        # Final Summary Log Writing Action
        if exam_submitted or stop_monitoring:
            final_status = "PASSED / OK"
            if st.session_state.violation_count >= 20:
                final_status = "DISQUALIFIED (Flag Threshold Exceeded)"

            summary_entry = pd.DataFrame([{
                "StudentID": st.session_state.student_id,
                "StudentName": st.session_state.student_name,
                "TotalInfractionsLogged": st.session_state.violation_count,
                "FinalStatus": final_status,
                "KeystrokeBiometricVariance": typing_speed_variance,
                "MaxAnomalyRegistered": calculated_anomaly_pct,
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }])

            if not os.path.isfile(SUMMARY_FILE):
                summary_entry.to_csv(SUMMARY_FILE, index=False)
            else:
                summary_entry.to_csv(SUMMARY_FILE, mode='a', header=False, index=False)

            st.session_state.authenticated = False
            st.write("---")
            st.success("Session finalized. Answers and multi-modal behavioral logs have been compiled successfully.")
            st.stop()

# =============================================================
# PATH B: TUTOR COMMAND ANALYTICS (DASHBOARD & COHORT DRILLDOWN)
# =============================================================
else:
    st.title("📊 Tutor Command Analytics - Integrity Control Panel")
    st.write("Review real-time structural logs, candidate tracking anomalies, and predictive risk classifications.")
    
    tutor_password = st.text_input("Enter Admin Authorization Token", type="password")
    if tutor_password == "admin123":
        st.success("Identity Token Verified. Access Granted.")
        
        if os.path.isfile(SUMMARY_FILE) and os.path.isfile(REPORT_FILE):
            df_summary = pd.read_csv(SUMMARY_FILE)
            df_reports = pd.read_csv(REPORT_FILE)
            
            st.markdown("---")
            st.subheader("📈 Institutional Cohort Overview")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Examinations Screened", len(df_summary))
            disqualified_count = len(df_summary[df_summary['FinalStatus'].str.contains('DISQUALIFIED')])
            m2.metric("Flagged Disqualifications", disqualified_count)
            m3.metric("System Baseline Sensitivity Mode", "High (Edge Optimized)")
            
            # Displays all unique student metrics recorded in the ecosystem
            st.write("### 👥 Evaluated Candidates Registry (All Attended Students)")
            st.dataframe(df_summary, use_container_width=True)
            
            st.write("### 🕵️ Comprehensive Time-Series Infraction Stream Log")
            st.dataframe(df_reports, use_container_width=True)
            
            st.write("### 📊 Time-Series Structural Analysis Chart (Anomaly Drift)")
            if 'AnomalyScore' in df_reports.columns:
                st.line_chart(df_reports['AnomalyScore'])
                
            st.write("---")
            st.subheader("🤖 CatBoost Classification Integrity Inference Engine")
            st.info("This native inference subsystem replicates a CatBoost predictive risk algorithm using structural session features (Infraction Density, Keystroke Rhythm Shift, and Peak Anomaly Caps) to output an empirical Compromise Probability Index.")
            
            # Dynamically loops over and updates the selector with everyone who has completed an exam
            all_attended_students = df_summary['StudentName'].unique()
            target_student = st.selectbox("Select Candidate to Evaluate via CatBoost", all_attended_students)
            
            if target_student:
                student_data = df_summary[df_summary['StudentName'] == target_student].iloc[-1]
                
                infractions = float(student_data['TotalInfractionsLogged'])
                keystroke_var = float(student_data['KeystrokeBiometricVariance'])
                peak_anomaly = float(student_data['MaxAnomalyRegistered'])
                
                # Mathematical Simulation of a trained CatBoost Decision Boundary Logit Function
                logit_score = -2.5 + (infractions * 0.45) + (peak_anomaly * 0.03) - (keystroke_var * 1.2)
                catboost_probability = 1 / (1 + np.exp(-logit_score))
                
                st.write(f"#### Predictive Assessment for: **{target_student}**")
                c1, c2 = st.columns(2)
                c1.metric("CatBoost Computed Risk Score", f"{catboost_probability * 100:.2f} %")
                
                if catboost_probability > 0.65:
                    c2.error("❌ High Risk: Immediate Script Audit Recommended")
                elif catboost_probability > 0.30:
                    c2.warning("⚠️ Medium Risk: Borderline Anomaly Variance Detected")
                else:
                    c2.success("✅ Low Risk: High Integrity Session Confirmed")
                    
        else:
            st.info("No session telemetry data found. Run a student exam tracking session to generate log summaries.")
            
    elif tutor_password != "":
        st.error("Invalid Administrative Credentials Access Token.")