import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import time
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from datetime import datetime, timedelta
from PIL import Image
from ultralytics import YOLO
from base64 import b64encode

# NEW: Import audio analysis dependencies
import numpy as np
import librosa
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# [KEEP ALL EXISTING CODE FROM wildeye_app.py UP TO THE SIDEBAR SECTION]
# Set page configuration
st.set_page_config(
    page_title="WildEye - Animal Behavior Detection",
    page_icon="🐘",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Set the full page background */
    body, .stApp {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
    }

    /* Header Styles */
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #4CAF50 !important;
        text-align: center;
        margin-bottom: 20px;
    }

    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #81C784 !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Sidebar, Cards, and Inputs */
    .card, .stats-card, .stSidebar, .stTextInput, .stSelectbox, .stNumberInput {
        background-color: #1E1E1E !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #E0E0E0 !important;
        border: 1px solid #4CAF50 !important;
    }

    /* Fix unreadable text issue (applies to all text in sidebar, main content, and inputs) */
    .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label, p, li, span {
        color: #E0E0E0 !important;
    }

    /* Sidebar text color */
    .stSidebar {
        color: #E0E0E0 !important;
    }

    /* Fix Latitude & Longitude input text */
    input[type="number"], input[type="text"] {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border: 1px solid #4CAF50 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 15px;
        font-size: 16px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #388E3C !important;
    }

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #76FF03 !important;
    }
</style>
""", unsafe_allow_html=True)



# App header
st.markdown('<div class="main-header">🐘 WildEye: Animal Behavior Detection</div>', unsafe_allow_html=True)

# Initialize session state for storing data
if 'detected_animals' not in st.session_state:
    st.session_state.detected_animals = []
    
if 'behavior_data' not in st.session_state:
    st.session_state.behavior_data = {'calm': 0, 'aggressive': 0}
    
if 'detection_locations' not in st.session_state:
    st.session_state.detection_locations = []
    
if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = []
    
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None



# NEW: Add audio analysis configuration to sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">Upload & Configure</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
    
    # Model selection
    model_option = st.selectbox(
        "Select Detection Model",
        ["YOLO Custom Model", "Elephant Detection Model", "Wildlife Behavior Model"]
    )
    
    # Detection options
    detection_threshold = st.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5,
        step=0.05
    )
    
    # Location input
    st.markdown("### Detection Location")
    location_lat = st.number_input("Latitude", value=8.5241, format="%.4f")
    location_lon = st.number_input("Longitude", value=76.9366, format="%.4f")
    location_name = st.text_input("Location Name", "Wildlife Sanctuary")

    # Define behavior types for the model
    behavior_types = {
        0: {"name": "Calm", "category": "calm"},
        1: {"name": "Eating", "category": "calm"},
        2: {"name": "Walking", "category": "calm"},
        3: {"name": "Charging", "category": "aggressive"},
        4: {"name": "Fighting", "category": "aggressive"}
    }

    
    # Add audio upload option
    st.markdown('<div class="sub-header">Audio Behavior Analysis</div>', unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'ogg'])
    
    # Audio analysis configuration
    audio_segment_length = st.slider(
        "Audio Segment Length (seconds)", 
        min_value=1, 
        max_value=10, 
        value=5,
        step=1
    )
    audio_confidence_threshold = st.slider(
        "Audio Detection Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.6,
        step=0.05
    )

# NEW: Audio Behavior Analysis Class (Copied from paste.txt with Streamlit-specific modifications)
class ElephantBehaviorAnalyzer:
    def __init__(self, audio_file, segment_length=5, sr=22050, max_pad_length=100, confidence_threshold=0.6):
        self.audio_file = audio_file
        self.segment_length = segment_length
        self.sr = sr
        self.max_pad_length = max_pad_length
        self.confidence_threshold = confidence_threshold
        
        # Load audio
        self.y, self.sr = librosa.load(audio_file, sr=self.sr)
        
        # Initialize results storage
        self.predictions = []
        self.timestamps = []
        self.confidences = []
        
    def extract_features_segmented(self):
        """Extract MFCC features from audio segments"""
        if len(self.y) == 0:
            st.error("❌ No audio detected!")
            return None

        segment_samples = self.segment_length * self.sr
        total_segments = len(self.y) // segment_samples

        features_list = []
        for i in range(total_segments):
            segment = self.y[i * segment_samples: (i + 1) * segment_samples]
            mfccs = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13)

            # Pad or truncate features to maintain a fixed length
            if mfccs.shape[1] < self.max_pad_length:
                pad_width = self.max_pad_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_pad_length]

            features_list.append(mfccs.flatten())

        return features_list

    def predict_continuous_behavior(self):
        """Predict behavior for each audio segment"""
        # ASSUMPTION: Models are pre-loaded
        model = joblib.load("elephant_behavior_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        
        features_list = self.extract_features_segmented()
        
        if features_list is None:
            st.error("❌ No valid audio detected!")
            return
        
        for i, features in enumerate(features_list):
            features = np.array(features).reshape(1, -1)  # Reshape for model input
            features_scaled = scaler.transform(features)  # Scale features

            # Get prediction and confidence score
            prediction = model.predict(features_scaled)[0]
            confidence = np.max(model.predict_proba(features_scaled))

            # Store results
            start_time = i * self.segment_length
            end_time = (i + 1) * self.segment_length
            
            if confidence < self.confidence_threshold:
                behavior = "No Detection"
            else:
                behavior = label_encoder.inverse_transform([prediction])[0]
            
            self.predictions.append(behavior)
            self.timestamps.append((start_time, end_time))
            self.confidences.append(confidence)
        
        return self
    
    def generate_visualizations(self, output_dir=None):
        """Generate comprehensive visualizations for Streamlit"""
        # Create DataFrame for easier analysis
        df = pd.DataFrame({
            'Start Time (s)': [t[0] for t in self.timestamps],
            'End Time (s)': [t[1] for t in self.timestamps],
            'Behavior': self.predictions,
            'Confidence': self.confidences
        })
        
        # Color palette for visualizations
        color_palette = {
            'Aggressive': '#FF6B6B',
            'Calm': '#4ECDC4', 
            'Hunger': '#FFA726', 
            'No Detection': '#BDBDBD'
        }
        
        # Create Plotly visualizations for Streamlit
        
        # 1. Behavior Distribution Pie Chart
        behavior_counts = df['Behavior'].value_counts()
        fig_pie = px.pie(
            names=behavior_counts.index, 
            values=behavior_counts.values,
            title='Audio Behavior Distribution',
            color=behavior_counts.index,
            color_discrete_map=color_palette
        )
        
        # 2. Behavior Timeline
        behavior_map = {b: i for i, b in enumerate(df['Behavior'].unique())}
        df['Behavior_Numeric'] = df['Behavior'].map(behavior_map)
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=df['Start Time (s)'], 
            y=df['Behavior_Numeric'],
            mode='lines+markers',
            name='Behavior Timeline',
            line=dict(color='#2C3E50', width=2),
            marker=dict(size=8)
        ))
        fig_timeline.update_layout(
            title='Audio Behavior Timeline',
            xaxis_title='Time (seconds)',
            yaxis_title='Behavior',
            yaxis=dict(
                tickmode='array',
                tickvals=list(behavior_map.values()),
                ticktext=list(behavior_map.keys())
            )
        )
        
        # 3. Confidence Boxplot
        fig_confidence = go.Figure()
        for behavior in behavior_counts.index:
            confidence_data = df[df['Behavior'] == behavior]['Confidence']
            fig_confidence.add_trace(go.Box(
                y=confidence_data,
                name=behavior,
                marker_color=color_palette.get(behavior, 'gray')
            ))
        fig_confidence.update_layout(
            title='Confidence Levels by Behavior',
            yaxis_title='Confidence Score'
        )
        
        # 4. Waveform Visualization
        time_axis = np.linspace(0, len(self.y)/self.sr, num=len(self.y))
        fig_waveform = go.Figure()
        fig_waveform.add_trace(go.Scatter(
            x=time_axis, 
            y=self.y, 
            mode='lines', 
            name='Waveform',
            line=dict(color='#2C3E50', width=1)
        ))
        
        # Add colored segments for behaviors
        for (start, end), behavior in zip(self.timestamps, self.predictions):
            fig_waveform.add_shape(type="rect",
                x0=start, x1=end, y0=min(self.y), y1=max(self.y),
                fillcolor=color_palette.get(behavior, 'gray'),
                opacity=0.2,
                layer='below',
                line_width=0,
            )
        
        fig_waveform.update_layout(
            title='Audio Waveform with Behavior Segments',
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitude'
        )
        
        return {
            'pie_chart': fig_pie,
            'timeline': fig_timeline,
            'confidence_boxplot': fig_confidence,
            'waveform': fig_waveform,
            'summary_df': df
        }

# [KEEP ALL EXISTING CODE FROM THE MAIN SECTION OF wildeye_app.py]

# Modify main processing section to include audio analysis
if uploaded_file is not None or uploaded_audio is not None:
    
    # Existing video processing logic...
    
    # New audio processing section
    if uploaded_audio is not None:
        st.markdown('<div class="sub-header">Audio Behavior Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing audio..."):
            # Save uploaded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(uploaded_audio.read())
                audio_path = temp_audio.name
            
            # Perform audio analysis
            try:
                analyzer = ElephantBehaviorAnalyzer(
                    audio_path, 
                    segment_length=audio_segment_length, 
                    confidence_threshold=audio_confidence_threshold
                )
                analyzer.predict_continuous_behavior()
                audio_results = analyzer.generate_visualizations()
                
                # Display audio analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(audio_results['pie_chart'], use_container_width=True)
                    st.plotly_chart(audio_results['confidence_boxplot'], use_container_width=True)
                
                with col2:
                    st.plotly_chart(audio_results['timeline'], use_container_width=True)
                    st.plotly_chart(audio_results['waveform'], use_container_width=True)
                
                # Summary table
                st.markdown("### Audio Behavior Summary")
                st.dataframe(audio_results['summary_df'])
                
                # Export options for audio data
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    if st.button("Export Audio Behavior Data (CSV)"):
                        csv = audio_results['summary_df'].to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="wildeye_audio_behaviors.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error in audio analysis: {e}")

# Define processing function
def process_video(video_file, confidence_threshold):
    # Create a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    
    # Load model (for demonstration, we'll use a placeholder path)
    model_path = "last_ED.pt"
    try:
        model = YOLO(model_path)
    except:
        # Fallback to YOLO default model if custom model not found
        st.warning("Custom model not found. Using YOLO default model.")
        model = YOLO("yolov8n.pt")
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video file
    temp_output_path = os.path.join(tempfile.gettempdir(), f"processed_video_{int(time.time())}.mp4")
    
    # Choose the appropriate fourcc codec based on OS
    if os.name == 'nt':  # Windows
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:  # Linux/Mac
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Colors for masks
    COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    # Progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    # Initialize tracking variables
    frame_number = 0
    animal_counts = {}
    behavior_counts = {"calm": 0, "aggressive": 0}
    timeline_detections = []
    
    # Stats display area
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    total_animals = stats_col1.empty()
    calm_count = stats_col2.empty()
    aggressive_count = stats_col3.empty()
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = int(frame_number / frame_count * 100)
        progress_bar.progress(progress)
        
        # Run YOLO detection on frame
        results = model(frame, conf=confidence_threshold)
        
        # Track detections for this frame
        frame_detections = {}
        timestamp = datetime.now() - timedelta(seconds=(frame_count-frame_number)/fps)
        
        # Draw results on frame
        for result in results:
            if result.masks is not None:
                for i, mask in enumerate(result.masks.xy):
                    mask = mask.astype(np.int32)
                    color = COLORS[i % len(COLORS)]
                    cv2.polylines(frame, [mask], isClosed=True, color=color, thickness=2)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [mask], color=color)
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Draw bounding boxes and labels
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = box.conf[0]
                
                # For demo, simulate behavior detection based on motion/size
                # In real application, this would come from the model
                animal_type = model.names[class_id] if class_id in model.names else f"Class_{class_id}"
                
                # Simulate behavior detection (in real app, this would be from model)
                behavior_id = frame_number % 5  # Cyclic behavior for demo
                behavior_info = behavior_types[behavior_id]
                
                # Update behavior counts
                behavior_counts[behavior_info["category"]] += 1
                
                # Create label with behavior
                label = f"{animal_type}: {behavior_info['name']} ({conf:.2f})"
                
                # Pick color based on behavior (red for aggressive, green for calm)
                box_color = (0, 0, 255) if behavior_info["category"] == "aggressive" else (0, 255, 0)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # Update animal counts
                if animal_type not in animal_counts:
                    animal_counts[animal_type] = 0
                animal_counts[animal_type] += 1
                
                # Add to frame detections
                if animal_type not in frame_detections:
                    frame_detections[animal_type] = 0
                frame_detections[animal_type] += 1
        
        # Add frame data to timeline
        if len(frame_detections) > 0:
            timeline_detections.append({
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "frame": frame_number,
                "detections": frame_detections,
                "behaviors": {"calm": behavior_counts["calm"], "aggressive": behavior_counts["aggressive"]}
            })
        
        # Update stats
        total_detected = sum(animal_counts.values())
        total_animals.markdown(f"<div class='stats-card'><h3>Total Detections</h3><h2>{total_detected}</h2></div>", unsafe_allow_html=True)
        calm_count.markdown(f"<div class='stats-card'><h3>Calm Behaviors</h3><h2>{behavior_counts['calm']}</h2></div>", unsafe_allow_html=True)
        aggressive_count.markdown(f"<div class='stats-card'><h3>Aggressive Behaviors</h3><h2>{behavior_counts['aggressive']}</h2></div>", unsafe_allow_html=True)
        
        # Write frame to output
        out.write(frame)
        frame_number += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Remove progress bar
    progress_bar.empty()
    
    # Update session state
    st.session_state.detected_animals = [{"name": k, "count": v} for k, v in animal_counts.items()]
    st.session_state.behavior_data = behavior_counts
    
    # Add location to detection locations
    st.session_state.detection_locations.append({
        "name": location_name,
        "lat": location_lat,
        "lon": location_lon,
        "animals": total_detected,
        "behaviors": behavior_counts
    })
    
    # Update timeline data
    st.session_state.timeline_data = timeline_detections
    
    # Store the path to processed video in session state
    st.session_state.processed_video_path = temp_output_path
    
    # Convert video to MP4 in a format suitable for streaming in browsers
    try:
        final_output_path = os.path.join(tempfile.gettempdir(), f"web_playable_{int(time.time())}.mp4")
        # Use more web-compatible parameters for ffmpeg
        os.system(f"ffmpeg -i {temp_output_path} -vcodec h264 -acodec aac -strict -2 -f mp4 {final_output_path}")
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            st.session_state.processed_video_path = final_output_path
        else:
            st.warning("Video conversion failed. Using original format which may not play in all browsers.")
    except Exception as e:
        st.error(f"Error during video conversion: {e}")
        # If conversion fails, we'll use the original output
        pass
    
    # Add video information for debugging
    try:
        cap = cv2.VideoCapture(st.session_state.processed_video_path)
        video_info = {
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        st.sidebar.write("Video info:", video_info)
    except Exception as e:
        st.sidebar.error(f"Error checking video format: {e}")
    
    # Return path to the processed video
    return st.session_state.processed_video_path

# Main dashboard layout
if uploaded_file is not None:
    # Process button
    if st.button("Process Video"):
        with st.spinner("Processing video... This may take a while."):
            output_file = process_video(uploaded_file, detection_threshold)
            
            # Store video path in session state
            st.session_state.processed_video_path = output_file
    
    # Display processed video if available
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        st.markdown('<div class="sub-header">Processed Video</div>', unsafe_allow_html=True)
        
        # Create a container for the video
        video_container = st.container()
        with video_container:
            try:
                # Display video with controls
                with open(st.session_state.processed_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"Error streaming video: {e}")
                # Provide immediate download option if streaming fails
                with open(st.session_state.processed_video_path, 'rb') as file:
                    st.download_button(
                        label="Download Video (Streaming failed)",
                        data=file.read(),
                        file_name="wildeye_processed_video.mp4",
                        mime="video/mp4"
                    )
        
        # Alternative video player option
        if st.checkbox("Video not playing? Try alternative player"):
            try:
                video_file = open(st.session_state.processed_video_path, 'rb')
                video_bytes = video_file.read()
                video_b64 = b64encode(video_bytes).decode()
                
                html_code = f"""
                <video width="100%" controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                st.markdown(html_code, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Alternative player failed: {e}")
        
        # Dashboard sections
        st.markdown('<div class="sub-header">Detection Dashboard</div>', unsafe_allow_html=True)
        
        # Create multi-column layout for visualizations
        col1, col2 = st.columns(2)
        
        # Pie chart for behavior
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Behavior Distribution")
            
            behavior_df = pd.DataFrame({
                "Behavior": ["Calm", "Aggressive"],
                "Count": [st.session_state.behavior_data["calm"], st.session_state.behavior_data["aggressive"]]
            })
            
            fig = px.pie(
                behavior_df, 
                values='Count', 
                names='Behavior',
                color='Behavior',
                color_discrete_map={'Calm': '#4CAF50', 'Aggressive': '#F44336'},
                hole=0.4
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bar chart for animal counts
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Animal Detections")
            
            animal_df = pd.DataFrame(st.session_state.detected_animals)
            if not animal_df.empty:
                # Filter for elephants or use all animals if none
                if "elephant" in animal_df['name'].str.lower().values:
                    filtered_df = animal_df[animal_df['name'].str.lower().str.contains('elephant')]
                else:
                    filtered_df = animal_df
                
                fig = px.bar(
                    filtered_df,
                    x='name',
                    y='count',
                    color='name',
                    labels={'name': 'Animal Type', 'count': 'Count'},
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No animals detected in this video.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Map showing detection locations
        st.markdown('<div class="sub-header">Detection Locations</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Create map
        m = folium.Map(location=[location_lat, location_lon], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)
        
        for loc in st.session_state.detection_locations:
            # Create popup text
            popup_text = f"""
            <b>{loc['name']}</b><br>
            Animals detected: {loc['animals']}<br>
            Calm behaviors: {loc['behaviors']['calm']}<br>
            Aggressive behaviors: {loc['behaviors']['aggressive']}
            """
            
            # Add marker with popup
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='green', icon='paw', prefix='fa')
            ).add_to(marker_cluster)
        
        # Display map
        folium_static(m, width=1100, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Timeline distribution
        st.markdown('<div class="sub-header">Timeline Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if len(st.session_state.timeline_data) > 0:
            # Create dataframe from timeline data
            timeline_rows = []
            for entry in st.session_state.timeline_data:
                row = {
                    "timestamp": entry["timestamp"],
                    "frame": entry["frame"],
                    "calm": entry["behaviors"]["calm"],
                    "aggressive": entry["behaviors"]["aggressive"]
                }
                
                # Add animal counts
                for animal, count in entry["detections"].items():
                    if animal not in row:
                        row[animal] = 0
                    row[animal] += count
                
                timeline_rows.append(row)
            
            timeline_df = pd.DataFrame(timeline_rows)
            
            # Create line chart
            fig = go.Figure()
            
            # Add calm behaviors
            fig.add_trace(go.Scatter(
                x=timeline_df["timestamp"],
                y=timeline_df["calm"],
                mode='lines+markers',
                name='Calm Behaviors',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=6)
            ))
            
            # Add aggressive behaviors
            fig.add_trace(go.Scatter(
                x=timeline_df["timestamp"],
                y=timeline_df["aggressive"],
                mode='lines+markers',
                name='Aggressive Behaviors',
                line=dict(color='#F44336', width=2),
                marker=dict(size=6)
            ))
            
            # Add animal counts if available
            for animal in [col for col in timeline_df.columns if col not in ["timestamp", "frame", "calm", "aggressive"]]:
                fig.add_trace(go.Scatter(
                    x=timeline_df["timestamp"],
                    y=timeline_df[animal],
                    mode='lines+markers',
                    name=f'{animal} Count',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            # Update layout
            fig.update_layout(
                title="Behavior and Detection Timeline",
                xaxis_title="Time",
                yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=70, b=40),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options
        st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            if st.button("Export Detection Data (CSV)"):
                # Create CSV for detections
                csv = pd.DataFrame(st.session_state.detected_animals).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="wildeye_detections.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("Export Behavior Data (CSV)"):
                # Create CSV for behaviors
                behavior_df = pd.DataFrame({
                    "Behavior": ["Calm", "Aggressive"],
                    "Count": [st.session_state.behavior_data["calm"], st.session_state.behavior_data["aggressive"]]
                })
                csv = behavior_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="wildeye_behaviors.csv",
                    mime="text/csv"
                )
        
        with export_col3:
            if st.button("Export Timeline Data (CSV)"):
                if len(st.session_state.timeline_data) > 0:
                    # Create dataframe from timeline data
                    timeline_rows = []
                    for entry in st.session_state.timeline_data:
                        row = {
                            "timestamp": entry["timestamp"],
                            "frame": entry["frame"],
                            "calm": entry["behaviors"]["calm"],
                            "aggressive": entry["behaviors"]["aggressive"]
                        }
                        
                        # Add animal counts
                        for animal, count in entry["detections"].items():
                            if animal not in row:
                                row[animal] = 0
                            row[animal] += count
                        
                        timeline_rows.append(row)
                    
                    timeline_df = pd.DataFrame(timeline_rows)
                    csv = timeline_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="wildeye_timeline.csv",
                        mime="text/csv"
                    )
        
        with export_col4:
            if st.button("Download Processed Video"):
                with open(st.session_state.processed_video_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label="Download Video",
                        data=video_bytes,
                        file_name="wildeye_processed_video.mp4",
                        mime="video/mp4"
                    )
        
        # Summary of findings
        st.markdown('<div class="sub-header">Analysis Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Generate summary text
        total_animals = sum(item["count"] for item in st.session_state.detected_animals)
        if st.session_state.behavior_data["calm"] + st.session_state.behavior_data["aggressive"] > 0:
            behavior_ratio = st.session_state.behavior_data["aggressive"] / (st.session_state.behavior_data["calm"] + st.session_state.behavior_data["aggressive"])
        else:
            behavior_ratio = 0
        
        st.markdown(f"""
        ### Key Findings:
        
        - Detected **{total_animals} animals** in the video footage
        - Observed **{st.session_state.behavior_data['calm']} calm behaviors** and **{st.session_state.behavior_data['aggressive']} aggressive behaviors**
        - The ratio of aggressive to total behaviors is **{behavior_ratio:.2%}**
        - Primary detection location: **{location_name}** ({location_lat:.4f}, {location_lon:.4f})
        
        This analysis provides insights into animal behavior patterns that can be used for conservation efforts and wildlife management.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Display instructions when no file is uploaded
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ## How to Use WildEye
        
        1. **Upload Video**: Use the sidebar to upload wildlife footage
        2. **Configure Settings**: Adjust detection parameters as needed
        3. **Process Video**: Click the "Process Video" button to start analysis
        4. **View Results**: Explore visualizations of animal behavior and detection data
        
        WildEye uses advanced AI to detect animals and analyze their behavior patterns. The dashboard provides:
        - Behavior classification (calm vs. aggressive)
        - Animal detection counts
        - Geographic mapping of detections
        - Timeline analysis of behaviors
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("https://via.placeholder.com/400x300?text=WildEye+Demo", caption="WildEye in action")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;">
    <p>WildEye - Animal Behavior Detection Dashboard | © 2025</p>
</div>
""", unsafe_allow_html=True)
