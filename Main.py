
import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import datetime
import pandas as pd
import altair as alt
from CannyEdgeDetector import CannyEdgeDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import time
import tempfile
import av
import sqlite3
import bcrypt
import base64

# ------------------------------- CONFIGURATION & STYLING -----------------------------------
st.set_page_config(
    page_title="Smart Traffic Control", page_icon="🚦", layout="wide"
)

# Function to encode local images for CSS
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css(page_type):
    """Loads custom CSS based on the page being displayed (auth or dashboard)."""
    
    # Common styles for both pages
    common_css = """
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #DDDDDD;
        }

        /* Custom Titles and Headers */
        .custom-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1E3A8A; /* Dark Blue */
            padding-top: 20px;
        }
        .custom-header {
            font-size: 1.75rem;
            font-weight: bold;
            color: #1E3A8A; /* Dark Blue */
            border-bottom: 2px solid #3B82F6; /* Lighter Blue Accent */
            padding-bottom: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Card container style */
        .card {
            background-color: #FFFFFF;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #DDDDDD;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        }

        /* Streamlit button styling */
        div.stButton > button {
            background-color: #3B82F6; /* Blue */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.2s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #2563EB; /* Darker Blue */
            transform: scale(1.02);
        }
        
        /* Text Input Styling */
        .stTextInput input {
            background-color: #F0F2F6;
            color: #333333;
            border: 1px solid #CCCCCC;
            border-radius: 8px;
        }

        /* Tab styling */
        [data-testid="stTabs"] button {
            color: #555555;
            border-radius: 8px 8px 0 0;
            background-color: transparent;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #FFFFFF;
            color: #3B82F6; /* Blue */
            border-bottom: 3px solid #3B82F6;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background-color: #F9FAFB;
            border: 1px solid #DDDDDD;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            color: #3B82F6;
        }

        /* Improve visibility of chart text */
        .stVegaLiteChart svg text {
            fill: #333333 !important;
        }
    </style>
    """
    st.markdown(common_css, unsafe_allow_html=True)

    if page_type == "auth":
        try:
            # Path to your local background image for the auth pages
            image_path = r"E:\SOURCE CODE\background.png"
            bg_image_base64 = get_base64_of_bin_file(image_path)
            auth_css = f"""
            <style>
                .stApp {{
                    background-image: linear-gradient(rgba(240, 242, 246, 0.8), rgba(240, 242, 246, 0.8)), url("data:image/png;base64,{bg_image_base64}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                .auth-form {{
                    background-color: rgba(255, 255, 255, 0.9);
                    backdrop-filter: blur(5px);
                    border-radius: 15px;
                    padding: 30px;
                    border: 1px solid #DDDDDD;
                    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
                }}
                .traffic-light-container {{
                    text-align: center;
                    padding-bottom: 20px;
                }}
            </style>
            """
            st.markdown(auth_css, unsafe_allow_html=True)
        except FileNotFoundError:
            # Fallback to a simple background if the image is not found
            st.markdown("<style>.stApp { background-color: #F0F2F6; }</style>", unsafe_allow_html=True)
    else: # Dashboard styles
        dashboard_css = """
        <style>
            .stApp {
                background-color: #F0F2F6; /* Light gray background for dashboard */
            }
        </style>
        """
        st.markdown(dashboard_css, unsafe_allow_html=True)


# ---------------------------- SQLITE DATABASE SETUP -----------------------------
DB_FILE = "traffic_data.db"

def initialize_sqlite():
    """Initializes a connection to the SQLite database and creates tables if they don't exist."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            density REAL NOT NULL,
            green_time INTEGER NOT NULL,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    con.commit()
    con.close()
    return True

# ---------------------------- PASSWORD HASHING UTILS -----------------------------
def hash_password(password: str) -> str:
    """Hashes a password using bcrypt for secure storage."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(hashed_password: str, plain_password: str) -> bool:
    """Verifies a plain password against its hashed version."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# ------------------------------- AUTHENTICATION PAGES --------------------------------
def render_auth_page(page_function):
    """Renders the login or signup page with a modern layout and theme."""
    st.markdown('<div class="custom-title">Smart Traffic Control</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        
        traffic_light_svg = """
        <div class="traffic-light-container">
            <svg width="80" height="200" viewBox="0 0 80 200" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="80" height="200" rx="10" fill="#E0E0E0"/>
            <circle cx="40" cy="40" r="25" fill="#D63031" stroke="#CCC" stroke-width="3"/>
            <circle cx="40" cy="100" r="25" fill="#FFD700" stroke="#CCC" stroke-width="3"/>
            <circle cx="40" cy="160" r="25" fill="#00B894" stroke="#CCC" stroke-width="3"/>
            </svg>
        </div>
        """
        st.markdown(traffic_light_svg, unsafe_allow_html=True)
        
        page_function()
        
        st.markdown('</div>', unsafe_allow_html=True)

def signup_page():
    """Displays the user registration page content."""
    st.subheader("Create New Account")
    with st.form("signup_form"):
        username = st.text_input("Username", key="signup_user").lower()
        password = st.text_input("Password", type="password", key="signup_pass")
        password_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm")
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            if not username or not password or not password_confirm:
                st.error("Please fill out all fields.")
            elif password != password_confirm:
                st.error("Passwords do not match.")
            else:
                con = sqlite3.connect(DB_FILE)
                cur = con.cursor()
                cur.execute("SELECT username FROM users WHERE username = ?", (username,))
                if cur.fetchone():
                    st.error("Username is already taken. Please choose another one.")
                else:
                    hashed_pass = hash_password(password)
                    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pass))
                    con.commit()
                    st.success("Account created successfully!")
                    st.info("Please navigate to the Login page to continue.")
                con.close()

def login_page():
    """Displays the user login page content."""
    st.subheader("Login to Your Account")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_user").lower()
        password = st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Login")

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                con = sqlite3.connect(DB_FILE)
                cur = con.cursor()
                cur.execute("SELECT password FROM users WHERE username = ?", (username,))
                result = cur.fetchone()
                con.close()
                if not result:
                    st.error("Invalid username or password.")
                else:
                    hashed_pass = result[0]
                    if check_password(hashed_pass, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

# ---------------------------- CORE APPLICATION -----------------------------
@st.cache_resource
def load_yolo_model():
    model = YOLO("yolov8n.pt")
    return model

VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
model = load_yolo_model()
CLASS_NAMES = model.names

def pil_to_cv2(img):
    if isinstance(img, Image.Image): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    raise TypeError("Unsupported image type for conversion.")

def ensure_gray(bgr_image: np.ndarray) -> np.ndarray:
    return bgr_image if len(bgr_image.shape) == 2 else cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

def count_white_pixels(img: np.ndarray) -> int:
    return int(np.sum(img == 255))

def time_allocation(sample_pixels: int, reference_pixels: int):
    """Calculates green light duration based on traffic density."""
    if reference_pixels <= 0: return 0.0, 21
    density = (sample_pixels / reference_pixels) * 100.0
    
    if density <= 40: t = 21
    elif 41 <= density <= 70: t = 30
    elif 71 <= density <= 90: t = 42
    else: t = 54
    
    return density, t

def run_analysis(frame: np.ndarray, sidebar_params: dict, ref_img: Image.Image):
    """Performs the full analysis pipeline on a given frame and displays results."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    results = model(frame, verbose=False)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # --- NEW: Initialize a dictionary to hold vehicle counts ---
    vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in VEHICLE_CLASSES:
                # --- NEW: Get class name and increment count ---
                class_name = model.names[cls_id]
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                
                # Draw rectangle on the mask for density calculation (existing logic)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    total_vehicles = sum(vehicle_counts.values())
    
    vehicles_only_bgr = cv2.bitwise_and(frame, frame, mask=mask)
    sample_gray = ensure_gray(vehicles_only_bgr); ref_bgr = pil_to_cv2(ref_img); ref_gray = ensure_gray(ref_bgr)
    detector = CannyEdgeDetector([sample_gray, ref_gray], sigma=sidebar_params['sigma'], kernel_size=sidebar_params['kernel_size'], lowthreshold=sidebar_params['low_t'], highthreshold=sidebar_params['high_t'], weak_pixel=100)
    canny_results = detector.detect(); sample_edges, ref_edges = canny_results

    st.markdown('<p class="custom-header">Visual Analysis Steps</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        st.image(frame, channels="BGR", caption="Input: Original Snapshot", use_container_width=True)
        st.image(vehicles_only_bgr, caption="Processing: Isolate Vehicles", use_container_width=True)
    with col2: 
        st.image(sample_edges, caption="Processing: Detect Vehicle Edges", use_container_width=True)
        st.image(ref_edges, caption="Reference: Empty Road Edges", use_container_width=True)
    
    st.markdown('</div><div class="card">', unsafe_allow_html=True)
    
    # --- MODIFIED: Display results with new vehicle counts ---
    st.markdown('<p class="custom-header">Analysis Results</p>', unsafe_allow_html=True)

    # Display Vehicle Count & Classification
    st.write("##### Vehicle Count & Classification")
    v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
    v_col1.metric("Total Vehicles", f"{total_vehicles}")
    v_col2.metric("Cars", f"{vehicle_counts['car']}")
    v_col3.metric("Motorcycles", f"{vehicle_counts['motorcycle']}")
    v_col4.metric("Buses", f"{vehicle_counts['bus']}")
    v_col5.metric("Trucks", f"{vehicle_counts['truck']}")
    
    st.markdown("---") # Visual separator

    # Display Density Calculation & Time Allocation
    st.write("##### Density & Time Metrics")
    sample_pixels = count_white_pixels(sample_edges); reference_pixels = count_white_pixels(ref_edges)
    density, green_time = time_allocation(sample_pixels, reference_pixels)
    st.session_state.latest_results = {"density": density, "green_time": green_time}
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"); current_user = st.session_state.username
    con = sqlite3.connect(DB_FILE); cur = con.cursor()
    cur.execute("INSERT INTO analysis_history (username, timestamp, density, green_time) VALUES (?, ?, ?, ?)", (current_user, timestamp, density, green_time)); con.commit(); con.close()

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Calculated Density", f"{density:.2f}%")
    mcol2.metric("Allocated Green Time", f"{green_time} sec")
    mcol3.metric("Vehicle Edge Pixels", f"{sample_pixels:,}")
    mcol4.metric("Reference Edge Pixels", f"{reference_pixels:,}")
    
    if "history" not in st.session_state: st.session_state["history"] = []
    st.session_state["history"].append({"timestamp": timestamp, "green_time": green_time, "density": round(density, 2)})
    hist_df = pd.DataFrame(st.session_state["history"][-10:])

    # Charting
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.write("##### Current Allocation Logic"); chart_data = pd.DataFrame({'Metric': ['Density (%)', 'Time (sec)'], 'Value': [round(density, 2), green_time]})
        bar_chart = alt.Chart(chart_data).mark_bar(width=50).encode(x=alt.X('Metric', sort=None, title=None), y=alt.Y('Value', title='Value'), color=alt.Color('Metric', legend=None, scale=alt.Scale(range=['#3B82F6', '#10B981'])), tooltip=['Metric', 'Value']).properties(height=300); st.altair_chart(bar_chart, use_container_width=True)
    with gcol2:
        st.write("##### Green Time History (This Session)"); line_chart = alt.Chart(hist_df).mark_line(point=True).encode(x=alt.X("timestamp:T", title="Time of Analysis"), y=alt.Y("green_time", title="Allocated Time (sec)", scale=alt.Scale(domain=[15, 65])), tooltip=['timestamp:T', 'green_time', 'density']).properties(height=300); st.altair_chart(line_chart, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main_dashboard():
    """The main dashboard of the application, shown after successful login."""
    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.markdown(f'<div class="custom-title" style="text-align: left; padding: 0;">Traffic Analysis Dashboard</div>', unsafe_allow_html=True)
        st.caption(f"Welcome, {st.session_state.username}!")
    with col2:
        image_path = r"E:\SOURCE CODE\traffic.png"
        try:
            st.image(image_path, use_container_width=True)
        except FileNotFoundError:
            st.error(f"Error: Image not found at path: {image_path}. Please ensure the image exists at this location.")

    tab1, tab2 = st.tabs(["🚦 Traffic Analysis Tool", "📊 Analytics Dashboard"])

    with tab1:
        with st.sidebar:
            st.header("⚙️ Parameters"); params = {'sigma': st.slider("Gaussian Sigma", 0.0, 5.0, 1.2, step=0.1), 'kernel_size': st.slider("Gaussian Kernel Size", 3, 21, 5, step=2), 'low_t': st.slider("Low Threshold", 0.01, 0.5, 0.09, step=0.01), 'high_t': st.slider("High Threshold", 0.1, 0.5, 0.2, step=0.01), 'motion_threshold': st.slider("Motion Threshold", 100, 5000, 500, step=50, help="Lower value is more sensitive to motion.")}
            st.markdown("---"); st.header("🛣️ Reference Image")
            ref_choice = st.radio("Select baseline:", ("Upload Empty Road Image (Recommended)", "Use Synthetic Box (for Demo)"), key="ref_radio")
            ref_img = None
            if ref_choice == "Upload Empty Road Image (Recommended)":
                ref_file = st.file_uploader("Upload reference...", type=["png", "jpg", "jpeg"], key="ref_uploader")
                if ref_file: ref_img = Image.open(ref_file).convert("RGB"); st.image(ref_img, caption="Empty Road Reference")
            else:
                synthetic = np.zeros((256, 256), dtype=np.uint8); cv2.rectangle(synthetic, (40, 40), (216, 216), 255, 3); ref_img = Image.fromarray(synthetic); st.image(ref_img, caption="Synthetic Reference")
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if ref_img is None: 
            st.info("👈 Please configure a reference image in the sidebar to begin.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown('<p class="custom-header">Choose Input Source</p>', unsafe_allow_html=True)
        input_mode = st.radio("Select input source:", ("Upload Image 🖼️", "Upload Video 🎥", "Live Webcam 📸"), horizontal=True, label_visibility="collapsed", key="input_mode")
        if input_mode == "Upload Image 🖼️":
            uploaded_image = st.file_uploader("Upload a traffic image...", type=["png", "jpg", "jpeg"], key="img_uploader")
            if uploaded_image: image = Image.open(uploaded_image).convert("RGB"); frame_to_analyze = pil_to_cv2(image); run_analysis(frame_to_analyze, params, ref_img)
        elif input_mode == "Upload Video 🎥":
            uploaded_video = st.file_uploader("Upload a traffic video...", type=["mp4", "mov", "avi"], key="vid_uploader")
            if uploaded_video and st.button("Analyze Video"):
                with st.spinner("Processing video..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile: tfile.write(uploaded_video.read()); video_path = tfile.name
                    cap = cv2.VideoCapture(video_path); prev_frame_gray, found_frame = None, None
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                        if prev_frame_gray is not None:
                            delta = cv2.absdiff(prev_frame_gray, gray); thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                            if np.sum(thresh) < params['motion_threshold']: found_frame = frame; break
                        prev_frame_gray = gray
                    cap.release()
                    if found_frame is not None: st.success("✅ Found a frame with stopped traffic!"); run_analysis(found_frame, params, ref_img)
                    else: st.error("❌ Could not find a suitable frame. Try adjusting the motion threshold.")
        elif input_mode == "Live Webcam 📸":
            st.info("Live feed shows real-time object detection. Analysis below is triggered when motion stops.")
            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24"); results = model(img, verbose=False)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]); cls_id = int(box.cls[0]); label = CLASS_NAMES[cls_id]
                    color = (0, 255, 0) if cls_id in VEHICLE_CLASSES else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2); cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if "latest_results" in st.session_state:
                    green_time = st.session_state.latest_results["green_time"]; text = f"Allocated Green Time: {green_time} sec"; cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            webrtc_streamer(key="webcam", mode=WebRtcMode.SENDRECV, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="custom-header">Advanced Data Analytics</p>', unsafe_allow_html=True)
        con = sqlite3.connect(DB_FILE)
        history_df = pd.read_sql_query("SELECT username, timestamp, density, green_time FROM analysis_history", con)
        
        if history_df.empty:
            st.info("No analysis data available yet. Perform some analyses in the main tool to see the dashboard.")
        else:
            @st.cache_data
            def to_excel(df_to_convert):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_to_convert.to_excel(writer, index=False, sheet_name='History')
                processed_data = output.getvalue()
                return processed_data

            st.write("#### Overall System Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Average Traffic Density by Hour")
                history_df['hour'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H')
                hourly_density = history_df.groupby('hour')['density'].mean().reset_index()
                density_chart = alt.Chart(hourly_density).mark_bar().encode(
                    x=alt.X('hour', title='Hour of Day', sort=None), 
                    y=alt.Y('density', title='Avg. Density (%)')
                ).properties(title='Peak Traffic Hours')
                st.altair_chart(density_chart, use_container_width=True)
            with col2:
                st.write("##### User Contribution")
                user_activity = history_df['username'].value_counts().reset_index()
                user_activity.columns = ['username', 'analysis_count']
                activity_chart = alt.Chart(user_activity).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="analysis_count", type="quantitative"), 
                    color=alt.Color(field="username", type="nominal"), 
                    tooltip=['username', 'analysis_count']
                ).properties(title='Analyses by User')
                st.altair_chart(activity_chart, use_container_width=True)
            
            st.markdown("---")

            st.subheader(f"My Analysis History ({st.session_state.username})")
            my_history_df = history_df[history_df['username'] == st.session_state.username].copy()
            
            if not my_history_df.empty:
                my_excel_data = to_excel(my_history_df[['timestamp', 'density', 'green_time']])
                st.download_button(
                    label="📥 Download My History as Excel",
                    data=my_excel_data,
                    file_name=f"{st.session_state.username}_analysis_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                display_df = my_history_df[['timestamp', 'density', 'green_time']].copy()
                display_df['density'] = display_df['density'].map('{:.2f}%'.format)
                display_df.rename(columns={'timestamp': 'Timestamp', 'density': 'Calculated Density', 'green_time': 'Green Time (sec)'}, inplace=True)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("You have not performed any analysis yet. Your results will appear here.")
        
        con.close()
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------- ENTRY POINT & ROUTING -----------------------------------
if __name__ == "__main__":
    initialize_sqlite()
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if "latest_results" not in st.session_state: st.session_state.latest_results = {"density": 0, "green_time": 21}

    if st.session_state.logged_in:
        load_css("dashboard") # Load dashboard-specific CSS
        with st.sidebar:
            st.success(f"Logged in as {st.session_state.username}")
            if st.button("Log out", use_container_width=True):
                st.session_state.logged_in = False; st.session_state.username = ""; st.rerun()
        main_dashboard()
    else:
        load_css("auth") # Load auth-specific CSS
        auth_choice = st.sidebar.radio("Welcome! Please select an option:", ["Login", "Sign Up"])
        if auth_choice == "Login":
            render_auth_page(login_page)
        else:
            render_auth_page(signup_page)