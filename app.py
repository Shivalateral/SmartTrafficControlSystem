"""
Streamlit App: Density-Based Smart Traffic Control using Canny Edge Detection
----------------------------------------------------------------------------

- Web-based real-time friendly.
- Session-based login (username/password stored in-memory).
- Upload or capture images from webcam.
- Tune Canny parameters.
- Compare white-pixel counts (edges) of sample vs reference.
- Compute green-light time allocation based on density.

Run locally:
1) pip install streamlit opencv-python-headless numpy pillow
2) streamlit run app.py
"""

import io
from typing import Tuple

import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ------------------------------- CONFIG -----------------------------------
st.set_page_config(
    page_title="Smart Traffic Control (Canny)",
    page_icon="🚦",
    layout="wide",
)

# ----------------------------- AUTH / LOGIN --------------------------------
VALID_USERS = {
    "admin": "admin123",
}


def login() -> bool:
    """Simple session-based login. Returns True if logged in."""
    if "auth" not in st.session_state:
        st.session_state.auth = {"is_auth": False, "user": None}

    if st.session_state.auth["is_auth"]:
        with st.sidebar:
            st.success(f"Logged in as: {st.session_state.auth['user']}")
            if st.button("Log out", use_container_width=True):
                st.session_state.auth = {"is_auth": False, "user": None}
                st.experimental_rerun()
        return True

    with st.sidebar:
        st.header("🔐 Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        col_a, col_b = st.columns([1, 1])
        login_clicked = col_a.button("Login", use_container_width=True)
        col_b.caption("Use admin / admin123 (demo)")
        if login_clicked:
            if u in VALID_USERS and VALID_USERS[u] == p:
                st.session_state.auth = {"is_auth": True, "user": u}
                st.success("Login successful! Redirecting…")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    return False

# ---------------------------- IMAGE HELPERS --------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def ensure_gray(bgr: np.ndarray) -> np.ndarray:
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def canny_edges(gray: np.ndarray, low_thresh: int, high_thresh: int, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges

def count_white_pixels(img: np.ndarray) -> int:
    return int(np.sum(img == 255))

def time_allocation(sample_pixels: int, reference_pixels: int) -> Tuple[float, int]:
    if reference_pixels <= 0:
        return 0.0, 20
    avg = (sample_pixels / reference_pixels) * 100.0
    if avg >= 90:
        t = 60
    elif avg > 85:
        t = 50
    elif avg > 75:
        t = 40
    elif avg > 50:
        t = 30
    else:
        t = 20
    return avg, t

# ------------------------------- UI ----------------------------------------

def main_app():
    st.title("🚦 Density-Based Smart Traffic Control (Canny)")
    st.caption("Web app built with Streamlit. Upload/capture an image, detect edges, and compute green time.")

    with st.sidebar:
        st.header("⚙️ Parameters")
        low = st.slider("Canny Low Threshold", 0, 255, 50)
        high = st.slider("Canny High Threshold", 1, 255, 150)
        if high <= low:
            st.warning("High threshold should be > Low threshold")
        ksize = st.slider("Gaussian Kernel Size", 3, 21, 5, step=2)
        sigma = st.slider("Gaussian Sigma", 0.0, 5.0, 1.2, step=0.1)

        st.markdown("---")
        st.subheader("Reference Image")
        ref_choice = st.radio("How to provide reference?", ["Upload", "Use sample"], horizontal=True)
        ref_img = None
        if ref_choice == "Upload":
            ref_upl = st.file_uploader("Upload reference (PNG/JPG)", type=["png", "jpg", "jpeg"], key="ref")
            if ref_upl is not None:
                ref_img = Image.open(ref_upl).convert("RGB")
        else:
            synthetic = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(synthetic, (40, 40), (216, 216), 255, 3)
            ref_img = Image.fromarray(cv2.cvtColor(synthetic, cv2.COLOR_GRAY2RGB))

    st.subheader("1) Provide a sample traffic image")
    col1, col2 = st.columns(2)
    sample_pil = None
    with col1:
        upl = st.file_uploader("Upload snapshot (PNG/JPG)", type=["png", "jpg", "jpeg"], key="sample")
        if upl is not None:
            sample_pil = Image.open(upl).convert("RGB")
    with col2:
        cam = st.camera_input("Or capture from webcam")
        if cam is not None:
            sample_pil = Image.open(cam).convert("RGB")

    if sample_pil is None:
        st.info("Upload or capture a traffic image to begin.")
        return

    # -------------------------- PROCESSING --------------------------------
    sample_bgr = pil_to_cv2(sample_pil)
    sample_gray = ensure_gray(sample_bgr)
    edges = canny_edges(sample_gray, low, high, ksize, sigma)

    if ref_img is not None:
        ref_bgr = pil_to_cv2(ref_img)
        ref_gray = ensure_gray(ref_bgr)
        ref_edges = canny_edges(ref_gray, low, high, ksize, sigma)
    else:
        ref_edges = np.zeros_like(edges)

    # -------------------------- DISPLAY -----------------------------------
    st.subheader("2) Visualize")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(sample_pil, caption="Sample (Original)", use_container_width=True)
    with c2:
        st.image(sample_gray, caption="Sample (Grayscale)", use_container_width=True, clamp=True)
    with c3:
        st.image(edges, caption="Sample (Canny Edges)", use_container_width=True, clamp=True)

    st.markdown("---")
    d1, d2, d3 = st.columns(3)
    with d1:
        if ref_img is not None:
            st.image(ref_img, caption="Reference (Original or Synthetic)", use_container_width=True)
    with d2:
        st.image(ref_edges, caption="Reference (Canny Edges)", use_container_width=True, clamp=True)
    with d3:
        sample_pixels = count_white_pixels(edges)
        reference_pixels = count_white_pixels(ref_edges)
        density, green_time = time_allocation(sample_pixels, reference_pixels)
        st.metric("Sample white pixels", f"{sample_pixels:,}")
        st.metric("Reference white pixels", f"{reference_pixels:,}")
        st.metric("Traffic Density %", f"{density:.2f}%")
        st.metric("Green Signal Time", f"{green_time} sec")

    # ------------------------- DOWNLOAD RESULTS ----------------------------
    st.subheader("3) Save results")
    edges_png = Image.fromarray(edges)
    buf = io.BytesIO()
    edges_png.save(buf, format="PNG")
    st.download_button(
        label="Download edge map (PNG)",
        data=buf.getvalue(),
        file_name="sample_edges.png",
        mime="image/png",
        use_container_width=True,
    )

    st.caption(
        "Tip: For real deployments, replace demo credentials with proper auth,\n"
        "and store reference images per-lane/camera for consistent density comparisons."
    )

# --------------------------------- ENTRY -----------------------------------
if __name__ == "__main__":
    if login():
        main_app()
    else:
        st.title("🚦 Smart Traffic Control")
        st.info("Please log in from the sidebar to continue.")
