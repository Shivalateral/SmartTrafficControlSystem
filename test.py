
"""
Streamlit Demo: Canny Edge Detection with Visualization
-------------------------------------------------------
- Upload/capture an image
- Detect edges using custom CannyEdgeDetector
- Visualize pixels and green-time allocation with graphs
"""

import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from CannyEdgeDetector import CannyEdgeDetector

# ---------------------------- IMAGE HELPERS --------------------------------
def pil_to_cv2(img):
    """Convert PIL.Image or numpy array to OpenCV BGR image."""
    if img is None:
        return None
    if isinstance(img, Image.Image):  # If it's PIL
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):  # If it's already NumPy
        if len(img.shape) == 2:  # grayscale
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img  # assume already BGR
    else:
        # Try conversion via np.array
        try:
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise TypeError(f"Unsupported type for pil_to_cv2: {type(img)}")

def ensure_gray(bgr: np.ndarray) -> np.ndarray:
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def count_white_pixels(img: np.ndarray) -> int:
    return int(np.sum(img == 255))

def time_allocation(sample_pixels: int, reference_pixels: int):
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

# ----------------------------- STREAMLIT APP -------------------------------
st.set_page_config(page_title="Canny Edge Detection Demo", page_icon="📷", layout="wide")
st.title("📷 Canny Edge Detection Demo")
st.caption("Upload/capture an image, detect edges, and see traffic insights.")

# Sidebar: Parameters
with st.sidebar:
    st.header("⚙️ Parameters")
    sigma = st.slider("Gaussian Sigma", 0.0, 5.0, 1.2, step=0.1)
    kernel_size = st.slider("Gaussian Kernel Size", 3, 21, 5, step=2)
    low_t = st.slider("Low Threshold (ratio)", 0.01, 0.5, 0.09, step=0.01)
    high_t = st.slider("High Threshold (ratio)", 0.1, 0.5, 0.2, step=0.01)

# -------------------------- INPUT IMAGE ------------------------------------
col1, col2 = st.columns(2)
with col1:
    upl = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="sample")
    if upl is not None:
        sample_pil = Image.open(upl).convert("RGB")
    else:
        sample_pil = None
with col2:
    cam = st.camera_input("Or capture from webcam")
    if cam is not None:
        sample_pil = Image.open(cam).convert("RGB")

if sample_pil is None:
    st.info("Upload or capture an image to begin.")
    st.stop()

# -------------------------- PROCESSING -------------------------------------
sample_bgr = pil_to_cv2(sample_pil)
sample_gray = ensure_gray(sample_bgr)

# Use synthetic reference image (simple square for demo)
synthetic = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(synthetic, (40, 40), (216, 216), 255, 3)
ref_img = synthetic
ref_bgr = pil_to_cv2(ref_img)
ref_gray = ensure_gray(ref_bgr)

detector = CannyEdgeDetector([sample_gray, ref_gray], sigma=sigma,
                             kernel_size=kernel_size,
                             lowthreshold=low_t, highthreshold=high_t,
                             weak_pixel=100)
results = detector.detect()
sample_edges, ref_edges = results

# -------------------------- DISPLAY ---------------------------------------
st.subheader("🔍 Visualization")
c1, c2, c3 = st.columns(3)
c1.image(sample_pil, caption="Original", use_container_width=True)
c2.image(sample_gray, caption="Grayscale", use_container_width=True, clamp=True)
c3.image(sample_edges, caption="Canny Edges", use_container_width=True, clamp=True)

d1, d2, d3 = st.columns(3)
d1.image(ref_img, caption="Reference (Synthetic)", use_container_width=True, clamp=True)
d2.image(ref_edges, caption="Reference Edges", use_container_width=True, clamp=True)
with d3:
    sample_pixels = count_white_pixels(sample_edges)
    reference_pixels = count_white_pixels(ref_edges)
    density, green_time = time_allocation(sample_pixels, reference_pixels)
    st.metric("Sample white pixels", f"{sample_pixels:,}")
    st.metric("Reference white pixels", f"{reference_pixels:,}")
    st.metric("Traffic Density %", f"{density:.2f}%")
    st.metric("Green Signal Time", f"{green_time} sec")

# -------------------------- GRAPHS ----------------------------------------
st.subheader("📊 Graphical Analysis")

# 1. Bar Chart
df_pixels = pd.DataFrame({
    "Type": ["Sample", "Reference"],
    "White Pixels": [sample_pixels, reference_pixels]
})
st.bar_chart(df_pixels.set_index("Type"))

# 2. Line Chart (traffic density vs green time)
densities = [10, 30, 55, 80, 87, 95]
times = []
for d in densities:
    _, t = time_allocation(d, 100)  # pretend reference=100
    times.append(t)
df_line = pd.DataFrame({"Density %": densities, "Green Time (sec)": times})
st.line_chart(df_line.set_index("Density %"))

# 3. Pie Chart (density vs free road)
fig, ax = plt.subplots()
ax.pie([density, 100-density], labels=["Traffic Density", "Free Road"],
       autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

# -------------------------- DOWNLOAD --------------------------------------
st.subheader("💾 Save Results")
edges_png = Image.fromarray(sample_edges)
buf = io.BytesIO()
edges_png.save(buf, format="PNG")
st.download_button(
    label="Download edge map (PNG)",
    data=buf.getvalue(),
    file_name="sample_edges.png",
    mime="image/png",
    use_container_width=True,
)
