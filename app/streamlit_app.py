"""Streamlit Web Application for GCP Marker Localization and Shape Classification."""

import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import streamlit as st

# Add src to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gcp_localizer.inference.predictor import GCPPredictor
from gcp_localizer.inference.visualization import annotate_prediction


@st.cache_resource
def load_predictor() -> GCPPredictor:
    """Load and cache the unified GCP predictor."""
    return GCPPredictor()


def create_demo_gcp_image() -> np.ndarray:
    """Generate a realistic synthetic aerial image with a Ground Control Point for demo testing."""
    import cv2
    np.random.seed(42)
    # Background texture resembling terrain/ground
    h, w = 1200, 1600
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :] = [90, 115, 80]  # Earthy green/brown
    noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add terrain details (soil patches, road line)
    cv2.line(img, (0, 700), (w, 550), (140, 135, 125), 45)
    cv2.line(img, (0, 700), (w, 550), (120, 115, 105), 35)

    # Draw an L-Shape GCP marker at (850, 620)
    gcp_x, gcp_y = 850, 620
    # Black border square
    cv2.rectangle(img, (gcp_x - 40, gcp_y - 40), (gcp_x + 40, gcp_y + 40), (20, 20, 20), -1)
    # White L-Shape
    cv2.rectangle(img, (gcp_x - 35, gcp_y - 35), (gcp_x + 35, gcp_y - 15), (245, 245, 245), -1)
    cv2.rectangle(img, (gcp_x - 35, gcp_y - 35), (gcp_x - 15, gcp_y + 35), (245, 245, 245), -1)

    return img


def main():
    st.set_page_config(
        page_title="GCP Marker Localizer",
        page_icon="🎯",
        layout="wide"
    )

    st.title("GCP Marker Localizer")
    st.markdown("Upload an aerial image to detect the GCP marker location and classify its shape.")

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Sidebar / Controls
    st.sidebar.header("Options")
    use_sample = st.sidebar.checkbox("Use Demo Aerial Image", value=False)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Upload an aerial survey or drone photograph"
    )

    image_np = None

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(pil_image)
        except Exception as e:
            st.error(f"Failed to load uploaded image: {e}")
            return
    elif use_sample:
        image_np = create_demo_gcp_image()
        st.info("Loaded sample aerial image containing a Ground Control Point marker.")

    if image_np is not None:
        st.divider()

        with st.spinner("Running unified localization & shape classification pipeline..."):
            try:
                prediction, annotated_img = predictor.predict_and_annotate(image_np)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Aerial Image")
            st.image(image_np, use_container_width=True, caption=f"Resolution: {image_np.shape[1]}x{image_np.shape[0]} px")

        with col2:
            st.subheader("Detected GCP Annotation")
            st.image(annotated_img, use_container_width=True, caption="Annotated Marker Location & Classification")

        st.divider()

        # Prediction Summary Card
        st.subheader("GCP Prediction")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.markdown("#### Location")
            st.write(f"**X:** `{prediction['x']:.2f} px`")
            st.write(f"**Y:** `{prediction['y']:.2f} px`")

        with metric_col2:
            st.markdown("#### Shape")
            st.write(f"**{prediction['shape']}**")

        with metric_col3:
            st.markdown("#### Confidence")
            conf_pct = prediction['confidence'] * 100.0 if prediction['confidence'] <= 1.0 else prediction['confidence']
            st.write(f"**{conf_pct:.1f}%**")


if __name__ == "__main__":
    main()
