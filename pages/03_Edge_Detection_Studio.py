import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.edge_detection import EdgeDetector
from ui.components import render_global_header, render_section_header, render_status_bar


st.set_page_config(page_title="SSR VISION | Edge Studio", page_icon="📐", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="EDGE RESEARCH LAB")


def render_image_card(title, image, caption=None):
    st.markdown(f"<div class='image-card'><div class='panel-kicker'>{title}</div>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    if caption:
        st.markdown(f"<div class='panel-note'>{caption}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def amplify_display(image_rgb, alpha):
    return cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=0)


def render_grid_in_order(outputs, ordered_names):
    for i in range(0, len(ordered_names), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx >= len(ordered_names):
                continue
            name = ordered_names[idx]
            with cols[j]:
                render_image_card(name, outputs[name])


st.markdown("<div class='research-shell'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='lab-banner'>
        <div>
            <div class='lab-banner-title'>EDGE DETECTION STUDIO</div>
            <div class='lab-banner-subtitle'>Full simultaneous edge-comparison laboratory for surveillance imagery and scene-structure research.</div>
        </div>
        <div class='lab-chip-row'>
            <span class='lab-chip'>ALL METHODS RUN TOGETHER</span>
            <span class='lab-chip'>GRADIENT ANALYTICS</span>
            <span class='lab-chip'>EDGE OVERLAY MAPPING</span>
            <span class='lab-chip'>DIFFERENCE HIGHLIGHTING</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, center_col, right_col = st.columns([1.0, 1.15, 1.15])

with left_col:
    render_section_header("PARAMETER CONTROL PANEL")
    lower = st.slider("Lower Threshold", min_value=0, max_value=255, value=50, help="Lower Canny threshold.")
    upper = st.slider("Upper Threshold", min_value=0, max_value=255, value=150, help="Upper Canny threshold.")
    kernel_size = st.slider("Kernel Size", min_value=3, max_value=9, value=3, step=2, help="Odd kernel size for Sobel and Laplacian operators.")
    sigma = st.slider("Sigma", min_value=0.6, max_value=3.0, value=1.2, step=0.2, help="Gaussian sigma for Canny and Difference of Gaussian.")
    gain = st.slider("Gradient Intensity Gain", min_value=0.8, max_value=2.4, value=1.2, step=0.1, help="Display amplification for comparison imagery.")
    compare_left = st.selectbox("Compare Method A", ["Sobel", "Prewitt", "Roberts", "Laplacian", "Canny", "Scharr", "Kirsch", "Robinson Compass", "Frei-Chen", "Difference of Gaussian"])
    compare_right = st.selectbox("Compare Method B", ["Sobel", "Prewitt", "Roberts", "Laplacian", "Canny", "Scharr", "Kirsch", "Robinson Compass", "Frei-Chen", "Difference of Gaussian"], index=4)

if "img_np" not in st.session_state:
    st.warning("No active image feed available. Upload an image here or analyze one in Image Analysis Center.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

img_np = st.session_state.img_np
outputs = EdgeDetector.all_algorithms(img_np, lower=lower, upper=upper, kernel_size=kernel_size, sigma=sigma)
display_outputs = {name: amplify_display(image, gain) for name, image in outputs.items() if name != "Original"}
display_outputs["Original"] = img_np

gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
canny_map = EdgeDetector.apply_canny(gray, lower=lower, upper=upper, sigma=sigma)
edge_density = float(np.mean(canny_map > 0))
magnitude_map = cv2.cvtColor(outputs["Magnitude"], cv2.COLOR_RGB2GRAY)
direction_map = cv2.cvtColor(outputs["Direction"], cv2.COLOR_RGB2GRAY)

with center_col:
    render_section_header("REFERENCE FEED")
    render_image_card("Original Scene", img_np, caption=f"{img_np.shape[1]}x{img_np.shape[0]} | Edge density {edge_density * 100:.1f}%")
    m1, m2 = st.columns(2)
    m1.metric("Mean Gradient", f"{np.mean(magnitude_map):.1f}")
    m2.metric("Mean Direction", f"{np.mean(direction_map):.1f}")

with right_col:
    render_section_header("EDGE OVERLAY MODE")
    render_image_card("Overlay Mapping", display_outputs["Edge Overlay"], caption="Canny edges fused onto the original frame for real-world boundary mapping.")
    m3, m4 = st.columns(2)
    m3.metric("Canny Thresholds", f"{lower}/{upper}")
    m4.metric("Sigma", f"{sigma:.1f}")

render_section_header("SIMULTANEOUS ALGORITHM GRID")
ordered_grid = [
    "Original",
    "Sobel",
    "Prewitt",
    "Roberts",
    "Laplacian",
    "Canny",
    "Scharr",
    "Kirsch",
    "Frei-Chen",
    "Robinson Compass",
    "Difference of Gaussian",
    "Edge Overlay",
]
render_grid_in_order(display_outputs, ordered_grid)

render_section_header("GRADIENT VISUALIZATION")
gradient_names = ["Gradient X", "Gradient Y", "Magnitude", "Direction"]
gradient_cols = st.columns(4)
for idx, name in enumerate(gradient_names):
    with gradient_cols[idx]:
        render_image_card(name, display_outputs[name])

render_section_header("COMPARISON TOOLS")
comparison_left = cv2.cvtColor(display_outputs[compare_left], cv2.COLOR_RGB2GRAY)
comparison_right = cv2.cvtColor(display_outputs[compare_right], cv2.COLOR_RGB2GRAY)
difference = cv2.absdiff(comparison_left, comparison_right)
difference_rgb = cv2.cvtColor(difference, cv2.COLOR_GRAY2RGB)
cmp1, cmp2, cmp3 = st.columns(3)
with cmp1:
    render_image_card(compare_left, display_outputs[compare_left], caption="Zoom on hover enabled via CSS.")
with cmp2:
    render_image_card(compare_right, display_outputs[compare_right], caption="Side-by-side structural comparison.")
with cmp3:
    render_image_card("Difference Highlight", difference_rgb, caption="Absolute pixel difference between the selected methods.")

with st.expander("Research Notes & Tooltips", expanded=False):
    st.markdown(
        """
        <span class='tooltip-chip'>Sobel / Scharr: first-order gradients</span>
        <span class='tooltip-chip'>Kirsch / Robinson: compass masks</span>
        <span class='tooltip-chip'>Frei-Chen: weighted gradient basis</span>
        <span class='tooltip-chip'>DoG: scale-space edge emphasis</span>
        <span class='tooltip-chip'>Edge Overlay: real-world edge mapping</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("All edge methods execute simultaneously on every rerun. The comparison selectors only choose which already-computed outputs to inspect more closely.")

st.markdown("</div>", unsafe_allow_html=True)