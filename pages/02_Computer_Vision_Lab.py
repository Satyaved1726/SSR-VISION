import cv2
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.edge_detection import EdgeDetector
from core.feature_extraction import FeatureExtractor
from core.image_processing import PreprocessingPipeline
from core.segmentation import SegmentationEngine
from core.vision import VisionAnalyzer
from ui.components import render_global_header, render_section_header, render_status_bar


st.set_page_config(page_title="SSR VISION | Vision Lab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="CV RESEARCH LAB")


@st.cache_resource
def get_vision_service():
    return VisionAnalyzer()


def image_signature(image_rgb):
    return (
        int(image_rgb.shape[0]),
        int(image_rgb.shape[1]),
        int(np.mean(image_rgb)),
        int(np.std(image_rgb)),
    )


def render_image_card(title, image, caption=None):
    st.markdown(f"<div class='image-card'><div class='panel-kicker'>{title}</div>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    if caption:
        st.markdown(f"<div class='panel-note'>{caption}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_image_grid(items, columns=3):
    names = list(items.keys())
    for i in range(0, len(names), columns):
        cols = st.columns(columns)
        for j in range(columns):
            idx = i + j
            if idx >= len(names):
                continue
            name = names[idx]
            with cols[j]:
                render_image_card(name, items[name])


def build_histogram_figure(histograms):
    figure = go.Figure()
    x_axis = list(range(len(next(iter(histograms.values())))))
    channel_styles = {
        "Red": "#ff5f6d",
        "Green": "#00ff9a",
        "Blue": "#00f7ff",
    }
    for channel_name, values in histograms.items():
        figure.add_trace(
            go.Scatter(
                x=x_axis,
                y=values,
                mode="lines",
                name=channel_name,
                line={"color": channel_styles.get(channel_name, "#ffffff"), "width": 2},
            )
        )
    figure.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,11,20,0.55)",
        font={"color": "#d9f8ff"},
        legend={"orientation": "h", "y": 1.05, "x": 0},
        xaxis={"showgrid": True, "gridcolor": "rgba(0,247,255,0.08)", "title": "Intensity Bin"},
        yaxis={"showgrid": True, "gridcolor": "rgba(0,247,255,0.08)", "title": "Frequency"},
    )
    return figure


st.markdown("<div class='research-shell'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='lab-banner'>
        <div>
            <div class='lab-banner-title'>COMPUTER VISION RESEARCH LAB</div>
            <div class='lab-banner-subtitle'>Full CVIP experimentation environment for surveillance, smart-city, and cyber intelligence pipelines.</div>
        </div>
        <div class='lab-chip-row'>
            <span class='lab-chip'>GPU-SAFE</span>
            <span class='lab-chip'>YOLOV8 TRAFFIC DETECTION</span>
            <span class='lab-chip'>MULTI-STAGE CVIP</span>
            <span class='lab-chip'>LIGHTWEIGHT CSS MOTION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, center_col, right_col = st.columns([1.05, 1.15, 1.15])

with left_col:
    render_section_header("CONTROL PANEL")

    with st.expander("Preprocessing", expanded=True):
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=11, step=2, value=5, help="Shared blur and morphology kernel size.")
        threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=127, help="Base value for binary, inverse, and trunc thresholding.")
        gamma_value = st.slider("Gamma", min_value=0.5, max_value=2.5, value=1.2, step=0.1, help="Gamma correction for illumination compensation.")
        clip_limit = st.slider("CLAHE Clip Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    with st.expander("Transformations", expanded=False):
        scale_value = st.slider("Scaling Factor", min_value=0.6, max_value=1.8, value=1.15, step=0.05)
        rotation_value = st.slider("Rotation", min_value=-45, max_value=45, value=12, step=1)
        translate_x = st.slider("Translate X", min_value=-120, max_value=120, value=25, step=5)
        translate_y = st.slider("Translate Y", min_value=-120, max_value=120, value=18, step=5)

    with st.expander("Detection & Review", expanded=True):
        run_yolo = st.checkbox("Run YOLOv8 Detection", value=True, help="Detect cars, bikes, trucks, buses, and pedestrians.")
        preview_family = st.selectbox(
            "Processed Output Family",
            ["Preprocessing", "Thresholding", "Morphology", "Transformations", "Segmentation", "Detection", "Features"],
            help="Select the pipeline family shown in the processed-output viewport.",
        )
        show_pipeline = st.checkbox("Show End-to-End Pipeline", value=True)

    st.markdown(
        """
        <div class='research-panel'>
            <div class='panel-kicker'>LAB BEHAVIOR</div>
            <div class='panel-note'>Algorithms run on CPU-safe OpenCV and lightweight numpy pipelines. Heavy inference is limited to YOLOv8 when enabled.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "img_np" not in st.session_state:
    st.warning("No active image feed available. Upload an image here or analyze one in Image Analysis Center.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

image_pil = st.session_state.get("uploaded_image")
if image_pil is None:
    image_pil = Image.fromarray(st.session_state.img_np)

img_np = st.session_state.img_np
gray = PreprocessingPipeline.to_grayscale(img_np)
source_name = st.session_state.get("cv_lab_source_name", "Session Feed")


def _np_hash(arr):
    """Fast perceptual hash for cache keying — downsamples before MD5."""
    small = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


@st.cache_data(show_spinner=False, max_entries=5, hash_funcs={np.ndarray: _np_hash})
def _compute_heavy_cv(img_arr):
    """Segmentation + feature extraction — only keyed on image, not sliders."""
    _seg = SegmentationEngine.full_segmentation(img_arr)
    _fea = FeatureExtractor.full_feature_pack(img_arr)
    return _seg, _fea


@st.cache_data(show_spinner=False, max_entries=20, hash_funcs={np.ndarray: _np_hash})
def _compute_preproc_cv(img_arr, k_size, thr_val, gamma, clip, scale, rot, tx, ty):
    """Preprocessing variants — re-runs only when image OR sliders change."""
    _gray = PreprocessingPipeline.to_grayscale(img_arr)
    _pre = PreprocessingPipeline.preprocessing_variants(img_arr, kernel_size=k_size, gamma=gamma, clip_limit=clip)
    _thr = {n: PreprocessingPipeline.ensure_rgb(v) for n, v in PreprocessingPipeline.threshold_variants(_gray, threshold_value=thr_val).items()}
    _mor = {n: PreprocessingPipeline.ensure_rgb(v) for n, v in PreprocessingPipeline.morphology_variants(_gray, kernel_size=k_size).items()}
    _trn = PreprocessingPipeline.transform_variants(img_arr, scale=scale, rotation=rot, translate_x=tx, translate_y=ty)
    return _pre, _thr, _mor, _trn


# Resize large images for faster lab processing (session state unchanged)
_LAB_MAX = 800
_lh, _lw = img_np.shape[:2]
if max(_lh, _lw) > _LAB_MAX:
    _ls = _LAB_MAX / max(_lh, _lw)
    _lab_img = cv2.resize(img_np, (int(_lw * _ls), int(_lh * _ls)), interpolation=cv2.INTER_AREA)
else:
    _lab_img = img_np

segmentation_outputs, feature_pack = _compute_heavy_cv(_lab_img)
preprocess_outputs, threshold_outputs, morphology_outputs, transform_outputs = _compute_preproc_cv(
    _lab_img, kernel_size, threshold_value, gamma_value, clip_limit,
    scale_value, rotation_value, translate_x, translate_y,
)

st.session_state.segmentation_outputs = segmentation_outputs
st.session_state.feature_pack = feature_pack

image_sig = image_signature(img_np)
detection_image = img_np.copy()
vision_results = st.session_state.get("vision_results", {})

if run_yolo:
    if st.session_state.get("cv_lab_detection_sig") != image_sig:
        with st.spinner("Running YOLOv8 traffic detection..."):
            _lab_pil = Image.fromarray(_lab_img)
            detection_image, vision_results = get_vision_service().analyze_image(_lab_pil)
        st.session_state.cv_lab_detection_sig = image_sig
        st.session_state.cv_lab_detection_output = detection_image
        st.session_state.cv_lab_detection_results = vision_results
        st.session_state.vision_results = vision_results
    detection_image = st.session_state.get("cv_lab_detection_output", img_np)
    vision_results = st.session_state.get("cv_lab_detection_results", vision_results)

feature_images = {
    "HOG": feature_pack["HOG Descriptors"],
    "ORB": feature_pack["ORB Keypoints"],
    "SIFT": feature_pack["SIFT Keypoints"],
    "SURF": feature_pack["SURF Keypoints"],
    "Contours": feature_pack["Contours"],
    "Blob Detection": feature_pack["Blob Detection"],
    "Shape Recognition": feature_pack["Shape Recognition"],
}

family_outputs = {
    "Preprocessing": preprocess_outputs,
    "Thresholding": threshold_outputs,
    "Morphology": morphology_outputs,
    "Transformations": transform_outputs,
    "Segmentation": {
        key: value
        for key, value in segmentation_outputs.items()
        if isinstance(value, np.ndarray)
    },
    "Detection": {"YOLOv8 Overlay": detection_image},
    "Features": feature_images,
}
selected_names = list(family_outputs[preview_family].keys())

with right_col:
    render_section_header("PROCESSED OUTPUT")
    processed_name = st.selectbox(
        "Output Variant",
        selected_names,
        key="cv_lab_processed_name",
        help="Choose the exact output shown in the processed viewport.",
    )

selected_output = family_outputs[preview_family][processed_name]

with center_col:
    render_section_header("ORIGINAL IMAGE")
    render_image_card("Source Feed", img_np, caption=f"{source_name} | {img_np.shape[1]}x{img_np.shape[0]} | {image_pil.mode}")
    st.markdown("<div class='metric-strip'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>RESOLUTION</div><div class='metric-tile-value'>{img_np.shape[1]}x{img_np.shape[0]}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>CHANNELS</div><div class='metric-tile-value'>{img_np.shape[2] if img_np.ndim == 3 else 1}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>MEAN INTENSITY</div><div class='metric-tile-value'>{np.mean(gray):.1f}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>CONTRAST STD</div><div class='metric-tile-value'>{np.std(gray):.1f}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    render_image_card(processed_name, selected_output, caption=f"Family: {preview_family}")
    st.markdown("<div class='metric-strip'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>LANE OCCUPANCY</div><div class='metric-tile-value'>{segmentation_outputs.get('lane_occupancy', 0.0) * 100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>WATERSHED</div><div class='metric-tile-value'>{segmentation_outputs.get('watershed_regions', 0)}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>ORB KEYPOINTS</div><div class='metric-tile-value'>{feature_pack.get('orb_count', 0)}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-tile'><div class='metric-tile-label'>BLOBS</div><div class='metric-tile-value'>{feature_pack.get('blob_count', 0)}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

render_section_header("MULTI-STAGE RESEARCH GRID")
tab_pre, tab_thr, tab_morph, tab_transform, tab_seg, tab_feat, tab_color, tab_detect, tab_pipeline = st.tabs(
    [
        "Preprocessing",
        "Thresholding",
        "Morphology",
        "Transformations",
        "Segmentation",
        "Features",
        "Color Analysis",
        "Detection",
        "Pipeline",
    ]
)

with tab_pre:
    render_image_grid(preprocess_outputs, columns=3)

with tab_thr:
    render_image_grid(threshold_outputs, columns=3)

with tab_morph:
    render_image_grid(morphology_outputs, columns=4)

with tab_transform:
    render_image_grid(transform_outputs, columns=3)

with tab_seg:
    render_image_grid(
        {
            "Threshold Segmentation": segmentation_outputs["Threshold Segmentation"],
            "Region-Based Segmentation": segmentation_outputs["Region Segmentation"],
            "Watershed Segmentation": segmentation_outputs["Watershed Segmentation"],
            "K-Means Segmentation": segmentation_outputs["K-Means Segmentation"],
            "Color-Based Segmentation": segmentation_outputs["Color-Based Segmentation"],
        },
        columns=3,
    )
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Lane Occupancy", f"{segmentation_outputs.get('lane_occupancy', 0.0) * 100:.1f}%")
    sm2.metric("Watershed Regions", segmentation_outputs.get("watershed_regions", 0))
    sm3.metric("Color Coverage", f"{segmentation_outputs.get('color_mask_coverage', 0.0) * 100:.1f}%")

with tab_feat:
    render_image_grid(feature_images, columns=3)
    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fm1.metric("ORB", feature_pack.get("orb_count", 0))
    fm2.metric("SIFT", feature_pack.get("sift_count", 0) if feature_pack.get("sift_available") else "N/A")
    fm3.metric("SURF", feature_pack.get("surf_count", 0) if feature_pack.get("surf_available") else "N/A")
    fm4.metric("Contours", feature_pack.get("contour_count", 0))
    fm5.metric("Blobs", feature_pack.get("blob_count", 0))
    shape_df = pd.DataFrame(
        [{"Shape": key, "Count": value} for key, value in feature_pack.get("shape_stats", {}).items()]
    )
    st.dataframe(shape_df, use_container_width=True, hide_index=True)

with tab_color:
    color_profile = feature_pack.get("color_profile", {})
    dominant_rgb = color_profile.get("dominant_rgb", [0, 0, 0])
    c1, c2 = st.columns([0.8, 1.2])
    with c1:
        swatch = np.zeros((140, 220, 3), dtype=np.uint8)
        swatch[:] = dominant_rgb
        render_image_card("Dominant Color Swatch", swatch, caption=f"{color_profile.get('dominant_color', 'Unknown')} | RGB {dominant_rgb}")
        if run_yolo and vision_results.get("vehicles"):
            vehicle_df = pd.DataFrame(
                [
                    {
                        "Type": item.get("vehicle_type", "Vehicle"),
                        "Color": item.get("vehicle_color", "Unknown"),
                        "Model": item.get("vehicle_model", "Unknown"),
                        "Confidence": round(item.get("conf", 0.0), 2),
                    }
                    for item in vision_results.get("vehicles", [])
                ]
            )
            st.dataframe(vehicle_df, use_container_width=True, hide_index=True)
        else:
            st.info("Vehicle color detection populates here when YOLOv8 analysis is enabled.")
    with c2:
        st.plotly_chart(build_histogram_figure(color_profile.get("histograms", {})), use_container_width=True)

with tab_detect:
    render_image_card("YOLOv8 Traffic Detection", detection_image)
    if run_yolo and vision_results:
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Cars / Bikes / Trucks", vision_results.get("vehicle_count", 0))
        d2.metric("Pedestrians", vision_results.get("pedestrian_count", 0))
        d3.metric("Density", vision_results.get("density_level", "LOW"))
        d4.metric("Road Condition", vision_results.get("road_condition", "UNKNOWN"))
        d5.metric("Vehicle Spacing", f"{vision_results.get('vehicle_spacing', 0.0):.1f}px")
    else:
        st.info("Enable YOLOv8 Detection in the control panel to overlay traffic object detections.")

with tab_pipeline:
    pipeline_outputs = {
        "Original": img_np,
        "Preprocessing": preprocess_outputs.get("CLAHE", img_np),
        "Segmentation": segmentation_outputs["K-Means Segmentation"],
        "Detection": detection_image,
        "Final Output": selected_output,
    }
    if show_pipeline:
        render_image_grid(pipeline_outputs, columns=5)
    else:
        st.info("Enable 'Show End-to-End Pipeline' in the control panel to display the full stage chain.")

st.markdown("</div>", unsafe_allow_html=True)