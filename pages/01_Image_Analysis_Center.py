import streamlit as st
from ui.components import render_global_header, render_section_header, render_status_bar

# Set page config FIRST on every page
st.set_page_config(
    page_title="SSR VISION | Image Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()

import time
import numpy as np
from PIL import Image
import cv2

# Core Modules
from core.vision import VisionAnalyzer
from core.ocr import PlateRecognizer
from core.web_mining import WebDataMiner
from core.analytics import AnalyticsEngine
from core.segmentation import SegmentationEngine

@st.cache_resource
def get_vision_service():
    return VisionAnalyzer()

@st.cache_resource
def get_ocr_service():
    return PlateRecognizer()


@st.cache_resource
def get_web_miner():
    return WebDataMiner()


def build_detection_overlay(image_rgb, vision_results, plate_regions):
    overlay_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for item in vision_results.get("vehicles", []):
        x1, y1, x2, y2 = item["box"]
        label = item.get("vehicle_type", "Vehicle")
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 243, 0), 2)
        cv2.putText(overlay_bgr, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 243, 0), 2)

    for x, y, w, h in plate_regions:
        cv2.rectangle(overlay_bgr, (x, y), (x + w, y + h), (0, 255, 120), 2)
        cv2.putText(overlay_bgr, "PLATE ROI", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 2)

    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def render_analysis_results():
    if "processed_img_np" not in st.session_state:
        return

    vision_results = st.session_state.vision_results
    plates = st.session_state.get("plates", [])
    render_section_header("MISSION OUTPUT")
    top_left, top_right = st.columns([1.3, 0.9])
    with top_left:
        st.markdown("<div class='edge-card'><h4>AI Processed Feed</h4></div>", unsafe_allow_html=True)
        st.markdown("<div class='ai-scanning-overlay'>", unsafe_allow_html=True)
        st.image(st.session_state.processed_img_np, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        st.markdown("<div class='edge-card'><h4>AI Diagnostics</h4></div>", unsafe_allow_html=True)
        st.markdown(f"**Risk Score:** {st.session_state.risk_score}/100")
        st.markdown(f"**Density:** {vision_results['density_level']}")
        st.markdown(f"**Vehicles:** {vision_results['vehicle_count']}")
        st.markdown(f"**Violations:** {len(vision_results['violations'])}")
        st.markdown(f"**Plates:** {len(plates)}")
        st.markdown(f"**Weather:** {st.session_state.web_data.get('weather', 'N/A')}")
        st.markdown(f"**Fusion Insight:** {st.session_state.fusion_insight}")
        if plates:
            st.markdown(f"**Detected Plate(s):** {', '.join(plates[:4])}")
        else:
            st.markdown("**Detected Plate(s):** No plate detected")

    render_section_header("INTELLIGENCE VIEWPORT")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("<div class='edge-card'><h4>Original Image</h4></div>", unsafe_allow_html=True)
        st.image(st.session_state.img_np, use_column_width=True)
    with d2:
        st.markdown("<div class='edge-card'><h4>Processed Image</h4></div>", unsafe_allow_html=True)
        st.image(st.session_state.processed_img_np, use_column_width=True)
    with d3:
        st.markdown("<div class='edge-card'><h4>Detection and Plate Overlay</h4></div>", unsafe_allow_html=True)
        st.markdown("<div class='ai-scanning-overlay'>", unsafe_allow_html=True)
        st.image(st.session_state.detection_overlay_np, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    meta1, meta2, meta3 = st.columns(3)
    meta1.markdown(f"**Lane Occupancy:** {int(vision_results.get('lane_occupancy', 0.0) * 100)}%")
    meta2.markdown(f"**Vehicle Spacing:** {vision_results.get('vehicle_spacing', 0.0):.1f}px")
    meta3.markdown(f"**Road Condition:** {vision_results.get('road_condition', 'UNKNOWN')}")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 📡 01_ IMAGE ANALYSIS CENTER")
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="READY FOR SCAN")
uploaded_file = st.file_uploader("UPLOAD TRAFFIC FEED (IMAGE)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)

    left, right = st.columns([1.35, 0.85])
    with left:
        render_section_header("RAW OP-FEED")
        st.image(image, use_column_width=True)
    with right:
        render_section_header("SCAN CONTROL")
        st.info("Uploader is available immediately. Heavy AI models load only after you start a scan.")
        analyze_clicked = st.button("INITIALIZE SECURE ANALYSIS", use_container_width=True)
        st.markdown("**Live status:** Waiting for operator command")
        st.markdown("**Mode:** GPU-safe / lightweight render path")

    if analyze_clicked:
        with st.spinner("RUNNING AI DIAGNOSTICS & COMPUTER VISION SCANS..."):
            progress_bar = st.progress(0)
            info_text = st.empty()

            info_text.text("Loading detection engine...")
            vision_service = get_vision_service()
            progress_bar.progress(10)

            info_text.text("Executing object detection...")
            processed_img_np, vision_results = vision_service.analyze_image(image)
            progress_bar.progress(35)

            info_text.text("Loading OCR engine...")
            ocr_service = get_ocr_service()
            progress_bar.progress(45)

            info_text.text("Extracting number plates...")
            plate_bundle = ocr_service.run_plate_pipeline(img_np, candidate_boxes=vision_results.get("vehicles", []))
            plates = plate_bundle["plates"]
            progress_bar.progress(58)

            info_text.text("Running lightweight road occupancy scan...")
            threshold_img, occupancy = SegmentationEngine.threshold_segmentation(img_np)
            segmentation_outputs = {
                "Threshold Segmentation": threshold_img,
                "lane_occupancy": round(occupancy, 3),
            }
            vision_results["lane_occupancy"] = segmentation_outputs.get("lane_occupancy", vision_results.get("lane_occupancy", 0.0))
            progress_bar.progress(72)

            info_text.text("Collecting live weather and traffic context...")
            web_data = get_web_miner().mine_weather_and_traffic()
            progress_bar.progress(88)

            info_text.text("Calculating intelligence score...")
            risk_score = AnalyticsEngine.calculate_risk_score(vision_results, web_data)
            alerts = AnalyticsEngine.generate_alerts(vision_results, web_data, risk_score)
            caption = AnalyticsEngine.generate_insight_caption(vision_results, web_data, risk_score)
            fusion_insight = AnalyticsEngine.fuse_intelligence(vision_results, web_data, risk_score)

            st.session_state.uploaded_image = image
            st.session_state.img_np = img_np
            st.session_state.processed_img_np = processed_img_np
            st.session_state.vision_results = vision_results
            st.session_state.plates = plates
            st.session_state.plate_regions = plate_bundle["regions"]
            st.session_state.web_data = web_data
            st.session_state.risk_score = risk_score
            st.session_state.alerts = alerts
            st.session_state.caption = caption
            st.session_state.fusion_insight = fusion_insight
            st.session_state.segmentation_outputs = segmentation_outputs
            st.session_state.feature_pack = st.session_state.get("feature_pack", {})
            st.session_state.detection_overlay_np = build_detection_overlay(img_np, vision_results, plate_bundle["regions"])

            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
            info_text.empty()

        st.success("🛰️ UPLINK SECURED. FEED ANALYZED.")

    if 'uploaded_image' in st.session_state and 'processed_img_np' in st.session_state:
        render_analysis_results()
else:
    st.info(">> AWAITING SATELLITE IMAGE UPLOAD...")

st.markdown('</div>', unsafe_allow_html=True)
