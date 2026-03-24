import streamlit as st
from ui.components import render_global_header, check_state_auth, render_section_header, render_alert, render_status_bar
import cv2
import numpy as np

st.set_page_config(page_title="SSR VISION | Road Intel", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="ROAD INTEL")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 🛣️ 04_ ROAD INTELLIGENCE")

road_cond = st.session_state.vision_results.get('road_condition', 'UNKNOWN')

col1, col2 = st.columns([1, 1])

with col1:
    render_section_header("INFRASTRUCTURE STATUS")
    
    if road_cond == "GOOD":
        render_alert("ROAD INTEGRITY: OPTIMAL (NO ANOMALIES DETECTED)", "info")
    elif road_cond == "FADED MARKINGS":
        render_alert("INFRASTRUCTURE WARNING: DEGRADED LANE MARKINGS DETECTED", "warning")
    else:
        render_alert("HAZARD DETECTED: STRUCTURAL DAMAGE / POTHOLES / OBSTRUCTIONS FOUND", "critical")
        
    st.markdown("""
    #### INTELLIGENCE REPORT
    The system analyzes the lower 40% of the image frame corresponding to the immediate road surface using specialized gradient analysis to detect anomalous textures indicative of potholes or major cracks.
    """)

    vr = st.session_state.vision_results
    occupancy_pct = int(vr.get("lane_occupancy", 0.0) * 100)
    st.markdown(f"**Lane Occupancy:** {occupancy_pct}%")
    st.markdown(f"**Estimated Vehicle Spacing:** {vr.get('vehicle_spacing', 0.0):.1f}px")

    # Condition flags derived from road_condition and vision results
    potholes = "YES" if road_cond == "DAMAGED/OBSTRUCTED" else "NO"
    cracks = "YES" if road_cond == "DAMAGED/OBSTRUCTED" else ("LIKELY" if road_cond == "FADED MARKINGS" else "NO")
    lane_deg = "HIGH" if road_cond == "FADED MARKINGS" else ("MEDIUM" if road_cond == "DAMAGED/OBSTRUCTED" else "LOW")
    obstruction = "POSSIBLE" if vr.get("density_level") in ["HIGH", "CRITICAL"] else "LOW"

    st.markdown("### Condition Flags")
    st.markdown(f"- Potholes: {potholes}")
    st.markdown(f"- Road Cracks: {cracks}")
    st.markdown(f"- Lane Marking Degradation: {lane_deg}")
    st.markdown(f"- Road Obstruction: {obstruction}")

with col2:
    render_section_header("ROAD SURFACE HEATMAP (CRACK DETECTION)")

    img_bgr = cv2.cvtColor(st.session_state.img_np.copy(), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    road_roi = img_bgr[int(h * 0.6):h, :]

    # Crack detection using morphological black-hat + adaptive thresholding
    gray_road = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_road)

    # Black-hat: isolates thin dark structures (cracks) against bright road surface
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    _, crack_mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)

    # Highlight detected cracks in orange
    overlay = road_roi.copy()
    overlay[crack_mask > 0] = [0, 120, 255]  # Orange in BGR
    blended = cv2.addWeighted(road_roi, 0.65, overlay, 0.55, 0)
    img_bgr[int(h * 0.6):h, :] = blended

    # Count crack pixels for reporting
    crack_pixel_ratio = float(np.sum(crack_mask > 0)) / float(crack_mask.size)
    crack_detected = crack_pixel_ratio > 0.01

    display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(display_rgb, use_column_width=True)
    if crack_detected:
        st.markdown(f"<div style='color:#ff6600;font-family:monospace;font-size:0.85rem;'>⚠ CRACK COVERAGE: {crack_pixel_ratio*100:.1f}% of road ROI — orange highlights indicate fracture zones.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#00ff66;font-family:monospace;font-size:0.85rem;'>✓ No significant crack patterns detected in road surface ROI.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
