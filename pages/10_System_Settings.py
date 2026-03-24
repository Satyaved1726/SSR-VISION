import streamlit as st
from ui.components import render_global_header, render_section_header, render_status_bar

st.set_page_config(page_title="SSR VISION | Settings", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="SETTINGS")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### ⚙️ 10_ SYSTEM SETTINGS")

st.markdown("Configure operational parameters for the SSR VISION platform. All configurations are stored locally.")

if "system_settings" not in st.session_state:
    st.session_state.system_settings = {
        "detector": "YOLOv8 Nano (Speed)",
        "confidence": 0.25,
        "ocr_region": "English (en)",
        "theme": "Cyber Neon (Default)",
        "animations": True,
        "labels": True,
        "gpu_safe_mode": True,
        "region": "Hyderabad",
    }

settings = st.session_state.system_settings

col1, col2 = st.columns(2)

with col1:
    render_section_header("AI MODEL CONFIGURATION")
    detector = st.selectbox("Object Detection Engine", ["YOLOv8 Nano (Speed)", "YOLOv8 Small (Balanced)", "YOLOv8 Medium (Accuracy)"], index=["YOLOv8 Nano (Speed)", "YOLOv8 Small (Balanced)", "YOLOv8 Medium (Accuracy)"].index(settings["detector"]))
    confidence = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=float(settings["confidence"]), step=0.05)
    ocr_region = st.selectbox("OCR Engine Region", ["English (en)", "European (eu)", "Global (auto)"], index=["English (en)", "European (eu)", "Global (auto)"].index(settings["ocr_region"]))
    region = st.text_input("Live Weather Region", value=settings.get("region", "Hyderabad"))

with col2:
    render_section_header("UI & THEME PREFERENCES")
    theme = st.radio("System Theme", ["Cyber Neon (Default)", "High Contrast Terminal", "Midnight Stealth"], index=["Cyber Neon (Default)", "High Contrast Terminal", "Midnight Stealth"].index(settings["theme"]))
    animations = st.checkbox("Enable Animations & Holographic Effects", value=bool(settings["animations"]))
    labels = st.checkbox("Show Bounding Box Labels", value=bool(settings["labels"]))
    gpu_safe_mode = st.checkbox("Enable GPU-safe mode (recommended)", value=bool(settings.get("gpu_safe_mode", True)))
    
if st.button("SAVE SYSTEM CONFIGURATION"):
    st.session_state.system_settings = {
        "detector": detector,
        "confidence": confidence,
        "ocr_region": ocr_region,
        "theme": theme,
        "animations": animations,
        "labels": labels,
        "gpu_safe_mode": gpu_safe_mode,
        "region": region,
    }
    st.success("⚙️ SECURITY PROTOCOLS UPDATED. SETTINGS SAVED TO LOCAL CACHE.")

st.markdown('</div>', unsafe_allow_html=True)
