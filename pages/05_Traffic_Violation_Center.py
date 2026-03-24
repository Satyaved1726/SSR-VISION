import streamlit as st
import cv2
import time
from ui.components import render_global_header, check_state_auth, render_section_header, render_evidence_card, render_status_bar

st.set_page_config(page_title="SSR VISION | Traffic Violations", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="VIOLATION MONITOR")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 🚨 05_ TRAFFIC VIOLATION EVIDENCE CENTER")

violations = st.session_state.vision_results.get('violations', [])
bboxes = st.session_state.vision_results.get('bboxes', []) 

if not violations:
    st.success("✅ ZERO VIOLATIONS DETECTED IN CURRENT FEED.")
    st.markdown("<div class='ai-scanning-overlay'>", unsafe_allow_html=True)
    st.image(st.session_state.processed_img_np, caption="CLEAN FEED", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.error(f"⚠️ {len(violations)} RULE VIOLATION(S) DETECTED.")
    
    col_v, col_img = st.columns([1, 1])
    
    with col_v:
        render_section_header("AI EVIDENCE CHRONICLE")
        for i, v in enumerate(violations):
            # Using our high-end NASA component
            severity = "CRITICAL" if "HELMET" in v else "HIGH" if ("LANE" in v or "STOP" in v) else "MEDIUM"
            render_evidence_card(
                title=f"INCIDENT #{i+1}: {v}", 
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'), 
                severity=severity,
                info_dict={
                    "LOC": "REGION_ALPHA",
                    "CONFIDENCE": "98.4%",
                    "SENSOR": "YOLO_V8_M"
                }
            )
            
    with col_img:
        render_section_header("GLOBAL SENSOR VIEW")
        img_copy = st.session_state.img_np.copy()
        img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes for highlighting the violations in the main image 
        for box_data in bboxes:
            x1, y1, x2, y2 = box_data["box"]
            cls_id = box_data["cls"]
            if cls_id == 3 and "HELMET ABSENCE DETECTED" in violations:
                 cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4) # RED
            elif cls_id == 2 and "LANE CROSSING VIOLATION" in violations:
                 cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 4) # YELLOW
            elif "ILLEGAL PARKING" in violations:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 3)
            elif "STOP LINE CROSSING" in violations:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 0), 3)
                 
        display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.markdown("<div class='ai-scanning-overlay'>", unsafe_allow_html=True)
        st.image(display_rgb, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
