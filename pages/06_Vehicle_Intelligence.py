import streamlit as st
import cv2

from ui.components import render_global_header, check_state_auth, render_section_header, render_status_bar
from core.vehicle_lookup import lookup_owner_details, PORTAL_LINKS

st.set_page_config(page_title="SSR VISION | Vehicle Intelligence", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="VEHICLE INTEL")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 🚗 06_ VEHICLE INTELLIGENCE")

vehicles = st.session_state.vision_results.get("vehicles", [])
plates = st.session_state.get("plates", [])
violations = st.session_state.vision_results.get("violations", [])
vehicle_count = st.session_state.vision_results.get("vehicle_count", 0)
img = st.session_state.img_np.copy()
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

if not vehicles:
    st.info("No vehicles detected in current scene.")
else:
    left, right = st.columns([1.1, 1.2])

    with left:
        render_section_header("Vehicle Intelligence Panel")
        for idx, v in enumerate(vehicles[:12], start=1):
            plate = plates[idx - 1] if idx - 1 < len(plates) else "Not Detected"
            st.markdown(f"""
            <div class='intel-feed-card'>
                <b>Vehicle #{idx}</b><br>
                Type: {v.get('vehicle_type', 'Vehicle')}<br>
                Color: {v.get('vehicle_color', 'Unknown')}<br>
                Detected Plate: {plate}
            </div>
            """, unsafe_allow_html=True)

    with right:
        render_section_header("Annotated Vehicle Feed")
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            label = f"{v.get('vehicle_type', 'Vehicle')} | {v.get('vehicle_color', 'Unknown')} | {v.get('vehicle_model', 'Unknown')}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 243, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 243, 0), 2)

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    render_section_header("Vehicle Count & Summary")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Detected Vehicles", vehicle_count)
    with s2:
        st.metric("Detected Plates", len(plates))
    with s3:
        st.metric("Detected Violations", len(violations))

render_section_header("Owner Lookup Demo Module")
selected_plate = st.selectbox("Select detected plate", options=plates if plates else ["No plate detected"])
if selected_plate and selected_plate != "No plate detected":
    owner = lookup_owner_details(selected_plate)
    if owner:
        st.success("Demo registry match found")
        st.write({
            "Vehicle Number": selected_plate,
            "Owner Name": owner["owner_name"],
            "Vehicle Model": owner["vehicle_model"],
            "City": owner["city"],
        })
    else:
        st.warning("No demo registry match available for this plate")

st.markdown("### Vehicle Information Lookup")
st.markdown("- Telangana e-Challan Portal")
st.markdown("- VAHAN Vehicle Registry")

c1, c2 = st.columns(2)
with c1:
    st.link_button("Check Challan Details", PORTAL_LINKS["challan"], use_container_width=True)
with c2:
    st.link_button("Check Vehicle Info", PORTAL_LINKS["vahan"], use_container_width=True)

if violations:
    st.markdown(f"**Detected Violations:** {', '.join(violations)}")
else:
    st.markdown("**Detected Violations:** None")

st.markdown('</div>', unsafe_allow_html=True)
