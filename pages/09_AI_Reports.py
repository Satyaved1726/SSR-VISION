import streamlit as st
from datetime import datetime
from ui.components import render_global_header, check_state_auth, render_status_bar, render_section_header, render_metric_card
from core.reporting import build_intelligence_pdf

st.set_page_config(page_title="SSR VISION | Reports", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="REPORTING")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 📄 09_ AI INTELLIGENCE BRIEFING")

report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
vision = st.session_state.vision_results
web_data = st.session_state.web_data
plates = st.session_state.get("plates", [])
violations = vision.get("violations", [])

st.markdown(
    f"<div class='holographic-card' style='padding:22px; margin-bottom:16px;'>"
    f"<div style='text-align:center; color:#ff4f6a; font-family:Orbitron, sans-serif; font-size:2.1rem; letter-spacing:2px;'>CYBER POLICE TRAFFIC DOSSIER</div>"
    f"<div style='text-align:center; color:#00f7ff; font-family:Orbitron, sans-serif; font-size:1.25rem; margin-top:10px;'>SSR VISION INCIDENT BRIEFING</div>"
    f"<div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-top:18px; font-family:monospace; color:#d9f8ff;'>"
    f"<span>Case Generated: {report_date}</span>"
    f"<span>Security Level: CYBER POLICE / INTERNAL</span>"
    f"<span>Region: {st.session_state.get('system_settings', {}).get('region', 'Hyderabad')}</span>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Risk Score", f"{st.session_state.risk_score}/100")
with m2:
    render_metric_card("Vehicles", vision.get("vehicle_count", 0))
with m3:
    render_metric_card("Violations", len(violations))
with m4:
    render_metric_card("Plates", len(plates))

render_section_header("EXECUTIVE SUMMARY")
st.markdown(f"<div class='intel-feed-card'>{st.session_state.caption}</div>", unsafe_allow_html=True)

render_section_header("FIELD INTELLIGENCE")
left, right = st.columns(2)
with left:
    st.markdown(f"**Traffic Density:** {vision.get('density_level', 'LOW')}")
    st.markdown(f"**Road Condition:** {vision.get('road_condition', 'UNKNOWN')}")
    st.markdown(f"**Weather:** {web_data.get('weather', 'N/A')}")
    st.markdown(f"**Fusion Insight:** {st.session_state.fusion_insight}")
with right:
    st.markdown(f"**Advisories:** {', '.join(web_data.get('advisories', [])) if web_data.get('advisories') else 'None'}")
    st.markdown(f"**Accident Locations:** {', '.join(web_data.get('accident_locations', [])) if web_data.get('accident_locations') else 'None'}")
    st.markdown(f"**Detected Plates:** {', '.join(plates) if plates else 'None'}")

render_section_header("VIOLATION LOG")
if violations:
    for item in violations:
        st.markdown(f"<div class='alert-card critical'><div class='alert-level'>CRITICAL</div><div>{item}</div></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='alert-card'><div class='alert-level'>INFO</div><div>No confirmed violations detected in the current scene.</div></div>", unsafe_allow_html=True)

st.download_button(
    label="EXPORT DOSSIER [TXT]", 
    data=f"SSR VISION INTELLIGENCE BRIEFING\nDATE: {report_date}\n\nSUMMARY: {st.session_state.caption}\nRISK SCORE: {st.session_state.risk_score}\nVEHICLES: {st.session_state.vision_results['vehicle_count']}\n",
    file_name=f"ssr_vision_report_{report_date.split()[0]}.txt",
    mime="text/plain"
)

pdf_bytes = build_intelligence_pdf(st.session_state)
st.download_button(
    label="EXPORT INTELLIGENCE BRIEFING [PDF]",
    data=pdf_bytes,
    file_name=f"ssr_vision_report_{report_date.split()[0]}.pdf",
    mime="application/pdf",
)

st.markdown('</div>', unsafe_allow_html=True)
