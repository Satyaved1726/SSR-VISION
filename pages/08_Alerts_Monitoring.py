import streamlit as st
from ui.components import render_global_header, check_state_auth, render_section_header, render_alert, render_status_bar

st.set_page_config(page_title="SSR VISION | Alerts", page_icon="⚠️", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="ALERT MONITOR")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### ⚠️ 08_ ALERTS & MONITORING CONSOLE")

alerts = st.session_state.alerts

if not alerts:
    st.success("🟢 SYS NORMAL. NO PENDING SECURITY OR TRAFFIC ALERTS.")
else:
    render_section_header(f"{len(alerts)} ACTIVE INTELLIGENCE ALERTS")
    for alert in alerts:
        level = alert.get("level", "info")
        msg = alert.get("msg", "")
        if level == "danger":
            level = "critical"
        render_alert(f"[TRX-{hash(msg) % 10000}] {msg}", level)

st.markdown('</div>', unsafe_allow_html=True)
