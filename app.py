import streamlit as st
import time
import numpy as np
import plotly.express as px
from ui.components import render_global_header, render_metric_card, render_section_header, render_status_bar, render_timeline

# --- PAGE 1: MAIN DASHBOARD ---
st.set_page_config(
    page_title="SSR VISION | Command Center",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
active_alerts = len(st.session_state.get("alerts", []))
render_status_bar(active_alerts=active_alerts, processing="MONITORING")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 🌐 HIGH-LEVEL SYSTEM OVERVIEW")

if 'vision_results' not in st.session_state:
    st.info(">> INTELLIGENCE FEED OFFLINE. NAVIGATE TO [01 - IMAGE ANALYSIS CENTER] TO INITIATE UPLINK.")
    
    # Show mock/empty state dashboard
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("RISK SCORE", "STANDBY")
    with c2: render_metric_card("VEHICLES", "0")
    with c3: render_metric_card("DENSITY", "UNKNOWN")
    with c4: render_metric_card("VIOLATIONS", "0")
    
else:
    vr = st.session_state.vision_results
    risk = st.session_state.risk_score
    
    # Dynamic live metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: 
        status = "danger" if risk > 75 else "warning" if risk > 40 else "normal"
        render_metric_card("RISK SCORE", f"{risk}/100", status)
    with c2: 
        render_metric_card("VEHICLES", vr["vehicle_count"])
    with c3: 
        dens = vr["density_level"]
        render_metric_card("DENSITY", dens, "danger" if dens == "HIGH" else "warning" if dens == "MEDIUM" else "normal")
    with c4: 
        viol_count = len(vr["violations"])
        render_metric_card("VIOLATIONS", viol_count, "danger" if viol_count > 0 else "normal")
    with c5:
        road = vr["road_condition"]
        render_metric_card("ROAD INTEL", "ANOMALY" if road != "GOOD" else "SECURE", "danger" if road != "GOOD" else "normal")
    with c6:
        render_metric_card("PEDESTRIANS", vr.get("pedestrian_count", 0))

    c7, c8 = st.columns(2)
    with c7:
        render_metric_card("LANE OCCUPANCY", f"{int(vr.get('lane_occupancy', 0.0)*100)}%")
    with c8:
        render_metric_card("VEHICLE SPACING", f"{vr.get('vehicle_spacing', 0.0):.1f}px")

    render_section_header("RECENT ACTIVITY LOG")
    now = time.strftime('%H:%M:%S')
    events = [
        (now, "Satellite feed uplink secured", "#00ff66"),
        (now, f"YOLOv8 completed - {vr['vehicle_count']} entities mapped", "#00f3ff"),
        (now, "WDM synchronization complete", "#ffb700"),
        (now, f"Traffic risk score computed - {risk}/100", "#ff003c" if risk > 70 else "#00ff66"),
    ]
    render_timeline(events)

    render_section_header("SYSTEM TELEMETRY PANELS")
    t1, t2, t3 = st.columns(3)
    with t1:
        density_base = 30 if vr.get("density_level") == "LOW" else 55 if vr.get("density_level") == "MEDIUM" else 75
        density_trend = np.clip(np.cumsum(np.random.randn(24) * 2.2) + density_base, 0, 100)
        fig = px.line(y=density_trend, color_discrete_sequence=["#00f3ff"])
        fig.update_layout(
            title="Traffic Density Trend",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e0faff",
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis_title="Time Slice",
            yaxis_title="Density Index",
        )
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        dist = vr.get("type_distribution", {})
        if dist:
            fig2 = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                color=list(dist.keys()),
                color_discrete_sequence=["#00f3ff", "#00ff66", "#ffb700", "#ff6a3d", "#7fffd4"],
                labels={"x": "Vehicle Type", "y": "Count"},
            )
            fig2.update_layout(
                title="Vehicle Distribution",
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#e0faff",
                margin=dict(l=8, r=8, t=36, b=8),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Awaiting class distribution signals.")
    with t3:
        violation_items = vr.get("violations", [])
        if violation_items:
            grouped = {}
            for item in violation_items:
                key = item.split(" ")[0]
                grouped[key] = grouped.get(key, 0) + 1
            fig3 = px.pie(
                values=list(grouped.values()),
                names=list(grouped.keys()),
                hole=0.55,
                color_discrete_sequence=["#ff4f6a", "#ffb700", "#00f7ff", "#00ff9a", "#ff7b7b"],
            )
            fig3.update_layout(
                title="Violation Statistics",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#e0faff",
                margin=dict(l=8, r=8, t=36, b=8),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No violations in active feed.")

    if "fusion_insight" in st.session_state:
        render_section_header("INTELLIGENCE FUSION")
        st.markdown(
            f"<div class='intel-feed-card'>{st.session_state.fusion_insight}</div>",
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)
