import streamlit as st
import pandas as pd
import plotly.express as px
from ui.components import render_global_header, check_state_auth, render_section_header, render_metric_card, render_status_bar
from core.analytics import AnalyticsEngine

st.set_page_config(page_title="SSR VISION | Analytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="ANALYTICS")

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 📈 07_ ANALYTICS CENTER")

vr = st.session_state.vision_results
traffic_metrics = AnalyticsEngine.compute_traffic_metrics(vr)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: render_metric_card("RISK RATING", st.session_state.risk_score)
with c2: render_metric_card("VEHICLES", vr["vehicle_count"])
with c3: render_metric_card("CONGESTION", vr["density_level"])
with c4: render_metric_card("VIOLATIONS", len(vr["violations"]))
with c5: render_metric_card("PEDESTRIANS", vr.get("pedestrian_count", 0))

c6, c7 = st.columns(2)
with c6: render_metric_card("LANE OCCUPANCY", f"{int(traffic_metrics['lane_occupancy']*100)}%")
with c7: render_metric_card("VEHICLE SPACING", f"{traffic_metrics['vehicle_spacing']:.1f}px")

render_section_header("VEHICLE TYPE DISTRIBUTION")

# Data preparation
labels_count = {}
for v in vr.get("bboxes", []):
    label = v.get("label", "Unknown")
    labels_count[label] = labels_count.get(label, 0) + 1

if labels_count:
    df = pd.DataFrame(list(labels_count.items()), columns=["Vehicle Type", "Count"])
    # Custom cyber colors
    fig = px.bar(df, x="Vehicle Type", y="Count", text="Count", 
                 color_discrete_sequence=["#00f3ff"])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="#e0faff",
        font_family="Rajdhani",
        xaxis_gridcolor="rgba(0, 243, 255, 0.1)",
        yaxis_gridcolor="rgba(0, 243, 255, 0.1)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW: Additional Intelligence Modules (Mocked for Demo based on constraints) ---
    st.markdown("<hr style='border-color: rgba(0, 243, 255, 0.2); margin: 30px 0;'>", unsafe_allow_html=True)
    c_sub1, c_sub2, c_sub3 = st.columns(3)
    
    with c_sub1:
        render_section_header("URBAN MOBILITY INDEX")
        # Mock Gauge
        fig_g = px.pie(values=[vr['vehicle_count'], 100-vr['vehicle_count']], names=['Mobility', 'Resistance'], hole=0.7, color_discrete_sequence=["#00ff66", "#1a1a1a"])
        fig_g.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0faff", height=250, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_g, use_container_width=True)
        
    with c_sub2:
        render_section_header("ENVIRONMENTAL IMPACT")
        st.markdown(f"""
        <div class="holographic-card" style="text-align:center; padding:30px 10px;">
            <h1 style="color:#ffb700; margin:0; font-family:'Orbitron', sans-serif;">{min(99, vr['vehicle_count'] * 1.5):.1f}</h1>
            <p style="color:#6b8f9e; font-family:monospace; font-size:0.8em;">EST. AQI IMPACT / NOX EMISSIONS</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c_sub3:
        render_section_header("TRAFFIC TREND PREDICTOR")
        import numpy as np
        # Mock trend line
        trend_data = np.cumsum(np.random.randn(20)) + vr['vehicle_count']
        fig_l = px.line(y=trend_data, color_discrete_sequence=["#ff003c"])
        fig_l.update_layout(xaxis_visible=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0faff", height=250, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_l, use_container_width=True)

else:
    st.info("No vehicles detected to map.")

render_section_header("FEATURE INTELLIGENCE METRICS")
feature_pack = st.session_state.get("feature_pack", {})
f1, f2, f3 = st.columns(3)
with f1:
    render_metric_card("ORB KEYPOINTS", feature_pack.get("orb_count", 0))
with f2:
    render_metric_card("CONTOUR COUNT", feature_pack.get("contour_count", 0))
with f3:
    shape_stats = feature_pack.get("shape_stats", {})
    render_metric_card("SHAPE TYPES", len([k for k, v in shape_stats.items() if v > 0]))

render_section_header("DATASET ANALYSIS MODULE")
uploaded_dataset = st.file_uploader("Upload traffic violations or vehicle records dataset", type=["csv"])
if uploaded_dataset is not None:
    df_in = pd.read_csv(uploaded_dataset)
    st.dataframe(df_in.head(20), use_container_width=True)

    if "violation" in [c.lower() for c in df_in.columns]:
        violation_col = next(c for c in df_in.columns if c.lower() == "violation")
        viol_df = df_in[violation_col].value_counts().reset_index()
        viol_df.columns = ["Violation", "Count"]
        vfig = px.bar(viol_df, x="Violation", y="Count", color_discrete_sequence=["#ff003c"])
        vfig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e0faff")
        st.plotly_chart(vfig, use_container_width=True)

    numeric_cols = df_in.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 1:
        trend_col = numeric_cols[0]
        tfig = px.line(df_in[trend_col], color_discrete_sequence=["#00f3ff"], title=f"Trend: {trend_col}")
        tfig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#e0faff")
        st.plotly_chart(tfig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
