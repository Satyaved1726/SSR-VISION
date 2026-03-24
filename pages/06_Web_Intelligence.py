import streamlit as st
import streamlit.components.v1 as components
import time as _time
from ui.components import render_global_header, check_state_auth, render_section_header, render_alert, render_status_bar

st.set_page_config(page_title="SSR VISION | Web Intel", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_global_header()
check_state_auth()
render_status_bar(active_alerts=len(st.session_state.get("alerts", [])), processing="WDM")

# ── Live JS clock (updates every second without page reload) ──────────────────
components.html("""
<div style="display:flex; justify-content:space-between; align-items:center;
            background:rgba(0,243,255,0.05); border:1px solid #00f3ff;
            padding:10px 18px; border-radius:4px; font-family:monospace;">
    <div>
        <span style="color:#6b8f9e; font-size:0.75rem; letter-spacing:2px;">SYSTEM CLOCK</span><br>
        <span id="live-clock" style="color:#00f3ff; font-size:1.6rem; font-weight:bold; letter-spacing:3px;">--:--:--</span>
    </div>
    <div style="text-align:right;">
        <span style="color:#6b8f9e; font-size:0.75rem; letter-spacing:2px;">DATE</span><br>
        <span id="live-date" style="color:#ffb700; font-size:1.1rem; letter-spacing:2px;">---</span>
    </div>
</div>
<script>
(function() {
    function pad(n){ return String(n).padStart(2,'0'); }
    function tick(){
        var now = new Date();
        var t = pad(now.getHours())+':'+pad(now.getMinutes())+':'+pad(now.getSeconds());
        var months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
        var d = pad(now.getDate())+' '+months[now.getMonth()]+' '+now.getFullYear();
        document.getElementById('live-clock').innerHTML = t;
        document.getElementById('live-date').innerHTML = d;
    }
    tick();
    setInterval(tick, 1000);
})();
</script>
""", height=80)

st.markdown('<div class="panel-container">', unsafe_allow_html=True)
st.markdown("### 🌐 06_ WEB DATA MINING (WDM)")

web_data = st.session_state.web_data

# ── Row 1: Weather + Traffic ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    render_section_header("WEATHER SATELLITE INTERCEPT")
    weather_raw = web_data.get("weather", "")
    is_offline = not weather_raw or weather_raw in ["Unknown", "N/A", "Data Unavailable (Offline)", "Weather link standby"]

    if is_offline:
        st.markdown("""
        <div style="background:rgba(255,0,60,0.08); border-left:4px solid #ff003c;
                    padding:15px; font-family:monospace; color:#ff6680;">
            ⚠ WEATHER SATELLITE LINK UNAVAILABLE<br>
            <span style="color:#6b8f9e; font-size:0.85rem;">Check network connectivity or wttr.in access.</span>
        </div>""", unsafe_allow_html=True)
        render_alert("METEOROLOGICAL FEED OFFLINE — RISK SCORE MAY BE INACCURATE", "warning")
    else:
        # Parse wttr.in format-3 output: "City: <emoji> <temp> <wind>"
        parts = weather_raw.split(":", 1)
        city_label = parts[0].strip() if len(parts) == 2 else "LOCATION"
        weather_detail = parts[1].strip() if len(parts) == 2 else weather_raw
        st.markdown(f"""
        <div style="background:rgba(0,243,255,0.06); border-left:4px solid #00f3ff;
                    padding:16px 18px; font-family:monospace; border-radius:4px;">
            <div style="color:#6b8f9e; font-size:0.75rem; letter-spacing:2px; margin-bottom:4px;">LOCATION</div>
            <div style="color:#00f3ff; font-size:1.1rem; font-weight:bold;">{city_label}</div>
            <div style="color:#e0faff; font-size:1.4rem; margin-top:8px;">{weather_detail}</div>
        </div>""", unsafe_allow_html=True)

        if web_data.get("is_bad_weather", False):
            render_alert("ADVERSE METEOROLOGICAL CONDITIONS DETECTED — RISK SCORE ELEVATED", "warning")
        else:
            render_alert("METEOROLOGICAL CONDITIONS NOMINAL", "info")

with col2:
    render_section_header("TRAFFIC COMMS INTERCEPT")
    traffic_news = web_data.get("traffic_news", "No updates available.")
    st.markdown(f"""
    <div style="background:rgba(255,183,0,0.07); border-left:4px solid #ffb700;
                padding:16px 18px; font-family:monospace; border-radius:4px;">
        <div style="color:#6b8f9e; font-size:0.75rem; letter-spacing:2px; margin-bottom:6px;">LATEST TRAFFIC BULLETIN</div>
        <div style="color:#ffda6a; font-size:1rem; line-height:1.6;">{traffic_news}</div>
    </div>""", unsafe_allow_html=True)

# ── Row 2: Gov Alerts + Accident Reports ─────────────────────────────────────
col3, col4 = st.columns([1, 1])

with col3:
    render_section_header("GOVERNMENT ALERT CHANNEL")
    gov_alerts = web_data.get("gov_alerts", [])
    if gov_alerts:
        for item in gov_alerts:
            st.markdown(f"""
            <div style="background:rgba(255,0,60,0.07); border-left:3px solid #ff003c;
                        padding:10px 14px; margin-bottom:8px; font-family:monospace;
                        color:#ff9aaa; border-radius:3px;">
                🔴 {item}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(0,255,102,0.06); border-left:3px solid #00ff66;
                    padding:10px 14px; font-family:monospace; color:#7dffb0; border-radius:3px;">
            ✅ No active government alerts.
        </div>""", unsafe_allow_html=True)

with col4:
    render_section_header("ACCIDENT REPORTS")
    accident_reports = web_data.get("accident_reports", [])
    if accident_reports:
        for item in accident_reports:
            st.markdown(f"""
            <div style="background:rgba(255,183,0,0.07); border-left:3px solid #ffb700;
                        padding:10px 14px; margin-bottom:8px; font-family:monospace;
                        color:#ffe08a; border-radius:3px;">
                ⚠ {item}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(0,243,255,0.05); border-left:3px solid #00f3ff;
                    padding:10px 14px; font-family:monospace; color:#7feeff; border-radius:3px;">
            ✅ No accident bulletins detected.
        </div>""", unsafe_allow_html=True)

# ── Row 3: Text Intelligence Analysis ────────────────────────────────────────
render_section_header("TEXT INTELLIGENCE ANALYSIS")
ti1, ti2, ti3 = st.columns(3)

with ti1:
    keywords = web_data.get("keywords", [])
    st.markdown(f"""
    <div style="background:rgba(0,243,255,0.05); border:1px solid rgba(0,243,255,0.2);
                padding:14px; border-radius:4px; font-family:monospace; min-height:90px;">
        <div style="color:#6b8f9e; font-size:0.72rem; letter-spacing:2px; margin-bottom:8px;">DETECTED KEYWORDS</div>
        <div style="color:#00f3ff;">{'&nbsp; &nbsp;'.join([f'<span style="background:rgba(0,243,255,0.1);padding:2px 7px;border-radius:3px;">{k}</span>' for k in keywords]) if keywords else '<span style="color:#555;">None detected</span>'}</div>
    </div>""", unsafe_allow_html=True)

with ti2:
    locations = (web_data.get("entities") or {}).get("locations", [])
    accident_locs = web_data.get("accident_locations", [])
    all_locs = sorted(set(locations + accident_locs))
    st.markdown(f"""
    <div style="background:rgba(0,255,102,0.04); border:1px solid rgba(0,255,102,0.2);
                padding:14px; border-radius:4px; font-family:monospace; min-height:90px;">
        <div style="color:#6b8f9e; font-size:0.72rem; letter-spacing:2px; margin-bottom:8px;">GEOLOCATED ENTITIES</div>
        <div style="color:#7dffb0;">{'<br>'.join([f'📍 {loc}' for loc in all_locs]) if all_locs else '<span style="color:#555;">No locations extracted</span>'}</div>
    </div>""", unsafe_allow_html=True)

with ti3:
    conditions = (web_data.get("entities") or {}).get("conditions", [])
    st.markdown(f"""
    <div style="background:rgba(255,183,0,0.04); border:1px solid rgba(255,183,0,0.2);
                padding:14px; border-radius:4px; font-family:monospace; min-height:90px;">
        <div style="color:#6b8f9e; font-size:0.72rem; letter-spacing:2px; margin-bottom:8px;">ROAD CONDITIONS</div>
        <div style="color:#ffda6a;">{'<br>'.join([f'⚡ {c.title()}' for c in conditions]) if conditions else '<span style="color:#555;">No conditions flagged</span>'}</div>
    </div>""", unsafe_allow_html=True)

# ── Row 4: Advisories + Road Closures ────────────────────────────────────────
adv_col, clos_col = st.columns([1, 1])
with adv_col:
    render_section_header("TRAFFIC ADVISORIES")
    advisories = web_data.get("advisories", [])
    if advisories:
        for adv in advisories:
            st.markdown(f"""
            <div style="background:rgba(127,255,212,0.06); border-left:3px solid #7fffd4;
                        padding:10px 14px; margin-bottom:8px; font-family:monospace;
                        color:#b2fff0; border-radius:3px;">
                ℹ {adv}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-family:monospace; color:#555;">No active advisories.</p>', unsafe_allow_html=True)

with clos_col:
    render_section_header("ROAD CLOSURES & CONSTRUCTION")
    closures = web_data.get("road_closures", [])
    if closures:
        for c in closures:
            st.markdown(f"""
            <div style="background:rgba(255,0,60,0.06); border-left:3px solid #ff003c;
                        padding:10px 14px; margin-bottom:8px; font-family:monospace;
                        color:#ff9aaa; border-radius:3px;">
                🚧 {c}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-family:monospace; color:#555;">No road closures reported.</p>', unsafe_allow_html=True)

# ── Intelligence Fusion ───────────────────────────────────────────────────────
if "fusion_insight" in st.session_state:
    render_section_header("INTELLIGENCE FUSION")
    st.markdown(
        f"<div class='intel-feed-card'>{st.session_state.fusion_insight}</div>",
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)
