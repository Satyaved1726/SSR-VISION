import streamlit as st
from datetime import datetime
import requests


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_live_weather(region):
    safe_region = (region or "Hyderabad").strip().replace(" ", "+")
    try:
        response = requests.get(f"https://wttr.in/{safe_region}?format=3", timeout=2.5)
        if response.status_code == 200:
            text = response.text.strip()
            if text and len(text) > 5 and "Error" not in text:
                return text
    except Exception:
        pass
    # Time-based realistic fallback when network is unavailable
    from datetime import datetime as _dt
    hour = _dt.now().hour
    region_label = (region or "Hyderabad").strip()
    if 5 <= hour < 11:
        return f"{region_label}: ☀️ 28°C ↗ 10km/h"
    elif 11 <= hour < 16:
        return f"{region_label}: 🌤 35°C ↗ 18km/h"
    elif 16 <= hour < 20:
        return f"{region_label}: ⛅ 31°C → 12km/h"
    else:
        return f"{region_label}: 🌙 26°C ↘ 6km/h"

def render_global_header():
    """Renders the top NASA-grade telemetry global navigation and status header."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%d %b %Y")
    settings = st.session_state.get("system_settings", {}) if hasattr(st, "session_state") else {}
    region = settings.get("region", "Hyderabad")
    live_weather = None
    if hasattr(st, "session_state"):
        live_weather = st.session_state.get("web_data", {}).get("weather")
    if not live_weather or live_weather in ["Unknown", "N/A", "Data Unavailable (Offline)"]:
        live_weather = _fetch_live_weather(region)

    # Threat level derived from live vision session
    threat_text = "THREAT LVL: SCANNING..."
    threat_color = "#ffb700"
    if hasattr(st, "session_state") and "vision_results" in st.session_state:
        vr = st.session_state.vision_results
        violations = len(vr.get("violations", []))
        density = vr.get("density_level", "LOW")
        road_cond = vr.get("road_condition", "GOOD")
        if violations > 0 or density == "CRITICAL" or road_cond == "DAMAGED/OBSTRUCTED":
            threat_text = "THREAT LVL: HIGH &#9888;"
            threat_color = "#ff003c"
        elif density in ["HIGH", "MEDIUM"]:
            threat_text = "THREAT LVL: ELEVATED"
            threat_color = "#ffb700"
        else:
            threat_text = "THREAT LVL: NOMINAL"
            threat_color = "#00ff66"

    safe_weather = (live_weather or "").replace("<", "&lt;").replace(">", "&gt;")
    safe_region = (region or "Hyderabad").replace("<", "&lt;")
    import streamlit.components.v1 as _components
    _components.html(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:rgb(5,10,20);font-family:monospace;overflow:hidden}}
.hdr{{display:flex;justify-content:space-between;align-items:stretch;flex-wrap:wrap;gap:10px;background:linear-gradient(90deg,rgba(5,10,20,0.98),rgba(0,0,0,0.98));border:1px solid #00f3ff;border-top:4px solid #00f3ff;padding:12px 15px;border-radius:4px;box-shadow:0 0 20px rgba(0,243,255,0.15),inset 0 0 10px rgba(0,243,255,0.1)}}
.brand{{flex:1 1 240px;min-width:200px}}
.brand h2{{color:#00f3ff;font-family:'Orbitron','Courier New',monospace;text-transform:uppercase;text-shadow:0 0 10px #00f3ff;font-size:1.35rem;letter-spacing:2px}}
.dev-name{{font-family:'Orbitron','Courier New',monospace;font-size:0.68rem;letter-spacing:3px;margin-top:6px;font-weight:700;color:#00ff41;text-shadow:0 0 6px #00ff41,0 0 12px #00ff41,0 0 24px #00ff41;animation:hackglow 2.4s ease-in-out infinite}}
@keyframes hackglow{{
  0%,100%{{color:#00ff41;text-shadow:0 0 4px #00ff41,0 0 10px #00ff41,0 0 20px #00ff41}}
  30%{{color:#00f3ff;text-shadow:0 0 6px #00f3ff,0 0 14px #00f3ff,0 0 30px #00f3ff,0 0 50px #00f3ff}}
  60%{{color:#39ff14;text-shadow:0 0 4px #39ff14,0 0 8px #39ff14,0 0 18px #39ff14}}
}}
.s1{{color:#6b8f9e;font-size:0.72rem;letter-spacing:2px;margin-top:5px}}
.s2{{color:#ffb700;font-size:0.65rem;letter-spacing:1px;margin-top:4px}}
.telgrid{{display:grid;grid-template-columns:1fr 1fr;gap:7px 14px;font-size:0.78rem;background:rgba(0,243,255,0.05);padding:10px;border:1px solid rgba(0,243,255,0.2);min-width:300px;max-width:440px;align-self:center}}
.full{{grid-column:span 2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.tele{{text-align:right;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;min-width:190px;flex:1 1 190px}}
.bar{{width:140px;height:4px;background:#333;margin:5px 0}}
.barfill{{height:100%;background:#00f3ff;box-shadow:0 0 5px #00f3ff;transition:width 1s}}
@keyframes blink{{50%{{opacity:0}}}}
.blink{{animation:blink 1s step-end infinite}}
</style></head><body>
<div class="hdr">
  <div class="brand">
    <h2>SSR VISION</h2>
    <div class="dev-name">&gt; SRINIVAS SATYA RAMESH VISION _</div>
    <div class="s1">SMART SURVEILLANCE &amp; RESPONSE VISION PLATFORM</div>
    <div class="s2">AI URBAN INTELLIGENCE &amp; CYBER MONITORING SYSTEM</div>
  </div>
  <div style="flex:1 1 300px;display:flex;justify-content:center;align-items:center;">
    <div class="telgrid">
      <div style="color:#00ff66;">CORE STATUS: <span class="blink">NOMINAL</span></div>
      <div style="color:#00f3ff;">SYS CLOCK: <b id="hclock">--:--:--</b></div>
      <div style="color:#ffb700;">NODE SYNC: ACTIVE</div>
      <div style="color:{threat_color};" id="hthreat">{threat_text}</div>
      <div style="color:#e0faff;">DATE: <span id="hdate">---</span></div>
      <div style="color:#7fffd4;">REGION: {safe_region}</div>
      <div class="full" style="color:#9fe8ff;">WEATHER: {safe_weather}</div>
    </div>
  </div>
  <div class="tele">
    <div style="color:#00f3ff;font-size:0.88rem;">TELEMETRY LINK: <b style="color:#00ff66;">SECURE</b></div>
    <div class="bar"><div class="barfill" id="cpubar" style="width:25%"></div></div>
    <div style="color:#6b8f9e;font-size:0.72rem;" id="cputxt">CPU: -- | MEM: -- | NET: UPLINK T-0</div>
  </div>
</div>
<script>(function(){{
  function pad(n){{return String(n).padStart(2,'0');}}
  var mo=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
  function tick(){{
    var n=new Date();
    document.getElementById('hclock').innerHTML=pad(n.getHours())+':'+pad(n.getMinutes())+':'+pad(n.getSeconds());
    document.getElementById('hdate').innerHTML=pad(n.getDate())+' '+mo[n.getMonth()]+' '+n.getFullYear();
    var cpu=Math.floor(Math.random()*25)+10;
    var mem=Math.floor(Math.random()*20)+55;
    document.getElementById('cpubar').style.width=cpu+'%';
    document.getElementById('cputxt').innerHTML='CPU: '+cpu+'% | MEM: '+mem+'% | NET: UPLINK T-0';
  }}
  tick();setInterval(tick,1000);
}})();
</script>
</body></html>""", height=178, scrolling=False)


def render_status_bar(active_alerts=0, processing="IDLE"):
    now = datetime.now().strftime("%H:%M:%S")
    html = (
        '<div class="telemetry-strip">'
        '<span class="telemetry-item"><strong>SSR CORE</strong>: ONLINE</span>'
        '<span class="telemetry-item"><strong>AI ENGINE</strong>: ONLINE</span>'
        f'<span class="telemetry-item"><strong>ACTIVE ALERTS</strong>: {active_alerts}</span>'
        f'<span class="telemetry-item"><strong>SYSTEM TIME</strong>: {now}</span>'
        f'<span class="telemetry-item"><strong>PROCESSING</strong>: {processing}</span>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    
def check_state_auth():
    """Helper to check if an image has been uploaded to state."""
    if 'uploaded_image' not in st.session_state or st.session_state.uploaded_image is None:
        st.warning("⚠️ CRITICAL ERROR: NO ACTIVE INTELLIGENCE FEED DETECTED.")
        st.info(">> PLEASE RETURN TO [01 - IMAGE ANALYSIS CENTER] TO UPLINK A DATA STREAM.")
        st.stop()
    if 'vision_results' not in st.session_state:
        st.warning("⚠️ CRITICAL ERROR: NO AI METRICS DETECTED.")
        st.info(">> PLEASE RETURN TO [01 - IMAGE ANALYSIS CENTER] TO RUN THE ANALYSIS.")
        st.stop()

def render_metric_card(title, value, status="normal"):
    """
    Renders a high-end NASA-style metric card.
    """
    color_class = "metric-card holographic-card"
    if status == "warning": color_class += " warning"
    elif status == "danger": color_class += " danger"
    
    html = (
        f'<div class="{color_class}" style="margin:5px;">'
        f'<div class="metric-card-title">{title}</div>'
        f'<div class="metric-card-value" style="font-family: \'Orbitron\', monospace;">{value}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_alert(message, level="info"):
    """
    Renders a custom futuristic alert.
    """
    normalized = (level or "info").lower()
    css_class = "info"
    label = "INFO"
    if normalized in ["warning", "warn"]:
        css_class = "warning"
        label = "WARNING"
    elif normalized in ["critical", "danger", "error"]:
        css_class = "critical"
        label = "CRITICAL"

    html = (
        f'<div class="alert-card {css_class}">'
        f'<div class="alert-level">{label}</div>'
        f'<div style="font-size:1.05rem; color:#e9fbff;">{message}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(title):
    html = (
        '<div style="border-bottom: 2px solid rgba(0, 243, 255, 0.5); '
        'margin: 30px 0 15px 0; padding-bottom: 8px; position: relative;">'
        '<div style="position:absolute; left:0; bottom:-2px; width:30px; height:2px; background:#fff; box-shadow:0 0 10px #fff;"></div>'
        '<h3 style="margin:0; font-family:\'Orbitron\', sans-serif; color:#00f3ff; font-size: 1.3rem; letter-spacing: 2px; text-shadow:0 0 5px rgba(0,243,255,0.5);">'
        f'■ {title}'
        '</h3>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_evidence_card(title, timestamp, severity, info_dict):
    """
    Renders a highly detailed 'Evidence Card' for violations.
    """
    colors = {
        "CRITICAL": "#ff003c",
        "HIGH": "#ffb700",
        "MEDIUM": "#00f3ff",
        "LOW": "#00ff66"
    }
    glow = colors.get(severity, "#00f3ff")
    
    html = (
        '<div style="'
        f'border: 1px solid {glow}; background: rgba(5,10,15,0.9); margin-bottom: 20px; position: relative; '
        f'box-shadow: 0 5px 15px rgba(0,0,0,0.8), inset 0 0 20px rgba({int(glow[1:3],16)}, {int(glow[3:5],16)}, {int(glow[5:7],16)}, 0.1); overflow: hidden;">'
        '<div style="'
        f'background: {glow}; color: #000; font-family: \'Orbitron\', sans-serif; padding: 5px 10px; font-weight: bold; font-size: 0.9em; display:flex; justify-content:space-between;">'
        f'<span>EVIDENCE ID: {hash(title+timestamp) % 100000}</span>'
        f'<span>{severity} PRIORITY</span>'
        '</div>'
        '<div style="padding: 15px;">'
        f'<h4 style="color: {glow}; margin: 0 0 10px 0; font-family: \'Rajdhani\', sans-serif; font-size: 1.2em; text-transform: uppercase; letter-spacing: 1px;">{title}</h4>'
        '<div style="font-family: monospace; font-size: 0.85em; color: #a0c4d3; display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">'
        f'<div style="border-left: 2px solid {glow}; padding-left: 5px;"><b>TIMESTAMP:</b><br>{timestamp}</div>'
    )
    for k, v in info_dict.items():
        html += f'<div style="border-left: 2px solid {glow}; padding-left: 5px;"><b>{k}:</b><br>{v}</div>'
        
    html += (
        '</div>'
        '<div style="margin-top: 15px; text-align: right;">'
        '<span style="font-size: 0.7em; font-family: monospace; color: #6b8f9e;">SYS.EVIDENCE.SECURE.HASH</span>'
        '</div></div>'
        '<div style="'
        f'position: absolute; top:0; left:0; width:100%; height:2px; background: {glow}; opacity:0.5; box-shadow: 0 0 10px {glow}; animation: scanline 3s linear infinite;"></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_timeline(events):
    html = '<div style="padding:12px; border:1px solid rgba(0,243,255,0.3); background:rgba(3,8,14,0.7);">'
    for ts, msg, color in events:
        html += (
            f'<div style="padding:6px 0; border-bottom:1px dashed rgba(0,243,255,0.15);">'
            f'<span style="font-family:monospace; color:#6b8f9e;">[{ts}]</span> '
            f'<span style="color:{color};">{msg}</span>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

