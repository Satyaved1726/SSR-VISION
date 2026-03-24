import sys
import os
import importlib.util
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

# Mock Streamlit session state
st.session_state = SessionState()
st.set_page_config = lambda *args, **kwargs: None

dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
st.session_state.img_np = dummy_img
st.session_state.uploaded_image = Image.fromarray(dummy_img)
st.session_state.processed_img_np = dummy_img
st.session_state.vision_results = {
    "vehicle_count": 0,
    "pedestrian_count": 0,
    "density_level": "LOW",
    "violations": [],
    "road_condition": "GOOD",
    "bboxes": [],
    "vehicles": [],
    "type_distribution": {},
    "lane_occupancy": 0.0,
    "vehicle_spacing": 0.0,
}
st.session_state.alerts = []
st.session_state.web_data = {"weather": "N/A", "is_bad_weather": False, "advisories": []}
st.session_state.risk_score = 0
st.session_state.caption = "Test"
st.session_state.plates = []

def load_page(path, label):
    try:
        module_name = label.lower().replace(" ", "_")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"{label} OK")
    except Exception as e:
        print(f"{label} Error: {e}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_page(os.path.join(BASE_DIR, "pages", "01_Image_Analysis_Center.py"), "Page 1")
load_page(os.path.join(BASE_DIR, "pages", "02_Computer_Vision_Lab.py"), "Page 2")
load_page(os.path.join(BASE_DIR, "pages", "03_Edge_Detection_Studio.py"), "Page 3")
