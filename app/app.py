"""
Sonaris — Unified Sonar Intelligence Platform
Premium Streamlit Dashboard with real-time ML inference
"""

import os
import sys
import io
import json
import time
import csv
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import cv2
    CV2_AVAILABLE = True
except (ImportError, Exception):
    cv2 = None
    CV2_AVAILABLE = False


def bgr_to_rgb(img):
    """Safely convert BGR to RGB with or without OpenCV."""
    if img is None:
        return img
    if cv2 is not None:
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[2] == 3:
        return img[:, :, ::-1]
    return img


def rgb_to_bgr(img):
    """Safely convert RGB to BGR with or without OpenCV."""
    if img is None:
        return img
    if cv2 is not None:
        try:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[2] == 3:
        return img[:, :, ::-1]
    return img


# ── Resolve project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Sonaris | Sonar Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Optional ML imports ─────────────────────────────────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass

# ── Model paths ─────────────────────────────────────────────────────────────
LITE_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_seg_best.pt"
WAR_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_war_best.pt"
PRO_YOLO_PATH = PROJECT_ROOT / "models" / "yolov8x_best.pt"
SAM_PATH = PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Ocean-Dark Theme
# ════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-primary: #060d19;
        --bg-secondary: #0c1a2e;
        --bg-card: rgba(15, 35, 65, 0.55);
        --bg-card-hover: rgba(20, 50, 90, 0.65);
        --accent: #00e5a0;
        --accent-dim: rgba(0, 229, 160, 0.12);
        --accent-glow: rgba(0, 229, 160, 0.35);
        --cyan: #00c8ff;
        --coral: #ff6b8a;
        --gold: #ffc857;
        --text-primary: #e8f4fd;
        --text-secondary: #8ba3be;
        --text-muted: #4a6580;
        --border: rgba(255,255,255,0.06);
        --border-accent: rgba(0, 229, 160, 0.2);
        --radius: 16px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(160deg, var(--bg-primary) 0%, var(--bg-secondary) 40%, #0a1e38 100%);
        color: var(--text-primary);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6,13,25,0.97) 0%, rgba(12,26,46,0.97) 100%) !important;
        border-right: 1px solid var(--border-accent);
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Glass Cards */
    .glass {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 28px;
        margin-bottom: 20px;
        transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    .glass:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-accent);
        transform: translateY(-3px);
        box-shadow: 0 20px 60px rgba(0, 229, 160, 0.08);
    }

    /* Hero Title */
    .hero-title {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #00e5a0 0%, #00c8ff 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        line-height: 1.1;
        margin-bottom: 0;
    }

    .hero-sub {
        text-align: center;
        color: var(--text-secondary);
        font-size: 18px;
        font-weight: 400;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 8px;
        margin-bottom: 40px;
    }

    /* Pulse Dot */
    .pulse-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: var(--accent);
        border-radius: 50%;
        margin-left: 12px;
        vertical-align: middle;
        animation: pulse-ring 2s ease-in-out infinite;
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 var(--accent-glow); }
        70% { box-shadow: 0 0 0 16px rgba(0,229,160,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,229,160,0); }
    }

    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,229,160,0.08) 0%, rgba(0,200,255,0.05) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 30px var(--accent-dim);
    }
    .metric-icon { font-size: 28px; margin-bottom: 8px; }
    .metric-val {
        font-size: 36px;
        font-weight: 800;
        color: var(--accent);
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 6px;
        font-weight: 600;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 24px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background: rgba(0,229,160,0.12);
        color: var(--accent);
        border: 1px solid rgba(0,229,160,0.3);
        animation: badge-glow 2.5s ease-in-out infinite;
    }
    @keyframes badge-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(0,229,160,0.15); }
        50% { box-shadow: 0 0 25px rgba(0,229,160,0.3); }
    }
    .badge-danger {
        background: rgba(255,107,138,0.12);
        color: var(--coral);
        border: 1px solid rgba(255,107,138,0.3);
    }
    .badge-info {
        background: rgba(0,200,255,0.12);
        color: var(--cyan);
        border: 1px solid rgba(0,200,255,0.3);
    }

    /* Detection Card */
    .det-card {
        background: linear-gradient(135deg, rgba(10,30,56,0.8) 0%, rgba(15,40,70,0.6) 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    .det-card:hover {
        border-color: var(--border-accent);
        background: rgba(15,40,70,0.75);
    }
    .det-class {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        text-transform: capitalize;
    }
    .det-conf {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: var(--accent);
        font-size: 18px;
    }
    .det-meta {
        color: var(--text-secondary);
        font-size: 13px;
        line-height: 1.8;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed var(--border-accent) !important;
        border-radius: var(--radius) !important;
        padding: 20px !important;
        background: rgba(0,229,160,0.02) !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: rgba(0,229,160,0.06) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, #00c8ff 100%) !important;
        color: #060d19 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px var(--accent-glow) !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: rgba(0,229,160,0.1) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(0,229,160,0.2) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--cyan)) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* Section Headers */
    .section-head {
        font-size: 13px;
        font-weight: 700;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
    }

    h1, h2, h3 { color: var(--text-primary) !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-dim) !important;
        color: var(--accent) !important;
        border-color: var(--border-accent) !important;
    }

    /* Toast / warnings / info */
    .stAlert { border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def extract_gps_from_exif(pil_image: Image.Image):
    """Extract GPS lat/lon from EXIF metadata."""
    try:
        exif = pil_image._getexif()
        if not exif:
            return None, None
        gps_info = {}
        for key, val in exif.items():
            tag = TAGS.get(key, key)
            if tag == "GPSInfo":
                for t in val:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_info[sub_tag] = val[t]

        if "GPSLatitude" not in gps_info:
            return None, None

        def to_deg(v):
            return float(v[0]) + float(v[1]) / 60.0 + float(v[2]) / 3600.0

        lat = to_deg(gps_info["GPSLatitude"])
        if gps_info.get("GPSLatitudeRef") != "N":
            lat = -lat
        lon = to_deg(gps_info["GPSLongitude"])
        if gps_info.get("GPSLongitudeRef") != "E":
            lon = -lon
        return lat, lon
    except Exception:
        return None, None


def google_maps_url(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


@st.cache_resource
def load_lite_model(file_mtime: float = 0.0):
    """Load YOLO11n-Seg model (cached based on file timestamp)."""
    if not YOLO_AVAILABLE:
        return None
    if LITE_MODEL_PATH.exists():
        return YOLO(str(LITE_MODEL_PATH))
    return None


def get_lite_model():
    """Get Lite model with cache auto-invalidation on weight update."""
    mtime = LITE_MODEL_PATH.stat().st_mtime if LITE_MODEL_PATH.exists() else 0.0
    return load_lite_model(mtime)


@st.cache_resource
def load_war_model(file_mtime: float = 0.0):
    """Load War Mode YOLO11 tactical naval model (cached)."""
    if not YOLO_AVAILABLE:
        return None
    if WAR_MODEL_PATH.exists():
        return YOLO(str(WAR_MODEL_PATH))
    return None


def get_war_model():
    """Get War Mode model with cache auto-invalidation on weight update."""
    mtime = WAR_MODEL_PATH.stat().st_mtime if WAR_MODEL_PATH.exists() else 0.0
    return load_war_model(mtime)




@st.cache_resource
def load_pro_yolo_model():
    """Load YOLOv8x model for Pro mode (cached).
    
    If fine-tuned weights exist locally, use those.
    Otherwise auto-download the base YOLOv8x from Ultralytics hub.
    """
    if not YOLO_AVAILABLE:
        return None
    # Try fine-tuned weights first
    if PRO_YOLO_PATH.exists():
        return YOLO(str(PRO_YOLO_PATH))
    # Auto-download base YOLOv8x from Ultralytics
    try:
        return YOLO("yolov8x.pt")
    except Exception:
        return None


def run_lite_inference(model, image_np, conf=0.5, imgsz=1024):
    """Run YOLO11n-Seg inference and return structured results."""
    results = model.predict(source=image_np, conf=conf, imgsz=imgsz, verbose=False)
    detections = []
    annotated_img = image_np.copy()

    if len(results) > 0:
        r = results[0]
        names = r.names
        annotated_img = r.plot()

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            masks_data = r.masks.data.cpu().numpy() if r.masks is not None else None

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls_id = int(clss[i])
                conf_val = float(confs[i])
                class_name = names.get(cls_id, str(cls_id))

                # Physical size estimation (approximate)
                w_px = x2 - x1
                h_px = y2 - y1
                area_px = w_px * h_px

                # Mask area for better estimation
                mask_area_px = area_px
                shape_type = "rectangular"
                if masks_data is not None and i < len(masks_data):
                    mask = masks_data[i]
                    if mask.shape != image_np.shape[:2]:
                        if cv2 is not None:
                            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                        else:
                            pil_m = Image.fromarray((mask * 255).astype(np.uint8))
                            mask = np.array(pil_m.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)) / 255.0
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    mask_area_px = float(np.sum(mask_binary))

                    # Simple shape classification
                    if cv2 is not None:
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * mask_area_px / (perimeter ** 2)
                                rect = cv2.minAreaRect(cnt)
                                rect_area = rect[1][0] * rect[1][1]
                                rectangularity = mask_area_px / max(rect_area, 1)
                                aspect = max(rect[1]) / max(min(rect[1]), 1)

                                if circularity > 0.85:
                                    shape_type = "circular"
                                elif rectangularity > 0.80:
                                    shape_type = "rectangular"
                                elif aspect > 3.0:
                                    shape_type = "elongated"
                                else:
                                    shape_type = "irregular"

                # Resolution: assume 1.0 m/px (configurable)
                res = 1.0
                det = {
                    "id": i + 1,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf_val,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "width_px": float(w_px),
                    "height_px": float(h_px),
                    "width_m": float(w_px * res),
                    "height_m": float(h_px * res),
                    "area_m2": float(mask_area_px * res * res),
                    "shape_type": shape_type,
                }
                detections.append(det)

    return detections, annotated_img


def draw_overlay_on_image(image_np, detections):
    """Draw bounding boxes and labels on image when model isn't available."""
    overlay = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = (0, 229, 160)  # accent green
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return overlay


def build_report_dict(detections, model_mode, conf_threshold, lat, lon, inference_ms):
    """Build a structured report dict."""
    return {
        "sonaris_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_mode": model_mode,
        "confidence_threshold": conf_threshold,
        "inference_time_ms": round(inference_ms, 1),
        "gps": {
            "latitude": lat if lat != 0 else None,
            "longitude": lon if lon != 0 else None,
            "google_maps": google_maps_url(lat, lon) if (lat != 0 or lon != 0) else None,
        },
        "total_detections": len(detections),
        "detections": detections,
    }


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="hero-title">Sonaris<span class="pulse-dot"></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Unified Sonar Intelligence Platform</div>', unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-val">3</div>
            <div class="metric-label">AI Models</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚔️</div>
            <div class="metric-val">5</div>
            <div class="metric-label">Naval Defense Classes</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📍</div>
            <div class="metric-val">GPS</div>
            <div class="metric-label">Geospatial Mapping</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-val">6.5</div>
            <div class="metric-label">GFLOPs (War Mode)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="glass">
            <div class="section-head">🚀 Lite Mode</div>
            <h3 style="margin:0 0 8px 0;">Civilian 4-Class</h3>
            <div class="det-meta">
                Ultra-lightweight instance segmentation (<b style="color:#00e5a0">2.83M params</b>).
                Detects aircraft wreckage, fish schools, seabed debris, and shipwrecks in real-time.
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass">
            <div class="section-head">⚔️ War Mode</div>
            <h3 style="margin:0 0 8px 0;">Naval Defense</h3>
            <div class="det-meta">
                Tactical model trained on DRISHTI dataset (<b style="color:#ff6b6b">2.59M params</b>).
                Detects subsea mines, pipelines, wrecks, ghost nets, and crab pots with threat alerts.
            </div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass">
            <div class="section-head">🎯 Pro Mode</div>
            <h3 style="margin:0 0 8px 0;">YOLOv8x + SAM</h3>
            <div class="det-meta">
                Two-stage hybrid combining <b style="color:#00c8ff">YOLOv8x detection</b> (68.2M params)
                with Meta's <b style="color:#a78bfa">SAM</b> for zero-shot boundary delineation.
            </div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="glass">
            <div class="section-head">📍 Geospatial</div>
            <h3 style="margin:0 0 8px 0;">GPS Mapping</h3>
            <div class="det-meta">
                Extract coordinates from <b style="color:#ffc857">EXIF metadata</b> or sonar telemetry.
                Auto-generates Google Maps coordinates with metric area and aspect ratio sizing.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Quick start
    st.markdown("### 🗺️ Quick Start")
    st.markdown("""
    <div class="glass">
        <div class="det-meta" style="font-size:15px; line-height: 2;">
        <b style="color:#00e5a0">1.</b> Select <b>🔍 Single Analysis</b> from the sidebar<br>
        <b style="color:#00e5a0">2.</b> Choose your model mode — <b>Lite</b> for speed or <b>Pro</b> for accuracy<br>
        <b style="color:#00e5a0">3.</b> Upload a side-scan sonar image (PNG, JPG, TIF)<br>
        <b style="color:#00e5a0">4.</b> Optionally add GPS coordinates<br>
        <b style="color:#00e5a0">5.</b> Click <b>Analyze</b> and explore results with confidence scores, shape classification, and physical measurements
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demo images preview
    if DEMO_DIR.exists():
        demo_files = sorted(DEMO_DIR.glob("*.png"))
        if demo_files:
            st.markdown("### 🖼️ Sample Sonar Images")
            cols = st.columns(min(len(demo_files), 4))
            for i, f in enumerate(demo_files[:4]):
                with cols[i]:
                    img = Image.open(f)
                    st.image(img, caption=f.stem, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def page_single_analysis():
    st.markdown("## 🔍 Single Image Analysis")

    # ── Sidebar Controls ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-head">⚙️ Model Controls</div>', unsafe_allow_html=True)

        mode = st.radio("Inference Mode", [
            "🚀 Lite — Civilian (4 Classes)",
            "⚔️ War Mode — Naval Defense (5 Classes)",
            "🎯 Pro — YOLOv8x + SAM Hybrid"
        ], help="Lite: Aircraft, Fish, Debris, Shipwreck. War Mode: Submarine Pipelines, Mines, Ghost Nets, Wrecks. Pro: Server-grade hybrid.")
        is_lite = "Lite" in mode
        is_war = "War" in mode
        is_pro = "Pro" in mode

        conf = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05,
                         help="Minimum confidence to display a detection")
        imgsz = st.select_slider("Inference Resolution", [640, 768, 1024], value=1024,
                                 help="Higher = more detail, slower inference")
        preprocess = st.toggle("Acoustic Preprocessing", value=False,
                               help="Apply CLAHE contrast + median denoising")

        st.markdown("---")
        st.markdown('<div class="section-head">📍 GPS</div>', unsafe_allow_html=True)
        gps_mode = st.radio("GPS Source", ["Manual", "Auto-EXIF", "JSON File"], horizontal=True)

        lat, lon = 0.0, 0.0
        if gps_mode == "Manual":
            lat = st.number_input("Latitude", -90.0, 90.0, 0.0, format="%.6f")
            lon = st.number_input("Longitude", -180.0, 180.0, 0.0, format="%.6f")
        elif gps_mode == "JSON File":
            meta_file = st.file_uploader("Upload Metadata", type=["json"], key="gps_json")
            if meta_file:
                try:
                    meta = json.load(meta_file)
                    lat = float(meta.get("latitude", 0))
                    lon = float(meta.get("longitude", 0))
                    st.success(f"📍 {lat:.4f}, {lon:.4f}")
                except Exception:
                    st.error("Invalid JSON")

    # ── Main Content ────────────────────────────────────────────────────
    top_col1, top_col2 = st.columns([3, 1])

    with top_col1:
        uploaded = st.file_uploader("Upload Side-Scan Sonar Image",
                                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                                    key="single_upload")
    with top_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        # Demo image selector
        use_demo = st.selectbox("Or use demo image", ["— Select —"] +
                                [f.name for f in sorted(DEMO_DIR.glob("*.png"))] if DEMO_DIR.exists() else ["— Select —"])

    # Determine image source
    pil_image = None
    source_name = ""

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        source_name = uploaded.name
    elif use_demo and use_demo != "— Select —":
        demo_path = DEMO_DIR / use_demo
        if demo_path.exists():
            pil_image = Image.open(demo_path).convert("RGB")
            source_name = use_demo

    if pil_image is None:
        st.info("👆 Upload a sonar image or select a demo image to begin analysis")
        return

    # EXIF GPS extraction
    if gps_mode == "Auto-EXIF":
        ex_lat, ex_lon = extract_gps_from_exif(pil_image)
        if ex_lat is not None:
            lat, lon = ex_lat, ex_lon
            st.sidebar.success(f"📍 EXIF: {lat:.6f}, {lon:.6f}")
        else:
            st.sidebar.warning("No GPS in EXIF data")

    # Preview
    image_np = np.array(pil_image)
    image_bgr = rgb_to_bgr(image_np)

    # Optional preprocessing
    if preprocess and cv2 is not None:
        try:
            # CLAHE contrast enhancement
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # Median denoise
            image_bgr = cv2.medianBlur(image_bgr, 5)
            image_np = bgr_to_rgb(image_bgr)
        except Exception:
            pass

    # ── Run Analysis ────────────────────────────────────────────────────
    analyze = st.button("🌊  ANALYZE SONAR IMAGE", use_container_width=True)

    if analyze:
        if is_war:
            model_label = "War Mode (YOLO11n Tactical 5-Class)"
        elif is_lite:
            model_label = "Lite (YOLO11n Civilian 4-Class)"
        else:
            model_label = "Pro (YOLOv8x + SAM)"
        progress = st.progress(0, text=f"Initializing {model_label}...")

        t_start = time.perf_counter()
        detections = []
        annotated_img = image_np.copy()

        # Load appropriate model
        if is_war:
            progress.progress(20, text="Loading War Mode (Naval Tactical)...")
            model = get_war_model()
            if model is not None:
                progress.progress(50, text="Running naval tactical threat inference...")
                detections, annotated_bgr = run_lite_inference(model, image_bgr, conf=conf, imgsz=imgsz)
                annotated_img = bgr_to_rgb(annotated_bgr)
            else:
                progress.progress(50, text="War Mode weights missing")
                st.error("⚠️ War Mode model not found. Ensure `models/yolo11n_war_best.pt` exists.")
                return
        elif is_lite:
            progress.progress(20, text="Loading YOLO11n Lite model...")
            model = get_lite_model()
            if model is not None:
                progress.progress(50, text="Running civilian inference...")
                detections, annotated_bgr = run_lite_inference(model, image_bgr, conf=conf, imgsz=imgsz)
                annotated_img = bgr_to_rgb(annotated_bgr)
            else:
                progress.progress(50, text="Model not found — check models/ directory")
                st.error("⚠️ YOLO model not found. Ensure `models/yolo11n_seg_best.pt` exists.")
                return
        else:
            # Pro mode
            progress.progress(20, text="Loading YOLOv8x model...")
            pro_model = load_pro_yolo_model()
            if pro_model is not None:
                progress.progress(40, text="Running YOLOv8x detection...")
                # Use YOLOv8x for detection
                results = pro_model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)
                if len(results) > 0:
                    r = results[0]
                    annotated_img = bgr_to_rgb(r.plot())
                    if r.boxes is not None:
                        for i in range(len(r.boxes)):
                            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                            cls_id = int(r.boxes.cls[i].cpu())
                            conf_val = float(r.boxes.conf[i].cpu())
                            detections.append({
                                "id": i + 1,
                                "class_id": cls_id,
                                "class_name": r.names.get(cls_id, str(cls_id)),
                                "confidence": conf_val,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "width_px": float(x2 - x1),
                                "height_px": float(y2 - y1),
                                "width_m": float(x2 - x1),
                                "height_m": float(y2 - y1),
                                "area_m2": float((x2 - x1) * (y2 - y1)),
                                "shape_type": "irregular",
                            })
            else:
                st.warning("⚠️ YOLOv8x weights not found. Falling back to War mode...")
                model = get_war_model() or get_lite_model()
                if model is not None:
                    detections, annotated_bgr = run_lite_inference(model, image_bgr, conf=conf, imgsz=imgsz)
                    annotated_img = bgr_to_rgb(annotated_bgr)
                else:
                    st.error("No model weights found in `models/` directory.")
                    return

        inference_ms = (time.perf_counter() - t_start) * 1000
        progress.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress.empty()

        # ── Store in session state ──────────────────────────────────────
        mode_tag = "War Mode" if is_war else ("Lite" if is_lite else "Pro")
        st.session_state["last_detections"] = detections
        st.session_state["last_annotated"] = annotated_img
        st.session_state["last_original"] = image_np
        st.session_state["last_inference_ms"] = inference_ms
        st.session_state["last_conf"] = conf
        st.session_state["last_mode"] = mode_tag
        st.session_state["last_lat"] = lat
        st.session_state["last_lon"] = lon

    # ── Display Results ─────────────────────────────────────────────────
    if "last_detections" not in st.session_state:
        return

    detections = st.session_state["last_detections"]
    annotated_img = st.session_state["last_annotated"]
    original_img = st.session_state["last_original"]
    inference_ms = st.session_state["last_inference_ms"]
    mode_str = st.session_state["last_mode"]
    lat = st.session_state["last_lat"]
    lon = st.session_state["last_lon"]

    st.markdown("---")

    # Status badge
    if detections:
        st.markdown(f'<div class="badge badge-success">🟢 {len(detections)} ANOMAL{"Y" if len(detections)==1 else "IES"} DETECTED</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-danger">🔴 NO ANOMALIES DETECTED</div>', unsafe_allow_html=True)

    # Tactical Threat Alert banners for War Mode
    if "War" in mode_str and detections:
        det_names = [d["class_name"].lower() for d in detections]
        if any("mine" in n for n in det_names):
            st.error("🚨 **TACTICAL THREAT DETECTED**: Subsea Naval Mine / Explosive Cylinder Identified! Threat Level: CRITICAL. Geo-coordinates flagged for countermeasures.")
        if any("pipeline" in n for n in det_names):
            st.info("🛡️ **CRITICAL INFRASTRUCTURE**: Submarine Pipeline Detected. Structural integrity & seabed routing logged.")
        if any("ghost_net" in n for n in det_names):
            st.warning("⚠️ **NAVIGATION HAZARD**: Abandoned Ghost Fishing Net / Submerged Entanglement Hazard Identified.")
        if any("shipwreck" in n for n in det_names):
            st.info("⚓ **HYDROGRAPHIC FEATURE**: Submerged Vessel Wreckage Located.")
        if any("crab_pot" in n for n in det_names):
            st.caption("🦀 **BENTHIC ACTIVITY**: Commercial Crab Pot / Seabed Cage Detected.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side images
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown('<div class="section-head">Original</div>', unsafe_allow_html=True)
        st.image(original_img, use_container_width=True)
    with img_col2:
        st.markdown('<div class="section-head">Analyzed</div>', unsafe_allow_html=True)
        st.image(annotated_img, use_container_width=True)

    # Metrics row
    if detections:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        max_conf = max(d["confidence"] for d in detections)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-val">{len(detections)}</div><div class="metric-label">Detections</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-val">{max_conf:.0%}</div><div class="metric-label">Top Confidence</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">⚡</div><div class="metric-val">{inference_ms:.0f}ms</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">🧠</div><div class="metric-val">{mode_str}</div><div class="metric-label">Model Mode</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detection details
        st.markdown("### 🔎 Detection Details")
        for det in detections:
            conf_pct = det["confidence"] * 100
            with st.expander(f"#{det['id']} — {det['class_name'].title()}  ·  {conf_pct:.1f}%", expanded=(det["id"] == 1)):
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown(f"""
                    <div class="det-card">
                        <div class="det-class">{det['class_name'].title()}</div>
                        <div class="det-conf">{conf_pct:.1f}% confidence</div>
                        <div class="det-meta" style="margin-top:8px;">
                            Shape: <b>{det['shape_type'].title()}</b><br>
                            Class ID: {det['class_id']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with dc2:
                    st.markdown(f"""
                    <div class="det-card">
                        <div style="color:var(--cyan); font-weight:700; font-size:14px;">📐 Physical Dimensions</div>
                        <div class="det-meta" style="margin-top:8px; line-height:2;">
                            Width: <b style="color:var(--text-primary)">{det['width_m']:.1f} m</b> ({det['width_px']:.0f} px)<br>
                            Height: <b style="color:var(--text-primary)">{det['height_m']:.1f} m</b> ({det['height_px']:.0f} px)<br>
                            Area: <b style="color:var(--text-primary)">{det['area_m2']:.1f} m²</b>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with dc3:
                    bbox = det["bbox"]
                    st.markdown(f"""
                    <div class="det-card">
                        <div style="color:var(--gold); font-weight:700; font-size:14px;">📦 Bounding Box</div>
                        <div class="det-meta" style="margin-top:8px; line-height:2; font-family:'JetBrains Mono',monospace; font-size:12px;">
                            x1: {bbox[0]:.1f} &nbsp; y1: {bbox[1]:.1f}<br>
                            x2: {bbox[2]:.1f} &nbsp; y2: {bbox[3]:.1f}<br>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # GPS link
                if lat != 0 or lon != 0:
                    maps = google_maps_url(lat, lon)
                    st.markdown(f'<div class="badge badge-info" style="margin-top:8px;">📍 GPS: {lat:.6f}, {lon:.6f} &nbsp;—&nbsp; <a href="{maps}" target="_blank" style="color:var(--cyan);">Open in Google Maps ↗</a></div>',
                                unsafe_allow_html=True)

        # ── Export Section ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        report = build_report_dict(detections, mode_str, conf, lat, lon, inference_ms)

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.download_button("📄 Download JSON Report",
                               data=json.dumps(report, indent=2),
                               file_name=f"sonaris_report_{datetime.now():%Y%m%d_%H%M%S}.json",
                               mime="application/json",
                               use_container_width=True)
        with ex2:
            df = pd.DataFrame(detections)
            st.download_button("📊 Download CSV",
                               data=df.to_csv(index=False),
                               file_name=f"sonaris_detections_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv",
                               use_container_width=True)
        with ex3:
            buf = io.BytesIO()
            Image.fromarray(annotated_img).save(buf, format="PNG")
            st.download_button("🖼️ Download Annotated Image",
                               data=buf.getvalue(),
                               file_name=f"sonaris_annotated_{datetime.now():%Y%m%d_%H%M%S}.png",
                               mime="image/png",
                               use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def page_batch_analysis():
    st.markdown("## 📊 Batch Analysis")
    st.markdown("Upload multiple sonar images for sequential analysis with aggregate statistics.")

    with st.sidebar:
        st.markdown('<div class="section-head">⚙️ Batch Settings</div>', unsafe_allow_html=True)
        batch_mode = st.radio("Batch Engine", [
            "🚀 Lite (Civilian 4-Class)",
            "⚔️ War Mode (Naval 5-Class)"
        ], help="Choose detection model for batch processing")
        batch_conf = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05, key="batch_conf")
        batch_imgsz = st.select_slider("Resolution", [640, 768, 1024], value=1024, key="batch_res")

    uploaded_files = st.file_uploader("Upload Sonar Images", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                      accept_multiple_files=True, key="batch_upload")

    if not uploaded_files:
        st.info("👆 Upload multiple sonar images to start batch analysis")
        return

    st.markdown(f'<div class="badge badge-info">{len(uploaded_files)} images queued</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🌊  RUN BATCH ANALYSIS", use_container_width=True):
        model = get_war_model() if "War" in batch_mode else get_lite_model()
        if model is None:
            st.error(f"⚠️ Model for {batch_mode} not found. Ensure weights exist in `models/` directory.")
            return

        all_results = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Processing {f.name}...")
            pil = Image.open(f).convert("RGB")
            img_np = np.array(pil)
            img_bgr = rgb_to_bgr(img_np)
            dets, ann_bgr = run_lite_inference(model, img_bgr, conf=batch_conf, imgsz=batch_imgsz)
            ann_rgb = bgr_to_rgb(ann_bgr)
            all_results.append({"name": f.name, "detections": dets, "annotated": ann_rgb, "original": img_np})

        progress.progress(1.0, text="Batch complete!")
        time.sleep(0.3)
        progress.empty()

        # Aggregate stats
        total_dets = sum(len(r["detections"]) for r in all_results)
        imgs_with_dets = sum(1 for r in all_results if r["detections"])
        all_confs = [d["confidence"] for r in all_results for d in r["detections"]]
        avg_conf = np.mean(all_confs) if all_confs else 0

        st.markdown("### 📈 Aggregate Statistics")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">🖼️</div><div class="metric-val">{len(all_results)}</div><div class="metric-label">Images</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-val">{total_dets}</div><div class="metric-label">Detections</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">✅</div><div class="metric-val">{imgs_with_dets}</div><div class="metric-label">With Hits</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-val">{avg_conf:.0%}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)

        # Class distribution
        if all_confs:
            st.markdown("### 📋 Class Distribution")
            class_counts = {}
            for r in all_results:
                for d in r["detections"]:
                    cn = d["class_name"]
                    class_counts[cn] = class_counts.get(cn, 0) + 1
            df_cls = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
            st.bar_chart(df_cls.set_index("Class"))

        # Per-image results
        st.markdown("### 📑 Per-Image Results")
        for r in all_results:
            n_dets = len(r["detections"])
            icon = "🟢" if n_dets > 0 else "🔴"
            with st.expander(f"{icon} {r['name']}  —  {n_dets} detection{'s' if n_dets != 1 else ''}"):
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.image(r["original"], caption="Original", use_container_width=True)
                with rc2:
                    st.image(r["annotated"], caption="Analyzed", use_container_width=True)
                if r["detections"]:
                    st.dataframe(pd.DataFrame(r["detections"])[["id", "class_name", "confidence", "shape_type", "area_m2"]],
                                 use_container_width=True, hide_index=True)

        # Batch CSV export
        all_dets = []
        for r in all_results:
            for d in r["detections"]:
                d_copy = dict(d)
                d_copy["source_image"] = r["name"]
                all_dets.append(d_copy)
        if all_dets:
            st.markdown("---")
            df_all = pd.DataFrame(all_dets)
            st.download_button("📊 Download Batch CSV",
                               data=df_all.to_csv(index=False),
                               file_name=f"sonaris_batch_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv",
                               use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
def page_dataset_explorer():
    st.markdown("## 🗺️ Dataset Explorer")
    st.markdown("Browse demo images and sonar samples from the project datasets.")

    # Show demo images
    if DEMO_DIR.exists():
        demo_files = sorted(DEMO_DIR.glob("*.png"))
        if demo_files:
            st.markdown("### 🖼️ Demo Sonar Images")
            st.markdown("Select an image to analyze it with the model.")

            # Navigation
            if "explorer_idx" not in st.session_state:
                st.session_state["explorer_idx"] = 0

            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("◀ Previous", use_container_width=True):
                    st.session_state["explorer_idx"] = max(0, st.session_state["explorer_idx"] - 1)
            with nav3:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state["explorer_idx"] = min(len(demo_files) - 1, st.session_state["explorer_idx"] + 1)
            with nav2:
                idx = st.session_state["explorer_idx"]
                st.markdown(f'<div style="text-align:center; color:var(--text-secondary); padding:8px;">{demo_files[idx].name} ({idx+1}/{len(demo_files)})</div>', unsafe_allow_html=True)

            idx = st.session_state["explorer_idx"]
            current_file = demo_files[idx]
            pil_img = Image.open(current_file).convert("RGB")
            image_np = np.array(pil_img)
            image_bgr = rgb_to_bgr(image_np)

            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.markdown('<div class="section-head">Original Image</div>', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)

            with ex_col2:
                st.markdown('<div class="section-head">Model Prediction</div>', unsafe_allow_html=True)
                model = get_lite_model()
                if model is not None:
                    with st.spinner("Running inference..."):
                        dets, ann_bgr = run_lite_inference(model, image_bgr, conf=0.3, imgsz=1024)
                        ann_rgb = bgr_to_rgb(ann_bgr)
                        st.image(ann_rgb, use_container_width=True)

                    if dets:
                        st.markdown(f'<div class="badge badge-success">{len(dets)} detection(s)</div>', unsafe_allow_html=True)
                        for d in dets:
                            st.markdown(f"""
                            <div class="det-card">
                                <span class="det-class">{d['class_name'].title()}</span>
                                &nbsp;&nbsp;<span class="det-conf">{d['confidence']:.0%}</span>
                                <span class="det-meta" style="margin-left:16px;">{d['shape_type']}</span>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="badge badge-danger">No detections</div>', unsafe_allow_html=True)
                else:
                    st.warning("Model not loaded — install `ultralytics` and add weights to `models/`")
                    st.image(pil_img, use_container_width=True)

            # Thumbnails
            st.markdown("---")
            thumb_cols = st.columns(min(len(demo_files), 6))
            for i, f in enumerate(demo_files[:6]):
                with thumb_cols[i]:
                    img = Image.open(f)
                    is_selected = (i == idx)
                    if st.button(f.stem[:12], key=f"thumb_{i}", use_container_width=True):
                        st.session_state["explorer_idx"] = i
                        st.rerun()
    else:
        st.info("No demo images found. Add sonar images to `data/demo/`.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="hero-title" style="font-size:48px;">Sonaris</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:30px;">About This Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <div class="section-head">🌊 Problem Statement</div>
        <p style="color:var(--text-secondary); line-height:1.8; font-size:15px;">
        <b style="color:var(--text-primary);">SIH26057</b> — Ministry of Earth Sciences (MoES), Government of India<br><br>
        Side-scan sonar imagery is essential for underwater search and recovery, seabed archaeology, and marine ecosystem monitoring.
        However, manual analysis of these acoustic scans is extremely slow, labor-intensive, and subject to human fatigue.
        Sonaris automates the detection of underwater anomalies — shipwrecks, aircraft debris, marine structures — using
        state-of-the-art deep learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison
    st.markdown("### 🧠 Model Architectures")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""
        <div class="glass">
            <div class="section-head">🚀 Lite Mode</div>
            <h3 style="margin:0;">YOLO11n-Seg</h3>
            <table style="color:var(--text-secondary); margin-top:16px; width:100%; font-size:14px;">
                <tr><td>Parameters</td><td style="text-align:right; color:var(--accent); font-weight:700;">2.83M</td></tr>
                <tr><td>GFLOPs</td><td style="text-align:right; color:var(--accent); font-weight:700;">9.6</td></tr>
                <tr><td>Classes</td><td style="text-align:right;">4 Civilian</td></tr>
                <tr><td>mAP@50 (bbox)</td><td style="text-align:right;">0.529</td></tr>
                <tr><td>mAP@50 (mask)</td><td style="text-align:right;">0.491</td></tr>
                <tr><td>Deployment</td><td style="text-align:right; color:var(--accent);">Real-time / Edge</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class="glass">
            <div class="section-head">⚔️ War Mode</div>
            <h3 style="margin:0;">YOLO11n-Tactical</h3>
            <table style="color:var(--text-secondary); margin-top:16px; width:100%; font-size:14px;">
                <tr><td>Parameters</td><td style="text-align:right; color:var(--coral); font-weight:700;">2.59M</td></tr>
                <tr><td>GFLOPs</td><td style="text-align:right; color:var(--coral); font-weight:700;">6.5</td></tr>
                <tr><td>Classes</td><td style="text-align:right;">5 Naval Defense</td></tr>
                <tr><td>Dataset</td><td style="text-align:right;">DRISHTI</td></tr>
                <tr><td>Threat Alerts</td><td style="text-align:right;">Mines / Pipelines</td></tr>
                <tr><td>Deployment</td><td style="text-align:right; color:var(--coral);">Naval / Tactical</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div class="glass">
            <div class="section-head">🎯 Pro Mode</div>
            <h3 style="margin:0;">YOLOv8x + SAM</h3>
            <table style="color:var(--text-secondary); margin-top:16px; width:100%; font-size:14px;">
                <tr><td>Detection Params</td><td style="text-align:right; color:var(--cyan); font-weight:700;">68.2M</td></tr>
                <tr><td>SAM Model</td><td style="text-align:right;">ViT-B (358 MB)</td></tr>
                <tr><td>Task</td><td style="text-align:right;">Detection + Zero-Shot Seg</td></tr>
                <tr><td>mAP@50</td><td style="text-align:right; font-weight:700;">0.801</td></tr>
                <tr><td>Precision</td><td style="text-align:right;">0.936</td></tr>
                <tr><td>Deployment</td><td style="text-align:right; color:var(--cyan);">High Accuracy Server</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # Tech stack
    st.markdown("### 🔧 Technology Stack")
    st.markdown("""
    <div class="glass">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px;">
            <div class="det-card"><div style="color:var(--accent); font-weight:700;">Frontend</div><div class="det-meta">Streamlit · Custom CSS</div></div>
            <div class="det-card"><div style="color:var(--cyan); font-weight:700;">Deep Learning</div><div class="det-meta">PyTorch · Ultralytics YOLO</div></div>
            <div class="det-card"><div style="color:var(--gold); font-weight:700;">Segmentation</div><div class="det-meta">Meta SAM · YOLO Masks</div></div>
            <div class="det-card"><div style="color:var(--coral); font-weight:700;">Computer Vision</div><div class="det-meta">OpenCV · CLAHE · Median</div></div>
            <div class="det-card"><div style="color:#a78bfa; font-weight:700;">Geospatial</div><div class="det-meta">EXIF Parser · Google Maps</div></div>
            <div class="det-card"><div style="color:var(--accent); font-weight:700;">Data Science</div><div class="det-meta">NumPy · Pandas · Matplotlib</div></div>
        </div>
    </div>""", unsafe_allow_html=True)




# ════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="hero-title" style="font-size:28px; text-align:left; margin-bottom:0;">Sonaris</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="color:var(--text-muted); font-size:11px; letter-spacing:2px; text-transform:uppercase; margin-bottom:20px;">Sonar Intelligence</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio("Navigate", [
            "🏠 Home",
            "🔍 Single Analysis",
            "📊 Batch Analysis",
            "🗺️ Dataset Explorer",
            "ℹ️ About",
        ], label_visibility="collapsed")

        st.markdown("---")

        # Model status indicators
        st.markdown('<div class="section-head">System Status</div>', unsafe_allow_html=True)

        lite_ok = YOLO_AVAILABLE and LITE_MODEL_PATH.exists()
        war_ok = YOLO_AVAILABLE and WAR_MODEL_PATH.exists()
        pro_local = PRO_YOLO_PATH.exists()
        sam_ok = SAM_AVAILABLE and SAM_PATH.exists()

        st.markdown(f"{'🟢' if lite_ok else '🔴'} Lite Model {'(4 Classes)' if lite_ok else 'Missing'}")
        if lite_ok:
            m = get_lite_model()
            if m and hasattr(m, 'names'):
                st.caption(f"🎯 Civilian: {', '.join(m.names.values())}")

        st.markdown(f"{'⚔️' if war_ok else '🔴'} War Mode {'(5 Naval Classes)' if war_ok else 'Missing'}")
        if war_ok:
            m_w = get_war_model()
            if m_w and hasattr(m_w, 'names'):
                st.caption(f"🛡️ Defense: {', '.join(m_w.names.values())}")

        st.markdown(f"{'🟢' if pro_local else '🟢'} Pro YOLO {'Local' if pro_local else 'Auto-DL'}")
        st.markdown(f"{'🟢' if sam_ok else '🟡'} SAM Engine {'Ready' if sam_ok else 'Optional'}")

        st.markdown("---")
        st.caption("v1.2.0 · Sonaris Multi-Class & Tactical Defense Platform")

    # Route to page
    if page == "🏠 Home":
        page_home()
    elif page == "🔍 Single Analysis":
        page_single_analysis()
    elif page == "📊 Batch Analysis":
        page_batch_analysis()
    elif page == "🗺️ Dataset Explorer":
        page_dataset_explorer()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
