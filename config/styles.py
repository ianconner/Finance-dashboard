# config/styles.py - Optimized Futuristic Theme

import streamlit as st

def inject_custom_css():
    """Inject optimized futuristic CSS theme"""
    st.markdown("""
    <style>
    /* ==================== MAIN APP ==================== */
    .stApp {
        background: linear-gradient(135deg, #000814 0%, #001d3d 50%, #000814 100%);
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: rgba(0, 8, 20, 0.95);
        border-right: 1px solid rgba(0, 180, 255, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #00d4ff !important;
    }
    
    /* ==================== METRIC CARDS ==================== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.1) 0%, rgba(0, 100, 200, 0.05) 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 180, 255, 0.3);
        box-shadow: 0 4px 12px rgba(0, 180, 255, 0.15);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 180, 255, 0.5);
    }
    
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 180, 255, 0.5);
    }
    
    /* ==================== HEADERS ==================== */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 180, 255, 0.4);
    }
    
    /* ==================== BUTTONS ==================== */
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.3) 0%, rgba(0, 100, 200, 0.3) 100%) !important;
        border: 1px solid rgba(0, 180, 255, 0.5) !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.4) 0%, rgba(0, 100, 200, 0.4) 100%) !important;
        border-color: rgba(0, 180, 255, 0.8) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }
    
    button[kind="secondary"] {
        background: rgba(0, 180, 255, 0.1) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(0, 180, 255, 0.2) !important;
        border-color: rgba(0, 180, 255, 0.5) !important;
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b4ff 0%, #00e4ff 100%) !important;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.4);
    }
    
    /* ==================== ALERTS ==================== */
    .stSuccess {
        background: rgba(0, 255, 157, 0.1) !important;
        border: 1px solid rgba(0, 255, 157, 0.4) !important;
        border-left: 3px solid #00ff9d !important;
        border-radius: 8px !important;
        color: #00ff9d !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.1) !important;
        border: 1px solid rgba(255, 193, 7, 0.4) !important;
        border-left: 3px solid #ffc107 !important;
        border-radius: 8px !important;
        color: #ffc107 !important;
    }
    
    .stError {
        background: rgba(255, 82, 82, 0.1) !important;
        border: 1px solid rgba(255, 82, 82, 0.4) !important;
        border-left: 3px solid #ff5252 !important;
        border-radius: 8px !important;
        color: #ff5252 !important;
    }
    
    .stInfo {
        background: rgba(0, 180, 255, 0.1) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        border-left: 3px solid #00b4ff !important;
        border-radius: 8px !important;
        color: #00d4ff !important;
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 180, 255, 0.05);
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(0, 180, 255, 0.2);
        border-bottom: none;
        color: #00d4ff;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 180, 255, 0.15);
        color: #ffffff !important;
        border-color: rgba(0, 180, 255, 0.5);
        border-bottom: 2px solid #00d4ff;
    }
    
    /* ==================== DATAFRAMES ==================== */
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid rgba(0, 180, 255, 0.2);
    }
    
    /* ==================== EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: rgba(0, 180, 255, 0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 180, 255, 0.12) !important;
        border-color: rgba(0, 180, 255, 0.4) !important;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: rgba(0, 180, 255, 0.05);
        border: 2px dashed rgba(0, 180, 255, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(0, 180, 255, 0.1);
        border-color: rgba(0, 180, 255, 0.5);
    }
    
    /* ==================== INPUTS ==================== */
    input, textarea, select {
        background: rgba(0, 180, 255, 0.05) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #00d4ff !important;
        transition: all 0.2s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: rgba(0, 180, 255, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(0, 180, 255, 0.2) !important;
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.4), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 8, 20, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 180, 255, 0.4);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 180, 255, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render a simple hero section"""
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0; margin-bottom: 1rem;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{title}</h1>
        <p style='color: #00d4ff; font-size: 1rem;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, subtitle=""):
    """Render a section header"""
    subtitle_html = f"<p style='color: #00d4ff; font-size: 0.875rem; margin-top: 0.5rem; opacity: 0.8;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
        <h2 style='font-size: 1.5rem; margin: 0;'>{icon} {title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)
