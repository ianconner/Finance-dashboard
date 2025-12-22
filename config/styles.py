# config/styles.py - Balanced JARVIS Theme (Performance + Style)

import streamlit as st

def inject_custom_css():
    """Inject balanced JARVIS-style CSS - optimized for performance"""
    st.markdown("""
    <style>
    /* Import lightweight sci-fi font */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap');
    
    /* ==================== MAIN APP BACKGROUND ==================== */
    .stApp {
        background: 
            radial-gradient(ellipse at top, rgba(13, 110, 253, 0.08) 0%, transparent 50%),
            linear-gradient(135deg, #000814 0%, #001d3d 50%, #000814 100%);
        font-family: 'Rajdhani', -apple-system, sans-serif;
    }
    
    /* Subtle scanline effect (lightweight) */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            0deg,
            rgba(88, 166, 255, 0.02) 0px,
            transparent 2px
        );
        pointer-events: none;
        opacity: 0.5;
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0, 8, 20, 0.95) 0%, rgba(0, 13, 26, 0.98) 100%);
        border-right: 2px solid rgba(0, 180, 255, 0.3);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] * {
        color: #00d4ff !important;
    }
    
    /* Glowing border on sidebar */
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: -2px;
        width: 2px;
        height: 100%;
        background: linear-gradient(180deg, transparent, rgba(0, 180, 255, 0.6), transparent);
        animation: sidebarGlow 4s ease-in-out infinite;
    }
    
    @keyframes sidebarGlow {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }
    
    /* ==================== HOLOGRAPHIC METRIC CARDS ==================== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.12) 0%, rgba(0, 100, 200, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 180, 255, 0.4);
        box-shadow: 
            0 8px 32px rgba(0, 180, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        transition: all 0.3s ease;
    }
    
    /* Shine effect on hover */
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.2), transparent);
        transition: left 0.6s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 
            0 12px 48px rgba(0, 180, 255, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    div[data-testid="stMetric"]:hover::before {
        left: 100%;
    }
    
    /* Corner accents */
    div[data-testid="stMetric"]::after {
        content: '';
        position: absolute;
        top: 8px;
        left: 8px;
        width: 20px;
        height: 20px;
        border-top: 2px solid rgba(0, 180, 255, 0.6);
        border-left: 2px solid rgba(0, 180, 255, 0.6);
    }
    
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 8px rgba(0, 180, 255, 0.6);
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 180, 255, 0.8);
        filter: brightness(1.1);
    }
    
    /* ==================== HEADERS ==================== */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 30px rgba(0, 180, 255, 0.6);
        letter-spacing: 1px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        animation: titlePulse 4s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 30px rgba(0, 180, 255, 0.6); }
        50% { text-shadow: 0 0 40px rgba(0, 180, 255, 0.9); }
    }
    
    /* ==================== HOLOGRAPHIC BUTTONS ==================== */
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.25) 0%, rgba(0, 100, 200, 0.25) 100%) !important;
        border: 2px solid rgba(0, 180, 255, 0.5) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(0, 180, 255, 0.4);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.4s, height 0.4s;
    }
    
    button[kind="primary"]:hover::before {
        width: 200px;
        height: 200px;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.35) 0%, rgba(0, 100, 200, 0.35) 100%) !important;
        border-color: rgba(0, 180, 255, 0.8) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.5) !important;
    }
    
    button[kind="secondary"] {
        background: rgba(0, 180, 255, 0.08) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(0, 180, 255, 0.15) !important;
        border-color: rgba(0, 180, 255, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.3) !important;
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b4ff 0%, #00e4ff 50%, #00b4ff 100%) !important;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.5);
        animation: progressPulse 2s ease-in-out infinite;
    }
    
    @keyframes progressPulse {
        0%, 100% { box-shadow: 0 0 15px rgba(0, 180, 255, 0.5); }
        50% { box-shadow: 0 0 25px rgba(0, 180, 255, 0.8); }
    }
    
    /* ==================== ALERT PANELS ==================== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.12) 0%, rgba(0, 255, 157, 0.08) 100%) !important;
        border: 1px solid rgba(0, 255, 157, 0.4) !important;
        border-left: 4px solid #00ff9d !important;
        border-radius: 10px !important;
        color: #00ff9d !important;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.15) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.12) 0%, rgba(255, 193, 7, 0.08) 100%) !important;
        border: 1px solid rgba(255, 193, 7, 0.4) !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 10px !important;
        color: #ffc107 !important;
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.15) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.12) 0%, rgba(255, 82, 82, 0.08) 100%) !important;
        border: 1px solid rgba(255, 82, 82, 0.4) !important;
        border-left: 4px solid #ff5252 !important;
        border-radius: 10px !important;
        color: #ff5252 !important;
        box-shadow: 0 0 20px rgba(255, 82, 82, 0.15) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.12) 0%, rgba(0, 180, 255, 0.08) 100%) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        border-left: 4px solid #00b4ff !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.15) !important;
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
        border-bottom: 2px solid rgba(0, 180, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 180, 255, 0.08);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(0, 180, 255, 0.3);
        border-bottom: none;
        color: #00d4ff;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 255, 0.12);
        box-shadow: 0 -5px 15px rgba(0, 180, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.2) 0%, rgba(0, 180, 255, 0.08) 100%);
        color: #ffffff !important;
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 0 -5px 20px rgba(0, 180, 255, 0.3);
        position: relative;
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 20%;
        right: 20%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.8);
    }
    
    /* ==================== DATAFRAMES ==================== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 180, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0, 180, 255, 0.1);
    }
    
    /* ==================== EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.1) 0%, rgba(0, 100, 200, 0.06) 100%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.15) 0%, rgba(0, 100, 200, 0.1) 100%) !important;
        border-color: rgba(0, 180, 255, 0.5) !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.2) !important;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: rgba(0, 180, 255, 0.06);
        border: 2px dashed rgba(0, 180, 255, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(0, 180, 255, 0.1);
        border-color: rgba(0, 180, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.2);
    }
    
    /* ==================== INPUT FIELDS ==================== */
    input, textarea, select {
        background: rgba(0, 180, 255, 0.06) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: rgba(0, 180, 255, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(0, 180, 255, 0.2) !important;
        background: rgba(0, 180, 255, 0.1) !important;
    }
    
    /* ==================== SLIDER ==================== */
    .stSlider [data-baseweb="slider"] {
        background: rgba(0, 180, 255, 0.2);
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #00d4ff, #0096ff);
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.5), transparent) !important;
        margin: 3rem 0 !important;
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.3);
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 8, 20, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.5), rgba(0, 100, 200, 0.5));
        border-radius: 10px;
        border: 2px solid rgba(0, 8, 20, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.7), rgba(0, 100, 200, 0.7));
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.5);
    }
    
    /* ==================== COLUMN SPACING ==================== */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render a JARVIS-style hero section"""
    st.markdown(f"""
    <div style='text-align: center; padding: 2.5rem 0 2rem 0; position: relative;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem; letter-spacing: 2px;'>{title}</h1>
        <p style='color: #00d4ff; font-size: 1.1rem; text-transform: uppercase; 
                  letter-spacing: 3px; text-shadow: 0 0 15px rgba(0, 180, 255, 0.6);'>{subtitle}</p>
        <div style='width: 150px; height: 2px; margin: 1.5rem auto;
                    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                    box-shadow: 0 0 15px rgba(0, 180, 255, 0.8);'></div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, subtitle=""):
    """Render a holographic section header"""
    subtitle_html = f"<p style='color: #00d4ff; font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem 0 2rem 0;'>
        <h2 style='font-size: 1.75rem; margin: 0; letter-spacing: 1px;'>{icon} {title}</h2>
        {subtitle_html}
        <div style='width: 100px; height: 2px; margin: 1rem auto;
                    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                    box-shadow: 0 0 10px rgba(0, 180, 255, 0.6);'></div>
    </div>
    """, unsafe_allow_html=True)
