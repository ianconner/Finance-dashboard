# config/styles.py - JARVIS Holographic Interface

import streamlit as st

def inject_custom_css():
    """Inject JARVIS-style holographic futuristic CSS theme"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* ==================== MAIN APP BACKGROUND ==================== */
    .stApp {
        background: 
            radial-gradient(ellipse at top, rgba(13, 110, 253, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom, rgba(88, 166, 255, 0.1) 0%, transparent 50%),
            linear-gradient(135deg, #000814 0%, #001d3d 50%, #000814 100%);
        font-family: 'Rajdhani', sans-serif;
        position: relative;
    }
    
    /* Animated scanlines effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            0deg,
            rgba(88, 166, 255, 0.03) 0px,
            transparent 1px,
            transparent 2px,
            rgba(88, 166, 255, 0.03) 3px
        );
        pointer-events: none;
        z-index: 1;
        animation: scanlines 8s linear infinite;
    }
    
    @keyframes scanlines {
        0% { transform: translateY(0); }
        100% { transform: translateY(10px); }
    }
    
    /* ==================== SIDEBAR STYLING ==================== */
    [data-testid="stSidebar"] {
        background: rgba(0, 8, 20, 0.85);
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(0, 180, 255, 0.3);
        box-shadow: 
            inset -1px 0 20px rgba(0, 180, 255, 0.1),
            4px 0 30px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 2px;
        height: 100%;
        background: linear-gradient(
            180deg,
            transparent 0%,
            rgba(0, 180, 255, 0.8) 50%,
            transparent 100%
        );
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    [data-testid="stSidebar"] * {
        color: #00d4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* ==================== HOLOGRAPHIC METRIC CARDS ==================== */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            rgba(0, 180, 255, 0.08) 0%,
            rgba(0, 100, 200, 0.05) 100%
        );
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 180, 255, 0.3);
        box-shadow: 
            0 8px 32px rgba(0, 180, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(0, 180, 255, 0.2),
            transparent
        );
        transition: left 0.5s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 
            0 12px 48px rgba(0, 180, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 30px rgba(0, 180, 255, 0.3);
    }
    
    div[data-testid="stMetric"]:hover::before {
        left: 100%;
    }
    
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 
            0 0 20px rgba(0, 180, 255, 0.8),
            0 0 40px rgba(0, 180, 255, 0.4);
        background: linear-gradient(135deg, #00d4ff 0%, #0096ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* ==================== HEADERS WITH HOLOGRAPHIC EFFECT ==================== */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00d4ff 0%, #0096ff 50%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 180, 255, 0.5);
        position: relative;
        letter-spacing: 2px;
    }
    
    h1 {
        animation: textGlow 3s ease-in-out infinite;
    }
    
    @keyframes textGlow {
        0%, 100% {
            filter: brightness(1) drop-shadow(0 0 20px rgba(0, 180, 255, 0.5));
        }
        50% {
            filter: brightness(1.2) drop-shadow(0 0 30px rgba(0, 180, 255, 0.8));
        }
    }
    
    /* ==================== HOLOGRAPHIC BUTTONS ==================== */
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.2) 0%, rgba(0, 100, 200, 0.2) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(0, 180, 255, 0.5) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 
            0 0 20px rgba(0, 180, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(0, 180, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    button[kind="primary"]:hover::before {
        width: 300px;
        height: 300px;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.3) 0%, rgba(0, 100, 200, 0.3) 100%) !important;
        border-color: rgba(0, 180, 255, 0.8) !important;
        box-shadow: 
            0 0 40px rgba(0, 180, 255, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    
    button[kind="secondary"] {
        background: rgba(0, 180, 255, 0.05) !important;
        backdrop-filter: blur(5px) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(0, 180, 255, 0.15) !important;
        border-color: rgba(0, 180, 255, 0.6) !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.3) !important;
    }
    
    /* ==================== HOLOGRAPHIC PROGRESS BAR ==================== */
    .stProgress > div > div {
        background: linear-gradient(
            90deg,
            rgba(0, 180, 255, 0.8) 0%,
            rgba(0, 255, 255, 0.9) 50%,
            rgba(0, 180, 255, 0.8) 100%
        ) !important;
        border-radius: 10px;
        box-shadow: 
            0 0 20px rgba(0, 180, 255, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% {
            box-shadow: 
                0 0 20px rgba(0, 180, 255, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        50% {
            box-shadow: 
                0 0 40px rgba(0, 180, 255, 0.9),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }
    }
    
    /* ==================== ALERT PANELS ==================== */
    .stSuccess {
        background: rgba(0, 255, 157, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 255, 157, 0.4) !important;
        border-left: 4px solid rgba(0, 255, 157, 0.8) !important;
        border-radius: 10px !important;
        color: #00ff9d !important;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.2) !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 193, 7, 0.4) !important;
        border-left: 4px solid rgba(255, 193, 7, 0.8) !important;
        border-radius: 10px !important;
        color: #ffc107 !important;
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.2) !important;
    }
    
    .stError {
        background: rgba(255, 82, 82, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 82, 82, 0.4) !important;
        border-left: 4px solid rgba(255, 82, 82, 0.8) !important;
        border-radius: 10px !important;
        color: #ff5252 !important;
        box-shadow: 0 0 20px rgba(255, 82, 82, 0.2) !important;
    }
    
    .stInfo {
        background: rgba(0, 180, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        border-left: 4px solid rgba(0, 180, 255, 0.8) !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.2) !important;
    }
    
    /* ==================== HOLOGRAPHIC TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: 2px solid rgba(0, 180, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 180, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(0, 180, 255, 0.2);
        border-bottom: none;
        color: #00d4ff;
        padding: 1rem 2rem;
        font-weight: 600;
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 255, 0.1);
        box-shadow: 0 -5px 20px rgba(0, 180, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            180deg,
            rgba(0, 180, 255, 0.2) 0%,
            rgba(0, 180, 255, 0.05) 100%
        );
        color: #ffffff !important;
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 
            0 -5px 30px rgba(0, 180, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        animation: tabGlow 2s ease-in-out infinite;
    }
    
    @keyframes tabGlow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* ==================== GLASS DATAFRAMES ==================== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 180, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 180, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* ==================== HOLOGRAPHIC EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: linear-gradient(
            135deg,
            rgba(0, 180, 255, 0.1) 0%,
            rgba(0, 100, 200, 0.05) 100%
        ) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(
            135deg,
            rgba(0, 180, 255, 0.15) 0%,
            rgba(0, 100, 200, 0.1) 100%
        ) !important;
        border-color: rgba(0, 180, 255, 0.5) !important;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.2) !important;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: rgba(0, 180, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(0, 180, 255, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(0, 180, 255, 0.1);
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.2);
    }
    
    /* ==================== INPUT FIELDS ==================== */
    input, textarea, select {
        background: rgba(0, 180, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 180, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #00d4ff !important;
        padding: 0.75rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: rgba(0, 180, 255, 0.8) !important;
        box-shadow: 
            0 0 0 3px rgba(0, 180, 255, 0.2),
            0 0 20px rgba(0, 180, 255, 0.3) !important;
        background: rgba(0, 180, 255, 0.1) !important;
    }
    
    /* ==================== SLIDER ==================== */
    .stSlider [data-baseweb="slider"] {
        background: rgba(0, 180, 255, 0.2);
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0096ff 100%);
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(0, 180, 255, 0.5) 50%,
            transparent 100%
        ) !important;
        margin: 3rem 0 !important;
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.3);
    }
    
    /* ==================== CUSTOM SCROLLBAR ==================== */
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
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.8), rgba(0, 100, 200, 0.8));
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.5);
    }
    
    /* ==================== HEXAGONAL CORNER ACCENTS ==================== */
    .metric-accent::before,
    .metric-accent::after {
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(0, 180, 255, 0.5);
    }
    
    .metric-accent::before {
        top: -1px;
        left: -1px;
        border-right: none;
        border-bottom: none;
    }
    
    .metric-accent::after {
        bottom: -1px;
        right: -1px;
        border-left: none;
        border-top: none;
    }
    
    /* ==================== PULSING GLOW ANIMATION ==================== */
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 180, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 40px rgba(0, 180, 255, 0.6);
        }
    }
    
    .glow-element {
        animation: pulseGlow 3s ease-in-out infinite;
    }
    
    /* ==================== DATA STREAM EFFECT ==================== */
    @keyframes dataStream {
        0% {
            transform: translateY(-100%);
            opacity: 0;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            transform: translateY(100%);
            opacity: 0;
        }
    }
    
    /* ==================== COLUMN SPACING ==================== */
    [data-testid="column"] {
        padding: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render a JARVIS-style hero section"""
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem 0; margin-bottom: 2rem; position: relative;'>
        <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    width: 400px; height: 400px; 
                    background: radial-gradient(circle, rgba(0, 180, 255, 0.1) 0%, transparent 70%);
                    filter: blur(40px); z-index: 0;'></div>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; position: relative; z-index: 1;
                   text-shadow: 0 0 40px rgba(0, 180, 255, 0.8);'>{title}</h1>
        <p style='color: #00d4ff; font-size: 1.2rem; font-family: "Orbitron", sans-serif;
                  text-transform: uppercase; letter-spacing: 3px; position: relative; z-index: 1;
                  text-shadow: 0 0 20px rgba(0, 180, 255, 0.6);'>{subtitle}</p>
        <div style='width: 200px; height: 2px; margin: 1rem auto;
                    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                    box-shadow: 0 0 10px rgba(0, 180, 255, 0.8);'></div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, subtitle=""):
    """Render a holographic section header"""
    subtitle_html = f"<p style='color: #00d4ff; font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8; font-family: \"Rajdhani\", sans-serif;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem 0 2rem 0; position: relative;'>
        <div style='display: inline-block; position: relative;'>
            <h2 style='font-size: 1.8rem; margin: 0; font-family: "Orbitron", sans-serif;
                       text-shadow: 0 0 30px rgba(0, 180, 255, 0.8);'>{icon} {title}</h2>
            <div style='position: absolute; bottom: -5px; left: 0; right: 0; height: 2px;
                        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                        box-shadow: 0 0 10px rgba(0, 180, 255, 0.8);'></div>
        </div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_glass_panel(content):
    """Render content in a glass morphism panel"""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(0, 180, 255, 0.05) 0%, rgba(0, 100, 200, 0.02) 100%);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(0, 180, 255, 0.2);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 180, 255, 0.1);
                margin: 1rem 0;'>
        {content}
    </div>
    """, unsafe_allow_html=True)
