# config/styles.py - Enhanced JARVIS Theme (Dialed Up!)

import streamlit as st

def inject_custom_css():
    """Inject enhanced JARVIS-style CSS - more intensity, balanced performance"""
    st.markdown("""
    <style>
    /* Import sci-fi fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600;700&display=swap');
    
    /* ==================== MAIN APP BACKGROUND ==================== */
    .stApp {
        background: 
            radial-gradient(ellipse at top right, rgba(13, 110, 253, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom left, rgba(0, 180, 255, 0.1) 0%, transparent 50%),
            linear-gradient(135deg, #000510 0%, #001a35 50%, #000510 100%);
        font-family: 'Rajdhani', sans-serif;
        position: relative;
    }
    
    /* Animated scanlines - more visible */
    .stApp::after {
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
        animation: scanlineMove 8s linear infinite;
        opacity: 0.7;
    }
    
    @keyframes scanlineMove {
        0% { transform: translateY(0); }
        100% { transform: translateY(20px); }
    }
    
    /* Floating particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(0, 180, 255, 0.3), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(0, 180, 255, 0.3), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(0, 180, 255, 0.3), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(0, 180, 255, 0.3), transparent),
            radial-gradient(2px 2px at 90% 60%, rgba(0, 180, 255, 0.3), transparent);
        background-size: 200% 200%;
        animation: particles 20s ease-in-out infinite;
        pointer-events: none;
        opacity: 0.4;
    }
    
    @keyframes particles {
        0%, 100% { transform: translate(0, 0); }
        25% { transform: translate(2%, -2%); }
        50% { transform: translate(-2%, 2%); }
        75% { transform: translate(2%, 2%); }
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0, 8, 20, 0.98) 0%, rgba(0, 13, 26, 0.98) 100%);
        border-right: 2px solid rgba(0, 180, 255, 0.4);
        box-shadow: 
            inset -1px 0 15px rgba(0, 180, 255, 0.15),
            4px 0 30px rgba(0, 0, 0, 0.6);
        position: relative;
    }
    
    [data-testid="stSidebar"] * {
        color: #00d4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* Animated glowing border */
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: -2px;
        width: 2px;
        height: 100%;
        background: linear-gradient(
            180deg,
            transparent 0%,
            rgba(0, 180, 255, 0.8) 30%,
            rgba(0, 255, 255, 1) 50%,
            rgba(0, 180, 255, 0.8) 70%,
            transparent 100%
        );
        animation: borderFlow 4s ease-in-out infinite;
    }
    
    @keyframes borderFlow {
        0%, 100% { opacity: 0.5; transform: translateY(-10%); }
        50% { opacity: 1; transform: translateY(10%); }
    }
    
    /* ==================== HOLOGRAPHIC METRIC CARDS ==================== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.15) 0%, rgba(0, 100, 200, 0.1) 100%);
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 180, 255, 0.5);
        box-shadow: 
            0 8px 32px rgba(0, 180, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            0 0 40px rgba(0, 180, 255, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Animated shine sweep */
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(0, 180, 255, 0.3) 50%,
            transparent 70%
        );
        transform: translateX(-100%) translateY(-100%) rotate(45deg);
        transition: transform 0.8s;
    }
    
    div[data-testid="stMetric"]:hover::before {
        transform: translateX(100%) translateY(100%) rotate(45deg);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: rgba(0, 180, 255, 0.8);
        box-shadow: 
            0 16px 64px rgba(0, 180, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 60px rgba(0, 180, 255, 0.2);
    }
    
    /* Corner brackets (HUD style) */
    div[data-testid="stMetric"]::after {
        content: '';
        position: absolute;
        top: 10px;
        left: 10px;
        width: 25px;
        height: 25px;
        border-top: 3px solid rgba(0, 180, 255, 0.7);
        border-left: 3px solid rgba(0, 180, 255, 0.7);
        animation: cornerPulse 3s ease-in-out infinite;
    }
    
    @keyframes cornerPulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 12px rgba(0, 212, 255, 0.8);
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.25rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 
            0 0 20px rgba(0, 180, 255, 1),
            0 0 40px rgba(0, 180, 255, 0.6),
            0 0 60px rgba(0, 180, 255, 0.3);
        filter: brightness(1.2);
        animation: valueGlow 3s ease-in-out infinite;
    }
    
    @keyframes valueGlow {
        0%, 100% { filter: brightness(1.1); }
        50% { filter: brightness(1.3); }
    }
    
    /* ==================== HEADERS ==================== */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 900 !important;
        text-shadow: 
            0 0 30px rgba(0, 180, 255, 0.8),
            0 0 60px rgba(0, 180, 255, 0.4);
        letter-spacing: 2px;
        position: relative;
    }
    
    h1 {
        font-size: 2.75rem !important;
        animation: titlePulse 4s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { 
            text-shadow: 
                0 0 30px rgba(0, 180, 255, 0.8),
                0 0 60px rgba(0, 180, 255, 0.4);
        }
        50% { 
            text-shadow: 
                0 0 40px rgba(0, 180, 255, 1),
                0 0 80px rgba(0, 180, 255, 0.6),
                0 0 120px rgba(0, 180, 255, 0.3);
        }
    }
    
    /* ==================== HOLOGRAPHIC BUTTONS ==================== */
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.3) 0%, rgba(0, 100, 200, 0.3) 100%) !important;
        border: 2px solid rgba(0, 180, 255, 0.6) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        border-radius: 12px !important;
        padding: 0.85rem 2.5rem !important;
        box-shadow: 
            0 0 25px rgba(0, 180, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Ripple effect */
    button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(0, 180, 255, 0.5);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    button[kind="primary"]:hover::before {
        width: 300px;
        height: 300px;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.4) 0%, rgba(0, 100, 200, 0.4) 100%) !important;
        border-color: rgba(0, 180, 255, 1) !important;
        color: #ffffff !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 0 40px rgba(0, 180, 255, 0.7),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    button[kind="secondary"] {
        background: rgba(0, 180, 255, 0.1) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        color: #00d4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(0, 180, 255, 0.2) !important;
        border-color: rgba(0, 180, 255, 0.7) !important;
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div {
        background: linear-gradient(
            90deg,
            rgba(0, 180, 255, 0.9) 0%,
            rgba(0, 255, 255, 1) 50%,
            rgba(0, 180, 255, 0.9) 100%
        ) !important;
        border-radius: 10px;
        box-shadow: 
            0 0 20px rgba(0, 180, 255, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: progressFlow 2s ease-in-out infinite;
    }
    
    @keyframes progressFlow {
        0%, 100% { 
            box-shadow: 
                0 0 20px rgba(0, 180, 255, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        50% { 
            box-shadow: 
                0 0 40px rgba(0, 180, 255, 1),
                0 0 60px rgba(0, 180, 255, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }
    }
    
    /* ==================== ALERT PANELS ==================== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.15) 0%, rgba(0, 255, 157, 0.1) 100%) !important;
        border: 1px solid rgba(0, 255, 157, 0.5) !important;
        border-left: 4px solid #00ff9d !important;
        border-radius: 12px !important;
        color: #00ff9d !important;
        box-shadow: 0 0 25px rgba(0, 255, 157, 0.2) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 193, 7, 0.1) 100%) !important;
        border: 1px solid rgba(255, 193, 7, 0.5) !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 12px !important;
        color: #ffc107 !important;
        box-shadow: 0 0 25px rgba(255, 193, 7, 0.2) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.15) 0%, rgba(255, 82, 82, 0.1) 100%) !important;
        border: 1px solid rgba(255, 82, 82, 0.5) !important;
        border-left: 4px solid #ff5252 !important;
        border-radius: 12px !important;
        color: #ff5252 !important;
        box-shadow: 0 0 25px rgba(255, 82, 82, 0.2) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.15) 0%, rgba(0, 180, 255, 0.1) 100%) !important;
        border: 1px solid rgba(0, 180, 255, 0.5) !important;
        border-left: 4px solid #00b4ff !important;
        border-radius: 12px !important;
        color: #00d4ff !important;
        box-shadow: 0 0 25px rgba(0, 180, 255, 0.2) !important;
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: 2px solid rgba(0, 180, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 180, 255, 0.1);
        border-radius: 14px 14px 0 0;
        border: 1px solid rgba(0, 180, 255, 0.4);
        border-bottom: none;
        color: #00d4ff;
        padding: 1.1rem 2.5rem;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 255, 0.15);
        box-shadow: 0 -8px 25px rgba(0, 180, 255, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            180deg,
            rgba(0, 180, 255, 0.25) 0%,
            rgba(0, 180, 255, 0.1) 100%
        );
        color: #ffffff !important;
        border-color: rgba(0, 180, 255, 0.7);
        box-shadow: 
            0 -8px 30px rgba(0, 180, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 15%;
        right: 15%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        box-shadow: 0 0 15px rgba(0, 180, 255, 1);
        animation: tabUnderline 2s ease-in-out infinite;
    }
    
    @keyframes tabUnderline {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    /* ==================== DATAFRAMES ==================== */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(0, 180, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 180, 255, 0.15);
    }
    
    /* ==================== EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.12) 0%, rgba(0, 100, 200, 0.08) 100%) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        padding: 1.1rem 1.75rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 0.18) 0%, rgba(0, 100, 200, 0.12) 100%) !important;
        border-color: rgba(0, 180, 255, 0.6) !important;
        box-shadow: 0 0 25px rgba(0, 180, 255, 0.3) !important;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: rgba(0, 180, 255, 0.08);
        border: 2px dashed rgba(0, 180, 255, 0.4);
        border-radius: 14px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(0, 180, 255, 0.12);
        border-color: rgba(0, 180, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.25);
    }
    
    /* ==================== INPUT FIELDS ==================== */
    input, textarea, select {
        background: rgba(0, 180, 255, 0.08) !important;
        border: 1px solid rgba(0, 180, 255, 0.4) !important;
        border-radius: 12px !important;
        color: #00d4ff !important;
        padding: 0.85rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: rgba(0, 180, 255, 0.8) !important;
        box-shadow: 
            0 0 0 3px rgba(0, 180, 255, 0.25),
            0 0 25px rgba(0, 180, 255, 0.4) !important;
        background: rgba(0, 180, 255, 0.12) !important;
    }
    
    /* ==================== SLIDER ==================== */
    .stSlider [data-baseweb="slider"] {
        background: rgba(0, 180, 255, 0.25);
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #00d4ff, #0096ff);
        box-shadow: 0 0 20px rgba(0, 180, 255, 0.8);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.6), transparent) !important;
        margin: 3rem 0 !important;
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.4);
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 8, 20, 0.6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.6), rgba(0, 100, 200, 0.6));
        border-radius: 10px;
        border: 2px solid rgba(0, 8, 20, 0.6);
        box-shadow: 0 0 10px rgba(0, 180, 255, 0.4);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(0, 180, 255, 0.9), rgba(0, 100, 200, 0.9));
        box-shadow: 0 0 15px rgba(0, 180, 255, 0.7);
    }
    
    /* ==================== COLUMN SPACING ==================== */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render an enhanced JARVIS-style hero section"""
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem 0 2.5rem 0; position: relative;'>
        <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    width: 500px; height: 500px;
                    background: radial-gradient(circle, rgba(0, 180, 255, 0.15) 0%, transparent 70%);
                    filter: blur(60px); animation: heroGlow 4s ease-in-out infinite;'></div>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.75rem; letter-spacing: 3px; position: relative;'>{title}</h1>
        <p style='color: #00d4ff; font-size: 1.2rem; font-family: "Orbitron", sans-serif;
                  text-transform: uppercase; letter-spacing: 4px; position: relative;
                  text-shadow: 0 0 20px rgba(0, 180, 255, 0.8);'>{subtitle}</p>
        <div style='width: 200px; height: 3px; margin: 1.5rem auto;
                    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                    box-shadow: 0 0 20px rgba(0, 180, 255, 1);
                    animation: dividerPulse 3s ease-in-out infinite;'></div>
    </div>
    <style>
        @keyframes heroGlow {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        @keyframes dividerPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 180, 255, 1); }
            50% { box-shadow: 0 0 40px rgba(0, 180, 255, 1); }
        }
    </style>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, subtitle=""):
    """Render an enhanced holographic section header"""
    subtitle_html = f"<p style='color: #00d4ff; font-size: 0.95rem; margin-top: 0.75rem; opacity: 0.85;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0 2.5rem 0; position: relative;'>
        <div style='display: inline-block; position: relative;'>
            <h2 style='font-size: 2rem; margin: 0; letter-spacing: 2px;
                       font-family: "Orbitron", sans-serif;'>{icon} {title}</h2>
            <div style='position: absolute; bottom: -8px; left: 10%; right: 10%; height: 3px;
                        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
                        box-shadow: 0 0 15px rgba(0, 180, 255, 0.8);
                        animation: headerUnderline 3s ease-in-out infinite;'></div>
        </div>
        {subtitle_html}
    </div>
    <style>
        @keyframes headerUnderline {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)
