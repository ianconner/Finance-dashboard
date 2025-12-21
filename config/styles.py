# config/styles.py - Modern Futuristic Theme

import streamlit as st

def inject_custom_css():
    """Inject modern futuristic CSS theme"""
    st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(99, 110, 250, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 110, 250, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 110, 250, 0.3);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.2);
    }
    
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Headers with Gradient */
    h1, h2, h3 {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }
    
    /* Primary Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(88, 166, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(88, 166, 255, 0.6) !important;
        transform: translateY(-2px);
    }
    
    /* Secondary Buttons */
    button[kind="secondary"] {
        background: rgba(99, 110, 250, 0.1) !important;
        border: 1px solid rgba(99, 110, 250, 0.4) !important;
        color: #58a6ff !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(99, 110, 250, 0.2) !important;
        border-color: rgba(99, 110, 250, 0.6) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 50%, #f97583 100%);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.4);
    }
    
    /* Alert Messages */
    .stSuccess {
        background: rgba(46, 160, 67, 0.15) !important;
        border: 1px solid rgba(46, 160, 67, 0.4) !important;
        border-radius: 10px !important;
        color: #3fb950 !important;
        border-left: 4px solid #3fb950 !important;
    }
    
    .stWarning {
        background: rgba(187, 128, 9, 0.15) !important;
        border: 1px solid rgba(187, 128, 9, 0.4) !important;
        border-radius: 10px !important;
        color: #d29922 !important;
        border-left: 4px solid #d29922 !important;
    }
    
    .stError {
        background: rgba(248, 81, 73, 0.15) !important;
        border: 1px solid rgba(248, 81, 73, 0.4) !important;
        border-radius: 10px !important;
        color: #f85149 !important;
        border-left: 4px solid #f85149 !important;
    }
    
    .stInfo {
        background: rgba(88, 166, 255, 0.15) !important;
        border: 1px solid rgba(88, 166, 255, 0.4) !important;
        border-radius: 10px !important;
        color: #58a6ff !important;
        border-left: 4px solid #58a6ff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(99, 110, 250, 0.1);
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(99, 110, 250, 0.2);
        color: #8b949e;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 110, 250, 0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 110, 250, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
        color: #58a6ff !important;
        border-bottom: 3px solid #58a6ff;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(99, 110, 250, 0.3);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(99, 110, 250, 0.1) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(99, 110, 250, 0.3) !important;
        color: #58a6ff !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 110, 250, 0.15) !important;
        border-color: rgba(99, 110, 250, 0.5) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(99, 110, 250, 0.05);
        border: 2px dashed rgba(99, 110, 250, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(99, 110, 250, 0.1);
        border-color: rgba(99, 110, 250, 0.5);
    }
    
    /* Input Fields */
    input, textarea, select {
        background: rgba(99, 110, 250, 0.05) !important;
        border: 1px solid rgba(99, 110, 250, 0.3) !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        padding: 0.5rem !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: rgba(88, 166, 255, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background: rgba(99, 110, 250, 0.2);
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.4);
    }
    
    /* Dividers */
    hr {
        border-color: rgba(99, 110, 250, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(99, 110, 250, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 110, 250, 0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 110, 250, 0.5);
    }
    
    /* Column Container Spacing */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Glow Animation */
    @keyframes glow {
        from {
            box-shadow: 0 0 5px rgba(88, 166, 255, 0.3), 0 0 10px rgba(88, 166, 255, 0.2);
        }
        to {
            box-shadow: 0 0 15px rgba(88, 166, 255, 0.5), 0 0 25px rgba(88, 166, 255, 0.3);
        }
    }
    
    .glow-element {
        animation: glow 2s ease-in-out infinite alternate;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_section(title, subtitle):
    """Render a hero section with gradient styling"""
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>{title}</h1>
        <p style='color: #8b949e; font-size: 1.1rem;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, subtitle=""):
    """Render a section header with icon"""
    subtitle_html = f"<p style='color: #8b949e; font-size: 0.875rem; margin-top: 0.5rem;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h2 style='font-size: 1.5rem; margin: 0;'>{icon} {title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)
