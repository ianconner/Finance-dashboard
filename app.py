# app.py - Main Streamlit entry point

import streamlit as st
import pandas as pd
from config.constants import peer_benchmark
from database.connection import DB_AVAILABLE, DB_ERROR, engine
from database.operations import (
    get_monthly_updates, get_retirement_goal,
    save_portfolio_csv, load_portfolio_csv
)
from data.parser import parse_portfolio_csv
from pages.dashboard import show_dashboard
from pages.ai_chat_page import show_ai_chat_page

# ------------------------------------------------------------------
# Page config and DB check
# ------------------------------------------------------------------
st.set_page_config(
    page_title="S.A.G.E. | Strategic Asset Growth Engine",
    layout="wide"
)

if not DB_AVAILABLE:
    st.error(f"⚠️ Database connection failed: {DB_ERROR or 'Unknown error'}")
    st.stop()

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = load_portfolio_csv()

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = get_monthly_updates()

# Handle empty data gracefully
if df.empty or 'date' not in df.columns:
    df = pd.DataFrame(columns=['date', 'person', 'account_type', 'value'])
else:
    df["date"] = pd.to_datetime(df["date"])

# Net worth for Sean + Kim
df_net = pd.DataFrame()
if not df.empty:
    df_net = (
        df[df["person"].isin(["Sean", "Kim"])]
        .groupby("date")["value"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

# Portfolio CSV handling
df_port = pd.DataFrame()
port_summary = {}

if st.session_state.portfolio_csv:
    try:
        import base64
        csv_bytes = base64.b64decode(st.session_state.portfolio_csv)
        df_port, port_summary = parse_portfolio_csv(csv_bytes.decode('utf-8'))
    except Exception as e:
        st.error(f"Failed to load saved portfolio: {e}")

# ------------------------------------------------------------------
# Page routing
# ------------------------------------------------------------------
if st.session_state.page == "ai":
    show_ai_chat_page(df, df_net, df_port, port_summary)
else:
    show_dashboard(df, df_net, df_port, port_summary)
