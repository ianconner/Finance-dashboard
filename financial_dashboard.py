import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Retries
from tenacity import retry, stop_after_attempt, wait_fixed

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint, insert
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Gemini
import google.generativeai as genai

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
PEER_NET_WORTH_40YO = 189_000
HISTORICAL_SP_MONTHLY = 0.07 / 12

# ----------------------------------------------------------------------
# --------------------------- DATABASE SETUP ---------------------------
# ----------------------------------------------------------------------
try:
    url = st.secrets["postgres_url"]
    if url.startswith("postgres://"):
        url = url.replace("postgres:", "postgresql+psycopg2:", 1)
    engine = create_engine(url)
    Base = declarative_base()

    class MonthlyUpdate(Base):
        __tablename__ = "monthly_updates"
        date = Column(Date, primary_key=True)
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        value = Column(Float)
        __table_args__ = (PrimaryKeyConstraint('date', 'person', 'account_type'),)

    class AccountConfig(Base):
        __tablename__ = "account_config"
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        __table_args__ = (PrimaryKeyConstraint('person', 'account_type'),)

    class Goal(Base):
        __tablename__ = "goals"
        name = Column(String, primary_key=True)
        target = Column(Float)
        by_year = Column(Integer)

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# ----------------------------------------------------------------------
# --------------------------- DB HELPERS -------------------------------
# ----------------------------------------------------------------------
def get_session():
    return Session()

def reset_database():
    sess = get_session()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    defaults = {
        'Sean': ['IRA', 'Roth IRA', 'TSP', 'Personal', 'T3W'],
        'Kim': ['Retirement'],
        'Taylor': ['Personal']
    }
    for p, types in defaults.items():
        for t in types:
            sess.merge(AccountConfig(person=p, account_type=t))
    sess.commit()
    sess.close()

def load_accounts():
    sess = get_session()
    cfg = sess.query(AccountConfig).all()
    accounts = {}
    for row in cfg:
        accounts.setdefault(row.person, []).append(row.account_type)
    sess.close()
    if not accounts:
        reset_database()
        return load_accounts()
    return accounts

def add_monthly_update(date, person, acc_type, value):
    sess = get_session()
    stmt = insert(MonthlyUpdate.__table__).values(
        date=date, person=person, account_type=acc_type, value=value
    ).on_conflict_do_update(
        index_elements=['date', 'person', 'account_type'],
        set_={'value': value}
    )
    sess.execute(stmt)
    sess.commit()
    sess.close()

def get_monthly_updates():
    sess = get_session()
    rows = sess.query(MonthlyUpdate).all()
    sess.close()
    return pd.DataFrame([
        {'date': r.date, 'person': r.person,
         'account_type': r.account_type, 'value': r.value}
        for r in rows
    ])

def get_goals():
    sess = get_session()
    goals = sess.query(Goal).all()
    sess.close()
    return goals

def add_goal(name, target, by_year):
    sess = get_session()
    sess.merge(Goal(name=name, target=target, by_year=by_year))
    sess.commit()
    sess.close()

# ----------------------------------------------------------------------
# ----------------------- CSV → PORTFOLIO SUMMARY ----------------------
# ----------------------------------------------------------------------
def parse_fidelity_csv(uploaded_file) -> pd.DataFrame:
    """Return a clean DataFrame with ticker, allocation % and market_value."""
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    df = pd.read_csv(uploaded_file)

    # Basic sanity
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name'])]

    if df.empty:
        st.error("No valid rows found in CSV.")
        return pd.DataFrame()

    # Clean numbers
    for col in ['Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)

    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['shares'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['Last Price'], errors='coerce')
    df['market_value'] = pd.to_numeric(df['Current Value'], errors='coerce')
    df['cost_basis'] = pd.to_numeric(df['Average Cost Basis'], errors='coerce')

    df = df.dropna(subset=['shares', 'price', 'market_value', 'cost_basis'])
    if df.empty:
        st.error("No numeric data after cleaning.")
        return pd.DataFrame()

    total = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total * 100
    return df[['ticker', 'allocation', 'market_value']]

# ----------------------------------------------------------------------
# ----------------------- AI REBALANCE CHAT ----------------------------
# ----------------------------------------------------------------------
def get_ai_response(model, prompt):
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"AI error: {str(e)}"

# ----------------------------------------------------------------------
# --------------------------- UI ---------------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

# Load net-worth data
df = get_monthly_updates()
df_net = pd.DataFrame(columns=['date', 'value'])
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df_net = (
        df[df["person"].isin(["Sean", "Kim"])]
        .groupby("date")["value"].sum()
        .reset_index()
        .sort_values("date")
    )
    df_net["date"] = df_net["date"].dt.tz_localize(None)

# ------------------------------------------------------------------
# SIDEBAR – ONLY CSV + AI BUTTON
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Upload Fidelity CSV")
    port_file = st.file_uploader("CSV file", type="csv", key="port")
    df_port = pd.DataFrame()
    if port_file:
        df_port = parse_fidelity_csv(port_file)
        if not df_port.empty:
            st.success(f"Parsed {len(df_port)} holdings – ready for AI!")
        else:
            st.warning("CSV loaded but no valid data.")

    st.subheader("AI Rebalance Advisor")
    if st.button("Ask AI Advisor"):
        if df_port.empty:
            st.error("Upload a CSV first!")
        else:
            st.session_state.page = "ai"
            st.rerun()

    # (Optional) keep the other admin buttons if you still want them
    if st.button("Reset DB (Admin)"):
        reset_database()
        st.rerun()

# ------------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- AI CHAT PAGE -------------------
if st.session_state.page == "ai":
    st.subheader("AI Rebalance Chat")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("GOOGLE_API_KEY missing – add it in Secrets.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            st.error(f"Cannot load Gemini: {e}")
            st.stop()

        # ---- INITIAL ADVICE ----
        if not st.session_state.messages:
            current = df_net['value'].iloc[-1] if not df_net.empty else 0
            portfolio_json = df_port[['ticker', 'allocation']].round(1).to_dict('records')
            init_prompt = (
                f"Net worth: ${current:,.0f}. "
                f"Portfolio: {portfolio_json}. "
                "Suggest 1-2 rebalance moves to maximize returns. "
                "Keep it fun, bold, and use emojis."
            )
            with st.spinner("AI is thinking..."):
                init_reply = get_ai_response(model, init_prompt)
            st.session_state.messages.append({"role": "assistant", "content": init_reply})

        # ---- DISPLAY CHAT ----
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ---- USER INPUT ----
        user_input = st.chat_input("Ask anything – change risk, why sell X, etc.")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            follow_prompt = f"Conversation history:\n{history}\nRespond helpfully, keep fun/bold/emojis."
            with st.spinner("AI is replying..."):
                reply = get_ai_response(model, follow_prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    if st.button("Back to Dashboard"):
        st.session_state.page = "home"
        st.session_state.messages = []   # clear chat
        st.rerun()

# ------------------- HOME PAGE (net-worth, goals, etc.) -------------------
else:
    if not df.empty:
        # Monthly Summary
        st.subheader("Monthly Summary (by Year)")
        df['year'] = df['date'].dt.year
        for yr in sorted(df['year'].unique(), reverse=True):
            with st.expander(f"{yr} – Click to Expand"):
                ydf = df[df['year'] == yr]
                piv = ydf.pivot_table(index="date", columns=["person", "account_type"],
                                      values="value", fill_value=0)
                st.dataframe(piv.style.format("${:,.0f}"))

        # Net Worth line
        st.subheader("Family Net Worth")
        fig = px.line(df_net, x="date", y="value", title="Net Worth")
        st.plotly_chart(fig, use_container_width=True)

        # Goals
        st.subheader("Goals")
        cur = df_net["value"].iloc[-1]
        for g in get_goals():
            prog = min(cur / g.target, 1.0)
            st.progress(prog)
            st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f}")

        # Export
        st.download_button("Export Monthly Data", df.to_csv(index=False).encode(), "monthly_data.csv")

    else:
        st.info("Add your first monthly update (or just upload a CSV and use the AI).")
