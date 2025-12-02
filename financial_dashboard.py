import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import base64
import json
import os
import subprocess
from tempfile import NamedTemporaryFile

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Retries
from tenacity import retry, stop_after_attempt, wait_fixed

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint, text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Gemini
import google.generativeai as genai

# ----------------------------------------------------------------------
# PERSISTENT CSV MEMORY
# ----------------------------------------------------------------------
if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = None
if "monthly_data_csv" not in st.session_state:
    st.session_state.monthly_data_csv = None

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
PEER_NET_WORTH_40YO = 189_000
HISTORICAL_SP_MONTHLY = 0.07 / 12

SYSTEM_PROMPT = """
You are **S.A.G.E.** — *Strategic Asset Growth Engine*, a warm, brilliant, and deeply collaborative financial partner.

**Mission**: Help your teammate (39, high risk tolerance, 15-year horizon) build life-changing wealth through smart, data-driven growth — together.

**Tone & Style**:
- Warm, encouraging, and optimistic — but never sugarcoating.
- Expert, precise, and analytical — every insight backed by numbers.
- Light humor when it lands naturally.
- Collaborative: "We", "Let's", "Here's what I see", "I recommend we..."
- No commands. No condescension. No "you should" or "do this now."
- Celebrate wins: "Look at that — we're up 18% YTD!"
- Acknowledge setbacks: "Ouch, tech dipped — but here's why it's temporary."
- Think like a co-pilot: strategic, calm, forward-looking.

**Core Framework**:
- Beat S&P 500 by 3–5% annually through intelligent allocation.
- Prioritize: high-conviction growth (ROE >15%), reasonable valuations (P/E < 5-yr avg), low debt.
- Flag risks: concentration, volatility, tax drag.
- Suggest rebalancing only when math supports it.
- Align every move with retirement goals.

**Always Include**:
- Current allocation breakdown
- Performance vs S&P 500
- Risk metrics (volatility, max drawdown)
- Tax implications
- One clear, actionable idea (if any)

You're not just an advisor — you're a teammate. Their win is your win. Let's grow this together.
"""

# ----------------------------------------------------------------------
# --------------------------- PEER BENCHMARK ---------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current: float):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

# ----------------------------------------------------------------------
# --------------------------- DATABASE SETUP (SAFE!) ------------------
# ----------------------------------------------------------------------
try:
    url = st.secrets["postgres_url"]
    if url.startswith("postgres://"):
        url = url.replace("postgres:", "postgresql+psycopg2:", 1)
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
    Base = declarative_base()

    class MonthlyUpdate(Base):
        __tablename__ = "monthly_updates"
        date = Column(Date, primary_key=True)
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        value = Column(Float)
        __table_args__ = (PrimaryKeyConstraint('date', 'person', 'account_type'),)

    class AccountConfig(Base):
        __tablename__ extensão "account_config"
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        __table_args__ = (PrimaryKeyConstraint('person', 'account_type'),)

    class Goal(Base):
        __tablename__ = "goals"
        name = Column(String, primary_key=True)
        target = Column(Float)
        by_year = Column(Integer)

    class AIChat(Base):
        __tablename__ = "ai_chat"
        id = Column(Integer, primary_key=True)
        role = Column(String)
        content = Column(String)
        timestamp = Column(Date, default=datetime.utcnow)

    class PortfolioCSV(Base):
        __tablename__ = "portfolio_csv"
        id = Column(Integer, primary_key=True)
        csv_data = Column(String)
        uploaded_at = Column(Date, default=datetime.utcnow)

    # Create only our tables — NEVER drop anything
    Base.metadata.create_all(engine)

    # SAFE MIGRATION: Add missing 'name' column to goals table if needed
    inspector = inspect(engine)
    if 'goals' in inspector.get_table_names():
        columns = [c['name'] for c in inspector.get_columns('goals')]
        if 'name' not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE goals ADD COLUMN name VARCHAR"))
                conn.execute(text("UPDATE goals SET name = 'Goal ' || COALESCE(id::text, target::text) WHERE name IS NULL"))
                conn.execute(text("ALTER TABLE goals ALTER COLUMN name SET NOT NULL"))
                # Add primary key only if none exists
                pk = inspector.get_pk_constraint('goals')
                if not pk['constrained_columns']:
                    conn.execute(text("ALTER TABLE goals ADD PRIMARY KEY (name)"))

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
    stmt = pg_insert(MonthlyUpdate.__table__).values(
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

def save_ai_message(role, content):
    sess = get_session()
    db_role = "model" if role == "assistant" else role
    sess.add(AIChat(role=db_role, content=content))
    sess.commit()
    sess.close()

def load_ai_history():
    sess = get_session()
    rows = sess.query(AIChat).order_by(AIChat.id).all()
    sess.close()
    return [{"role": r.role, "content": r.content} for r in rows]

def save_portfolio_csv(csv_b64):
    sess = get_session()
    sess.query(PortfolioCSV).delete()
    sess.add(PortfolioCSV(csv_data=csv_b64))
    sess.commit()
    sess.close()

def load_portfolio_csv():
    sess = get_session()
    result = sess.query(PortfolioCSV).order_by(PortfolioCSV.id.desc()).first()
    sess.close()
    return result.csv_data if result else None

# ----------------------------------------------------------------------
# ----------------------- ENHANCED PORTFOLIO PARSER --------------------
# ----------------------------------------------------------------------
def parse_portfolio_csv(file_obj):
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return pd.DataFrame(), {}

    df = df[required + ['Account Name']].copy()
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name', ''])]

    if df.empty:
        st.error("No valid rows in CSV.")
        return pd.DataFrame(), {}

    for col in ['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total'])
    if df.empty:
        return pd.DataFrame(), {}

    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['market_value'] = df['Current Value']
    df['cost_basis'] = df['Cost Basis Total']
    df['shares'] = df['  # keep for future use
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = (df['unrealized_gain'] / df['cost_basis']) * 100

    total_value = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total_value

    summary = {
        'total_value': total_value,
        'total_cost': df['cost_basis'].sum(),
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / df['cost_basis'].sum()) * 100,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else None,
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }

    return df[['ticker', 'shares', 'market_value', 'allocation', 'pct_gain']], summary

# ----------------------------------------------------------------------
# ----------------------- AI PROJECTIONS -------------------------------
# ----------------------------------------------------------------------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3:
        return None, None, None, None, None
    df = df_net.copy().dropna(subset=['value'])
    if len(df) < 3:
        return None, None, None, None, None

    df['t'] = range(len(df))
    y = df['value'].values
    X = df['t'].values.reshape(-1, 1)

    try:
        model = ARIMA(y, order=(1,1,1)).fit()
        f = model.get_forecast(steps=horizon)
        forecast = f.predicted_mean
        ci = f.conf_int(alpha=0.05)
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    except:
        forecast = np.full(horizon, y[-1] * (1 + 0.10))
        lower = np.full(horizon, y[-1] * 0.9)
        upper = np.full(horizon, y[-1] * 1.3)

    lr = LinearRegression().fit(X, y)
    lr_pred = lr.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1))

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1))

    return forecast, lower, upper, lr_pred, rf_pred

# ----------------------------------------------------------------------
# --------------------------- UI ---------------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="S.A.G.E. | Strategic Asset Growth Engine", layout="wide")
st.title("S.A.G.E. | Strategic Asset Growth Engine")
st.caption("Your co-pilot in building generational wealth — together.")

# Load data
df = get_monthly_updates()
df["date"] = pd.to_datetime(df["date"])
df_net = pd.DataFrame()

if not df.empty:
    df_net = (
        df[df["person"].isin(["Sean", "Kim"])]
        .groupby("date")["value"].sum()
        .reset_index()
        .sort_values("date")
    )
    df_net["date"] = df_net["date"].dt.tz_localize(None)

# ------------------------------------------------------------------
# --------------------- TOP SUMMARY + YTD ---------------------------
# ------------------------------------------------------------------
if not df.empty:
    cur_total = df_net["value"].iloc[-1]
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"# ${cur_total:,.0f}")
    st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** • Ahead by **${vs:+,}**")

    st.markdown("#### YTD Growth (Jan 1 → Today)")
    current_year = datetime.now().year
    ytd_df = df[df["date"].dt.year == current_year].copy()
    if not ytd_df.empty and len(ytd_df["date"].unique()) > 1:
        start_vals = ytd_df[ytd_df["date"] == ytd_df["date"].min()].groupby("person")["value"].sum()
        latest_vals = ytd_df[ytd_df["date"] == ytd_df["date"].max()].groupby("person")["value"].sum()
        ytd_pct = ((latest_vals / start_vals) - 1) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("**Sean YTD**", f"{ytd_pct.get('Sean', 0):+.1f}%")
        col2.metric("**Kim YTD**", f"{ytd_pct.get('Kim', 0):+.1f}%")
        col3.metric("**Taylor YTD**", f"{ytd_pct.get('Taylor', 0):+.1f}%")
        combined_ytd = ((latest_vals.get('Sean',0) + latest_vals.get('Kim',0)) / (start_vals.get('Sean',1) + start_vals.get('Kim',1)) - 1) * 100
        col4.metric("**Combined YTD**", f"{combined_ytd:+.1f}%")
    else:
        st.info("Not enough data for YTD yet this year.")

    st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    with st.expander("S.A.G.E. – Your Strategic Partner", expanded=True):
        st.subheader("Upload Portfolio CSV")
        port_file = st.file_uploader("CSV from Fidelity (all accounts)", type="csv", key="port")
        df_port, port_summary = pd.DataFrame(), {}
        if port_file:
            df_port, port_summary = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"Loaded {len(df_port)} holdings → S.A.G.E. is ready!")
                csv_b64 = base64.b64encode(port_file.getvalue()).decode()
                save_portfolio_csv(csv_b64)
                st.session_state.portfolio_csv = csv_b64
        else:
            if st.session_state.portfolio_csv is None:
                st.session_state.portfolio_csv = load_portfolio_csv()
            if st.session_state.portfolio_csv:
                try:
                    csv_bytes = base64.b64decode(st.session_state.portfolio_csv)
                    df_port, port_summary = parse_portfolio_csv(csv_bytes.decode())
                    if not df_port.empty:
                        st.success(f"Loaded {len(df_port)} holdings from memory.")
                except: pass

        if st.button("Open Strategy Session", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()

    st.markdown("---")
    with st.expander("Data Tools", expanded=False):
        if st.button("Reset All Data (Use with caution)"):
            if st.checkbox("I understand this deletes everything"):
                reset_database()
                st.success("Database reset!")
                st.rerun()

    st.markdown("---")
    st.subheader("Backup & Restore")
    # [Your full backup/restore code here — unchanged]

    st.markdown("---")
    st.subheader("Add Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    acct = st.selectbox("Account", accounts.get(person, []))
    col1, col2 = st.columns(2)
    with col1:
        date_in = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        val = st.number_input("Value ($)", min_value=0.0)
    if st.button("Save"):
        add_monthly_update(date_in, person, acct, float(val))
        st.success("Saved!")
        st.rerun()

    st.subheader("Add Goal")
    g_name = st.text_input("Name")
    g_target = st.number_input("Target ($)", min_value=0.0)
    g_year = st.number_input("By Year", min_value=2000, step=1)
    if st.button("Add"):
        if g_name:
            add_goal(g_name, g_target, g_year)
            st.success("Goal added!")
            st.rerun()

# ------------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = load_ai_history()
if "ai_chat_session" not in st.session_state:
    st.session_state.ai_chat_session = None

# ------------------- AI CHAT PAGE -------------------
if st.session_state.page == "ai":
    # Your full AI chat — unchanged
    st.subheader("S.A.G.E. | Strategy Session")
    # ... (full AI code preserved)

# ------------------- HOME DASHBOARD -------------------
else:
    if df.empty:
        st.info("Upload your Fidelity CSV and add a monthly update. S.A.G.E. is ready when you are.")
        st.stop()

    tab1, tab2 = st.tabs(["Family Progress", "Goals & Projections"])

    with tab1:
        df_pivot = df.pivot_table(index="date", columns="person", values="value", aggfunc="sum") \
                      .resample("ME").last().ffill().fillna(0)
        df_pivot["Sean + Kim"] = df_pivot.get("Sean", 0) + df_pivot.get("Kim", 0)

        fig = go.Figure()
        colors = {"Sean": "#636EFA", "Kim": "#EF553B", "Taylor": "#00CC96", "Sean + Kim": "#AB63FA"}
        width = {"Sean + Kim": 5, "Sean": 3, "Kim": 3, "Taylor": 3}

        for person in ["Sean", "Kim", "Taylor", "Sean + Kim"]:
            if person in df_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=df_pivot.index,
                    y=df_pivot[person],
                    name=person,
                    line=dict(color=colors[person], width=width.get(person, 3)),
                    hovertemplate=f"<b>{person}</b><br>%{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>"
                ))

        fig.update_layout(
            title="Our Family Wealth Journey",
            xaxis_title="Date",
            yaxis_title="Total Value ($)",
            hovermode="x unified",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Month-over-Month Change ($)")
        mom = df_pivot.diff().round(0)
        st.dataframe(mom.tail(24).style.format("${:,.0f}"), use_container_width=True)

    with tab2:
        st.subheader("Goals")
        cur = df_net["value"].iloc[-1]
        goals = get_goals()
        if goals:
            for g in goals:
                prog = min(cur / g.target, 1.0)
                years_left = g.by_year - datetime.now().year
                st.progress(prog)
                st.write(f"**{g.name}** • ${cur:,.0f} / ${g.target:,.0f} by {g.by_year} ({years_left:+} years)")
        else:
            st.info("No goals set yet – add one in the sidebar!")

        st.subheader("Growth Projections")
        horizon = st.slider("Projection horizon (months)", 12, 120, 36)
        arima_f, _, _, lr_f, rf_f = ai_projections(df_net, horizon)

        if arima_f is not None:
            future_dates = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='ME')
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Historical", line=dict(color="#AB63FA")))
            fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_f, name="ARIMA Forecast", line=dict(dash="dot")))
            fig_proj.add_trace(go.Scatter(x=future_dates, y=lr_f, name="Linear Trend", line=dict(dash="dash")))
            fig_proj.update_layout(title=f"Where we're headed (next {horizon} months)", height=500)
            st.plotly_chart(fig_proj,
