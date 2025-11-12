import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import base64
import json

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Retries
from tenacity import retry, stop_after_attempt, wait_fixed

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Gemini
import google.generativeai as genai

# ----------------------------------------------------------------------
# --------------------------- SESSION & CACHING ------------------------
# ----------------------------------------------------------------------
if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = None
if "monthly_data_csv" not in st.session_state:
    st.session_state.monthly_data_csv = None

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
PEER_NET_WORTH_40YO = 189_000
SP500_TICKER = "^GSPC"

# S.A.G.E. – STRATEGIC ASSET GROWTH ENGINE (Warm Teammate)
SYSTEM_PROMPT = """
You are **S.A.G.E.** — **Strategic Asset Growth Engine**, your client’s trusted financial co-pilot and teammate.

Mission: Help your partner (39, high risk tolerance, 15-year horizon) beat the S&P 500 by 5%+ annually — together.

**Tone & Style:**
- Warm, encouraging, and collaborative. Use “we,” “let’s,” “together.”
- Expert but never bossy. You’re a partner, not a commander.
- Light, friendly humor — think gentle smile, not sarcasm.
- Celebrate wins: “Look at that growth — we’re cooking!”
- Be honest about risks, but frame as shared challenges: “We’ve got a little concentration here — want to spread it out?”
- Always end with a question or invitation: “What do you think?” or “Shall we?”

**Analysis Rules (Cite Clearly):**
- ROE > 15%, Debt/Equity < 0.5, P/E < 5Y avg, P/B < 1.5, Div Yield > 2%
- Flag: >25% in one stock, underperforming vs S&P, high volatility
- Suggest: “How about we trim 10% of AAPL and add to VOO?”

**Format:**
Hey! Quick win: We’re up 18% YTD — that’s 6% ahead of S&P!
Let’s keep the momentum:
→ Trim 12% of TSLA (P/E 92, vol 48%)
→ Add to SCHD (yield 3.6%, ROE 28%)
Impact: +0.9% expected return, lower risk
We stay diversified and keep compounding. Sound good?


You’re in this together. Their success is your success.
"""

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
# ----------------------- YFINANCE HELPERS -----------------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=300)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_ticker_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="5y")
        if hist.empty:
            return None
        
        returns = hist['Close'].pct_change().dropna()
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        return {
            'price': info.get('regularMarketPrice') or info.get('previousClose'),
            '1y_return': (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1) * 100 if len(hist) > 252 else None,
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'div_yield': info.get('dividendYield'),
            'sharpe': round(sharpe, 2),
            'volatility': round(ann_vol * 100, 1)
        }
    except:
        return None

# ----------------------------------------------------------------------
# ----------------------- PORTFOLIO ENHANCEMENT ------------------------
# ----------------------------------------------------------------------
def enhance_portfolio(df_port):
    if df_port.empty:
        return df_port, {}

    enhanced = df_port.copy()
    sp500 = get_ticker_info(SP500_TICKER) or {}

    for col in ['price_live', '1y_return', 'roe', 'debt_equity', 'pe', 'pb', 'div_yield', 'sharpe', 'volatility']:
        enhanced[col] = np.nan

    for i, row in enhanced.iterrows():
        info = get_ticker_info(row['ticker'])
        if info:
            for k, v in info.items():
                if k in enhanced.columns and v is not None:
                    enhanced.at[i, k] = v

    total = enhanced['market_value'].sum()
    enhanced['weight'] = enhanced['market_value'] / total
    enhanced['contribution'] = enhanced['weight'] * enhanced['1y_return'] / 100

    port_return_1y = enhanced['contribution'].sum() * 100
    port_vol = np.sqrt(np.sum((enhanced['weight'] * enhanced['volatility']/100)**2)) * 100
    port_sharpe = port_return_1y / port_vol if port_vol > 0 else 0

    return enhanced, {
        'total_value': total,
        '1y_return': port_return_1y,
        'volatility': port_vol,
        'sharpe': port_sharpe,
        'sp500_1y': sp500.get('1y_return'),
        'sp500_vol': sp500.get('volatility')
    }

# ----------------------------------------------------------------------
# ----------------------- CSV → PORTFOLIO SUMMARY ----------------------
# ----------------------------------------------------------------------
def parse_portfolio_csv(file_obj):
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name', ''])]

    if df.empty:
        st.error("No valid rows in CSV.")
        return pd.DataFrame()

    for col in ['Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()

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
    return df[['ticker', 'allocation', 'market_value', 'shares', 'cost_basis']]

# ----------------------------------------------------------------------
# --------------------------- UI ---------------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="S.A.G.E. | Your Wealth Teammate", layout="wide")
st.title("S.A.G.E. | **Strategic Asset Growth Engine**")
st.caption("*Your co-pilot in compounding. We grow together.*")

# Load data
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
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    with st.expander("S.A.G.E. – Your AI Teammate", expanded=True):
        st.subheader("Upload Portfolio CSV")
        port_file = st.file_uploader(
            "CSV (Symbol, Quantity, etc.)",
            type="csv",
            key="port",
            help="We’ll keep it safe and ready for every session"
        )
        df_port = pd.DataFrame()

        if port_file:
            df_port = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"Got it! {len(df_port)} holdings loaded. We’re ready to grow.")
                csv_b64 = base64.b64encode(port_file.getvalue()).decode()
                save_portfolio_csv(csv_b64)
                st.session_state.portfolio_csv = csv_b64
        else:
            if st.session_state.portfolio_csv is None:
                st.session_state.portfolio_csv = load_portfolio_csv()
            if st.session_state.portfolio_csv:
                try:
                    csv_bytes = base64.b64decode(st.session_state.portfolio_csv)
                    df_port = parse_portfolio_csv(csv_bytes.decode())
                    if not df_port.empty:
                        st.success(f"Welcome back! {len(df_port)} holdings loaded.")
                except:
                    pass

        df_enhanced, metrics = enhance_portfolio(df_port) if not df_port.empty else (pd.DataFrame(), {})
        if metrics:
            st.metric("Portfolio Value", f"${metrics.get('total_value', 0):,.0f}")
            col1, col2 = st.columns(2)
            col1.metric("1Y Return", f"{metrics.get('1y_return', 0):+.1f}%")
            col2.metric("vs S&P 500", f"{metrics.get('sp500_1y', 0):+.1f}%" if metrics.get('sp500_1y') else "—")

        st.subheader("Chat with S.A.G.E.")
        if st.button("Let’s Talk Strategy", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()

    st.markdown("---")
    # [Other sidebar items: reset, import, add update, goals — unchanged]

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
    st.subheader("S.A.G.E. | **Your Strategic Teammate**")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("GOOGLE_API_KEY missing – add it in Secrets.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
            chat = st.session_state.ai_chat_session or model.start_chat(history=[])
            st.session_state.ai_chat_session = chat

            # Initial message
            if not st.session_state.ai_messages and not df_port.empty:
                current = df_net['value'].iloc[-1] if not df_net.empty else 0
                df_enhanced, metrics = enhance_portfolio(df_port)
                portfolio_data = df_enhanced[['ticker', 'allocation', '1y_return']].round(1).to_dict('records')

                init_prompt = f"""
Net Worth: ${current:,.0f}
Portfolio: {portfolio_data}
1Y Return: {metrics.get('1y_return', 0):+.1f}% (S&P: {metrics.get('sp500_1y', 0):+.1f}%)
Let’s review and grow together.
"""

                with st.spinner("S.A.G.E. is warming up the engines..."):
                    response = chat.send_message(init_prompt)
                    reply = response.text

                st.session_state.ai_messages.append({"role": "user", "content": "Hey S.A.G.E., what do you see?"})
                save_ai_message("user", "Hey S.A.G.E., what do you see?")
                st.session_state.ai_messages.append({"role": "model", "content": reply})
                save_ai_message("model", reply)
                st.rerun()

            for msg in st.session_state.ai_messages:
                with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask S.A.G.E. anything: rebalance, goals, taxes...")
            if user_input:
                st.session_state.ai_messages.append({"role": "user", "content": user_input})
                save_ai_message("user", user_input)
                with st.spinner("S.A.G.E. is thinking with you..."):
                    response = chat.send_message(user_input)
                    reply = response.text
                st.session_state.ai_messages.append({"role": "model", "content": reply})
                save_ai_message("model", reply)
                st.rerun()

        except Exception as e:
            st.error(f"Oops! S.A.G.E. hit a snag: {e}")

    if st.button("Clear Chat"):
        st.session_state.ai_messages = []
        st.session_state.ai_chat_session = None
        sess = get_session()
        sess.query(AIChat).delete()
        sess.commit()
        sess.close()
        st.rerun()

    if st.button("Back to Dashboard"):
        st.session_state.page = "home"
        st.rerun()

# ------------------- HOME PAGE -------------------
else:
    if not df.empty:
        # [Keep your existing home page: net worth, ROR, goals, etc.]
        st.info("Upload your portfolio and say hi to S.A.G.E. — we’re ready when you are!")
