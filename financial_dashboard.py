import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    PrimaryKeyConstraint, text
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

# S.A.G.E. – Strategic Asset Growth Engine
SYSTEM_PROMPT = """
You are **S.A.G.E.** — *Strategic Asset Growth Engine*, a warm, brilliant, and deeply collaborative financial partner.

**Mission**: Help your teammate (39, high risk tolerance, 15-year horizon) build life-changing wealth through smart, data-driven growth — together.

**Tone & Style**:
- Warm, encouraging, and optimistic — but never sugarcoating.
- Expert, precise, and analytical — every insight backed by numbers.
- Light humor when it lands naturally (a quip, not a routine).
- Collaborative: "We", "Let’s", "Here’s what I see", "I recommend we..."
- No commands. No condescension. No "you should" or "do this now."
- Celebrate wins: "Look at that — we’re up 18% YTD!"
- Acknowledge setbacks: "Ouch, tech dipped — but here’s why it’s temporary."
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

You're not just an advisor — you're a teammate. Their win is your win. Let’s grow this together.
"""

# ----------------------------------------------------------------------
# --------------------------- PEER BENCHMARK ---------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current: float):
    """
    Returns:
        pct (0-100) – how far above the average 40-year-old net-worth we are
        vs  – dollar difference (positive = ahead)
    """
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

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
# ----------------------- ENHANCED PORTFOLIO PARSER --------------------
# ----------------------------------------------------------------------
def parse_portfolio_csv(file_obj):
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Cost Basis Total', 'Average Cost Basis']
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

    for col in ['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total'])
    if df.empty:
        return pd.DataFrame(), {}

    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['market_value'] = df['Current Value']
    df['cost_basis'] = df['Cost Basis Total']
    df['shares'] = df['Quantity']
    df['price'] = df['Last Price']
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = (df['unrealized_gain'] / df['cost_basis']) * 100

    total_value = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total_value

    account_summary = df.groupby('Account Name')['market_value'].sum().to_dict()

    summary = {
        'total_value': total_value,
        'total_cost': df['cost_basis'].sum(),
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / df['cost_basis'].sum()) * 100,
        'account_breakdown': account_summary,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else None,
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }

    return df[['ticker', 'shares', 'price', 'market_value', 'cost_basis', 'unrealized_gain', 'pct_gain', 'allocation']], summary

# ----------------------------------------------------------------------
# ----------------------- YFINANCE + RISK METRICS ----------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_ticker(ticker: str, period: str = "5y"):
    try:
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=10,
        )
        if data.empty or "Close" not in data.columns:
            return None
        return data[["Close"]].rename(columns={"Close": "price"})
    except Exception as e:
        st.warning(f"yfinance failed for {ticker}: {e}")
        return None

def get_portfolio_metrics(df_port, df_net):
    if df_port.empty or df_net.empty:
        return {}

    tickers = df_port['ticker'].tolist()
    weights = df_port['allocation'].values

    returns = []
    for t in tickers:
        data = fetch_ticker(t, period="2y")
        if data is not None and len(data) > 60:
            monthly = data['price'].resample('M').last().pct_change().dropna()
            returns.append(monthly)

    if len(returns) < 2:
        return {}

    port_ret = pd.concat(returns, axis=1).mean(axis=1)
    port_vol = port_ret.std() * np.sqrt(12)
    port_sharpe = (port_ret.mean() * 12) / port_vol if port_vol > 0 else 0

    sp_data = fetch_ticker('^GSPC', '2y')
    sp_ret = sp_data['price'].resample('M').last().pct_change().dropna() if sp_data is not None else pd.Series()
    sp_vol = sp_ret.std() * np.sqrt(12)
    sp_sharpe = (sp_ret.mean() * 12) / sp_vol if sp_vol > 0 else 0

    return {
        'portfolio_vol': port_vol * 100,
        'sp500_vol': sp_vol * 100,
        'portfolio_sharpe': port_sharpe,
        'sp500_sharpe': sp_sharpe,
        'vs_sp_sharpe': port_sharpe - sp_sharpe
    }

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
st.caption("Your warm, expert co-pilot in building wealth — together.")

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
    df_net["date"] = df_net["date"].dt.tz_localization(None)

# ------------------------------------------------------------------
# --------------------- TOP SUMMARY (Peer + YTD) -------------------
# ------------------------------------------------------------------
if not df.empty:
    cur_total = df_net["value"].iloc[-1]
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** | Ahead by **${vs:+,}**")

    st.markdown("#### YTD Growth")
    col1, col2, col3 = st.columns(3)
    current_year = datetime.now().year
    for person, col in zip(["Sean", "Kim", "Taylor"], [col1, col2, col3]):
        person_df = df[df["person"] == person].copy()
        if not person_df.empty:
            person_df = person_df.sort_values("date")
            ytd_data = person_df[person_df["date"].dt.year == current_year]
            if len(ytd_data) > 1:
                start_val = ytd_data["value"].iloc[0]
                current_val = ytd_data["value"].iloc[-1]
                ytd_pct = (current_val / start_val - 1) * 100
                col.metric(f"**{person}'s YTD**", f"{ytd_pct:+.1f}%")
            else:
                col.metric(f"**{person}'s YTD**", "—")
        else:
            col.metric(f"**{person}'s YTD**", "—")

    st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR – S.A.G.E. AI + PERSISTENT CSV
# ------------------------------------------------------------------
with st.sidebar:
    with st.expander("S.A.G.E. – Your Strategic Partner", expanded=True):
        st.subheader("Upload Portfolio CSV")
        port_file = st.file_uploader(
            "CSV from Fidelity (all accounts)",
            type="csv",
            key="port",
            help="Saved across sessions"
        )
        df_port = pd.DataFrame()
        port_summary = {}

        if port_file:
            df_port, port_summary = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"Loaded {len(df_port)} holdings → S.A.G.E. is ready!")
                csv_b64 = base64.b64encode(port_file.getvalue()).decode()
                save_portfolio_csv(csv_b64)
                st.session_state.portfolio_csv = csv_b64
            else:
                st.warning("CSV loaded but no valid data.")
        else:
            if st.session_state.portfolio_csv is None:
                st.session_state.portfolio_csv = load_portfolio_csv()
            if st.session_state.portfolio_csv:
                try:
                    csv_bytes = base64.b64decode(st.session_state.portfolio_csv)
                    df_port, port_summary = parse_portfolio_csv(csv_bytes.decode())
                    if not df_port.empty:
                        st.success(f"Loaded {len(df_port)} holdings from memory.")
                except Exception as e:
                    st.error(f"Failed to load saved portfolio: {e}")

        st.subheader("Talk to S.A.G.E.")
        if st.button("Open Strategy Session", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()

    st.markdown("---")
    with st.expander("Data Tools", expanded=False):
        st.subheader("Bulk Import Monthly")
        monthly_file = st.file_uploader("CSV (date,person,account_type,value)", type="csv", key="monthly")
        if monthly_file:
            try:
                df_import = pd.read_csv(monthly_file)
                req = ['date', 'person', 'account_type', 'value']
                if all(c in df_import.columns for c in req):
                    df_import['date'] = pd.to_datetime(df_import['date']).dt.date
                    for _, r in df_import.iterrows():
                        add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value']))
                    st.success(f"Imported {len(df_import)} rows!")
                else:
                    st.error(f"Need: {req}")
            except Exception as e:
                st.error(f"Import error: {e}")

        if st.button("Reset Database"):
            if st.checkbox("I understand this deletes all data", key="confirm"):
                reset_database()
                sess = get_session()
                sess.query(PortfolioCSV).delete()
                sess.commit()
                sess.close()
                st.session_state.portfolio_csv = None
                st.success("Reset complete.")
                st.rerun()

    st.markdown("---")
    st.subheader("Add Update")
    accounts = load_db_accounts()
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

# ------------------- AI CHAT PAGE (S.A.G.E.) -------------------
if st.session_state.page == "ai":
    st.subheader("S.A.G.E. | Strategic Asset Growth Engine")
    st.caption("Let’s review, refine, and grow — together.")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("Add `GOOGLE_API_KEY` in Streamlit Secrets to enable S.A.G.E.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
            formatted_history = [
                {"role": m["role"], "parts": [m["content"]]} 
                for m in st.session_state.ai_messages 
                if isinstance(m, dict)
            ]
            chat = model.start_chat(history=formatted_history) if not st.session_state.ai_chat_session else st.session_state.ai_chat_session
            st.session_state.ai_chat_session = chat
        except Exception as e:
            st.error(f"AI init failed: {e}")
            st.stop()

        # Auto-init message
        if not st.session_state.ai_messages and not df_port.empty:
            metrics = get_portfolio_metrics(df_port, df_net)
            init_prompt = f"""
Net worth: ${df_net['value'].iloc[-1]:,.0f}
Portfolio value: ${port_summary.get('total_value', 0):,.0f}
YTD gain: {port_summary.get('total_gain_pct', 0):+.1f}%
Top holding: {port_summary.get('top_holding', 'N/A')} ({port_summary.get('top_allocation', 0):.1f}%)
Volatility: {metrics.get('portfolio_vol', 0):.1f}% | Sharpe: {metrics.get('portfolio_sharpe', 0):.2f}
S&P 500 Sharpe: {metrics.get('sp500_sharpe', 0):.2f}
Portfolio: {df_port[['ticker', 'allocation']].round(3).to_dict('records')}
            """.strip()

            with st.spinner("S.A.G.E. is analyzing your full picture..."):
                try:
                    response = chat.send_message(init_prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"AI error: {e}"

            st.session_state.ai_messages.append({"role": "user", "content": init_prompt})
            save_ai_message("user", init_prompt)
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply)
            st.rerun()

        # Display chat
        for msg in st.session_state.ai_messages:
            role = "assistant" if msg["role"] == "model" else "user"
            with st.chat_message(role):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask S.A.G.E.: rebalance? risk? taxes? retirement?")
        if user_input:
            st.session_state.ai_messages.append({"role": "user", "content": user_input})
            save_ai_message("user", user_input)
            with st.spinner("S.A.G.E. is thinking..."):
                try:
                    response = chat.send_message(user_input)
                    reply = response.text
                except Exception as e:
                    reply = f"AI error: {e}"
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply)
            st.rerun()

    if st.button("Clear Chat"):
        st.session_state.ai_messages = []
        st.session_state.ai_chat_session = None
        sess = get_session()
        sess.query(AIChat).delete()
        sess.commit()
        sess.close()
        st.success("Chat cleared.")
        st.rerun()

    if st.button("Back to Dashboard"):
        st.session_state.page = "home"
        st.rerun()

# ------------------- HOME DASHBOARD -------------------
else:
    if not df.empty:
        st.subheader("Net Worth Over Time")
        fig = px.line(df_net, x="date", y="value", markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Breakdown")
        if not df_port.empty:
            fig_pie = px.pie(df_port, names='ticker', values='market_value', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("ROR vs S&P 500")
        df_net['ror'] = df_net['value'].pct_change() * 100
        df_ror = df_net.dropna(subset=['ror']).copy()
        if len(df_ror) >= 2:
            sp_data = fetch_ticker('^GSPC', period="5y")
            if sp_data is not None and not sp_data.empty:
                sp_df = sp_data.reset_index()
                sp_df['Date'] = pd.to_datetime(sp_df['Date']).dt.tz_localize(None)
                sp_df['sp_ror'] = sp_df['price'].pct_change() * 100
                sp_df = sp_df.dropna(subset=['sp_ror'])
            else:
                st.info("Live S&P 500 data unavailable – using historic average.")
                df_ror['sp_ror'] = HISTORICAL_SP_MONTHLY * 100
                sp_df = pd.DataFrame({"price": [1.0, 1.0 * (1 + HISTORICAL_SP_MONTHLY * 12)]})

            df_ror = pd.merge_asof(df_ror[['date', 'ror']].sort_values('date'),
                                   sp_df[['Date', 'sp_ror']].sort_values('Date'),
                                   left_on='date', right_on='Date',
                                   direction='nearest', tolerance=pd.Timedelta('1M'))
            df_ror = df_ror.dropna(subset=['sp_ror'])

            if not df_ror.empty:
                fig_ror = go.Figure()
                fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['ror'], name='Personal'))
                fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'], name='S&P 500'))
                fig_ror.update_layout(title="Monthly ROR", barmode='group')
                st.plotly_chart(fig_ror, use_container_width=True)

                periods = len(df_net) / 12
                ann_p = (df_net['value'].iloc[-1] / df_net['value'].iloc[0]) ** (1/periods) - 1
                ann_s = (sp_df['price'].iloc[-1] / sp_df['price'].iloc[0]) ** (1/periods) - 1
                st.metric("Annualized Personal ROR", f"{ann_p*100:.2f}%")
                st.metric("Annualized S&P 500 ROR", f"{ann_s*100:.2f}%")
            else:
                st.info("Not enough overlapping data.")
        else:
            st.info("S&P 500 data unavailable – using static avg.")
            df_ror['sp_ror'] = HISTORICAL_SP_MONTHLY * 100
            fig_ror = go.Figure()
            fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['ror'], name='Personal'))
            fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'], name='S&P Avg'))
            fig_ror.update_layout(title="Monthly ROR (vs static avg)", barmode='group')
            st.plotly_chart(fig_ror, use_container_width=True)

        tab1, tab2 = st.tabs(["YTD", "M2M"])
        with tab1:
            df_ytd = df_net.copy()
            df_ytd['year'] = df_ytd['date'].dt.year
            fig_ytd = px.line(df_ytd, x="date", y="value", color="year")
            st.plotly_chart(fig_ytd, use_container_width=True)
        with tab2:
            df_m2m = df_net.copy()
            df_m2m['gain'] = df_m2m['value'].diff()
            df_m2m = df_m2m.dropna()
            fig_m2m = px.bar(df_m2m, x="date", y="gain")
            st.plotly_chart(fig_m2m, use_container_width=True)

        st.subheader("AI Growth Projections")
        horizon = st.slider("Months", 12, 60, 24)
        arima_f, ar_l, ar_u, lr_f, rf_f = ai_projections(df_net, horizon)
        if arima_f is not None:
            future = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='ME')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Historical"))
            fig.add_trace(go.Scatter(x=future, y=arima_f, name="ARIMA"))
            fig.add_trace(go.Scatter(x=future, y=lr_f, name="Linear"))
            fig.add_trace(go.Scatter(x=future, y=rf_f, name="RF"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Goals")
        cur = df_net["value"].iloc[-1]
        for g in get_goals():
            prog = min(cur / g.target, 1.0)
            st.progress(prog)
            st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f}")

        st.subheader("Delete Entry")
        choice = st.selectbox("Select", df.index,
                              format_func=lambda i: f"{df.loc[i,'date']} – ${df.loc[i,'value']:,.0f}")
        if st.button("Delete"):
            row = df.loc[choice]
            sess = get_session()
            sess.query(MonthlyUpdate).filter_by(date=row["date"], person=row["person"],
                                                account_type=row["account_type"]).delete()
            sess.commit()
            sess.close()
            st.rerun()

        st.download_button("Export Monthly Data", df.to_csv(index=False).encode(), "monthly_data.csv")

    else:
        st.info("Upload your Fidelity CSV and add a monthly update. S.A.G.E. is ready when you are.")
