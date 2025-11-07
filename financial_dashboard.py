import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Retries
from tenacity import retry, stop_after_attempt, wait_fixed

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date,
    PrimaryKeyConstraint, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

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

# ----------------------------------------------------------------------
# ----------------------- CSV → PORTFOLIO SUMMARY ----------------------
# ----------------------------------------------------------------------
def parse_fidelity_csv(uploaded_file) -> pd.DataFrame:
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    df = pd.read_csv(uploaded_file)
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
        st.error("No valid rows in CSV.")
        return pd.DataFrame()

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
# ----------------------- YFINANCE HELPERS -----------------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_ticker(ticker, period="5y"):
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': 'price'})
    except:
        pass
    return None

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
# ----------------------- PEER BENCHMARK -------------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

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
        model = ARIMA(y, order=(1,1,0)).fit()
        f = model.get_forecast(steps=horizon)
        forecast = f.predicted_mean * 0.95
        ci = f.conf_int(alpha=0.05)
        lower, upper = ci[:, 0] * 0.95, ci[:, 1] * 0.95
    except:
        forecast = np.full(horizon, y[-1] * 1.05)
        lower = np.full(horizon, y[-1] * 0.95)
        upper = np.full(horizon, y[-1] * 1.05)

    lr = LinearRegression().fit(X, y)
    lr_pred = lr.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1)) * 0.95

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1)) * 0.95

    return forecast, lower, upper, lr_pred, rf_pred

# ----------------------------------------------------------------------
# --------------------------- UI ---------------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

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
# --------------------- TOP SUMMARY (Peer + YTD) -----------------------
# ------------------------------------------------------------------
if not df.empty:
    cur_total = df_net["value"].iloc[-1]

    # Peer Benchmark
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"### **vs. Avg 40yo: Top {100-int(pct)}%** | Delta: **{vs:+,}**")

    # YTD Gain/Loss per Person
    st.markdown("#### **YTD Performance**")
    col1, col2, col3 = st.columns(3)
    for person, col in zip(["Sean", "Kim", "Taylor"], [col1, col2, col3]):
        person_df = df[df["person"] == person].copy()
        if not person_df.empty:
            person_df["date"] = pd.to_datetime(person_df["date"])
            person_df = person_df.sort_values("date")
            ytd_start = person_df[person_df["date"].dt.year == datetime.now().year]["value"].iloc[0]
            ytd_now = person_df["value"].iloc[-1]
            ytd_pct = (ytd_now / ytd_start - 1) * 100
            col.metric(f"**{person}'s YTD**", f"{ytd_pct:+.1f}%")
        else:
            col.metric(f"**{person}'s YTD**", "—")

    st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Upload Fidelity CSV (for AI)")
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

    st.subheader("Add Monthly Update")
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
    if st.button("Add Goal"):
        if g_name:
            add_goal(g_name, g_target, g_year)
            st.success("Added!")
            st.rerun()

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
            model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.error(f"Cannot load Gemini: {e}")
            st.stop()

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

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

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
        st.session_state.messages = []
        st.rerun()

# ------------------- HOME PAGE -------------------
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

        # Net Worth
        st.subheader("Family Net Worth")
        fig = px.line(df_net, x="date", y="value", title="Net Worth")
        st.plotly_chart(fig, use_container_width=True)

        # ROR vs S&P 500
        st.subheader("ROR vs S&P 500")
        df_net['ror'] = df_net['value'].pct_change() * 100
        df_ror = df_net.dropna(subset=['ror']).copy()
        if len(df_ror) >= 2:
            sp_data = fetch_ticker('^GSPC', period="5y")
            if sp_data is not None:
                sp_df = sp_data.reset_index()
                sp_df['Date'] = pd.to_datetime(sp_df['Date']).dt.tz_localize(None)
                sp_df['sp_ror'] = sp_df['price'].pct_change() * 100
                sp_df = sp_df.dropna(subset=['sp_ror'])
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

        # YTD & M2M
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

        # AI Projections
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

        # Goals
        st.subheader("Goals")
        cur = df_net["value"].iloc[-1]
        for g in get_goals():
            prog = min(cur / g.target, 1.0)
            st.progress(prog)
            st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f}")

        # Delete / Export
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
        st.info("Add your first monthly update or upload a CSV to use the AI.")
