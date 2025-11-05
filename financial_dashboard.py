import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import openai
import random

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint, insert
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
HISTORICAL_SP_ANNUAL_REAL = 0.07
HISTORICAL_SP_MONTHLY = HISTORICAL_SP_ANNUAL_REAL / 12
VOLATILITY_STD = 0.04
PEER_NET_WORTH_40YO = 150_000  # BLS median for 35-44

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

def update_goal(name, target, by_year):
    sess = get_session()
    goal = sess.query(Goal).filter_by(name=name).first()
    if goal:
        goal.target = target
        goal.by_year = by_year
        sess.commit()
    sess.close()

def delete_goal(name):
    sess = get_session()
    sess.query(Goal).filter_by(name=name).delete()
    sess.commit()
    sess.close()

# ----------------------------------------------------------------------
# ----------------------- YFINANCE HELPER ------------------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_ticker(ticker, period="1d", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': 'price'})
    except Exception as e:
        st.warning(f"Failed to fetch {ticker}: {e}")
    return None

# ----------------------------------------------------------------------
# ----------------------- PORTFOLIO ANALYZER ---------------------------
# ----------------------------------------------------------------------
def analyze_portfolio(df_port):
    if df_port.empty or 'Symbol' not in df_port.columns:
        return None, None, None

    results = []
    total_value = 0.0

    for _, row in df_port.iterrows():
        ticker = str(row['Symbol']).strip().upper()
        shares = float(row.get('Quantity', 0))
        cost_basis = float(row.get('Average Cost Basis', 0))
        if shares <= 0:
            continue

        price_df = fetch_ticker(ticker, period="1d")
        if price_df is None:
            st.warning(f"Price not available for {ticker}")
            continue
        price = price_df['price'].iloc[-1]

        market = shares * price
        total_value += market
        gain = market - (shares * cost_basis)

        results.append({
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'market_value': market,
            'allocation': 0.0,
            'gain': gain,
            '6mo_return': 0.0
        })

    if total_value == 0:
        return None, None, None

    out = pd.DataFrame(results)
    out['allocation'] = out['market_value'] / total_value * 100

    # 6-month return
    for i, row in out.iterrows():
        hist = fetch_ticker(row['ticker'], period="6mo")
        if hist is not None and len(hist) > 1:
            out.at[i, '6mo_return'] = (hist['price'].iloc[-1] /
                                      hist['price'].iloc[0] - 1) * 100

    # Health score
    std_alloc = out['allocation'].std()
    health = max(0, min(100, 100 - std_alloc * 8))

    # Recommendations
    recs = []
    over = out[out['allocation'] > 25]['ticker'].tolist()
    if over:
        recs.append(f"Overweight: {', '.join(over)} — consider trimming.")
    if not out.empty:
        top = out.loc[out['6mo_return'].idxmax()]
        recs.append(f"Hot pick: {top['ticker']} (+{top['6mo_return']:.1f}%) — add 5%?")

    return out, health, recs

# ----------------------------------------------------------------------
# ----------------------- PEER BENCHMARK -------------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current):
    vs_peer = current - PEER_NET_WORTH_40YO
    percentile = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return percentile, vs_peer

# ----------------------------------------------------------------------
# ----------------------- AI PROJECTIONS -------------------------------
# ----------------------------------------------------------------------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3:
        return None, None, None, None, None
    df_net = df_net.copy().dropna(subset=['value'])
    if len(df_net) < 3:
        return None, None, None, None, None

    df_net['time_idx'] = range(len(df_net))
    y = df_net['value'].values
    X = df_net['time_idx'].values.reshape(-1, 1)

    try:
        model = ARIMA(y, order=(1,1,0))
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        ci = forecast_result.conf_int(alpha=0.05)
        lower, upper = ci[:, 0], ci[:, 1]
        forecast = np.array(forecast) * 0.95
        lower *= 0.95
        upper *= 0.95
    except Exception:
        forecast = np.full(horizon, y[-1] * 1.05)
        lower = np.full(horizon, y[-1] * 0.95)
        upper = np.full(horizon, y[-1] * 1.05)

    lr = LinearRegression().fit(X, y)
    future_x = np.array(range(len(df_net), len(df_net) + horizon)).reshape(-1, 1)
    lr_pred = lr.predict(future_x) * 0.95

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(future_x) * 0.95

    return forecast, lower, upper, lr_pred, rf_pred

# ----------------------------------------------------------------------
# --------------------------- UI START ---------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

# Load data
df = get_monthly_updates()

# EARLY df_net (prevents NameError)
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

# ONE-TIME SEED
if df.empty:
    st.subheader("Seed Database with CSV (One-Time)")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        try:
            up = pd.read_csv(uploaded)
            if all(c in up.columns for c in ['date', 'person', 'account_type', 'value']):
                if st.button("Import CSV"):
                    for _, r in up.iterrows():
                        d = pd.to_datetime(r['date'], errors='coerce').date()
                        if pd.isna(d): continue
                        add_monthly_update(d, str(r['person']), str(r['account_type']), float(r['value']))
                    st.success("Seeded!")
                    st.rerun()
            else:
                st.error("CSV needs: date, person, account_type, value")
        except Exception as e:
            st.error(f"CSV error: {e}")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Add Monthly Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    acct = st.selectbox("Account", accounts.get(person, []))
    col1, col2 = st.columns(2)
    with col1:
        date_in = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        val = st.number_input("Value ($)", min_value=0.0, format="%.2f")
    if st.button("Save"):
        add_monthly_update(date_in, person, acct, float(val))
        st.success("Saved!")
        st.rerun()

    st.subheader("Add Goal")
    g_name = st.text_input("Goal name")
    g_target = st.number_input("Target ($)", min_value=0.0)
    g_year = st.number_input("By year", min_value=2000, step=1)
    if st.button("Add Goal"):
        if g_name:
            add_goal(g_name, g_target, g_year)
            st.success("Goal added!")
            st.rerun()

    if st.button("Reset DB (Admin)"):
        reset_database()
        st.rerun()

    # --- Portfolio Analyzer ---
    st.subheader("1. Portfolio Analyzer")
    port_file = st.file_uploader("Upload Fidelity CSV", type="csv", key="port")
    df_port = pd.DataFrame()
    if port_file:
        try:
            df_port = pd.read_csv(port_file)
            if all(c in df_port.columns for c in ['Symbol', 'Quantity', 'Average Cost Basis']):
                df_res, health, recs = analyze_portfolio(df_port)
                if df_res is not None:
                    st.dataframe(
                        df_res[['ticker', 'allocation', '6mo_return']]
                        .style.format({'allocation': '{:.1f}%', '6mo_return': '{:.1f}%'})
                    )
                    st.metric("Health Score", f"{health:.0f}/100")
                    for r in recs:
                        st.info(r)
                else:
                    st.error("No valid holdings.")
            else:
                st.error("CSV must contain: Symbol, Quantity, Average Cost Basis")
        except Exception as e:
            st.error(f"CSV error: {e}")

    # --- Peer Benchmark ---
    st.subheader("9. Peer Benchmark")
    if not df_net.empty:
        cur = df_net["value"].iloc[-1]
        pct, vs = peer_benchmark(cur)
        st.metric("vs. Avg 40yo", f"Top {100-int(pct)}%", delta=f"{vs:+,}")
        if pct > 80:
            st.balloons()
    else:
        st.info("Enter data to see peer rank.")

# ------------------------------------------------------------------
# MAIN CONTENT (only if data exists)
# ------------------------------------------------------------------
if not df.empty:
    # --- Monthly Summary (by Year) - ONLY ONCE ---
    st.subheader("Monthly Summary (by Year)")
    df['year'] = df['date'].dt.year
    for yr in sorted(df['year'].unique(), reverse=True):
        with st.expander(f"{yr} – Click to Expand"):
            ydf = df[df['year'] == yr]
            piv = ydf.pivot_table(
                index="date",
                columns=["person", "account_type"],
                values="value",
                aggfunc="sum",
                fill_value=0
            )
            st.dataframe(piv.style.format("${:,.0f}"))

    # --- Net Worth Chart ---
    st.subheader("Family Net Worth")
    fig = px.line(df_net, x="date", y="value", title="Family Net Worth",
                  labels={"value": "Total ($)"})
    max_val = df_net['value'].max()
    fig.update_yaxes(range=[0, np.ceil(max_val/50_000)*50_000], tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # --- ROR ---
    if len(df_net) >= 2:
        periods = len(df_net) / 12
        ann_ror = (df_net['value'].iloc[-1] / df_net['value'].iloc[0]) ** (1/periods) - 1
        st.metric("Annualized ROR", f"{ann_ror*100:.2f}%")

    # --- AI Projections ---
    st.subheader("AI Growth Projections")
    horizon = st.slider("Months Ahead", 12, 60, 24)
    arima_f, ar_lower, ar_upper, lr_f, rf_f = ai_projections(df_net, horizon)
    if arima_f is not None:
        future_dates = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1),
                                     periods=horizon, freq='ME')
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"],
                                      name="Historical", line=dict(color="blue")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_f,
                                      name="ARIMA", line=dict(color="green")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=ar_lower, fill=None,
                                      line=dict(color="lightgreen", dash="dash"),
                                      showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=ar_upper,
                                      fill='tonexty', line=dict(color="lightgreen"),
                                      name="ARIMA CI"))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=lr_f,
                                      name="Linear", line=dict(color="orange")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=rf_f,
                                      name="Random Forest", line=dict(color="red")))
        fig_proj.update_layout(title=f"Projections ({horizon} months)", yaxis_title="$")
        st.plotly_chart(fig_proj, use_container_width=True)

    # --- Goals ---
    st.subheader("Financial Goals")
    cur = df_net["value"].iloc[-1]
    for g in get_goals():
        prog = min(cur / g.target, 1.0)
        st.progress(prog)
        st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f} → {g.by_year}")

    with st.expander("Edit/Delete Goals"):
        for g in get_goals():
            with st.expander(f"Edit {g.name}"):
                t = st.number_input("Target", value=g.target, key=f"t_{g.name}")
                y = st.number_input("Year", value=g.by_year, key=f"y_{g.name}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Update", key=f"u_{g.name}"):
                        update_goal(g.name, t, y)
                        st.rerun()
                with c2:
                    if st.button("Delete", key=f"d_{g.name}"):
                        delete_goal(g.name)
                        st.rerun()

    # --- Export ---
    st.download_button("Export Monthly Values", df.to_csv(index=False).encode(),
                       "monthly_values.csv", "text/csv")

else:
    st.info("Add your first monthly update to get started!")
