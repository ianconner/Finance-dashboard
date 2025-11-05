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
PEER_NET_WORTH_40YO = 150_000          # BLS median 35-44

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

    class Contribution(Base):
        __tablename__ = "contributions"
        date = Column(Date, primary_key=True)
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        contribution = Column(Float)
        __table_args__ = (PrimaryKeyConstraint('date', 'person', 'account_type'),)

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

def add_contribution(date, person, acc_type, amount):
    sess = get_session()
    stmt = insert(Contribution.__table__).values(
        date=date, person=person, account_type=acc_type, contribution=amount
    ).on_conflict_do_update(
        index_elements=['date', 'person', 'account_type'],
        set_={'contribution': amount}
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

def get_contributions():
    sess = get_session()
    rows = sess.query(Contribution).all()
    sess.close()
    return pd.DataFrame([
        {'date': r.date, 'person': r.person,
         'account_type': r.account_type, 'contribution': r.contribution}
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
    """Return DataFrame with 'price' column or None."""
    try:
        data = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': 'price'})
    except Exception as e:
        st.warning(f"yfinance error for {ticker}: {e}")
    return None

# ----------------------------------------------------------------------
# ----------------------- FEATURE 1: PORTFOLIO ANALYZER ----------------
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

    std_alloc = out['allocation'].std()
    health = max(0, min(100, 100 - std_alloc * 8))

    recs = []
    over = out[out['allocation'] > 25]['ticker'].tolist()
    if over:
        recs.append(f"Overweight: {', '.join(over)} — consider trimming.")
    if not out.empty:
        top = out.loc[out['6mo_return'].idxmax()]
        recs.append(f"Hot pick: {top['ticker']} (+{top['6mo_return']:.1f}%)")

    return out, health, recs

# ----------------------------------------------------------------------
# ----------------------- FEATURE 2: TREND ALERTS ----------------------
# ----------------------------------------------------------------------
SECTOR_ETFS = {
    "Tech": "XLK",
    "Health": "XLV",
    "Finance": "XLF",
    "Energy": "XLE",
    "Consumer": "XLY",
    "Industrials": "XLI",
    "AI": "BOTZ"
}
MEME_GIFS = [
    "https://media.giphy.com/media/l0MYC0LdjMMD9R3n2/giphy.gif",
    "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif"
]

def get_trend_alerts():
    alerts = []
    for name, ticker in SECTOR_ETFS.items():
        hist = fetch_ticker(ticker, period="1mo")
        if hist is not None and len(hist) > 1:
            ret = (hist['price'].iloc[-1] / hist['price'].iloc[0] - 1) * 100
            if ret > 8:
                alerts.append((name, ret, random.choice(MEME_GIFS)))
    return alerts

# ----------------------------------------------------------------------
# ----------------------- FEATURE 7: AI REBALANCE BOT ------------------
# ----------------------------------------------------------------------
def get_ai_rebalance(df_port, df_net):
    if df_port.empty:
        return "Upload portfolio to get AI advice."
    current = df_net['value'].iloc[-1] if not df_net.empty else 0
    prompt = f"""
    You are a fun, bold wealth advisor. User net worth: ${current:,.0f}.
    Portfolio: {df_port[['ticker', 'allocation']].round(1).to_dict('records')}.
    Suggest 1-2 rebalance moves to boost growth. Be concise, fun, use emojis.
    """
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI offline: {e}"

# ----------------------------------------------------------------------
# ----------------------- FEATURE 8: DIVIDEND SNOWBALL -----------------
# ----------------------------------------------------------------------
def dividend_snowball(df_port, years=10):
    if df_port.empty:
        return None
    total = df_port['market_value'].sum()
    ann_yield = 0.02
    values = [total]
    for _ in range(years):
        div = values[-1] * ann_yield
        values.append(values[-1] + div)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2025, 2025+years+1)), y=values,
                             mode='lines+markers', name="Snowball",
                             line=dict(width=4, color='gold')))
    fig.update_layout(title="Dividend Snowball", yaxis_title="$")
    return fig

# ----------------------------------------------------------------------
# ----------------------- FEATURE 9: PEER BENCHMARK --------------------
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
    df_net = df_net.copy().dropna(subset=['value'])
    if len(df_net) < 3:
        return None, None, None, None, None

    df_net['time_idx'] = range(len(df_net))
    y = df_net['value'].values
    X = df_net['time_idx'].values.reshape(-1, 1)

    try:
        model = ARIMA(y, order=(1,1,0))
        fitted = model.fit()
        f = fitted.get_forecast(steps=horizon)
        forecast = f.predicted_mean
        ci = f.conf_int(alpha=0.05)
        lower, upper = ci[:, 0], ci[:, 1]
        forecast = np.array(forecast) * 0.95
        lower *= 0.95
        upper *= 0.95
    except Exception:
        forecast = np.full(horizon, y[-1] * 1.05)
        lower = np.full(horizon, y[-1] * 0.95)
        upper = np.full(horizon, y[-1] * 1.05)

    lr = LinearRegression().fit(X, y)
    future_x = np.arange(len(df_net), len(df_net)+horizon).reshape(-1, 1)
    lr_pred = lr.predict(future_x) * 0.95

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(future_x) * 0.95

    return forecast, lower, upper, lr_pred, rf_pred

# ----------------------------------------------------------------------
# --------------------------- UI START ---------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

# ------------------------------------------------------------------
# 1. Load data + **early df_net**
# ------------------------------------------------------------------
df = get_monthly_updates()
df_contrib = get_contributions()

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
# 2. ONE-TIME CSV SEED
# ------------------------------------------------------------------
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
# 3. SIDEBAR – INPUTS & FUN TOOLS
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

    st.subheader("Add Contribution")
    contrib = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
    if st.button("Save Contribution"):
        add_contribution(date_in, person, acct, float(contrib))
        st.success("Contribution saved!")
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

    # ---------- 1. Portfolio Analyzer ----------
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

    # ---------- 2. Trend Alerts ----------
    st.subheader("2. Trend Alerts")
    alerts = get_trend_alerts()
    if alerts:
        for name, ret, gif in alerts:
            st.success(f"{name} +{ret:.1f}% MoM")
            st.image(gif, width=100)
    else:
        st.info("No hot sectors right now.")

    # ---------- 7. AI Rebalance Bot ----------
    st.subheader("7. AI Rebalance Bot")
    if st.button("Ask AI Advisor"):
        if df.empty:
            st.warning("Add data first.")
        else:
            with st.spinner("Thinking..."):
                advice = get_ai_rebalance(df_port, df_net)
                st.markdown(advice)

    # ---------- 8. Dividend Snowball ----------
    st.subheader("8. Dividend Snowball")
    if not df_port.empty and st.button("Project 10Y"):
        fig = dividend_snowball(df_port)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ---------- 9. Peer Benchmark ----------
    st.subheader("9. Peer Benchmark")
    if not df_net.empty:
        cur = df_net["value"].iloc[-1]
        pct, vs = peer_benchmark(cur)
        st.metric("vs. Avg 40yo", f"Top {100-int(pct)}%", delta=f"{vs:+,}")
        if pct > 80:
            st.balloons()
    else:
        st.info("Add data to see ranking.")

# ------------------------------------------------------------------
# 4. MAIN CONTENT (only if data exists)
# ------------------------------------------------------------------
if not df.empty:
    # ---- Monthly Summary (by Year) – ONLY ONCE ----
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

    # ---- Net Worth + Benchmark ----
    st.subheader("Family Net Worth")
    fig = px.line(df_net, x="date", y="value", title="Family Net Worth",
                  labels={"value": "Total ($)"})
    max_val = df_net['value'].max()
    fig.update_yaxes(range=[0, np.ceil(max_val/50_000)*50_000], tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ---- ROR vs S&P 500 (monthly) ----
    st.subheader("Rate of Return (ROR) vs S&P 500")
    df_net['personal_ror'] = df_net['value'].pct_change() * 100
    df_ror = df_net.dropna(subset=['personal_ror']).copy()

    sp_data = fetch_ticker('^GSPC', period="5y")
    if sp_data is not None:
        sp_df = sp_data.reset_index()
        sp_df['Date'] = pd.to_datetime(sp_df['Date']).dt.tz_localize(None)
        sp_df['sp_ror'] = sp_df['price'].pct_change() * 100
        sp_df = sp_df.dropna(subset=['sp_ror'])

        df_ror = pd.merge_asof(
            df_ror[['date', 'personal_ror']].sort_values('date'),
            sp_df[['Date', 'sp_ror']].sort_values('Date'),
            left_on='date', right_on='Date',
            direction='nearest', tolerance=pd.Timedelta('1M')
        ).dropna(subset=['sp_ror'])

        if not df_ror.empty:
            fig_ror = go.Figure()
            fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['personal_ror'],
                                     name='Personal ROR', marker_color='blue'))
            fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'],
                                     name='S&P 500 ROR', marker_color='gray'))
            fig_ror.update_layout(title="Monthly ROR", barmode='group')
            st.plotly_chart(fig_ror, use_container_width=True)

            periods = len(df_net) / 12
            ann_p = (df_net['value'].iloc[-1] / df_net['value'].iloc[0]) ** (1/periods) - 1
            ann_s = (sp_df['price'].iloc[-1] / sp_df['price'].iloc[0]) ** (1/periods) - 1
            st.metric("Annualized Personal ROR", f"{ann_p*100:.2f}%")
            st.metric("Annualized S&P 500 ROR (Your Period)", f"{ann_s*100:.2f}%")
            st.metric("Historical S&P Avg (1950-)", "7.00%")
        else:
            st.info("Not enough overlapping data for ROR chart.")
    else:
        st.info("S&P 500 data unavailable – using static historical avg.")
        df_ror['sp_ror'] = HISTORICAL_SP_MONTHLY * 100
        fig_ror = go.Figure()
        fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['personal_ror'],
                                 name='Personal ROR', marker_color='blue'))
        fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'],
                                 name='S&P Avg', marker_color='gray'))
        fig_ror.update_layout(title="Monthly ROR (vs static avg)", barmode='group')
        st.plotly_chart(fig_ror, use_container_width=True)

    # ---- Per-Person ROR ----
    st.subheader("Per-Person ROR vs S&P 500")
    fig_pp = go.Figure()
    for p in ['Sean', 'Kim']:
        dp = df[df['person'] == p].groupby("date")["value"].sum().reset_index()
        dp = dp.sort_values("date")
        dp["date"] = dp["date"].dt.tz_localize(None)
        dp['ror'] = dp['value'].pct_change() * 100
        dp = dp.dropna(subset=['ror'])
        if not dp.empty:
            fig_pp.add_trace(go.Scatter(x=dp['date'], y=dp['ror'],
                                        mode='lines+markers', name=f"{p} ROR"))
    if 'df_ror' in locals() and not df_ror.empty:
        fig_pp.add_trace(go.Scatter(x=df_ror['date'], y=df_ror['sp_ror'],
                                    mode='lines', name='S&P 500', line=dict(dash='dot')))
    fig_pp.update_layout(title="Per-Person Monthly ROR", hovermode='x unified')
    st.plotly_chart(fig_pp, use_container_width=True)

    # ---- YTD & M2M Gains ----
    tab1, tab2 = st.tabs(["YTD Gain/Loss", "Month-to-Month Gain/Loss"])
    with tab1:
        df_ytd = df_net.copy()
        df_ytd['year'] = df_ytd['date'].dt.year
        fig_ytd = px.line(df_ytd, x="date", y="value", color="year",
                          title="YTD Net Worth")
        st.plotly_chart(fig_ytd, use_container_width=True)
    with tab2:
        df_m2m = df_net.copy()
        df_m2m['prev'] = df_m2m['value'].shift(1)
        df_m2m['gain'] = df_m2m['value'] - df_m2m['prev']
        df_m2m = df_m2m.dropna()
        fig_m2m = px.bar(df_m2m, x="date", y="gain", title="Month-to-Month Gain/Loss")
        st.plotly_chart(fig_m2m, use_container_width=True)

    # ---- AI Growth Projections ----
    st.subheader("AI Growth Projections")
    horizon = st.slider("Months Ahead", 12, 60, 24)
    arima_f, ar_l, ar_u, lr_f, rf_f = ai_projections(df_net, horizon)
    if arima_f is not None:
        future = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1),
                               periods=horizon, freq='ME')
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"],
                                      name="Historical", line=dict(color="blue")))
        fig_proj.add_trace(go.Scatter(x=future, y=arima_f, name="ARIMA",
                                      line=dict(color="green")))
        fig_proj.add_trace(go.Scatter(x=future, y=ar_l, fill=None,
                                      line=dict(color="lightgreen", dash="dash"),
                                      showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future, y=ar_u, fill='tonexty',
                                      line=dict(color="lightgreen"), name="CI"))
        fig_proj.add_trace(go.Scatter(x=future, y=lr_f, name="Linear",
                                      line=dict(color="orange")))
        fig_proj.add_trace(go.Scatter(x=future, y=rf_f, name="Random Forest",
                                      line=dict(color="red")))
        fig_proj.update_layout(title=f"Projections ({horizon} mo)", yaxis_title="$")
        st.plotly_chart(fig_proj, use_container_width=True)

    # ---- Goals ----
    st.subheader("Financial Goals")
    cur = df_net["value"].iloc[-1]
    for g in get_goals():
        prog = min(cur / g.target, 1.0)
        st.progress(prog)
        st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f} → {g.by_year}")

    with st.expander("Edit / Delete Goals"):
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

    # ---- Delete Entry ----
    st.subheader("Delete Entry")
    choice = st.selectbox("Select", df.index,
                          format_func=lambda i: f"{df.loc[i,'date']} – {df.loc[i,'person']} – ${df.loc[i,'value']:,.0f}")
    if st.button("Delete"):
        row = df.loc[choice]
        sess = get_session()
        sess.query(MonthlyUpdate).filter_by(
            date=row["date"], person=row["person"], account_type=row["account_type"]
        ).delete()
        sess.commit()
        sess.close()
        st.success("Deleted!")
        st.rerun()

    # ---- Export ----
    st.download_button("Export Monthly Values", df.to_csv(index=False).encode(),
                       "monthly_values.csv", "text/csv")
    if not df_contrib.empty:
        st.download_button("Export Contributions", df_contrib.to_csv(index=False).encode(),
                           "contributions.csv", "text/csv")
else:
    st.info("Add your first monthly update to get started!")
