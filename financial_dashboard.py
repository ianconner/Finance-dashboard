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
# ----------------------- YFINANCE HELPER (only for benchmarks) -------
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_ticker(ticker, period="1d", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': 'price'})
    except Exception:
        pass
    return None

# ----------------------------------------------------------------------
# ----------------------- PORTFOLIO ANALYZER ---------------------------
# ----------------------------------------------------------------------
def analyze_portfolio(df_port):
    """
    Uses Fidelity CSV columns:
        Symbol, Quantity, Last Price, Average Cost Basis, Current Value
    """
    required = ['Symbol', 'Quantity', 'Last Price', 'Average Cost Basis', 'Current Value']
    missing = [c for c in required if c not in df_port.columns]
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return None, None, None

    # ---- SAFE TYPE CONVERSION ----
    df_port = df_port[required].copy()
    df_port = df_port.dropna(subset=['Symbol', 'Quantity', 'Last Price', 'Current Value'])

    # Convert Symbol to string safely
    df_port['ticker'] = df_port['Symbol'].astype(str).str.upper().str.strip()

    # Convert numeric columns safely
    df_port['shares'] = pd.to_numeric(df_port['Quantity'], errors='coerce')
    df_port['price'] = pd.to_numeric(df_port['Last Price'], errors='coerce')
    df_port['cost_basis'] = pd.to_numeric(df_port['Average Cost Basis'], errors='coerce')
    df_port['market_value'] = pd.to_numeric(
        df_port['Current Value'].astype(str).str.replace(',', ''), errors='coerce'
    )

    df_port = df_port.dropna(subset=['shares', 'price', 'cost_basis', 'market_value'])
    if df_port.empty:
        st.error("No valid rows after cleaning.")
        return None, None, None

    total_value = df_port['market_value'].sum()
    df_port['allocation'] = df_port['market_value'] / total_value * 100
    df_port['gain'] = df_port['market_value'] - (df_port['shares'] * df_port['cost_basis'])

    # 6-month return (optional)
    df_port['6mo_return'] = 0.0
    for i, row in df_port.iterrows():
        hist = fetch_ticker(row['ticker'], period="6mo")
        if hist is not None and len(hist) > 1:
            df_port.at[i, '6mo_return'] = (hist['price'].iloc[-1] / hist['price'].iloc[0] - 1) * 100

    # Health score
    std_alloc = df_port['allocation'].std()
    health = max(0, min(100, 100 - std_alloc * 8))

    # Recommendations
    recs = []
    over = df_port[df_port['allocation'] > 25]['ticker'].tolist()
    if over:
        recs.append(f"Overweight: {', '.join(over)} — consider trimming.")
    if not df_port.empty:
        top = df_port.loc[df_port['6mo_return'].idxmax()]
        recs.append(f"Hot pick: {top['ticker']} (+{top['6mo_return']:.1f}%)")

    return df_port, health, recs

# ----------------------------------------------------------------------
# ----------------------- TREND ALERTS ---------------------------------
# ----------------------------------------------------------------------
SECTOR_ETFS = {
    "Tech": "XLK", "Health": "XLV", "Finance": "XLF",
    "Energy": "XLE", "Consumer": "XLY", "Industrials": "XLI", "AI": "BOTZ"
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
# ----------------------- AI REBALANCE BOT -----------------------------
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
# ----------------------- DIVIDEND SNOWBALL ----------------------------
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

# Load data
df = get_monthly_updates()
df_contrib = get_contributions()

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

    # --- Portfolio Analyzer ---
    st.subheader("1. Portfolio Analyzer")
    port_file = st.file_uploader("Upload Fidelity CSV", type="csv", key="port")
    df_port = pd.DataFrame()
    if port_file:
        try:
            df_port = pd.read_csv(port_file)
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
        except Exception as e:
            st.error(f"CSV error: {e}")

    # --- Trend Alerts ---
    st.subheader("2. Trend Alerts")
    alerts = get_trend_alerts()
    if alerts:
        for name, ret, gif in alerts:
            st.success(f"{name} +{ret:.1f}% MoM")
            st.image(gif, width=100)
    else:
        st.info("No hot sectors right now.")

    # --- AI Rebalance Bot ---
    st.subheader("7. AI Rebalance Bot")
    if st.button("Ask AI Advisor"):
        if df.empty:
            st.warning("Add data first.")
        else:
            with st.spinner("Thinking..."):
                advice = get_ai_rebalance(df_port, df_net)
                st.markdown(advice)

    # --- Dividend Snowball ---
    st.subheader("8. Dividend Snowball")
    if not df_port.empty and st.button("Project 10Y"):
        fig = dividend_snowball(df_port)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # --- Peer Benchmark (no balloons) ---
    st.subheader("9. Peer Benchmark")
    if not df_net.empty:
        cur = df_net["value"].iloc[-1]
        pct, vs = peer_benchmark(cur)
        st.metric("vs. Avg 40yo", f"Top {100-int(pct)}%", delta=f"{vs:+,}")
        # Removed st.balloons()
    else:
        st.info("Enter data to see peer rank.")

# ------------------------------------------------------------------
# MAIN CONTENT
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

    # --- Delete Entry ---
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

    # --- Export ---
    st.download_button("Export Monthly Values", df.to_csv(index=False).encode(),
                       "monthly_values.csv", "text/csv")

else:
    st.info("Add your first monthly update to get started!")
