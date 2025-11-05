import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf  # <-- REQUIRED for benchmarks
import time  # For retries

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
from sqlalchemy.exc import SQLAlchemyError

# ---------- DATABASE SETUP ----------
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
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# ---------- DB HELPERS ----------
def get_session():
    return Session()

def reset_database():
    try:
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
    except Exception as e:
        st.error(f"Reset failed: {e}")

def load_accounts():
    try:
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
    except Exception as e:
        st.error(f"Load accounts error: {e}")
        return {}

def add_person(name):
    try:
        sess = get_session()
        sess.merge(AccountConfig(person=name, account_type='Personal'))
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add person error: {e}")

def add_account_type(person, acc_type):
    try:
        sess = get_session()
        sess.merge(AccountConfig(person=person, account_type=acc_type))
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add account error: {e}")

def add_monthly_update(date, person, acc_type, value):
    try:
        sess = get_session()
        stmt = insert(MonthlyUpdate.__table__).values(
            date=date, person=person, account_type=acc_type, value=value
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['date', 'person', 'account_type'],
            set_={'value': value}
        )
        sess.execute(stmt)
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add update error: {e}")

def add_contribution(date, person, acc_type, amount):
    try:
        sess = get_session()
        stmt = insert(Contribution.__table__).values(
            date=date, person=person, account_type=acc_type, contribution=amount
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['date', 'person', 'account_type'],
            set_={'contribution': amount}
        )
        sess.execute(stmt)
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add contribution error: {e}")

def get_monthly_updates():
    try:
        sess = get_session()
        rows = sess.query(MonthlyUpdate).all()
        sess.close()
        return pd.DataFrame([
            {'date': r.date, 'person': r.person, 'account_type': r.account_type, 'value': r.value}
            for r in rows
        ])
    except Exception as e:
        st.error(f"Get updates error: {e}")
        return pd.DataFrame()

def get_contributions():
    try:
        sess = get_session()
        rows = sess.query(Contribution).all()
        sess.close()
        return pd.DataFrame([
            {'date': r.date, 'person': r.person, 'account_type': r.account_type, 'contribution': r.contribution}
            for r in rows
        ])
    except Exception as e:
        st.error(f"Get contributions error: {e}")
        return pd.DataFrame()

def get_goals():
    try:
        sess = get_session()
        goals = sess.query(Goal).all()
        sess.close()
        return goals
    except Exception as e:
        st.error(f"Get goals error: {e}")
        return []

def add_goal(name, target, by_year):
    try:
        sess = get_session()
        sess.merge(Goal(name=name, target=target, by_year=by_year))
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add goal error: {e}")

def update_goal(name, target, by_year):
    try:
        sess = get_session()
        goal = sess.query(Goal).filter_by(name=name).first()
        if goal:
            goal.target = target
            goal.by_year = by_year
            sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Update goal error: {e}")

def delete_goal(name):
    try:
        sess = get_session()
        sess.query(Goal).filter_by(name=name).delete()
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Delete goal error: {e}")

# ---------- ONE-TIME CSV SEED ----------
def seed_database_from_csv(df_uploaded):
    try:
        sess = get_session()
        count = sess.query(MonthlyUpdate).count()
        if count > 0:
            st.info("Database already has data — skipping seed.")
            return

        for _, row in df_uploaded.iterrows():
            date = pd.to_datetime(row['date'], format='%b-%y', errors='coerce').date()
            if pd.isna(date):
                continue
            person = str(row['person'])
            account_type = str(row['account_type'])
            value = float(row['value'])
            stmt = insert(MonthlyUpdate.__table__).values(
                date=date, person=person, account_type=account_type, value=value
            )
            stmt = stmt.on_conflict_do_nothing()
            sess.execute(stmt)
        sess.commit()
        sess.close()
        st.success(f"Seeded {len(df_uploaded)} rows!")
    except Exception as e:
        st.error(f"Seed failed: {e}")

# ---------- CACHED YFINANCE FETCH WITH RETRIES ----------
@st.cache_data(ttl=3600)  # Cache 1hr
def fetch_benchmark_data(ticker, start_date, end_date, max_retries=3):
    """Fetch with retries and tz-aware dates."""
    start_tz = pd.Timestamp(start_date, tz='UTC').isoformat()
    end_tz = pd.Timestamp(end_date, tz='UTC').isoformat()
    
    for attempt in range(max_retries):
        try:
            st.write(f"Fetching {ticker} (attempt {attempt+1}/{max_retries})...")
            data = yf.download(ticker, start=start_tz, end=end_tz, progress=False)
            if not data.empty and 'Adj Close' in data.columns and len(data) > 0:
                st.success(f"✅ Fetched {len(data)} rows for {ticker}")
                return data
            else:
                raise ValueError("Empty data or no 'Adj Close'")
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5s
            else:
                st.error(f"All retries failed for {ticker}. Using static fallback.")
                return None  # Trigger fallback
    
    return None

# ---------- STATIC FALLBACK FOR S&P (Historical Avg) ----------
def get_sp_fallback_data(user_start_date, end_date, num_points=50):
    """Static conservative S&P data: 7% annual real return since 1950, normalized from user start."""
    historical_avg_ror = 0.07  # Conservative real return
    dates = pd.date_range(start=user_start_date, end=end_date, freq='M')[:num_points]
    initial_value = 100  # Arbitrary base
    values = [initial_value * (1 + historical_avg_ror / 12) ** i for i in range(len(dates))]
    fallback_df = pd.DataFrame({'Date': dates, 'Adj Close': values})
    fallback_df['Date'] = fallback_df['Date'].dt.tz_localize(None)
    st.info("Using static S&P fallback (7% historical avg return).")
    return fallback_df

# ---------- AI PROJECTIONS ----------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3:
        return None, None, None, None, None
    df_net = df_net.copy()
    df_net['time_idx'] = range(len(df_net))
    y = df_net['value'].values
    X = df_net['time_idx'].values.reshape(-1, 1)

    # Conservative ARIMA
    try:
        model = ARIMA(y, order=(1,1,0))  # Simple for stability
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        ci = fitted.get_forecast(steps=horizon).conf_int(alpha=0.05)  # Wider CI
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        forecast *= 0.95  # Dampen for conservatism
    except Exception as arima_err:
        st.warning(f"ARIMA failed, using conservative linear trend.")
        forecast = np.full(horizon, y[-1] * 1.05)  # Cap at 5% growth
        lower = np.full(horizon, y[-1]*0.95)
        upper = np.full(horizon, y[-1]*1.05)

    lr = LinearRegression().fit(X, y)
    future_x = np.array(range(len(df_net), len(df_net) + horizon)).reshape(-1, 1)
    lr_pred = lr.predict(future_x) * 0.95  # Dampen

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(future_x) * 0.95  # Dampen

    return forecast, lower, upper, lr_pred, rf_pred

# ---------- UI START ----------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

df = get_monthly_updates()
df_contrib = get_contributions()

# ONE-TIME SEED
if df.empty:
    st.subheader("Seed Database with CSV (One-Time)")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            if all(c in df_up.columns for c in ['date', 'person', 'account_type', 'value']):
                if st.button("Import CSV"):
                    seed_database_from_csv(df_up)
                    st.rerun()
            else:
                st.error("CSV needs: date, person, account_type, value")
        except Exception as e:
            st.error(f"CSV error: {e}")

# SIDEBAR
with st.sidebar:
    st.subheader("Add Monthly Update")
    accounts_dict = load_accounts()
    persons = list(accounts_dict.keys())
    person = st.selectbox("Person", persons, key="person_sel")
    acct_opts = accounts_dict.get(person, [])
    account_type = st.selectbox("Account", acct_opts or ["Personal"], key="acct_sel")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        value = st.number_input("Value ($)", min_value=0.0, format="%.2f")
    if st.button("Save Entry"):
        add_monthly_update(date, person, account_type, float(value))
        st.success("Saved!")
        st.rerun()

    st.subheader("Add Contribution")
    contrib = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
    if st.button("Save Contribution"):
        add_contribution(date, person, account_type, float(contrib))
        st.success("Contribution saved!")
        st.rerun()

    st.subheader("Add Goal")
    g_name = st.text_input("Goal Name")
    g_target = st.number_input("Target ($)", min_value=0.0)
    g_year = st.number_input("By Year", min_value=2000, step=1)
    if st.button("Add Goal"):
        if g_name and g_target > 0:
            add_goal(g_name, g_target, g_year)
            st.success("Goal added!")
            st.rerun()

    if st.button("Reset DB (Admin)"):
        reset_database()
        st.rerun()

# MAIN CONTENT
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])

    # Fix df_contrib safely
    if not df_contrib.empty and 'date' in df_contrib.columns:
        df_contrib["date"] = pd.to_datetime(df_contrib["date"])
    else:
        df_contrib = pd.DataFrame(columns=['date', 'person', 'account_type', 'contribution'])

    # COLLAPSIBLE YEARLY SUMMARY
    st.subheader("Monthly Summary (by Year)")
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique(), reverse=True):
        with st.expander(f"{year} – Click to Expand"):
            year_df = df[df['year'] == year]
            pivot = year_df.pivot_table(
                index="date", columns=["person", "account_type"],
                values="value", aggfunc="sum", fill_value=0
            )
            st.dataframe(pivot.style.format("${:,.0f}"))

    # NET WORTH + BENCHMARK
    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index()
    df_net = df_net.sort_values("date")
    df_net["date"] = df_net["date"].dt.tz_localize(None)
    st.dataframe(df_net.head(5))  # Sample data - share this output!

    fig_net = px.line(df_net, x="date", y="value", title="Family Net Worth", labels={"value": "Total ($)"})
    max_value = df_net['value'].max()
    y_max = np.ceil(max_value / 50000) * 50000
    fig_net.update_yaxes(range=[0, y_max])
    fig_net.update_layout(yaxis_tickformat="$,.0f")

    # Benchmark
    benchmark = st.selectbox("Benchmark", ["S&P 500 (^GSPC)", "Dow Jones (^DJI)", "Other Ticker"])
    if benchmark == "Other Ticker":
        ticker = st.text_input("Ticker", "AAPL")
    else:
        ticker = benchmark.split(" ")[-1].strip("()")

    start_date = df_net["date"].min().date()
    end_date = datetime.now().date()
    historical_start = '1950-01-01'  # Long-term historical

    # Use cached fetch
    data = fetch_benchmark_data(ticker, historical_start, end_date)
    if data is not None and not data.empty:
        bench = data['Adj Close'].reset_index()
        bench['Date'] = pd.to_datetime(bench['Date']).dt.tz_localize(None)
        # Filter to user's data range for normalization
        bench_user_range = bench[(bench['Date'] >= pd.Timestamp(start_date)) & (bench['Date'] <= pd.Timestamp(end_date))]
        if not bench_user_range.empty:
            initial_net = df_net["value"].iloc[0]
            initial_bench = bench_user_range['Adj Close'].iloc[0]
            bench_user_range['norm'] = (bench_user_range['Adj Close'] / initial_bench) * initial_net
            fig_net.add_trace(go.Scatter(
                x=bench_user_range['Date'], y=bench_user_range['norm'],
                name=f"{ticker} (Historical Normalized)", line=dict(dash='dot', color='gray')
            ))
        else:
            # Fallback to full historical normalized from user start
            initial_bench = bench['Adj Close'].loc[bench['Date'] >= pd.Timestamp(start_date)].iloc[0] if len(bench) > 0 else 100
            bench['norm'] = (bench['Adj Close'] / initial_bench) * initial_net
            fig_net.add_trace(go.Scatter(
                x=bench['Date'], y=bench['norm'],
                name=f"{ticker} (Historical)", line=dict(dash='dot', color='gray')
            ))
    else:
        # Static fallback for S&P specifically
        if ticker == '^GSPC':
            fallback_bench = get_sp_fallback_data(start_date, end_date)
            if not fallback_bench.empty:
                initial_net = df_net["value"].iloc[0]
                initial_bench = fallback_bench['Adj Close'].iloc[0]
                fallback_bench['norm'] = (fallback_bench['Adj Close'] / initial_bench) * initial_net
                fig_net.add_trace(go.Scatter(
                    x=fallback_bench['Date'], y=fallback_bench['norm'],
                    name="S&P 500 (Static Historical Avg)", line=dict(dash='dot', color='gray')
                ))
        else:
            st.warning("Benchmark unavailable—using family net worth only.")

    st.plotly_chart(fig_net, use_container_width=True)

    # ROR vs S&P 500 (with historical)
    st.subheader("Rate of Return (ROR) vs S&P 500")
    
    df_net['personal_ror'] = df_net['value'].pct_change() * 100
    df_net_ror = df_net.dropna(subset=['personal_ror']).copy()
    
    if len(df_net_ror) < 2:
        st.warning("Need at least 2 months of data for ROR.")
    else:
        sp_ticker = '^GSPC'
        sp_data = fetch_benchmark_data(sp_ticker, historical_start, end_date)
        if sp_data is not None and not sp_data.empty:
            sp_df = sp_data['Adj Close'].reset_index()
            sp_df['Date'] = pd.to_datetime(sp_df['Date']).dt.tz_localize(None)
            # User-range filter for merge
            sp_user_range = sp_df[(sp_df['Date'] >= pd.Timestamp(start_date)) & (sp_df['Date'] <= pd.Timestamp(end_date))]
            sp_user_range['sp_ror'] = sp_user_range['Adj Close'].pct_change() * 100
            sp_user_range = sp_user_range.dropna(subset=['sp_ror'])
            
            if not sp_user_range.empty:
                df_ror = pd.merge_asof(
                    df_net_ror[['date', 'personal_ror']].sort_values('date'),
                    sp_user_range[['Date', 'sp_ror']].sort_values('Date'),
                    left_on='date', right_on='Date', direction='nearest', tolerance=pd.Timedelta('1M')
                )
                df_ror = df_ror.dropna(subset=['sp_ror'])
                
                if not df_ror.empty:
                    fig_ror = go.Figure()
                    fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['personal_ror'], name='Personal ROR', marker_color='blue'))
                    fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'], name='S&P 500 ROR (Historical Period)', marker_color='gray'))
                    fig_ror.update_layout(
                        title="Monthly ROR Comparison",
                        xaxis_title="Date",
                        yaxis_title="Return (%)",
                        barmode='group'
                    )
                    st.plotly_chart(fig_ror, use_container_width=True)
                    
                    periods = len(df_net) / 12
                    personal_annual_ror = (df_net['value'].iloc[-1] / df_net['value'].iloc[0]) ** (1 / periods) - 1
                    sp_annual_ror = (sp_user_range['Adj Close'].iloc[-1] / sp_user_range['Adj Close'].iloc[0]) ** (1 / periods) - 1
                    historical_sp_avg = 0.07  # Conservative real return since 1950
                    st.metric("Annualized Personal ROR", f"{personal_annual_ror * 100:.2f}%")
                    st.metric("Annualized S&P 500 ROR (Your Period)", f"{sp_annual_ror * 100:.2f}%")
                    st.metric("Historical S&P Avg ROR (Since 1950)", f"{historical_sp_avg * 100:.2f}% (Conservative Real)")
                    outperformance = personal_annual_ror - historical_sp_avg
                    st.metric("Outperformance vs Historical S&P", f"{outperformance * 100:+.2f}%")
                else:
                    st.warning("No overlapping dates for ROR merge.")
            else:
                st.warning("No S&P data in your range.")
        else:
            # Static fallback ROR (constant historical avg monthly)
            historical_monthly = historical_sp_avg / 12
            df_ror_fallback = df_net_ror[['date', 'personal_ror']].copy()
            df_ror_fallback['sp_ror'] = historical_monthly * 100
            fig_ror_f = go.Figure()
            fig_ror_f.add_trace(go.Bar(x=df_ror_fallback['date'], y=df_ror_fallback['personal_ror'], name='Personal ROR', marker_color='blue'))
            fig_ror_f.add_trace(go.Bar(x=df_ror_fallback['date'], y=df_ror_fallback['sp_ror'], name='S&P Historical Avg', marker_color='gray'))
            fig_ror_f.update_layout(title="Monthly ROR vs Static S&P Avg", barmode='group')
            st.plotly_chart(fig_ror_f, use_container_width=True)
            st.info("Using static S&P monthly avg (0.58%) for comparison.")

    # Per-Person ROR
    st.subheader("Per-Person ROR vs S&P 500")
    persons = ['Sean', 'Kim']
    fig_person_ror = go.Figure()
    for p in persons:
        df_p = df[df['person'] == p].groupby("date")["value"].sum().reset_index()
        df_p = df_p.sort_values("date")
        df_p["date"] = df_p["date"].dt.tz_localize(None)
        df_p['ror'] = df_p['value'].pct_change() * 100
        df_p = df_p.dropna(subset=['ror'])
        if not df_p.empty:
            fig_person_ror.add_trace(go.Scatter(x=df_p['date'], y=df_p['ror'], mode='lines+markers', name=f"{p} Monthly ROR", line=dict(width=2)))
    
    # Overlay S&P (from above or fallback)
    if 'df_ror' in locals() and not df_ror.empty:
        fig_person_ror.add_trace(go.Scatter(x=df_ror['date'], y=df_ror['sp_ror'], mode='lines', name='S&P 500 ROR', line=dict(dash='dot', color='gray')))
    else:
        # Fallback line
        sp_fallback_dates = df_net_ror['date']
        sp_fallback_ror = [historical_monthly * 100] * len(sp_fallback_dates)
        fig_person_ror.add_trace(go.Scatter(x=sp_fallback_dates, y=sp_fallback_ror, mode='lines', name='S&P Historical Avg', line=dict(dash='dot', color='gray')))
    
    fig_person_ror.update_layout(
        title="Per-Person Monthly ROR vs S&P",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_person_ror, use_container_width=True)

    # YTD & M2M GAINS
    tab1, tab2 = st.tabs(["YTD Gain/Loss", "Month-to-Month Gain/Loss"])

    with tab1:
        df_ytd = df_net.copy()
        df_ytd['year'] = df_ytd['date'].dt.year
        df_ytd['ytd'] = df_ytd.groupby('year')['value'].cumsum()
        fig_ytd = px.line(df_ytd, x="date", y="value", color="year", title="YTD Net Worth")
        st.plotly_chart(fig_ytd, use_container_width=True)

    with tab2:
        df_m2m = df_net.copy()
        df_m2m['prev'] = df_m2m['value'].shift(1)
        df_m2m['gain'] = df_m2m['value'] - df_m2m['prev']
        df_m2m = df_m2m.dropna()
        fig_m2m = px.bar(df_m2m, x="date", y="gain", title="Month-to-Month Gain/Loss")
        st.plotly_chart(fig_m2m, use_container_width=True)

    # AI PROJECTIONS
    st.subheader("AI Growth Projections")
    horizon = st.slider("Months Ahead", 12, 60, 24)
    arima_f, ar_lower, ar_upper, lr_f, rf_f = ai_projections(df_net, horizon)

    if arima_f is not None:
        future_dates = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='MS')
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Historical", line=dict(color="blue")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_f, name="ARIMA (Conservative)", line=dict(color="green")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=ar_lower, fill=None, line=dict(color="lightgreen", dash="dash"), showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=ar_upper, fill='tonexty', line=dict(color="lightgreen"), name="ARIMA CI"))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=lr_f, name="Linear (Conservative)", line=dict(color="orange")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=rf_f, name="Random Forest (Conservative)", line=dict(color="red")))
        fig_proj.update_layout(title=f"Conservative Projections ({horizon} months)", yaxis_title="Net Worth ($)")
        st.plotly_chart(fig_proj, use_container_width=True)

        proj_df = pd.DataFrame({
            "Model": ["ARIMA", "Linear", "Random Forest"],
            "24 mo": [arima_f[23] if len(arima_f) > 23 else np.nan, lr_f[23], rf_f[23]],
            "60 mo": [arima_f[59] if len(arima_f) > 59 else np.nan, lr_f[59] if len(lr_f) > 59 else np.nan, rf_f[59] if len(rf_f) > 59 else np.nan]
        }).round(0).style.format({"24 mo": "${:,.0f}", "60 mo": "${:,.0f}"})
        st.dataframe(proj_df)
    else:
        st.info("Need 3+ months of data.")

    # GOALS
    st.subheader("Financial Goals")
    current = df_net["value"].iloc[-1]
    for g in get_goals():
        prog = min(current / g.target, 1.0)
        st.progress(prog)
        st.write(f"**{g.name}**: ${current:,.0f} / ${g.target:,.0f} → {g.by_year}")

    with st.expander("Edit/Delete Goals"):
        for g in get_goals():
            with st.expander(f"Edit {g.name}"):
                t = st.number_input("Target", value=g.target, key=f"t_{g.name}")
                y = st.number_input("Year", value=g.by_year, key=f"y_{g.name}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Update", key=f"u_{g.name}"):
                        update_goal(g.name, t, y)
                        st.success("Updated!")
                        st.rerun()
                with c2:
                    if st.button("Delete", key=f"d_{g.name}"):
                        delete_goal(g.name)
                        st.success("Deleted!")
                        st.rerun()

    # DELETE ENTRY
    st.subheader("Delete Entry")
    choice = st.selectbox("Select", df.index, format_func=lambda i: f"{df.loc[i,'date']} – {df.loc[i,'person']} – ${df.loc[i,'value']:,.0f}")
    if st.button("Delete"):
        row = df.loc[choice]
        sess = get_session()
        sess.query(MonthlyUpdate).filter_by(date=row["date"], person=row["person"], account_type=row["account_type"]).delete()
        sess.commit()
        sess.close()
        st.success("Deleted!")
        st.rerun()

    # EXPORT
    st.download_button("Export Values", df.to_csv(index=False).encode(), "values.csv")
    if not df_contrib.empty:
        st.download_button("Export Contributions", df_contrib.to_csv(index=False).encode(), "contributions.csv")
