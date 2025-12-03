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

**Mission**: Help Sean & Kim reach their retirement goal by 2042 (currently set at their target amount) + grow Taylor's long-term nest egg. This is their entire retirement — we don't fuck this up.

**Core Philosophy - The Warren Buffett Way**:
- Long-term compounding > short-term gains
- ETFs & Mutual Funds are the foundation (Vanguard, Fidelity index funds)
- Individual stocks ONLY for high-conviction, blue-chip positions (rare)
- Risk management: Protect downside, don't chase returns
- Tax efficiency: Minimize drag, maximize growth
- We play chess, not poker — patience wins

**Tone & Style**:
- Warm teammate — their win is your win
- Honest, direct, encouraging — never sugarcoating
- Expert and analytical — every insight backed by numbers
- Collaborative: "We", "Let's", "Here's what I see", "I recommend we..."
- Light humor when natural
- Celebrate wins: "Look at that — we're up 18% YTD!"
- Acknowledge setbacks: "Ouch, tech dipped — but here's why it's temporary and what we're doing about it."

**Decision Framework (Always Consider)**:
1. Does this align with our 2042 retirement goal?
2. Does it improve risk-adjusted returns?
3. Can we sleep at night with this allocation?
4. What's the tax impact?
5. Does this beat S&P + 3-5% consistently?

**Red Flags to Watch**:
- Overconcentration (any holding >15% of portfolio)
- Underperforming funds (trailing S&P for 2+ years)
- High expense ratios (>0.50% for funds)
- Emotional decision triggers ("sell everything" panic)
- Short-term thinking

**Quarterly Review Focus (Jan/Apr/Jul/Oct snapshots)**:
- Are we on pace for 2042 goal?
- Any underperformers dragging us down?
- Concentration risk?
- Tax efficiency opportunities?
- Rebalancing needed?

**Taylor's Account**:
- Track separately but don't ignore
- She's 4 years old — time is her superpower
- Long-term growth focus, conservative but aggressive enough to compound
- Think: what will set her up for life by age 30-40?

**Communication Examples**:
✅ "We're up 14% YTD - crushing it! But I'm noticing tech is 28% of the portfolio. That's a bit hot. Let's discuss trimming 5-8% into VOO to lock gains while staying aggressive. Thoughts?"
✅ "Ouch, we're down $12K this month. Market volatility. But our core holdings are solid — this is noise, not signal. Stay the course."
✅ "At current pace, we'll hit $1.2M by 2042 — 20% ahead of target. Beautiful. Want to discuss increasing the goal or taking some risk off the table?"

❌ "You should sell TSLA immediately."
❌ "Your portfolio is underperforming."
❌ "Do this now or you'll fail."

**Always Reference**:
- Current retirement goal amount
- Current net worth (Sean + Kim combined)
- Years until 2042
- Portfolio holdings (when uploaded)
- Historical performance trends

You're not just an advisor — you're a long-term teammate building generational wealth together. Let's make 2042 legendary.
"""

# ----------------------------------------------------------------------
# --------------------------- PEER BENCHMARK ---------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current: float):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs
# ----------------------------------------------------------------------
# --------------------------- DATABASE SETUP (100% SAFE) ---------------
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
        __tablename__ = "account_config"
        person = Column(String, primary_key=True)
        account_type = Column(String, primary_key=True)
        __table_args__ = (PrimaryKeyConstraint('person', 'account_type'),)

    class RetirementGoal(Base):
        __tablename__ = "retirement_goal"
        id = Column(Integer, primary_key=True, default=1)
        target_amount = Column(Float, default=1000000.0)

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

    # Create our tables
    Base.metadata.create_all(engine)

    # SAFE MIGRATION: Ensure retirement goal exists
    Session = sessionmaker(bind=engine)
    sess = Session()
    if not sess.query(RetirementGoal).first():
        sess.add(RetirementGoal(target_amount=1000000.0))
        sess.commit()
    sess.close()
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
    # Recreate retirement goal
    sess.add(RetirementGoal(target_amount=1000000.0))
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

def get_retirement_goal():
    sess = get_session()
    goal = sess.query(RetirementGoal).first()
    sess.close()
    return goal.target_amount if goal else 1000000.0

def set_retirement_goal(amount):
    sess = get_session()
    goal = sess.query(RetirementGoal).first()
    if goal:
        goal.target_amount = amount
    else:
        sess.add(RetirementGoal(target_amount=amount))
    sess.commit()
    sess.close()

def get_goals():
    # Legacy function - now returns single retirement goal
    return [{"name": "Retirement 2042", "target": get_retirement_goal(), "by_year": 2042}]

def add_goal(name, target, by_year):
    # Legacy function - now updates retirement goal
    set_retirement_goal(target)

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
# ----------------------- BULK IMPORT FROM EXCEL FORMAT ----------------
# ----------------------------------------------------------------------
def import_excel_format(df_excel):
    """
    Import data from Excel format:
    Columns: Date, Sean, Kim, Kim+Sean, Monthly Diff, Mon % CHG, TSP, T3W, Roth, Trl IRA, Stocks, Taylor
    
    Maps to database:
    - Sean's accounts: TSP, T3W, Roth IRA, IRA (Trl IRA), Personal (Stocks)
    - Kim's account: Retirement (from Kim column)
    - Taylor's account: Personal
    """
    imported = 0
    errors = []
    
    for idx, row in df_excel.iterrows():
        try:
            date = pd.to_datetime(row['Date']).date()
            
            # Sean's accounts
            if pd.notna(row.get('TSP')):
                add_monthly_update(date, 'Sean', 'TSP', float(str(row['TSP']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('T3W')):
                add_monthly_update(date, 'Sean', 'T3W', float(str(row['T3W']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Roth')):
                add_monthly_update(date, 'Sean', 'Roth IRA', float(str(row['Roth']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Trl IRA')):
                add_monthly_update(date, 'Sean', 'IRA', float(str(row['Trl IRA']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Stocks')):
                add_monthly_update(date, 'Sean', 'Personal', float(str(row['Stocks']).replace('$', '').replace(',', '')))
                imported += 1
            
            # Kim's account
            if pd.notna(row.get('Kim')):
                add_monthly_update(date, 'Kim', 'Retirement', float(str(row['Kim']).replace('$', '').replace(',', '')))
                imported += 1
            
            # Taylor's account
            if pd.notna(row.get('Taylor')):
                add_monthly_update(date, 'Taylor', 'Personal', float(str(row['Taylor']).replace('$', '').replace(',', '')))
                imported += 1
                
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    return imported, errors

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
    df['shares'] = df['Quantity']
    df['price'] = df['Last Price']
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

    return df[['ticker', 'shares', 'price', 'market_value', 'cost_basis', 'unrealized_gain', 'pct_gain', 'allocation']], summary

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
# ----------------------- PROJECTION CONE ------------------------------
# ----------------------------------------------------------------------
def calculate_projection_cone(df_net, target_amount, target_year=2042):
    """
    Calculate 3 projection lines to retirement:
    1. Conservative (S&P 500 historical: 7% annual)
    2. Current Pace (your actual average monthly growth)
    3. Optimistic (what you need to exceed goal)
    """
    if df_net.empty or len(df_net) < 2:
        return None, None, None, None
    
    current_value = df_net['value'].iloc[-1]
    current_date = df_net['date'].iloc[-1]
    
    # Calculate months to retirement
    months_to_retirement = (target_year - current_date.year) * 12 - current_date.month
    if months_to_retirement <= 0:
        return None, None, None, None
    
    # Generate future dates (monthly)
    future_dates = pd.date_range(
        start=current_date + pd.DateOffset(months=1),
        periods=months_to_retirement,
        freq='ME'
    )
    
    # 1. CONSERVATIVE: S&P 500 baseline (7% annual real returns)
    sp500_annual_rate = 0.07
    sp500_monthly_rate = (1 + sp500_annual_rate) ** (1/12) - 1
    conservative = [current_value * ((1 + sp500_monthly_rate) ** (i + 1)) for i in range(months_to_retirement)]
    
    # 2. CURRENT PACE: Calculate actual historical growth rate
    # Use ALL monthly data, not just quarters
    df_sorted = df_net.sort_values('date').copy()
    
    # Calculate total time period in years
    start_date = df_sorted['date'].iloc[0]
    end_date = df_sorted['date'].iloc[-1]
    years_elapsed = (end_date - start_date).days / 365.25
    
    if years_elapsed > 0:
        start_value = df_sorted['value'].iloc[0]
        end_value = df_sorted['value'].iloc[-1]
        
        # Calculate CAGR (Compound Annual Growth Rate)
        cagr = (end_value / start_value) ** (1 / years_elapsed) - 1
        
        # Convert to monthly rate
        monthly_growth_rate = (1 + cagr) ** (1/12) - 1
    else:
        monthly_growth_rate = sp500_monthly_rate  # Fallback
    
    current_pace = [current_value * ((1 + monthly_growth_rate) ** (i + 1)) for i in range(months_to_retirement)]
    
    # 3. OPTIMISTIC: What rate do we need to hit 1.5x target?
    optimistic_target = target_amount * 1.5
    years_to_retirement = months_to_retirement / 12
    required_annual_rate = (optimistic_target / current_value) ** (1 / years_to_retirement) - 1
    required_monthly_rate = (1 + required_annual_rate) ** (1/12) - 1
    optimistic = [current_value * ((1 + required_monthly_rate) ** (i + 1)) for i in range(months_to_retirement)]
    
    return future_dates, conservative, current_pace, optimistic

def calculate_confidence_score(df_net, target_amount, target_year=2042):
    """
    Calculate probability of hitting retirement goal based on:
    - Current pace projection vs target
    - Historical volatility
    - Time remaining
    """
    if df_net.empty or len(df_net) < 2:
        return 50.0, "Insufficient data"
    
    current_value = df_net['value'].iloc[-1]
    current_date = df_net['date'].iloc[-1]
    months_remaining = (target_year - current_date.year) * 12 - current_date.month
    
    if months_remaining <= 0:
        return 100.0 if current_value >= target_amount else 0.0, "At retirement date"
    
    # Calculate historical CAGR
    df_sorted = df_net.sort_values('date').copy()
    start_date = df_sorted['date'].iloc[0]
    end_date = df_sorted['date'].iloc[-1]
    years_elapsed = (end_date - start_date).days / 365.25
    
    if years_elapsed > 0:
        start_value = df_sorted['value'].iloc[0]
        end_value = df_sorted['value'].iloc[-1]
        cagr = (end_value / start_value) ** (1 / years_elapsed) - 1
        monthly_rate = (1 + cagr) ** (1/12) - 1
    else:
        return 50.0, "Insufficient data"
    
    # Project current pace to 2042
    projected_value = current_value * ((1 + monthly_rate) ** months_remaining)
    
    # Simple confidence calculation
    if projected_value >= target_amount * 1.2:
        # Well above target
        confidence = 95.0
        method = "On track - exceeding target"
    elif projected_value >= target_amount:
        # Above target
        confidence = 85.0
        method = "On track - meeting target"
    elif projected_value >= target_amount * 0.8:
        # Close to target
        confidence = 70.0
        method = "Close - minor adjustment may help"
    elif projected_value >= target_amount * 0.6:
        # Below target
        confidence = 50.0
        method = "Below target - adjustment needed"
    else:
        # Well below target
        confidence = 30.0
        method = "Significant adjustment needed"
    
    # Adjust for volatility
    df_sorted['monthly_return'] = df_sorted['value'].pct_change()
    volatility = df_sorted['monthly_return'].std()
    
    if pd.notna(volatility) and volatility > 0:
        # Higher volatility = lower confidence
        volatility_adjustment = min(15, volatility * 100)
        confidence = max(5, confidence - volatility_adjustment)
    
    return round(confidence, 1), method
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
# --------------------- TOP RETIREMENT GOAL -------------------------
# ------------------------------------------------------------------
if not df.empty:
    cur_total = df_net["value"].iloc[-1]
    retirement_target = get_retirement_goal()
    progress_pct = (cur_total / retirement_target) * 100
    years_remaining = 2042 - datetime.now().year
    
    # Calculate confidence score
    confidence, confidence_method = calculate_confidence_score(df_net, retirement_target, 2042)
    
    st.markdown("# 🎯 RETIREMENT 2042")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric("Current Net Worth (Sean + Kim)", f"${cur_total:,.0f}")
    with col2:
        st.metric("Target", f"${retirement_target:,.0f}")
    with col3:
        st.metric("Progress", f"{progress_pct:.1f}%")
    with col4:
        # Color code confidence
        if confidence >= 80:
            st.metric("Confidence", f"{confidence:.0f}%", delta="On track", delta_color="normal")
        elif confidence >= 60:
            st.metric("Confidence", f"{confidence:.0f}%", delta="Monitor", delta_color="off")
        else:
            st.metric("Confidence", f"{confidence:.0f}%", delta="Action needed", delta_color="inverse")
    
    # Progress bar with color coding
    if progress_pct >= 100:
        st.success(f"🎉 Goal achieved! You're at {progress_pct:.1f}% of target!")
    elif confidence >= 80:
        st.progress(min(progress_pct / 100, 1.0))
        st.success(f"✅ On track! {years_remaining} years remaining • {confidence:.0f}% confidence")
    elif confidence >= 60:
        st.progress(min(progress_pct / 100, 1.0))
        st.warning(f"⚠️ Watch closely • {years_remaining} years remaining • {confidence:.0f}% confidence")
    else:
        st.progress(min(progress_pct / 100, 1.0))
        st.error(f"🚨 Adjustment needed • {years_remaining} years remaining • {confidence:.0f}% confidence")
    
    # Goal adjustment slider
    st.markdown("#### Adjust Retirement Goal")
    new_target = st.slider(
        "Target Amount",
        min_value=500000,
        max_value=5000000,
        value=int(retirement_target),
        step=50000,
        format="$%d",
        help="Adjust your retirement goal - SAGE will recalculate projections"
    )
    
    if new_target != retirement_target:
        set_retirement_goal(new_target)
        st.success(f"Goal updated to ${new_target:,.0f}!")
        st.rerun()
    
    st.markdown("---")
    
    # PROJECTION CONE GRAPH
    st.markdown("## 📊 Projection Cone: Sept 2020 → Dec 2042")
    future_dates, conservative, current_pace, optimistic = calculate_projection_cone(df_net, retirement_target, 2042)
    
    if future_dates is not None:
        fig_cone = go.Figure()
        
        # Historical data (Sean + Kim)
        fig_cone.add_trace(go.Scatter(
            x=df_net['date'],
            y=df_net['value'],
            name='Historical (Sean + Kim)',
            line=dict(color='#AB63FA', width=4),
            mode='lines'
        ))
        
        # Conservative projection (S&P 500)
        fig_cone.add_trace(go.Scatter(
            x=future_dates,
            y=conservative,
            name='Conservative (S&P 7%)',
            line=dict(color='#FFA15A', width=2, dash='dot'),
            mode='lines'
        ))
        
        # Current pace projection
        fig_cone.add_trace(go.Scatter(
            x=future_dates,
            y=current_pace,
            name='Current Pace',
            line=dict(color='#00CC96', width=3),
            mode='lines'
        ))
        
        # Optimistic projection
        fig_cone.add_trace(go.Scatter(
            x=future_dates,
            y=optimistic,
            name='Optimistic (1.5x Target)',
            line=dict(color='#636EFA', width=2, dash='dash'),
            mode='lines'
        ))
        
        # Add target line at 2042
        fig_cone.add_hline(
            y=retirement_target,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: ${retirement_target:,.0f}",
            annotation_position="right"
        )
        
        # Add shaded confidence interval
        fig_cone.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=conservative + optimistic[::-1],
            fill='toself',
            fillcolor='rgba(0,100,250,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Confidence Range'
        ))
        
        fig_cone.update_layout(
            title=f"Retirement Projection Cone • {confidence:.0f}% Confidence",
            xaxis_title="Date",
            yaxis_title="Net Worth ($)",
            hovermode="x unified",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_cone, use_container_width=True)
        
        # Projection summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Conservative (2042)", f"${conservative[-1]:,.0f}", 
                     delta=f"{((conservative[-1]/retirement_target - 1) * 100):+.0f}% vs target")
        with col2:
            st.metric("Current Pace (2042)", f"${current_pace[-1]:,.0f}",
                     delta=f"{((current_pace[-1]/retirement_target - 1) * 100):+.0f}% vs target")
        with col3:
            st.metric("Optimistic (2042)", f"${optimistic[-1]:,.0f}",
                     delta=f"{((optimistic[-1]/retirement_target - 1) * 100):+.0f}% vs target")
    
    st.markdown("---")
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
# SIDEBAR – FULLY INTACT
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
        st.subheader("Bulk Import - Excel Format")
        excel_file = st.file_uploader(
            "Upload your historical Excel data (Date, Sean, Kim, TSP, T3W, Roth, Trl IRA, Stocks, Taylor columns)",
            type=["csv", "xlsx"],
            key="excel_import"
        )
        if excel_file:
            try:
                if excel_file.name.endswith('.xlsx'):
                    df_import = pd.read_excel(excel_file)
                else:
                    df_import = pd.read_csv(excel_file)
                
                imported, errors = import_excel_format(df_import)
                
                if imported > 0:
                    st.success(f"✅ Imported {imported} records!")
                if errors:
                    st.warning(f"⚠️ {len(errors)} errors occurred")
                    with st.expander("View Errors"):
                        for err in errors[:10]:  # Show first 10 errors
                            st.text(err)
                
                if imported > 0:
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
        
        st.markdown("---")
        st.subheader("Bulk Import - Standard Format")
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
    st.subheader("Backup & Restore")

    if st.button("Download Full Database Backup (.dump)", type="primary", use_container_width=True):
        with st.spinner("Creating complete backup..."):
            try:
                conn_url = engine.url
                host = conn_url.host
                port = conn_url.port or 5432
                dbname = conn_url.database
                user = conn_url.username
                password = str(conn_url.password) if conn_url.password else ""

                with NamedTemporaryFile(delete=False, suffix=".dump") as tmpfile:
                    dump_path = tmpfile.name

                    cmd = [
                        "pg_dump",
                        f"--host={host}",
                        f"--port={port}",
                        f"--username={user}",
                        f"--dbname={dbname}",
                        "--format=custom",
                        "--compress=9",
                        "--verbose",
                        "--no-owner",
                        "--no-acl",
                        f"--file={dump_path}"
                    ]

                    env = os.environ.copy()
                    env["PGPASSWORD"] = password

                    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)

                    if result.returncode != 0:
                        st.error(f"Backup failed:\n{result.stderr}")
                    else:
                        with open(dump_path, "rb") as f:
                            st.download_button(
                                label=f"Download Backup – {datetime.now().strftime('%Y-%m-%d')}.dump",
                                data=f,
                                file_name=f"sage-backup-{datetime.now().strftime('%Y-%m-%d')}.dump",
                                mime="application/octet-stream",
                                type="secondary",
                                use_container_width=True
                            )
                        st.success("Backup ready!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if 'dump_path' in locals() and os.path.exists(dump_path):
                    os.unlink(dump_path)

    st.markdown("#### Restore from Backup")
    restore_file = st.file_uploader("Upload a .dump file", type=["dump"], key="restore")
    if restore_file and st.button("Restore Database (OVERWRITES ALL)", type="secondary"):
        if st.checkbox("I understand this will overwrite everything", key="confirm_restore"):
            with st.spinner("Restoring..."):
                try:
                    with NamedTemporaryFile(delete=False) as tmpfile:
                        tmpfile.write(restore_file.getvalue())
                        restore_path = tmpfile.name

                    cmd = [
                        "pg_restore",
                        "--clean", "--if-exists", "--no-owner", "--no-acl",
                        f"--host={engine.url.host}",
                        f"--port={engine.url.port or 5432}",
                        f"--username={engine.url.username}",
                        f"--dbname={engine.url.database}",
                        restore_path
                    ]
                    env = os.environ.copy()
                    env["PGPASSWORD"] = str(engine.url.password or "")
                    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
                    if result.returncode == 0:
                        st.success("Restore complete! Refreshing...")
                        st.balloons()
                    else:
                        st.error(f"Restore failed:\n{result.stderr}")
                except Exception as e:
                    st.error(f"Error: {e}")

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

# ------------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = load_ai_history()
if "ai_chat_session" not in st.session_state:
    st.session_state.ai_chat_session = None

# ------------------- AI CHAT PAGE (FULLY INTACT) -------------------
if st.session_state.page == "ai":
    st.subheader("S.A.G.E. | Strategic Asset Growth Engine")
    st.caption("Let's review, refine, and grow — together.")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("Add `GOOGLE_API_KEY` in Streamlit Secrets to enable S.A.G.E.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
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

        if not st.session_state.ai_messages and not df_port.empty:
            retirement_target = get_retirement_goal()
            years_to_retirement = 2042 - datetime.now().year
            init_prompt = f"""Here's our current situation:
            
**Retirement Goal**: ${retirement_target:,.0f} by 2042 ({years_to_retirement} years remaining)
**Current Net Worth (Sean + Kim)**: ${df_net['value'].iloc[-1]:,.0f}
**Progress**: {(df_net['value'].iloc[-1] / retirement_target) * 100:.1f}%

**Portfolio**: Loaded {len(df_port)} holdings from Fidelity
**Total Portfolio Value**: ${port_summary.get('total_value', 0):,.0f}
**Total Gain**: ${port_summary.get('total_gain', 0):,.0f} ({port_summary.get('total_gain_pct', 0):.1f}%)

What's your initial analysis? Are we on track?"""
            with st.spinner("S.A.G.E. is analyzing your full picture..."):
                response = chat.send_message(init_prompt)
                reply = response.text
            st.session_state.ai_messages.append({"role": "user", "content": init_prompt})
            save_ai_message("user", init_prompt)
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply)
            st.rerun()

        for msg in st.session_state.ai_messages:
            role = "assistant" if msg["role"] == "model" else "user"
            with st.chat_message(role):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask S.A.G.E.: rebalance? risk? taxes? retirement?")
        if user_input:
            st.session_state.ai_messages.append({"role": "user", "content": user_input})
            save_ai_message("user", user_input)
            with st.spinner("S.A.G.E. is thinking..."):
                response = chat.send_message(user_input)
                reply = response.text
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply)
            st.rerun()

    if st.button("Back to Dashboard"):
        st.session_state.page = "home"
        st.rerun()
        # ------------------- HOME DASHBOARD (YOUR FULL BEAUTIFUL VERSION) -------------------
else:
    if df.empty:
        st.info("Upload your Fidelity CSV and add a monthly update. S.A.G.E. is ready when you are.")
        st.stop()

tab1, tab2 = st.tabs(["Retirement (Sean + Kim)", "Taylor's Nest Egg"])

    with tab1:
        # Sean + Kim wealth journey graph
        df_pivot = df.pivot_table(index="date", columns="person", values="value", aggfunc="sum") \
                      .resample("ME").last().ffill().fillna(0)
        
        # Only Sean + Kim for this tab
        df_sean_kim = df_pivot[['Sean', 'Kim']].copy() if 'Sean' in df_pivot.columns and 'Kim' in df_pivot.columns else df_pivot
        df_sean_kim["Sean + Kim"] = df_sean_kim.get("Sean", 0) + df_sean_kim.get("Kim", 0)

        fig = go.Figure()
        colors = {"Sean": "#636EFA", "Kim": "#EF553B", "Sean + Kim": "#AB63FA"}
        width = {"Sean + Kim": 5, "Sean": 3, "Kim": 3}

        for person in ["Sean", "Kim", "Sean + Kim"]:
            if person in df_sean_kim.columns:
                fig.add_trace(go.Scatter(
                    x=df_sean_kim.index,
                    y=df_sean_kim[person],
                    name=person,
                    line=dict(color=colors[person], width=width.get(person, 3)),
                    hovertemplate=f"<b>{person}</b><br>%{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>"
                ))

        fig.update_layout(
            title="Retirement Portfolio Growth (Sean + Kim)",
            xaxis_title="Date",
            yaxis_title="Total Value ($)",
            hovermode="x unified",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📈 Month-over-Month Analysis")
        
        # Calculate MoM changes ($ and %)
        mom_dollar = df_sean_kim.diff().round(0)
        mom_pct = df_sean_kim.pct_change() * 100
        
        # Get available years
        available_years = sorted(df_sean_kim.index.year.unique(), reverse=True)
        
        # Create tabs for each year
        year_tabs = st.tabs([str(year) for year in available_years])
        
        for year_tab, year in zip(year_tabs, available_years):
            with year_tab:
                # Filter data for this year
                year_mask = df_sean_kim.index.year == year
                year_data_dollar = mom_dollar[year_mask]
                year_data_pct = mom_pct[year_mask]
                
                if year_data_dollar.empty:
                    st.info(f"No data for {year}")
                    continue
                
                # Build display DataFrame
                display_data = []
                for date in year_data_dollar.index:
                    row = {'Date': date.strftime('%b %Y')}
                    
                    for person in ['Sean', 'Kim', 'Sean + Kim']:
                        if person in year_data_dollar.columns:
                            dollar_val = year_data_dollar.loc[date, person]
                            pct_val = year_data_pct.loc[date, person]
                            
                            if pd.notna(dollar_val) and pd.notna(pct_val):
                                row[f'{person} $'] = dollar_val
                                row[f'{person} %'] = pct_val
                            else:
                                row[f'{person} $'] = 0
                                row[f'{person} %'] = 0
                    
                    display_data.append(row)
                
                df_display = pd.DataFrame(display_data)
                
                # Style function for color coding
                def color_negative_red(val):
                    if isinstance(val, (int, float)):
                        color = '#90EE90' if val > 0 else ('#FF6B6B' if val < 0 else 'white')
                        return f'background-color: {color}; color: black'
                    return ''
                
                # Apply styling
                styled_df = df_display.style.applymap(
                    color_negative_red,
                    subset=[col for col in df_display.columns if col != 'Date']
                ).format({
                    col: '${:,.0f}' if '$' in col else '{:+.2f}%'
                    for col in df_display.columns if col != 'Date'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("---")
        st.markdown("## 📅 Year-over-Year (December to December)")
        
        # Get December data for each year
        december_data = df_sean_kim[df_sean_kim.index.month == 12].copy()
        
        if len(december_data) >= 2:
            yoy_data = []
            years = sorted(december_data.index.year.unique())
            
            for i in range(len(years) - 1):
                prev_year = years[i]
                curr_year = years[i + 1]
                
                prev_data = december_data[december_data.index.year == prev_year].iloc[0]
                curr_data = december_data[december_data.index.year == curr_year].iloc[0]
                
                row = {'Period': f'Dec {prev_year} → Dec {curr_year}'}
                
                for person in ['Sean', 'Kim', 'Sean + Kim']:
                    if person in prev_data.index and person in curr_data.index:
                        prev_val = prev_data[person]
                        curr_val = curr_data[person]
                        
                        if prev_val > 0:
                            dollar_change = curr_val - prev_val
                            pct_change = ((curr_val / prev_val) - 1) * 100
                            
                            row[f'{person} $'] = dollar_change
                            row[f'{person} %'] = pct_change
                
                yoy_data.append(row)
            
            df_yoy = pd.DataFrame(yoy_data)
            
            # Style YoY data
            styled_yoy = df_yoy.style.applymap(
                color_negative_red,
                subset=[col for col in df_yoy.columns if col != 'Period']
            ).format({
                col: '${:,.0f}' if '$' in col else '{:+.2f}%'
                for col in df_yoy.columns if col != 'Period'
            })
            
            st.dataframe(styled_yoy, use_container_width=True)
        else:
            st.info("Need at least 2 years of December data for YoY comparison")

    with tab2:
        st.markdown("# 💎 Taylor's Nest Egg")
        st.caption("Building long-term wealth for Taylor's future")
        
        taylor_df = df[df["person"] == "Taylor"].sort_values("date")
        
        if not taylor_df.empty:
            taylor_current = taylor_df["value"].iloc[-1]
            taylor_start = taylor_df["value"].iloc[0]
            taylor_growth = ((taylor_current / taylor_start) - 1) * 100 if taylor_start > 0 else 0
            
            # Calculate Taylor's CAGR
            taylor_start_date = taylor_df['date'].iloc[0]
            taylor_end_date = taylor_df['date'].iloc[-1]
            taylor_years = (taylor_end_date - taylor_start_date).days / 365.25
            
            if taylor_years > 0 and taylor_start > 0:
                taylor_cagr = ((taylor_current / taylor_start) ** (1 / taylor_years) - 1) * 100
            else:
                taylor_cagr = 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Value", f"${taylor_current:,.0f}")
            col2.metric("Total Growth", f"{taylor_growth:+.1f}%")
            col3.metric("Annual Growth Rate (CAGR)", f"{taylor_cagr:.1f}%")
            
            # Taylor's growth chart
            fig_taylor = go.Figure()
            fig_taylor.add_trace(go.Scatter(
                x=taylor_df["date"],
                y=taylor_df["value"],
                name="Taylor's Portfolio",
                line=dict(color="#00CC96", width=3),
                fill='tozeroy',
                fillcolor='rgba(0,204,150,0.1)',
                hovertemplate="<b>Taylor</b><br>%{x|%b %Y}: $%{y:,.0f}<extra></extra>"
            ))
            
            fig_taylor.update_layout(
                title="Taylor's Portfolio Growth Over Time",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_taylor, use_container_width=True)
            
            st.markdown("---")
            st.markdown("## 📊 Taylor's Month-over-Month")
            
            # Taylor MoM table
            taylor_pivot = taylor_df.set_index('date')['value'].resample('ME').last().to_frame()
            taylor_mom_dollar = taylor_pivot.diff().round(0)
            taylor_mom_pct = taylor_pivot.pct_change() * 100
            
            # Get Taylor's years
            taylor_years = sorted(taylor_pivot.index.year.unique(), reverse=True)
            taylor_year_tabs = st.tabs([str(year) for year in taylor_years])
            
            for year_tab, year in zip(taylor_year_tabs, taylor_years):
                with year_tab:
                    year_mask = taylor_pivot.index.year == year
                    year_dollar = taylor_mom_dollar[year_mask]
                    year_pct = taylor_mom_pct[year_mask]
                    
                    if year_dollar.empty:
                        st.info(f"No data for {year}")
                        continue
                    
                    display_data = []
                    for date in year_dollar.index:
                        dollar_val = year_dollar.loc[date, 'value']
                        pct_val = year_pct.loc[date, 'value']
                        
                        if pd.notna(dollar_val) and pd.notna(pct_val):
                            display_data.append({
                                'Date': date.strftime('%b %Y'),
                                'Change $': dollar_val,
                                'Change %': pct_val
                            })
                    
                    df_taylor_display = pd.DataFrame(display_data)
                    
                    def color_negative_red(val):
                        if isinstance(val, (int, float)):
                            color = '#90EE90' if val > 0 else ('#FF6B6B' if val < 0 else 'white')
                            return f'background-color: {color}; color: black'
                        return ''
                    
                    styled_taylor = df_taylor_display.style.applymap(
                        color_negative_red,
                        subset=['Change $', 'Change %']
                    ).format({
                        'Change $': '${:,.0f}',
                        'Change %': '{:+.2f}%'
                    })
                    
                    st.dataframe(styled_taylor, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🎯 Long-Term Outlook for Taylor")
            st.info(f"""
            **Taylor is {2025 - 2021} years old.** Time is her greatest asset.
            
            At her current growth rate of {taylor_cagr:.1f}% annually:
            - By age 18 (2039): ~${taylor_current * ((1 + taylor_cagr/100) ** 14):,.0f}
            - By age 30 (2051): ~${taylor_current * ((1 + taylor_cagr/100) ** 26):,.0f}
            - By age 40 (2061): ~${taylor_current * ((1 + taylor_cagr/100) ** 36):,.0f}
            
            Let compounding work its magic. 🚀
            """)
            
        else:
            st.info("No data for Taylor yet. Add her first monthly update to get started!")

    st.download_button(
        "Export All Monthly Data",
        df.to_csv(index=False).encode(),
        f"sage-data-{datetime.now().strftime('%Y-%m-%d')}.csv",
        "text/csv"
    )
