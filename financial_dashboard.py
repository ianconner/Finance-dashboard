import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# SQLAlchemy
from sqlalchemy import create_engine, Column, String, Float, Date, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# ---------- DATABASE ----------
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

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
except Exception as e:
    st.error(f"DB Error: {e}")
    st.stop()

# ---------- HELPERS ----------
def get_session(): return Session()

def reset_database():
    sess = get_session()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    defaults = {'Sean': ['IRA', 'Roth IRA', 'TSP', 'Personal', 'T3W'], 'Kim': ['Retirement'], 'Taylor': ['Personal']}
    for p, types in defaults.items():
        for t in types: sess.merge(AccountConfig(person=p, account_type=t))
    sess.commit(); sess.close()

def load_accounts():
    sess = get_session()
    cfg = sess.query(AccountConfig).all()
    accounts = {row.person: [row.account_type] for row in cfg}
    sess.close()
    if not accounts: reset_database(); return load_accounts()
    return {p: [t for row in cfg if row.person == p for t in [row.account_type]]}

def add_person(name): sess = get_session(); sess.merge(AccountConfig(person=name, account_type="Personal")); sess.commit(); sess.close()
def add_account_type(p, t): sess = get_session(); sess.merge(AccountConfig(person=p, account_type=t)); sess.commit(); sess.close()
def add_monthly_update(d, p, a, v): sess = get_session(); sess.merge(MonthlyUpdate(date=d, person=p, account_type=a, value=v)); sess.commit(); sess.close()
def add_contribution(d, p, a, c): sess = get_session(); sess.merge(Contribution(date=d, person=p, account_type=a, contribution=c)); sess.commit(); sess.close()

def get_monthly_updates():
    sess = get_session()
    rows = sess.query(MonthlyUpdate).all()
    sess.close()
    return pd.DataFrame([{'date': r.date, 'person': r.person, 'account_type': r.account_type, 'value': r.value} for r in rows])

def get_contributions():
    sess = get_session()
    rows = sess.query(Contribution).all()
    sess.close()
    return pd.DataFrame([{'date': r.date, 'person': r.person, 'account_type': r.account_type, 'contribution': r.contribution} for r in rows])

def seed_database_from_csv(df):
    sess = get_session()
    if sess.query(MonthlyUpdate).count() > 0: return
    for _, r in df.iterrows():
        sess.merge(MonthlyUpdate(date=pd.to_datetime(r['date']).date(), person=str(r['person']), account_type=str(r['account_type']), value=float(r['value'])))
    sess.commit(); sess.close()
    st.success("Seeded!")

# ---------- AI PROJECTIONS ----------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3: return None, None, None, None, None
    df_net['time_idx'] = range(len(df_net))
    y, X = df_net['value'].values, df_net['time_idx'].values.reshape(-1, 1)

    try:
        model = ARIMA(y, order=(1,1,1)).fit()
        forecast = model.forecast(steps=horizon)
        ci = model.get_forecast(steps=horizon).conf_int()
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    except:
        forecast = np.full(horizon, y[-1])
        lower = np.full(horizon, y[-1]*0.9)
        upper = np.full(horizon, y[-1]*1.1)

    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    future_x = np.array(range(len(df_net), len(df_net) + horizon)).reshape(-1, 1)
    lr_pred = lr.predict(future_x)
    rf_pred = rf.predict(future_x)

    return forecast, lower, upper, lr_pred, rf_pred

# ---------- UI ----------
st.set_page_config(page_title="Finance AI", layout="wide")
st.title("Personal Finance Tracker")

df = get_monthly_updates()
df_contrib = get_contributions()

# CSV Seed
if df.empty:
    st.subheader("Seed with CSV")
    file = st.file_uploader("Upload CSV", type="csv")
    if file and st.button("Import"):
        df_up = pd.read_csv(file)
        if all(c in df_up.columns for c in ['date', 'person', 'account_type', 'value']):
            seed_database_from_csv(df_up)
            st.rerun()

# Sidebar
with st.sidebar:
    st.subheader("Add Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    acct = st.selectbox("Account", accounts[person])
    col1, col2 = st.columns(2)
    with col1: date = st.date_input("Date", pd.Timestamp("today").date())
    with col2: value = st.number_input("Value ($)", 0.0, format="%.2f")
    if st.button("Save"): add_monthly_update(date, person, acct, value); st.success("Saved!"); st.rerun()

    st.subheader("Add Contribution")
    contrib = st.number_input("Amount ($)", 0.0, format="%.2f")
    if st.button("Save Contrib"): add_contribution(date, person, acct, contrib); st.success("Saved!"); st.rerun()

    st.subheader("Add Person")
    new_p = st.text_input("Name")
    if st.button("Add") and new_p.strip(): add_person(new_p.strip()); st.success("Added!"); st.rerun()

    st.subheader("Add Account Type")
    p_for_acct = st.selectbox("For", list(accounts.keys()), key="p_acct")
    new_a = st.text_input("Type")
    if st.button("Add Acct") and new_a.strip(): add_account_type(p_for_acct, new_a.strip()); st.success("Added!"); st.rerun()

# Main
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns=["person", "account_type"], values="value", aggfunc="sum", fill_value=0)
    st.subheader("Monthly Summary")
    st.dataframe(pivot.style.format("${:,.0f}"))

    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index().sort_values("date")
    st.subheader("Net Worth")
    fig = px.line(df_net, x="date", y="value", title="Sean + Kim", labels={"value": "$"})
    fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # AI
    st.subheader("AI Projections")
    horizon = st.slider("Months", 12, 60, 24)
    arima_f, lower, upper, lr_f, rf_f = ai_projections(df_net, horizon)

    if arima_f is not None:
        future = pd.date_range(start=df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='MS')
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Past", line=dict(color="blue")))
        fig_proj.add_trace(go.Scatter(x=future, y=arima_f, name="ARIMA", line=dict(color="green")))
        fig_proj.add_trace(go.Scatter(x=future, y=lower, line=dict(color="green", dash="dash"), showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future, y=upper, fill="tonexty", line=dict(color="green"), name="95% CI"))
        fig_proj.add_trace(go.Scatter(x=future, y=lr_f, name="Linear", line=dict(color="orange")))
        fig_proj.add_trace(go.Scatter(x=future, y=rf_f, name="Random Forest", line=dict(color="red")))
        fig_proj.update_layout(title=f"AI Forecast ({horizon} months)", yaxis_title="$", xaxis_title="Date")
        st.plotly_chart(fig_proj, use_container_width=True)

        # Safe Table
        proj = pd.DataFrame({
            'Model': ['ARIMA', 'Linear', 'Random Forest'],
            '24 Months': [
                f"${arima_f[23]:,.0f}" if horizon > 23 else "N/A",
                f"${lr_f[23]:,.0f}" if horizon > 23 else "N/A",
                f"${rf_f[23]:,.0f}" if horizon > 23 else "N/A"
            ],
            '60 Months': [
                f"${arima_f[59]:,.0f}" if horizon > 59 else "N/A",
                f"${lr_f[59]:,.0f}" if horizon > 59 else "N/A",
                f"${rf_f[59]:,.0f}" if horizon > 59 else "N/A"
            ]
        })
        st.dataframe(proj)

    # Goals, Growth, Delete, Export (same as before — omitted for brevity)
    # ... [Keep your existing Goals, Growth, Delete, Export code here] ...

# ===== RENDER: CORRECT START COMMAND =====
import os
if "RENDER" in os.environ:
    # Render uses `streamlit run app.py --server.port $PORT`
    # No code needed — just remove st.run_server
    pass
