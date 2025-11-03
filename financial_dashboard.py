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

def add_person(name):
    sess = get_session()
    sess.merge(AccountConfig(person=name, account_type="Personal"))
    sess.commit()
    sess.close()

def add_account_type(person, acct_type):
    sess = get_session()
    sess.merge(AccountConfig(person=person, account_type=acct_type))
    sess.commit()
    sess.close()

def add_monthly_update(date, person, acct, value):
    sess = get_session()
    sess.merge(MonthlyUpdate(date=date, person=person, account_type=acct, value=value))
    sess.commit()
    sess.close()

def add_contribution(date, person, acct, amount):
    sess = get_session()
    sess.merge(Contribution(date=date, person=person, account_type=acct, contribution=amount))
    sess.commit()
    sess.close()

def get_monthly_updates():
    sess = get_session()
    rows = sess.query(MonthlyUpdate).all()
    sess.close()
    return pd.DataFrame([
        {'date': r.date, 'person': r.person, 'account_type': r.account_type, 'value': r.value}
        for r in rows
    ])

def get_contributions():
    sess = get_session()
    rows = sess.query(Contribution).all()
    sess.close()
    return pd.DataFrame([
        {'date': r.date, 'person': r.person, 'account_type': r.account_type, 'contribution': r.contribution}
        for r in rows
    ])

def seed_database_from_csv(df):
    sess = get_session()
    if sess.query(MonthlyUpdate).count() > 0:
        st.info("Already seeded.")
        return
    for _, r in df.iterrows():
        sess.merge(MonthlyUpdate(
            date=pd.to_datetime(r['date']).date(),
            person=str(r['person']),
            account_type=str(r['account_type']),
            value=float(r['value'])
        ))
    sess.commit()
    sess.close()
    st.success(f"Seeded {len(df)} rows!")

# ---------- AI PROJECTIONS ----------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3:
        return None, None, None, None, None
    df_net['time_idx'] = range(len(df_net))
    y = df_net['value'].values
    X = df_net['time_idx'].values.reshape(-1, 1)

    try:
        model = ARIMA(y, order=(1,1,1)).fit()
        forecast = model.forecast(steps=horizon)
        ci = model.get_forecast(steps=horizon).conf_int()
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    except:
        forecast = np.full(horizon, y[-1])
        lower = np.full(horizon, y[-1] * 0.9)
        upper = np.full(horizon, y[-1] * 1.1)

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
    st.subheader("Seed Database (One-Time)")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded and st.button("Import"):
        df_up = pd.read_csv(uploaded)
        if all(c in df_up.columns for c in ['date', 'person', 'account_type', 'value']):
            seed_database_from_csv(df_up)
            st.rerun()
        else:
            st.error("CSV needs: date, person, account_type, value")

# Sidebar
with st.sidebar:
    st.subheader("Add Monthly Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    account_type = st.selectbox("Account", accounts[person])

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", pd.Timestamp("today").date())
    with col2:
        value = st.number_input("Value ($)", 0.0, format="%.2f")

    if st.button("Save Update"):
        add_monthly_update(date, person, account_type, value)
        st.success("Saved!")
        st.rerun()

    st.subheader("Add Contribution")
    contrib = st.number_input("Amount ($)", 0.0, format="%.2f")
    if st.button("Save Contrib"):
        add_contribution(date, person, account_type, contrib)
        st.success("Saved!")
        st.rerun()

    st.subheader("Add New Person")
    new_person = st.text_input("Name")
    if st.button("Add Person") and new_person.strip():
        add_person(new_person.strip())
        st.success(f"Added {new_person.strip()}!")
        st.rerun()

    st.subheader("Add New Account Type")
    acct_person = st.selectbox("For Person", list(accounts.keys()), key="acct_p")
    new_acct = st.text_input("Account Name")
    if st.button("Add Account") and new_acct.strip():
        add_account_type(acct_person, new_acct.strip())
        st.success(f"Added {new_acct.strip()}!")
        st.rerun()

    if st.button("Reset DB (Admin)"):
        reset_database()
        st.rerun()

# Main Content
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns=["person", "account_type"], values="value", aggfunc="sum", fill_value=0)
    st.subheader("Monthly Summary")
    st.dataframe(pivot.style.format("${:,.0f}"))

    # Net Worth
    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index().sort_values("date")
    st.subheader("Family Net Worth (Sean + Kim)")
    fig = px.line(df_net, x="date", y="value", title="Net Worth Over Time", labels={"value": "$"})
    fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # AI Projections
    st.subheader("AI Growth Projections")
    horizon = st.slider("Forecast (months)", 12, 60, 24)
    arima_f, lower, upper, lr_f, rf_f = ai_projections(df_net, horizon)

    if arima_f is not None:
        future_dates = pd.date_range(start=df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='MS')
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Past", line=dict(color="blue")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_f, name="ARIMA", line=dict(color="green")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=lower, line=dict(color="lightgreen", dash="dash"), showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=upper, fill="tonexty", line=dict(color="lightgreen"), name="95% CI"))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=lr_f, name="Linear", line=dict(color="orange")))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=rf_f, name="Random Forest", line=dict(color="red")))
        fig_proj.update_layout(title=f"AI Forecast ({horizon} months)", yaxis_title="$", xaxis_title="Date")
        st.plotly_chart(fig_proj, use_container_width=True)

        # Safe Summary Table
        proj_data = {
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
        }
        st.dataframe(pd.DataFrame(proj_data))

    # Goals
    st.subheader("Goals")
    goals = [
        {"name": "Millionaire", "target": 1_000_000, "by": "2030"},
        {"name": "Retirement", "target": 2_000_000, "by": "2035"},
        {"name": "Legacy", "target": 3_000_000, "by": "2040"}
    ]
    current = df_net["value"].iloc[-1]
    for g in goals:
        progress = min(current / g["target"], 1.0)
        st.progress(progress)
        st.write(f"**{g['name']}**: ${current:,.0f} / ${g['target']:,.0f} → {g['by']}")

    # Growth Rates
    st.subheader("Monthly Growth")
    latest = df["date"].max()
    prev = latest - pd.DateOffset(months=1)
    if prev in df["date"].values:
        curr = df[df["date"] == latest]
        prev_df = df[df["date"] == prev]
        merged = curr.merge(prev_df, on=["person", "account_type"], suffixes=("_c", "_p"))
        merged["growth"] = (merged["value_c"] - merged["value_p"]) / merged["value_p"]
        growth = merged[["person", "account_type", "growth"]].copy()
        growth["growth"] = (growth["growth"] * 100).round(2)
        st.dataframe(growth.sort_values("growth", ascending=False).style.format({"growth": "{:.2f}%"}))
    else:
        st.info("Need 2 months for growth rates")

    # Delete
    st.subheader("Delete Entry")
    choice = st.selectbox("Select", df.index, format_func=lambda i: f"{df.loc[i,'date']} – {df.loc[i,'person']} – {df.loc[i,'account_type']} – ${df.loc[i,'value']:,.0f}")
    if st.button("Delete"):
        row = df.loc[choice]
        sess = get_session()
        sess.query(MonthlyUpdate).filter_by(date=row["date"], person=row["person"], account_type=row["account_type"]).delete()
        sess.commit()
        sess.close()
        st.success("Deleted!")
        st.rerun()

    # Export
    st.download_button("Export Values", df.to_csv(index=False).encode(), "values.csv", "text/csv")
    if not df_contrib.empty:
        st.download_button("Export Contributions", df_contrib.to_csv(index=False).encode(), "contributions.csv", "text/csv")
