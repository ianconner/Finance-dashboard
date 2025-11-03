import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

# AI/ML imports
from pmdarima import auto_arima  # New for auto ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ---------- SQLAlchemy (persistent DB) ----------
from sqlalchemy import (
    create_engine, Column, String, Float, Date, text,
    PrimaryKeyConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# ---------- Database Setup ----------
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
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# ---------- Helper DB Functions ----------
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
        if row.person not in accounts:
            accounts[row.person] = []
        accounts[row.person].append(row.account_type)
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

def add_account_type(person, acc_type):
    sess = get_session()
    sess.merge(AccountConfig(person=person, account_type=acc_type))
    sess.commit()
    sess.close()

def add_monthly_update(date, person, acc_type, value):
    sess = get_session()
    sess.merge(MonthlyUpdate(date=date, person=person, account_type=acc_type, value=value))
    sess.commit()
    sess.close()

def add_contribution(date, person, acc_type, amount):
    sess = get_session()
    sess.merge(Contribution(date=date, person=person, account_type=acc_type, contribution=amount))
    sess.commit()
    sess.close()

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

# ---------- ONE-TIME CSV UPLOAD TO SEED DATABASE ----------
def seed_database_from_csv(df_uploaded):
    try:
        sess = get_session()
        count = sess.query(MonthlyUpdate).count()
        if count > 0:
            st.info("Database already has data — skipping seed.")
            return
        
        for _, row in df_uploaded.iterrows():
            date = pd.to_datetime(row['date']).date()
            person = str(row['person'])
            account_type = str(row['account_type'])
            value = float(row['value'])
            sess.merge(MonthlyUpdate(date=date, person=person, account_type=account_type, value=value))
        
        sess.commit()
        sess.close()
        st.success(f"Seeded {len(df_uploaded)} rows into database!")
    except Exception as e:
        st.error(f"Seed failed: {e}")

# ---------- AI ANALYTICS: Growth Projections ----------
def ai_projections(df_net, horizon=24):
    if len(df_net) < 3:
        return None, None, None, None, None
    df_net['time_idx'] = range(len(df_net))
    y = df_net['value'].values
    X = df_net['time_idx'].values.reshape(-1, 1)

    # Improved ARIMA with auto_arima
    try:
        auto_model = auto_arima(y, seasonal=False, suppress_warnings=True)
        model = ARIMA(y, order=auto_model.order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        ci = fitted.get_forecast(steps=horizon).conf_int()
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
    except:
        forecast = np.full(horizon, y[-1])
        lower = np.full(horizon, y[-1]*0.9)
        upper = np.full(horizon, y[-1]*1.1)

    # Linear Regression
    lr = LinearRegression().fit(X, y)
    future_x = np.array(range(len(df_net), len(df_net) + horizon)).reshape(-1, 1)
    lr_pred = lr.predict(future_x)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    rf_pred = rf.predict(future_x)

    # LSTM
    try:
        y_scaled = (y - y.min()) / (y.max() - y.min())
        X_lstm = y_scaled[:-1].reshape(-1, 1, 1)
        y_lstm = y_scaled[1:]
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_lstm, y_lstm, epochs=100, verbose=0)
        lstm_pred = []
        last= y_scaled[-1].reshape(1, 1, 1)
        for _ in range(horizon):
            pred = model.predict(last, verbose=0)[0][0]
            lstm_pred.append(pred)
            last = np.array(pred).reshape(1, 1, 1)
        lstm_pred = np.array(lstm_pred) * (y.max() - y.min()) + y.min()
    except:
        lstm_pred = np.full(horizon, y[-1])

    return forecast, lower, upper, lr_pred, rf_pred, lstm_pred

# ---------- UI Starts Here ----------
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Tracker")

# Load data
df = get_monthly_updates()
df_contrib = get_contributions()

# ----- ONE-TIME CSV SEED -----
if df.empty:
    st.subheader("Seed Database with CSV (One-Time Setup)")
    uploaded_file = st.file_uploader("Upload your old CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            expected_cols = ['date', 'person', 'account_type', 'value']
            if all(col in df_upload.columns for col in expected_cols):
                if st.button("Import CSV to Database"):
                    seed_database_from_csv(df_upload)
                    st.rerun()
            else:
                st.error(f"CSV must have columns: {expected_cols}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ----- Sidebar -----
with st.sidebar:
    st.subheader("Add Monthly Update")
    accounts_dict = load_accounts()
    persons = list(accounts_dict.keys())
    person = st.selectbox("Person", persons, key="person_selector")

    acct_opts = accounts_dict.get(person, [])
    if not acct_opts:
        acct_opts = ["TSP", "T3W", "Stocks"] if person == "Sean" else ["Kim Total"]
        st.info(f"No accounts for **{person}** yet – defaults shown.")
    account_type = st.selectbox("Account type", acct_opts, key=f"acct_{person}")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        value = st.number_input("Value ($)", min_value=0.0, format="%.2f")

    if st.button("Save entry"):
        add_monthly_update(date, person, account_type, float(value))
        st.success(f"Saved {person} → {account_type} = ${value:,.2f}")
        st.rerun()

    st.subheader("Add Contribution")
    contrib_amount = st.number_input("Contribution ($)", min_value=0.0, format="%.2f", key="contrib_amt")
    if st.button("Save contribution"):
        add_contribution(date, person, account_type, float(contrib_amount))
        st.success(f"Contribution ${contrib_amount:,.2f} saved")
        st.rerun()

    # ----- Add New Person -----
    st.subheader("Add New Person")
    new_person = st.text_input("New Person Name")
    if st.button("Add Person"):
        if new_person.strip():
            add_person(new_person.strip())
            st.success(f"Added {new_person.strip()}!")
            st.rerun()
        else:
            st.error("Enter a name")

    # ----- Add New Account Type -----
    st.subheader("Add New Account Type")
    new_acct_person = st.selectbox("For Person", persons, key="new_acct_person")
    new_acct_type = st.text_input("New Account Type")
    if st.button("Add Account Type"):
        if new_acct_type.strip():
            add_account_type(new_acct_person, new_acct_type.strip())
            st.success(f"Added {new_acct_type.strip()} for {new_acct_person}!")
            st.rerun()
        else:
            st.error("Enter an account type")

    # Admin Reset
    if st.button("Reset & Re-Seed (Admin Only)"):
        reset_database()
        st.rerun()

# ----- Main Data Display -----
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    if not df_contrib.empty:
        df_contrib["date"] = pd.to_datetime(df_contrib["date"])

    # Collapsible Monthly Summary by Year
    st.subheader("Monthly Summary (By Year)")
    df['year'] = df["date"].dt.year
    years = sorted(df['year'].unique(), reverse=True)
    for year in years:
        with st.expander(f"Year {year}"):
            df_year = df[df['year'] == year]
            pivot = df_year.pivot_table(
                index="date",
                columns=["person", "account_type"],
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            st.dataframe(pivot.style.format("${:,.0f}"))

    # Tabbed Gain/Loss Graphs
    st.subheader("Account Gain/Loss Views")
    tabs = st.tabs(["YTD Gain/Loss", "Month-to-Month Gain/Loss"])

    with tabs[0]:
        st.write("Year-to-Date Gain/Loss (Cumulative from Jan 1)")
        for year in years:
            df_year = df[df['year'] == year]
            df_year = df_year.sort_values("date")
            df_year['month'] = df_year["date"].dt.month
            df_year['ytd'] = df_year.groupby(['person', 'account_type'])['value'].cumsum()
            fig_ytd = px.line(df_year, x="month", y="ytd", color="person", line_group="account_type", title=f"{year} YTD Gain/Loss")
            st.plotly_chart(fig_ytd, use_container_width=True)

    with tabs[1]:
        st.write("Month-to-Month Gain/Loss")
        df_monthly = df.sort_values("date")
        df_monthly['prev_value'] = df_monthly.groupby(['person', 'account_type'])['value'].shift(1)
        df_monthly['monthly_gain'] = df_monthly['value'] - df_monthly['prev_value'].fillna(0)
        fig_monthly = px.bar(df_monthly, x="date", y="monthly_gain", color="person", barmode="group", title="Month-to-Month Gain/Loss")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # [Your existing AI Projections, Goals, Growth Rates, Delete, Export - keep as is]

# Render Port Fix (for Render hosting)
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    st.run_server(port=port, host="0.0.0.0")
