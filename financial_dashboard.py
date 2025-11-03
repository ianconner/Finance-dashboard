import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

# AI/ML imports
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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
            if row.person not in accounts:
                accounts[row.person] = []
            accounts[row.person].append(row.account_type)
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

def delete_account_type(person, acc_type):
    try:
        sess = get_session()
        sess.query(AccountConfig).filter_by(person=person, account_type=acc_type).delete()
        sess.query(MonthlyUpdate).filter_by(person=person, account_type=acc_type).delete()
        sess.query(Contribution).filter_by(person=person, account_type=acc_type).delete()
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Delete account error: {e}")

def delete_person(person):
    try:
        sess = get_session()
        sess.query(AccountConfig).filter_by(person=person).delete()
        sess.query(MonthlyUpdate).filter_by(person=person).delete()
        sess.query(Contribution).filter_by(person=person).delete()
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Delete person error: {e}")

def add_monthly_update(date, person, acc_type, value):
    try:
        sess = get_session()
        stmt = insert(MonthlyUpdate.__table__).values(date=date, person=person, account_type=acc_type, value=value)
        stmt = stmt.on_conflict_do_update(index_elements=['date', 'person', 'account_type'], set_={'value': value})
        sess.execute(stmt)
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Add update error: {e}")

def add_contribution(date, person, acc_type, amount):
    try:
        sess = get_session()
        stmt = insert(Contribution.__table__).values(date=date, person=person, account_type=acc_type, contribution=amount)
        stmt = stmt.on_conflict_do_update(index_elements=['date', 'person', 'account_type'], set_={'contribution': amount})
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
            {'date': r.date, 'person': r.person,
             'account_type': r.account_type, 'value': r.value}
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
            {'date': r.date, 'person': r.person,
             'account_type': r.account_type, 'contribution': r.contribution}
            for r in rows
        ])
    except Exception as e:
        st.error(f"Get contributions error: {e}")
        return pd.DataFrame()

# ---------- ONE-TIME CSV UPLOAD TO SEED DATABASE ----------
def seed_database_from_csv(df_uploaded):
    try:
        sess = get_session()
        count = sess.query(MonthlyUpdate).count()
        if count > 0:
            st.info("Database already has data — skipping seed.")
            return
        
        for _, row in df_uploaded.iterrows():
            stmt = insert(MonthlyUpdate.__table__).values(date=pd.to_datetime(row['date']).date(), person=str(row['person']), account_type=str(row['account_type']), value=float(row['value']))
            stmt = stmt.on_conflict_do_update(index_elements=['date', 'person', 'account_type'], set_={'value': value})
            sess.execute(stmt)
        
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

    # ARIMA
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
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    rf_pred = rf.predict(future_x)

    return forecast, lower, upper, lr_pred, rf_pred

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

    # Pivot
    pivot = df.pivot_table(
        index="date",
        columns=["person", 'account_type'],
        values="value",
        aggfunc="sum",
        fill_value=0,
    )
    st.subheader("Monthly Summary")
    st.dataframe(pivot.style.format("${:,.0f}"))

    # Net Worth Chart
    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index()
    df_net = df_net.sort_values("date")
    fig_net = px.line(
        df_net, x="date", y="value",
        title="Family Net Worth (Sean + Kim)",
        labels={"value": "Total ($)"}
    )
    fig_net.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_net, use_container_width=True)

    # AI Projections
    st.subheader("AI Growth Projections")
    horizon = st.slider("Forecast Horizon (months)", 12, 60, 24)
    arima_f, arima_lower, arima_upper, lr_f, rf_f = ai_projections(df_net, horizon)
    
    if arima_f is not None:
        future_dates = pd.date_range(start=df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Historical", line=dict(color='blue')))
        
        # ARIMA
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_f, name="ARIMA Forecast", line=dict(color='green')))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_lower, fill=None, mode='lines', line=dict(color='green', dash='dash'), showlegend=False))
        fig_proj.add_trace(go.Scatter(x=future_dates, y=arima_upper, fill='tonexty', mode='lines', line=dict(color='green'), name='ARIMA CI'))
        
        # Linear
        fig_proj.add_trace(go.Scatter(x=future_dates, y=lr_f, name="Linear Trend", line=dict(color='orange')))
        
        # RF
        fig_proj.add_trace(go.Scatter(x=future_dates, y=rf_f, name="Random Forest", line=dict(color='red')))
        
        fig_proj.update_layout(title=f"AI Projections ({horizon} months)", yaxis_title="Net Worth ($)", xaxis_title="Date")
        st.plotly_chart(fig_proj, use_container_width=True)
        
        # Summary Table
        proj_df = pd.DataFrame({
            'Model': ['ARIMA (Median)', 'Linear Regression', 'Random Forest'],
            '24 Months': [arima_f[23] if len(arima_f) > 23 else np.nan, lr_f[23] if len(lr_f) > 23 else np.nan, rf_f[23] if len(rf_f) > 23 else np.nan],
            '60 Months': [arima_f[59] if len(arima_f) > 59 else np.nan, lr_f[59] if len(lr_f) > 59 else np.nan, rf_f[59] if len(rf_f) > 59 else np.nan]
        })
        proj_df = proj_df.round(0).style.format({"24 Months": "${:,.0f}", "60 Months": "${:,.0f}"})
        st.dataframe(proj_df)
    else:
        st.info("Need 3+ months of data for projections.")

    # Goals
    st.subheader("Financial Goals")
    goals = get_goals()
    current = df_net["value"].iloc[-1] if not df_net.empty else 0

    for g in goals:
        progress = min(current / g.target, 1.0)
        st.progress(progress)
        st.write(f"**{g.name}**: ${current:,.0f} / ${g.target:,.0f} ({progress*100:.1f}%) → {g.by_year}")

    # Edit/Delete Goals
    st.subheader("Edit/Delete Goals")
    for g in goals:
        with st.expander(f"Edit {g.name}"):
            new_target = st.number_input("New Target", value=g.target)
            new_year = st.number_input("New Year", value=g.by_year)
            if st.button("Update Goal", key=f"update_{g.name}"):
                update_goal(g.name, new_target, new_year)
                st.success("Updated!")
                st.rerun()
            if st.button("Delete Goal", key=f"delete_{g.name}"):
                delete_goal(g.name)
                st.success("Deleted!")
                st.rerun()

    # Growth Rates
    st.subheader("Monthly Growth Rates")
    latest_date = df["date"].max()
    prev_date = latest_date - pd.DateOffset(months=1)
    latest = df[df["date"] == latest_date]
    prev = df[df["date"] == prev_date]

    if len(prev) > 0:
        merged = latest.merge(prev, on=["person", "account_type"], suffixes=("_curr", "_prev"))
        merged["growth"] = (merged["value_curr"] - merged["value_prev"]) / merged["value_prev"]
        growth_df = merged[["person", "account_type", "growth"]].copy()
        growth_df["growth"] = (growth_df["growth"] * 100).round(2)
        growth_df = growth_df.sort_values("growth", ascending=False)
        st.dataframe(growth_df.style.format({"growth": "{:.2f}%"}))
    else:
        st.info("Not enough data for growth rates (need 2+ months)")

    # Delete Entry
    st.subheader("Delete an Entry")
    df_disp = df.reset_index(drop=True)
    choice = st.selectbox(
        "Select row",
        options=df_disp.index,
        format_func=lambda i: f"{df_disp.loc[i,'date']} – {df_disp.loc[i,'person']} – {df_disp.loc[i,'account_type']} – ${df_disp.loc[i,'value']:,.0f}"
    )
    if st.button("Delete"):
        row = df_disp.loc[choice]
        sess = get_session()
        sess.query(MonthlyUpdate).filter_by(
            date=row["date"], person=row["person"], account_type=row["account_type"]
        ).delete()
        sess.commit()
        sess.close()
        st.success("Deleted!")
        st.rerun()

    # Export
    csv_vals = df.to_csv(index=False).encode()
    st.download_button("Export Values CSV", csv_vals, "values.csv", "text/csv")
    if not df_contrib.empty:
        csv_cont = df_contrib.to_csv(index=False).encode()
        st.download_button("Export Contributions CSV", csv_cont, "contributions.csv", "text/csv")
