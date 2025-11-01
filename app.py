import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, text
from sqlalchemy.orm import declarative_base, sessionmaker  # New import to fix warning!
from pathlib import Path
import os

# -------------------------------------------------
# 1. CONNECT TO PERSISTENT DB
# -------------------------------------------------
DB_PATH = Path("data") / "finances.db"
DB_PATH.parent.mkdir(exist_ok=True)

# For local dev → SQLite
engine = create_engine(f"sqlite:///{DB_PATH}")

# For cloud (Streamlit Community Cloud, Railway, etc.) you can use Postgres:
# engine = create_engine(st.secrets["postgres_url"])

Base = declarative_base()

class Entry(Base):
    __tablename__ = "entries"
    id          = Column(Integer, primary_key=True)
    date        = Column(Date, nullable=False, index=True)   # YYYY-MM-DD
    person      = Column(String(20), nullable=False)
    account_type= Column(String(50), nullable=False)
    value       = Column(Float, nullable=False)

# Build the table only if it doesn't exist (fixes the crash!)
try:
    Base.metadata.create_all(engine)
except Exception as e:
    if "table entries already exists" not in str(e):
        raise  # Only ignore if it's the "already exists" error

Session = sessionmaker(bind=engine)

def get_session():
    return Session()

def df_from_db():
    """Return the whole table as a pandas DataFrame (sorted by date)."""
    with engine.connect() as conn:
        df = pd.read_sql_table("entries", conn)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date")

def add_entry(date, person, account_type, value):
    sess = get_session()
    sess.add(Entry(date=date, person=person, account_type=account_type, value=value))
    sess.commit()
    sess.close()

def unique_accounts(person: str):
    """Return list of account_types that belong to a person."""
    sql = text("SELECT DISTINCT account_type FROM entries WHERE person = :p ORDER BY account_type")
    with engine.connect() as conn:
        res = conn.execute(sql, {"p": person}).fetchall()
    return [row[0] for row in res] if res else []

def add_monthly_update():
    st.subheader("Add Monthly Update")

    # ---- 1. PERSON SELECTOR -------------------------------------------------
    persons = ["Sean", "Kim"]                     # add more if you have them
    person = st.selectbox("Person", persons, key="person_selector")

    # ---- 2. ACCOUNT TYPE (dynamic, resets when person changes) ------------
    account_options = unique_accounts(person)
    if not account_options:
        account_options = ["TSP", "T3W", "Stocks"] if person == "Sean" else ["Kim Total"]
        st.info(f"No prior entries for **{person}**. Using defaults – they will be saved automatically.")
    account_type = st.selectbox(
        "Account type",
        account_options,
        key=f"acct_{person}"   # unique key per person → no cross-talk
    )

    # ---- 3. DATE & VALUE ----------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        value = st.number_input("Value ($)", min_value=0.0, format="%.2f")

    if st.button("Save entry"):
        add_entry(date, person, account_type, float(value))
        st.success(f"Saved {person} → {account_type} = ${value:,.2f} on {date}")
        st.experimental_rerun()   # refresh UI instantly

def main_page():
    st.title("Personal Finance Tracker")

    # -----------------------------------------------------------------
    # 1. Show current data (always fresh from DB)
    # -----------------------------------------------------------------
    df = df_from_db()
    if df.empty:
        st.info("No data yet – add your first entry above!")
    else:
        # Pivot for the “wide” view you love
        pivot = df.pivot_table(
            index="date",
            columns=["person", "account_type"],
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        st.dataframe(pivot.style.format("${:,.0f}"))

        # Optional: download as CSV (long format, ready for re-import)
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV (long format)",
            csv,
            "finances_long.csv",
            "text/csv",
        )

    # -----------------------------------------------------------------
    # 2. Add new entry
    # -----------------------------------------------------------------
    add_monthly_update()

    # -----------------------------------------------------------------
    # 3. (Optional) Delete an entry
    # -----------------------------------------------------------------
    if not df.empty:
        st.subheader("Delete an entry")
        # Build a list of ids for the selectbox
        df_disp = df.reset_index(drop=True)
        choice = st.selectbox(
            "Select row to delete",
            options=df_disp.index,
            format_func=lambda i: f"{df_disp.loc[i, 'date']} – {df_disp.loc[i, 'person']} – {df_disp.loc[i, 'account_type']} – ${df_disp.loc[i, 'value']:,.0f}"
        )
        if st.button("Delete"):
            entry_id = int(df_disp.loc[choice, "id"])
            sess = get_session()
            sess.query(Entry).filter(Entry.id == entry_id).delete()
            sess.commit()
            sess.close()
            st.success("Deleted!")
            st.experimental_rerun()
