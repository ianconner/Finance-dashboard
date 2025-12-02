# family_wealth_tracker.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# ========================== FIREBASE & GEMINI ==========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

if not firebase_admin._apps:
    cred_dict = dict(st.secrets["firebase_admin"])
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    firebase_admin.initialize_app(credentials.Certificate(cred_dict))

db = firestore.client()
model = genai.GenerativeModel("gemini-1.5-flash")

# ========================== DATA HELPERS ==========================
COLLECTION = "monthly_data"

def save_monthly(date_obj, sean, kim, total):
    doc_id = date_obj.strftime("%Y-%m")
    db.collection(COLLECTION).document(doc_id).set({
        "date": date_obj.isoformat(),
        "sean": float(sean),
        "kim": float(kim),
        "total": float(total)
    })

def load_data():
    docs = db.collection(COLLECTION).order_by("date").stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        d["date"] = datetime.fromisoformat(d["date"]).date()
        data.append(d)
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["date","sean","kim","total"])

def export_csv():
    df = load_data()
    return df.to_csv(index=False).encode() if not df.empty else None

# ========================== PAGE SETUP ==========================
st.set_page_config(page_title="Family Wealth Tracker", layout="wide")
st.title("Family Wealth Tracker")
st.markdown("#### Sean • Kim • Combined  |  Powered by S.A.G.E. your financial co-pilot")

# ========================== LOAD DATA ==========================
df = load_data()

# ========================== SIDEBAR (ALWAYS VISIBLE) ==========================
with st.sidebar:
    st.header("Add / Update Month")
    
    # This works 100% on every Streamlit version
    new_date = st.date_input(
        "Month",
        value=datetime.today().replace(day=1),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.today()
    )
    
    # Optional: show it nicely
    st.write(f"Selected: **{new_date:%b %Y}**")
    
    sean = st.number_input("Sean's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    kim = st.number_input("Kim's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    total = sean + kim

    if st.button("Save Month", type="primary", use_container_width=True):
        save_monthly(new_date, sean, kim, total)
        st.success(f"{new_date:%b %Y} saved!")
        st.rerun()

    st.divider()
    st.subheader("Data Tools")

    csv_data = export_csv()
    if csv_data:
        st.download_button(
            "Download Full Backup CSV",
            csv_data,
            f"family_wealth_backup_{datetime.today():%Y-%m-%d}.csv",
            "text/csv"
        )

    if st.button("Clear All Data (DANGER)", type="secondary"):
        for doc in db.collection(COLLECTION).stream():
            doc.reference.delete()
        st.success("All data cleared!")
        st.rerun()

    st.divider()
    st.subheader("Migration Tools")
    
    uploaded = st.file_uploader("Upload old CSV to import", type=["csv"])
    if uploaded:
        try:
            import_df = pd.read_csv(uploaded)
            if all(col in import_df.columns for col in ["date","sean","kim","total"]):
                import_df["date"] = pd.to_datetime(import_df["date"]).dt.date
                bar = st.progress(0)
                for i, row in import_df.iterrows():
                    save_monthly(row["date"], row["sean"], row["kim"], row["total"])
                    bar.progress((i+1)/len(import_df))
                st.success(f"Imported {len(import_df)} months!")
                st.balloons()
                st.rerun()
            else:
                st.error("CSV needs columns: date, sean, kim, total")
        except Exception as e:
            st.error(f"Error: {e}")

# ========================== MAIN DASHBOARD ==========================
if df.empty:
    st.info("No data yet — add your first month in the sidebar on the left!")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sean's Net Worth", f"${latest.sean:,.0f}", f"{latest.sean-prev.sean:+,.0f}")
c2.metric("Kim's Net Worth", f"${latest.kim:,.0f}", f"{latest.kim-prev.kim:+,.0f}")
c3.metric("Total Family Net Worth", f"${latest.total:,.0f}", f"{latest.total-prev.total:+,.0f}")
c4.metric("Monthly Change", f"{(latest.total/prev.total-1)*100:+.2f}%")

# Growth chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.date, y=df.sean, name="Sean", line=dict(width=4)))
fig.add_trace(go.Scatter(x=df.date, y=df.kim, name="Kim", line=dict(color="orange", width=4)))
fig.add_trace(go.Scatter(x=df.date, y=df.total, name="Total", line=dict(color="green", width=6)))
fig.update_layout(title="Family Net Worth Over Time", height=500)
st.plotly_chart(fig, use_container_width=True)

# S&P 500 comparison
start_date = df.date.min().date()
start_value = df.total.iloc[0]
sp500 = yf.download("^GSPC", start=start_date, progress=False)["Adj Close"]
if not sp500.empty:
    sp_growth = (sp500 / sp500.iloc[0]) * start_value
    sp_df = pd.DataFrame({"date": sp500.index, "S&P 500": sp_growth.values})
    fig2 = px.line(df, x="date", y="total", title="You vs S&P 500")
    fig2.add_scatter(x=sp_df.date, y=sp_df["S&P 500"], name="S&P 500", line=dict(dash="dot", color="gray"))
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Built with love for Sean, Kim & family — let’s beat the market together.")
