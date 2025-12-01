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

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

if not firebase_admin._apps:
    cred_dict = {
        "type": st.secrets["firebase_admin"]["type"],
        "project_id": st.secrets["firebase_admin"]["project_id"],
        "private_key_id": st.secrets["firebase_admin"]["private_key_id"],
        "private_key": st.secrets["firebase_admin"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["firebase_admin"]["client_email"],
        "client_id": st.secrets["firebase_admin"]["client_id"],
        "auth_uri": st.secrets["firebase_admin"]["auth_uri"],
        "token_uri": st.secrets["firebase_admin"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase_admin"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase_admin"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firebase_admin"]["universe_domain"]
    }
    firebase_admin.initialize_app(credentials.Certificate(cred_dict))

db = firestore.client()
# Gemini (already in your code)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ========================== DATA HELPERS ==========================
COLLECTION = "monthly_data"

def save_monthly(date, sean, kim, total):
    doc_id = date.strftime("%Y-%m")
    db.collection(COLLECTION).document(doc_id).set({
        "date": date.isoformat(),
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
    if df.empty:
        return None
    csv = df.to_csv(index=False).encode()
    return csv

# ========================== PAGE CONFIG ==========================
st.set_page_config(page_title="Family Wealth Tracker", layout="wide")
st.title("Family Wealth Tracker")
st.markdown("#### Sean • Kim • Combined  |  Powered by S.A.G.E. your financial co-pilot")

# ========================== LOAD DATA ==========================
df = load_data()

# ========================== SIDEBAR – INPUT (always visible) ==========================
with st.sidebar:
    st.header("Add / Update Month")
    new_date = st.date_input("Month", value=datetime.today().replace(day=1), format="MM/DD/YYYY")
    sean = st.number_input("Sean's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    kim = st.number_input("Kim's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    total = sean + kim

    if st.button("Save Month", type="primary", use_container_width=True):
        save_monthly(new_date, sean, kim, total)
        st.success(f"{new_date:%b %Y} saved!")
        st.rerun()

    st.divider()
    st.header("Data Tools")
    csv_data = export_csv()
    if csv_data:
        st.download_button("Download Backup CSV", csv_data, "family_wealth_backup.csv", "text/csv")
    
    st.divider()
    st.subheader("🔽 Migration Tools")

    # DOWNLOAD EVERYTHING FROM THE CURRENT (OLD) DB
    if st.button("Download ALL Historical Data as CSV", type="primary"):
        full_csv = export_csv()
        if full_csv:
            st.download_button(
                label="⬇️ Save Your Complete History Now",
                data=full_csv,
                file_name=f"family_wealth_full_backup_{datetime.today():%Y-%m-%d}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("Ready! Click the button above to download.")
        else:
            st.warning("No data yet – add some months first.")

    # UPLOAD AND IMPORT OLD DATA (works with the CSV you just downloaded or any old one)
    uploaded_file = st.file_uploader("Upload old CSV to import everything", type=["csv"])
    if uploaded_file is not None:
        try:
            import_df = pd.read_csv(uploaded_file)
            required_cols = ["date", "sean", "kim", "total"]
            if not all(col in import_df.columns for col in required_cols):
                st.error(f"CSV must have columns: {required_cols}")
            else:
                import_df["date"] = pd.to_datetime(import_df["date"]).dt.date
                progress_bar = st.progress(0)
                for i, row in import_df.iterrows():
                    save_monthly(row["date"], row["sean"], row["kim"], row["total"])
                    progress_bar.progress((i + 1) / len(import_df))
                st.success(f"Imported {len(import_df)} months successfully!")
                st.balloons()
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ========================== MAIN DASHBOARD (only if data exists) ==========================
if df.empty:
    st.info("No data yet – add your first month in the sidebar on the left!")
    st.stop()

# Now we know df has at least one row
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# Rest of your beautiful dashboard (metrics, charts, etc.) stays exactly the same below this line
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sean's Net Worth", f"${latest.sean:,.0f}", f"{latest.sean-prev.sean:+,.0f}")
col2.metric("Kim's Net Worth", f"${latest.kim:,.0f}", f"{latest.kim-prev.kim:+,.0f}")
col3.metric("Total Family Net Worth", f"${latest.total:,.0f}", f"{latest.total-prev.total:+,.0f}", delta_color="normal")
col4.metric("Monthly Change %", f"{(latest.total/prev.total-1)*100:+.2f}%")

# ... (all your charts, YoY table, goals, S.A.G.E. stay exactly as they are)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Current numbers
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df)>1 else latest

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sean's Net Worth", f"${latest.sean:,.0f}", f"{latest.sean-prev.sean:+,.0f}")
col2.metric("Kim's Net Worth", f"${latest.kim:,.0f}", f"{latest.kim-prev.kim:+,.0f}")
col3.metric("Total Family Net Worth", f"${latest.total:,.0f}", f"{latest.total-prev.total:+,.0f}", delta_color="normal")
col4.metric("Monthly Change %", f"{(latest.total/prev.total-1)*100:+.2f}%")

# Growth Chart – Sean, Kim, Total
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.date, y=df.sean, name="Sean", line=dict(color="#636EFA", width=4)))
fig.add_trace(go.Scatter(x=df.date, y=df.kim, name="Kim", line=dict(color="#EF553B", width=4)))
fig.add_trace(go.Scatter(x=df.date, y=df.total, name="Total Family", line=dict(color="#00CC96", width=6)))
fig.update_layout(title="Family Net Worth Over Time", xaxis_title="", yaxis_title="Net Worth ($)", height=500)
st.plotly_chart(fig, use_container_width=True)

# S&P 500 Comparison
start_date = df.date.iloc[0]
start_value = df.total.iloc[0]
sp500 = yf.download("^GSPC", start=start_date, end=datetime.today(), progress=False)["Adj Close"]
if not sp500.empty:
    sp500_growth = (sp500 / sp500.iloc[0]) * start_value
    sp_df = pd.DataFrame({"date": sp500.index, "SP500": sp500_growth.values})
    sp_df["date"] = pd.to_datetime(sp_df["date"])
    fig2 = px.line(df, x="date", y="total", title="Your Family vs S&P 500")
    fig2.add_scatter(x=sp_df.date, y=sp_df.SP500, name="S&P 500 if you invested the same amount", line=dict(dash="dot", color="gray"))
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("S&P 500 data temporarily unavailable – check internet connection")

# Year-over-Year Table (exactly like your Excel)
st.subheader("Year-over-Year Summary")
yoy = df.groupby(df.date.dt.year)["total"].agg("last").reset_index()
yoy["YoY $"] = yoy["total"].diff()
yoy["YoY %"] = yoy["total"].pct_change()
yoy = yoy.round(0)
yoy.columns = ["Year", "Net Worth", "YoY $", "YoY %"]
yoy["YoY %"] = yoy["YoY %"].map("{:+.1%}".format)
yoy["YoY $"] = yoy["YoY $"].map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "")
st.dataframe(yoy, use_container_width=True, hide_index=True)

# Goals Section
st.subheader("Goals")
goal1 = st.number_input("Goal 1 – Target Net Worth ($)", value=1_000_000, step=100_000)
goal1_year = st.number_input("…by year", value=2035, step=1)
if latest.total > 0:
    progress1 = min(latest.total / goal1, 1.0)
    st.progress(progress1)
    st.write(f"**${latest.total:,.0f} → ${goal1:,.0f} by {goal1_year}** → {progress1:.1%} complete")

goal2 = st.number_input("Goal 2 – Target ($)", value=2_000_000, step=100_000)
goal2_year = st.number_input("…by year", value=2040, step=1)
if latest.total > 0:
    progress2 = min(latest.total / goal2, 1.0)
    st.progress(progress2)
    st.write(f"**${latest.total:,.0f} → ${goal2:,.0f} by {goal2_year}** → {progress2:.1%} complete")

# Monthly Table (exactly like your Excel)
st.subheader("Monthly History (2020–present)")
display_df = df.copy()
display_df["date"] = display_df["date"].dt.strftime("%b %Y")
display_df["Sean"] = display_df["sean"].apply(lambda x: f"${x:,.0f}")
display_df["Kim"] = display_df["kim"].apply(lambda x: f"${x:,.0f}")
display_df["Total"] = display_df["total"].apply(lambda x: f"${x:,.0f}")
display_df = display_df[["date", "Sean", "Kim", "Total"]]
st.dataframe(display_df, use_container_width=True, hide_index=True)

# S.A.G.E. – Your Financial Co-Pilot
st.markdown("---")
st.subheader("S.A.G.E. – Strategic Asset Growth Engine")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask S.A.G.E. anything – strategy, risk, goals, beating the market…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context for Gemini
    context = f"""
Current family net worth: ${latest.total:,.0f}
Last month change: {(latest.total/prev.total-1)*100:+.2f}%
YoY growth: {yoy.iloc[-1]['YoY %'] if len(yoy)>1 else 'N/A'}
Historical average annual return: ~12%
Goal 1: ${goal1:,} by {goal1_year} → {progress1:.1%} complete
Goal 2: ${goal2:,} by {goal2_year} → {progress2:.1%} complete
Full monthly history (last 12 months shown):
{df.tail(12)[['date','total']].to_string(index=False)}
"""
    full_prompt = f"{context}\n\nUser question: {prompt}\n\nRespond as S.A.G.E. – warm, brilliant, collaborative financial co-pilot. Use first-person plural ('we'), celebrate wins, be honest about risks, and always give one clear actionable idea."

    with st.chat_message("assistant"):
        with st.spinner("S.A.G.E. is thinking…"):
            response = model.generate_content(full_prompt)
            reply = response.text
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("---")
st.caption("Built with ❤️ for Sean, Kim & family – let’s beat the market together.")
