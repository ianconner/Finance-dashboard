# family_wealth_tracker.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Firebase — this is the line that works on Streamlit Cloud
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(st.secrets["firebase_admin"]))

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

# ========================== SIDEBAR – INPUT ==========================
with st.sidebar:
    st.header("Add / Update Month")
    new_date = st.date_input("Month", value=datetime.today().replace(day=1), format="YYYY-MM")
    sean = st.number_input("Sean's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    kim = st.number_input("Kim's Net Worth ($)", value=0.0, step=1000.0, format="%.0f")
    total = sean + kim
    if st.button("Save Month"):
        save_monthly(new_date, sean, kim, total)
        st.success(f"{new_date:%b %Y} saved!")
        st.rerun()

    st.divider()
    st.header("Data Tools")
    csv_data = export_csv()
    if csv_data:
        st.download_button("Download Full Backup CSV", csv_data, "family_wealth_backup.csv", "text/csv")

    if st.button("Clear All Data (careful!)"):
        for doc in db.collection(COLLECTION).stream():
            doc.reference.delete()
        st.success("All data cleared")
        st.rerun()

# ========================== MAIN DASHBOARD ==========================
if df.empty:
    st.info("No data yet – add your first month on the left!")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Current numbers
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df)>1 else latest

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sean's Net Worth", f"${latest.sean:,.0f}", f"{latest.sean-prev.sean:+,.0f}")
col2.metric("Kim's Net Worth", f"${latest.kim:,.0f}", f"{latest.kim-prev.kim:+,.0f}")
col3.metric("Total Family Net Worth", f"${latest.total:,.0f}", f"{latest.total-prev.total:+,.1f}", delta_color="normal")
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
yoy["YoY $"] = yoy["last"].diff()
yoy["YoY %"] = yoy["last"].pct_change()
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
