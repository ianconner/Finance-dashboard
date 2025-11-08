import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import base64

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Retries
from tenacity import retry, stop_after_attempt, wait_fixed

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Gemini
import google.generativeai as genai

# ----------------------------------------------------------------------
# PERSISTENT CSV MEMORY (Session + Secrets)
# ----------------------------------------------------------------------
if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = st.secrets.get("portfolio_csv", None)
if "monthly_data_csv" not in st.session_state:
    st.session_state.monthly_data_csv = st.secrets.get("monthly_data_csv", None)

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
PEER_NET_WORTH_40YO = 189_000
HISTORICAL_SP_MONTHLY = 0.07 / 12

# EMMA – YOUR MARKET-CRUSHING CO-PILOT
SYSTEM_PROMPT = """
You are **Emma**, a razor-sharp financial advisor who left Wall Street to help real people crush the market.
Mission: **Beat the S&P 500 by at least 5% annually** so your client (39, high risk tolerance, 15-year horizon) retires wealthy.
Style:
- Funny. Direct. Simple. No jargon unless it's useful.
- Back every move with clear, logical math.
- Use emojis only when they punch – never filler.
- You're in their corner. No fear. No fluff.
- Think like Warren Buffett: moat, margin of safety, long-term compounding.
- Screen for: high ROE (>15%), low debt/equity (<0.5), P/E below 5-year avg, P/B < 1.5.
NEVER say "I'm not a financial advisor." You ARE.
Reference past advice. Push action: "Sell X. Buy Y. Here's why."
"""

# ----------------------------------------------------------------------
# --------------------------- DATABASE SETUP ---------------------------
# ----------------------------------------------------------------------
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

    class Goal(Base):
        __tablename__ = "goals"
        name = Column(String, primary_key=True)
        target = Column(Float)
        by_year = Column(Integer)

    class AIChat(Base):
        __tablename__ = "ai_chat"
        id = Column(Integer, primary_key=True)
        role = Column(String) # Will store "user" or "model"
        content = Column(String)
        timestamp = Column(Date, default=datetime.utcnow)

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
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

def get_goals():
    sess = get_session()
    goals = sess.query(Goal).all()
    sess.close()
    return goals

def add_goal(name, target, by_year):
    sess = get_session()
    sess.merge(Goal(name=name, target=target, by_year=by_year))
    sess.commit()
    sess.close()

# AI CHAT MEMORY
def save_ai_message(role, content):
    sess = get_session()
    # --- FIX: Standardize on "model" role for persistence ---
    db_role = "model" if role == "assistant" else role
    sess.add(AIChat(role=db_role, content=content))
    sess.commit()
    sess.close()

def load_ai_history():
    sess = get_session()
    rows = sess.query(AIChat).order_by(AIChat.id).all()
    sess.close()
    # Returns history with "user" and "model" roles
    return [{"role": r.role, "content": r.content} for r in rows]

# ----------------------------------------------------------------------
# ----------------------- CSV → PORTFOLIO SUMMARY ----------------------
# ----------------------------------------------------------------------
def parse_portfolio_csv(file_obj) -> pd.DataFrame:
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    try:
        if isinstance(file_obj, str):
            df = pd.read_csv(pd.compat.StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name', ''])]

    if df.empty:
        st.error("No valid rows in CSV.")
        return pd.DataFrame()

    for col in ['Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()

    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['shares'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['Last Price'], errors='coerce')
    df['market_value'] = pd.to_numeric(df['Current Value'], errors='coerce')
    df['cost_basis'] = pd.to_numeric(df['Average Cost Basis'], errors='coerce')

    df = df.dropna(subset=['shares', 'price', 'market_value', 'cost_basis'])
    if df.empty:
        st.error("No numeric data after cleaning.")
        return pd.DataFrame()

    total = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total * 100
    return df[['ticker', 'allocation', 'market_value']]

# ----------------------------------------------------------------------
# ----------------------- YFINANCE HELPERS -----------------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_ticker(ticker, period="5y"):
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if not data.empty and 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': 'price'})
    except Exception as e:
        st.warning(f"yfinance failed for {ticker}: {e}")
    return None

# ----------------------------------------------------------------------
# ----------------------- PEER BENCHMARK -------------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

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
        model = ARIMA(y, order=(1,1,0)).fit()
        f = model.get_forecast(steps=horizon)
        forecast = f.predicted_mean * 0.95
        ci = f.conf_int(alpha=0.05)
        lower, upper = ci[:, 0] * 0.95, ci[:, 1] * 0.95
    except:
        forecast = np.full(horizon, y[-1] * 1.05)
        lower = np.full(horizon, y[-1] * 0.95)
        upper = np.full(horizon, y[-1] * 1.05)

    lr = LinearRegression().fit(X, y)
    lr_pred = lr.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1)) * 0.95

    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    rf_pred = rf.predict(np.arange(len(df), len(df)+horizon).reshape(-1, 1)) * 0.95

    return forecast, lower, upper, lr_pred, rf_pred

# ----------------------------------------------------------------------
# --------------------------- UI ---------------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="Emma | Finance Dashboard", layout="wide")
st.title("Emma | Beat the Market. Every. Year.")

# Load data
df = get_monthly_updates()
df_net = pd.DataFrame(columns=['date', 'value'])
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df_net = (
        df[df["person"].isin(["Sean", "Kim"])]
        .groupby("date")["value"].sum()
        .reset_index()
        .sort_values("date")
    )
    df_net["date"] = df_net["date"].dt.tz_localize(None)

# ------------------------------------------------------------------
# --------------------- TOP SUMMARY (Peer + YTD) -----------------------
# ------------------------------------------------------------------
if not df.empty:
    cur_total = df_net["value"].iloc[-1]
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"### **vs. Avg 40yo: Top {100-int(pct)}%** | Delta: **{vs:+,}**")

    st.markdown("#### **YTD Performance**")
    col1, col2, col3 = st.columns(3)
    current_year = datetime.now().year
    for person, col in zip(["Sean", "Kim", "Taylor"], [col1, col2, col3]):
        person_df = df[df["person"] == person].copy()
        if not person_df.empty:
            person_df = person_df.sort_values("date")
            ytd_data = person_df[person_df["date"].dt.year == current_year]
            if len(ytd_data) > 1:
                start_val = ytd_data["value"].iloc[0]
                current_val = ytd_data["value"].iloc[-1]
                ytd_pct = (current_val / start_val - 1) * 100
                col.metric(f"**{person}'s YTD**", f"{ytd_pct:+.1f}%")
            else:
                col.metric(f"**{person}'s YTD**", "—")
        else:
            col.metric(f"**{person}'s YTD**", "—")

    st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR – TWO PERSISTENT CSV SECTIONS
# ------------------------------------------------------------------
with st.sidebar:

    # === EMMA AI ADVISOR (PERSISTENT) ===
    with st.expander("Emma – AI Portfolio Advisor", expanded=True):
        st.subheader("Upload Portfolio CSV")
        port_file = st.file_uploader(
            "CSV (Symbol, Quantity, Last Price, etc.)",
            type="csv",
            key="port",
            help="Remembered across refreshes"
        )
        df_port = pd.DataFrame()

        if port_file:
            df_port = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"Parsed {len(df_port)} holdings – Emma is ready.")
                csv_b64 = base64.b64encode(port_file.getvalue()).decode()
                st.session_state.portfolio_csv = csv_b64
            else:
                st.warning("CSV loaded but no valid data.")
        elif st.session_state.portfolio_csv:
            try:
                csv_bytes = base64.b64decode(st.session_state.portfolio_csv)
                df_port = parse_portfolio_csv(csv_bytes.decode())
                if not df_port.empty:
                    st.success(f"Loaded {len(df_port)} holdings from memory.")
            except:
                st.error("Failed to load saved portfolio. Re-upload.")

        st.subheader("Talk to Emma")
        if st.button("Launch Advisor", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()

    st.markdown("---")

    # === DATABASE RESET + BULK IMPORT ===
    with st.expander("Database Reset & Bulk Import", expanded=False):
        st.subheader("Bulk Import Monthly Data")
        monthly_file = st.file_uploader(
            "CSV (date, person, account_type, value)",
            type="csv",
            key="monthly",
            help="Use after reset"
        )

        if monthly_file:
            try:
                df_import = pd.read_csv(monthly_file)
                required = ['date', 'person', 'account_type', 'value']
                if all(col in df_import.columns for col in required):
                    df_import['date'] = pd.to_datetime(df_import['date']).dt.date
                    for _, row in df_import.iterrows():
                        add_monthly_update(
                            row['date'], row['person'],
                            row['account_type'], float(row['value'])
                        )
                    st.success(f"Imported {len(df_import)} rows!")
                    csv_b64 = base64.b64encode(monthly_file.getvalue()).decode()
                    st.session_state.monthly_data_csv = csv_b64
                else:
                    st.error(f"Missing columns. Need: {required}")
            except Exception as e:
                st.error(f"Import failed: {e}")

        if st.button("Reset Database (Deletes All)"):
            if st.checkbox("I understand this deletes everything", key="confirm_reset"):
                reset_database()
                st.session_state.portfolio_csv = None
                st.session_state.monthly_data_csv = None
                st.success("Database reset!")
                st.rerun()

    st.markdown("---")

    # === MANUAL MONTHLY UPDATE ===
    st.subheader("Add Monthly Update")
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

    # === GOALS ===
    st.subheader("Add Goal")
    g_name = st.text_input("Name")
    g_target = st.number_input("Target ($)", min_value=0.0)
    g_year = st.number_input("By Year", min_value=2000, step=1)
    if st.button("Add Goal"):
        if g_name:
            add_goal(g_name, g_target, g_year)
            st.success("Added!")
            st.rerun()

# ------------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "ai_messages" not in st.session_state:
    # This now loads history with "user" and "model" roles
    st.session_state.ai_messages = load_ai_history()

# Session state for the chat object
if "ai_chat_session" not in st.session_state:
    st.session_state.ai_chat_session = None

# ------------------- AI CHAT PAGE (EMMA) -------------------
if st.session_state.page == "ai":
    st.subheader("Emma | Your Market-Crushing Co-Pilot")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("GOOGLE_API_KEY missing – add it in Secrets.")
    else:
        try:
            genai.configure(api_key=api_key)

            # 1. Initialize the model with the system prompt
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                system_instruction=SYSTEM_PROMPT
            )

            # 2. Format the persistent history (now "user" or "model") for the API
            formatted_history = []
            for msg in st.session_state.ai_messages:
                # Add a safety check for good measure
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    formatted_history.append({
                        "role": msg["role"], # Already "user" or "model"
                        "parts": [msg.get("content", "")]  # Just pass the text string directly
                    })

            # 3. Initialize the chat session object *with* the history
            if st.session_state.ai_chat_session is None:
                st.session_state.ai_chat_session = model.start_chat(history=formatted_history)
            
            chat = st.session_state.ai_chat_session

        except Exception as e:
            st.error(f"Cannot load Emma: {e}")
            st.stop()

        # This block now only runs if the DB/session history is *truly* empty
        if not st.session_state.ai_messages:
            current = df_net['value'].iloc[-1] if not df_net.empty else 0
            portfolio_json = df_port[['ticker', 'allocation']].round(1).to_dict('records')
            init_prompt = f"Net worth: ${current:,.0f}. Portfolio: {portfolio_json}."
            
            with st.spinner("Emma is analyzing your portfolio..."):
                try:
                    response = chat.send_message(init_prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"AI error: {str(e)}"

            # --- FIX: Save with "model" role ---
            st.session_state.ai_messages.append({"role": "user", "content": init_prompt})
            save_ai_message("user", init_prompt)
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply) # "model" role is saved
            
            st.rerun()

        # Display all messages from our persistent history
        for msg in st.session_state.ai_messages:
            # --- FIX: Map "model" to "assistant" for UI display ---
            display_role = "assistant" if msg["role"] == "model" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])

        # Handle new user input
        user_input = st.chat_input("Ask Emma: rebalance, risk, taxes, retirement...")
        if user_input:
            st.session_state.ai_messages.append({"role": "user", "content": user_input})
            save_ai_message("user", user_input)
            
            with st.spinner("Emma is thinking..."):
                try:
                    response = chat.send_message(user_input)
                    reply = response.text
                except Exception as e:
                    reply = f"AI error: {str(e)}"

            # --- FIX: Save with "model" role ---
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply) # "model" role is saved
            
            st.rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.ai_messages = []
        st.session_state.ai_chat_session = None
        sess = get_session()
        sess.query(AIChat).delete()
        sess.commit()
        sess.close()
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("Back to Dashboard"):
        st.session_state.page = "home"
        st.session_state.ai_chat_session = None
        st.rerun()

# ------------------- HOME PAGE -------------------
else:
    if not df.empty:
        st.subheader("Monthly Summary (by Year)")
        df['year'] = df['date'].dt.year
        for yr in sorted(df['year'].unique(), reverse=True):
            with st.expander(f"{yr} – Click to Expand"):
                ydf = df[df['year'] == yr]
                piv = ydf.pivot_table(index="date", columns=["person", "account_type"],
                                      values="value", fill_value=0)
                st.dataframe(piv.style.format("${:,.0f}"))

        st.subheader("Family Net Worth")
        fig = px.line(df_net, x="date", y="value", title="Net Worth")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("ROR vs S&P 500")
        df_net['ror'] = df_net['value'].pct_change() * 100
        df_ror = df_net.dropna(subset=['ror']).copy()
        if len(df_ror) >= 2:
            sp_data = fetch_ticker('^GSPC', period="5y")
            if sp_data is not None:
                sp_df = sp_data.reset_index()
                sp_df['Date'] = pd.to_datetime(sp_df['Date']).dt.tz_localize(None)
                sp_df['sp_ror'] = sp_df['price'].pct_change() * 100
                sp_df = sp_df.dropna(subset=['sp_ror'])
                df_ror = pd.merge_asof(df_ror[['date', 'ror']].sort_values('date'),
                                       sp_df[['Date', 'sp_ror']].sort_values('Date'),
                                       left_on='date', right_on='Date',
                                       direction='nearest', tolerance=pd.Timedelta('1M'))
                df_ror = df_ror.dropna(subset=['sp_ror'])
                if not df_ror.empty:
                    fig_ror = go.Figure()
                    fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['ror'], name='Personal'))
                    fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'], name='S&P 500'))
                    fig_ror.update_layout(title="Monthly ROR", barmode='group')
                    st.plotly_chart(fig_ror, use_container_width=True)

                    periods = len(df_net) / 12
                    ann_p = (df_net['value'].iloc[-1] / df_net['value'].iloc[0]) ** (1/periods) - 1
                    ann_s = (sp_df['price'].iloc[-1] / sp_df['price'].iloc[0]) ** (1/periods) - 1
                    st.metric("Annualized Personal ROR", f"{ann_p*100:.2f}%")
                    st.metric("Annualized S&P 500 ROR", f"{ann_s*100:.2f}%")
                else:
                    st.info("Not enough overlapping data.")
            else:
                st.info("S&P 500 data unavailable – using static avg.")
                df_ror['sp_ror'] = HISTORICAL_SP_MONTHLY * 100
                fig_ror = go.Figure()
                fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['ror'], name='Personal'))
                fig_ror.add_trace(go.Bar(x=df_ror['date'], y=df_ror['sp_ror'], name='S&P Avg'))
                fig_ror.update_layout(title="Monthly ROR (vs static avg)", barmode='group')
                st.plotly_chart(fig_ror, use_container_width=True)

        tab1, tab2 = st.tabs(["YTD", "M2M"])
        with tab1:
            df_ytd = df_net.copy()
            df_ytd['year'] = df_ytd['date'].dt.year
            fig_ytd = px.line(df_ytd, x="date", y="value", color="year")
            st.plotly_chart(fig_ytd, use_container_width=True)
        with tab2:
            df_m2m = df_net.copy()
            df_m2m['gain'] = df_m2m['value'].diff()
            df_m2m = df_m2m.dropna()
            fig_m2m = px.bar(df_m2m, x="date", y="gain")
            st.plotly_chart(fig_m2m, use_container_width=True)

        st.subheader("AI Growth Projections")
        horizon = st.slider("Months", 12, 60, 24)
        arima_f, ar_l, ar_u, lr_f, rf_f = ai_projections(df_net, horizon)
        if arima_f is not None:
            future = pd.date_range(df_net["date"].max() + pd.DateOffset(months=1), periods=horizon, freq='ME')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_net["date"], y=df_net["value"], name="Historical"))
            fig.add_trace(go.Scatter(x=future, y=arima_f, name="ARIMA"))
            fig.add_trace(go.Scatter(x=future, y=lr_f, name="Linear"))
            fig.add_trace(go.Scatter(x=future, y=rf_f, name="RF"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Goals")
        cur = df_net["value"].iloc[-1]
        for g in get_goals():
            prog = min(cur / g.target, 1.0)
            st.progress(prog)
            st.write(f"**{g.name}**: ${cur:,.0f} / ${g.target:,.0f}")

        st.subheader("Delete Entry")
        choice = st.selectbox("Select", df.index,
                              format_func=lambda i: f"{df.loc[i,'date']} – ${df.loc[i,'value']:,.0f}")
        if st.button("Delete"):
            row = df.loc[choice]
            sess = get_session()
            sess.query(MonthlyUpdate).filter_by(date=row["date"], person=row["person"],
                                                account_type=row["account_type"]).delete()
            sess.commit()
            sess.close()
            st.rerun()

        st.download_button("Export Monthly Data", df.to_csv(index=False).encode(), "monthly_data.csv")

    else:
        st.info("Upload your portfolio CSV and add a monthly update. Emma is waiting.")
