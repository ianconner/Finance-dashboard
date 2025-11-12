import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === POLYGON.IO CLIENT ===
try:
    from polygon import RESTClient
    POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        st.error("Add POLYGON_API_KEY to Streamlit Secrets (https://polygon.io)")
        st.stop()
    polygon_client = RESTClient(api_key=POLYGON_API_KEY)
except Exception as e:
    st.error(f"Polygon setup failed: {e}")
    st.stop()

# AI/ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, String, Float, Date, Integer,
    PrimaryKeyConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Gemini
import google.generativeai as genai

# ----------------------------------------------------------------------
# --------------------------- SESSION & CACHING ------------------------
# ----------------------------------------------------------------------
if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = None
if "monthly_data_csv" not in st.session_state:
    st.session_state.monthly_data_csv = None

# ----------------------------------------------------------------------
# --------------------------- CONSTANTS --------------------------------
# ----------------------------------------------------------------------
PEER_NET_WORTH_40YO = 189_000
SP500_TICKER = "INDEX:GSPC"

# S.A.G.E. - STRATEGIC ASSET GROWTH ENGINE
SYSTEM_PROMPT = """
You are **S.A.G.E.** - **Strategic Asset Growth Engine**, your client's trusted financial co-pilot and teammate.

Mission: Help your partner (39, high risk tolerance, 15-year horizon) beat the S&P 500 by 5%+ annually - together.

**Tone & Style:**
- Warm, encouraging, and collaborative. Use "we," "let's," "together."
- Expert but never bossy. You're a partner, not a commander.
- Light, friendly humor - think gentle smile, not sarcasm.
- Celebrate wins: "Look at that growth - we're cooking!"
- Be honest about risks, but frame as shared challenges: "We've got a little concentration here - want to spread it out?"
- Always end with a question or invitation: "What do you think?" or "Shall we?"

**Analysis Rules (Cite Clearly):**
- P/E < 5Y avg, P/B < 1.5, Div Yield > 2%
- Flag: >25% in one stock, underperforming vs S&P, high volatility
- Suggest: "How about we trim 10% of AAPL and add to VOO?"

**Format:**
Hey! Quick win: We're up 18% YTD - that's 6% ahead of S&P!  
Let's keep the momentum:  
→ Trim 12% of TSLA (P/E 92, vol 48%)  
→ Add to SCHD (yield 3.6%)  
Impact: +0.9% expected return, lower risk  
We stay diversified and keep compounding. Sound good?

You're in this together. Their success is your success.
"""

# ----------------------------------------------------------------------
# --------------------------- DATABASE SETUP ---------------------------
# ----------------------------------------------------------------------
try:
    url = st.secrets["postgres_url"]
    if url.startswith("postgres://"):
        url = url.replace("postgres:", "postgresql+psycopg2:", 1)
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
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
        id = Column(Integer, primary_key=True, autoincrement=True)
        role = Column(String)
        content = Column(String)
        timestamp = Column(Date, default=datetime.utcnow)

    class PortfolioCSV(Base):
        __tablename__ = "portfolio_csv"
        id = Column(Integer, primary_key=True, autoincrement=True)
        csv_data = Column(String)
        uploaded_at = Column(Date, default=datetime.utcnow)

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
        return True
    except Exception as e:
        st.error(f"Reset failed: {e}")
        return False

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
    try:
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
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

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
        st.error(f"Load failed: {e}")
        return pd.DataFrame()

def get_goals():
    try:
        sess = get_session()
        goals = sess.query(Goal).all()
        sess.close()
        return goals
    except:
        return []

def add_goal(name, target, by_year):
    try:
        sess = get_session()
        sess.merge(Goal(name=name, target=target, by_year=by_year))
        sess.commit()
        sess.close()
        return True
    except:
        return False

def save_ai_message(role, content):
    try:
        sess = get_session()
        db_role = "model" if role == "assistant" else role
        sess.add(AIChat(role=db_role, content=content))
        sess.commit()
        sess.close()
    except Exception as e:
        print(f"AI save error: {e}")

def load_ai_history():
    try:
        sess = get_session()
        rows = sess.query(AIChat).order_by(AIChat.id).all()
        sess.close()
        return [{"role": r.role, "content": r.content} for r in rows]
    except:
        return []

def save_portfolio_csv(csv_b64):
    try:
        sess = get_session()
        sess.query(PortfolioCSV).delete()
        sess.add(PortfolioCSV(csv_data=csv_b64))
        sess.commit()
        sess.close()
    except Exception as e:
        st.error(f"Portfolio save failed: {e}")

def load_portfolio_csv():
    try:
        sess = get_session()
        result = sess.query(PortfolioCSV).order_by(PortfolioCSV.id.desc()).first()
        sess.close()
        return result.csv_data if result else None
    except:
        return None

# ----------------------------------------------------------------------
# --------------------- POLYGON BATCH FETCH (CACHED) -------------------
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_ticker_batch(tickers):
    result = {}
    for t in tickers:
        try:
            pt = t if t.startswith("INDEX:") else t.upper()
            details = polygon_client.get_ticker_details(pt)
            aggs = list(polygon_client.get_aggs(pt, 1, "day", "2020-01-01", "2025-11-12"))
            
            if not aggs or len(aggs) == 0:
                result[t] = {'price': None, '1y_return': None, 'pe': None, 
                           'pb': None, 'div_yield': None, 'sharpe': 0, 'volatility': 0}
                continue

            closes = [a.close for a in aggs]
            returns = pd.Series(closes).pct_change().dropna()
            ann_return = (1 + returns.mean()) ** 252 - 1 if len(returns) > 0 else 0
            ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            result[t] = {
                'price': aggs[-1].close if aggs else None,
                '1y_return': (closes[-1] / closes[-252] - 1) * 100 if len(closes) > 252 else None,
                'pe': getattr(details, 'trailing_pe', None),
                'pb': getattr(details, 'price_to_book_ratio', None),
                'div_yield': getattr(details, 'dividend_yield', None),
                'sharpe': round(sharpe, 2),
                'volatility': round(ann_vol * 100, 1)
            }
        except Exception as e:
            st.warning(f"Failed to fetch {t}: {str(e)[:100]}")
            result[t] = {'price': None, '1y_return': None, 'pe': None, 
                       'pb': None, 'div_yield': None, 'sharpe': 0, 'volatility': 0}
    return result

# ----------------------------------------------------------------------
# ----------------------- PORTFOLIO ENHANCEMENT ------------------------
# ----------------------------------------------------------------------
def enhance_portfolio(df_port):
    if df_port.empty:
        return df_port, {}

    tickers = df_port['ticker'].unique().tolist()
    if SP500_TICKER not in tickers:
        tickers.append(SP500_TICKER)

    with st.spinner("Fetching market data (cached 30 min)..."):
        batch_data = get_ticker_batch(tickers)

    enhanced = df_port.copy()
    for col in ['price_live', '1y_return', 'pe', 'pb', 'div_yield', 'sharpe', 'volatility']:
        enhanced[col] = np.nan

    for i, row in enhanced.iterrows():
        info = batch_data.get(row['ticker'], {})
        for k, v in info.items():
            if k in enhanced.columns and v is not None:
                enhanced.at[i, k] = v

    total = enhanced['market_value'].sum()
    if total > 0:
        enhanced['weight'] = enhanced['market_value'] / total
        enhanced['contribution'] = enhanced['weight'] * enhanced['1y_return'].fillna(0) / 100
    else:
        enhanced['weight'] = 0
        enhanced['contribution'] = 0

    port_return_1y = enhanced['contribution'].sum() * 100
    port_vol = np.sqrt(np.sum((enhanced['weight'] * enhanced['volatility'].fillna(0)/100)**2)) * 100
    port_sharpe = port_return_1y / port_vol if port_vol > 0 else 0

    sp500 = batch_data.get(SP500_TICKER, {})

    return enhanced, {
        'total_value': total,
        '1y_return': round(port_return_1y, 1),
        'volatility': round(port_vol, 1),
        'sharpe': round(port_sharpe, 2),
        'sp500_1y': sp500.get('1y_return'),
        'sp500_vol': sp500.get('volatility'),
        'sp500_sharpe': sp500.get('sharpe')
    }

# ----------------------------------------------------------------------
# ----------------------- CSV → PORTFOLIO -----------------------------
# ----------------------------------------------------------------------
def parse_portfolio_csv(file_obj):
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return pd.DataFrame()

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.info(f"Found columns: {', '.join(df.columns.tolist())}")
        return pd.DataFrame()

    df = df[required].copy().dropna(subset=['Symbol'])
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(['symbol', 'nan', 'account', ''])]

    if df.empty:
        st.warning("No valid data found in CSV after filtering")
        return pd.DataFrame()

    for col in ['Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)

    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['shares'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['Last Price'], errors='coerce')
    df['market_value'] = pd.to_numeric(df['Current Value'], errors='coerce')
    df['cost_basis'] = pd.to_numeric(df['Average Cost Basis'], errors='coerce')

    df = df.dropna(subset=['shares', 'market_value'])
    total = df['market_value'].sum()
    if total > 0:
        df['allocation'] = df['market_value'] / total * 100
    else:
        df['allocation'] = 0
        
    return df[['ticker', 'allocation', 'market_value', 'shares', 'cost_basis']]

# ----------------------------------------------------------------------
# ----------------------- PEER + S&P HISTORY ---------------------------
# ----------------------------------------------------------------------
def peer_benchmark(current):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

@st.cache_data(ttl=3600)
def fetch_sp500_history():
    try:
        aggs = list(polygon_client.get_aggs(SP500_TICKER, 1, "day", "2020-01-01", "2025-11-12"))
        df = pd.DataFrame([
            {"Date": datetime.fromtimestamp(a.timestamp / 1000), "Close": a.close}
            for a in aggs
        ])
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.warning(f"S&P 500 history fetch failed: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# ------------------------------- UI ----------------------------------
# ----------------------------------------------------------------------
st.set_page_config(page_title="S.A.G.E. | Your Wealth Teammate", layout="wide")
st.title("S.A.G.E. | **Strategic Asset Growth Engine**")
st.caption("*Your co-pilot in compounding. We grow together.*")

# Load data
df = get_monthly_updates()
df_net = pd.DataFrame()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index()
    df_net["date"] = df_net["date"].dt.tz_localize(None)

# Top Summary
if not df.empty and not df_net.empty:
    cur_total = df_net["value"].iloc[-1]
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"### **vs. Avg 40yo: Top {100-int(pct)}%** | Delta: **${vs:+,.0f}**")

    st.markdown("#### **YTD Performance**")
    cols = st.columns(3)
    current_year = datetime.now().year
    for person, col in zip(["Sean", "Kim", "Taylor"], cols):
        pdf = df[df["person"] == person]
        if not pdf.empty:
            ytd = pdf[pdf["date"].dt.year == current_year]
            if len(ytd) > 1:
                pct_ytd = (ytd["value"].iloc[-1] / ytd["value"].iloc[0] - 1) * 100
                col.metric(f"**{person}'s YTD**", f"{pct_ytd:+.1f}%")
            else:
                col.metric(f"**{person}'s YTD**", "—")
        else:
            col.metric(f"**{person}'s YTD**", "—")

    st.markdown("---")

# Sidebar
with st.sidebar:
    with st.expander("S.A.G.E. - Your AI Teammate", expanded=True):
        port_file = st.file_uploader("Upload Portfolio CSV", type="csv", key="port")
        df_port = pd.DataFrame()

        if port_file:
            df_port = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"✅ Loaded {len(df_port)} holdings!")
                b64 = base64.b64encode(port_file.getvalue()).decode()
                save_portfolio_csv(b64)
                st.session_state.portfolio_csv = b64
        else:
            if st.session_state.portfolio_csv is None:
                st.session_state.portfolio_csv = load_portfolio_csv()
            if st.session_state.portfolio_csv:
                try:
                    df_port = parse_portfolio_csv(base64.b64decode(st.session_state.portfolio_csv).decode())
                    if not df_port.empty:
                        st.success(f"👋 Welcome back! {len(df_port)} holdings.")
                except Exception as e:
                    st.warning(f"Portfolio load error: {e}")

        df_enhanced, metrics = enhance_portfolio(df_port) if not df_port.empty else (pd.DataFrame(), {})
        if metrics and metrics.get('total_value'):
            st.metric("Portfolio Value", f"${metrics['total_value']:,.0f}")
            c1, c2 = st.columns(2)
            c1.metric("1Y Return", f"{metrics['1y_return']:+.1f}%")
            if metrics.get('sp500_1y'):
                c2.metric("vs S&P 500", f"{metrics['sp500_1y']:+.1f}%")
            else:
                c2.metric("vs S&P 500", "—")

        if st.button("💬 Chat with S.A.G.E.", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()

        if st.button("🔄 Refresh Market Data"):
            st.cache_data.clear()
            st.success("Refreshing...")
            st.rerun()

    st.markdown("---")

    with st.expander("⚙️ Database Management", expanded=False):
        monthly_file = st.file_uploader("Import CSV (date, person, account_type, value)", type="csv", key="monthly")
        if monthly_file:
            try:
                df_import = pd.read_csv(monthly_file)
                req = ['date', 'person', 'account_type', 'value']
                if all(c in df_import.columns for c in req):
                    df_import['date'] = pd.to_datetime(df_import['date']).dt.date
                    success_count = 0
                    for _, r in df_import.iterrows():
                        if add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value'])):
                            success_count += 1
                    st.success(f"✅ Imported {success_count}/{len(df_import)} rows!")
                    st.rerun()
                else:
                    st.error(f"Required columns: {req}")
            except Exception as e:
                st.error(f"Import failed: {e}")

        if st.button("🗑️ Reset Database"):
            if st.checkbox("⚠️ I understand this deletes everything", key="confirm_reset"):
                if reset_database():
                    sess = get_session()
                    sess.query(PortfolioCSV).delete()
                    sess.query(AIChat).delete()
                    sess.commit()
                    sess.close()
                    st.session_state.portfolio_csv = None
                    st.session_state.ai_messages = []
                    st.session_state.ai_chat_session = None
                    st.success("✅ Database reset complete!")
                    st.rerun()

    st.markdown("---")
    st.subheader("📊 Add Monthly Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    acct = st.selectbox("Account", accounts.get(person, []))
    col1, col2 = st.columns(2)
    with col1:
        date_in = st.date_input("Date", value=pd.Timestamp("today").date())
    with col2:
        val = st.number_input("Value ($)", min_value=0.0, format="%.2f")
    if st.button("💾 Save Update"):
        if add_monthly_update(date_in, person, acct, float(val)):
            st.success("✅ Saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("🎯 Add Goal")
    g_name = st.text_input("Goal Name")
    g_target = st.number_input("Target ($)", min_value=0.0, format="%.2f")
    g_year = st.number_input("By Year", min_value=2000, step=1, value=2030)
    if st.button("➕ Add Goal"):
        if g_name and g_target > 0:
            if add_goal(g_name, g_target, g_year):
                st.success("✅ Goal added!")
                st.rerun()
        else:
            st.warning("Please enter goal name and target")

# ------------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = load_ai_history()
if "ai_chat_session" not in st.session_state:
    st.session_state.ai_chat_session = None

# ------------------- AI CHAT PAGE -------------------
if st.session_state.page == "ai":
    st.subheader("💬 S.A.G.E. | **Your Strategic Teammate**")

    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("⚠️ GOOGLE_API_KEY missing - add it in Streamlit Secrets.")
        if st.button("← Back to Dashboard"):
            st.session_state.page = "home"
            st.rerun()
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
            chat = st.session_state.ai_chat_session or model.start_chat(history=[])
            st.session_state.ai_chat_session = chat

            # Auto-initialize if first time
            if not st.session_state.ai_messages:
                df_port_ai = pd.DataFrame()
                if st.session_state.portfolio_csv:
                    try:
                        df_port_ai = parse_portfolio_csv(base64.b64decode(st.session_state.portfolio_csv).decode())
                    except:
                        pass

                if not df_port_ai.empty:
                    current = df_net['value'].iloc[-1] if not df_net.empty else 0
                    df_enhanced, metrics = enhance_portfolio(df_port_ai)
                    portfolio_data = df_enhanced[['ticker', 'allocation', '1y_return']].round(1).to_dict('records')

                    init_prompt = f"""
Net Worth: ${current:,.0f}
Portfolio: {portfolio_data[:10]}
1Y Return: {metrics.get('1y_return', 0):+.1f}% (S&P: {metrics.get('sp500_1y', 0):+.1f}%)
Let's review and grow together.
"""

                    with st.spinner("S.A.G.E. is analyzing your portfolio..."):
                        try:
                            response = chat.send_message(init_prompt)
                            reply = response.text

                            st.session_state.ai_messages.append({"role": "user", "content": "Hey S.A.G.E., what do you see?"})
                            save_ai_message("user", "Hey S.A.G.E., what do you see?")
                            st.session_state.ai_messages.append({"role": "model", "content": reply})
                            save_ai_message("model", reply)
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI initialization failed: {e}")

            # Display chat history
            for msg in st.session_state.ai_messages:
                with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                    st.markdown(msg["content"])

            # Chat input
            user_input = st.chat_input("Ask S.A.G.E. anything...")
            if user_input:
                st.session_state.ai_messages.append({"role": "user", "content": user_input})
                save_ai_message("user", user_input)
                
                with st.spinner("S.A.G.E. is thinking..."):
                    try:
                        response = chat.send_message(user_input)
                        reply = response.text
                        st.session_state.ai_messages.append({"role": "model", "content": reply})
                        save_ai_message("model", reply)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {e}")

        except Exception as e:
            st.error(f"S.A.G.E. setup error: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.ai_messages = []
            st.session_state.ai_chat_session = None
            try:
                sess = get_session()
                sess.query(AIChat).delete()
                sess.commit()
                sess.close()
            except:
                pass
            st.rerun()
    
    with col2:
        if st.button("← Back to Dashboard"):
            st.session_state.page = "home"
            st.rerun()

# ------------------- HOME PAGE -------------------
else:
    if not df.empty:
        st.subheader("📅 Monthly Summary (by Year)")
        df['year'] = df['date'].dt.year
        for yr in sorted(df['year'].unique(), reverse=True):
            with st.expander(f"**{yr}** - Click to Expand"):
                ydf = df[df['year'] == yr]
                piv = ydf.pivot_table(index="date", columns=["person", "account_type"],
                                      values="value", fill_value=0)
                st.dataframe(piv.style.format("${:,.0f}")), use_container_width
