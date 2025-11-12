import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import time

# === POLYGON.IO CLIENT ===
try:
    from polygon import RESTClient
    POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        st.error("Add POLYGON_API_KEY to Streamlit Secrets")
        st.stop()
    polygon_client = RESTClient(api_key=POLYGON_API_KEY)
except Exception as e:
    st.error(f"Polygon setup failed: {e}")
    st.stop()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine, Column, String, Float, Date, Integer, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
import google.generativeai as genai

if "portfolio_csv" not in st.session_state:
    st.session_state.portfolio_csv = None

PEER_NET_WORTH_40YO = 189_000
SP500_TICKER = "SPY"

SYSTEM_PROMPT = """You are S.A.G.E. - Strategic Asset Growth Engine, a friendly financial co-pilot.
Mission: Help beat S&P 500 by 5%+ annually.
Tone: Warm, collaborative, expert but never bossy. Use "we" and "let's".
Celebrate wins, frame risks as shared challenges. Always end with a question."""

# Database Setup
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
    st.error(f"Database failed: {e}")
    st.stop()

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
    except:
        return False

def get_monthly_updates():
    try:
        sess = get_session()
        rows = sess.query(MonthlyUpdate).all()
        sess.close()
        return pd.DataFrame([{'date': r.date, 'person': r.person, 'account_type': r.account_type, 'value': r.value} for r in rows])
    except:
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
        sess.add(AIChat(role="model" if role == "assistant" else role, content=content))
        sess.commit()
        sess.close()
    except:
        pass

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
    except:
        pass

def load_portfolio_csv():
    try:
        sess = get_session()
        result = sess.query(PortfolioCSV).order_by(PortfolioCSV.id.desc()).first()
        sess.close()
        return result.csv_data if result else None
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_ticker_batch(tickers):
    result = {}
    for i, t in enumerate(tickers):
        if i > 0 and i % 5 == 0:
            time.sleep(1)
        try:
            pt = t.strip().upper()
            if not pt or pt in ['', 'SYMBOL', 'NAN']:
                continue
            
            details = polygon_client.get_ticker_details(pt)
            aggs = list(polygon_client.get_aggs(pt, 1, "day", "2020-01-01", "2025-11-12", limit=50000))
            
            if not aggs:
                result[t] = {'price': None, '1y_return': None, 'sharpe': 0, 'volatility': 0}
                continue

            closes = [a.close for a in aggs]
            returns = pd.Series(closes).pct_change().dropna()
            ann_return = (1 + returns.mean()) ** 252 - 1 if len(returns) > 0 else 0
            ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            result[t] = {
                'price': aggs[-1].close,
                '1y_return': (closes[-1] / closes[-252] - 1) * 100 if len(closes) > 252 else None,
                'sharpe': round(sharpe, 2),
                'volatility': round(ann_vol * 100, 1)
            }
        except:
            result[t] = {'price': None, '1y_return': None, 'sharpe': 0, 'volatility': 0}
    return result

def enhance_portfolio(df_port):
    if df_port.empty:
        return df_port, {}
    tickers = df_port['ticker'].unique().tolist()
    if SP500_TICKER not in tickers:
        tickers.append(SP500_TICKER)
    
    batch_data = get_ticker_batch(tickers)
    enhanced = df_port.copy()
    for col in ['price_live', '1y_return', 'sharpe', 'volatility']:
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
    
    port_return = enhanced['contribution'].sum() * 100
    sp500 = batch_data.get(SP500_TICKER, {})
    
    return enhanced, {
        'total_value': total,
        '1y_return': round(port_return, 1),
        'sp500_1y': sp500.get('1y_return')
    }

def parse_portfolio_csv(file_obj):
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"CSV error: {e}")
        return pd.DataFrame()
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
        return pd.DataFrame()
    
    df = df[required].copy().dropna(subset=['Symbol'])
    df = df[df['Symbol'].astype(str).str.strip() != '']
    
    for col in ['Quantity', 'Last Price', 'Current Value', 'Average Cost Basis']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
    
    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['shares'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['market_value'] = pd.to_numeric(df['Current Value'], errors='coerce')
    df['cost_basis'] = pd.to_numeric(df['Average Cost Basis'], errors='coerce')
    
    df = df.dropna(subset=['shares', 'market_value'])
    total = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total * 100 if total > 0 else 0
    
    return df[['ticker', 'allocation', 'market_value', 'shares', 'cost_basis']]

def peer_benchmark(current):
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs
    # PASTE THIS AFTER PART 1

# UI
st.set_page_config(page_title="S.A.G.E. Dashboard", layout="wide")
st.title("S.A.G.E. | Strategic Asset Growth Engine")

df = get_monthly_updates()
df_net = pd.DataFrame()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df_net = df[df["person"].isin(["Sean", "Kim"])].groupby("date")["value"].sum().reset_index()

if not df.empty and not df_net.empty:
    cur_total = df_net["value"].iloc[-1]
    pct, vs = peer_benchmark(cur_total)
    st.markdown(f"### vs. Avg 40yo: Top {100-int(pct)}% | Delta: ${vs:+,.0f}")
    
    cols = st.columns(3)
    current_year = datetime.now().year
    for person, col in zip(["Sean", "Kim", "Taylor"], cols):
        pdf = df[df["person"] == person]
        if not pdf.empty:
            ytd = pdf[pdf["date"].dt.year == current_year]
            if len(ytd) > 1:
                pct_ytd = (ytd["value"].iloc[-1] / ytd["value"].iloc[0] - 1) * 100
                col.metric(f"{person} YTD", f"{pct_ytd:+.1f}%")

with st.sidebar:
    with st.expander("Portfolio", expanded=True):
        port_file = st.file_uploader("Upload CSV", type="csv")
        df_port = pd.DataFrame()
        
        if port_file:
            df_port = parse_portfolio_csv(port_file)
            if not df_port.empty:
                st.success(f"Loaded {len(df_port)} holdings")
                b64 = base64.b64encode(port_file.getvalue()).decode()
                save_portfolio_csv(b64)
                st.session_state.portfolio_csv = b64
        else:
            if st.session_state.portfolio_csv is None:
                st.session_state.portfolio_csv = load_portfolio_csv()
            if st.session_state.portfolio_csv:
                try:
                    df_port = parse_portfolio_csv(base64.b64decode(st.session_state.portfolio_csv).decode())
                except:
                    pass
        
        df_enhanced, metrics = enhance_portfolio(df_port) if not df_port.empty else (pd.DataFrame(), {})
        if metrics and metrics.get('total_value'):
            st.metric("Value", f"${metrics['total_value']:,.0f}")
            c1, c2 = st.columns(2)
            c1.metric("1Y", f"{metrics['1y_return']:+.1f}%")
            if metrics.get('sp500_1y'):
                c2.metric("S&P", f"{metrics['sp500_1y']:+.1f}%")
        
        if st.button("Chat", disabled=df_port.empty):
            st.session_state.page = "ai"
            st.rerun()
    
    st.markdown("---")
    st.subheader("Monthly Update")
    accounts = load_accounts()
    person = st.selectbox("Person", list(accounts.keys()))
    acct = st.selectbox("Account", accounts.get(person, []))
    date_in = st.date_input("Date")
    val = st.number_input("Value", min_value=0.0)
    if st.button("Save"):
        if add_monthly_update(date_in, person, acct, val):
            st.success("Saved!")
            st.rerun()
    
    st.subheader("Goals")
    g_name = st.text_input("Name")
    g_target = st.number_input("Target", min_value=0.0)
    g_year = st.number_input("Year", min_value=2025, step=1)
    if st.button("Add Goal"):
        if g_name and add_goal(g_name, g_target, g_year):
            st.success("Added!")
            st.rerun()
    
    st.markdown("---")
    with st.expander("⚙️ Database Tools"):
        monthly_file = st.file_uploader("Import Monthly CSV", type="csv", key="monthly")
        if monthly_file:
            try:
                df_import = pd.read_csv(monthly_file)
                req = ['date', 'person', 'account_type', 'value']
                if all(c in df_import.columns for c in req):
                    df_import['date'] = pd.to_datetime(df_import['date']).dt.date
                    count = 0
                    for _, r in df_import.iterrows():
                        if add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value'])):
                            count += 1
                    st.success(f"✅ Imported {count} rows!")
                    st.rerun()
                else:
                    st.error(f"Need columns: {', '.join(req)}")
            except Exception as e:
                st.error(f"Import error: {e}")
        
        if st.button("🗑️ Reset Database"):
            if st.checkbox("⚠️ I understand this deletes ALL data", key="confirm_reset"):
                if reset_database():
                    sess = get_session()
                    try:
                        sess.query(PortfolioCSV).delete()
                        sess.query(AIChat).delete()
                        sess.commit()
                    except:
                        pass
                    finally:
                        sess.close()
                    
                    st.session_state.portfolio_csv = None
                    st.session_state.ai_messages = []
                    st.session_state.ai_chat_session = None
                    st.success("✅ Database reset complete!")
                    st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = load_ai_history()
if "ai_chat_session" not in st.session_state:
    st.session_state.ai_chat_session = None

if st.session_state.page == "ai":
    st.subheader("Chat with S.A.G.E.")
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    
    if not api_key:
        st.warning("Add GOOGLE_API_KEY to secrets")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
            chat = st.session_state.ai_chat_session or model.start_chat()
            st.session_state.ai_chat_session = chat
            
            if not st.session_state.ai_messages and st.session_state.portfolio_csv:
                try:
                    df_p = parse_portfolio_csv(base64.b64decode(st.session_state.portfolio_csv).decode())
                    if not df_p.empty:
                        current = df_net['value'].iloc[-1] if not df_net.empty else 0
                        data = df_p[['ticker', 'allocation']].head(5).to_dict('records')
                        prompt = f"Net Worth: ${current:,.0f}, Holdings: {data}. Brief analysis?"
                        
                        response = chat.send_message(prompt)
                        st.session_state.ai_messages.append({"role": "user", "content": "Analyze portfolio"})
                        st.session_state.ai_messages.append({"role": "model", "content": response.text})
                        save_ai_message("user", "Analyze portfolio")
                        save_ai_message("model", response.text)
                        st.rerun()
                except:
                    pass
            
            for msg in st.session_state.ai_messages:
                with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                    st.markdown(msg["content"])
            
            user_input = st.chat_input("Ask anything...")
            if user_input:
                st.session_state.ai_messages.append({"role": "user", "content": user_input})
                save_ai_message("user", user_input)
                response = chat.send_message(user_input)
                st.session_state.ai_messages.append({"role": "model", "content": response.text})
                save_ai_message("model", response.text)
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()

else:
    if not df.empty:
        st.subheader("Monthly Data")
        df['year'] = df['date'].dt.year
        for yr in sorted(df['year'].unique(), reverse=True):
            with st.expander(f"{yr}"):
                ydf = df[df['year'] == yr]
                piv = ydf.pivot_table(index="date", columns=["person", "account_type"], values="value", fill_value=0)
                st.dataframe(piv.style.format("${:,.0f}"), use_container_width=True)
        
        st.subheader("Net Worth")
        fig = px.line(df_net, x="date", y="value")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Goals")
        goals = get_goals()
        if goals and not df_net.empty:
            cur = df_net["value"].iloc[-1]
            for g in goals:
                prog = min(cur / g.target, 1.0) if g.target > 0 else 0
                st.progress(prog)
                st.write(f"{g.name}: ${cur:,.0f} / ${g.target:,.0f} ({g.by_year})")
        
        if not df_enhanced.empty:
            st.subheader("Portfolio Holdings")
            display_df = df_enhanced[['ticker', 'allocation', 'market_value', '1y_return']].copy()
            st.dataframe(
                display_df.style.format({
                    'allocation': '{:.1f}%',
                    'market_value': '${:,.0f}',
                    '1y_return': '{:+.1f}%'
                }),
                use_container_width=True
            )
        
        st.download_button(
            "📥 Export Data",
            df.to_csv(index=False).encode(),
            "monthly_data.csv",
            "text/csv"
        )
    else:
        st.info("👋 Welcome! Upload your portfolio CSV in the sidebar and add monthly updates to get started.")
        st.markdown("""
        ### Quick Start:
        1. **Upload Portfolio CSV** - Include columns: Symbol, Quantity, Last Price, Current Value, Average Cost Basis
        2. **Add Monthly Updates** - Track your net worth over time
        3. **Set Goals** - Monitor your progress toward financial targets
        4. **Chat with S.A.G.E.** - Get AI-powered portfolio insights
        """)
