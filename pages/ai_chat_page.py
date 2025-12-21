# pages/ai_chat_page.py - FIXED to use all data sources

import streamlit as st
import pandas as pd
from datetime import datetime
import base64

from database.operations import load_ai_history, save_ai_message, get_session, get_retirement_goal, get_monthly_updates
from database.models import AIChat
from ai.sage_chat import init_chat, generate_initial_analysis, send_message
from data.parser import parse_portfolio_csv, merge_portfolios

def show_ai_chat_page(df, df_net, df_port, port_summary):
    st.title("🧠 S.A.G.E. | Strategic Asset Growth Engine")
    st.caption("Your best-friend genius financial team — always here, always remembering.")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Add `GOOGLE_API_KEY` in Streamlit Secrets")
        st.stop()

    # Load ALL data sources for comprehensive analysis
    
    # 1. Get historical monthly data from database
    df = get_monthly_updates()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    
    # 2. Get current portfolio from all uploaded CSVs
    from database.operations import load_all_portfolios
    
    all_b64 = load_all_portfolios()
    raw_portfolio_data = []
    
    for slot, b64_data in all_b64.items():
        try:
            decoded = base64.b64decode(b64_data).decode('utf-8')
            parsed_df, _ = parse_portfolio_csv(decoded, show_analysis=False)
            if not parsed_df.empty:
                raw_portfolio_data.append(parsed_df)
        except Exception as e:
            st.error(f"Error loading portfolio slot {slot}: {e}")
    
    # Merge all portfolios
    if raw_portfolio_data:
        df_port, port_summary = merge_portfolios(raw_portfolio_data)
    else:
        df_port = pd.DataFrame()
        port_summary = {}
    
    # 3. Calculate net worth from database history
    df_sean_kim = df[df["person"].isin(["Sean", "Kim"])]
    df_net = df_sean_kim.groupby("date")["value"].sum().reset_index().sort_values("date")
    
    # 4. Get current totals from portfolios (most accurate)
    current_sean = 0
    current_kim = 0
    current_taylor = 0
    
    if not df_port.empty and 'person' in df_port.columns:
        person_totals = df_port.groupby('person')['market_value'].sum()
        current_sean = person_totals.get('sean', 0)
        current_kim = person_totals.get('kim', 0)
        current_taylor = person_totals.get('taylor', 0)
    
    current_net_worth = current_sean + current_kim
    
    # Show data summary at top
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Historical Data Points", len(df))
            st.metric("Date Range", f"{df['date'].min().strftime('%b %Y') if not df.empty else 'N/A'} - {df['date'].max().strftime('%b %Y') if not df.empty else 'N/A'}")
        with col2:
            st.metric("Current Portfolio Holdings", len(df_port))
            st.metric("CSV Files Loaded", len(all_b64))
        with col3:
            st.metric("Sean Total", f"${current_sean:,.0f}")
            st.metric("Kim Total", f"${current_kim:,.0f}")
            st.metric("Taylor Total", f"${current_taylor:,.0f}")

    # Session state
    if "sage_chat" not in st.session_state:
        history = load_ai_history()
        st.session_state.ai_messages = history
        st.session_state.sage_chat = init_chat(api_key, history)

    chat = st.session_state.sage_chat
    retirement_target = get_retirement_goal()

    # Header with buttons
    col1, col2, col3 = st.columns([5, 2, 2])
    with col1:
        st.markdown(f"**Retirement Goal:** ${retirement_target:,.0f} by 2042 • **Current:** ${current_net_worth:,.0f}")
    with col2:
        if st.button("🔄 Run Full Analysis", use_container_width=True, type="primary"):
            with st.spinner("S.A.G.E. is running a fresh full review..."):
                user_prompt, reply = generate_initial_analysis(
                    chat, df_net, df_port, port_summary, retirement_target
                )
                if reply:
                    st.session_state.ai_messages.append({"role": "user", "content": "[Full Analysis Requested]\n\n" + user_prompt})
                    save_ai_message("user", user_prompt)
                    st.session_state.ai_messages.append({"role": "model", "content": reply})
                    save_ai_message("model", reply)
                    st.rerun()
    with col3:
        if st.button("🗑️ New Conversation", use_container_width=True):
            st.session_state.ai_messages = []
            st.session_state.sage_chat = init_chat(api_key, [])
            sess = get_session()
            try:
                sess.query(AIChat).delete()
                sess.commit()
            finally:
                sess.close()
            st.success("Started fresh conversation!")
            st.rerun()

    # Auto deep analysis on first load
    if not st.session_state.ai_messages and not df_net.empty:
        with st.spinner("S.A.G.E. is running a full strategic review..."):
            user_prompt, reply = generate_initial_analysis(
                chat, df_net, df_port, port_summary, retirement_target
            )
            if reply:
                st.session_state.ai_messages.append({"role": "user", "content": user_prompt})
                save_ai_message("user", user_prompt)
                st.session_state.ai_messages.append({"role": "model", "content": reply})
                save_ai_message("model", reply)
                st.rerun()

    # Display history
    for msg in st.session_state.ai_messages:
        role = "assistant" if msg["role"] == "model" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Talk strategy, risks, opportunities, or anything..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        save_ai_message("user", prompt)
        
        with st.spinner("S.A.G.E. team analyzing..."):
            reply = send_message(chat, prompt)
            st.session_state.ai_messages.append({"role": "model", "content": reply})
            save_ai_message("model", reply)
            st.rerun()

    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
