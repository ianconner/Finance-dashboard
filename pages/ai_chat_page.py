# pages/ai_chat_page.py

import streamlit as st
from datetime import datetime

from database.operations import load_ai_history, save_ai_message
from ai.sage_chat import init_chat, generate_comprehensive_analysis, send_message

def show_ai_chat_page(df, df_net, df_port, port_summary):
    st.title("🧠 S.A.G.E. | Strategic Asset Growth Engine")
    st.caption("Your best-friend genius financial team — always here, always remembering.")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Add `GOOGLE_API_KEY` in Streamlit Secrets")
        st.stop()

    # Session state
    if "sage_chat" not in st.session_state:
        history = load_ai_history()
        st.session_state.ai_messages = history
        st.session_state.sage_chat = init_chat(api_key, history)

    chat = st.session_state.sage_chat
    retirement_target = st.session_state.get("retirement_goal", 1000000.0)

    # New Conversation Button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.write(f"**Goal:** ${retirement_target:,.0f} by 2042")
    with col2:
        if st.button("🗑️ New Conversation"):
            st.session_state.ai_messages = []
            st.session_state.sage_chat = init_chat(api_key, [])
            from database.operations import get_session
            sess = get_session()
            sess.query(type('AIChat', (), {})).delete()  # clear DB history
            sess.commit()
            sess.close()
            st.rerun()

    # Auto deep analysis on first load
    if not st.session_state.ai_messages and not df_net.empty:
        with st.spinner("S.A.G.E. is running a full strategic review..."):
            user_prompt, reply = generate_comprehensive_analysis(
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
