# pages/ai_chat_page.py

import streamlit as st
import google.generativeai as genai
from datetime import datetime

from config.constants import SYSTEM_PROMPT
from database.operations import load_ai_history, save_ai_message
from ai.sage_chat import init_chat, generate_initial_analysis, send_user_message

def show_ai_chat_page(df, df_net, df_port, port_summary):
    st.subheader("🧠 S.A.G.E. | Strategic Asset Growth Engine")
    st.caption("Your long-term teammate — let's review and refine together.")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Add `GOOGLE_API_KEY` in Streamlit Secrets to enable S.A.G.E.")
        st.stop()

    # Initialize chat session
    if "sage_chat" not in st.session_state:
        history = load_ai_history()
        st.session_state.ai_messages = history
        st.session_state.sage_chat = init_chat(api_key, history)

    chat = st.session_state.sage_chat

    # Auto-generate initial analysis if chat is empty and we have data
    if not st.session_state.ai_messages and not df_net.empty and not df_port.empty:
        retirement_target = st.session_state.get("retirement_goal", 1000000)
        if "retirement_goal" not in st.session_state:
            from database.operations import get_retirement_goal
            retirement_target = get_retirement_goal()
            st.session_state.retirement_goal = retirement_target

        with st.spinner("S.A.G.E. is analyzing your full financial picture..."):
            user_prompt, reply = generate_initial_analysis(
                chat, df_net, df_port, port_summary, retirement_target
            )
            if reply:
                st.session_state.ai_messages.append({"role": "user", "content": user_prompt})
                save_ai_message("user", user_prompt)
                st.session_state.ai_messages.append({"role": "model", "content": reply})
                save_ai_message("model", reply)
                st.rerun()

    # Display chat history
    for msg in st.session_state.ai_messages:
        role = "assistant" if msg["role"] == "model" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Ask S.A.G.E. anything — rebalance? risk? taxes? retirement?"):
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        save_ai_message("user", user_input)
        
        with st.spinner("S.A.G.E. is thinking..."):
            reply = send_user_message(chat, user_input)
            if reply:
                st.session_state.ai_messages.append({"role": "model", "content": reply})
                save_ai_message("model", reply)
            st.rerun()

    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
