# ai/sage_chat.py

import google.generativeai as genai
import streamlit as st
from config.constants import SYSTEM_PROMPT
from database.operations import save_ai_message
from analysis.projections import calculate_projection_cone, calculate_confidence_score

def init_chat(api_key, history):
    """Initialize Gemini chat with system prompt and history"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
    
    formatted_history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in history
        if isinstance(m, dict) and "role" in m and "content" in m
    ]
    
    return model.start_chat(history=formatted_history)

def generate_initial_analysis(chat, df_net, df_port, port_summary, retirement_target):
    """Generate the automatic first message from S.A.G.E. when opening chat"""
    from datetime import datetime
    
    years_to_retirement = 2042 - datetime.now().year
    
    # Get projections
    future_dates, conservative, current_pace, optimistic = calculate_projection_cone(
        df_net, retirement_target, 2042
    )
    projected_2042 = current_pace[-1] if current_pace is not None else df_net['value'].iloc[-1]
    
    # Get confidence
    confidence, conf_method = calculate_confidence_score(df_net, retirement_target, 2042)
    
    init_prompt = f"""Here's our current situation (as of {datetime.now().strftime('%B %d, %Y')}):

**RETIREMENT GOAL**: ${retirement_target:,.0f} by 2042 ({years_to_retirement} years remaining)

**CURRENT STATUS**:
- Net Worth (Sean + Kim): ${df_net['value'].iloc[-1]:,.0f}
- Progress: {(df_net['value'].iloc[-1] / retirement_target) * 100:.1f}%
- Confidence Score: {confidence:.0f}% ({conf_method})

**PROJECTIONS TO 2042**:
- Conservative (S&P 7%): ${conservative[-1]:,.0f if conservative else 0:,.0f}
- Current Pace: ${projected_2042:,.0f}
- Optimistic (1.5x): ${optimistic[-1]:,.0f if optimistic else 0:,.0f}

**PORTFOLIO SNAPSHOT**:
- Holdings: {len(df_port)} positions
- Total Value: ${port_summary.get('total_value', 0):,.0f}
- Total Gain: ${port_summary.get('total_gain', 0):,.0f} ({port_summary.get('total_gain_pct', 0):+.1f}%)
- Top Holding: {port_summary.get('top_holding', 'N/A')} ({port_summary.get('top_allocation', 0):.1f}%)

Give me your full analysis: Are we on track? Any red flags? What should we focus on next?"""

    try:
        response = chat.send_message(init_prompt)
        return init_prompt, response.text
    except Exception as e:
        st.error(f"Failed to generate initial S.A.G.E. analysis: {e}")
        return init_prompt, None

def send_user_message(chat, user_input):
    """Send a user message and get response"""
    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        st.error(f"AI response failed: {e}")
        return None

def save_message(role, content):
    """Save message to database (wrapper)"""
    save_ai_message(role, content)
