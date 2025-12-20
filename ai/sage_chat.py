# ai/sage_chat.py - Upgraded to Claude 3.5 Sonnet

import anthropic
import streamlit as st
from datetime import datetime

from config.constants import SYSTEM_PROMPT
from database.operations import save_ai_message
from analysis.projections import calculate_projection_cone, calculate_confidence_score

def get_real_time_context():
    """Placeholder — we'll add real news later with tools"""
    return "(Using latest knowledge of markets and economy for analysis)"

def generate_initial_analysis(client, df_net, df_port, port_summary, retirement_target):
    confidence, conf_method = calculate_confidence_score(df_net, retirement_target)
    years_left = 2042 - datetime.now().year
    current_nw = df_net['value'].iloc[-1] if not df_net.empty else 0
    progress = (current_nw / retirement_target) * 100 if retirement_target > 0 else 0
    
    future_dates, conservative, current_pace, optimistic = calculate_projection_cone(
        df_net, retirement_target
    )
    
    projected_2042 = current_pace[-1] if current_pace and len(current_pace) > 0 else current_nw
    conservative_2042 = conservative[-1] if conservative and len(conservative) > 0 else 'N/A'
    optimistic_2042 = optimistic[-1] if optimistic and len(optimistic) > 0 else 'N/A'
    
    real_time = get_real_time_context()
    
    conservative_str = f"${conservative_2042:,.0f}" if isinstance(conservative_2042, (int, float)) else str(conservative_2042)
    optimistic_str = f"${optimistic_2042:,.0f}" if isinstance(optimistic_2042, (int, float)) else str(optimistic_2042)

    prompt = f"""
Current date: {datetime.now().strftime('%B %d, %Y')}

RETIREMENT GOAL: ${retirement_target:,.0f} by 2042 ({years_left} years left)
CURRENT NET WORTH (Sean + Kim): ${current_nw:,.0f}
PROGRESS: {progress:.1f}%
CONFIDENCE: {confidence:.0f}% ({conf_method})

PROJECTIONS TO 2042:
- Conservative (7% real): {conservative_str}
- Current Pace: ${projected_2042:,.0f}
- Optimistic (1.5x pace): {optimistic_str}

PORTFOLIO SNAPSHOT:
- Holdings: {len(df_port)} positions
- Total Value: ${port_summary.get('total_value', 0):,.0f}
- Unrealized Gain: {port_summary.get('total_gain_pct', 0):+.1f}%
- Top Holding: {port_summary.get('top_holding', 'N/A')} ({port_summary.get('top_allocation', 0):.1f}%)

REAL-TIME CONTEXT:
{real_time}

{SYSTEM_PROMPT}

Give me your full, deep strategic analysis as my best-friend financial genius team.
Be warm, direct, proactive, and back everything with logic.
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.7,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return prompt, response.content[0].text
    except Exception as e:
        st.error(f"Initial analysis failed: {e}")
        return prompt, None

def init_chat(api_key, history):
    client = anthropic.Anthropic(api_key=api_key)
    return client

def send_message(client, user_input, history):
    messages = [{"role": m["role"] if m["role"] == "assistant" else "user", "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.7,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"Sorry, I hit an error: {str(e)}"
