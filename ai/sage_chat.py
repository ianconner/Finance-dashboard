# ai/sage_chat.py

import google.generativeai as genai
import streamlit as st
from datetime import datetime
import json

from config.constants import SYSTEM_PROMPT
from database.operations import save_ai_message, load_ai_history
from analysis.projections import calculate_projection_cone, calculate_confidence_score

# ------------------------------------------------------------------
# Tool: Real-time news & events (web search + browse)
# ------------------------------------------------------------------
def get_real_time_context():
    """Fetch current market-moving news and events"""
    try:
        from xai.tools import web_search, browse_page
        
        # Search for today's key financial news
        results = web_search(
            query="top financial market news today OR Fed OR inflation OR tariffs OR recession OR earnings site:reuters.com OR site:bloomberg.com OR site:cnbc.com OR site:wsj.com",
            num_results=8
        )
        
        news_summary = "Recent key events impacting markets:\n"
        for r in results[:5]:
            title = r.get('title', 'No title')
            snippet = r.get('snippet', '')[:200]
            news_summary += f"• {title}: {snippet}\n"
        
        return news_summary.strip()
    
    except Exception as e:
        return f"(Real-time news fetch failed: {str(e)} — proceeding with stored knowledge)"

# ------------------------------------------------------------------
# Enhanced initial analysis
# ------------------------------------------------------------------
def generate_comprehensive_analysis(chat, df_net, df_port, port_summary, retirement_target):
    confidence, conf_method = calculate_confidence_score(df_net, retirement_target)
    years_left = 2042 - datetime.now().year
    current_nw = df_net['value'].iloc[-1]
    progress = (current_nw / retirement_target) * 100
    
    future_dates, conservative, current_pace, optimistic = calculate_projection_cone(
        df_net, retirement_target
    )
    projected_2042 = current_pace[-1] if current_pace else current_nw
    
    # Real-time context
    real_time = get_real_time_context()
    
    prompt = f"""
Current date: {datetime.now().strftime('%B %d, %Y')}

RETIREMENT GOAL: ${retirement_target:,.0f} by 2042 ({years_left} years left)
CURRENT NET WORTH (Sean + Kim): ${current_nw:,.0f}
PROGRESS: {progress:.1f}%
CONFIDENCE: {confidence:.0f}% ({conf_method})

PROJECTIONS TO 2042:
- Conservative (7% real): ${conservative[-1]:,.0f if conservative else 0:,.0f}
- Current Pace: ${projected_2042:,.0f}
- Optimistic (1.5x pace): ${optimistic[-1]:,.0f if optimistic else 0:,.0f}

PORTFOLIO SNAPSHOT:
- {len(df_port)} holdings
- Total value: ${port_summary.get('total_value', 0):,.0f}
- Unrealized gain: {port_summary.get('total_gain_pct', 0):+.1f}%
- Top holding: {port_summary.get('top_holding', 'N/A')} ({port_summary.get('top_allocation', 0):.1f}%)

REAL-TIME MARKET CONTEXT:
{real_time}

Give me your full, deep analysis as my best-friend financial genius team:
- Are we on/exceeding/behind pace?
- Biggest risks and opportunities right now?
- Any concentration or sector imbalances?
- Tax moves worth considering?
- Specific rebalance or action recommendations?
- How current news/events could affect us and what we should do?

Be direct, warm, proactive, and back everything with logic.
"""

    try:
        response = chat.send_message(prompt)
        return prompt, response.text
    except Exception as e:
        st.error(f"Analysis generation failed: {e}")
        return prompt, None

# ------------------------------------------------------------------
# Main functions
# ------------------------------------------------------------------
def init_chat(api_key, history):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro', system_instruction=SYSTEM_PROMPT)  # upgraded to pro for depth
    formatted = [{"role": m["role"], "parts": [m["content"]]} for m in history]
    return model.start_chat(history=formatted)

def send_message(chat, user_input):
    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"
