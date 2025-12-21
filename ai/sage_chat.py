# ai/sage_chat.py - Enhanced with comprehensive portfolio analysis

import google.generativeai as genai
import streamlit as st
from datetime import datetime

from config.constants import SYSTEM_PROMPT
from database.operations import save_ai_message
from analysis.projections import calculate_projection_cone, calculate_confidence_score

def get_real_time_context():
    """Simple placeholder — Gemini has up-to-date knowledge"""
    return "(Using latest market and economic knowledge)"

def generate_initial_analysis(chat, df_net, df_port, port_summary, retirement_target):
    """Generate deep initial strategic analysis with ALL available data"""
    
    # Calculate confidence and projections
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
    
    # Safe string formatting
    conservative_str = f"${conservative_2042:,.0f}" if isinstance(conservative_2042, (int, float)) else str(conservative_2042)
    optimistic_str = f"${optimistic_2042:,.0f}" if isinstance(optimistic_2042, (int, float)) else str(optimistic_2042)

    # Build comprehensive portfolio breakdown
    portfolio_details = ""
    if not df_port.empty:
        # Breakdown by person
        if 'person' in df_port.columns:
            person_totals = df_port.groupby('person')['market_value'].sum()
            portfolio_details += "\n\nPORTFOLIO BREAKDOWN BY PERSON:\n"
            for person, total in person_totals.items():
                portfolio_details += f"- {person.upper()}: ${total:,.2f}\n"
        
        # Top holdings
        if len(df_port) > 0:
            top_10 = df_port.nlargest(10, 'market_value')[['ticker', 'name', 'market_value', 'allocation', 'pct_gain', 'person']]
            portfolio_details += "\n\nTOP 10 HOLDINGS:\n"
            for idx, row in top_10.iterrows():
                portfolio_details += f"- {row['ticker']} ({row.get('name', 'N/A')}): ${row['market_value']:,.2f} ({row['allocation']*100:.1f}%) - Gain: {row['pct_gain']:+.1f}% - Owner: {row.get('person', 'unknown').upper()}\n"
        
        # Sector/allocation analysis
        if 'person' in df_port.columns:
            portfolio_details += "\n\nACCOUNT ALLOCATION:\n"
            account_summary = df_port.groupby(['person', 'account_name'])['market_value'].sum().reset_index()
            for _, row in account_summary.iterrows():
                portfolio_details += f"- {row['person'].upper()} - {row['account_name']}: ${row['market_value']:,.2f}\n"
        
        # Performance summary
        total_gain = df_port['unrealized_gain'].sum()
        total_cost = df_port['cost_basis'].sum()
        overall_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
        portfolio_details += f"\n\nOVERALL PERFORMANCE:\n"
        portfolio_details += f"- Total Cost Basis: ${total_cost:,.2f}\n"
        portfolio_details += f"- Current Value: ${port_summary.get('total_value', 0):,.2f}\n"
        portfolio_details += f"- Unrealized Gain: ${total_gain:,.2f} ({overall_gain_pct:+.1f}%)\n"

    # Historical performance
    historical_summary = ""
    if not df_net.empty and len(df_net) >= 2:
        start_date = df_net['date'].iloc[0]
        end_date = df_net['date'].iloc[-1]
        start_value = df_net['value'].iloc[0]
        end_value = df_net['value'].iloc[-1]
        years_elapsed = (end_date - start_date).days / 365.25
        
        if years_elapsed > 0 and start_value > 0:
            total_return = ((end_value / start_value) - 1) * 100
            cagr = ((end_value / start_value) ** (1 / years_elapsed) - 1) * 100
            
            historical_summary = f"\n\nHISTORICAL PERFORMANCE ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}):\n"
            historical_summary += f"- Starting Value: ${start_value:,.2f}\n"
            historical_summary += f"- Current Value: ${end_value:,.2f}\n"
            historical_summary += f"- Total Return: {total_return:+.1f}%\n"
            historical_summary += f"- CAGR: {cagr:.1f}%\n"
            historical_summary += f"- Time Period: {years_elapsed:.1f} years\n"
            
            # Recent performance (last 3 months)
            if len(df_net) >= 4:
                recent_df = df_net.tail(4)
                recent_start = recent_df['value'].iloc[0]
                recent_end = recent_df['value'].iloc[-1]
                recent_return = ((recent_end / recent_start) - 1) * 100
                historical_summary += f"- Last 3 Months: {recent_return:+.1f}%\n"

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
{portfolio_details}
{historical_summary}

REAL-TIME CONTEXT:
{real_time}

{SYSTEM_PROMPT}

Give me your full, deep strategic analysis as my best-friend financial genius team.
Analyze ALL the data provided - individual accounts, holdings, performance, trends, everything.
Be warm, direct, proactive, and back everything with logic.
Focus on actionable insights and specific recommendations based on what you see.
"""

    try:
        response = chat.send_message(prompt)
        return prompt, response.text
    except Exception as e:
        st.error(f"Initial analysis failed: {e}")
        return prompt, None

def init_chat(api_key, history):
    """Initialize Gemini 2.5 Flash chat"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=SYSTEM_PROMPT
    )
    formatted_history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in history
    ]
    return model.start_chat(history=formatted_history)

def send_message(chat, user_input):
    """Send a user message and get Gemini response"""
    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"
