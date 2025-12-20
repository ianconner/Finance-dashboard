# analysis/projections.py - Fixed with better error handling

import pandas as pd
from datetime import datetime

def calculate_projection_cone(df_net, target_amount, target_year=2042):
    """
    Returns future_dates, conservative, current_pace, optimistic projections
    Fixed: Added division by zero protection
    """
    if df_net.empty or len(df_net) < 2:
        return None, None, None, None
    
    current_value = df_net['value'].iloc[-1]
    current_date = df_net['date'].iloc[-1]
    
    if current_value <= 0:
        return None, None, None, None
    
    # Months until retirement
    months_to_retirement = (target_year - current_date.year) * 12 + (12 - current_date.month)
    if months_to_retirement <= 0:
        return None, None, None, None
    
    future_dates = pd.date_range(
        start=current_date + pd.DateOffset(months=1),
        periods=months_to_retirement,
        freq='ME'
    )
    
    # Conservative: 7% annual (historical S&P real return)
    sp_monthly = (1 + 0.07) ** (1/12) - 1
    conservative = [current_value * ((1 + sp_monthly) ** (i + 1)) for i in range(months_to_retirement)]
    
    # Current Pace: Historical CAGR
    df_sorted = df_net.sort_values('date')
    years_elapsed = (df_sorted['date'].iloc[-1] - df_sorted['date'].iloc[0]).days / 365.25
    start_val = df_sorted['value'].iloc[0]
    
    if years_elapsed > 0 and start_val > 0:
        cagr = (current_value / start_val) ** (1 / years_elapsed) - 1
        cagr = max(-0.5, min(0.5, cagr))  # Cap at +/-50% to avoid extremes
    else:
        cagr = 0.07  # fallback to 7%
    
    current_monthly = (1 + cagr) ** (1/12) - 1
    current_pace = [current_value * ((1 + current_monthly) ** (i + 1)) for i in range(months_to_retirement)]
    
    # Optimistic: 1.5x current pace (capped at 20% annual)
    opt_cagr = min(cagr * 1.5, 0.20)
    opt_monthly = (1 + opt_cagr) ** (1/12) - 1
    optimistic = [current_value * ((1 + opt_monthly) ** (i + 1)) for i in range(months_to_retirement)]
    
    return future_dates, conservative, current_pace, optimistic


def calculate_confidence_score(df_net, target_amount, target_year=2042):
    """
    Calculate confidence percentage with detailed reasoning
    Fixed: Added better error handling and validation
    """
    if df_net.empty or len(df_net) < 2:
        return 50.0, "Insufficient data"
    
    current_value = df_net['value'].iloc[-1]
    
    if current_value <= 0:
        return 10.0, "Invalid portfolio value"
    
    current_date = df_net['date'].iloc[-1]
    months_remaining = (target_year - current_date.year) * 12 + (12 - current_date.month)
    
    if months_remaining <= 0:
        return 100.0 if current_value >= target_amount else 0.0, "At/past retirement"
    
    df_sorted = df_net.sort_values('date')
    years_elapsed = (df_sorted['date'].iloc[-1] - df_sorted['date'].iloc[0]).days / 365.25
    
    if years_elapsed < 0.5:
        return 50.0, "Need 6+ months of data"
    
    start_value = df_sorted['value'].iloc[0]
    if start_value <= 0:
        return 50.0, "Invalid starting value"
    
    # Calculate CAGR with safety checks
    try:
        cagr = (current_value / start_value) ** (1 / years_elapsed) - 1
        cagr = max(-0.5, min(0.5, cagr))  # cap extremes
    except (ZeroDivisionError, ValueError):
        return 50.0, "Calculation error"
    
    monthly_rate = (1 + cagr) ** (1/12) - 1
    projected = current_value * ((1 + monthly_rate) ** months_remaining)
    
    if target_amount <= 0:
        return 50.0, "Invalid target amount"
    
    ratio = projected / target_amount
    
    # Base confidence calculation
    if ratio >= 1.5:
        confidence = 95.0
        method = "Exceeding target significantly"
    elif ratio >= 1.2:
        confidence = 90.0
        method = "Well above target"
    elif ratio >= 1.0:
        confidence = 80.0
        method = "On track"
    elif ratio >= 0.9:
        confidence = 70.0
        method = "Close – minor gap"
    elif ratio >= 0.75:
        confidence = 55.0
        method = "Below target – adjustment recommended"
    elif ratio >= 0.5:
        confidence = 35.0
        method = "Significant gap"
    else:
        confidence = 20.0
        method = "Major adjustment required"
    
    # Volatility penalty
    if len(df_sorted) > 2:
        df_sorted['monthly_return'] = df_sorted['value'].pct_change()
        volatility = df_sorted['monthly_return'].std()
        if pd.notna(volatility) and volatility > 0:
            volatility_penalty = min(20, volatility * 100)
            confidence = max(5, confidence - volatility_penalty)
    
    # Time penalty for < 5 years remaining
    years_remaining = months_remaining / 12
    if years_remaining < 5:
        time_penalty = (5 - years_remaining) * 2
        confidence = max(5, confidence - time_penalty)
    
    return round(confidence, 1), method
