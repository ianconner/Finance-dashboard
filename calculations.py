# ============================================================================
# CALCULATIONS.PY - All Financial Calculations (WITH FIXES)
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from config import PEER_NET_WORTH_40YO

# ============================================================================
# PEER BENCHMARK
# ============================================================================

def peer_benchmark(current: float):
    """Calculate peer benchmark vs average 40-year-old"""
    vs = current - PEER_NET_WORTH_40YO
    pct = min(100, max(0, (current / PEER_NET_WORTH_40YO) * 50))
    return pct, vs

# ============================================================================
# PORTFOLIO PARSING
# ============================================================================

def parse_portfolio_csv(file_obj):
    """Parse Fidelity portfolio CSV"""
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']
    
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        return pd.DataFrame(), {}

    # Check for required columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame(), {}

    # Clean data
    df = df[required + ['Account Name']].copy()
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name', ''])]

    if df.empty:
        return pd.DataFrame(), {}

    # Convert numeric columns
    for col in ['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total'])
    if df.empty:
        return pd.DataFrame(), {}

    # Calculate metrics
    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['market_value'] = df['Current Value']
    df['cost_basis'] = df['Cost Basis Total']
    df['shares'] = df['Quantity']
    df['price'] = df['Last Price']
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = (df['unrealized_gain'] / df['cost_basis']) * 100

    total_value = df['market_value'].sum()
    df['allocation'] = df['market_value'] / total_value

    # Summary stats
    summary = {
        'total_value': total_value,
        'total_cost': df['cost_basis'].sum(),
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / df['cost_basis'].sum()) * 100,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else None,
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }

    return df[['ticker', 'shares', 'price', 'market_value', 'cost_basis', 'unrealized_gain', 'pct_gain', 'allocation']], summary

# ============================================================================
# PROJECTION CONE (FIXED)
# ============================================================================

def calculate_projection_cone(df_net, target_amount, target_year=2042):
    """
    Calculate 3 projection lines to retirement - FIXED VERSION
    
    FIXES:
    - Requires minimum 6 months of data for CAGR
    - Caps growth rates at ±50% annually
    - Falls back to S&P baseline if data insufficient
    """
    if df_net.empty or len(df_net) < 2:
        return None, None, None, None
    
    current_value = df_net['value'].iloc[-1]
    current_date = df_net['date'].iloc[-1]
    
    months_to_retirement = (target_year - current_date.year) * 12 - current_date.month
    if months_to_retirement <= 0:
        return None, None, None, None
    
    future_dates = pd.date_range(
        start=current_date + pd.DateOffset(months=1),
        periods=months_to_retirement,
        freq='ME'
    )
    
    # 1. CONSERVATIVE: S&P 500 baseline (7% annual real returns)
    sp500_annual_rate = 0.07
    sp500_monthly_rate = (1 + sp500_annual_rate) ** (1/12) - 1
    conservative = [current_value * ((1 + sp500_monthly_rate) ** (i + 1)) 
                   for i in range(months_to_retirement)]
    
    # 2. CURRENT PACE: Calculate actual historical growth rate (FIXED)
    df_sorted = df_net.sort_values('date').copy()
    start_date = df_sorted['date'].iloc[0]
    end_date = df_sorted['date'].iloc[-1]
    years_elapsed = (end_date - start_date).days / 365.25
    
    # FIX: Require minimum 6 months of data for reliable CAGR
    if years_elapsed >= 0.5:  # At least 6 months
        start_value = df_sorted['value'].iloc[0]
        end_value = df_sorted['value'].iloc[-1]
        
        if start_value > 0:
            # Calculate CAGR
            cagr = (end_value / start_value) ** (1 / years_elapsed) - 1
            
            # Cap unrealistic growth rates (both positive and negative)
            cagr = max(-0.5, min(0.5, cagr))  # Cap at ±50% annual
            
            # Convert to monthly rate
            monthly_growth_rate = (1 + cagr) ** (1/12) - 1
        else:
            monthly_growth_rate = sp500_monthly_rate
    else:
        # Fallback to simple average monthly return if insufficient data
        df_sorted['monthly_return'] = df_sorted['value'].pct_change()
        monthly_growth_rate = df_sorted['monthly_return'].mean()
        
        # If still unrealistic, use S&P baseline
        if abs(monthly_growth_rate) > 0.05:  # More than 5% per month is suspect
            monthly_growth_rate = sp500_monthly_rate
    
    current_pace = [current_value * ((1 + monthly_growth_rate) ** (i + 1)) 
                   for i in range(months_to_retirement)]
    
    # 3. OPTIMISTIC: What rate do we need to hit 1.5x target?
    optimistic_target = target_amount * 1.5
    years_to_retirement = months_to_retirement / 12
    
    if years_to_retirement > 0 and current_value > 0:
        required_annual_rate = (optimistic_target / current_value) ** (1 / years_to_retirement) - 1
        required_monthly_rate = (1 + required_annual_rate) ** (1/12) - 1
        optimistic = [current_value * ((1 + required_monthly_rate) ** (i + 1)) 
                     for i in range(months_to_retirement)]
    else:
        optimistic = conservative  # Fallback
    
    return future_dates, conservative, current_pace, optimistic

# ============================================================================
# CONFIDENCE SCORE (FIXED)
# ============================================================================

def calculate_confidence_score(df_net, target_amount, target_year=2042):
    """
    Calculate probability of hitting retirement goal - ENHANCED VERSION
    
    FIXES:
    - Requires minimum 6 months of data
    - Caps CAGR at ±50% annually
    - Adjusts for volatility and time remaining
    - Better error handling
    """
    if df_net.empty or len(df_net) < 2:
        return 50.0, "Insufficient data"
    
    current_value = df_net['value'].iloc[-1]
    current_date = df_net['date'].iloc[-1]
    months_remaining = (target_year - current_date.year) * 12 - current_date.month
    
    if months_remaining <= 0:
        return 100.0 if current_value >= target_amount else 0.0, "At retirement date"
    
    # Calculate historical CAGR with safeguards
    df_sorted = df_net.sort_values('date').copy()
    start_date = df_sorted['date'].iloc[0]
    end_date = df_sorted['date'].iloc[-1]
    years_elapsed = (end_date - start_date).days / 365.25
    
    # Require meaningful time period
    if years_elapsed < 0.5:
        return 50.0, "Need 6+ months of data for reliable projection"
    
    start_value = df_sorted['value'].iloc[0]
    end_value = df_sorted['value'].iloc[-1]
    
    # Check for data quality
    if start_value <= 0:
        return 50.0, "Invalid starting value"
    
    # Calculate CAGR
    cagr = (end_value / start_value) ** (1 / years_elapsed) - 1
    
    # Cap extreme CAGRs
    cagr = max(-0.5, min(0.5, cagr))  # ±50% annual max
    
    # Project to retirement
    monthly_rate = (1 + cagr) ** (1/12) - 1
    projected_value = current_value * ((1 + monthly_rate) ** months_remaining)
    
    # Enhanced confidence calculation
    ratio = projected_value / target_amount
    
    if ratio >= 1.5:
        confidence = 95.0
        method = "Exceeding target significantly"
    elif ratio >= 1.2:
        confidence = 90.0
        method = "Well above target"
    elif ratio >= 1.0:
        confidence = 80.0
        method = "On track to meet target"
    elif ratio >= 0.9:
        confidence = 70.0
        method = "Close - minor gap"
    elif ratio >= 0.75:
        confidence = 55.0
        method = "Below target - adjustment recommended"
    elif ratio >= 0.5:
        confidence = 35.0
        method = "Significant gap - action needed"
    else:
        confidence = 20.0
        method = "Major adjustment required"
    
    # Adjust for volatility
    df_sorted['monthly_return'] = df_sorted['value'].pct_change()
    volatility = df_sorted['monthly_return'].std()
    
    if pd.notna(volatility) and volatility > 0:
        # Higher volatility = lower confidence (uncertainty)
        volatility_penalty = min(20, volatility * 100)
        confidence = max(5, confidence - volatility_penalty)
    
    # Adjust for time remaining
    years_remaining = months_remaining / 12
    if years_remaining < 5:
        # Less time = less room for recovery
        time_penalty = (5 - years_remaining) * 2
        confidence = max(5, confidence - time_penalty)
    
    return round(confidence, 1), method
