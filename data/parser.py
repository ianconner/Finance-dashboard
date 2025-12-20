# data/parser.py

import pandas as pd
import streamlit as st

def parse_portfolio_csv(file_obj):
    """
    Parse Fidelity portfolio CSV and return cleaned DataFrame + summary dict
    """
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']
    
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV missing required columns: {', '.join(missing)}")
        return pd.DataFrame(), {}

    # Keep only needed columns
    cols_to_keep = required + (['Account Name'] if 'Account Name' in df.columns else [])
    df = df[cols_to_keep].copy()

    # Clean data
    df = df.dropna(subset=required, how='any')
    df = df[df['Symbol'].astype(str).str.strip() != '']
    df = df[~df['Symbol'].astype(str).str.strip().str.lower().isin(
        ['symbol', 'account number', 'nan', 'account name', '']
    )]

    if df.empty:
        st.error("No valid holdings found in CSV.")
        return pd.DataFrame(), {}

    # Clean numeric columns
    for col in ['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required)
    if df.empty:
        return pd.DataFrame(), {}

    # Add calculated columns
    df['ticker'] = df['Symbol'].str.upper().str.strip()
    df['market_value'] = df['Current Value']
    df['cost_basis'] = df['Cost Basis Total']
    df['shares'] = df['Quantity']
    df['price'] = df['Last Price']
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = (df['unrealized_gain'] / df['cost_basis']) * 100

    total_value = df['market_value'].sum()
    if total_value > 0:
        df['allocation'] = df['market_value'] / total_value
    else:
        df['allocation'] = 0.0

    # Summary stats
    summary = {
        'total_value': total_value,
        'total_cost': df['cost_basis'].sum(),
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / df['cost_basis'].sum()) * 100 
                          if df['cost_basis'].sum() > 0 else 0,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else 'N/A',
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }

    # Return only the clean columns we need
    clean_df = df[['ticker', 'shares', 'price', 'market_value', 'cost_basis',
                   'unrealized_gain', 'pct_gain', 'allocation']]

    return clean_df, summary

# Add this function at the end of data/parser.py

def merge_portfolios(portfolio_dfs):
    """Combine multiple portfolio DataFrames into one total view"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}

    # Concatenate all
    merged = pd.concat(portfolio_dfs, ignore_index=True)

    # Group by ticker to combine same holdings across accounts
    merged = merged.groupby('ticker', as_index=False).agg({
        'shares': 'sum',
        'price': 'last',  # use most recent price
        'market_value': 'sum',
        'cost_basis': 'sum',
        'unrealized_gain': 'sum'
    })

    # Recalculate derived fields
    merged['pct_gain'] = merged.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) 
        if row['cost_basis'] > 0 else 0, axis=1
    )

    total_value = merged['market_value'].sum()
    if total_value > 0:
        merged['allocation'] = merged['market_value'] / total_value
    else:
        merged['allocation'] = 0.0

    # Summary
    total_cost = merged['cost_basis'].sum()
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': merged['unrealized_gain'].sum(),
        'total_gain_pct': (merged['unrealized_gain'].sum() / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': merged.loc[merged['market_value'].idxmax(), 'ticker'] if not merged.empty else 'N/A',
        'top_allocation': merged['allocation'].max() * 100 if not merged.empty else 0
    }

    return merged, summary
