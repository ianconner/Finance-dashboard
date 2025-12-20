# data/parser.py - Full version with cash handling and multi-portfolio merge

import pandas as pd
import streamlit as st

def parse_portfolio_csv(file_obj):
    """
    Parse Fidelity portfolio CSV, including cash/money market rows (Symbol ends with **)
    """
    try:
        if isinstance(file_obj, str):
            from io import StringIO
            df = pd.read_csv(StringIO(file_obj))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}

    if df.empty:
        st.error("CSV is empty")
        return pd.DataFrame(), {}

    # Clean column names
    df.columns = df.columns.str.strip()

    # Required columns
    required = ['Symbol', 'Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return pd.DataFrame(), {}

    # Clean numeric columns
    for col in ['Quantity', 'Last Price', 'Current Value', 'Cost Basis Total']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN with 0
    df[required] = df[required].fillna(0)

    # Ticker column
    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['ticker'] = df['ticker'].replace({'NAN': 'CASH', '<NA>': 'CASH'})

    # Identify cash rows
    df['is_cash'] = df['ticker'].str.endswith('**') | df['ticker'].isin(['FCASH**', 'SPAXX**'])

    # Market value is always Current Value
    df['market_value'] = df['Current Value']

    # Cost basis: for cash, same as market value (no gain/loss)
    df['cost_basis'] = df.apply(
        lambda row: row['Current Value'] if row['is_cash'] else row['Cost Basis Total'],
        axis=1
    )

    # Standard calculations
    df['shares'] = df['Quantity']
    df['price'] = df['Last Price']
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = df.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
        axis=1
    )

    total_value = df['market_value'].sum()
    if total_value > 0:
        df['allocation'] = df['market_value'] / total_value
    else:
        df['allocation'] = 0.0

    # Summary
    total_cost = df['cost_basis'].sum()
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else 'CASH',
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }

    # Clean output
    clean_df = df[['ticker', 'shares', 'price', 'market_value', 'cost_basis',
                   'unrealized_gain', 'pct_gain', 'allocation']]

    return clean_df, summary

def merge_portfolios(portfolio_dfs):
    """Merge multiple parsed portfolios into one combined view"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}

    merged = pd.concat(portfolio_dfs, ignore_index=True)

    # Group by ticker
    merged = merged.groupby('ticker', as_index=False).agg({
        'shares': 'sum',
        'price': 'last',
        'market_value': 'sum',
        'cost_basis': 'sum',
        'unrealized_gain': 'sum'
    })

    merged['pct_gain'] = merged.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
        axis=1
    )

    total_value = merged['market_value'].sum()
    if total_value > 0:
        merged['allocation'] = merged['market_value'] / total_value

    total_cost = merged['cost_basis'].sum()
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': merged['unrealized_gain'].sum(),
        'total_gain_pct': (merged['unrealized_gain'].sum() / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': merged.loc[merged['market_value'].idxmax(), 'ticker'] if not merged.empty else 'CASH',
        'top_allocation': merged['allocation'].max() * 100 if not merged.empty else 0
    }

    return merged, summary
