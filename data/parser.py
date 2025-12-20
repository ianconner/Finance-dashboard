# data/parser.py - ROBUST VERSION with better CSV handling

import pandas as pd
import streamlit as st

def parse_portfolio_csv(file_obj):
    """
    Parse portfolio CSV with Account Name column support
    Handles both Fidelity format and custom format with Account Name
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

    # Check if this is the custom format with Account Name
    has_account_name = 'Account Name' in df.columns
    
    if has_account_name:
        # Custom format parsing
        required = ['Account Name', 'Symbol', 'Current Value']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return pd.DataFrame(), {}
        
        # CRITICAL: Filter out rows where Account Name is actually a symbol
        # Real account names contain spaces or apostrophes, symbols don't
        df = df[df['Account Name'].astype(str).str.contains(r"[\s']", regex=True, na=False)]
        
        if df.empty:
            st.error("No valid account names found in CSV")
            return pd.DataFrame(), {}
        
        # Clean numeric columns
        df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
        
        # Optional columns
        for col in ['Quantity', 'Last Price', 'Last Price Change', 
                    "Today's Gain/Loss Dollar", "Today's Gain/Loss Percent",
                    "Total Gain/Loss Dollar", "Total Gain/Loss Percent"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[\$,%]', '', regex=True).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Extract ticker
        df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
        df['ticker'] = df['ticker'].replace({'NAN': 'CASH', '<NA>': 'CASH', '': 'CASH'})
        
        # Identify cash rows
        df['is_cash'] = df['ticker'].str.contains(r'\*\*$', regex=True, na=False) | df['ticker'].isin(['FCASH', 'SPAXX', 'CASH'])
        
        # Set values
        df['market_value'] = df['Current Value']
        df['shares'] = df.get('Quantity', 0)
        df['price'] = df.get('Last Price', 0)
        
        # Calculate cost basis from total gain if available
        if 'Total Gain/Loss Dollar' in df.columns:
            df['cost_basis'] = df['market_value'] - df['Total Gain/Loss Dollar']
            df['unrealized_gain'] = df['Total Gain/Loss Dollar']
        else:
            df['cost_basis'] = df['market_value']
            df['unrealized_gain'] = 0
        
        # Ensure cost basis is never negative
        df['cost_basis'] = df['cost_basis'].clip(lower=0)
        
        # Calculate percentage gain
        df['pct_gain'] = df.apply(
            lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
            axis=1
        )
        
        # Store account name
        df['account_name'] = df['Account Name']
        
    else:
        # Standard Fidelity format
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
        df['ticker'] = df['ticker'].replace({'NAN': 'CASH', '<NA>': 'CASH', '': 'CASH'})

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
        
        df['account_name'] = 'Unknown'

    # Calculate allocation
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
                   'unrealized_gain', 'pct_gain', 'allocation', 'account_name']].copy()

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
        'unrealized_gain': 'sum',
        'account_name': 'first'
    })

    merged['pct_gain'] = merged.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
        axis=1
    )

    total_value = merged['market_value'].sum()
    if total_value > 0:
        merged['allocation'] = merged['market_value'] / total_value
    else:
        merged['allocation'] = 0

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

def calculate_net_worth_from_csv(csv_data_b64):
    """
    Calculate net worth from CSV data by Account Name
    Returns: (sean_kim_total, taylor_total)
    """
    import base64
    import io
    
    try:
        decoded = base64.b64decode(csv_data_b64).decode('utf-8')
        raw_df = pd.read_csv(io.StringIO(decoded))
        raw_df.columns = raw_df.columns.str.strip()
        
        if 'Account Name' not in raw_df.columns or 'Current Value' not in raw_df.columns:
            return 0, 0
        
        # CRITICAL FIX: Filter to only rows with real account names (contain space or apostrophe)
        raw_df = raw_df[raw_df['Account Name'].astype(str).str.contains(r"[\s']", regex=True, na=False)]
        
        if raw_df.empty:
            return 0, 0
        
        # Clean Current Value column
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        sean_kim_total = 0
        taylor_total = 0
        
        for _, row in raw_df.iterrows():
            account_name = str(row.get('Account Name', '')).lower()
            value = row.get('Current Value', 0)
            
            if 'sean' in account_name or 'kim' in account_name:
                sean_kim_total += value
            elif 'taylor' in account_name:
                taylor_total += value
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        print(f"Error calculating net worth from CSV: {e}")
        return 0, 0
