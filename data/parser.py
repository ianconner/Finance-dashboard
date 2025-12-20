# data/parser.py - EXPLICIT column detection

import pandas as pd
import streamlit as st
import io

def parse_portfolio_csv(file_obj):
    """
    Parse portfolio CSV with explicit column mapping
    """
    try:
        # Read the raw content
        if isinstance(file_obj, str):
            content = file_obj
        else:
            file_obj.seek(0)
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        
        # Clean trailing commas
        lines = content.split('\n')
        cleaned_lines = [line.rstrip(',') for line in lines]
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(cleaned_content))
        
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}

    if df.empty:
        st.error("CSV is empty")
        return pd.DataFrame(), {}

    # Clean column names
    original_columns = list(df.columns)
    df.columns = df.columns.str.strip()
    
    st.write("### 🔍 CSV Structure Analysis")
    st.write(f"**Columns ({len(df.columns)}):** {list(df.columns)}")
    
    # Show first few rows of each column to identify the real Account Name column
    st.write("\n**Column Preview (first 5 values):**")
    for i, col in enumerate(df.columns[:8]):  # First 8 columns
        sample = df[col].head(5).tolist()
        st.write(f"  [{i}] **{col}**: {sample}")
    
    # Find Account Name column by checking content (look for apostrophes or specific patterns)
    account_name_col = None
    
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(20)
        # Check if this column contains account names (has apostrophes like "Sean's" or "Taylor's")
        has_apostrophes = any("'" in str(val) for val in sample_values)
        has_spaces = any(" " in str(val) for val in sample_values)
        
        # Also check if it contains common account name patterns
        has_account_pattern = any(
            any(name in str(val).lower() for name in ['sean', 'kim', 'taylor', 'roth', 'ira'])
            for val in sample_values
        )
        
        if (has_apostrophes or has_account_pattern) and has_spaces:
            account_name_col = col
            st.write(f"\n✅ **Found Account Name column:** '{col}'")
            st.write(f"   Sample: {list(sample_values[:5])}")
            break
    
    if not account_name_col:
        st.error("❌ Could not identify Account Name column")
        st.write("Please check that your CSV contains account names like 'Sean's Personal', 'Kim's IRA', etc.")
        return pd.DataFrame(), {}
    
    # Check for required columns
    required = ['Symbol', 'Current Value']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return pd.DataFrame(), {}
    
    # Clean numeric columns
    df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
    df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
    
    for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Build output dataframe
    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['market_value'] = df['Current Value']
    df['shares'] = df.get('Quantity', 0)
    df['price'] = df.get('Last Price', 0)
    df['cost_basis'] = df.get('Cost Basis Total', df['Current Value'])
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = df.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
        axis=1
    )
    df['account_name'] = df[account_name_col]  # Use the detected column
    
    total_value = df['market_value'].sum()
    if total_value > 0:
        df['allocation'] = df['market_value'] / total_value
    else:
        df['allocation'] = 0.0
    
    # Show what accounts we found
    unique_accounts = df['account_name'].unique()
    st.write(f"\n✅ **Accounts in portfolio:** {list(unique_accounts)}")
    st.write(f"✅ **Total Value:** ${total_value:,.2f}")
    
    summary = {
        'total_value': total_value,
        'total_cost': df['cost_basis'].sum(),
        'total_gain': df['unrealized_gain'].sum(),
        'total_gain_pct': (df['unrealized_gain'].sum() / df['cost_basis'].sum() * 100) if df['cost_basis'].sum() > 0 else 0,
        'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else 'CASH',
        'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
    }
    
    clean_df = df[['ticker', 'shares', 'price', 'market_value', 'cost_basis',
                   'unrealized_gain', 'pct_gain', 'allocation', 'account_name']].copy()
    
    return clean_df, summary

def merge_portfolios(portfolio_dfs):
    """Merge multiple parsed portfolios"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}

    merged = pd.concat(portfolio_dfs, ignore_index=True)

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
    Calculate net worth from CSV by finding and using the real Account Name column
    """
    import base64
    
    try:
        decoded = base64.b64decode(csv_data_b64).decode('utf-8')
        
        # Clean trailing commas
        lines = decoded.split('\n')
        cleaned_lines = [line.rstrip(',') for line in lines]
        cleaned_content = '\n'.join(cleaned_lines)
        
        raw_df = pd.read_csv(io.StringIO(cleaned_content))
        raw_df.columns = raw_df.columns.str.strip()
        
        st.write("### 💰 Net Worth Calculation")
        
        # Find the real Account Name column
        account_name_col = None
        
        for col in raw_df.columns:
            sample_values = raw_df[col].dropna().astype(str).head(20)
            has_apostrophes = any("'" in str(val) for val in sample_values)
            has_account_pattern = any(
                any(name in str(val).lower() for name in ['sean', 'kim', 'taylor', 'roth', 'ira'])
                for val in sample_values
            )
            has_spaces = any(" " in str(val) for val in sample_values)
            
            if (has_apostrophes or has_account_pattern) and has_spaces:
                account_name_col = col
                st.write(f"✅ Using column: '{col}'")
                break
        
        if not account_name_col or 'Current Value' not in raw_df.columns:
            st.error("Missing required columns")
            return 0, 0
        
        # Clean Current Value
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        # Group by the real account name column
        account_summary = raw_df.groupby(account_name_col)['Current Value'].sum()
        
        sean_kim_total = 0
        taylor_total = 0
        
        st.write("\n**Breakdown:**")
        for account_name, total in account_summary.items():
            name_lower = str(account_name).lower()
            st.write(f"  - {account_name}: ${total:,.2f}")
            
            if 'sean' in name_lower or 'kim' in name_lower:
                sean_kim_total += total
                st.write(f"    → Sean/Kim")
            elif 'taylor' in name_lower:
                taylor_total += total
                st.write(f"    → Taylor")
        
        st.write(f"\n✅ **Sean+Kim: ${sean_kim_total:,.2f} | Taylor: ${taylor_total:,.2f}**")
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return 0, 0
