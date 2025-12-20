# data/parser.py - FIX for misaligned columns

import pandas as pd
import streamlit as st

def parse_portfolio_csv(file_obj):
    """
    Parse portfolio CSV with proper column detection
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
    
    st.write("### 🔍 Column Detection")
    st.write(f"Columns found: {list(df.columns)}")
    
    # Find the actual Account Name column by checking content
    account_name_col = None
    
    # Check if we have the expected column name
    if 'Account Name' in df.columns:
        # Verify it has actual account names (contains spaces or apostrophes)
        sample_values = df['Account Name'].dropna().head(10).astype(str)
        has_real_names = any("'" in str(val) or " " in str(val) for val in sample_values)
        
        if has_real_names:
            account_name_col = 'Account Name'
            st.write(f"✅ Found Account Name column: '{account_name_col}'")
        else:
            st.write(f"⚠️ 'Account Name' column exists but contains symbols, not names")
            st.write(f"Sample values: {list(sample_values)}")
            
            # Try to find the right column by looking at second column (usually account name in Fidelity CSVs)
            if len(df.columns) > 1:
                second_col = df.columns[1]
                sample_values_2 = df[second_col].dropna().head(10).astype(str)
                st.write(f"Checking second column '{second_col}': {list(sample_values_2)}")
                
                has_real_names_2 = any("'" in str(val) or " " in str(val) for val in sample_values_2)
                if has_real_names_2:
                    account_name_col = second_col
                    st.write(f"✅ Using column '{second_col}' as Account Name")
    
    if not account_name_col:
        st.error("❌ Could not find Account Name column")
        return pd.DataFrame(), {}
    
    # Check for required columns
    required = ['Symbol', 'Current Value']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.write(f"Available: {list(df.columns)}")
        return pd.DataFrame(), {}
    
    # Show sample of what we're using
    st.write(f"\n**Using Account Name from:** '{account_name_col}'")
    st.write("Sample account names:")
    st.write(df[account_name_col].unique()[:10])
    
    # Clean numeric columns
    df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
    df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
    
    # Optional columns
    for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Set up output columns
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
    
    # Use the correct account name column
    df['account_name'] = df[account_name_col]
    
    total_value = df['market_value'].sum()
    if total_value > 0:
        df['allocation'] = df['market_value'] / total_value
    else:
        df['allocation'] = 0.0
    
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
        
        st.write("### 💰 Net Worth Calculation")
        
        # Find the actual Account Name column
        account_name_col = None
        
        if 'Account Name' in raw_df.columns:
            sample_values = raw_df['Account Name'].dropna().head(10).astype(str)
            has_real_names = any("'" in str(val) or " " in str(val) for val in sample_values)
            
            if has_real_names:
                account_name_col = 'Account Name'
            elif len(raw_df.columns) > 1:
                # Try second column
                second_col = raw_df.columns[1]
                sample_values_2 = raw_df[second_col].dropna().head(10).astype(str)
                has_real_names_2 = any("'" in str(val) or " " in str(val) for val in sample_values_2)
                if has_real_names_2:
                    account_name_col = second_col
        
        if not account_name_col:
            st.error("❌ Could not find Account Name column")
            return 0, 0
        
        if 'Current Value' not in raw_df.columns:
            st.error("❌ 'Current Value' column not found")
            return 0, 0
        
        st.write(f"Using account names from column: '{account_name_col}'")
        
        # Clean Current Value
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        sean_kim_total = 0
        taylor_total = 0
        
        # Group by account name
        account_summary = raw_df.groupby(account_name_col)['Current Value'].sum()
        
        st.write("\n**Breakdown by Account:**")
        for account_name, total in account_summary.items():
            name_lower = str(account_name).lower()
            st.write(f"  - {account_name}: ${total:,.2f}")
            
            if 'sean' in name_lower or 'kim' in name_lower:
                sean_kim_total += total
                st.write(f"    ✅ Sean/Kim")
            elif 'taylor' in name_lower:
                taylor_total += total
                st.write(f"    ✅ Taylor")
        
        st.write(f"\n**TOTALS:** Sean+Kim=${sean_kim_total:,.2f} | Taylor=${taylor_total:,.2f}")
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return 0, 0
