# data/parser.py - CSV INSPECTOR - Shows exact structure

import pandas as pd
import streamlit as st

def parse_portfolio_csv(file_obj):
    """
    Parse portfolio CSV - INSPECTOR VERSION
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
    
    # SHOW FULL DEBUG INFO
    st.write("### 🔍 CSV INSPECTOR")
    st.write(f"**Total rows:** {len(df)}")
    st.write(f"**Columns:** {list(df.columns)}")
    
    st.write("\n**First 10 rows of raw data:**")
    st.dataframe(df.head(10))
    
    if 'Account Name' in df.columns:
        st.write("\n**All unique Account Name values:**")
        st.write(df['Account Name'].unique())
        
        st.write("\n**Account Name column (first 20):**")
        for idx, val in df['Account Name'].head(20).items():
            st.write(f"Row {idx}: '{val}' (type: {type(val).__name__})")
    
    # Try basic parsing
    required = ['Symbol', 'Current Value']
    if all(c in df.columns for c in required):
        df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
        
        df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
        df['market_value'] = df['Current Value']
        df['shares'] = df.get('Quantity', 0)
        df['price'] = df.get('Last Price', 0)
        df['cost_basis'] = df.get('Cost Basis Total', df['Current Value'])
        df['unrealized_gain'] = 0
        df['pct_gain'] = 0
        
        total_value = df['market_value'].sum()
        if total_value > 0:
            df['allocation'] = df['market_value'] / total_value
        else:
            df['allocation'] = 0.0
        
        df['account_name'] = df.get('Account Name', 'Unknown')
        
        summary = {
            'total_value': total_value,
            'total_cost': df['cost_basis'].sum(),
            'total_gain': 0,
            'total_gain_pct': 0,
            'top_holding': df.loc[df['market_value'].idxmax(), 'ticker'] if not df.empty else 'CASH',
            'top_allocation': df['allocation'].max() * 100 if not df.empty else 0
        }
        
        clean_df = df[['ticker', 'shares', 'price', 'market_value', 'cost_basis',
                       'unrealized_gain', 'pct_gain', 'allocation', 'account_name']].copy()
        
        return clean_df, summary
    
    return pd.DataFrame(), {}

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
        
        st.write("### 💰 CALCULATE NET WORTH DEBUG")
        st.write(f"Total rows: {len(raw_df)}")
        st.write(f"Columns: {list(raw_df.columns)}")
        
        if 'Account Name' not in raw_df.columns:
            st.error("❌ 'Account Name' column not found")
            return 0, 0
            
        if 'Current Value' not in raw_df.columns:
            st.error("❌ 'Current Value' column not found")
            return 0, 0
        
        st.write("\n**Unique Account Names:**")
        st.write(raw_df['Account Name'].unique())
        
        # Clean Current Value column
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        st.write(f"\n**Total value in CSV:** ${raw_df['Current Value'].sum():,.2f}")
        
        sean_kim_total = 0
        taylor_total = 0
        
        # Show breakdown by account name
        st.write("\n**Breakdown by Account Name:**")
        for account_name in raw_df['Account Name'].unique():
            account_df = raw_df[raw_df['Account Name'] == account_name]
            account_total = account_df['Current Value'].sum()
            st.write(f"  - '{account_name}': ${account_total:,.2f}")
            
            # Check for matching names
            name_lower = str(account_name).lower()
            if 'sean' in name_lower or 'kim' in name_lower:
                sean_kim_total += account_total
                st.write(f"    ✅ Added to Sean+Kim")
            elif 'taylor' in name_lower:
                taylor_total += account_total
                st.write(f"    ✅ Added to Taylor")
            else:
                st.write(f"    ⚠️ No match")
        
        st.write(f"\n**FINAL TOTALS:**")
        st.write(f"  - Sean + Kim: ${sean_kim_total:,.2f}")
        st.write(f"  - Taylor: ${taylor_total:,.2f}")
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return 0, 0
