# data/parser.py - CLEAN: No persistent analysis output

import pandas as pd
import streamlit as st
import io
import base64

def parse_portfolio_csv(file_obj, show_analysis=False):
    """
    Parse portfolio CSV with explicit column mapping and automatic Fidelity footer removal
    Handles both tab-delimited and comma-separated formats
    
    Args:
        file_obj: File object or string content
        show_analysis: Whether to show the CSV structure analysis (default False)
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
        
        # --- Aggressive Fidelity footer removal ---
        lines = content.splitlines()
        cleaned_lines = []
        data_section = True
        for line in lines:
            stripped = line.strip()
            # Stop at any known Fidelity footer indicators
            if (stripped.startswith('"The data and information') or 
                'Date downloaded' in stripped or 
                'Brokerage services' in stripped or 
                'Fidelity.com' in stripped or 
                stripped.startswith('"Brokerage services') or
                stripped.startswith('"Brokerage services are provided')):
                data_section = False
                continue
            if stripped == '':  # Skip blank lines after footer starts
                if not data_section:
                    continue
                cleaned_lines.append(line)  # Keep blanks during data section
                continue
            
            if data_section:
                cleaned_lines.append(line.rstrip(', '))  # Clean trailing commas/spaces
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Auto-detect delimiter: tab or comma
        first_line = cleaned_lines[0] if cleaned_lines else ""
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
        
        # Parse CSV with detected delimiter
        df = pd.read_csv(io.StringIO(cleaned_content), sep=delimiter)
        
    except Exception as e:
        if show_analysis:
            st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}

    if df.empty:
        if show_analysis:
            st.error("CSV is empty after cleaning")
        return pd.DataFrame(), {}

    # Clean column names
    df.columns = df.columns.str.strip()
    
    if show_analysis:
        st.write(f"**Detected delimiter:** {'Tab' if delimiter == '\t' else 'Comma'}")
        st.write(f"**Columns found ({len(df.columns)}):** {list(df.columns)}")
        st.write(f"**Rows:** {len(df)}")
    
    # Try to find Account Name column - try exact match first, then fuzzy
    account_name_col = None
    
    # Method 1: Exact match
    if 'Account Name' in df.columns:
        account_name_col = 'Account Name'
    # Method 2: Case-insensitive search
    else:
        for col in df.columns:
            if 'account' in col.lower() and 'name' in col.lower():
                account_name_col = col
                break
    
    # Method 3: Look for column with apostrophes (Sean's, Kim's, Taylor's)
    if not account_name_col:
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(20)
            has_apostrophes = any("'" in str(val) for val in sample_values)
            has_account_pattern = any(
                any(name in str(val).lower() for name in ['sean', 'kim', 'taylor', 'roth', 'ira', 'personal'])
                for val in sample_values
            )
            has_spaces = any(" " in str(val) for val in sample_values)
            
            if (has_apostrophes or has_account_pattern) and has_spaces:
                account_name_col = col
                break
    
    if not account_name_col:
        if show_analysis:
            st.error("❌ Could not find Account Name column.")
            st.write(f"**Searched for:** 'Account Name' (exact), columns containing 'account' and 'name', or columns with Sean/Kim/Taylor patterns")
            st.write(f"**Available columns:** {list(df.columns)}")
            st.write("**First few rows:**")
            st.dataframe(df.head(3))
        return pd.DataFrame(), {}
    
    # Required columns check
    required = ['Symbol', 'Current Value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if show_analysis:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.write(f"Available columns: {list(df.columns)}")
        return pd.DataFrame(), {}

    # Clean Current Value - handle values in quotes with commas like "$1,120.95 "
    df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
    df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
    
    # Clean other numeric columns
    for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Sum by account (don't group by Symbol since we want total per account)
    account_summary = df.groupby(account_name_col).agg({
        'Current Value': 'sum',
        'Cost Basis Total': 'sum' if 'Cost Basis Total' in df.columns else 'first'
    }).reset_index()
    
    total_value = account_summary['Current Value'].sum()
    total_cost = account_summary['Cost Basis Total'].sum() if 'Cost Basis Total' in df.columns else 0
    
    # Create summary
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': total_value - total_cost,
        'total_gain_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': df.loc[df['Current Value'].idxmax(), 'Symbol'] if not df.empty and df['Current Value'].max() > 0 else 'CASH',
        'top_allocation': (df['Current Value'].max() / total_value * 100) if total_value > 0 else 0
    }
    
    # For detailed holdings, keep individual rows
    df['account_name'] = df[account_name_col]
    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['name'] = df.get('Description', df['Symbol'])  # Use Description if available
    df['market_value'] = df['Current Value']
    df['shares'] = df.get('Quantity', 0)
    df['price'] = df.get('Last Price', 0)
    df['cost_basis'] = df.get('Cost Basis Total', 0)
    df['unrealized_gain'] = df['market_value'] - df['cost_basis']
    df['pct_gain'] = df.apply(
        lambda row: (row['unrealized_gain'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0,
        axis=1
    )
    
    if total_value > 0:
        df['allocation'] = df['market_value'] / total_value
    else:
        df['allocation'] = 0.0
    
    # Only show analysis if explicitly requested
    if show_analysis:
        unique_accounts = df['account_name'].unique()
        st.success(f"✅ Parsed successfully! Found accounts: {', '.join(unique_accounts)}")
        st.info(f"💰 Total Value: ${total_value:,.2f}")
        
        # Show breakdown by account
        st.markdown("**Account Breakdown:**")
        for account in unique_accounts:
            account_total = df[df['account_name'] == account]['market_value'].sum()
            st.write(f"  - {account}: ${account_total:,.2f}")
    
    clean_df = df[['ticker', 'name', 'shares', 'price', 'market_value', 'cost_basis',
                   'unrealized_gain', 'pct_gain', 'allocation', 'account_name']].copy()
    
    return clean_df, summary

def merge_portfolios(portfolio_dfs):
    """Merge multiple parsed portfolio DataFrames (for multi-upload support)"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}
    
    combined = pd.concat(portfolio_dfs, ignore_index=True)
    
    # Portfolio data already has the columns we need from parse_portfolio_csv
    # Just sum up totals
    total_value = combined['market_value'].sum()
    total_cost = combined['cost_basis'].sum()
    
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': total_value - total_cost,
        'total_gain_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': combined.loc[combined['market_value'].idxmax(), 'ticker'] if not combined.empty else 'CASH',
        'top_allocation': (combined['market_value'].max() / total_value * 100) if total_value > 0 else 0
    }

    return combined, summary

def calculate_net_worth_from_csv(csv_data_b64):
    """
    Calculate net worth from saved base64 CSV using the same robust cleaning
    NO UI OUTPUT - silent calculation only
    """
    import base64
    
    try:
        decoded = base64.b64decode(csv_data_b64).decode('utf-8')
        
        # Same aggressive cleaning as above
        lines = decoded.splitlines()
        cleaned_lines = []
        data_section = True
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('"The data and information') or 
                'Date downloaded' in stripped or 
                'Brokerage services' in stripped or 
                'Fidelity.com' in stripped or 
                stripped.startswith('"Brokerage services')):
                data_section = False
                continue
            if stripped == '':
                if not data_section:
                    continue
                cleaned_lines.append(line)
                continue
            if data_section:
                cleaned_lines.append(line.rstrip(', '))
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Auto-detect delimiter
        first_line = cleaned_lines[0] if cleaned_lines else ""
        delimiter = '\t' if '\t' in first_line else ','
        
        # Parse CSV with detected delimiter
        raw_df = pd.read_csv(io.StringIO(cleaned_content), sep=delimiter)
        raw_df.columns = raw_df.columns.str.strip()
        
        # Find Account Name column - try multiple methods
        account_name_col = None
        if 'Account Name' in raw_df.columns:
            account_name_col = 'Account Name'
        else:
            for col in raw_df.columns:
                if 'account' in col.lower() and 'name' in col.lower():
                    account_name_col = col
                    break
        
        # Fallback: look for apostrophes
        if not account_name_col:
            for col in raw_df.columns:
                sample_values = raw_df[col].dropna().astype(str).head(20)
                has_apostrophes = any("'" in str(val) for val in sample_values)
                has_account_pattern = any(
                    any(name in str(val).lower() for name in ['sean', 'kim', 'taylor', 'roth', 'ira'])
                    for val in sample_values
                )
                if has_apostrophes or has_account_pattern:
                    account_name_col = col
                    break
        
        if not account_name_col or 'Current Value' not in raw_df.columns:
            return 0, 0
        
        # Clean Current Value - handle quotes and commas
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        # Group and sum by account
        account_summary = raw_df.groupby(account_name_col)['Current Value'].sum()
        
        sean_kim_total = 0
        taylor_total = 0
        
        for account_name, total in account_summary.items():
            name_lower = str(account_name).lower()
            # Sean's accounts include: Personal, ROTH IRA, IRA
            if 'sean' in name_lower:
                sean_kim_total += total
            elif 'kim' in name_lower:
                sean_kim_total += total
            elif 'taylor' in name_lower:
                taylor_total += total
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        # Silent failure for background calculations
        return 0, 0
