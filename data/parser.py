# data/parser.py - CLEAN: No persistent analysis output

import pandas as pd
import streamlit as st
import io
import base64

def parse_portfolio_csv(file_obj, show_analysis=False):
    """
    Parse portfolio CSV with explicit column mapping and automatic Fidelity footer removal
    
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
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(cleaned_content))
        
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
    
    # Find Account Name column (looks for apostrophes like "Sean's" or known names)
    account_name_col = None
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
            st.error("❌ Could not identify Account Name column automatically.")
            st.write("Tip: Look for the column containing values like \"Sean's Personal\", \"Kim's IRA\", etc.")
        return pd.DataFrame(), {}
    
    # Required columns check
    required = ['Symbol', 'Current Value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if show_analysis:
            st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame(), {}

    # Clean numeric columns
    df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
    df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
    
    for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Group by account and aggregate holdings
    grouped = df.groupby(account_name_col).agg({
        'Symbol': 'first',
        'Description': 'first',
        'Quantity': 'sum',
        'Last Price': 'last',
        'Current Value': 'sum',
        'Cost Basis Total': 'sum'
    }).reset_index()
    
    # Calculate gains
    grouped['unrealized_gain'] = grouped['Current Value'] - grouped['Cost Basis Total']
    
    total_value = grouped['Current Value'].sum()
    if total_value > 0:
        grouped['allocation'] = grouped['Current Value'] / total_value
    else:
        grouped['allocation'] = 0

    total_cost = grouped['Cost Basis Total'].sum()
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': grouped['unrealized_gain'].sum(),
        'total_gain_pct': (grouped['unrealized_gain'].sum() / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': grouped.loc[grouped['Current Value'].idxmax(), 'Symbol'] if not grouped.empty else 'CASH',
        'top_allocation': grouped['allocation'].max() * 100 if not grouped.empty else 0
    }

    # Rename for consistency
    merged = grouped.rename(columns={
        account_name_col: 'account',
        'Symbol': 'ticker',
        'Description': 'name',
        'Quantity': 'quantity',
        'Last Price': 'price',
        'Current Value': 'market_value',
        'Cost Basis Total': 'cost_basis',
        'unrealized_gain': 'unrealized_gain',
        'allocation': 'allocation'
    })
    
    # Only show analysis if explicitly requested
    if show_analysis:
        unique_accounts = merged['account'].unique()
        st.success(f"✅ Parsed successfully! Found accounts: {', '.join(unique_accounts)}")
        st.info(f"💰 Total Value: ${total_value:,.2f}")

    return merged, summary

def merge_portfolios(portfolio_dfs):
    """Merge multiple parsed portfolio DataFrames (for multi-upload support)"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}
    
    combined = pd.concat(portfolio_dfs, ignore_index=True)
    
    # Re-group after merge to consolidate same tickers across accounts
    grouped = combined.groupby('ticker').agg({
        'name': 'first',
        'quantity': 'sum',
        'price': 'last',
        'market_value': 'sum',
        'cost_basis': 'sum'
    }).reset_index()
    
    grouped['unrealized_gain'] = grouped['market_value'] - grouped['cost_basis']
    
    total_value = grouped['market_value'].sum()
    if total_value > 0:
        grouped['allocation'] = grouped['market_value'] / total_value
    else:
        grouped['allocation'] = 0

    total_cost = grouped['cost_basis'].sum()
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': grouped['unrealized_gain'].sum(),
        'total_gain_pct': (grouped['unrealized_gain'].sum() / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': grouped.loc[grouped['market_value'].idxmax(), 'ticker'] if not grouped.empty else 'CASH',
        'top_allocation': grouped['allocation'].max() * 100 if not grouped.empty else 0
    }

    return grouped, summary

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
        
        raw_df = pd.read_csv(io.StringIO(cleaned_content))
        raw_df.columns = raw_df.columns.str.strip()
        
        # Find Account Name column (same logic) - NO UI OUTPUT
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
                break
        
        if not account_name_col or 'Current Value' not in raw_df.columns:
            return 0, 0
        
        # Clean Current Value
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        # Group and sum
        account_summary = raw_df.groupby(account_name_col)['Current Value'].sum()
        
        sean_kim_total = 0
        taylor_total = 0
        
        for account_name, total in account_summary.items():
            name_lower = str(account_name).lower()
            if 'sean' in name_lower or 'kim' in name_lower:
                sean_kim_total += total
            elif 'taylor' in name_lower:
                taylor_total += total
        
        return sean_kim_total, taylor_total
        
    except Exception as e:
        # Silent failure for background calculations
        return 0, 0
