# data/parser.py - ACCOUNT NUMBER BASED PARSER

import pandas as pd
import streamlit as st
import io
import base64

# CONFIGURE YOUR ACCOUNT NUMBERS HERE
ACCOUNT_MAPPING = {
    'sean': [
        'X64950612',  # Sean's Personal
        '261525666',  # Sean's ROTH
        '263338794',  # Sean's IRA
    ],
    'kim': [
        '235293295',  # Kim's IRA
        '238106811',  # Kim's ROTH
    ],
    'taylor': [
        'Z21361724',  # Taylor's
    ]
}

def identify_person_by_account(account_number):
    """Identify person by account number"""
    account_str = str(account_number).strip()
    
    # Check full match or last 4 digits
    for person, account_list in ACCOUNT_MAPPING.items():
        for acc in account_list:
            acc_str = str(acc).strip()
            # Match full number or last 4 digits
            if account_str == acc_str or account_str.endswith(acc_str[-4:]) or acc_str.endswith(account_str[-4:]):
                return person
    
    return 'unknown'

def parse_portfolio_csv(file_obj, show_analysis=False):
    """
    Parse portfolio CSV using Account Number for identification
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
        
        # Aggressive Fidelity footer removal
        lines = content.splitlines()
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
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
        
        # Parse CSV
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
        st.write(f"**Columns found:** {list(df.columns)}")
        st.write(f"**Rows:** {len(df)}")
    
    # Find Account Number column
    account_number_col = None
    for col in df.columns:
        if 'account' in col.lower() and 'number' in col.lower():
            account_number_col = col
            break
    
    if not account_number_col:
        if show_analysis:
            st.error("❌ Could not find Account Number column")
            st.write("Available columns:", list(df.columns))
        return pd.DataFrame(), {}
    
    # Find Account Name column (for display purposes)
    account_name_col = None
    for col in df.columns:
        if 'account' in col.lower() and 'name' in col.lower():
            account_name_col = col
            break
    
    # Required columns check
    required = ['Symbol', 'Current Value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if show_analysis:
            st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame(), {}

    # Clean Current Value
    df['Current Value'] = df['Current Value'].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
    df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce').fillna(0)
    
    # Clean other numeric columns
    for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Identify person for each row based on account number
    df['person'] = df[account_number_col].apply(identify_person_by_account)
    
    # Calculate totals
    total_value = df['Current Value'].sum()
    total_cost = df['Cost Basis Total'].sum() if 'Cost Basis Total' in df.columns else 0
    
    # Create summary
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': total_value - total_cost,
        'total_gain_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': df.loc[df['Current Value'].idxmax(), 'Symbol'] if not df.empty and df['Current Value'].max() > 0 else 'CASH',
        'top_allocation': (df['Current Value'].max() / total_value * 100) if total_value > 0 else 0
    }
    
    # Build output dataframe
    df['account_number'] = df[account_number_col]
    df['account_name'] = df[account_name_col] if account_name_col else df[account_number_col]
    df['ticker'] = df['Symbol'].astype(str).str.upper().str.strip()
    df['name'] = df.get('Description', df['Symbol'])
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
    
    if show_analysis:
        # Show what we found
        st.markdown("**🔍 Account Detection Results:**")
        account_totals = df.groupby(['account_number', 'account_name', 'person'])['market_value'].sum().reset_index()
        for _, row in account_totals.iterrows():
            st.write(f"  - Account {row['account_number']} ({row['account_name']}): ${row['market_value']:,.2f} → **{row['person'].upper()}**")
        
        # Summary by person
        person_totals = df.groupby('person')['market_value'].sum()
        st.markdown("**💰 Totals by Person:**")
        for person, total in person_totals.items():
            st.write(f"  - {person.upper()}: ${total:,.2f}")
    
    clean_df = df[['ticker', 'name', 'shares', 'price', 'market_value', 'cost_basis',
                   'unrealized_gain', 'pct_gain', 'allocation', 'account_name', 'account_number', 'person']].copy()
    
    return clean_df, summary

def merge_portfolios(portfolio_dfs):
    """Merge multiple portfolios"""
    if not portfolio_dfs:
        return pd.DataFrame(), {}
    
    combined = pd.concat(portfolio_dfs, ignore_index=True)
    
    total_value = combined['market_value'].sum()
    total_cost = combined['cost_basis'].sum()
    
    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain': total_value - total_cost,
        'total_gain_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        'top_holding': combined.loc[combined['market_value'].idxmax(), 'ticker'] if not combined.empty and combined['market_value'].max() > 0 else 'CASH',
        'top_allocation': (combined['market_value'].max() / total_value * 100) if total_value > 0 else 0
    }

    return combined, summary

def calculate_net_worth_from_csv(csv_data_b64):
    """
    Calculate net worth from saved base64 CSV using account numbers
    Returns (sean_kim_total, taylor_total)
    """
    import base64
    
    try:
        decoded = base64.b64decode(csv_data_b64).decode('utf-8')
        
        # Same aggressive cleaning
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
        
        # Parse CSV
        raw_df = pd.read_csv(io.StringIO(cleaned_content), sep=delimiter)
        raw_df.columns = raw_df.columns.str.strip()
        
        # Find Account Number column
        account_number_col = None
        for col in raw_df.columns:
            if 'account' in col.lower() and 'number' in col.lower():
                account_number_col = col
                break
        
        if not account_number_col or 'Current Value' not in raw_df.columns:
            return 0, 0
        
        # Clean Current Value
        raw_df['Current Value'] = raw_df['Current Value'].astype(str).str.replace(r'[\$,"\s]', '', regex=True).str.strip()
        raw_df['Current Value'] = pd.to_numeric(raw_df['Current Value'], errors='coerce').fillna(0)
        
        # Identify person for each row
        raw_df['person'] = raw_df[account_number_col].apply(identify_person_by_account)
        
        # Sum by person
        person_totals = raw_df.groupby('person')['Current Value'].sum()
        
        sean_total = person_totals.get('sean', 0)
        kim_total = person_totals.get('kim', 0)
        taylor_total = person_totals.get('taylor', 0)
        
        # Return Sean+Kim combined, then Taylor
        return sean_total + kim_total, taylor_total
        
    except Exception as e:
        return 0, 0
