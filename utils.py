# ============================================================================
# UTILS.PY - Import/Export, Validation, Backup/Restore
# ============================================================================

import pandas as pd
import streamlit as st
from database import add_monthly_update, get_session, PortfolioCSV, engine, Base
import os
import subprocess
from tempfile import NamedTemporaryFile
from datetime import datetime

# ============================================================================
# DATA IMPORT
# ============================================================================

def import_excel_format(df_excel):
    """
    Import data from Excel format:
    Columns: Date, Sean, Kim, Kim+Sean, Monthly Diff, Mon % CHG, TSP, T3W, Roth, Trl IRA, Stocks, Taylor
    
    Maps to database:
    - Sean's accounts: TSP, T3W, Roth IRA, IRA (Trl IRA), Personal (Stocks)
    - Kim's account: Retirement (from Kim column)
    - Taylor's account: Personal
    """
    imported = 0
    errors = []
    
    for idx, row in df_excel.iterrows():
        try:
            date = pd.to_datetime(row['Date']).date()
            
            # Sean's accounts
            if pd.notna(row.get('TSP')):
                add_monthly_update(date, 'Sean', 'TSP', float(str(row['TSP']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('T3W')):
                add_monthly_update(date, 'Sean', 'T3W', float(str(row['T3W']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Roth')):
                add_monthly_update(date, 'Sean', 'Roth IRA', float(str(row['Roth']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Trl IRA')):
                add_monthly_update(date, 'Sean', 'IRA', float(str(row['Trl IRA']).replace('$', '').replace(',', '')))
                imported += 1
            if pd.notna(row.get('Stocks')):
                add_monthly_update(date, 'Sean', 'Personal', float(str(row['Stocks']).replace('$', '').replace(',', '')))
                imported += 1
            
            # Kim's account
            if pd.notna(row.get('Kim')):
                add_monthly_update(date, 'Kim', 'Retirement', float(str(row['Kim']).replace('$', '').replace(',', '')))
                imported += 1
            
            # Taylor's account
            if pd.notna(row.get('Taylor')):
                add_monthly_update(date, 'Taylor', 'Personal', float(str(row['Taylor']).replace('$', '').replace(',', '')))
                imported += 1
                
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    return imported, errors

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_monthly_data(df):
    """Check for common data issues"""
    issues = []
    
    # Check for duplicate dates
    duplicates = df[df.duplicated(subset=['date', 'person', 'account_type'], keep=False)]
    if not duplicates.empty:
        issues.append(f"Found {len(duplicates)} duplicate entries")
    
    # Check for negative values
    negatives = df[df['value'] < 0]
    if not negatives.empty:
        issues.append(f"Found {len(negatives)} negative values")
    
    # Check for unrealistic monthly changes (>50%)
    df_sorted = df.sort_values(['person', 'account_type', 'date'])
    df_sorted['pct_change'] = df_sorted.groupby(['person', 'account_type'])['value'].pct_change()
    extreme = df_sorted[abs(df_sorted['pct_change']) > 0.5]
    if not extreme.empty:
        issues.append(f"Found {len(extreme)} extreme month-over-month changes (>50%)")
    
    # Check for gaps in monthly data
    for person in df['person'].unique():
        person_data = df[df['person'] == person].sort_values('date')
        if len(person_data) > 1:
            date_diff = person_data['date'].diff()
            gaps = date_diff[date_diff > pd.Timedelta(days=45)]  # More than 1.5 months
            if not gaps.empty:
                issues.append(f"{person}: Found {len(gaps)} gaps in monthly data")
    
    return issues

# ============================================================================
# BACKUP & RESTORE
# ============================================================================

def create_database_backup():
    """Create a pg_dump backup file"""
    try:
        conn_url = engine.url
        host = conn_url.host
        port = conn_url.port or 5432
        dbname = conn_url.database
        user = conn_url.username
        password = str(conn_url.password) if conn_url.password else ""

        with NamedTemporaryFile(delete=False, suffix=".dump") as tmpfile:
            dump_path = tmpfile.name

        cmd = [
            "pg_dump",
            f"--host={host}",
            f"--port={port}",
            f"--username={user}",
            f"--dbname={dbname}",
            "--format=custom",
            "--compress=9",
            "--verbose",
            "--no-owner",
            "--no-acl",
            f"--file={dump_path}"
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = password

        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)

        if result.returncode != 0:
            return None, f"Backup failed: {result.stderr}"
        
        return dump_path, None
    
    except Exception as e:
        return None, str(e)

def restore_database_backup(restore_file):
    """Restore database from pg_dump file"""
    try:
        with NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(restore_file.getvalue())
            restore_path = tmpfile.name

        cmd = [
            "pg_restore",
            "--clean", "--if-exists", "--no-owner", "--no-acl",
            f"--host={engine.url.host}",
            f"--port={engine.url.port or 5432}",
            f"--username={engine.url.username}",
            f"--dbname={engine.url.database}",
            restore_path
        ]
        
        env = os.environ.copy()
        env["PGPASSWORD"] = str(engine.url.password or "")
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            return True, "Restore complete!"
        else:
            return False, f"Restore failed: {result.stderr}"
    
    except Exception as e:
        return False, str(e)
    finally:
        if 'restore_path' in locals() and os.path.exists(restore_path):
            os.unlink(restore_path)
