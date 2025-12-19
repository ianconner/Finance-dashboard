# data/importers.py

import pandas as pd
import streamlit as st
from database.operations import add_monthly_update

def import_excel_format(df_excel):
    """
    Import historical data from the old Excel format.
    Columns expected: Date, Sean, Kim, TSP, T3W, Roth, Tri IRA (or Trl IRA), Stocks, Taylor
    """
    imported = 0
    errors = []

    for idx, row in df_excel.iterrows():
        try:
            # Parse date flexibly
            date_str = str(row['Date']).strip()
            date = pd.to_datetime(date_str).date()

            # Helper to clean monetary values
            def clean_value(val):
                if pd.isna(val):
                    return None
                val_str = str(val).replace('$', '').replace(',', '').strip()
                if val_str == '' or val_str.lower() == 'nan':
                    return None
                try:
                    return float(val_str)
                except:
                    return None

            # Flag if we have detailed Sean accounts
            sean_detailed = False

            # Sean detailed accounts
            for col, acct in [('TSP', 'TSP'), ('T3W', 'T3W'), ('Roth', 'Roth IRA'), ('Stocks', 'Personal')]:
                val = clean_value(row.get(col))
                if val is not None:
                    add_monthly_update(date, 'Sean', acct, val)
                    imported += 1
                    sean_detailed = True

            # Tri IRA (handles both "Tri IRA" and "Trl IRA" spellings)
            tri_val = clean_value(row.get('Tri IRA')) or clean_value(row.get('Trl IRA'))
            if tri_val is not None:
                add_monthly_update(date, 'Sean', 'IRA', tri_val)
                imported += 1
                sean_detailed = True

            # Fallback: if no detailed Sean data, use the "Sean" total column
            if not sean_detailed:
                sean_total = clean_value(row.get('Sean'))
                if sean_total is not None:
                    add_monthly_update(date, 'Sean', 'Personal', sean_total)
                    imported += 1

            # Kim
            kim_val = clean_value(row.get('Kim'))
            if kim_val is not None:
                add_monthly_update(date, 'Kim', 'Retirement', kim_val)
                imported += 1

            # Taylor
            taylor_val = clean_value(row.get('Taylor'))
            if taylor_val is not None:
                add_monthly_update(date, 'Taylor', 'Personal', taylor_val)
                imported += 1

        except Exception as e:
            errors.append(f"Row {idx} (Date: {row.get('Date', 'unknown')}): {str(e)}")

    return imported, errors
