# pages/dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import base64
import subprocess
from tempfile import NamedTemporaryFile
import os

from config.constants import peer_benchmark
from database.operations import (
    load_accounts, add_monthly_update, reset_database,
    get_retirement_goal, set_retirement_goal,
    save_portfolio_csv
)
from data.parser import parse_portfolio_csv
from data.importers import import_excel_format
from analysis.projections import calculate_confidence_score

def show_dashboard(df, df_net, df_port, port_summary):
    # ------------------------------------------------------------------
    # Top Retirement Goal Section
    # ------------------------------------------------------------------
    if not df.empty and not df_net.empty:
        cur_total = df_net["value"].iloc[-1]
        retirement_target = get_retirement_goal()
        progress_pct = (cur_total / retirement_target) * 100
        years_remaining = 2042 - datetime.now().year
        
        confidence, confidence_method = calculate_confidence_score(df_net, retirement_target)
        
        st.markdown("# 🎯 RETIREMENT 2042")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.metric("Current Net Worth (Sean + Kim)", f"${cur_total:,.0f}")
        with col2:
            st.metric("Target", f"${retirement_target:,.0f}")
        with col3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col4:
            delta = "On track" if confidence >= 80 else ("Monitor" if confidence >= 60 else "Action needed")
            st.metric("Confidence", f"{confidence:.0f}%", delta=delta)
        
        if progress_pct >= 100:
            st.success(f"🎉 Goal achieved! {progress_pct:.1f}% of target")
        elif confidence >= 80:
            st.progress(min(progress_pct / 100, 1.0))
            st.success(f"✅ On track • {years_remaining} years • {confidence:.0f}% confidence")
        elif confidence >= 60:
            st.progress(min(progress_pct / 100, 1.0))
            st.warning(f"⚠️ Watch closely • {years_remaining} years • {confidence:.0f}% confidence")
        else:
            st.progress(min(progress_pct / 100, 1.0))
            st.error(f"🚨 Adjustment needed • {years_remaining} years • {confidence:.0f}% confidence")
        
        st.markdown("#### Adjust Retirement Goal")
        new_target = st.slider(
            "Target Amount",
            500000, 5000000, int(retirement_target), step=50000, format="$%d"
        )
        if new_target != retirement_target:
            set_retirement_goal(new_target)
            st.rerun()
        
        st.markdown("---")

    # ------------------------------------------------------------------
    # Peer Benchmark & YTD
    # ------------------------------------------------------------------
    if not df.empty and not df_net.empty:
        cur_total = df_net["value"].iloc[-1]
        pct, vs = peer_benchmark(cur_total)
        st.markdown(f"# ${cur_total:,.0f}")
        st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** • Ahead by **${vs:+,}**")

        st.markdown("#### YTD Growth")
        current_year = datetime.now().year
        ytd_df = df[df["date"].dt.year == current_year].copy()
        if len(ytd_df["date"].unique()) > 1:
            start_vals = ytd_df[ytd_df["date"] == ytd_df["date"].min()].groupby("person")["value"].sum()
            latest_vals = ytd_df[ytd_df["date"] == ytd_df["date"].max()].groupby("person")["value"].sum()
            ytd_pct = ((latest_vals / start_vals) - 1) * 100
            combined_ytd = ((latest_vals.get('Sean',0) + latest_vals.get('Kim',0)) / 
                            (start_vals.get('Sean',1) + start_vals.get('Kim',1)) - 1) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sean YTD", f"{ytd_pct.get('Sean', 0):+.1f}%")
            col2.metric("Kim YTD", f"{ytd_pct.get('Kim', 0):+.1f}%")
            col3.metric("Taylor YTD", f"{ytd_pct.get('Taylor', 0):+.1f}%")
            col4.metric("Combined YTD", f"{combined_ytd:+.1f}%")
        else:
            st.info("Not enough data for YTD yet.")
        
        st.markdown("---")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        with st.expander("S.A.G.E. – Your Strategic Partner", expanded=True):
            st.subheader("Upload Portfolio CSV")
            port_file = st.file_uploader("Fidelity CSV (all accounts)", type="csv")
            
            if port_file:
                df_port_new, port_summary_new = parse_portfolio_csv(port_file)
                if not df_port_new.empty:
                    st.success(f"Loaded {len(df_port_new)} holdings")
                    csv_b64 = base64.b64encode(port_file.getvalue()).decode()
                    save_portfolio_csv(csv_b64)
                    st.rerun()

            if st.button("🧠 Talk to S.A.G.E.", use_container_width=True):
                st.session_state.page = "ai"
                st.rerun()

        st.markdown("---")
        st.subheader("Data Tools")
        
        excel_file = st.file_uploader("Bulk Import - Excel Format", type=["csv", "xlsx"])
        if excel_file:
            try:
                df_import = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                imported, errors = import_excel_format(df_import)
                if imported:
                    st.success(f"Imported {imported} records!")
                    st.rerun()
                if errors:
                    st.warning(f"{len(errors)} errors")
            except Exception as e:
                st.error(f"Import failed: {e}")

        monthly_file = st.file_uploader("Standard CSV Import", type="csv", key="std")
        if monthly_file:
            try:
                df_std = pd.read_csv(monthly_file)
                req = ['date', 'person', 'account_type', 'value']
                if all(c in df_std.columns for c in req):
                    df_std['date'] = pd.to_datetime(df_std['date']).dt.date
                    for _, r in df_std.iterrows():
                        add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value']))
                    st.success(f"Imported {len(df_std)} rows!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Reset Database"):
            if st.checkbox("I understand — delete everything"):
                reset_database()
                st.success("Database reset!")
                st.rerun()

        st.markdown("---")
        st.subheader("Backup & Restore")
        if st.button("Download Full Backup"):
            with st.spinner("Creating backup..."):
                # (backup code omitted for brevity — we can add later if you want)
                st.info("Backup feature coming soon!")

        st.markdown("---")
        st.subheader("Add Update")
        accounts = load_accounts()
        person = st.selectbox("Person", list(accounts.keys()))
        acct = st.selectbox("Account", accounts.get(person, []))
        date_in = st.date_input("Date", datetime.today())
        val = st.number_input("Value ($)", min_value=0.0)
        if st.button("Save"):
            add_monthly_update(date_in, person, acct, val)
            st.success("Saved!")
            st.rerun()

    # ------------------------------------------------------------------
    # Main Tabs
    # ------------------------------------------------------------------
    if df.empty:
        st.info("Add data to see the dashboard!")
        return

    tab1, tab2 = st.tabs(["Retirement (Sean + Kim)", "Taylor's Nest Egg"])

    with tab1:
        # Growth chart and MoM/YoY tables (your original code here — shortened for message length)
        st.info("Full charts and tables will be added in the final file — coming next!")

    with tab2:
        st.info("Taylor's section coming in final file!")

    # Note: The full chart code is very long — we'll complete it in the last file
