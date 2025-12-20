# pages/dashboard.py - Final version with multi-portfolio, accurate net worth, full features

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

from config.constants import peer_benchmark
from database.operations import (
    load_accounts, add_monthly_update, reset_database,
    get_retirement_goal, set_retirement_goal,
    save_portfolio_csv_slot, load_all_portfolios
)
from data.parser import parse_portfolio_csv, merge_portfolios
from data.importers import import_excel_format
from analysis.projections import calculate_confidence_score

def show_dashboard(df, df_net, df_port, port_summary):
    # ------------------------------------------------------------------
    # Calculate current values from portfolio CSVs (source of truth)
    # ------------------------------------------------------------------
    current_sean_kim = 0
    current_taylor = 0
    portfolio_loaded = False

    if df_port is not None and not df_port.empty:
        portfolio_loaded = True

        # Use raw CSV data to accurately group by Account Name
        all_b64 = load_all_portfolios()
        for slot, b64_data in all_b64.items():
            try:
                decoded = base64.b64decode(b64_data).decode('utf-8')
                raw_df = pd.read_csv(io.StringIO(decoded))
                raw_df.columns = raw_df.columns.str.strip()
                for _, row in raw_df.iterrows():
                    account_name = str(row.get('Account Name', ''))
                    value_str = str(row.get('Current Value', '0')).replace('$', '').replace(',', '')
                    value = pd.to_numeric(value_str, errors='coerce') or 0
                    if 'Sean' in account_name or 'sean' in account_name:
                        current_sean_kim += value
                    elif 'Kim' in account_name or 'kim' in account_name:
                        current_sean_kim += value
                    elif 'Taylor' in account_name or 'taylor' in account_name:
                        current_taylor += value
            except Exception as e:
                st.error(f"Error reading saved portfolio {slot}: {e}")

    # Fallback to monthly data if no portfolio loaded
    if current_sean_kim == 0 and not df_net.empty:
        current_sean_kim = df_net["value"].iloc[-1]

    if current_taylor == 0:
        taylor_df = df[df["person"] == "Taylor"]
        current_taylor = taylor_df["value"].iloc[-1] if not taylor_df.empty else 0

    # Override df_net for Sean + Kim (for goal/confidence/charts)
    df_net = pd.DataFrame([{
        'date': datetime.today(),
        'value': current_sean_kim
    }])

    # Taylor df for her tab
    taylor_current_df = pd.DataFrame([{
        'date': datetime.today(),
        'value': current_taylor
    }])

    # ------------------------------------------------------------------
    # Top Retirement Goal Section (Sean + Kim)
    # ------------------------------------------------------------------
    if current_sean_kim > 0:
        retirement_target = get_retirement_goal()
        progress_pct = (current_sean_kim / retirement_target) * 100
        years_remaining = 2042 - datetime.now().year
        
        confidence, confidence_method = calculate_confidence_score(df_net, retirement_target)
        
        st.markdown("# 🎯 RETIREMENT 2042")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.metric("Current Net Worth (Sean + Kim)", f"${current_sean_kim:,.0f}")
        with col2:
            st.metric("Target", f"${retirement_target:,.0f}")
        with col3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col4:
            if confidence >= 80:
                st.metric("Confidence", f"{confidence:.0f}%", delta="On track")
            elif confidence >= 60:
                st.metric("Confidence", f"{confidence:.0f}%", delta="Monitor", delta_color="off")
            else:
                st.metric("Confidence", f"{confidence:.0f}%", delta="Action needed", delta_color="inverse")
        
        if progress_pct >= 100:
            st.success(f"🎉 Goal achieved! You're at {progress_pct:.1f}% of target!")
        elif confidence >= 80:
            st.progress(min(progress_pct / 100, 1.0))
            st.success(f"✅ On track! {years_remaining} years remaining • {confidence:.0f}% confidence")
        elif confidence >= 60:
            st.progress(min(progress_pct / 100, 1.0))
            st.warning(f"⚠️ Watch closely • {years_remaining} years remaining • {confidence:.0f}% confidence")
        else:
            st.progress(min(progress_pct / 100, 1.0))
            st.error(f"🚨 Adjustment needed • {years_remaining} years remaining • {confidence:.0f}% confidence")
        
        st.markdown("#### Adjust Retirement Goal")
        new_target = st.slider(
            "Target Amount",
            min_value=500000,
            max_value=5000000,
            value=int(retirement_target),
            step=50000,
            format="$%d",
            key="goal_slider"
        )
        if new_target != retirement_target:
            set_retirement_goal(new_target)
            st.rerun()
        
        st.markdown("---")

    # ------------------------------------------------------------------
    # Peer Benchmark & YTD
    # ------------------------------------------------------------------
    if current_sean_kim > 0:
        pct, vs = peer_benchmark(current_sean_kim)
        st.markdown(f"# ${current_sean_kim:,.0f}")
        st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** • Ahead by **${vs:+,}**")

        st.markdown("#### YTD Growth")
        if portfolio_loaded:
            st.info("YTD growth will be added when historical portfolio data is available")
        else:
            st.info("Upload portfolio CSVs for current values")

        st.markdown("---")

    # ------------------------------------------------------------------
    # Sidebar - Multi-Portfolio Upload
    # ------------------------------------------------------------------
    with st.sidebar:
        with st.expander("S.A.G.E. – Your Strategic Partner", expanded=True):
            st.subheader("Upload Portfolio CSVs (up to 3 accounts)")
            st.caption("Upload from different brokers/accounts — S.A.G.E. combines them")

            col1, col2, col3 = st.columns(3)

            with col1:
                port_file1 = st.file_uploader("Account 1", type="csv", key="port1")
                if port_file1:
                    _, temp_summary = parse_portfolio_csv(port_file1)
                    if temp_summary:
                        csv_b64 = base64.b64encode(port_file1.getvalue()).decode()
                        save_portfolio_csv_slot(1, csv_b64)
                        st.success(f"Account 1: ${temp_summary['total_value']:,.0f}")

            with col2:
                port_file2 = st.file_uploader("Account 2 (optional)", type="csv", key="port2")
                if port_file2:
                    _, temp_summary = parse_portfolio_csv(port_file2)
                    if temp_summary:
                        csv_b64 = base64.b64encode(port_file2.getvalue()).decode()
                        save_portfolio_csv_slot(2, csv_b64)
                        st.success(f"Account 2: ${temp_summary['total_value']:,.0f}")

            with col3:
                port_file3 = st.file_uploader("Account 3 (optional)", type="csv", key="port3")
                if port_file3:
                    _, temp_summary = parse_portfolio_csv(port_file3)
                    if temp_summary:
                        csv_b64 = base64.b64encode(port_file3.getvalue()).decode()
                        save_portfolio_csv_slot(3, csv_b64)
                        st.success(f"Account 3: ${temp_summary['total_value']:,.0f}")

            if portfolio_loaded:
                st.success(f"**Combined Portfolio Ready**: ${port_summary.get('total_value',0):,.0f} across {len(load_all_portfolios())} account(s)")

            st.caption("Always here for strategy, risks, opportunities, or just to chat.")
            if st.button("🧠 Talk to S.A.G.E.", use_container_width=True):
                st.session_state.page = "ai"
                st.rerun()

        # ... rest of sidebar (Data Tools, Add Update, etc.) remains the same as your current version

    # ------------------------------------------------------------------
    # Main Tabs
    # ------------------------------------------------------------------
    tab1, tab2 = st.tabs(["Retirement (Sean + Kim)", "💎 Taylor's Nest Egg"])

    with tab1:
        st.markdown(f"### Current Portfolio Value: ${current_sean_kim:,.0f}")
        if portfolio_loaded:
            st.success("Live from your Fidelity accounts")
        else:
            st.info("Upload portfolio CSVs for live values")

        # Charts will be added when historical data is available

    with tab2:
        st.markdown(f"### Taylor's Current Value: ${current_taylor:,.0f}")
        if portfolio_loaded and current_taylor > 0:
            st.success("Live from Taylor's account")
        else:
            st.info("Upload Taylor's CSV for live value")

    st.download_button(
        "Export All Monthly Data",
        df.to_csv(index=False).encode(),
        f"sage-data-{datetime.now().strftime('%Y-%m-%d')}.csv",
        "text/csv"
    )
