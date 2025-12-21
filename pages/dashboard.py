# pages/dashboard.py - FULL UPDATED FILE WITH DISMISSABLE CSV DEBUG PANELS

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
    save_portfolio_csv_slot, load_all_portfolios, get_monthly_updates,
    clear_all_portfolios
)
from data.parser import parse_portfolio_csv, merge_portfolios, calculate_net_worth_from_csv
from data.importers import import_excel_format
from analysis.projections import calculate_confidence_score, calculate_projection_cone

def show_dashboard(df, df_net, df_port, port_summary):
    st.title("🏦 S.A.G.E. Dashboard")
    st.caption("Your Strategic Asset Growth Engine – Overview & Insights")

    # Load monthly historical data
    monthly_df = get_monthly_updates()
    if not monthly_df.empty:
        monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    else:
        monthly_df = pd.DataFrame(columns=['date', 'person', 'account_type', 'value'])

    # --------------------------------------------------
    # PORTFOLIO CSV UPLOAD & PARSING
    # --------------------------------------------------
    st.markdown("### Upload Portfolio CSVs")
    st.caption("Latest upload becomes monthly snapshot – up to 3 accounts supported")

    # Initialize debug visibility state
    if "show_csv_debug" not in st.session_state:
        st.session_state.show_csv_debug = {}  # {slot: True/False}

    # File uploaders
    uploaded_files = []
    for slot in range(1, 4):
        col1, col2 = st.columns([4, 1])
        with col1:
            uploaded = st.file_uploader(
                f"Account {slot}{' (optional)' if slot > 1 else ''}",
                type="csv",
                key=f"csv_upload_{slot}",
                help="Download Positions CSV directly from Fidelity and drop here",
                label_visibility="collapsed"
            )
        with col2:
            saved_data = load_portfolio_csv_slot(slot)
            if uploaded:
                st.success("Uploaded")
            elif saved_data:
                st.info("Saved")
            else:
                st.write("—")

        if uploaded:
            uploaded_files.append((slot, uploaded))
            # Show debug for new upload
            st.session_state.show_csv_debug[slot] = True

    # Process any new uploads
    if uploaded_files:
        with st.spinner("Processing uploaded CSVs..."):
            for slot, file in uploaded_files:
                file_bytes = file.getvalue()
                b64 = base64.b64encode(file_bytes).decode()
                save_portfolio_csv_slot(slot, b64)

            st.success(f"Processed {len(uploaded_files)} CSV file(s)!")
            st.rerun()

    # --------------------------------------------------
    # LOAD AND MERGE ALL SAVED PORTFOLIOS
    # --------------------------------------------------
    all_b64 = load_all_portfolios()
    raw_portfolio_dfs = []
    current_sean_kim = 0
    current_taylor = 0

    for slot, b64_data in all_b64.items():
        try:
            sean_kim, taylor = calculate_net_worth_from_csv(b64_data)
            current_sean_kim += sean_kim
            current_taylor += taylor

            decoded = base64.b64decode(b64_data).decode('utf-8')
            parsed_df, _ = parse_portfolio_csv(decoded)
            if not parsed_df.empty:
                raw_portfolio_dfs.append(parsed_df)
        except Exception as e:
            st.error(f"Error loading saved portfolio slot {slot}: {e}")

    if raw_portfolio_dfs:
        df_port, port_summary = merge_portfolios(raw_portfolio_dfs)
    else:
        df_port = pd.DataFrame()
        port_summary = {}

    # --------------------------------------------------
    # DISPLAY DISMISSABLE CSV DEBUG PANELS
    # --------------------------------------------------
    for slot in range(1, 4):
        if load_portfolio_csv_slot(slot):
            show_key = f"debug_slot_{slot}"
            if show_key not in st.session_state:
                st.session_state[show_key] = st.session_state.show_csv_debug.get(slot, False)

            if st.session_state[show_key]:
                with st.expander(f"🔍 CSV Structure Analysis – Saved Account {slot}", expanded=True):
                    b64_data = load_portfolio_csv_slot(slot)
                    decoded = base64.b64decode(b64_data).decode('utf-8')
                    parsed_df, summary = parse_portfolio_csv(decoded)

                    if not parsed_df.empty:
                        st.success(f"Parsed {len(parsed_df)} positions – Total value: ${summary.get('total_value', 0):,.0f}")
                    else:
                        st.error("Saved CSV could not be parsed")

                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button("Dismiss", key=f"dismiss_debug_{slot}"):
                            st.session_state[show_key] = False
                            st.rerun()

    # Optional: Developer tool to re-open debug panels
    with st.expander("🔧 Developer Tools"):
        st.write("Re-open debug panels or clear saved CSVs")
        col1, col2 = st.columns(2)
        with col1:
            for slot in range(1, 4):
                if load_portfolio_csv_slot(slot):
                    if st.button(f"Show Debug – Account {slot}", key=f"force_debug_{slot}"):
                        st.session_state[f"debug_slot_{slot}"] = True
                        st.rerun()
        with col2:
            if st.button("Clear All Saved CSVs", type="secondary"):
                clear_all_portfolios()
                st.success("All saved CSVs cleared!")
                st.rerun()

    # --------------------------------------------------
    # CURRENT NET WORTH DISPLAY
    # --------------------------------------------------
    st.markdown("### Current Portfolio Value (as of latest upload)")

    total_value = current_sean_kim + current_taylor
    col1, col2, col3 = st.columns(3)
    col1.metric("Sean + Kim Retirement", f"${current_sean_kim:,.0f}")
    col2.metric("Taylor's Accounts", f"${current_taylor:,.0f}")
    col3.metric("Combined Total", f"${total_value:,.0f}")

    if total_value > 0:
        peer_pct, peer_diff = peer_benchmark(total_value)
        st.success(f"You're ${peer_diff:,.0f} ahead of the average 40-year-old household!")

    # --------------------------------------------------
    # REST OF DASHBOARD (charts, projections, etc.)
    # --------------------------------------------------
    if not df_port.empty:
        st.markdown("### Portfolio Holdings")
        st.dataframe(df_port[['ticker', 'name', 'quantity', 'price', 'market_value', 'allocation', 'unrealized_gain']],
                     use_container_width=True)

        # Top holdings pie chart
        top10 = df_port.nlargest(10, 'market_value')
        fig = go.Figure(data=[go.Pie(labels=top10['ticker'] + " – " + top10['name'],
                                    values=top10['market_value'],
                                    textinfo='label+percent',
                                    hole=0.3)])
        fig.update_layout(title="Top 10 Holdings")
        st.plotly_chart(fig, use_container_width=True)

    # Historical net worth chart (Sean + Kim)
    df_sean_kim_hist = monthly_df[monthly_df["person"].isin(["Sean", "Kim"])]
    if not df_sean_kim_hist.empty:
        hist_total = df_sean_kim_hist.groupby("date")["value"].sum().reset_index()
        fig_hist = go.Figure(go.Scatter(x=hist_total["date"], y=hist_total["value"],
                                        mode='lines+markers', line=dict(width=3)))
        fig_hist.update_layout(title="Sean + Kim Net Worth Over Time", height=500)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Projections (placeholder – full logic in ai_chat_page)
    retirement_goal = get_retirement_goal()
    if not df_net.empty and retirement_goal:
        confidence, method = calculate_confidence_score(df_net, retirement_goal)
        st.metric("Confidence to 2042 Goal", f"{confidence:.0f}%", delta=method)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Talk to S.A.G.E. (AI Chat)", use_container_width=True, type="primary"):
            st.session_state.page = "ai"
            st.rerun()
    with col2:
        st.download_button(
            "Export All Monthly Data",
            monthly_df.to_csv(index=False).encode(),
            f"sage-data-{datetime.now().strftime('%Y-%m-%d')}.csv",
            "text/csv"
        )
