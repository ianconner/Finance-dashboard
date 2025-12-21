# pages/dashboard.py - COMPLETE FIXED VERSION

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
    save_portfolio_csv_slot, load_all_portfolios, get_monthly_updates, get_session
)
from data.parser import parse_portfolio_csv, merge_portfolios, calculate_net_worth_from_csv
from data.importers import import_excel_format
from analysis.projections import calculate_confidence_score

def show_dashboard(df, df_net, df_port, port_summary):
    # Initialize session state for portfolio details toggle
    if "show_csv_detail" not in st.session_state:
        st.session_state.show_csv_detail = False
    
    # Load historical monthly data
    df = get_monthly_updates()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = pd.DataFrame(columns=['date', 'person', 'account_type', 'value'])

    # Calculate current values from portfolio CSVs (source of truth)
    current_sean_kim = 0
    current_taylor = 0
    portfolio_loaded = False

    all_b64 = load_all_portfolios()
    raw_portfolio_data = []

    for slot, b64_data in all_b64.items():
        try:
            # Calculate net worth from CSV (NO ANALYSIS)
            sean_kim_val, taylor_val = calculate_net_worth_from_csv(b64_data)
            current_sean_kim += sean_kim_val
            current_taylor += taylor_val
            
            # Also parse for portfolio details (NO ANALYSIS)
            decoded = base64.b64decode(b64_data).decode('utf-8')
            parsed_df, _ = parse_portfolio_csv(decoded, show_analysis=False)
            if not parsed_df.empty:
                raw_portfolio_data.append(parsed_df)
                portfolio_loaded = True
                
        except Exception as e:
            st.error(f"Error reading saved portfolio {slot}: {e}")

    # Merge all portfolio data for display
    if raw_portfolio_data:
        df_port, port_summary = merge_portfolios(raw_portfolio_data)

    # Net worth from monthly data
    df_sean_kim = df[df["person"].isin(["Sean", "Kim"])]
    df_sean_kim_total = df_sean_kim.groupby("date")["value"].sum().reset_index().sort_values("date")
    
    if not df_sean_kim_total.empty:
        current_sean_kim_from_db = df_sean_kim_total["value"].iloc[-1]
        # Use portfolio value if available AND it's non-zero
        if current_sean_kim == 0:
            current_sean_kim = current_sean_kim_from_db

    df_taylor = df[df["person"] == "Taylor"]
    df_taylor_total = df_taylor.groupby("date")["value"].sum().reset_index().sort_values("date")
    
    if not df_taylor_total.empty:
        current_taylor_from_db = df_taylor_total["value"].iloc[-1]
        if current_taylor == 0:
            current_taylor = current_taylor_from_db

    # ------------------------------------------------------------------
    # RETIREMENT GOAL SECTION
    # ------------------------------------------------------------------
    retirement_target = get_retirement_goal()
    
    if current_sean_kim > 0:
        progress_pct = (current_sean_kim / retirement_target) * 100
        years_remaining = 2042 - datetime.now().year
        
        # Use historical data for confidence calculation
        confidence, confidence_method = calculate_confidence_score(df_sean_kim_total, retirement_target)
        
        st.markdown("# 🎯 RETIREMENT 2042")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.metric("Current Net Worth (Sean + Kim)", f"${current_sean_kim:,.0f}")
        with col2:
            st.metric("Target", f"${retirement_target:,.0f}")
        with col3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col4:
            delta_text = "On track" if confidence >= 80 else ("Monitor" if confidence >= 60 else "Action needed")
            st.metric("Confidence", f"{confidence:.0f}%", delta=delta_text)
        
        st.progress(min(progress_pct / 100, 1.0))
        if confidence >= 80:
            st.success(f"✅ On track! {years_remaining} years remaining • {confidence:.0f}% confidence")
        elif confidence >= 60:
            st.warning(f"⚠️ Watch closely • {years_remaining} years remaining • {confidence:.0f}% confidence")
        else:
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
            try:
                set_retirement_goal(new_target)
                st.rerun()
            except Exception as e:
                st.error(f"Could not update goal: {e}")
        
        st.markdown("---")

    # ------------------------------------------------------------------
    # Peer Benchmark
    # ------------------------------------------------------------------
    if current_sean_kim > 0:
        pct, vs = peer_benchmark(current_sean_kim)
        st.markdown(f"# ${current_sean_kim:,.0f}")
        st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** • Ahead by **${vs:+,}**")
        st.markdown("---")

    # ------------------------------------------------------------------
    # Sidebar - Clean Uploads
    # ------------------------------------------------------------------
    with st.sidebar:
        st.subheader("📊 Upload Portfolio CSVs")
        st.caption("Latest upload becomes monthly snapshot — up to 3 accounts")

        # Check which slots already have data
        existing_slots = list(all_b64.keys())

        st.markdown("#### Account 1")
        if 1 in existing_slots:
            st.success("✅ Account 1 loaded")
        port_file1 = st.file_uploader("Portfolio CSV", type="csv", key="port1", label_visibility="collapsed")
        if port_file1:
            try:
                parsed_df, temp_summary = parse_portfolio_csv(port_file1, show_analysis=False)
                if temp_summary:
                    csv_b64 = base64.b64encode(port_file1.getvalue()).decode()
                    save_portfolio_csv_slot(1, csv_b64)
                    st.success(f"✅ Account 1: ${temp_summary.get('total_value', 0):,.0f}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("#### Account 2 (optional)")
        if 2 in existing_slots:
            st.success("✅ Account 2 loaded")
        port_file2 = st.file_uploader("Portfolio CSV", type="csv", key="port2", label_visibility="collapsed")
        if port_file2:
            try:
                parsed_df, temp_summary = parse_portfolio_csv(port_file2, show_analysis=False)
                if temp_summary:
                    csv_b64 = base64.b64encode(port_file2.getvalue()).decode()
                    save_portfolio_csv_slot(2, csv_b64)
                    st.success(f"✅ Account 2: ${temp_summary.get('total_value', 0):,.0f}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("#### Account 3 (optional)")
        if 3 in existing_slots:
            st.success("✅ Account 3 loaded")
        port_file3 = st.file_uploader("Portfolio CSV", type="csv", key="port3", label_visibility="collapsed")
        if port_file3:
            try:
                parsed_df, temp_summary = parse_portfolio_csv(port_file3, show_analysis=False)
                if temp_summary:
                    csv_b64 = base64.b64encode(port_file3.getvalue()).decode()
                    save_portfolio_csv_slot(3, csv_b64)
                    st.success(f"✅ Account 3: ${temp_summary.get('total_value', 0):,.0f}")
            except Exception as e:
                st.error(f"Error: {e}")

        # Refresh button to reload after uploads
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

        # Show summary and save button when portfolios are loaded
        if portfolio_loaded:
            st.markdown("---")
            st.info(f"**Combined Total**\nSean+Kim: ${current_sean_kim:,.0f}\nTaylor: ${current_taylor:,.0f}")
            
            # Manual snapshot save button
            if st.button("💾 Save Current Snapshot", use_container_width=True, type="primary"):
                today = pd.Timestamp.today()
                snapshot_date = (today + pd.offsets.MonthEnd(0)).date()
                
                if current_sean_kim <= 0:
                    st.error("❌ Cannot save: Sean+Kim total is $0. Please check CSV upload.")
                else:
                    try:
                        add_monthly_update(snapshot_date, 'Sean', 'Personal', current_sean_kim)
                        if current_taylor > 0:
                            add_monthly_update(snapshot_date, 'Taylor', 'Personal', current_taylor)
                        st.success(f"✅ Snapshot saved for {snapshot_date.strftime('%B %Y')}")
                        st.info(f"Sean+Kim: ${current_sean_kim:,.0f}" + (f"\nTaylor: ${current_taylor:,.0f}" if current_taylor > 0 else ""))
                    except Exception as e:
                        st.error(f"❌ Could not save snapshot: {e}")

            st.caption("Ready when you are.")
            if st.button("🧠 Talk to S.A.G.E.", use_container_width=True):
                st.session_state.page = "ai"
                st.rerun()

        # CSV Data Preview
        if portfolio_loaded and not df_port.empty:
            st.markdown("---")
            st.markdown("### 📄 Portfolio Details")
            
            if st.button(
                "👁️ Show Details" if not st.session_state.show_csv_detail else "👁️ Hide Details",
                use_container_width=True,
                key="toggle_details"
            ):
                st.session_state.show_csv_detail = not st.session_state.show_csv_detail
                st.rerun()
            
            if st.session_state.show_csv_detail:
                st.markdown("#### Holdings by Account")
                if 'account_name' in df_port.columns:
                    for account in df_port['account_name'].unique():
                        account_data = df_port[df_port['account_name'] == account]
                        account_total = account_data['market_value'].sum()
                        
                        with st.expander(f"{account} (${account_total:,.2f})", expanded=False):
                            display_cols = ['ticker', 'shares', 'price', 'market_value', 'cost_basis', 'unrealized_gain', 'pct_gain']
                            if 'name' in account_data.columns:
                                display_cols = ['ticker', 'name', 'shares', 'price', 'market_value', 'cost_basis', 'unrealized_gain', 'pct_gain']
                            
                            display_data = account_data[display_cols].copy()
                            
                            st.dataframe(
                                display_data,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                                    "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                                    "unrealized_gain": st.column_config.NumberColumn("Gain/Loss", format="$%.2f"),
                                    "pct_gain": st.column_config.NumberColumn("Gain %", format="%.2f%%"),
                                    "shares": st.column_config.NumberColumn("Shares", format="%.4f")
                                }
                            )
                
                st.markdown("#### Summary by Person")
                summary_data = []
                for account in df_port['account_name'].unique():
                    account_total = df_port[df_port['account_name'] == account]['market_value'].sum()
                    account_lower = account.lower()
                    if 'sean' in account_lower or 'kim' in account_lower:
                        person = 'Sean/Kim'
                    elif 'taylor' in account_lower:
                        person = 'Taylor'
                    else:
                        person = 'Unknown'
                    summary_data.append({
                        'Account': account,
                        'Person': person,
                        'Total Value': account_total
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total Value": st.column_config.NumberColumn("Total Value", format="$%.2f")
                    }
                )

        st.markdown("---")
        st.subheader("Data Tools")
        
        with st.expander("🗑️ Delete Data Entries"):
            st.caption("Remove corrupted or incorrect monthly entries")
            
            if not df.empty:
                recent_dates = sorted(df['date'].unique(), reverse=True)[:6]
                
                for date in recent_dates:
                    date_data = df[df['date'] == date]
                    date_str = date.strftime('%B %Y')
                    
                    with st.container():
                        st.markdown(f"**{date_str}**")
                        for idx, row in date_data.iterrows():
                            col1, col2, col3 = st.columns([2, 2, 1])
                            col1.write(f"{row['person']}")
                            col2.write(f"${row['value']:,.0f}")
                            if col3.button("Delete", key=f"del_{date.strftime('%Y%m%d')}_{row['person']}_{row['account_type']}_{idx}"):
                                try:
                                    from database.models import MonthlyUpdate
                                    sess = get_session()
                                    try:
                                        sess.query(MonthlyUpdate).filter(
                                            MonthlyUpdate.date == date.date(),
                                            MonthlyUpdate.person == row['person'],
                                            MonthlyUpdate.account_type == row['account_type']
                                        ).delete()
                                        sess.commit()
                                        st.success(f"Deleted {row['person']} entry for {date_str}")
                                        st.rerun()
                                    except Exception as e:
                                        sess.rollback()
                                        st.error(f"Delete failed: {e}")
                                    finally:
                                        sess.close()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        st.markdown("---")
            else:
                st.info("No data entries to delete")
        
        st.markdown("**Bulk Import - Excel Format**")
        excel_file = st.file_uploader("Upload historical Excel data", type=["csv", "xlsx"], key="excel_import")
        if excel_file:
            try:
                df_import = pd.read_excel(excel_file) if excel_file.name.endswith('.xlsx') else pd.read_csv(excel_file)
                imported, errors = import_excel_format(df_import)
                if imported > 0:
                    st.success(f"Imported {imported} records!")
                    st.rerun()
                if errors:
                    with st.expander("View Errors"):
                        for err in errors:
                            st.text(err)
            except Exception as e:
                st.error(f"Import failed: {e}")

        st.markdown("**Standard CSV Import**")
        monthly_file = st.file_uploader("CSV (date,person,account_type,value)", type="csv", key="monthly")
        if monthly_file:
            try:
                df_std = pd.read_csv(monthly_file)
                req = ['date', 'person', 'account_type', 'value']
                if all(c in df_std.columns for c in req):
                    df_std['date'] = pd.to_datetime(df_std['date']).dt.date
                    count = 0
                    for _, r in df_std.iterrows():
                        try:
                            add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value']))
                            count += 1
                        except Exception as e:
                            st.warning(f"Skipped row: {e}")
                    st.success(f"Imported {count} rows!")
                    st.rerun()
                else:
                    st.error(f"Missing columns. Need: {', '.join(req)}")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Reset Database"):
            if st.checkbox("I understand this deletes all data", key="confirm_reset"):
                try:
                    reset_database()
                    st.success("Database reset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {e}")

        st.markdown("---")
        st.subheader("Add Update")
        accounts = load_accounts()
        person = st.selectbox("Person", list(accounts.keys()))
        acct = st.selectbox("Account", accounts.get(person, []))
        col1, col2 = st.columns(2)
        with col1:
            date_in = st.date_input("Date", value=datetime.today())
        with col2:
            val = st.number_input("Value ($)", min_value=0.0)
        if st.button("Save"):
            try:
                add_monthly_update(date_in, person, acct, float(val))
                st.success("Saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ------------------------------------------------------------------
    # Main Tabs
    # ------------------------------------------------------------------
    if df.empty:
        st.info("Add data to see the full dashboard.")
        return

    tab1, tab2 = st.tabs(["Retirement (Sean + Kim)", "💎 Taylor's Nest Egg"])

    with tab1:
        df_pivot = df.pivot_table(index="date", columns="person", values="value", aggfunc="sum")
        df_pivot = df_pivot.resample("ME").last().ffill()
        
        df_sean_kim_plot = pd.DataFrame()
        df_sean_kim_plot['Sean'] = df_pivot.get('Sean', 0)
        df_sean_kim_plot['Kim'] = df_pivot.get('Kim', 0)
        df_sean_kim_plot["Sean + Kim"] = df_sean_kim_plot["Sean"] + df_sean_kim_plot["Kim"]

        fig = go.Figure()
        colors = {"Sean": "#636EFA", "Kim": "#EF553B", "Sean + Kim": "#AB63FA"}
        widths = {"Sean + Kim": 5, "Sean": 3, "Kim": 3}

        for person in ["Sean", "Kim", "Sean + Kim"]:
            if person in df_sean_kim_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_sean_kim_plot.index,
                    y=df_sean_kim_plot[person],
                    name=person,
                    mode='lines',
                    line=dict(color=colors[person], width=widths.get(person, 3)),
                    hovertemplate=f"<b>{person}</b><br>%{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>"
                ))

        fig.update_layout(
            title="Retirement Portfolio Growth (Sean + Kim)",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1),
            yaxis=dict(rangemode='tozero')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### YTD Growth (Jan 1 → Today)")
        current_year = datetime.now().year
        ytd_data = df_sean_kim_plot[df_sean_kim_plot.index.year == current_year].copy()
        
        if not ytd_data.empty and len(ytd_data) >= 2:
            jan_data = ytd_data.iloc[0]
            latest_data = ytd_data.iloc[-1]
            
            ytd_pct = {}
            for person in ['Sean', 'Kim', 'Taylor']:
                if person in ytd_data.columns:
                    start_val = jan_data[person]
                    end_val = latest_data[person]
                    ytd_pct[person] = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0
                else:
                    ytd_pct[person] = 0
            
            if 'Sean + Kim' in ytd_data.columns:
                combined_start = jan_data['Sean + Kim']
                combined_end = latest_data['Sean + Kim']
                combined_ytd = ((combined_end / combined_start) - 1) * 100 if combined_start > 0 else 0
            else:
                combined_ytd = 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("**Sean YTD**", f"{ytd_pct.get('Sean', 0):+.1f}%")
            col2.metric("**Kim YTD**", f"{ytd_pct.get('Kim', 0):+.1f}%")
            col3.metric("**Taylor YTD**", f"{ytd_pct.get('Taylor', 0):+.1f}%")
            col4.metric("**Combined YTD**", f"{combined_ytd:+.1f}%")
        else:
            st.info(f"Not enough data for YTD {current_year}.")

        st.markdown("---")
        st.markdown("## 📈 Month-over-Month Analysis")

        def color_val(val):
            if isinstance(val, (int, float)):
                return 'background-color: #90EE90; color: black' if val > 0 else ('background-color: #FF6B6B; color: black' if val < 0 else '')
            return ''

        mom_dollar = df_sean_kim_plot.diff().round(0)
        mom_pct = df_sean_kim_plot.pct_change() * 100
        years = sorted(df_sean_kim_plot.index.year.unique(), reverse=True)
        year_tabs = st.tabs([str(y) for y in years])

        for tab, year in zip(year_tabs, years):
            with tab:
                mask = df_sean_kim_plot.index.year == year
                display_data = []
                for date in df_sean_kim_plot[mask].index:
                    row = {'Date': date.strftime('%b %Y')}
                    for p in ['Sean', 'Kim', 'Sean + Kim']:
                        if p in df_sean_kim_plot.columns:
                            row[f'{p} $'] = mom_dollar.loc[date, p] if date in mom_dollar.index else 0
                            row[f'{p} %'] = mom_pct.loc[date, p] if date in mom_pct.index else 0
                    display_data.append(row)
                df_display = pd.DataFrame(display_data)

                styled = df_display.style.map(color_val, subset=[c for c in df_display.columns if c != 'Date'])\
                    .format({c: '${:,.0f}' if '$' in c else '{:+.2f}%' for c in df_display.columns if c != 'Date'})
                st.dataframe(styled, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📅 Year-over-Year (December to December)")
        december_data = df_sean_kim_plot[df_sean_kim_plot.index.month == 12]
        if len(december_data) >= 2:
            yoy_data = []
            years = sorted(december_data.index.year.unique())
            for i in range(len(years) - 1):
                prev = december_data[december_data.index.year == years[i]].iloc[0]
                curr = december_data[december_data.index.year == years[i+1]].iloc[0]
                row = {'Period': f'Dec {years[i]} → Dec {years[i+1]}'}
                for p in ['Sean', 'Kim', 'Sean + Kim']:
                    if p in prev and p in curr and prev[p] > 0:
                        row[f'{p} $'] = curr[p] - prev[p]
                        row[f'{p} %'] = ((curr[p] / prev[p]) - 1) * 100
                yoy_data.append(row)
            df_yoy = pd.DataFrame(yoy_data)
            styled_yoy = df_yoy.style.map(color_val, subset=[c for c in df_yoy.columns if c != 'Period'])\
                .format({c: '${:,.0f}' if '$' in c else '{:+.2f}%' for c in df_yoy.columns if c != 'Period'})
            st.dataframe(styled_yoy, use_container_width=True)
        else:
            st.info("Need at least 2 years of December data")

    with tab2:
        st.markdown("# 💎 Taylor's Nest Egg")
        st.caption("Building long-term wealth for Taylor's future")
        
        taylor_df = df[df["person"] == "Taylor"].sort_values("date")
        
        if not taylor_df.empty:
            current = taylor_df["value"].iloc[-1]
            start = taylor_df["value"].iloc[0]
            growth = ((current / start) - 1) * 100 if start > 0 else 0
            years = (taylor_df['date'].iloc[-1] - taylor_df['date'].iloc[0]).days / 365.25
            cagr = ((current / start) ** (1/years) - 1) * 100 if years > 0 and start > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Value", f"${current:,.0f}")
            col2.metric("Total Growth", f"{growth:+.1f}%")
            col3.metric("CAGR", f"{cagr:.1f}%")

            fig_t = go.Figure(go.Scatter(
                x=taylor_df["date"], y=taylor_df["value"],
                line=dict(color="#00CC96", width=3),
                fill='tozeroy', fillcolor='rgba(0,204,150,0.1)',
                hovertemplate="<b>Taylor</b><br>%{x|%b %Y}: $%{y:,.0f}<extra></extra>"
            ))
            fig_t.update_layout(
                title="Taylor's Portfolio Growth Over Time",
                height=500,
                yaxis=dict(rangemode='tozero')
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎯 Long-Term Outlook for Taylor")
            taylor_age = datetime.now().year - 2021
            
            if cagr > 0:
                proj_18 = current * ((1 + cagr/100) ** (2039 - datetime.now().year))
                proj_30 = current * ((1 + cagr/100) ** (2051 - datetime.now().year))
                proj_40 = current * ((1 + cagr/100) ** (2061 - datetime.now().year))
                
                st.info(f"**Taylor is approximately {taylor_age} years old.** At {cagr:.1f}% CAGR: Age 18 (2039): ~${proj_18:,.0f}, Age 30 (2051): ~${proj_30:,.0f}, Age 40 (2061): ~${proj_40:,.0f}")
            else:
                st.info("Add more historical data to see projections")
        else:
            st.info("No data for Taylor yet.")

    st.download_button(
        "Export All Monthly Data",
        df.to_csv(index=False).encode(),
        f"sage-data-{datetime.now().strftime('%Y-%m-%d')}.csv",
        "text/csv"
    )
