# pages/dashboard.py - Full dashboard with multi-portfolio support

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import base64

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
    if not df.empty and not df_net.empty:
        cur_total = df_net["value"].iloc[-1]
        pct, vs = peer_benchmark(cur_total)
        st.markdown(f"# ${cur_total:,.0f}")
        st.markdown(f"### vs. Avg 40yo: **Top {100-int(pct)}%** • Ahead by **${vs:+,}**")

        st.markdown("#### YTD Growth (Jan 1 → Today)")
        current_year = datetime.now().year
        ytd_df = df[df["date"].dt.year == current_year].copy()
        if len(ytd_df["date"].unique()) > 1:
            start_vals = ytd_df[ytd_df["date"] == ytd_df["date"].min()].groupby("person")["value"].sum()
            latest_vals = ytd_df[ytd_df["date"] == ytd_df["date"].max()].groupby("person")["value"].sum()
            ytd_pct = ((latest_vals / start_vals) - 1) * 100
            combined_ytd = ((latest_vals.get('Sean',0) + latest_vals.get('Kim',0)) / 
                            (start_vals.get('Sean',1) + start_vals.get('Kim',1)) - 1) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("**Sean YTD**", f"{ytd_pct.get('Sean', 0):+.1f}%")
            col2.metric("**Kim YTD**", f"{ytd_pct.get('Kim', 0):+.1f}%")
            col3.metric("**Taylor YTD**", f"{ytd_pct.get('Taylor', 0):+.1f}%")
            col4.metric("**Combined YTD**", f"{combined_ytd:+.1f}%")
        else:
            st.info("Not enough data for YTD yet this year.")

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

            # Load and merge all saved portfolios
            all_b64 = load_all_portfolios()
            portfolio_dfs = []
            for slot, b64_data in all_b64.items():
                try:
                    from io import StringIO
                    decoded = base64.b64decode(b64_data).decode('utf-8')
                    df_slot, _ = parse_portfolio_csv(StringIO(decoded))
                    if not df_slot.empty:
                        portfolio_dfs.append(df_slot)
                except Exception as e:
                    st.error(f"Error loading saved portfolio {slot}")

            if portfolio_dfs:
                df_port, port_summary = merge_portfolios(portfolio_dfs)
                st.success(f"**Combined Portfolio Ready**: ${port_summary['total_value']:,.0f} across {len(all_b64)} account(s)")
            else:
                df_port = pd.DataFrame()
                port_summary = {}

            st.caption("Always here for strategy, risks, opportunities, or just to chat.")
            if st.button("🧠 Talk to S.A.G.E.", use_container_width=True):
                st.session_state.page = "ai"
                st.rerun()

        st.markdown("---")
        st.subheader("Data Tools")
        
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
                    st.warning(f"{len(errors)} errors")
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
                    for _, r in df_std.iterrows():
                        add_monthly_update(r['date'], r['person'], r['account_type'], float(r['value']))
                    st.success(f"Imported {len(df_std)} rows!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Reset Database"):
            if st.checkbox("I understand this deletes all data", key="confirm_reset"):
                reset_database()
                st.success("Database reset!")
                st.rerun()

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
            add_monthly_update(date_in, person, acct, float(val))
            st.success("Saved!")
            st.rerun()

    # ------------------------------------------------------------------
    # Main Tabs - Charts & Tables
    # ------------------------------------------------------------------
    if df.empty:
        st.info("Add data to see the full dashboard.")
        return

    tab1, tab2 = st.tabs(["Retirement (Sean + Kim)", "💎 Taylor's Nest Egg"])

    with tab1:
        # Growth Chart
        df_pivot = df.pivot_table(index="date", columns="person", values="value", aggfunc="sum") \
                      .resample("ME").last().ffill().fillna(0)
        
        df_sean_kim = df_pivot[['Sean', 'Kim']].copy() if 'Sean' in df_pivot.columns and 'Kim' in df_pivot.columns else df_pivot
        df_sean_kim["Sean + Kim"] = df_sean_kim.get("Sean", 0) + df_sean_kim.get("Kim", 0)

        fig = go.Figure()
        colors = {"Sean": "#636EFA", "Kim": "#EF553B", "Sean + Kim": "#AB63FA"}
        widths = {"Sean + Kim": 5, "Sean": 3, "Kim": 3}

        for person in ["Sean", "Kim", "Sean + Kim"]:
            if person in df_sean_kim.columns:
                fig.add_trace(go.Scatter(
                    x=df_sean_kim.index,
                    y=df_sean_kim[person],
                    name=person,
                    line=dict(color=colors[person], width=widths.get(person, 3)),
                    hovertemplate=f"<b>{person}</b><br>%{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>"
                ))

        fig.update_layout(
            title="Retirement Portfolio Growth (Sean + Kim)",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📈 Month-over-Month Analysis")

        mom_dollar = df_sean_kim.diff().round(0)
        mom_pct = df_sean_kim.pct_change() * 100
        years = sorted(df_sean_kim.index.year.unique(), reverse=True)
        year_tabs = st.tabs([str(y) for y in years])

        for tab, year in zip(year_tabs, years):
            with tab:
                mask = df_sean_kim.index.year == year
                display_data = []
                for date in df_sean_kim[mask].index:
                    row = {'Date': date.strftime('%b %Y')}
                    for p in ['Sean', 'Kim', 'Sean + Kim']:
                        if p in df_sean_kim.columns:
                            row[f'{p} $'] = mom_dollar.loc[date, p] if date in mom_dollar.index else 0
                            row[f'{p} %'] = mom_pct.loc[date, p] if date in mom_pct.index else 0
                    display_data.append(row)
                df_display = pd.DataFrame(display_data)

                def color_val(val):
                    if isinstance(val, (int, float)):
                        return 'background-color: #90EE90; color: black' if val > 0 else ('background-color: #FF6B6B; color: black' if val < 0 else '')
                    return ''

                styled = df_display.style.map(color_val, subset=[c for c in df_display.columns if c != 'Date'])\
                    .format({c: '${:,.0f}' if '$' in c else '{:+.2f}%' for c in df_display.columns if c != 'Date'})
                st.dataframe(styled, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📅 Year-over-Year (December to December)")
        december_data = df_sean_kim[df_sean_kim.index.month == 12]
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
            growth = ((current / taylor_df["value"].iloc[0]) - 1) * 100
            years = (taylor_df['date'].iloc[-1] - taylor_df['date'].iloc[0]).days / 365.25
            cagr = ((current / taylor_df["value"].iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0

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
            fig_t.update_layout(title="Taylor's Portfolio Growth Over Time", height=500)
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎯 Long-Term Outlook for Taylor")
            taylor_age = datetime.now().year - 2021
            st.info(f"""
            **Taylor is approximately {taylor_age} years old.** Time is her superpower.
            
            At {cagr:.1f}% CAGR:
            - Age 18 (2039): ~${current * ((1 + cagr/100) ** (2039 - datetime.now().year)):,.0f}
            - Age 30 (2051): ~${current * ((1 + cagr/100) ** (2051 - datetime.now().year)):,.0f}
            - Age 40 (2061): ~${current * ((1 + cagr/100) ** (2061 - datetime.now().year)):,.0f}
            
            Let compounding work its magic. 🚀
            """)

            # Taylor MoM tables
            taylor_pivot = taylor_df.set_index('date')['value'].resample('ME').last().to_frame()
            taylor_mom_dollar = taylor_pivot.diff().round(0)
            taylor_mom_pct = taylor_pivot.pct_change() * 100
            taylor_years = sorted(taylor_pivot.index.year.unique(), reverse=True)
            taylor_tabs = st.tabs([str(y) for y in taylor_years])

            for tab, year in zip(taylor_tabs, taylor_years):
                with tab:
                    mask = taylor_pivot.index.year == year
                    display = []
                    for date in taylor_pivot[mask].index:
                        display.append({
                            'Date': date.strftime('%b %Y'),
                            'Change $': taylor_mom_dollar.loc[date, 'value'],
                            'Change %': taylor_mom_pct.loc[date, 'value']
                        })
                    df_disp = pd.DataFrame(display)
                    styled = df_disp.style.map(color_val, subset=['Change $', 'Change %'])\
                        .format({'Change $': '${:,.0f}', 'Change %': '{:+.2f}%'})
                    st.dataframe(styled, use_container_width=True)
        else:
            st.info("No data for Taylor yet.")

    st.download_button(
        "Export All Monthly Data",
        df.to_csv(index=False).encode(),
        f"sage-data-{datetime.now().strftime('%Y-%m-%d')}.csv",
        "text/csv"
    )
