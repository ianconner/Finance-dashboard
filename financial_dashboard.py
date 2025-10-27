```python
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is not installed. Please run 'pip install plotly' and restart the app.")
    st.stop()
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("Statsmodels is not installed. Please run 'pip install statsmodels' and restart the app.")
    st.stop()

# Debug mode
DEBUG = False

# Database connection
conn = sqlite3.connect('family_finance.db')
c = conn.cursor()

# Function to reset database
def reset_database():
    c.execute("DROP TABLE IF EXISTS monthly_updates")
    c.execute("DROP TABLE IF EXISTS accounts")
    c.execute("DROP TABLE IF EXISTS account_config")
    c.execute("DROP TABLE IF EXISTS contributions")
    c.execute('''CREATE TABLE accounts 
                 (person TEXT, account_type TEXT, initial_value REAL, start_date TEXT)''')
    c.execute('''CREATE TABLE monthly_updates 
                 (date TEXT, person TEXT, account_type TEXT, value REAL, 
                  PRIMARY KEY (date, person, account_type))''')
    c.execute('''CREATE TABLE account_config 
                 (person TEXT, account_type TEXT, PRIMARY KEY (person, account_type))''')
    c.execute('''CREATE TABLE contributions 
                 (date TEXT, person TEXT, account_type TEXT, contribution REAL, 
                  PRIMARY KEY (date, person, account_type))''')
    default_accounts = {
        'Sean': ['IRA', 'Roth IRA', 'TSP', 'Personal', 'T3W'],
        'Kim': ['Retirement'],
        'Taylor': ['Personal']
    }
    for person, acc_types in default_accounts.items():
        for acc_type in acc_types:
            c.execute("INSERT OR IGNORE INTO account_config (person, account_type) VALUES (?, ?)", (person, acc_type))
    conn.commit()

# Load accounts from database
def load_accounts():
    c.execute("SELECT person, account_type FROM account_config")
    rows = c.fetchall()
    accounts = {}
    for person, account_type in rows:
        if person not in accounts:
            accounts[person] = []
        accounts[person].append(account_type)
    if not accounts:
        reset_database()
        return load_accounts()
    return accounts

# Function to add new person
def add_person(person_name):
    c.execute("INSERT OR IGNORE INTO account_config (person, account_type) VALUES (?, ?)", (person_name, 'Default'))
    conn.commit()

# Function to add account type
def add_account_type(person, account_type):
    c.execute("INSERT OR IGNORE INTO account_config (person, account_type) VALUES (?, ?)", (person, account_type))
    conn.commit()

# Function to delete account type
def delete_account_type(person, account_type):
    c.execute("DELETE FROM account_config WHERE person=? AND account_type=?", (person, account_type))
    c.execute("DELETE FROM monthly_updates WHERE person=? AND account_type=?", (person, account_type))
    c.execute("DELETE FROM contributions WHERE person=? AND account_type=?", (person, account_type))
    conn.commit()

# Function to delete person
def delete_person(person):
    c.execute("DELETE FROM account_config WHERE person=?", (person,))
    c.execute("DELETE FROM monthly_updates WHERE person=?", (person,))
    c.execute("DELETE FROM contributions WHERE person=?", (person,))
    conn.commit()

# Function to add/update monthly values
def add_monthly_update(date, person, account_type, value):
    c.execute("INSERT OR REPLACE INTO monthly_updates (date, person, account_type, value) VALUES (?, ?, ?, ?)",
              (date, person, account_type, value))
    conn.commit()

# Function to add/update monthly contributions
def add_monthly_contribution(date, person, account_type, contribution):
    c.execute("INSERT OR REPLACE INTO contributions (date, person, account_type, contribution) VALUES (?, ?, ?, ?)",
              (date, person, account_type, contribution))
    conn.commit()

# Function to import CSV data with duplicate preview
def import_csv_data(file, accounts, table='monthly_updates'):
    try:
        df = pd.read_csv(file)
        expected_columns = ['date', 'person', 'account_type', 'value' if table == 'monthly_updates' else 'contribution']
        if not all(col in df.columns for col in expected_columns):
            return False, f"CSV must have columns: {', '.join(expected_columns)}"
        
        duplicates = []
        added, updated, skipped = 0, 0, 0
        for index, row in df.iterrows():
            if (pd.isna(row['date']) or 
                row['person'] not in accounts or 
                row['account_type'] not in accounts[row['person']] or 
                not isinstance(row[expected_columns[-1]], (int, float))):
                skipped += 1
                continue
            
            c.execute(f"SELECT {expected_columns[-1]} FROM {table} WHERE date=? AND person=? AND account_type=?", 
                      (str(row['date']), row['person'], row['account_type']))
            existing = c.fetchone()
            
            if existing:
                if abs(float(existing[0]) - float(row[expected_columns[-1]])) < 0.01:
                    duplicates.append(row.to_dict())
                    skipped += 1
                    continue
                else:
                    updated += 1
            else:
                added += 1
            
            if table == 'monthly_updates':
                add_monthly_update(str(row['date']), row['person'], row['account_type'], float(row['value']))
            else:
                add_monthly_contribution(str(row['date']), row['person'], row['account_type'], float(row['contribution']))
        
        message = f"CSV imported successfully to {table}! Added {added}, updated {updated}, skipped {skipped} records."
        if duplicates:
            st.warning("Potential duplicates detected:")
            st.dataframe(pd.DataFrame(duplicates))
            if st.button("Import duplicates anyway"):
                for row in duplicates:
                    if table == 'monthly_updates':
                        add_monthly_update(str(row['date']), row['person'], row['account_type'], float(row['value']))
                    else:
                        add_monthly_contribution(str(row['date']), row['person'], row['account_type'], float(row['contribution']))
                    added += 1
                message += f"\nImported {len(duplicates)} duplicates upon user confirmation."
        return True, message
    except Exception as e:
        return False, f"Error importing CSV: {str(e)}"

# Function to get all data as DataFrame
def get_data(table='monthly_updates'):
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])
    return df.sort_values('date')

# Function for inflation adjustment
def adjust_for_inflation(value, start_date, end_date, inflation_rate=0.025):
    years = (end_date.year - start_date.year) + (end_date.month - start_date.month) / 12
    return value / (1 + inflation_rate) ** years

# Function for baseline prediction
def baseline_prediction(current_value, monthly_contribution, start_date, end_date=datetime(2042, 12, 31), annual_rate=0.07):
    years = (end_date.year - start_date.year) + (end_date.month - start_date.month) / 12
    months = int(years * 12)
    monthly_rate = annual_rate / 12
    fv_current = current_value * (1 + monthly_rate) ** months
    fv_contributions = monthly_contribution * ((1 + monthly_rate) ** months - 1) / monthly_rate if monthly_contribution else 0
    return fv_current + fv_contributions

# Function for ML prediction using ARIMA
def ml_prediction(df_pivot, monthly_contribution, future_date=datetime(2042, 12, 31)):
    if df_pivot.empty or 'Sean_Kim' not in df_pivot.columns:
        return None, "No Sean + Kim data for ML prediction"
    
    data = df_pivot['Sean_Kim'].values
    if len(data) < 12:
        return None, "Insufficient data for ARIMA prediction"
    
    try:
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit()
        
        last_date = pd.to_datetime(df_pivot['month'].iloc[-1] + '-01')
        future_months = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)
        
        forecast = model_fit.forecast(steps=future_months)
        predicted_value = forecast[-1]
        
        if monthly_contribution:
            monthly_rate = 0.07 / 12
            fv_contributions = monthly_contribution * ((1 + monthly_rate) ** future_months - 1) / monthly_rate
            predicted_value += fv_contributions
        
        return predicted_value, None
    except Exception as e:
        return None, f"ARIMA prediction failed: {str(e)}"

# Function for Monte Carlo simulation
def monte_carlo_simulation(initial_value, years, expected_return=0.07, volatility=0.15, num_simulations=10000):
    monthly_return = expected_return / 12
    monthly_volatility = volatility / np.sqrt(12)
    simulations = []
    for _ in range(num_simulations):
        value = initial_value
        for _ in range(years * 12):
            monthly_rate = np.random.normal(monthly_return, monthly_volatility)
            value *= (1 + monthly_rate)
        simulations.append(value)
    return np.array(simulations)

# Streamlit app
st.set_page_config(layout="wide")
st.markdown("""
<style>
.main .block-container { max-width: 100%; padding: 1rem; }
@media (max-width: 768px) { .main .block-container { padding: 0.5rem; } }
</style>
""", unsafe_allow_html=True)
st.title("Family Finance Dashboard")

# Sidebar for data import and updates
st.sidebar.header("Manage Data")
accounts = load_accounts()

# Month-to-month update form
with st.sidebar.form("monthly_update_form"):
    st.subheader("Add Monthly Update")
    update_date = st.text_input("Date (YYYY-MM, e.g., 2025-11)", value=datetime.now().strftime("%Y-%m"))
    update_person = st.selectbox("Person", list(accounts.keys()))
    update_account_type = st.selectbox("Account Type", accounts[update_person])
    update_value = st.number_input("Value/Contribution ($)", min_value=0.0, step=100.0)
    update_table = st.selectbox("Table", ["monthly_updates (Values)", "contributions (Contributions)"])
    submit_button = st.form_submit_button("Add Update")
    if submit_button:
        if not re.match(r"^\d{4}-\d{2}$", update_date):
            st.sidebar.error("Date must be in YYYY-MM format (e.g., 2025-11)")
        else:
            try:
                pd.to_datetime(update_date + "-01", format="%Y-%m-%d")
                table = 'monthly_updates' if update_table.startswith('monthly_updates') else 'contributions'
                if table == 'monthly_updates':
                    add_monthly_update(update_date, update_person, update_account_type, update_value)
                    st.sidebar.success(f"Added {update_person} {update_account_type} value ${update_value:,.2f} for {update_date} to monthly_updates")
                else:
                    add_monthly_contribution(update_date, update_person, update_account_type, update_value)
                    st.sidebar.success(f"Added {update_person} {update_account_type} contribution ${update_value:,.2f} for {update_date} to contributions")
            except ValueError:
                st.sidebar.error("Invalid date. Use YYYY-MM (e.g., 2025-11)")

# Sidebar for forecast inputs
st.sidebar.header("Forecast Settings")
monthly_contribution = st.sidebar.number_input("Monthly Contribution (Sean + Kim, $)", min_value=0.0, value=0.0, step=100.0)
goal_amounts = st.sidebar.multiselect("Retirement Goals ($)", [1000000, 2000000, 3000000], default=[1000000])
annual_rate = st.sidebar.selectbox("S&P 500 Return Rate:", ["7% (Real)", "10% (Nominal)"], index=0)
annual_rate_value = 0.07 if annual_rate == "7% (Real)" else 0.10

# Import CSV forms
with st.sidebar.form("import_values_form"):
    st.subheader("Import Values")
    values_file = st.file_uploader("Upload CSV (Values)", type="csv", key="values")
    if st.form_submit_button("Import Values"):
        if values_file:
            success, message = import_csv_data(values_file, accounts, table='monthly_updates')
            st.sidebar.write(message)

with st.sidebar.form("import_contributions_form"):
    st.subheader("Import Contributions")
    contributions_file = st.file_uploader("Upload CSV (Contributions)", type="csv", key="contributions")
    if st.form_submit_button("Import Contributions"):
        if contributions_file:
            success, message = import_csv_data(contributions_file, accounts, table='contributions')
            st.sidebar.write(message)

# Reset database option
if st.sidebar.button("Reset Database"):
    reset_database()
    st.sidebar.success("Database reset successfully!")

# Refresh button
if st.sidebar.button("Refresh Dashboard"):
    st.cache_data.clear()
    st.experimental_rerun()

# Load and combine data
df_values = get_data('monthly_updates')
df_contrib = get_data('contributions')
df_values = df_values.rename(columns={'value': 'contribution'})
df = pd.concat([df_values, df_contrib], ignore_index=True)
if not df.empty:
    df['contribution'] = df['contribution'].astype(float)
    df = df.sort_values('date')

# Debug data
if DEBUG:
    st.write("Debug: DataFrame Head", df.head())
    st.write("Debug: DataFrame Columns", df.columns)

# Month-to-Month Breakdown Table (in tabs per year)
with st.expander("Month-to-Month Breakdown", expanded=True):
    if not df.empty:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.strftime('%Y-%m')
        df_pivot = df.pivot_table(index=['year', 'month'], columns='person', values='contribution', aggfunc='sum', fill_value=0).reset_index()
        df_pivot['Sean_Kim'] = df_pivot['Sean'] + df_pivot['Kim']
        
        df_pivot['Sean_%_prev'] = df_pivot['Sean'].pct_change() * 100
        df_pivot['Kim_%_prev'] = df_pivot['Kim'].pct_change() * 100
        df_pivot['Taylor_%_prev'] = df_pivot.get('Taylor', 0).pct_change() * 100
        df_pivot['Sean_Kim_%_prev'] = df_pivot['Sean_Kim'].pct_change() * 100
        
        df_dec_prev = df[df['month'].str.endswith('-12')].pivot_table(index='year', columns='person', values='contribution', aggfunc='sum', fill_value=0)
        df_dec_prev['Sean_Kim'] = df_dec_prev['Sean'] + df_dec_prev['Kim']
        df_pivot['Sean_%_ytd'] = 0.0
        df_pivot['Kim_%_ytd'] = 0.0
        df_pivot['Taylor_%_ytd'] = 0.0
        df_pivot['Sean_Kim_%_ytd'] = 0.0
        
        for i, row in df_pivot.iterrows():
            year = row['year']
            prev_year = year - 1
            if prev_year in df_dec_prev.index:
                prev_dec = df_dec_prev.loc[prev_year]
                for person in ['Sean', 'Kim', 'Taylor', 'Sean_Kim']:
                    if person in prev_dec and prev_dec[person] != 0:
                        df_pivot.at[i, f'{person}_%_ytd'] = (row[person] - prev_dec[person]) / prev_dec[person] * 100
        
        df_pivot = df_pivot[['year', 'month', 'Sean', 'Kim', 'Taylor', 'Sean_Kim', 
                             'Sean_%_prev', 'Kim_%_prev', 'Taylor_%_prev', 'Sean_Kim_%_prev',
                             'Sean_%_ytd', 'Kim_%_ytd', 'Taylor_%_ytd', 'Sean_Kim_%_ytd']]
        
        if DEBUG:
            st.write("Debug: df_pivot", df_pivot.tail())
        
        years = sorted(df_pivot['year'].unique())
        tabs = st.tabs([str(year) for year in years])
        for tab, year in zip(tabs, years):
            with tab:
                year_df = df_pivot[df_pivot['year'] == year].drop(columns=['year'])
                st.dataframe(year_df.style.format({
                    'Sean': '${:,.2f}', 'Kim': '${:,.2f}', 'Taylor': '${:,.2f}', 'Sean_Kim': '${:,.2f}',
                    'Sean_%_prev': '{:.1f}%', 'Kim_%_prev': '{:.1f}%', 'Taylor_%_prev': '{:.1f}%', 'Sean_Kim_%_prev': '{:.1f}%',
                    'Sean_%_ytd': '{:.1f}%', 'Kim_%_ytd': '{:.1f}%', 'Taylor_%_ytd': '{:.1f}%', 'Sean_Kim_%_ytd': '{:.1f}%'
                }))
    else:
        st.write("No data available. Please import a CSV file or add monthly updates.")

# Calculate monthly totals for Kim and Sean
current_date = datetime.now()
if not df.empty:
    df_total = df[df['person'].isin(['Kim', 'Sean'])].groupby('date')['contribution'].sum().reset_index()
    df_total = df_total.rename(columns={'contribution': 'monthly_total'})
    latest_month_total = df_total['monthly_total'].iloc[-1] if not df_total.empty else 0
    latest_sean_kim = df_pivot['Sean_Kim'].iloc[-1] if not df_pivot.empty else 0
else:
    df_total = pd.DataFrame()
    latest_month_total = 0
    latest_sean_kim = 0

# Debug totals
if DEBUG:
    st.write("Debug: Latest Sean + Kim", latest_sean_kim)
    st.write("Debug: Latest Month Total", latest_month_total)

# Retirement Forecast (Monthly Progress)
with st.expander("Retirement Forecast (Monthly Progress)", expanded=True):
    if not df.empty:
        df['year'] = df['date'].dt.year
        df_monthly = df[df['person'].isin(['Kim', 'Sean'])].groupby(['year', df['date'].dt.strftime('%Y-%m')])['contribution'].sum().reset_index()
        df_monthly = df_monthly.rename(columns={'contribution': 'monthly_total', 'date': 'month'})
        df_monthly['inflation_adjusted'] = [adjust_for_inflation(v, pd.to_datetime(m + '-01'), current_date) 
                                           for v, m in zip(df_monthly['monthly_total'], df_monthly['month'])]
    else:
        df_monthly = pd.DataFrame()
    
    future_years = list(range(current_date.year, 2043))
    future_months = (2042 - current_date.year) * 12 + (12 - current_date.month + 1)
    
    if not df_total.empty:
        baseline_pred = baseline_prediction(latest_sean_kim, monthly_contribution, current_date, datetime(2042, 12, 31), annual_rate=annual_rate_value)
        ml_pred, error = ml_prediction(df_pivot, monthly_contribution, datetime(2042, 12, 31))
        
        arima_data = df_pivot['Sean_Kim'].values
        if len(arima_data) >= 12:
            model = ARIMA(arima_data, order=(1, 1, 1))
            model_fit = model.fit()
            arima_forecast = model_fit.forecast(steps=future_months)
            future_dates = [pd.to_datetime(df_pivot['month'].iloc[-1] + '-01') + relativedelta(months=i+1) for i in range(future_months)]
            ml_monthly = arima_forecast + np.array([monthly_contribution * (1 + annual_rate_value / 12) ** i for i in range(future_months)])
            ml_monthly_inflated = [adjust_for_inflation(v, d, current_date) for v, d in zip(ml_monthly, future_dates)]
        else:
            ml_monthly = []
            ml_monthly_inflated = []
        
        baseline_monthly = [latest_sean_kim * (1 + annual_rate_value / 12) ** i + monthly_contribution * ((1 + annual_rate_value / 12) ** i - 1) / (annual_rate_value / 12) if monthly_contribution else latest_sean_kim * (1 + annual_rate_value / 12) ** i for i in range(1, future_months + 1)]
        baseline_monthly_inflated = [adjust_for_inflation(v, d, current_date) for v, d in zip(baseline_monthly, future_dates)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pivot['month'], y=df_pivot['Sean_Kim'], mode='lines+markers', name='Historical Sean + Kim'))
        fig.add_trace(go.Scatter(x=df_pivot['month'], y=[adjust_for_inflation(v, pd.to_datetime(m + '-01'), current_date) for v, m in zip(df_pivot['Sean_Kim'], df_pivot['month'])], mode='lines+markers', name='Historical Sean + Kim (Inflation-Adjusted)'))
        if ml_monthly.size > 0:
            fig.add_trace(go.Scatter(x=[d.strftime('%Y-%m') for d in future_dates], y=ml_monthly, mode='lines+markers', name='ML Prediction (ARIMA)', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=[d.strftime('%Y-%m') for d in future_dates], y=ml_monthly_inflated, mode='lines+markers', name='ML Prediction (Inflation-Adjusted)', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=[d.strftime('%Y-%m') for d in future_dates], y=baseline_monthly, mode='lines+markers', name=f'Baseline {annual_rate}', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=[d.strftime('%Y-%m') for d in future_dates], y=baseline_monthly_inflated, mode='lines+markers', name=f'Baseline {annual_rate} (Inflation-Adjusted)', line=dict(dash='dash')))
        fig.update_layout(xaxis_title='Month', yaxis_title='Balance ($)', xaxis_tickangle=-45, hovermode='x unified', dragmode='zoom')
        st.plotly_chart(fig, use_container_width=True)
        
        # Export forecast data
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m') for d in future_dates],
            'Baseline': baseline_monthly,
            'Baseline (Inflation-Adjusted)': baseline_monthly_inflated,
            'ML (ARIMA)': ml_monthly if ml_monthly.size > 0 else [None] * len(future_dates),
            'ML (Inflation-Adjusted)': ml_monthly_inflated if ml_monthly.size > 0 else [None] * len(future_dates)
        })
        st.download_button("Export Forecast Data as CSV", forecast_df.to_csv(index=False), "forecast_data.csv")
        
        if DEBUG:
            st.write("Debug: Baseline Forecast", baseline_pred)
            st.write("Debug: ML Forecast", ml_pred)
            st.write("Debug: Future Months", future_months)

# Goal Progress Tracker
with st.expander("Retirement Goal Progress"):
    if latest_sean_kim > 0:
        progress = min(latest_sean_kim / goal_amounts[0], 1.0) if goal_amounts else 0
        goal_data = {
            'Metric': ['Current (Latest Month)', 'Baseline Forecast (2042)', 'ML Forecast (2042)'],
            'Value': [latest_sean_kim, baseline_pred, ml_pred if ml_pred else 'N/A'],
            **{f'Progress (%) - Goal ${g/1000000:.1f}M': [latest_sean_kim/g*100, baseline_pred/g*100, (ml_pred/g*100) if ml_pred else 'N/A'] for g in goal_amounts}
        }
        def format_value(x):
            return '${:,.2f}'.format(x) if isinstance(x, (int, float)) else x
        def format_progress(x):
            return '{:.1f}%'.format(x) if isinstance(x, (int, float)) else x
        def style_progress(val):
            if isinstance(val, (int, float)):
                color = 'green' if val > 100 else 'yellow' if val > 50 else 'red'
                return f'color: {color}'
            return ''
        
        st.progress(progress)
        st.write(f"Current Progress (Goal ${goal_amounts[0]/1000000:.1f}M): {progress*100:.1f}% (${latest_sean_kim:,.2f} of ${goal_amounts[0]:,.2f})")
        st.table(pd.DataFrame(goal_data).style.format({
            'Value': format_value,
            **{f'Progress (%) - Goal ${g/1000000:.1f}M': format_progress for g in goal_amounts}
        }).applymap(style_progress, subset=[f'Progress (%) - Goal ${g/1000000:.1f}M' for g in goal_amounts]))
    else:
        st.write("No Sean + Kim data available for goal progress.")

# Portfolio Performance
with st.expander("Portfolio Performance"):
    if len(df_total) > 1:
        monthly_returns = df_total['monthly_total'].pct_change().dropna()
        annualized_return = ((1 + monthly_returns.mean()) ** 12 - 1) * 100
        annualized_volatility = monthly_returns.std() * np.sqrt(12) * 100
        st.write(f"Annualized Return (Kim and Sean, Monthly): {annualized_return:.2f}%")
        st.write(f"Annualized Volatility: {annualized_volatility:.2f}%")
        
        years_to_2042 = 2042 - current_date.year
        simulations = monte_carlo_simulation(latest_sean_kim, years_to_2042, annual_rate_value, 0.15, 10000)
        percentiles = np.percentile(simulations, [25, 50, 75])
        st.write(f"Monte Carlo 2042 Outcomes (10,000 simulations, based on latest Sean + Kim):")
        st.write(f"25th Percentile: ${percentiles[0]:,.2f}")
        st.write(f"Median: ${percentiles[1]:,.2f}")
        st.write(f"75th Percentile: ${percentiles[2]:,.2f}")
        
        fig = px.histogram(simulations, nbins=50, title='Monte Carlo Outcomes Distribution (based on latest Sean + Kim)')
        fig.add_vline(x=percentiles[0], line_dash="dash", line_color="red", annotation_text="25th")
        fig.add_vline(x=percentiles[1], line_dash="dash", line_color="green", annotation_text="Median")
        fig.add_vline(x=percentiles[2], line_dash="dash", line_color="blue", annotation_text="75th")
        fig.update_layout(hovermode='x unified', dragmode='zoom')
        st.plotly_chart(fig, use_container_width=True)

# Benchmark Comparison
with st.expander(f"Performance vs. S&P 500 Benchmark ({annual_rate})"):
    if len(df_total) > 1:
        actual_growth = (df_total['monthly_total'].iloc[-1] / df_total['monthly_total'].iloc[0] - 1) * 100
        months_elapsed = len(df_total) - 1
        benchmark_growth = ((1 + annual_rate_value / 12) ** months_elapsed - 1) * 100
        st.write(f"Kim and Sean annualized monthly growth: {actual_growth / (months_elapsed / 12):.2f}%")
        st.write(f"S&P Benchmark: {annual_rate_value*100:.2f}%")
        if actual_growth > benchmark_growth:
            st.success("Beating the market!")
        else:
            st.warning("Underperforming the market.")

# Year-to-Year Progress
with st.expander("Year-to-Year Progress"):
    if not df.empty:
        df_yearly = df_pivot.groupby('year')['Sean_Kim'].last().reset_index()
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['Sean_Kim'], mode='lines+markers', name='Sean + Kim'))
        fig_yearly.update_layout(xaxis_title='Year', yaxis_title='End-of-Year Balance ($)', xaxis_tickangle=-45, hovermode='x unified', dragmode='zoom')
        st.plotly_chart(fig_yearly, use_container_width=True)

# Export Data
if not df.empty:
    st.download_button("Export Values as CSV", df.to_csv(index=False), "family_finance_values.csv")
    if not df_contrib.empty:
        st.download_button("Export Contributions as CSV", df_contrib.to_csv(index=False), "family_finance_contributions.csv")

conn.close()
```
