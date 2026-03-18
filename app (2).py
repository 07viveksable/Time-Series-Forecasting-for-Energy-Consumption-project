import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Time Series Forecasting — Energy Consumption")
st.markdown("Upload your **AEP_hourly.csv** and explore SARIMA & Prophet forecasts.")

# ─── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload AEP_hourly.csv", type="csv")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
model_choice  = st.sidebar.selectbox("Model", ["Prophet", "SARIMA", "Both"])
train_split   = st.sidebar.slider("Train size (%)", 70, 95, 80)

# ─── Load Data ───────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df_daily = df.resample('D').mean().asfreq('D')
    df_daily = df_daily.interpolate(method='linear')
    return df_daily

if uploaded_file:
    df_daily = load_data(uploaded_file)
else:
    st.info("👈 Upload your CSV from the sidebar to get started.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📉 Decomposition", "🤖 Forecast", "📋 Metrics"])

# ── Tab 1: EDA ───────────────────────────────────────────────
with tab1:
    st.subheader("Raw Daily Energy Consumption")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['AEP_MW'],
        mode='lines', name='AEP_MW',
        line=dict(color='steelblue', width=1)
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="MW", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df_daily):,}")
    col2.metric("Date Range", f"{df_daily.index.min().date()} → {df_daily.index.max().date()}")
    col3.metric("Avg Consumption", f"{df_daily['AEP_MW'].mean():,.0f} MW")

    st.subheader("Monthly Average Consumption")
    df_daily['Month'] = df_daily.index.month
    monthly_avg = df_daily.groupby('Month')['AEP_MW'].mean()
    fig2, ax = plt.subplots(figsize=(10, 3))
    monthly_avg.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg MW")
    ax.set_title("Average Monthly Consumption")
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=0)
    st.pyplot(fig2)

# ── Tab 2: Decomposition ─────────────────────────────────────
with tab2:
    st.subheader("Seasonal Decomposition")
    with st.spinner("Decomposing series..."):
        decomp = seasonal_decompose(df_daily['AEP_MW'], model='additive', period=365)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        decomp.observed.plot(ax=axes[0], color='#1D5E8A'); axes[0].set_ylabel('Observed')
        decomp.trend.plot(ax=axes[1],    color='#1D7A55'); axes[1].set_ylabel('Trend')
        decomp.seasonal.plot(ax=axes[2], color='#E09B1D'); axes[2].set_ylabel('Seasonal')
        decomp.resid.plot(ax=axes[3],    color='#C04828', alpha=0.7); axes[3].set_ylabel('Residual')
        for ax in axes:
            ax.set_xlabel('')
        plt.suptitle('Seasonal Decomposition of Energy Consumption', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Stationarity Test (ADF)")
    result = adfuller(df_daily['AEP_MW'].dropna())
    p_val = result[1]
    st.write(f"**Test Statistic:** {result[0]:.4f}")
    st.write(f"**p-value:** {p_val:.4f}")
    if p_val < 0.05:
        st.success("✅ Series is STATIONARY — good to use directly!")
    else:
        st.warning("⚠️ Series is NON-STATIONARY — differencing recommended.")

# ── Tab 3: Forecast ──────────────────────────────────────────
with tab3:
    split_idx = int(len(df_daily) * train_split / 100)
    train = df_daily.iloc[:split_idx]
    test  = df_daily.iloc[split_idx:]

    st.info(f"Train: {len(train):,} days | Test: {len(test):,} days | Forecast: {forecast_days} days ahead")

    results = {}

    # Prophet
    if model_choice in ["Prophet", "Both"]:
        st.subheader("🔮 Prophet Forecast")
        with st.spinner("Training Prophet model..."):
            prophet_df = train.reset_index().rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(prophet_df)

            future   = m.make_future_dataframe(periods=forecast_days + len(test))
            forecast = m.predict(future)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=train.index, y=train['AEP_MW'],
                                       name='Train', line=dict(color='royalblue')))
            fig_p.add_trace(go.Scatter(x=test.index, y=test['AEP_MW'],
                                       name='Actual Test', line=dict(color='green')))
            fig_p.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                       name='Forecast', line=dict(color='orange')))
            fig_p.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(255,165,0,0.15)',
                line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'
            ))
            fig_p.update_layout(height=420, xaxis_title="Date", yaxis_title="MW")
            st.plotly_chart(fig_p, use_container_width=True)

            test_forecast = forecast[forecast['ds'].isin(test.index)]
            if len(test_forecast) > 0:
                rmse = np.sqrt(mean_squared_error(test['AEP_MW'], test_forecast['yhat']))
                mae  = mean_absolute_error(test['AEP_MW'], test_forecast['yhat'])
                mape = np.mean(np.abs((test['AEP_MW'].values - test_forecast['yhat'].values)
                                      / test['AEP_MW'].values)) * 100
                results['Prophet'] = {'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'MAPE (%)': round(mape, 2)}

    # SARIMA
    if model_choice in ["SARIMA", "Both"]:
        st.subheader("📈 SARIMA Forecast")
        with st.spinner("Training SARIMA model (may take ~1 min)..."):
            model     = SARIMAX(train['AEP_MW'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            sarima_fit = model.fit(disp=False)
            steps      = len(test) + forecast_days
            pred       = sarima_fit.forecast(steps=steps)
            pred_index = pd.date_range(start=test.index[0], periods=steps, freq='D')
            pred.index = pred_index

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=train.index[-180:], y=train['AEP_MW'][-180:],
                                       name='Train (last 6 mo)', line=dict(color='royalblue')))
            fig_s.add_trace(go.Scatter(x=test.index, y=test['AEP_MW'],
                                       name='Actual', line=dict(color='green')))
            fig_s.add_trace(go.Scatter(x=pred.index, y=pred.values,
                                       name='SARIMA', line=dict(color='red')))
            fig_s.update_layout(height=420, xaxis_title="Date", yaxis_title="MW")
            st.plotly_chart(fig_s, use_container_width=True)

            rmse = np.sqrt(mean_squared_error(test['AEP_MW'], pred[:len(test)]))
            mae  = mean_absolute_error(test['AEP_MW'], pred[:len(test)])
            mape = np.mean(np.abs((test['AEP_MW'].values - pred[:len(test)].values)
                                  / test['AEP_MW'].values)) * 100
            results['SARIMA'] = {'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'MAPE (%)': round(mape, 2)}

    st.session_state['results'] = results

# ── Tab 4: Metrics ───────────────────────────────────────────
with tab4:
    st.subheader("📋 Model Comparison")
    results = st.session_state.get('results', {})
    if results:
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_r, ax = plt.subplots()
            results_df['RMSE'].plot(kind='bar', ax=ax, color=['orange','steelblue'][:len(results_df)], edgecolor='white')
            ax.set_title("RMSE Comparison"); ax.set_ylabel("RMSE"); ax.set_xlabel("")
            plt.xticks(rotation=0); st.pyplot(fig_r)
        with col2:
            fig_m, ax = plt.subplots()
            results_df['MAPE (%)'].plot(kind='bar', ax=ax, color=['green','red'][:len(results_df)], edgecolor='white')
            ax.set_title("MAPE Comparison"); ax.set_ylabel("MAPE (%)"); ax.set_xlabel("")
            plt.xticks(rotation=0); st.pyplot(fig_m)

        best = results_df['RMSE'].idxmin()
        st.success(f"🏆 Best model by RMSE: **{best}** with RMSE = {results_df.loc[best,'RMSE']:,.2f}")
    else:
        st.info("Run a forecast in the **🤖 Forecast** tab first, then come back here.")
