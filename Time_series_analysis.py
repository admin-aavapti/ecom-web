# # ==========================
# # Time Series Analysis App
# # ==========================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from prophet import Prophet
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import re

# # --------------------------
# # Streamlit Page Config
# # --------------------------
# st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# st.title("📈 Time Series Forecasting Dashboard")
# st.write("Upload your dataset, choose target column, and compare **Prophet** vs **LSTM** models.")

# # --------------------------
# # File Upload
# # --------------------------
# uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

# @st.cache_data
# def load_timeseries_from_file(f):
#     # Try reading with proper handling for ₹, commas etc.
#     df = pd.read_csv(f, low_memory=False)

#     # Normalize column names
#     df.columns = [c.strip().lower() for c in df.columns]

#     # Identify date column
#     date_col = None
#     for col in df.columns:
#         if "date" in col:
#             date_col = col
#             break

#     if date_col is None:
#         st.error("No column containing 'date' found in the dataset.")
#         st.stop()

#     # Convert to datetime
#     df["ds"] = pd.to_datetime(df[date_col], errors="coerce")
#     df = df.dropna(subset=["ds"])
#     df = df.sort_values("ds")

#     return df

# if uploaded:
#     df = load_timeseries_from_file(uploaded)
#     st.write("✅ Data Loaded Successfully")
#     st.write(df.head())

#     # --------------------------
#     # Column Selection
#     # --------------------------
#     target_col = st.selectbox("Select Target Column for Forecasting", [c for c in df.columns if c not in ["ds"]])

#     # Clean target column (remove ₹, commas, convert to float)
#     df["y"] = (
#         df[target_col]
#         .astype(str)
#         .apply(lambda x: re.sub(r"[^\d\.\-]", "", x))  # keep only numbers, dot, minus
#     )

#     df["y"] = pd.to_numeric(df["y"], errors="coerce")
#     df = df.dropna(subset=["y"])

#     # --------------------------
#     # Train/Test Split
#     # --------------------------
#     train_size = int(len(df) * 0.8)
#     train = df.iloc[:train_size]
#     test = df.iloc[train_size:]

#     # --------------------------
#     # Prophet Model
#     # --------------------------
#     st.subheader("🔮 Prophet Forecast")

#     prophet = Prophet(daily_seasonality=True)
#     prophet.fit(train[["ds", "y"]])

#     future = prophet.make_future_dataframe(periods=len(test))
#     forecast_prophet = prophet.predict(future)

#     # Evaluate Prophet
#     y_true = test["y"].values
#     y_pred_prophet = forecast_prophet.iloc[-len(test):]["yhat"].values

#     mse_prophet = mean_squared_error(y_true, y_pred_prophet)
#     mae_prophet = mean_absolute_error(y_true, y_pred_prophet)

#     st.write(f"**Prophet MSE:** {mse_prophet:.2f}, **MAE:** {mae_prophet:.2f}")

#     fig1 = prophet.plot(forecast_prophet)
#     st.pyplot(fig1)

#     # --------------------------
#     # LSTM Model
#     # --------------------------
#     st.subheader("🤖 LSTM Forecast")

#     # Scaling
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(df[["y"]].values)

#     seq_len = 10
#     X, y = [], []
#     for i in range(len(scaled) - seq_len):
#         X.append(scaled[i:i+seq_len])
#         y.append(scaled[i+seq_len])
#     X, y = np.array(X), np.array(y)

#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     # LSTM Model
#     model = Sequential([
#         LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
#         Dropout(0.2),
#         LSTM(32),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mse")
#     model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

#     y_pred_lstm = model.predict(X_test)
#     y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
#     y_test_rescaled = scaler.inverse_transform(y_test)

#     mse_lstm = mean_squared_error(y_test_rescaled, y_pred_lstm)
#     mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_lstm)

#     st.write(f"**LSTM MSE:** {mse_lstm:.2f}, **MAE:** {mae_lstm:.2f}")

#     forecast_lstm = pd.DataFrame({
#         "ds": df["ds"].iloc[-len(y_test):].values,
#         "yhat": y_pred_lstm.flatten()
#     })

#     fig2, ax2 = plt.subplots()
#     ax2.plot(df["ds"], df["y"], label="Actual", color="blue")
#     ax2.plot(forecast_lstm["ds"], forecast_lstm["yhat"], label="LSTM Forecast", color="green")
#     ax2.legend()
#     st.pyplot(fig2)

#     # --------------------------
#     # Combined Overlay Plot
#     # --------------------------
#     st.subheader(f"📊 Combined Forecast Comparison for {target_col}")

#     fig3, ax3 = plt.subplots()
#     ax3.plot(df["ds"], df["y"], label="Actual", color="blue")
#     ax3.plot(forecast_prophet["ds"], forecast_prophet["yhat"], label="Prophet Forecast", color="orange")
#     ax3.plot(forecast_lstm["ds"], forecast_lstm["yhat"], label="LSTM Forecast", color="green")
#     ax3.legend()
#     st.pyplot(fig3)

#     # --------------------------
#     # Metrics Comparison Table
#     # --------------------------
#     st.subheader("📑 Model Performance Metrics")

#     metrics_df = pd.DataFrame({
#         "Model": ["Prophet", "LSTM"],
#         "MSE": [mse_prophet, mse_lstm],
#         "MAE": [mae_prophet, mae_lstm]
#     })

#     st.table(metrics_df)


# time_series_forecaster.py
# Streamlit app: Prophet + LSTM hybrid forecaster for your amazon_timeseries_fixed.csv
# Requirements: pandas, numpy, streamlit, plotly, prophet, scikit-learn, tensorflow, statsmodels

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta
import re
import os

# --------------------------
# Helpers
# --------------------------
st.set_page_config(page_title="Brand & Category Review Forecaster", layout="wide")
st.title("Brand & Category Product Review Forecaster")

@st.cache_data
def load_data_default(path="/mnt/data/amazon_timeseries_fixed.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        return df, path
    return None, None

def parse_dates(df):
    # Find likely date column
    date_col = None
    for c in df.columns:
        if re.search(r"^(date|ds|review_date|review_date_enriched)$", c, re.I):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                date_col = c
                break

    if date_col:
        # try formats robustly
        df['__date_parsed'] = pd.to_datetime(df[date_col].astype(str), errors='coerce', dayfirst=True)
        # if too many NaT, try dayfirst=False
        if df['__date_parsed'].isna().mean() > 0.4:
            df['__date_parsed'] = pd.to_datetime(df[date_col].astype(str), errors='coerce', dayfirst=False)
    else:
        df['__date_parsed'] = pd.NaT

    # Fill missing dates with random dates within realistic range if any remain
    mask = df['__date_parsed'].isna()
    if mask.any():
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2025-09-30")
        n = mask.sum()
        rand_seconds = np.random.randint(start.value // 10**9, end.value // 10**9 + 1, size=n)
        df.loc[mask, '__date_parsed'] = pd.to_datetime(rand_seconds, unit='s')

    # final standardized date column
    df['ds'] = pd.to_datetime(df['__date_parsed'])
    # add date parts
    df['review_month'] = df['ds'].dt.month_name()
    df['review_dayname'] = df['ds'].dt.day_name()
    df['review_year'] = df['ds'].dt.year
    return df

def clean_currency_col(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.replace('₹', '').replace('Rs.', '').replace(',', '').strip()
    s = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(s)
    except:
        return np.nan

def extract_brand(product_name):
    if pd.isna(product_name):
        return None
    # common heuristics: brand often first token or before '-' or '|'
    pn = str(product_name).strip()
    # try patterns "Brand ..." or "Brand - ..." or "Brand | ..."
    m = re.match(r"^([A-Za-z0-9&\-\.\']+)", pn)
    if m:
        brand = m.group(1)
        # if brand all caps or long token, keep it; else fallback to first token
        return brand
    return pn.split()[0]

def mase(y_true, y_pred, train_series):
    # Mean Absolute Scaled Error
    # scale = mean(|y_t - y_{t-1}|) for training series
    n = len(train_series)
    if n < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(train_series)))
    if scale == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# --------------------------
# UI: load file or default
# --------------------------
default_df, default_path = load_data_default()
uploaded = st.file_uploader("Upload your time-series CSV (or leave to use existing file on server)", type=["csv"])
if uploaded is None and default_df is None:
    st.warning("No data found at /mnt/data/amazon_timeseries_fixed.csv. Please upload your CSV.")
    st.stop()

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, low_memory=False)
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    df = default_df
    st.info(f"Loaded dataset from `{default_path}` (change by uploading a file).")

# --------------------------
# Preprocessing
# --------------------------
with st.expander("Preview & Preprocessing (click to expand)"):
    st.write("Raw preview (top 5 rows):")
    st.dataframe(df.head())

# Parse date
df = parse_dates(df)

# Standardize/clean price columns if present
price_cols = [c for c in df.columns if re.search(r"discounted_price|discount_price|price", c, re.I)]
for pc in price_cols:
    newc = pc + "_num"
    df[newc] = df[pc].apply(clean_currency_col)

# rating_count: find a reasonable column
rating_count_col = None
for c in df.columns:
    if re.search(r"rating[_\s-]*count|ratingcount|rating_cnt|ratingcnt|rating_count_enriched|review_count", c, re.I):
        rating_count_col = c
        break

if rating_count_col:
    df['rating_count'] = pd.to_numeric(df[rating_count_col].astype(str).str.replace(r"[^\d]", "", regex=True), errors='coerce')
else:
    # fallback: try 'rating_count' present or create from review_count
    if 'rating_count' not in df.columns:
        df['rating_count'] = pd.to_numeric(df.get('rating_count', pd.Series([np.nan]*len(df))), errors='coerce')

# ensure product_name, category columns exist
if 'product_name' not in df.columns and 'product_title' in df.columns:
    df.rename(columns={'product_title':'product_name'}, inplace=True)

if 'category' not in df.columns and 'product_category' in df.columns:
    df.rename(columns={'product_category':'category'}, inplace=True)

# brand extraction
if 'brand' not in df.columns:
    df['brand'] = df['product_name'].apply(extract_brand)

# marketplace detection
if 'market_place' not in df.columns:
    # try to find something similar
    marketplace_candidates = [c for c in df.columns if re.search(r"market", c, re.I)]
    if marketplace_candidates:
        df.rename(columns={marketplace_candidates[0]:'market_place'}, inplace=True)
    else:
        df['market_place'] = np.random.choice(['US','IN','UK'], size=len(df))

# create a per-row review_count field if missing: one row = one review -> count =1
if 'review_count' not in df.columns:
    df['review_count'] = 1

# show enriched preview
with st.expander("Enriched preview (ds, product_name, brand, category, rating_count)"):
    cols_show = [c for c in ['ds','product_name','brand','category','market_place','rating_count','review_count'] if c in df.columns]
    st.dataframe(df[cols_show].head(10))

# --------------------------
# Sidebar filters & controls
# --------------------------
st.sidebar.header("Filters & Model Settings")
marketplaces = df['market_place'].dropna().unique().tolist()
market_select = st.sidebar.multiselect("Marketplace", options=marketplaces, default=marketplaces)

categories = df['category'].dropna().astype(str).unique().tolist() if 'category' in df.columns else []
if categories:
    category_select = st.sidebar.selectbox("Category (optional)", options=["All"] + categories, index=0)
else:
    category_select = "All"

mode = st.sidebar.radio("Mode", ["Single Brand", "All Brands in Category"])
granularity = st.sidebar.radio("Granularity", ["Daily", "Weekly"])
model_choice = st.sidebar.selectbox("Model", ["Prophet", "Hybrid (Prophet + LSTM)"])
forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)
seq_len = st.sidebar.slider("LSTM sequence length (if hybrid)", min_value=3, max_value=60, value=30, step=1)
epochs = st.sidebar.slider("LSTM epochs", min_value=1, max_value=200, value=20, step=1)

selected_brand = None
if mode == "Single Brand":
    brands = df[df['market_place'].isin(market_select)]
    if category_select != "All":
        brands = brands[brands['category'].astype(str) == category_select]
    brands = brands['brand'].dropna().astype(str).unique().tolist()
    if not brands:
        st.sidebar.warning("No brands found for your filters.")
        brands = [""]
    selected_brand = st.sidebar.selectbox("Brand", options=brands)

# --------------------------
# Series builder
# --------------------------
def build_timeseries(df_input, granularity="Daily"):
    df_ts = df_input.copy()
    df_ts = df_ts.groupby('ds')['review_count'].sum().reset_index()
    df_ts = df_ts.sort_values('ds')
    if granularity == "Weekly":
        df_ts = df_ts.set_index('ds').resample('W')['review_count'].sum().reset_index()
    # fill gaps
    idx = pd.date_range(start=df_ts['ds'].min(), end=df_ts['ds'].max(), freq='D' if granularity=='Daily' else 'W')
    df_ts = df_ts.set_index('ds').reindex(idx).rename_axis('ds').fillna(0).reset_index()
    return df_ts

# Apply filters to data
df_filtered = df[df['market_place'].isin(market_select)]
if category_select != "All":
    df_filtered = df_filtered[df_filtered['category'].astype(str) == category_select]
if mode == "Single Brand" and selected_brand:
    df_filtered = df_filtered[df_filtered['brand'].astype(str) == selected_brand]

if df_filtered.empty:
    st.warning("No data after applying filters. Adjust filters or upload different file.")
    st.stop()

# Build aggregated time series
ts = build_timeseries(df_filtered, granularity=granularity)

# --------------------------
# Forecast functions
# --------------------------
def train_prophet(ts_df, periods, freq):
    m = Prophet()
    m.fit(ts_df.rename(columns={'ds':'ds','review_count':'y'}))
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast

def train_lstm_forecast(ts_df, seq_len, periods, epochs=20):
    # ts_df: columns ds, review_count (daily or weekly)
    vals = ts_df['review_count'].values.astype(float).reshape(-1,1)
    scaler = MinMaxScaler()
    vals_scaled = scaler.fit_transform(vals)
    # prepare sequences
    X, y = [], []
    for i in range(seq_len, len(vals_scaled)):
        X.append(vals_scaled[i-seq_len:i])
        y.append(vals_scaled[i])
    X = np.array(X); y = np.array(y)
    if len(X) == 0:
        return None, None, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1],1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    # recursive forecast
    input_seq = vals_scaled[-seq_len:].reshape(seq_len,1)
    preds_scaled = []
    for _ in range(periods):
        x = input_seq.reshape(1, seq_len, 1)
        p = model.predict(x, verbose=0)[0][0]
        preds_scaled.append(p)
        # append and slide
        input_seq = np.vstack([input_seq[1:], [[p]]])
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return model, preds, scaler, (X, y)

# --------------------------
# Run forecasting
# --------------------------
st.header("Time Series & Forecast")
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Historical time series")
    fig_hist = px.line(ts, x='ds', y='review_count', labels={'ds':'Date','review_count':'Reviews'}, title="Historical review counts")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # KPIs
    total_reviews = int(ts['review_count'].sum())
    recent_mean = float(ts.tail(30)['review_count'].mean()) if len(ts)>=30 else float(ts['review_count'].mean())
    st.metric("Total reviews (filtered)", f"{total_reviews:,}")
    st.metric("Recent daily avg", f"{recent_mean:.2f}")

# Prepare Prophet input (rename)
prophet_df = ts.rename(columns={'ds':'ds','review_count':'y'})

freq = 'D' if granularity == "Daily" else 'W'
try:
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    prophet_future = prophet_model.make_future_dataframe(periods=forecast_days, freq=freq)
    prophet_forecast = prophet_model.predict(prophet_future)
except Exception as e:
    st.error(f"Prophet training failed: {e}")
    st.stop()

# LSTM only if requested by hybrid
lstm_preds = None
lstm_model_obj = None
lstm_eval = None
if model_choice.startswith("Hybrid"):
    if len(prophet_df) < seq_len + 2:
        st.warning("Not enough historical points for hybrid LSTM training given seq_len. Hybrid disabled.")
    else:
        lstm_model_obj, lstm_preds, scaler_obj, (X_train, y_train) = train_lstm_forecast(prophet_df.rename(columns={'y':'review_count'}).rename(columns={'review_count':'review_count'}).rename(columns={'y':'review_count'}).assign(review_count=prophet_df['y']), seq_len=seq_len, periods=forecast_days, epochs=epochs)
        # compute in-sample performance for LSTM if available
        if lstm_model_obj is not None:
            try:
                preds_inscaled = lstm_model_obj.predict(X_train, verbose=0)
                y_true_inv = scaler_obj.inverse_transform(y_train.reshape(-1,1)).flatten()
                y_pred_inv = scaler_obj.inverse_transform(preds_inscaled).flatten()
                mse = mean_squared_error(y_true_inv, y_pred_inv)
                mae = mean_absolute_error(y_true_inv, y_pred_inv)
                lstm_eval = (mse, mae)
            except Exception:
                lstm_eval = None

# Build forecast dataframes to plot
prophet_plot_df = prophet_forecast[['ds','yhat','yhat_lower','yhat_upper']].copy()
# hybrid: if LSTM preds present, average Prophet future tail with LSTM preds
hybrid_df = None
if lstm_preds is not None:
    future_dates = pd.date_range(start=ts['ds'].max() + (timedelta(days=1) if freq=='D' else timedelta(weeks=1)), periods=forecast_days, freq=freq)
    # get Prophet's future tail (only future portion)
    pfuture = prophet_plot_df[prophet_plot_df['ds'] > ts['ds'].max()].copy().reset_index(drop=True)
    # If Prophet has a different number of future rows (freq mismatch), align by length
    n = min(len(pfuture), len(lstm_preds))
    pfuture = pfuture.head(n)
    hybrid_vals = (pfuture['yhat'].values[:n] + np.array(lstm_preds[:n])) / 2.0
    hybrid_df = pd.DataFrame({'ds': pfuture['ds'].values[:n], 'yhat': hybrid_vals})

# --------------------------
# Plots: history + prophet CI + hybrid
# --------------------------
fig = go.Figure()
# historical
fig.add_trace(go.Scatter(x=ts['ds'], y=ts['review_count'], mode='lines', name='Historical'))
# prophet forecast & CI
fig.add_trace(go.Scatter(x=prophet_plot_df['ds'], y=prophet_plot_df['yhat'], mode='lines', name='Prophet forecast', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=prophet_plot_df['ds'], y=prophet_plot_df['yhat_upper'], mode='lines', name='Prophet upper', line=dict(color='orange', width=0), showlegend=False))
fig.add_trace(go.Scatter(x=prophet_plot_df['ds'], y=prophet_plot_df['yhat_lower'], mode='lines', name='Prophet lower', line=dict(color='orange', width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.1)', showlegend=False))

# hybrid
if hybrid_df is not None and (model_choice.startswith("Hybrid")):
    fig.add_trace(go.Scatter(x=hybrid_df['ds'], y=hybrid_df['yhat'], mode='lines', name='Hybrid (averaged) forecast', line=dict(color='green')))

fig.update_layout(title="Forecast (Prophet ± CI) and Hybrid", xaxis_title="Date", yaxis_title="Review Count")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# KPI block: predicted next period, MASE, LSTM eval
# --------------------------
st.subheader("Forecast KPIs")
# predicted total next horizon from Prophet
prophet_future_only = prophet_plot_df[prophet_plot_df['ds'] > ts['ds'].max()].copy()
pred_total_prophet = float(prophet_future_only['yhat'].sum()) if not prophet_future_only.empty else np.nan
st.metric("Prophet predicted total (next {} days)".format(forecast_days), f"{pred_total_prophet:.1f}")

if hybrid_df is not None:
    st.metric("Hybrid predicted total (next {} days)".format(len(hybrid_df)), f"{hybrid_df['yhat'].sum():.1f}")

# compute MASE using last 30 days as test if possible
if len(ts) >= max(30, seq_len+1):
    train = ts['review_count'].values[:-forecast_days] if len(ts)>forecast_days else ts['review_count'].values
    # prepare test y_true (if available) - here we don't have future ground truth; we compute in-sample residuals MASE using last in-sample portion as pseudo-test
    pseudo_test = ts['review_count'].values[-min(len(ts), forecast_days):]
    # get prophet in-sample predicted (use model.predict on historical)
    in_sample_preds = prophet_forecast[prophet_forecast['ds'] <= ts['ds'].max()]['yhat'].values
    in_sample_true = ts['review_count'].values[-len(in_sample_preds):] if len(in_sample_preds) <= len(ts) else ts['review_count'].values
    try:
        mase_val = mase(in_sample_true, in_sample_preds, ts['review_count'].values)
    except Exception:
        mase_val = np.nan
    st.write(f"MASE (in-sample, Prophet) = {mase_val:.3f}")
else:
    st.info("Not enough data for MASE calculation.")

if lstm_eval:
    mse, mae = lstm_eval
    st.write(f"LSTM in-sample MSE = {mse:.3f} | MAE = {mae:.3f}")

# --------------------------
# Anomaly detection (simple): z-score on residuals of Prophet's in-sample fit
# --------------------------
st.subheader("Anomaly detection (simple residual z-score)")
hist_pred = prophet_forecast[prophet_forecast['ds'] <= ts['ds'].max()].copy()
hist_pred = hist_pred.set_index('ds').reindex(ts['ds']).reset_index()
residuals = ts['review_count'].values - hist_pred['yhat'].values
z = (residuals - residuals.mean()) / (residuals.std() + 1e-9)
anom_mask = np.abs(z) > 2.5
anoms = pd.DataFrame({'ds': ts['ds'][anom_mask], 'review_count': ts['review_count'][anom_mask], 'z': z[anom_mask]})
if not anoms.empty:
    st.write("Detected anomalies (|z| > 2.5):")
    st.dataframe(anoms)
    # plot anomalies
    fig_anom = px.scatter(ts, x='ds', y='review_count', title='Anomalies (red)', labels={'ds':'Date','review_count':'Reviews'})
    fig_anom.add_scatter(x=anoms['ds'], y=anoms['review_count'], mode='markers', marker=dict(color='red', size=8), name='Anomaly')
    st.plotly_chart(fig_anom, use_container_width=True)
else:
    st.write("No strong anomalies detected (|z| > 2.5).")

# --------------------------
# Download forecast CSV
# --------------------------
st.subheader("Export")
# export Prophet forecast future subset
export_df = prophet_plot_df[prophet_plot_df['ds'] > ts['ds'].max()][['ds','yhat','yhat_lower','yhat_upper']].copy()
export_df = export_df.rename(columns={'yhat':'prophet_yhat','yhat_lower':'prophet_lower','yhat_upper':'prophet_upper'})
if hybrid_df is not None:
    export_df = export_df.reset_index(drop=True)
    hybrid_df2 = hybrid_df.rename(columns={'yhat':'hybrid_yhat'}).reset_index(drop=True)
    export_df = pd.concat([export_df, hybrid_df2[['hybrid_yhat']]], axis=1)
csv = export_df.to_csv(index=False)
st.download_button("Download forecast CSV", data=csv, file_name="forecast_export.csv", mime="text/csv")

# --------------------------
# End
# --------------------------
st.write("Done — use the controls on the left to change filters, model choice, granularity and LSTM params. If you want the app to auto-load a different CSV file path, upload it or change the default path at the top of the script.")
