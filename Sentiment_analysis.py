# =====================================
# Amazon Product Review Analysis Dashboard
# BiLSTM + Ratings + Radar + Trend
# =====================================
import pandas as pd
import numpy as np
import re
import os
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# =====================================
# Load Dataset
# =====================================
DATA_PATH = "amazon_pc_Data_enriched.csv"  # update if .xlsx
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df.columns = df.columns.str.strip()

st.title("Product Review Sentiment Dashboard")

# =====================================
# Handle Rating Columns (rename properly)
# =====================================
rating_cols = ["1 rating", "2 ratings", "3 ratings", "4 rating", "5 rating"]

# Ensure columns exist
for col in rating_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

# Compute weighted average rating
df["avg_rating"] = (
    df["1 rating"]*1 + df["2 ratings"]*2 + df["3 ratings"]*3 +
    df["4 rating"]*4 + df["5 rating"]*5
) / df[rating_cols].sum(axis=1).replace(0, np.nan)

# Map average rating to sentiment
df["rating_sentiment"] = df["avg_rating"].apply(
    lambda x: "Positive" if pd.notnull(x) and x >= 3 else "Negative"
)

# Convert review_date if available
if "review_date" in df.columns:
    df["date"] = pd.to_datetime(df["review_date"], errors="coerce", dayfirst=True)

# =====================================
# BiLSTM Model for Review Text Sentiment
# =====================================
MODEL_PATH = "bilstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

texts = df["review_body"].astype(str).values
labels = df["Sentiment_pc"].astype(str).values

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Tokenization
max_words = 20000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = labels

# Train or load BiLSTM
if not os.path.exists(MODEL_PATH):
    st.info("Training BiLSTM model... Please wait ⏳")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    embedding_dim = 128
    num_classes = len(np.unique(y))

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Dropout(0.3),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128)

    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
else:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

# =====================================
# Sidebar Filters
# =====================================
marketplaces = st.sidebar.multiselect(
    "🌍 Filter by Marketplace",
    options=sorted(df["market_place"].dropna().unique())
)
categories = st.sidebar.multiselect(
    "📂 Filter by Product Category",
    options=sorted(df["product_category"].dropna().unique())
)

filtered_df = df.copy()
if marketplaces:
    filtered_df = filtered_df[filtered_df["market_place"].isin(marketplaces)]
if categories:
    filtered_df = filtered_df[filtered_df["product_category"].isin(categories)]

# =====================================
# Rating Distribution (Bar Chart)
# =====================================
st.header("⭐ Rating Distribution per Product")

fig1 = px.bar(
    filtered_df,
    x="product_title",
    y=rating_cols,
    title="Rating Breakdown per Product",
    labels={"value": "Count", "variable": "Star Rating"},
    barmode="group"
)
st.plotly_chart(fig1)

# =====================================
# Sentiment by Marketplace
# =====================================
st.header("🌍 Sentiment by Marketplace")

fig2 = px.histogram(
    filtered_df,
    x="market_place",
    color="rating_sentiment",
    barmode="group",
    title="Sentiment by Marketplace (based on ratings)"
)
st.plotly_chart(fig2)

# =====================================
# Pie Chart of Sentiment
# =====================================
sentiment_counts = filtered_df.groupby("rating_sentiment").agg(
    count=("product_id", "count")
).reset_index()

fig3 = px.pie(
    sentiment_counts,
    values="count",
    names="rating_sentiment",
    title="Overall Sentiment Distribution"
)
fig3.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig3)

# =====================================
# Radar Chart + Time Series for Selected Product
# =====================================
st.header("📊 Product Rating Analysis (Radar + Trend)")

selected_product = st.selectbox(
    "Choose a product to analyze:",
    options=sorted(filtered_df["product_title"].dropna().unique())
)

if selected_product:
    prod_data = filtered_df[filtered_df["product_title"] == selected_product]

    # --- Radar Chart ---
    rating_sums = prod_data[rating_cols].sum()

    categories_radar = ["1★", "2★", "3★", "4★", "5★"]
    values = rating_sums.values.tolist()
    values += values[:1]  # close loop

    radar_fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories_radar + [categories_radar[0]],
            fill="toself",
            name=selected_product
        )
    )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.1])),
        showlegend=False,
        title=f"⭐ Rating Distribution for '{selected_product}'"
    )

    # --- Time Series Trend ---
    if "date" in prod_data.columns:
        trend_fig = px.line(
            prod_data.sort_values("date"),
            x="date",
            y="avg_rating",
            title=f"Avg Rating Trend Over Time for '{selected_product}'",
            markers=True
        )
    else:
        trend_fig = px.line(
            prod_data.reset_index(),
            x="index",
            y="avg_rating",
            title=f"Avg Rating Trend Over Time for '{selected_product}'",
            markers=True
        )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(radar_fig, use_container_width=True)
    with col2:
        st.plotly_chart(trend_fig, use_container_width=True)

# =====================================
# Predict Sentiment for New Review
# =====================================
st.header("Predict Sentiment for New Review (BiLSTM)")
new_review = st.text_area("Enter review text:")

if st.button("Predict Sentiment"):
    if new_review.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([new_review])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded)
        sentiment_idx = np.argmax(pred)
        sentiment_label = le.inverse_transform([sentiment_idx])[0]
        st.success(f"Predicted Sentiment: **{sentiment_label}**")

# =====================================
# Show Sample Data
# =====================================
st.subheader("Sample Product Ratings & Reviews")
st.write(filtered_df.head(10)[[
    "product_id", "product_title", "market_place",
    "1 rating", "2 ratings", "3 ratings", "4 rating", "5 rating",
    "avg_rating", "rating_sentiment", "Sentiment_pc"
]])

