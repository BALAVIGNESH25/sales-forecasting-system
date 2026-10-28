import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    return joblib.load("sales_forecasting_model.pkl")

artifact = load_artifact()

model = artifact["model"]
model_name = artifact["model_name"]
feature_cols = artifact["feature_columns"]
history_df = artifact["history"].copy()
families = artifact["families"]

history_df["date"] = pd.to_datetime(history_df["date"])

def forecast_sales(history_data, family, horizon):
    values = (
        history_data[history_data["family"] == family]
        .sort_values("date")["sales"]
        .tolist()
    )

    last_date = history_data["date"].max()
    forecasts = []

    family_cols = [
        c for c in feature_cols if c.startswith("family_")
    ]

    for step in range(1, horizon + 1):
        date = last_date + pd.Timedelta(days=step)

        row = {
            "day": date.day,
            "month": date.month,
            "year": date.year,
            "day_of_week": date.dayofweek,
            "day_of_year": date.dayofyear,
            "is_weekend": int(date.dayofweek >= 5),
            "lag_1": values[-1],
            "lag_7": values[-7],
            "lag_14": values[-14],
            "rolling_mean_7": np.mean(values[-7:]),
            "rolling_mean_14": np.mean(values[-14:])
        }

        for col in family_cols:
            row[col] = int(col == f"family_{family}")

        X_future = pd.DataFrame([row]).reindex(
            columns=feature_cols, fill_value=0
        )

        prediction = max(0.0, float(model.predict(X_future)[0]))
        values.append(prediction)

        forecasts.append({
            "Date": date,
            "Predicted Sales": prediction
        })

    return pd.DataFrame(forecasts)

st.title("📈 Sales Forecasting System")
st.write(
    "Forecast future product-family sales using a trained "
    "machine learning model."
)

st.caption(f"Selected model: {model_name}")

family = st.selectbox(
    "Select product family",
    families
)

horizon = st.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=30,
    value=16
)

if st.button("Generate Forecast", type="primary"):
    result = forecast_sales(history_df, family, horizon)

    col1, col2, col3 = st.columns(3)

    col1.metric("Forecast Days", len(result))
    col2.metric(
        "Total Forecast Sales",
        f"{result['Predicted Sales'].sum():,.0f}"
    )
    col3.metric(
        "Average Daily Sales",
        f"{result['Predicted Sales'].mean():,.2f}"
    )

    st.subheader("Future Sales Forecast")
    st.line_chart(
        result.set_index("Date")["Predicted Sales"]
    )

    st.subheader("Forecast Data")
    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download Forecast CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

st.info(
    "Forecasts are model estimates based on historical sales. "
    "Future promotions and holidays are not included in this version."
)
