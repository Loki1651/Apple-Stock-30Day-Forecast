import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Apple Stock Forecasting App")

st.title(" Apple Stock Forecasting App")
st.write("30-Day Forecast using SARIMA model")

try:
    # Load model
    with open("sarima_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Forecast
    forecast_obj = model.get_forecast(steps=30)
    forecast_df = forecast_obj.summary_frame()

    # Prepare DataFrame
    df = pd.DataFrame({
        "Day": range(1, 31),
        "Forecast Price": forecast_df["mean"].values,
        "Lower CI": forecast_df["mean_ci_lower"].values,
        "Upper CI": forecast_df["mean_ci_upper"].values
    })

    st.success("SARIMA model loaded successfully ")
    st.dataframe(df)

    # ===============================
    # DAY SELECT (SLIDER)
    # ===============================
    st.subheader(" Select Day to Explore Forecast")

    selected_day = st.slider(
        "Select Forecast Day",
        min_value=1,
        max_value=30,
        value=1
    )

    selected_price = df.loc[
        df["Day"] == selected_day, "Forecast Price"
    ].values[0]

    st.info(
        f" Day {selected_day} Forecast Price: ₹ {selected_price:.2f}"
    )

    # ===============================
    # TREND UP TO SELECTED DAY
    # ===============================
    st.subheader(" Forecast Trend up to Selected Day")

    st.line_chart(
        df[df["Day"] <= selected_day]
        .set_index("Day")[["Forecast Price"]]
    )

    # ===============================
    # MOVING AVERAGE TREND
    # ===============================
    df["MA_7"] = df["Forecast Price"].rolling(7).mean()
    df["MA_14"] = df["Forecast Price"].rolling(14).mean()

    st.subheader(" Moving Average Trend")
    st.line_chart(
        df.set_index("Day")[["Forecast Price", "MA_7", "MA_14"]]
    )

    # ===============================
    # CONFIDENCE INTERVAL BANDS
    # ===============================
    st.subheader(" Forecast Confidence Interval")
    st.line_chart(
        df.set_index("Day")[["Lower CI", "Forecast Price", "Upper CI"]]
    )

except Exception as e:
    st.error("Unable to load SARIMA model")
    st.error(str(e))
