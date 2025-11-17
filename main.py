import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Streamlit App Title
# -------------------------------
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
st.title("Demand Forecasting Dashboard")

#  Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

     #    Data Preprocessing
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Feature Engineering
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    #  Filters for Stakeholders
    
    st.subheader("Filter Data")
    product_filter = st.multiselect("Select Products", options=df['Product'].unique(), default=df['Product'].unique())
    filtered_df = df[df['Product'].isin(product_filter)]
    st.dataframe(filtered_df)

    # CSV Download Button for Filtered Data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(filtered_df)
    st.download_button(label="Download Filtered Data as CSV", data=csv_data, file_name='filtered_demand.csv', mime='text/csv')

#    Model Selection 
    
    st.subheader("Select Features & Target for Forecasting")
    default_features = ['Month', 'Day', 'Weekday', 'Price', 'Promotion']
    features = st.multiselect("Select Features", options=filtered_df.columns.tolist(), default=default_features)
    target = st.selectbox("Select Target Column", options=filtered_df.columns.tolist(), index=filtered_df.columns.get_loc('Demand'))

    X = filtered_df[features]
    y = filtered_df[target]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Metrics
    st.subheader("Model Performance")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Visualization: Actual vs Predicted
   
    st.subheader("Actual vs Predicted Demand")
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(len(y_test)), y=y_test, label="Actual")
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
    plt.xlabel("Samples")
    plt.ylabel("Demand")
    plt.legend()
    st.pyplot(plt)

    # 6 Forecast Future Demand
   
    st.subheader("Forecast Future Demand")
    days_to_forecast = st.number_input("Days to Forecast", min_value=1, max_value=60, value=30)
    if st.button("Forecast"):
        future_dates = pd.date_range(start=filtered_df['Date'].max() + pd.Timedelta(days=1), periods=days_to_forecast)
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['Weekday'] = future_df['Date'].dt.weekday

        # For other features like Price/Promotion, use mean of historical values
        for col in features:
            if col not in ['Month', 'Day', 'Weekday']:
                future_df[col] = filtered_df[col].mean()

        X_future = future_df[features]
        future_df['Predicted_Demand'] = model.predict(X_future)

        st.subheader("Future Demand Forecast")
        st.dataframe(future_df[['Date', 'Predicted_Demand']])
        st.line_chart(future_df.set_index('Date')['Predicted_Demand'])

        # Download Forecast as CSV
        forecast_csv = convert_df_to_csv(future_df[['Date', 'Predicted_Demand']])
        st.download_button("Download Forecast as CSV", data=forecast_csv, file_name='future_demand_forecast.csv', mime='text/csv')
