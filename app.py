import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📊",
    layout="wide"
)


model = joblib.load("sales_prediction_xgboost.pkl")



st.title("Sales Prediction System")
st.write(
    "Predict product sales using historical sales patterns and business information."
)

st.divider()



st.sidebar.header("Sales Information")


order_year = st.sidebar.number_input(
    "Order Year",
    min_value=2010,
    max_value=2030,
    value=2017
)

order_month = st.sidebar.slider(
    "Order Month",
    1,
    12,
    6
)

order_day = st.sidebar.slider(
    "Order Day",
    1,
    31,
    15
)

day_week = st.sidebar.slider(
    "Order Day Of Week",
    0,
    6,
    3
)

quarter = st.sidebar.selectbox(
    "Quarter",
    [1,2,3,4]
)

shipping_days = st.sidebar.slider(
    "Shipping Days",
    0,
    10,
    3
)




category = st.selectbox(
    "Category",
    [
        "Furniture",
        "Office Supplies",
        "Technology"
    ]
)


sub_category = st.selectbox(
    "Sub Category",
    [
        "Accessories",
        "Appliances",
        "Art",
        "Binders",
        "Bookcases",
        "Chairs",
        "Copiers",
        "Phones",
        "Storage",
        "Tables"
    ]
)


region = st.selectbox(
    "Region",
    [
        "East",
        "West",
        "Central",
        "South"
    ]
)


segment = st.selectbox(
    "Customer Segment",
    [
        "Consumer",
        "Corporate",
        "Home Office"
    ]
)


ship_mode = st.selectbox(
    "Shipping Mode",
    [
        "Standard Class",
        "Second Class",
        "First Class",
        "Same Day"
    ]
)


state = st.text_input(
    "State",
    "California"
)


city = st.text_input(
    "City",
    "Los Angeles"
)


country = st.text_input(
    "Country",
    "United States"
)


if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "Year":[order_year],
        "Order_Year":[order_year],
        "Order_Month":[order_month],
        "Order_Day":[order_day],
        "Order_DayOfWeek":[day_week],
        "Order_Quarter":[quarter],
        "Shipping_Days":[shipping_days],

        "Category_"+category:[1],
        "Sub-Category_"+sub_category:[1],
        "Region_"+region:[1],
        "Segment_"+segment:[1],
        "Ship Mode_"+ship_mode:[1],
        "Country_"+country:[1],
        "State_"+state:[1],
        "City_"+city:[1]
    })



    model_features = model.get_booster().feature_names


    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0


    input_data = input_data[model_features]


    prediction = model.predict(input_data)[0]


    st.success("Prediction Completed")

    st.metric(
        label="Predicted Sales",
        value=f"${prediction:,.2f}"
    )


st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Model\n\nXGBoost Regressor")

with col2:
    st.info("Task\n\nSales Forecasting")

with col3:
    st.info("Input\n\nBusiness & Order Data")