import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import base64

# -------------------------
# Load Model and Dataset
# -------------------------
model = pk.load(open("C:/Users/user/Desktop/ML2/Car_Price_Predictor/notebooks/FC212035_Shenal/model.pkl", "rb"))
cars_data = pd.read_csv("C:/Users/user/Desktop/ML2/Car_Price_Predictor/data/raw/car_data.csv")
brand_labels = pk.load(open("C:/Users/user/Desktop/ML2/Car_Price_Predictor/notebooks/FC212035_Shenal/brand_labels.pkl", "rb"))

# Extract brand name
def get_brand_name(car_name):
    parts = car_name.split()
    return " ".join(parts[:2]).strip()

cars_data["name"] = cars_data["name"].apply(get_brand_name)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# -------------------------
# Background Image + Overlay
# -------------------------
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.6);
            z-index: 0;
        }}
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1y4p8pa, .block-container {{
            position: relative;
            z-index: 1;
        }}
        div.stButton > button:hover {{
            background-color: #00b050 !important;
            color: white !important;
            border: none !important;
        }}
        .stTabs [role="tablist"] {{
            border-bottom: 2px solid #2b9348;
        }}
        .stTabs [role="tab"] {{
            background-color: rgba(255,255,255,0.1);
            color: white !important;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #00b050 !important;
            color: white !important;
        }}
        </style>
        """, unsafe_allow_html=True
    )

add_bg_from_local("C:/Users/user/Desktop/ML2/Car_Price_Predictor/2.png")

# -------------------------
# Hero Header
# -------------------------
st.markdown("""
<div style="background-color:rgba(43,45,66,0.85);padding:20px;border-radius:12px;">
    <h1 style="color:white;text-align:center;">🚘 Car Price Prediction</h1>
    <p style="color:#edf2f4;text-align:center;font-size:18px;">
    Estimate your car's value in seconds. Just provide a few details below 👇
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Tabs Layout
# -------------------------
tab1, tab2, tab3 = st.tabs(["🚗 Car Details", "⚙️ Specifications", "💰 Prediction"])

# --- Tab 1: Car Details ---
with tab1:
    st.subheader("Car Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox("Car Brand", cars_data["name"].unique())
        year = st.slider("Manufacture Year", 1994, 2024, 2015)
    with col2:
        km_driven = st.number_input("Kilometers Driven", 500, 200000, step=500, value=50000)
        owner = st.selectbox("Owner", cars_data["owner"].unique())

# --- Tab 2: Specifications ---
with tab2:
    st.subheader("Specifications")
    col3, col4 = st.columns(2)
    with col3:
        fuel = st.selectbox("Fuel Type", cars_data["fuel"].unique())
        transmission = st.radio("Transmission", cars_data["transmission"].unique())
        seats = st.slider("Seats", 2, 10, 5)
    with col4:
        mileage = st.slider("Mileage (km/l)", 10, 40, 20)
        engine = st.slider("Engine Capacity (CC)", 700, 5000, 1500, step=100)
        max_power = st.slider("Max Power (bhp)", 0, 200, 80)

# -------------------------
# Tab 3: Prediction
# -------------------------
with tab3:
    st.subheader("Prediction")
    st.markdown("Click below to predict your car price 🎯")

    col1, col2 = st.columns([1, 1])

    with col1:
        predict_btn = st.button("🧠 Predict Price", use_container_width=True)
    with col2:
        reset_btn = st.button("♻️ Reset", use_container_width=True)

    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None

    # Predict
    if predict_btn:
     input_data = pd.DataFrame(
         [[name, year, km_driven, fuel, "Individual",
           transmission, owner, mileage, engine,
           max_power, seats]],
         columns=[
             "name","year","km_driven","fuel",
             "seller_type","transmission","owner",
             "mileage","engine","max_power","seats"
         ]
     )

     input_data["name"] = input_data["name"].map(brand_labels)

     input_data["fuel"] = input_data["fuel"].map({
         "Diesel":1,
         "Petrol":2,
         "LPG":3,
         "CNG":4
     })

     input_data["seller_type"] = input_data["seller_type"].map({
         "Individual":1,
         "Dealer":2,
         "Trustmark Dealer":3
     })

     input_data["transmission"] = input_data["transmission"].map({
         "Manual":1,
         "Automatic":2
     })

     input_data["owner"] = input_data["owner"].map({
         "First Owner":1,
         "Second Owner":2,
         "Third Owner":3,
         "Fourth & Above Owner":4,
         "Test Drive Car":5
     })

     input_data = input_data.astype(float)

     st.session_state.predicted_price = model.predict(input_data)[0]
    

     

    

    # Reset only price
    if reset_btn:
        st.session_state.predicted_price = None

    # Show Result
    if st.session_state.predicted_price is not None:
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:20px;">
                <h2 style="color:#00ff88; background-color:rgba(0,0,0,0.7);
                padding:15px; border-radius:10px; display:inline-block;
                box-shadow:0 0 15px rgba(0,255,0,0.5);">
                💰 Estimated Price: ₹ {st.session_state.predicted_price:,.2f}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
