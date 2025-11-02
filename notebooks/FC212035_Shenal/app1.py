import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import base64

# -------------------------
# Load Model and Dataset
# -------------------------
model = pk.load(open("C:/Users/user/Desktop/ML1/Car_Price_Predictor/notebooks/FC212035_Shenal/model.pkl", "rb"))
cars_data = pd.read_csv("C:/Users/user/Desktop/ML1/Car_Price_Predictor/data/raw/car_data.csv")

# Extract brand name
def get_brand_name(car_name):
    return car_name.split(" ")[0].strip()
cars_data["name"] = cars_data["name"].apply(get_brand_name)

# -------------------------
# Add Background + Overlay
# -------------------------
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        /* Background Image */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark Overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 0;
        }}

        /* Keep all content above overlay */
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1y4p8pa, .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* ✅ Change only button hover color */
        div.stButton > button:hover {{
            background-color: #00b050 !important;  /* Green hover color */
            color: white !important;
            border: none !important;
        }}

        /* ✅ Style Streamlit Tabs */
        .stTabs [role="tablist"] {{
            border-bottom: 2px solid #2b9348;
        }}
        .stTabs [role="tab"] {{
            background-color: rgba(255,255,255,0.1);
            color: white !important;
            border-radius: 8px 8px 0 0;
            margin-right: 4px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #00b050 !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Add your image path here
add_bg_from_local("C:/Users/user/Desktop/ML1/Car_Price_Predictor/2.png")

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# -------------------------
# Hero Header
# -------------------------
st.markdown(
    """
    <div style="background-color:rgba(43,45,66,0.85);padding:20px;border-radius:12px;">
        <h1 style="color:white;text-align:center;">🚘 Car Price Prediction</h1>
        <p style="color:#edf2f4;text-align:center;font-size:18px;">
        Estimate your car's value in seconds. Just provide a few details below 👇
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar
# -------------------------
# st.sidebar.header("📊 Dataset Insights")
# st.sidebar.write("Cars in Dataset:", cars_data.shape[0])
# st.sidebar.write("Unique Brands:", cars_data['name'].nunique())
# st.sidebar.write("Fuel Types:", ", ".join(cars_data['fuel'].unique()))
# st.sidebar.info("💡 Tip: More recent models with low mileage usually get higher prices.")

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
        km_driven = st.number_input("Kilometers Driven", min_value=500, max_value=200000, step=500, value=50000)
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

# --- Tab 3: Prediction ---
with tab3:
    st.subheader("Prediction")
    st.markdown("Click below to predict your car price 🎯")

    if st.button("🔮 Predict Price"):
        # Prepare input data
        input_data = pd.DataFrame(
            [[name, year, km_driven, fuel, "Individual", transmission, owner, mileage, engine, max_power, seats]],
            columns=["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats"]
        )

        # Encoding categorical data
        input_data["owner"].replace(
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
            [1, 2, 3, 4, 5], inplace=True
        )
        input_data["fuel"].replace(["Diesel", "Petrol", "LPG", "CNG"], [1, 2, 3, 4], inplace=True)
        input_data["seller_type"].replace(["Individual", "Dealer", "Trustmark Dealer"], [1, 2, 3], inplace=True)
        input_data["transmission"].replace(["Manual", "Automatic"], [1, 2], inplace=True)
        input_data["name"].replace(
            ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
             'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
             'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
             'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
             'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
            list(range(1, 32)), inplace=True
        )

        # Make Prediction
        car_price = model.predict(input_data)

        # Display Result with Highlight
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:20px;">
                <h2 style="color:#00ff88; background-color:rgba(0,0,0,0.7); 
                padding:15px; border-radius:10px; display:inline-block;
                box-shadow:0 0 15px rgba(0,255,0,0.5);">
                💰 Estimated Price: ₹ {car_price[0]:,.2f}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        # st.balloons()
