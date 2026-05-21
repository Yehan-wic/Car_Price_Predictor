import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "FC212027-random_forest" / "car_price_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "FC212027-random_forest" / "scaler.pkl"

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "car_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned-random_forest_car_data-FC212027.csv"

IMAGE_1 = BASE_DIR / "1.jpg"
IMAGE_2 = BASE_DIR / "2.png"

DEFAULT_INR_TO_LKR = 3.60

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.25), transparent 35%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.18), transparent 30%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #ffffff;
    }

    .main-container {
        padding: 20px 40px;
    }

    .hero-section {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 28px;
        padding: 45px;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.45);
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 58px;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 18px;
        background: linear-gradient(90deg, #ffffff, #dbeafe, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #cbd5e1;
        line-height: 1.6;
        max-width: 720px;
        margin-bottom: 25px;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 20px;
    }

    .custom-badge {
        background: rgba(59, 130, 246, 0.14);
        border: 1px solid rgba(96, 165, 250, 0.35);
        color: #bfdbfe;
        padding: 10px 16px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }

    .section-title {
        font-size: 32px;
        font-weight: 850;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-card {
        background: linear-gradient(135deg, #16a34a, #15803d);
        padding: 34px;
        border-radius: 26px;
        border: 1px solid rgba(187, 247, 208, 0.3);
        box-shadow: 0 18px 60px rgba(22, 163, 74, 0.28);
        margin-top: 20px;
        text-align: center;
    }

    .result-label {
        font-size: 18px;
        color: #dcfce7;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-price {
        font-size: 52px;
        line-height: 1.1;
        font-weight: 950;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .result-note {
        font-size: 14px;
        color: #bbf7d0;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 20px;
        border-radius: 20px;
        min-height: 130px;
    }

    .info-title {
        font-size: 18px;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 8px;
    }

    .info-text {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.55;
    }

    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 26px;
        padding: 32px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 18px;
        border-radius: 18px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 16px;
        border: none;
        padding: 15px 22px;
        font-size: 18px;
        font-weight: 800;
        color: white;
        background: linear-gradient(135deg, #2563eb, #14b8a6);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.30);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 45px rgba(20, 184, 166, 0.35);
    }

    .small-caption {
        color: #94a3b8;
        font-size: 13px;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# --------------------------------------------------
# Load Category Mappings
# --------------------------------------------------
@st.cache_data
def load_category_mappings():
    fallback_maps = {
        "fuel": {
            "CNG": 0,
            "Diesel": 1,
            "LPG": 2,
            "Petrol": 3
        },
        "seller_type": {
            "Dealer": 0,
            "Individual": 1,
            "Trustmark Dealer": 2
        },
        "transmission": {
            "Automatic": 0,
            "Manual": 1
        },
        "owner": {
            "First Owner": 0,
            "Fourth & Above Owner": 1,
            "Second Owner": 2,
            "Test Drive Car": 3,
            "Third Owner": 4
        },
        "brand": {
            "Brand Code 0": 0,
            "Brand Code 1": 1,
            "Brand Code 2": 2,
            "Brand Code 3": 3,
            "Brand Code 4": 4,
            "Brand Code 5": 5,
            "Brand Code 6": 6,
            "Brand Code 7": 7,
            "Brand Code 8": 8,
            "Brand Code 9": 9,
            "Brand Code 10": 10,
            "Brand Code 11": 11,
            "Brand Code 12": 12,
            "Brand Code 13": 13,
            "Brand Code 14": 14,
            "Brand Code 15": 15,
            "Brand Code 16": 16,
            "Brand Code 17": 17,
            "Brand Code 18": 18,
            "Brand Code 19": 19,
            "Brand Code 20": 20,
            "Brand Code 21": 21,
            "Brand Code 22": 22,
            "Brand Code 23": 23,
            "Brand Code 24": 24,
            "Brand Code 25": 25,
            "Brand Code 26": 26,
            "Brand Code 27": 27,
            "Brand Code 28": 28,
            "Brand Code 29": 29,
            "Brand Code 30": 30
        }
    }

    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        processed_df = pd.read_csv(PROCESSED_DATA_PATH)

        min_len = min(len(raw_df), len(processed_df))
        raw_df = raw_df.iloc[:min_len].copy()
        processed_df = processed_df.iloc[:min_len].copy()

        combined = pd.DataFrame()
        combined["fuel_name"] = raw_df["fuel"]
        combined["fuel_code"] = processed_df["fuel"]

        combined["seller_name"] = raw_df["seller_type"]
        combined["seller_code"] = processed_df["seller_type"]

        combined["transmission_name"] = raw_df["transmission"]
        combined["transmission_code"] = processed_df["transmission"]

        combined["owner_name"] = raw_df["owner"]
        combined["owner_code"] = processed_df["owner"]

        combined["brand_name"] = raw_df["name"].astype(str).str.split().str[0]
        combined["brand_code"] = processed_df["brand"]

        def create_map(name_col, code_col):
            temp = combined[[name_col, code_col]].dropna().drop_duplicates()
            temp[code_col] = temp[code_col].astype(int)
            return dict(sorted(zip(temp[name_col], temp[code_col])))

        return {
            "fuel": create_map("fuel_name", "fuel_code"),
            "seller_type": create_map("seller_name", "seller_code"),
            "transmission": create_map("transmission_name", "transmission_code"),
            "owner": create_map("owner_name", "owner_code"),
            "brand": create_map("brand_name", "brand_code")
        }

    except Exception:
        return fallback_maps


try:
    model, scaler = load_model_and_scaler()
    category_maps = load_category_mappings()
except Exception as e:
    st.error("Model or scaler loading failed.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🚗 Car Price AI")
    st.write("ML-based car selling price prediction system.")

    if IMAGE_2.exists():
        st.image(str(IMAGE_2), use_container_width=True)

    st.markdown("---")

    st.markdown("### Currency Settings")
    inr_to_lkr = st.number_input(
        "INR to LKR conversion rate",
        min_value=1.00,
        max_value=10.00,
        value=DEFAULT_INR_TO_LKR,
        step=0.01,
        help="Your model predicts based on the original dataset currency. This converts that value to LKR."
    )

    st.info("Update this rate if you want a more accurate LKR conversion.")

    st.markdown("---")

    st.markdown("### Model Information")
    st.write("Algorithm: Random Forest")
    st.write("Output: Estimated car selling price")
    st.write("Display Currency: LKR")

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

hero_left, hero_right = st.columns([1.4, 0.8], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">Smart Car Price Predictor</div>
            <div class="hero-subtitle">
                Predict the estimated selling price of a used car using machine learning.
                Enter vehicle details such as brand, year, mileage, fuel type, engine capacity,
                and ownership status to get an instant price prediction in LKR.
            </div>
            <div class="badge-row">
                <span class="custom-badge">🤖 Machine Learning</span>
                <span class="custom-badge">🚘 Used Car Valuation</span>
                <span class="custom-badge">💰 LKR Price Output</span>
                <span class="custom-badge">📊 Group Project</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_right:
    if IMAGE_1.exists():
        st.image(str(IMAGE_1), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:95px;">🚗</div>
                <h2>AI Price Engine</h2>
                <p style="color:#94a3b8;">
                    Fast prediction using trained ML model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Info Cards
# --------------------------------------------------
info1, info2, info3 = st.columns(3, gap="medium")

with info1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">⚡ Instant Prediction</div>
            <div class="info-text">
                Enter car details and get a predicted selling price within seconds.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">📈 ML-Based Output</div>
            <div class="info-text">
                The prediction is generated using your trained machine learning model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with info3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">🇱🇰 LKR Display</div>
            <div class="info-text">
                The model output is converted and displayed as Sri Lankan Rupees.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Input Form
# --------------------------------------------------
st.markdown('<div class="section-title">Enter Car Details</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Fill the details below to estimate the selling price.</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2019,
            step=1
        )

        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=63000,
            step=1000
        )

        fuel = st.selectbox(
            "Fuel Type",
            options=list(category_maps["fuel"].keys())
        )

        seller_type = st.selectbox(
            "Seller Type",
            options=list(category_maps["seller_type"].keys())
        )

    with col2:
        brand = st.selectbox(
            "Car Brand",
            options=list(category_maps["brand"].keys())
        )

        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=0.1
        )

        engine = st.number_input(
            "Engine Capacity CC",
            min_value=500.0,
            max_value=5000.0,
            value=1200.0,
            step=50.0
        )

        max_power = st.number_input(
            "Max Power BHP",
            min_value=20.0,
            max_value=500.0,
            value=80.0,
            step=1.0
        )

    with col3:
        transmission = st.selectbox(
            "Transmission",
            options=list(category_maps["transmission"].keys())
        )

        owner = st.selectbox(
            "Owner Type",
            options=list(category_maps["owner"].keys())
        )

        seats = st.number_input(
            "Number of Seats",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=1.0
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Predict Price in LKR")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submit:
    input_data = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "fuel": category_maps["fuel"][fuel],
        "seller_type": category_maps["seller_type"][seller_type],
        "transmission": category_maps["transmission"][transmission],
        "owner": category_maps["owner"][owner],
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "brand": category_maps["brand"][brand]
    }])

    feature_order = [
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "brand"
    ]

    input_data = input_data[feature_order]

    try:
        scaled_input = scaler.transform(input_data)
        raw_prediction = model.predict(scaled_input)[0]

        lkr_prediction = raw_prediction * inr_to_lkr

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Selling Price</div>
                <div class="result-price">LKR {lkr_prediction:,.2f}</div>
                <div class="result-note">
                    Converted using rate: 1 INR = {inr_to_lkr:.2f} LKR
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Raw Model Output", f"{raw_prediction:,.2f}")

        with m2:
            st.metric("Conversion Rate", f"{inr_to_lkr:.2f}")

        with m3:
            st.metric("Final Currency", "LKR")

        with st.expander("View input data used for prediction"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Check whether the model, scaler, and input feature order are correct.")
        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="small-caption">
        Developed for Machine Learning Course Module | Car Price Prediction Group Project
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)