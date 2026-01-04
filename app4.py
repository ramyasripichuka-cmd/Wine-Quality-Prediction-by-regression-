import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ---------------- CSS + ANIMATIONS ----------------
st.markdown("""
<style>

/* GLOBAL TEXT COLOR */
* {
    color: black !important;
}

/* BACKGROUND */
.stApp {
    background-color: lavender;
    overflow-x: hidden;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: violet;
}

/* INPUTS */
input {
    background-color: #c6f7c6 !important;
    color: black !important;
}

/* ================= BUTTON STYLE ================= */
div.stButton > button {
    background: linear-gradient(135deg, #00bcd4, #ff2f92);
    color: white !important;
    font-size: 20px;
    font-weight: 900;
    border-radius: 14px;
    padding: 12px 30px;
    border: none;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.35);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #ff2f92, #00bcd4);
    transform: scale(1.05);
}

/* RESULT */
.result {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 18px;
}

/* GRAPE FALL ANIMATION */
.grape {
    position: fixed;
    top: -50px;
    font-size: 40px;
    animation: fall 6s linear infinite;
    z-index: 1;
}

@keyframes fall {
    0% { top: -50px; opacity: 1; }
    100% { top: 110%; opacity: 0; }
}

/* GRAPE BLAST */
.blast {
    animation: blast 1s ease-out;
}

@keyframes blast {
    0% { transform: scale(0.2); opacity: 0; }
    100% { transform: scale(1.5); opacity: 1; }
}

/* WINE GLASSES */
.cheers {
    position: fixed;
    top: 40%;
    font-size: 65px;
    animation: cheers 4s infinite;
    z-index: 2;
}

.left {
    left: 20px;
}

.right {
    right: 20px;
}

@keyframes cheers {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(15deg); }
}

</style>
""", unsafe_allow_html=True)

# ---------------- FALLING GRAPES ----------------
for i in range(10):
    st.markdown(
        f"<div class='grape' style='left:{i*10 + 5}%; animation-delay:{i*0.5}s;'>🍇</div>",
        unsafe_allow_html=True
    )

# ---------------- WINE GLASSES ----------------
st.markdown("<div class='cheers left'>🍷</div>", unsafe_allow_html=True)
st.markdown("<div class='cheers right'>🍷</div>", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🍷 Wine Quality Prediction (Regression Model)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Wine Features")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 5.0)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.08)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    X = np.random.rand(300, 6)
    y = np.random.rand(300) * 10
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# ---------------- PREDICTION ----------------
if st.button("🍇 Predict Wine Quality 🍇"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, alcohol]])
    prediction = model.predict(input_data)[0]

    st.markdown(
        "<div class='result blast'>🍇 Predicted Wine Quality: {:.2f} 🍇</div>".format(prediction),
        unsafe_allow_html=True
    )

    for i in range(6):
        st.markdown(
            f"<div class='grape blast' style='left:{i*15 + 10}%;'>🍇</div>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>✨ Project by <b>Ramyasri</b> ✨</div>",
    unsafe_allow_html=True
)
