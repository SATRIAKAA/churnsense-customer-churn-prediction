
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ChurnSense | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 ChurnSense")
st.caption(
    "Customer Churn Intelligence · Machine Learning powered"
)

st.markdown(
    "Analisis risiko pelanggan berhenti berlangganan "
    "berdasarkan profil, layanan, kontrak, dan informasi billing."
)

# Sidebar
with st.sidebar:
    st.header("Tentang Model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Decision threshold:** 0.35")
    st.write("**ROC-AUC:** 0.8419")
    st.divider()
    st.caption(
        "Model menggunakan dataset Telco Customer Churn. "
        "Prediksi merupakan estimasi, bukan kepastian."
    )

# Customer form
with st.form("customer_form"):
    st.subheader("1. Informasi pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.number_input(
            "Tenure (months)", min_value=0,
            max_value=100, value=12, step=1,
        )
        phone_service = st.selectbox(
            "Phone service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )
        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox(
            "Paperless billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.subheader("2. Layanan pelanggan")

    c1, c2, c3 = st.columns(3)

    internet_options = ["No", "Yes", "No internet service"]

    with c1:
        online_security = st.selectbox(
            "Online security", internet_options
        )
        online_backup = st.selectbox(
            "Online backup", internet_options
        )
        device_protection = st.selectbox(
            "Device protection", internet_options
        )

    with c2:
        tech_support = st.selectbox(
            "Tech support", internet_options
        )
        streaming_tv = st.selectbox(
            "Streaming TV", internet_options
        )
        streaming_movies = st.selectbox(
            "Streaming movies", internet_options
        )

    with c3:
        monthly_charges = st.number_input(
            "Monthly charges ($)",
            min_value=0.0, value=75.5, step=1.0,
        )
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0, value=906.0, step=10.0,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn Risk",
        use_container_width=True,
        type="primary",
    )

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(
                API_URL, json=payload, timeout=15
            )

            if response.ok:
                result = response.json()
                st.session_state["prediction_result"] = result
            else:
                st.error(
                    f"API error {response.status_code}: "
                    f"{response.text}"
                )
        except requests.RequestException:
            st.error(
                "Tidak dapat terhubung ke FastAPI. "
                "Pastikan server API sudah berjalan."
            )

result = st.session_state.get("prediction_result")

if result:
    st.divider()
    st.subheader("3. Prediction results")

    probability = result["churn_probability"]
    prediction = result["churn_prediction"]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Churn probability",
        f"{probability:.2%}",
    )
    metric2.metric(
        "Decision threshold",
        f"{result['threshold']:.0%}",
    )
    metric3.metric(
        "Prediction",
        result["churn_label"],
    )

    st.progress(
        max(0.0, min(1.0, probability)),
        text=f"Estimated churn probability: {probability:.2%}",
    )

    if prediction == 1:
        st.warning(
            "Pelanggan teridentifikasi berisiko churn. "
            "Pertimbangkan evaluasi retensi lebih lanjut."
        )
    else:
        st.success(
            "Pelanggan tidak melewati threshold churn "
            "yang ditetapkan model."
        )

    st.caption(
        "Prediksi adalah estimasi model berdasarkan pola historis. "
        "Jangan gunakan sebagai satu-satunya dasar keputusan bisnis."
    )
