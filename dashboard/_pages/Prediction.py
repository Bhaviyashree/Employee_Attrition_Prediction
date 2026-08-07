"""Prediction page - employee attrition form and results."""

from __future__ import annotations

import streamlit as st

from utils.helper import api_request, get_risk_badge_html, load_default_dataset


def render() -> None:
    st.markdown('<p class="section-header">Attrition Prediction</p>', unsafe_allow_html=True)

    df = load_default_dataset()

    # Default options from dataset
    departments = sorted(df["Department"].unique().tolist()) if not df.empty and "Department" in df.columns else ["Sales", "Research & Development", "Human Resources"]
    job_roles = sorted(df["JobRole"].unique().tolist()) if not df.empty and "JobRole" in df.columns else ["Sales Executive", "Research Scientist", "Manager"]
    travel_opts = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]

    with st.form("prediction_form"):
        st.subheader("Employee Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            department = st.selectbox("Department", departments)
            job_role = st.selectbox("Job Role", job_roles)
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=100)
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)

        with col2:
            years_in_role = st.number_input("Years In Current Role", min_value=0, max_value=40, value=3)
            distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=5)
            business_travel = st.selectbox("Business Travel", travel_opts)
            job_sat = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
            env_sat = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3)
            work_life = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=3)
            training = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
            performance = st.select_slider("Performance Rating", options=[1, 2, 3, 4], value=3)
            overtime = st.selectbox("Overtime", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Attrition", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "Age": age,
            "Gender": gender,
            "Department": department,
            "JobRole": job_role,
            "Education": education,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "DistanceFromHome": distance,
            "BusinessTravel": business_travel,
            "JobSatisfaction": job_sat,
            "EnvironmentSatisfaction": env_sat,
            "WorkLifeBalance": work_life,
            "TrainingTimesLastYear": training,
            "PerformanceRating": performance,
            "OverTime": overtime,
        }

        with st.spinner("Running prediction model..."):
            result = api_request("POST", "/api/v1/predict", payload)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            st.info("Ensure the API is running: `uvicorn api.main:app --reload`")
            st.info("And model is trained: `python training/train_model.py`")
        else:
            st.session_state["last_prediction"] = result

            st.divider()
            st.subheader("Prediction Results")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Prediction", result["prediction"])
            with r2:
                st.metric("Confidence Score", f"{result['confidence_score'] * 100:.1f}%")
            with r3:
                st.metric("Probability", f"{result['probability'] * 100:.1f}%")

            st.markdown(
                f"<h4>Risk Level: {get_risk_badge_html(result['risk_level'])}</h4>",
                unsafe_allow_html=True,
            )

            st.subheader("Recommendation")
            st.info(result["recommendation"])

            # Feature explainability note
            with st.expander("Model Explainability"):
                fi = api_request("GET", "/api/v1/model/feature-importance")
                if "error" not in fi:
                    import pandas as pd

                    fi_df = pd.DataFrame({"Feature": fi["features"], "Importance": fi["importances"]})
                    fi_df = fi_df.sort_values("Importance", ascending=False).head(10)
                    st.bar_chart(fi_df.set_index("Feature"))
                    st.caption("Top 10 features driving this model's predictions.")
