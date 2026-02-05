import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess
from anomaly_models import detect_anomalies
from forecasting import forecast_cost
from recommendations import generate_recommendations

# ======================
# Streamlit Config
# ======================
st.set_page_config(page_title="Cloud Cost Anomaly Detection", layout="wide")

st.title("☁️ Cloud Cost Anomaly Detection & Optimization Dashboard")
st.caption("AI-driven GCP billing analysis with FinOps insights")

# ======================
# File Upload
# ======================
uploaded = st.file_uploader("📤 Upload GCP Billing CSV", type=["csv"])

if uploaded:
    # ======================
    # Data Pipeline
    # ======================
    df = load_and_preprocess(uploaded)
    df = detect_anomalies(df)
    df = generate_recommendations(df)

    # ======================
    # KPI Calculation
    # ======================
    total_spend = df["Cost"].sum()
    anomaly_spend = df[df["Final_Anomaly"] == 1]["Cost"].sum()
    total_savings = df["Estimated_Saving_INR"].sum()
    avg_cost = df["Cost"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Cloud Spend (INR)", f"₹ {total_spend:,.0f}")
    col2.metric("🚨 Anomalous Spend (INR)", f"₹ {anomaly_spend:,.0f}")
    col3.metric("💡 Estimated Savings (INR)", f"₹ {total_savings:,.0f}")

    st.success(f"💰 Total Identified Savings Opportunity: ₹{total_savings:,.0f}")
    st.divider()

    # ======================
    # Monthly Budget Alert
    # ======================
    st.subheader("💼 Monthly Budget Monitoring")

    monthly_budget = st.number_input(
        "Set Monthly Budget (INR)",
        min_value=1000,
        value=50000,
        step=5000
    )

    if total_spend > monthly_budget:
        st.error(f"⚠️ Budget Exceeded by ₹{total_spend - monthly_budget:,.0f}")
    else:
        st.success("✅ Spending is within budget")

    # ======================
    # Severity Scoring
    # ======================
    def calculate_severity(row):
        if row["Cost"] > avg_cost * 2 or row["Cost_Change"] > 0.6:
            return "High"
        elif row["Cost"] > avg_cost * 1.2 or row["Cost_Change"] > 0.3:
            return "Medium"
        return "Low"

    df["Severity"] = df.apply(calculate_severity, axis=1)

    # ======================
    # Explainable AI
    # ======================
    def explain_anomaly(row):
        if row["Final_Anomaly"] == 0:
            return "Normal spending pattern"
        if row["Cost_Change"] > 0.5:
            return "Sudden cost spike compared to previous usage"
        if row["Cost"] > avg_cost:
            return "Unusually high cost compared to historical average"
        return "Irregular spending pattern detected by AI models"

    df["Why_Anomaly"] = df.apply(explain_anomaly, axis=1)

    # ======================
    # Savings Priority
    # ======================
    def saving_priority(amount):
        if amount > avg_cost * 0.4:
            return "High Savings"
        elif amount > avg_cost * 0.2:
            return "Medium Savings"
        elif amount > 0:
            return "Low Savings"
        return "No Savings"

    df["Savings_Priority"] = df["Estimated_Saving_INR"].apply(saving_priority)

    # ======================
    # Top-N Cost Drivers
    # ======================
    st.subheader("🏆 Top Cost Drivers")

    top_n = st.slider("Select Top N Cost Drivers", 3, 10, 5)

    top_days = (
        df[df["Final_Anomaly"] == 1]
        .sort_values("Cost", ascending=False)
        .head(top_n)
    )

    st.dataframe(
        top_days[["Date", "Cost", "Estimated_Saving_INR"]],
        use_container_width=True
    )

    # ======================
    # Cost Trend
    # ======================
    st.subheader("📈 Cloud Cost Trend (Normal vs Anomalous)")

    fig, ax = plt.subplots(figsize=(12, 5))

    normal_df = df[df["Final_Anomaly"] == 0]
    anomaly_df = df[df["Final_Anomaly"] == 1]

    ax.plot(normal_df["Date"], normal_df["Cost"], color="green", label="Normal Cost")
    ax.scatter(anomaly_df["Date"], anomaly_df["Cost"], color="red", label="Anomaly", s=60)

    ax.set_ylabel("Cost (INR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ======================
    # 🔥 Recommendation-focused Highlighting
    # ======================
    st.subheader("💡 Cost-Saving Recommendations")

    display_df = df[df["Final_Anomaly"] == 1][[
        "Date",
        "Cost",
        "Estimated_Saving_INR",
        "Savings_Priority",
        "Severity",
        "Why_Anomaly",
        "Recommendation"
    ]].copy()

    def highlight_recommendation(row):
        if row["Savings_Priority"] == "High Savings":
            return [
                ""] * (len(row) - 1) + ["background-color: #2e7d32; color: white; font-weight: 600"]
        elif row["Savings_Priority"] == "Medium Savings":
            return [
                ""] * (len(row) - 1) + ["background-color: #f9a825; color: black; font-weight: 600"]
        elif row["Savings_Priority"] == "Low Savings":
            return [
                ""] * (len(row) - 1) + ["background-color: #c62828; color: white; font-weight: 600"]
        return [""] * len(row)

    styled_df = display_df.style.apply(highlight_recommendation, axis=1)

    st.dataframe(styled_df, use_container_width=True)

    st.markdown("""
    **Legend**
    - 🟢 Green → High savings action  
    - 🟡 Yellow → Medium savings action  
    - 🔴 Red → Low savings / monitor  
    """)

    # ======================
    # Executive Summary
    # ======================
    st.subheader("📄 Executive Summary")

    high_severity_count = (df["Severity"] == "High").sum()

    st.markdown(f"""
    - **Total Cloud Spend:** ₹{total_spend:,.0f}  
    - **Anomalous Spend:** ₹{anomaly_spend:,.0f}  
    - **Estimated Savings:** ₹{total_savings:,.0f}  
    - **High Severity Incidents:** {high_severity_count}  
    - **Primary Actions:** Rightsizing, Idle Resource Cleanup, Commitment Discounts  
    """)

    # ======================
    # Download Report
    # ======================
    st.subheader("⬇️ Download Anomaly Report")

    report_df = df[df["Final_Anomaly"] == 1]
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name="cloud_cost_anomaly_report.csv",
        mime="text/csv"
    )

    # ======================
    # Forecast
    # ======================
    st.subheader("🔮 Cost Forecast (Next 30 Days)")
    forecast = forecast_cost(df)
    st.line_chart(forecast)

else:
    st.info("⬆️ Upload your GCP billing CSV file to start analysis.")
