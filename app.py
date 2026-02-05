import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import load_and_preprocess
from anomaly_models import detect_anomalies
from forecasting import forecast_cost
from recommendations import generate_recommendations
from metrics import compute_kpis

st.set_page_config(
    page_title="Cloud Cost Anomaly Detection",
    layout="wide"
)

st.title("☁️ Cloud Cost Anomaly Detection & Optimization Dashboard")
st.caption("AI-driven analysis of GCP billing data")

uploaded = st.file_uploader("📤 Upload GCP Billing CSV", type=["csv"])

if uploaded:
    df = load_and_preprocess(uploaded)
    df = detect_anomalies(df)
    df = generate_recommendations(df)


    # KPI SECTION
    
    total_spend, anomaly_spend, potential_savings = compute_kpis(df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="💰 Total Cloud Spend (INR)",
        value=f"₹ {total_spend:,.0f}"
    )

    col2.metric(
        label="🚨 Spend in Anomalies (INR)",
        value=f"₹ {anomaly_spend:,.0f}",
        delta=f"{(anomaly_spend/total_spend)*100:.1f}% of total"
    )

    col3.metric(
        label="💡 Potential Monthly Savings",
        value=f"₹ {potential_savings:,.0f}"
    )

    st.divider()

    # COST + ANOMALY GRAPH

    st.subheader(" Cloud Cost Trend with Detected Anomalies")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Cost"], label="Daily Cost", linewidth=2)

    ax.scatter(
        df[df["Final_Anomaly"] == 1]["Date"],
        df[df["Final_Anomaly"] == 1]["Cost"],
        color="red",
        label="Anomalies",
        s=60
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cost (INR)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


    # ANOMALY COST BREAKDOWN

    st.subheader(" Where the Money Was Wasted")

    anomaly_df = df[df["Final_Anomaly"] == 1].copy()
    anomaly_df["Estimated_Savable"] = anomaly_df["Cost"] * 0.30

    st.dataframe(
        anomaly_df[[
            "Date",
            "Cost",
            "Estimated_Savable",
            "Recommendation"
        ]],
        use_container_width=True
    )

  
    # SAVINGS VISUALIZATION
    st.subheader(" Cost vs Potential Savings")

    savings_df = pd.DataFrame({
        "Category": ["Normal Spend", "Potentially Savable"],
        "Amount": [
            total_spend - potential_savings,
            potential_savings
        ]
    })

    fig2, ax2 = plt.subplots()

    ax2.bar(
        savings_df["Category"],
        savings_df["Amount"]
    )

    ax2.set_ylabel("Amount (INR)")
    ax2.set_title("Cloud Spend Optimization Opportunity")

    
    ax2.ticklabel_format(style='plain', axis='y')

    st.pyplot(fig2)

    # FORECAST
   
    st.subheader(" Future Cost Forecast (Next 30 Days)")
    forecast = forecast_cost(df)
    st.line_chart(forecast)

else:
    st.info("⬆️ Upload your GCP billing CSV to start analysis")
