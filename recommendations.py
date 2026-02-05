import numpy as np

def generate_recommendations(df):
    recommendations = []
    saving_amounts = []

    avg_cost = df["Cost"].mean()
    avg_usage = df["Usage"].mean()

    for i, row in df.iterrows():
        if row["Final_Anomaly"] == 1:

            # Rule 1: Sudden spike
            if row["Cost_Change"] > 0.3:
                rec = (
                    "Sudden cost spike detected. "
                    "Review recent VM scaling events, autoscaling policies, "
                    "or newly deployed services."
                )
                saving = row["Cost"] * 0.30

            # Rule 2: High cost but low usage
            elif row["Cost"] > avg_cost and row["Usage"] < avg_usage:
                rec = (
                    "High cost with low usage detected. "
                    "Possible idle or over-provisioned resources. "
                    "Consider shutting down unused VMs or downsizing instances."
                )
                saving = row["Cost"] * 0.50

            # Rule 3: Repeated anomalies (commitment missing)
            elif (
                df["Final_Anomaly"].iloc[max(0, i-3):i].sum() >= 2
            ):
                rec = (
                    "Recurring cost anomalies detected. "
                    "Consider Committed Use Discounts or long-term reservations."
                )
                saving = row["Cost"] * 0.35

            # Rule 4: General anomaly
            else:
                rec = (
                    "Abnormal cost behavior detected. "
                    "Audit service-level usage, storage growth, "
                    "and data egress charges."
                )
                saving = row["Cost"] * 0.25

        else:
            rec = "Normal usage pattern detected."
            saving = 0.0

        recommendations.append(rec)
        saving_amounts.append(saving)

    df["Recommendation"] = recommendations
    df["Estimated_Saving_INR"] = saving_amounts

    return df
