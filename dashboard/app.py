import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fee Defaulter Dashboard", layout="wide")

st.title("Smart Fee Defaulter Prediction Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("outputs/risk_report.csv")

df = load_data()

latest = df.groupby("student_id").last().reset_index()

st.sidebar.header("Filters")

classes = sorted(latest["class"].unique())

selected_classes = st.sidebar.multiselect(
    "Select Classes",
    classes
)

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    0,
    100,
    70
)

if selected_classes:
    latest = latest[latest["class"].isin(selected_classes)]

high_risk = latest[latest["risk_score"] >= risk_threshold]

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Risk Analysis",
    "Class/Section",
    "Trends"
])

with tab1:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(latest))
    c2.metric("High Risk Students", len(high_risk))
    c3.metric("Total Dues", f"₹{latest['fee_due'].sum():,.0f}")
    c4.metric("Expected Collection", f"₹{latest['fee_paid'].sum():,.0f}")

    fig = px.histogram(
        latest,
        x="risk_score",
        nbins=20,
        title="Risk Score Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader("Top High Risk Students")

    cols = [
        "student_name",
        "class",
        "section",
        "risk_score",
        "default_prob",
        "predicted_delay_days"
    ]

    st.dataframe(
        high_risk[cols]
        .sort_values("risk_score", ascending=False)
        .head(30),
        use_container_width=True
    )

with tab3:

    heatmap_data = (
        latest.groupby(["class", "section"])["default_prob"]
        .mean()
        .reset_index()
    )

    fig2 = px.density_heatmap(
        heatmap_data,
        x="class",
        y="section",
        z="default_prob",
        title="Default Probability Heatmap"
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab4:

    monthly = (
        df.groupby("month")
        .agg(
            dues=("fee_due", "sum"),
            collected=("fee_paid", "sum")
        )
        .reset_index()
    )

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["dues"],
            name="Dues"
        )
    )

    fig3.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["collected"],
            name="Collected"
        )
    )

    fig3.update_layout(
        title="Monthly Collection vs Dues"
    )

    st.plotly_chart(fig3, use_container_width=True)