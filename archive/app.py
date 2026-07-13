import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# --- Absolute paths anchored to this script's location ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # …/archive
DATA_PATH  = os.path.join(BASE_DIR, "train.csv")                 # …/archive/train.csv
CHARTS_DIR = os.path.join(BASE_DIR, "..", "charts")              # …/charts
CHARTS_DIR = os.path.normpath(CHARTS_DIR)                        # clean up the ".."

# --- Page Configuration ---
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide", page_icon="📊")

# --- Helper: load image by filename from CHARTS_DIR ---
def render_image(filename, caption=""):
    path = os.path.join(CHARTS_DIR, filename)
    if os.path.isfile(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.error(f"Image not found: {path}")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year']  = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Dashboard:",
                        ["Sales Overview (EDA)", "Forecast Explorer",
                         "Anomaly Report", "Product Segments"])
st.sidebar.markdown("---")
st.sidebar.info("End-to-End Sales Forecasting & Demand Intelligence System")

# ==========================================
# PAGE 1: SALES OVERVIEW (EDA)
# ==========================================
if page == "Sales Overview (EDA)":
    st.title("📈 Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into historical sales performance, regional growth, and seasonal trends.")

    # KPIs
    total_sales  = df['Sales'].sum()
    total_orders = df['Order ID'].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (4 Years)", f"${total_sales:,.0f}")
    col2.metric("Total Unique Orders",     f"{total_orders:,}")
    col3.metric("Average Order Value",     f"${total_sales/total_orders:,.2f}")
    st.markdown("---")

    # Interactive filters
    col_f1, col_f2 = st.columns(2)
    selected_region   = col_f1.selectbox("Filter by Region:",   ["All"] + sorted(df['Region'].unique().tolist()))
    selected_category = col_f2.selectbox("Filter by Category:", ["All"] + sorted(df['Category'].unique().tolist()))

    filtered_df = df.copy()
    if selected_region   != "All": filtered_df = filtered_df[filtered_df['Region']   == selected_region]
    if selected_category != "All": filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # Live charts from data
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Total Sales by Year")
        yearly = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=yearly, x='Year', y='Sales', palette='Blues_d', ax=ax)
        ax.set_ylabel("Revenue ($)")
        st.pyplot(fig)
        plt.close(fig)

    with r2:
        st.subheader("Monthly Sales Trend")
        monthly = filtered_df.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
        monthly.columns = ['Date', 'Sales']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=monthly, x='Date', y='Sales', color='teal', linewidth=2, ax=ax)
        ax.set_ylabel("Revenue ($)")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("📊 Pre-Generated Analysis Charts")
    tab1, tab2, tab3 = st.tabs(["Overall Trend", "Seasonality", "Category Revenue"])

    with tab1:
        st.subheader("4-Year Macro Trend")
        render_image("overall_monthly_sales_trend(4_years).png",
                     "Steady upward growth trajectory from 2015 to 2018.")

    with tab2:
        st.subheader("Monthly Seasonality Spikes")
        render_image("average_sales_by_month_(seasonality).png",
                     "Clear Q4 spikes (Nov/Dec) and Q1 dips (Feb).")

    with tab3:
        st.subheader("Revenue by Product Category")
        render_image("total_revenue_by_product_category.png",
                     "Technology dominates overall revenue volume.")

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif page == "Forecast Explorer":
    st.title("🔮 Time-Series Forecasting")
    st.markdown("Comparing statistical models against machine learning algorithms to predict Q1 2019.")

    tab1, tab2, tab3 = st.tabs(["Time Series Decomposition", "Model Comparison", "Segment Forecasts"])

    with tab1:
        st.subheader("Deconstructing the Signal")
        st.markdown("Broke the data into core components to prove stationarity before modelling.")
        render_image("time_series_destribution.png",
                     "Trend, Seasonality, and Residual Noise.")

    with tab2:
        st.subheader("SARIMA vs. XGBoost")
        col1, col2 = st.columns(2)
        with col1:
            st.info("🏆 Winning Model: SARIMA — MAE: 20,632 | RMSE: 22,313 | MAPE: 21.96%")
            render_image("SARIMA_Model_Future_Forecast_(Q1_2019).png",
                         "SARIMA captures the seasonal Q1 drop accurately.")
        with col2:
            st.error("❌ Baseline: XGBoost — RMSE: 26,994")
            render_image("XGBoost_Model_Future_Forecast_(Q1_2019).png",
                         "Tree-based models struggle with pure extrapolation.")

    with tab3:
        st.subheader("Segment-Level Predictions (SARIMA)")
        render_image("SARIMA_3-Month_Forecast_Comparison_by_Segment_(Q1_2019).png",
                     "Technology and West Region project the highest recovery in March.")

# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif page == "Anomaly Report":
    st.title("🚨 Sales Anomaly Detection")
    st.markdown("Comparing ML (Isolation Forest) vs Statistical (Z-Score) outlier detection on weekly sales.")

    render_image("Anomaly_Detection_Comparison.png",
                 "Black Friday and Back-to-School spikes heavily flagged by both systems.")

    st.markdown("""
    **Business Context:**
    * **Isolation Forest:** Flagged 11 weeks — great for finding absolute seasonal volume ceilings.
    * **Z-Score (Rolling):** Flagged 0 weeks with strict thresholds, confirming most spikes built
      up gradually over the preceding month rather than being sudden shocks.
    """)

# ==========================================
# PAGE 4: PRODUCT SEGMENTS
# ==========================================
elif page == "Product Segments":
    st.title("📦 Product Demand Segments")
    st.markdown("K-Means Clustering + PCA grouping products by Volume, Growth, Volatility, and Order Value.")

    tab1, tab2 = st.tabs(["2D Cluster Map (PCA)", "Elbow Method"])

    with tab1:
        render_image("Product_Segmentation_Clusters.png",
                     "Products grouped into 4 distinct supply chain strategies.")
        st.markdown("""
        **Recommended Strategies:**
        * **Cluster 0 (Rising Stars):** Low ticket, high growth — increase safety stock.
        * **Cluster 2 (High-Ticket Volatile):** e.g. Copiers — implement Just-In-Time (JIT).
        * **Cluster 3 (Cash Cows):** e.g. Phones, Chairs — automate bulk reorders.
        """)

    with tab2:
        st.subheader("Mathematical Validation for K = 4")
        render_image("Elbow_Method_for_Optimal_K.png",
                     "Inertia curve flattens significantly after 4 clusters.")
