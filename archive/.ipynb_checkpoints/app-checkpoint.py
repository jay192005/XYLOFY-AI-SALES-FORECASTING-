import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide", page_icon="📊")

# --- Data Loading (Cached for speed) ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Dashboard:", 
                        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

st.sidebar.markdown("---")
st.sidebar.info("End-to-End Sales Forecasting & Demand Intelligence System")

# --- Page 1: Sales Overview Dashboard ---
if page == "Sales Overview":
    st.title("📈 Sales Overview Dashboard")
    st.markdown("Explore historical sales data across regions, categories, and time.")
    
    # KPIs
    total_sales = df['Sales'].sum()
    total_orders = df['Order ID'].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (4 Years)", f"${total_sales:,.0f}")
    col2.metric("Total Unique Orders", f"{total_orders:,}")
    col3.metric("Average Order Value", f"${total_sales/total_orders:,.2f}")
    
    st.markdown("---")
    
    # Interactive Filters
    col_f1, col_f2 = st.columns(2)
    selected_region = col_f1.selectbox("Filter by Region:", ["All"] + list(df['Region'].unique()))
    selected_category = col_f2.selectbox("Filter by Category:", ["All"] + list(df['Category'].unique()))
    
    # Apply Filters
    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        
    # Visualizations
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.subheader("Total Sales by Year")
        yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=yearly_sales, x='Year', y='Sales', palette='Blues_d', ax=ax1)
        st.pyplot(fig1)
        
    with fig_col2:
        st.subheader("Monthly Sales Trend")
        monthly_trend = filtered_df.set_index('Order Date').resample('ME')['Sales'].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(x=monthly_trend.index, y=monthly_trend.values, color='teal', linewidth=2, ax=ax2)
        ax2.set_ylabel("Sales ($)")
        st.pyplot(fig2)

# --- Page 2: Forecast Explorer ---
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer (SARIMA)")
    st.markdown("View the 3-month projected growth trajectory for our key business segments based on the winning SARIMA statistical model.")
    
    st.info("Winning Model Metrics: MAE: 20,632 | RMSE: 22,313 | MAPE: 21.96%")
    
    try:
        image = Image.open("charts/SARIMA_3-Month_Forecast_Comparison_by_Segment_(Q1_2019).png")
        st.image(image, caption="SARIMA Segment Forecast (Q1 2019)", use_container_width=True)
    except FileNotFoundError:
        st.error("Chart image not found. Ensure 'SARIMA_3-Month_Forecast_Comparison_by_Segment_(Q1_2019).png' is saved in the 'charts' folder.")

# --- Page 3: Anomaly Report ---
elif page == "Anomaly Report":
    st.title("🚨 Sales Anomaly Detection")
    st.markdown("Comparing Machine Learning (Isolation Forest) vs Statistical (Z-Score) outlier detection to identify irregular sales weeks.")
    
    try:
        image = Image.open("charts/Anomaly_Detection_Comparison.png")
        st.image(image, caption="Anomaly Detection Scatter Plot", use_container_width=True)
    except FileNotFoundError:
        st.error("Chart image not found. Ensure 'Anomaly_Detection_Comparison.png' is saved in the 'charts' folder.")
        
    st.markdown("""
    **Key Findings:**
    * **Black Friday/Holidays:** Massive expected volume spikes successfully captured.
    * **Back to School:** Mid-September corporate & academic purchasing rushes flagged across both models.
    """)

# --- Page 4: Product Demand Segments ---
elif page == "Product Segments":
    st.title("📦 Product Demand Segments (K-Means)")
    st.markdown("AI-driven inventory strategy grouping sub-categories by Volume, Growth, Volatility, and Order Value.")
    
    try:
        image = Image.open("charts/Product_Segmentation_Clusters.png")
        st.image(image, caption="PCA 2D Cluster Visualization", use_container_width=True)
    except FileNotFoundError:
        st.error("Chart image not found. Ensure 'Product_Segmentation_Clusters.png' is saved in the 'charts' folder.")
        
    st.markdown("""
    **Recommended Stocking Strategies:**
    * **High Volume, Stable:** Automate bulk reorders.
    * **Low Volume, Volatile:** Just-In-Time (JIT) minimal inventory.
    * **High Growth:** Increase safety stock allocations.
    """)