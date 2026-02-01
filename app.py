import streamlit as st
import pandas as pd
import numpy as np
from utils.dataa import (load_data, get_basic_info, get_missing_values,
                               get_numeric_summary, get_categorical_summary, detect_outliers,
                               get_date_columns, get_date_range)
from utils.visual import (plot_missing_values, plot_distribution, plot_boxplot,
                              plot_correlation_heatmap, plot_categorical_bars,
                              plot_scatter, plot_pairplot_interactive, plot_time_series,
                              plot_sales_by_category)

# Page configuration
st.set_page_config(
    page_title="Smart EDA Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Smart EDA Analyzer</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file",
                                     type=['csv', 'xlsx', 'xls'])

    st.markdown("---")
    st.info("💡 **Tip**: Upload karne ke baad automatic analysis shuru ho jayega!")

# Main app logic
if uploaded_file is not None:
    # Load data
    with st.spinner('Loading data...'):
        df = load_data(uploaded_file)

    if df is not None:
        st.success("✅ Data successfully loaded!")

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Overview",
            "📊 Statistical Summary",
            "🔍 Missing Values",
            "📈 Visualizations",
            "🔗 Relationships",
            "📅 Time Analysis"
        ])

        # TAB 1: Overview
        with tab1:
            st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)

            # Basic info
            info = get_basic_info(df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{info['rows']:,}")
            with col2:
                st.metric("Total Columns", info['columns'])
            with col3:
                st.metric("Memory Usage", f"{info['memory_usage']:.2f} MB")
            with col4:
                numeric_count = df.select_dtypes(include=[np.number]).shape[1]
                st.metric("Numeric Columns", numeric_count)

            st.markdown("---")

            # Data preview
            st.subheader("📄 Data Preview (First 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)

            # Column info
            st.subheader("📝 Column Information")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)

        # TAB 2: Statistical Summary
        with tab2:
            st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)

            # Numeric summary
            numeric_summary = get_numeric_summary(df)
            if numeric_summary is not None:
                st.subheader("🔢 Numeric Columns Summary")
                st.dataframe(numeric_summary, use_container_width=True)
            else:
                st.warning("No numeric columns found in the dataset.")

            # Categorical summary
            cat_summary = get_categorical_summary(df)
            if cat_summary:
                st.subheader("📝 Categorical Columns Summary")
                for col, data in cat_summary.items():
                    with st.expander(f"**{col}** - {data['unique_values']} unique values"):
                        st.write("Top 5 values:")
                        st.write(pd.DataFrame(list(data['top_5_values'].items()),
                                              columns=['Value', 'Count']))

        # TAB 3: Missing Values
        with tab3:
            st.markdown('<p class="sub-header">Missing Values Analysis</p>', unsafe_allow_html=True)

            missing_df = get_missing_values(df)

            if not missing_df.empty:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.dataframe(missing_df, use_container_width=True)

                with col2:
                    fig = plot_missing_values(missing_df)
                    if fig:
                        st.pyplot(fig)
            else:
                st.success("🎉 No missing values found in the dataset!")

        # TAB 4: Visualizations
        with tab4:
            st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Numeric column visualizations
            if numeric_cols:
                st.subheader("📊 Numeric Column Analysis")
                selected_num_col = st.selectbox("Select a numeric column", numeric_cols)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Distribution Plot**")
                    fig1 = plot_distribution(df, selected_num_col)
                    st.pyplot(fig1)

                with col2:
                    st.write("**Box Plot (Outlier Detection)**")
                    fig2 = plot_boxplot(df, selected_num_col)
                    st.pyplot(fig2)
                # Outlier info
                outlier_count, lower, upper = detect_outliers(df, selected_num_col)
                st.info(f"📍 **Outliers detected**: {outlier_count} | Range: [{lower:.2f}, {upper:.2f}]")

            # Categorical column visualizations
            if categorical_cols:
                st.markdown("---")
                st.subheader("📝 Categorical Column Analysis")
                selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)

                top_n = st.slider("Show top N values", 5, 20, 10)
                fig3 = plot_categorical_bars(df, selected_cat_col, top_n)
                st.pyplot(fig3)

            # Correlation heatmap
            if len(numeric_cols) >= 2:
                st.markdown("---")
                st.subheader("🔥 Correlation Heatmap")
                fig4 = plot_correlation_heatmap(df)
                if fig4:
                    st.pyplot(fig4)

        # TAB 5: Relationships
        with tab5:
            st.markdown('<p class="sub-header">Variable Relationships</p>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) >= 2:
                st.subheader("🔗 Scatter Plot")
                col1, col2 = st.columns(2)

                with col1:
                    x_var = st.selectbox("Select X-axis variable", numeric_cols)
                with col2:
                    y_var = st.selectbox("Select Y-axis variable",
                                         [col for col in numeric_cols if col != x_var])

                fig_scatter = plot_scatter(df, x_var, y_var)
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Pairplot
                if len(numeric_cols) >= 3:
                    st.markdown("---")
                    st.subheader("📊 Interactive Pairplot")
                    selected_cols = st.multiselect(
                        "Select columns for pairplot (3-5 recommended)",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )

                    if len(selected_cols) >= 2:
                        with st.spinner('Creating pairplot...'):
                            fig_pair = plot_pairplot_interactive(df, selected_cols)
                            st.plotly_chart(fig_pair, use_container_width=True)
            else:
                st.warning("At least 2 numeric columns are required for relationship analysis.")

    else:
        st.error("❌ Error loading file. Please check the file format.")

else:
    # Welcome screen
    st.info("👈 Please upload a CSV or Excel file from the sidebar to begin analysis.")

    # Features
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Data Overview**")
        st.write("• Dataset dimensions")
        st.write("• Column types")
        st.write("• Memory usage")

    with col2:
        st.markdown("**📈 Visualizations**")
        st.write("• Distribution plots")
        st.write("• Correlation heatmaps")
        st.write("• Box plots")

    with col3:
        st.markdown("**🔍 Analysis**")
        st.write("• Missing values")
        st.write("• Statistical summary")
        st.write("• Outlier detection")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with ❤️ using Streamlit | Made by You</div>",
    unsafe_allow_html=True
)