import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# App configuration
st.set_page_config(page_title="Sonaura Demo App", layout="wide")

def calculate_confidence_interval(data, confidence=0.99):
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean - interval, mean + interval

def perform_t_test(data1, data2):
    """ Perform two-sample t-test """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
    return t_stat, p_value

# Streamlit Application
def main():
    st.title("PCA Data Analysis and Visualization Tool")
    st.text("By David Pearl and Matthew Murphy")

    # 1. File Upload
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type="csv")
    dataframes = []

    if uploaded_files:
        for file in uploaded_files:
            # Determine star category from file name (assuming file name contains star info)
            if 'one_star' in file.name:
                category = '1-star'
            elif 'two_star' in file.name:
                category = '2-star'
            elif 'three_star' in file.name:
                category = '3-star'
            elif 'four_star' in file.name:
                category = '4-star'
            elif 'five_star' in file.name:
                category = '5-star'
            else:
                category = 'Unknown'

            # Read each CSV file into a DataFrame
            df = pd.read_csv(file)
            # Add a column for the star category
            df['Category'] = category
            dataframes.append(df)
        
        # 2. Combine DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        st.write("### Combined Data")
        st.write(combined_df)

        # 3. Calculate and Display Combined Statistics in Tables
        st.write("## Combined Statistics")
        numerical_columns = combined_df.select_dtypes(include=[np.number]).columns

        combined_stats = []

        for col in numerical_columns:
            col_data = combined_df[col].dropna()

            if not col_data.empty:
                mean_value = np.mean(col_data)
                median_value = np.median(col_data)
                min_value = np.min(col_data)
                max_value = np.max(col_data)
                std_dev_value = np.std(col_data)
                std_error_value = stats.sem(col_data)
                confidence_interval_99 = calculate_confidence_interval(col_data)

                combined_stats.append({
                    "Column": col,
                    "Mean": mean_value,
                    "Median": median_value,
                    "Min": min_value,
                    "Max": max_value,
                    "Std Dev": std_dev_value,
                    "Std Error": std_error_value,
                    "99% CI Lower": confidence_interval_99[0],
                    "99% CI Upper": confidence_interval_99[1]
                })

        # Display combined statistics in a table
        if combined_stats:
            combined_stats_df = pd.DataFrame(combined_stats)
            st.write("### Combined Statistics Table")
            st.table(combined_stats_df)

            # Combined histograms for each numerical column
            for col in numerical_columns:
                st.write(f"#### Distribution of {col}")
                fig = px.histogram(combined_df, x=col, color='Category', barmode='overlay')
                st.plotly_chart(fig)

        # 4. Calculate and Display Statistics by Category in Tables
        st.write("## Statistics by Category")
        categories = combined_df['Category'].unique()
        
        for category in categories:
            st.write(f"### {category} Statistics")
            category_df = combined_df[combined_df['Category'] == category]

            category_stats = []

            for col in numerical_columns:
                col_data = category_df[col].dropna()

                if not col_data.empty:
                    mean_value = np.mean(col_data)
                    median_value = np.median(col_data)
                    min_value = np.min(col_data)
                    max_value = np.max(col_data)
                    std_dev_value = np.std(col_data)
                    std_error_value = stats.sem(col_data)
                    confidence_interval_99 = calculate_confidence_interval(col_data)

                    category_stats.append({
                        "Column": col,
                        "Mean": mean_value,
                        "Median": median_value,
                        "Min": min_value,
                        "Max": max_value,
                        "Std Dev": std_dev_value,
                        "Std Error": std_error_value,
                        "99% CI Lower": confidence_interval_99[0],
                        "99% CI Upper": confidence_interval_99[1]
                    })

            # Display category statistics in a table
            if category_stats:
                category_stats_df = pd.DataFrame(category_stats)
                st.table(category_stats_df)

        # 5. Perform Statistical Significance Tests (T-test)
        st.write("## Statistical Significance Testing (T-Test)")

        # Select categories for comparison
        if len(categories) > 1:
            category_1 = st.selectbox("Select first category for comparison", categories, key='cat1')
            category_2 = st.selectbox("Select second category for comparison", categories, key='cat2')

            # Select numerical column for comparison
            col_for_ttest = st.selectbox("Select column for t-test", numerical_columns, key='ttest_col')

            # Extract data for the selected categories
            data_cat1 = combined_df[combined_df['Category'] == category_1][col_for_ttest].dropna()
            data_cat2 = combined_df[combined_df['Category'] == category_2][col_for_ttest].dropna()

            if not data_cat1.empty and not data_cat2.empty:
                t_stat, p_value = perform_t_test(data_cat1, data_cat2)
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"P-value: {p_value:.4f}")

                # Display result of significance test
                if p_value < 0.05:
                    st.write(f"Result: The difference in {col_for_ttest} between {category_1} and {category_2} is statistically significant.")
                else:
                    st.write(f"Result: No significant difference in {col_for_ttest} between {category_1} and {category_2}.")

        # 6. 2D Scatter Plot with Toggle (if applicable)
        if len(numerical_columns) >= 2:
            st.write("## 2D Scatter Plot")
            x_col_2d = st.selectbox("Select X axis for 2D scatter plot", numerical_columns, key='2d_x')
            y_col_2d = st.selectbox("Select Y axis for 2D scatter plot", numerical_columns, key='2d_y')
            coordinate_system = st.radio("Select coordinate system", ["Cartesian", "Polar"])

            if coordinate_system == "Cartesian":
                # Cartesian scatter plot
                fig_2d = px.scatter(combined_df, x=x_col_2d, y=y_col_2d, color='Category')
            else:
                # Polar scatter plot
                fig_2d = go.Figure()
                for category in categories:
                    category_data = combined_df[combined_df['Category'] == category]
                    fig_2d.add_trace(go.Scatterpolar(
                        r=category_data[y_col_2d],
                        theta=category_data[x_col_2d],
                        mode='markers',
                        name=category
                    ))
                fig_2d.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True)
                    ),
                    showlegend=True
                )

            st.plotly_chart(fig_2d)

        # 7. 3D Scatter Plot (if applicable)
        if len(numerical_columns) >= 3:
            st.write("## 3D Scatter Plot")
            x_col = st.selectbox("Select X axis for 3D scatter plot", numerical_columns, key='3d_x')
            y_col = st.selectbox("Select Y axis for 3D scatter plot", numerical_columns, key='3d_y')
            z_col = st.selectbox("Select Z axis for 3D scatter plot", numerical_columns, key='3d_z')

            fig_3d = px.scatter_3d(combined_df, x=x_col, y=y_col, z=z_col, color='Category')
            st.plotly_chart(fig_3d)

if __name__ == '__main__':
    main()
