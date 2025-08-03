import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
import re  # For regular expressions
from patsy import dmatrix  # For spline modeling

# App configuration
st.set_page_config(
    page_title="PCF Data Analysis and Visualization Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

def calculate_confidence_interval(data, confidence=0.99):
    """Calculate the confidence interval for a given dataset."""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - interval, mean + interval

def perform_t_test(data1, data2):
    """Perform a two-sample t-test and return the t-statistic and p-value."""
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
    return t_stat, p_value

def perform_anova(df, dependent_var, independent_var):
    """Perform one-way ANOVA and return the F-statistic and p-value."""
    groups = df[independent_var].unique()
    group_data = [df[df[independent_var] == group][dependent_var].dropna() for group in groups]
    f_stat, p_value = stats.f_oneway(*group_data)
    return f_stat, p_value

def perform_chi_square(df, var1, var2):
    """Perform Chi-Square test of independence and return the statistic and p-value."""
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    return chi2, p, dof, ex

def perform_linear_regression(df, dependent_var, independent_vars):
    """Perform linear regression and return the model and summary."""
    X = df[independent_vars].dropna()
    Y = df[dependent_var].dropna()
    # Align the indices
    X, Y = X.align(Y, join='inner', axis=0)
    model = LinearRegression()
    model.fit(X, Y)
    r_squared = model.score(X, Y)
    coefficients = pd.Series(model.coef_, index=independent_vars)
    intercept = model.intercept_
    return model, r_squared, coefficients, intercept

def perform_linear_regression_statsmodels(df, dependent_var, independent_vars):
    """Perform linear regression using statsmodels and return the summary."""
    formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
    model = ols(formula, data=df).fit()
    return model.summary()

def calculate_significance_across_categories(df, column):
    """Calculate pairwise t-test p-values for a specific column between all categories."""
    categories = df['category'].unique()
    p_values = {}
    
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i + 1:]:
            data_cat1 = df[df['category'] == cat1][column].dropna()
            data_cat2 = df[df['category'] == cat2][column].dropna()
            
            if not data_cat1.empty and not data_cat2.empty:
                _, p_value = perform_t_test(data_cat1, data_cat2)
                p_values[f"{cat1} vs {cat2}"] = p_value
    
    return p_values

def clean_column_names(df):
    """Standardize the column names by removing extra spaces, making lowercase, and replacing spaces/special characters with underscores."""
    df.columns = df.columns.str.strip().str.lower()
    # Replace spaces and non-alphanumeric characters with underscores
    df.columns = df.columns.str.replace(r'\W+', '_', regex=True)
    return df

@st.cache_data
def get_sampled_data(df, category, max_points):
    """Sample data for a specific category to improve performance."""
    category_data = df[df['category'] == category]
    if len(category_data) > max_points:
        return category_data.sample(n=max_points, random_state=42)
    return category_data

def main():
    st.title("PCA Data Analysis and Visualization Tool")
    st.markdown("**By David Pearl and Matthew Murphy**")

    # 1. File Upload
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type="csv")
    dataframes = []

    if uploaded_files:
        for file in uploaded_files:
            # Determine star category from file name (assuming file name contains star info)
            filename_lower = file.name.lower()
            if 'one_star' in filename_lower:
                category = '1-star'
            elif 'two_star' in filename_lower:
                category = '2-star'
            elif 'three_star' in filename_lower:
                category = '3-star'
            elif 'four_star' in filename_lower:
                category = '4-star'
            elif 'five_star' in filename_lower:
                category = '5-star'
            else:
                category = 'unknown'

            # Read each CSV file into a DataFrame
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue

            # Add a column for the star category
            df['category'] = category
            dataframes.append(df)
        
        # 2. Combine DataFrames
        with st.spinner("Combining and cleaning data..."):
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df = clean_column_names(combined_df)  # Clean column names

        # Display the column names for debugging
        st.write("### Column Names in Combined Data")
        st.write(combined_df.columns.tolist())

        # 3. Display Combined Data with Option to View All
        st.write("### Combined Data")
        
        # Provide an option to view all data or a subset
        view_all = st.checkbox("View all combined data (this may take a moment if the dataset is large)")

        if view_all:
            st.dataframe(combined_df)
        else:
            st.dataframe(combined_df.head(10))  # Show first 10 rows for better visibility

        # Update significant_columns to lowercase and sanitized
        significant_columns = ['total_time_per_meal', 'satisfaction_score']

        # 4. Calculate and Display Combined Statistics in Tables
        st.write("## Combined Statistics")
        numerical_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()

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
                    "Column": col.replace('_', ' ').capitalize(),
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
            st.dataframe(combined_stats_df)

            # Combined histograms for each numerical column
            for col in numerical_columns:
                st.write(f"#### Distribution of {col.replace('_', ' ').capitalize()}")
                fig = px.histogram(
                    combined_df,
                    x=col,
                    color='category',
                    barmode='overlay',
                    nbins=30,
                    title=f"Distribution of {col.replace('_', ' ').capitalize()} by Category",
                    labels={col: col.replace('_', ' ').capitalize(), 'count': 'Count'},
                    hover_data=['category']
                )
                fig.update_layout(
                    xaxis_title=col.replace('_', ' ').capitalize(),
                    yaxis_title="Count",
                    legend_title="Category",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        # 5. Calculate and Display Statistics by Category in Tables
        st.write("## Statistics by Category")
        categories = combined_df['category'].unique()

        for category in categories:
            st.write(f"### {category.capitalize()} Statistics")
            category_df = combined_df[combined_df['category'] == category]

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
                        "Column": col.replace('_', ' ').capitalize(),
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
                st.dataframe(category_stats_df)

        # 6. Statistical Significance Testing for specified columns
        st.write("## Statistical Significance Testing for 'Total Time Per Meal' and 'Satisfaction Score'")

        for col in significant_columns:
            st.write(f"### Statistical Significance for {col.replace('_', ' ').capitalize()}")
            if col in combined_df.columns:
                p_values = calculate_significance_across_categories(combined_df, col)
                if p_values:
                    p_values_df = pd.DataFrame(list(p_values.items()), columns=["Comparison", "P-Value"])
                    st.write("#### P-values for Pairwise Comparisons")
                    st.dataframe(p_values_df)

                    # Highlight significance
                    significance_text = []
                    for comparison, p_value in p_values.items():
                        if p_value < 0.05:
                            significance_text.append(f"**Significant difference** in {col.replace('_', ' ').capitalize()} between {comparison} (p = {p_value:.4f})")
                        else:
                            significance_text.append(f"No significant difference in {col.replace('_', ' ').capitalize()} between {comparison} (p = {p_value:.4f})")
                    
                    st.write("\n".join(significance_text))
                else:
                    st.write(f"No valid comparisons for {col}. Ensure that the data contains values for multiple categories.")
            else:
                st.write(f"'{col}' column not found in the data. Please check the uploaded files.")

        # 7. Advanced Statistical Tests
        st.write("## Advanced Statistical Tests")
        test_type = st.selectbox("Select a statistical test to perform", ["Linear Regression", "ANOVA", "Chi-Square"])

        if test_type == "Linear Regression":
            st.write("### Linear Regression Analysis")
            dependent_var_lr = st.selectbox("Select the dependent variable (Numerical)", numerical_columns, key='lr_dep')
            independent_vars_lr = st.multiselect("Select independent variable(s)", numerical_columns, key='lr_ind')
            if st.button("Run Linear Regression"):
                if dependent_var_lr and independent_vars_lr:
                    if dependent_var_lr in independent_vars_lr:
                        st.error("Dependent variable cannot be one of the independent variables.")
                    else:
                        with st.spinner("Performing Linear Regression..."):
                            try:
                                formula = f"{dependent_var_lr} ~ " + " + ".join(independent_vars_lr)
                                st.write(f"**Constructed Formula:** `{formula}`")
                                
                                model, r_sq, coefficients, intercept = perform_linear_regression(combined_df, dependent_var_lr, independent_vars_lr)
                                st.write(f"**R-squared:** {r_sq:.4f}")
                                st.write("**Intercept:**", intercept)
                                st.write("**Coefficients:**")
                                st.write(coefficients)
                                
                                # Display statsmodels summary
                                st.write("**Detailed Regression Summary:**")
                                summary = perform_linear_regression_statsmodels(combined_df, dependent_var_lr, independent_vars_lr)
                                st.text(summary)
                                
                                # Plot actual vs predicted
                                X = combined_df[independent_vars_lr].dropna()
                                Y = combined_df[dependent_var_lr].dropna()
                                X, Y = X.align(Y, join='inner', axis=0)
                                predictions = model.predict(X)
                                fig_lr = px.scatter(x=Y, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
                                fig_lr.add_shape(
                                    type='line',
                                    x0=Y.min(), y0=Y.min(),
                                    x1=Y.max(), y1=Y.max(),
                                    line=dict(color='Red', dash='dash')
                                )
                                st.plotly_chart(fig_lr, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating regression summary: {e}")
                else:
                    st.error("Please select both dependent and independent variables.")

        elif test_type == "ANOVA":
            st.write("### ANOVA (Analysis of Variance)")
            dependent_var_anova = st.selectbox("Select the dependent variable (Numerical)", numerical_columns, key='anova_dep')
            categorical_columns_for_anova = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_columns_for_anova = [col for col in categorical_columns_for_anova if col != 'category']  # Exclude 'category' if present
            independent_var_anova = st.selectbox("Select the independent variable (Categorical)", categorical_columns_for_anova, key='anova_ind')
            if st.button("Run ANOVA"):
                if dependent_var_anova and independent_var_anova:
                    with st.spinner("Performing ANOVA..."):
                        try:
                            f_stat, p_value = perform_anova(combined_df, dependent_var_anova, independent_var_anova)
                            st.write(f"**F-Statistic:** {f_stat:.4f}")
                            st.write(f"**P-Value:** {p_value:.4f}")
                            if p_value < 0.05:
                                st.write("**Result:** Significant differences exist between groups.")
                            else:
                                st.write("**Result:** No significant differences between groups.")
                            
                            # Boxplot to visualize
                            fig_anova = px.box(
                                combined_df, 
                                x=independent_var_anova, 
                                y=dependent_var_anova, 
                                color=independent_var_anova,
                                title=f"Boxplot of {dependent_var_anova.replace('_', ' ').capitalize()} by {independent_var_anova.replace('_', ' ').capitalize()}",
                                labels={independent_var_anova: independent_var_anova.replace('_', ' ').capitalize(), dependent_var_anova: dependent_var_anova.replace('_', ' ').capitalize()},
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_anova, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error performing ANOVA: {e}")
                else:
                    st.error("Please select both dependent and independent variables.")

        elif test_type == "Chi-Square":
            st.write("### Chi-Square Test of Independence")
            categorical_columns_for_chi = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_columns_for_chi = [col for col in categorical_columns_for_chi if col != 'category']  # Exclude 'category'
            var1_chi = st.selectbox("Select the first categorical variable", categorical_columns_for_chi, key='chi_var1')
            var2_chi = st.selectbox("Select the second categorical variable", categorical_columns_for_chi, key='chi_var2')
            if st.button("Run Chi-Square Test"):
                if var1_chi and var2_chi:
                    if var1_chi == var2_chi:
                        st.error("Please select two different variables.")
                    else:
                        with st.spinner("Performing Chi-Square Test..."):
                            try:
                                chi2, p, dof, ex = perform_chi_square(combined_df, var1_chi, var2_chi)
                                st.write(f"**Chi-Square Statistic:** {chi2:.4f}")
                                st.write(f"**Degrees of Freedom:** {dof}")
                                st.write(f"**P-Value:** {p:.4f}")
                                if p < 0.05:
                                    st.write("**Result:** Significant association between the variables.")
                                else:
                                    st.write("**Result:** No significant association between the variables.")
                                
                                # Display contingency table
                                st.write("**Contingency Table:**")
                                contingency_table = pd.crosstab(combined_df[var1_chi], combined_df[var2_chi])
                                st.dataframe(contingency_table)
                                
                                # Heatmap of contingency table
                                fig_chi = px.imshow(contingency_table, 
                                                    labels=dict(x=var2_chi.replace('_', ' ').capitalize(), y=var1_chi.replace('_', ' ').capitalize(), color="Count"),
                                                    x=contingency_table.columns,
                                                    y=contingency_table.index,
                                                    title=f"Heatmap of {var1_chi.replace('_', ' ').capitalize()} vs {var2_chi.replace('_', ' ').capitalize()}",
                                                    template="plotly_white")
                                st.plotly_chart(fig_chi, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error performing Chi-Square Test: {e}")
                else:
                    st.error("Please select both variables.")

        # 8. 2D Scatter Plot with Toggle (if applicable)
        if len(numerical_columns) >= 2:
            st.write("## 2D Scatter Plot")
            x_col_2d = st.selectbox("Select X axis for 2D scatter plot", numerical_columns, key='2d_x')
            y_col_2d = st.selectbox("Select Y axis for 2D scatter plot", numerical_columns, key='2d_y')
            coordinate_system = st.radio("Select coordinate system", ["Cartesian", "Polar"])

            if coordinate_system == "Cartesian":
                with st.spinner("Generating Cartesian Scatter Plot..."):
                    fig_2d = px.scatter(
                        combined_df,
                        x=x_col_2d,
                        y=y_col_2d,
                        color='category',
                        title=f"Scatter Plot of {x_col_2d.replace('_',' ').capitalize()} vs {y_col_2d.replace('_',' ').capitalize()}",
                        labels={x_col_2d: x_col_2d.replace('_',' ').capitalize(), y_col_2d: y_col_2d.replace('_',' ').capitalize()},
                        hover_data=['category'],
                        template="plotly_white"
                    )
                    fig_2d.update_layout(legend_title_text='Category')
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                st.write("### Polar Scatter Plot Optimization")
                MAX_POINTS = st.slider(
                    "Select maximum number of points to display per category",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100
                )

                with st.spinner("Generating Polar Scatter Plot..."):
                    fig_2d = go.Figure()
                    for category in categories:
                        category_data = get_sampled_data(combined_df, category, MAX_POINTS)
                        if len(combined_df[combined_df['category'] == category]) > MAX_POINTS:
                            st.write(f"**Sampling applied:** {category.capitalize()} category has more than {MAX_POINTS} points. Displaying {MAX_POINTS} randomly sampled points.")

                        fig_2d.add_trace(go.Scatterpolar(
                            r=category_data[y_col_2d],
                            theta=category_data[x_col_2d],
                            mode='markers',
                            name=category.capitalize(),
                            marker=dict(size=6, opacity=0.5)
                        ))
                    
                    if fig_2d.data:
                        fig_2d.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, title=y_col_2d.replace('_', ' ').capitalize())
                            ),
                            title=f"Polar Scatter Plot of {y_col_2d.replace('_', ' ').capitalize()} vs {x_col_2d.replace('_', ' ').capitalize()}",
                            showlegend=True,
                            template="plotly_white"
                        )
                    else:
                        st.warning("No data available to plot in Polar Scatter Plot.")
                        fig_2d = None

                if fig_2d:
                    st.plotly_chart(fig_2d, use_container_width=True)

        # 9. 3D Scatter Plot (if applicable)
        if len(numerical_columns) >= 3:
            st.write("## 3D Scatter Plot")
            x_col = st.selectbox("Select X axis for 3D scatter plot", numerical_columns, key='3d_x')
            y_col = st.selectbox("Select Y axis for 3D scatter plot", numerical_columns, key='3d_y')
            z_col = st.selectbox("Select Z axis for 3D scatter plot", numerical_columns, key='3d_z')

            with st.spinner("Generating 3D Scatter Plot..."):
                fig_3d = px.scatter_3d(
                    combined_df,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    color='category',
                    title=f"3D Scatter Plot of {x_col.replace('_', ' ').capitalize()}, {y_col.replace('_', ' ').capitalize()}, and {z_col.replace('_', ' ').capitalize()}",
                    labels={
                        x_col: x_col.replace('_', ' ').capitalize(),
                        y_col: y_col.replace('_', ' ').capitalize(),
                        z_col: z_col.replace('_', ' ').capitalize()
                    },
                    hover_data=['category'],
                    template="plotly_white"
                )
                fig_3d.update_layout(legend_title_text='Category')
            st.plotly_chart(fig_3d, use_container_width=True)

        # Convert category to a numeric Star_Level for non-linear modeling
        if 'category' in combined_df.columns:
            combined_df['Star_Level'] = combined_df['category'].str.extract(r'(\d)-star').astype(float)
            # Drop rows without a valid star level if necessary
            combined_df = combined_df.dropna(subset=['Star_Level'])
            combined_df['Star_Level'] = combined_df['Star_Level'].astype(int)

        # Non-Linear Modeling (Spline Regression)
        st.write("## Non-Linear Modeling (Spline Regression)")
        st.markdown("""
        Here we explore non-linear relationships using spline regressions. 
        Select a dependent and predictor variable, along with the star level variable,
        to see how satisfaction changes non-linearly with meal time.
        """)

        # Check conditions for non-linear modeling
        if len(numerical_columns) > 1 and 'Star_Level' in combined_df.columns:
            dependent_var_nl = st.selectbox("Select the dependent variable (Numerical, e.g. satisfaction_score)", numerical_columns, key='nl_dep')
            predictor_var_nl = st.selectbox("Select the predictor variable (Numerical, e.g. total_time_per_meal)", numerical_columns, key='nl_pred')
            
            downsample = st.checkbox("Downsample data for spline modeling (recommended if dataset is large)", value=True)
            downsample_size = st.number_input("Downsample size", min_value=1000, max_value=200000, value=10000, step=1000) if downsample else None
            
            if st.button("Run Non-Linear Spline Regression"):
                if dependent_var_nl and predictor_var_nl and dependent_var_nl != predictor_var_nl:
                    df_for_nl = combined_df.copy()
                    if downsample and len(df_for_nl) > downsample_size:
                        df_for_nl = df_for_nl.sample(n=downsample_size, random_state=42)
                    
                    if 'Star_Level' not in df_for_nl.columns:
                        st.error("Star_Level column not found. Please ensure categories are correctly parsed.")
                    else:
                        try:
                            spline_formula = f"{dependent_var_nl} ~ Star_Level + bs({predictor_var_nl}, degree=3, df=5)"
                            nonlinear_model = ols(spline_formula, data=df_for_nl).fit()
                            
                            st.write("### Non-Linear Model Summary")
                            st.text(nonlinear_model.summary())
                            
                            time_range = np.linspace(df_for_nl[predictor_var_nl].min(),
                                                     df_for_nl[predictor_var_nl].max(), 100)
                            star_levels = sorted(df_for_nl['Star_Level'].unique())
                            
                            fig_spline = go.Figure()
                            for sl in star_levels:
                                pred_df = pd.DataFrame({
                                    'Star_Level': sl,
                                    predictor_var_nl: time_range
                                })
                                predictions = nonlinear_model.predict(pred_df)
                                fig_spline.add_trace(go.Scatter(
                                    x=time_range,
                                    y=predictions,
                                    mode='lines',
                                    name=f"{sl}-Star"
                                ))
                            
                            fig_spline.update_layout(
                                title=f"Non-Linear Effects of {predictor_var_nl.replace('_',' ').capitalize()} on {dependent_var_nl.replace('_',' ').capitalize()} by Star Level",
                                xaxis_title=predictor_var_nl.replace('_',' ').capitalize(),
                                yaxis_title=dependent_var_nl.replace('_',' ').capitalize(),
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig_spline, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error fitting non-linear model: {e}")
                else:
                    st.error("Please select distinct dependent and predictor variables.")
        else:
            st.write("Not enough numerical columns or Star_Level data available for non-linear modeling.")

if __name__ == '__main__':
    main()

