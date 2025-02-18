# PCF Data Analysis and Visualization Tool (formerly PCA in code title)

**By David Pearl and Matthew Murphy**

## Overview

This Streamlit application is designed to facilitate the analysis and visualization of data, particularly focusing on datasets categorized by star ratings (e.g., 1-star to 5-star). It allows users to upload CSV files, perform various statistical analyses, and generate insightful visualizations to understand data distributions, relationships, and significant differences across categories.

## Features

This tool offers a range of features for data exploration and statistical analysis:

*   **Data Upload and Combination:**
    *   Upload multiple CSV files simultaneously.
    *   Automatically categorize data based on filenames (e.g., files named `one_star.csv`, `two_star.csv` will be categorized as '1-star', '2-star', etc.).
    *   Combines uploaded data into a single DataFrame for unified analysis.
    *   Provides an option to view the full combined dataset or a preview.

*   **Data Cleaning and Preprocessing:**
    *   Standardizes column names by converting them to lowercase, removing extra spaces, and replacing spaces and special characters with underscores.

*   **Descriptive Statistics:**
    *   Calculates and displays combined descriptive statistics (Mean, Median, Min, Max, Standard Deviation, Standard Error, and 99% Confidence Intervals) for all numerical columns.
    *   Generates histograms to visualize the distribution of numerical columns, broken down by category.
    *   Provides category-specific descriptive statistics tables for each star rating.

*   **Statistical Significance Testing:**
    *   Performs pairwise t-tests to assess the statistical significance of differences in specified numerical columns (e.g., 'total_time_per_meal', 'satisfaction_score') between different categories.
    *   Highlights statistically significant differences based on p-values.

*   **Advanced Statistical Tests:**
    *   **Linear Regression:**
        *   Performs linear regression analysis using both `scikit-learn` and `statsmodels` for detailed summaries.
        *   Allows users to select dependent and independent numerical variables.
        *   Displays R-squared, intercept, coefficients, and a comprehensive summary from `statsmodels`.
        *   Generates scatter plots of actual vs. predicted values.
    *   **ANOVA (Analysis of Variance):**
        *   Performs one-way ANOVA to compare means across different categories for a selected numerical dependent variable and a categorical independent variable.
        *   Displays F-statistic, p-value, and interpretation of significance.
        *   Generates box plots to visualize the distribution of the dependent variable across categories.
    *   **Chi-Square Test of Independence:**
        *   Performs Chi-Square test to assess the independence of two categorical variables.
        *   Displays Chi-Square statistic, degrees of freedom, p-value, and interpretation of significance.
        *   Presents a contingency table and a heatmap for visual representation of the relationship.

*   **Interactive Visualizations:**
    *   **2D Scatter Plots:**
        *   Generates Cartesian and Polar scatter plots for visualizing relationships between two numerical variables.
        *   Option to toggle between Cartesian and Polar coordinate systems.
        *   Polar scatter plots include data point sampling for performance optimization with large datasets.
    *   **3D Scatter Plots:**
        *   Creates 3D scatter plots to explore relationships between three numerical variables, colored by category.

*   **Non-Linear Modeling (Spline Regression):**
    *   Explores non-linear relationships between a dependent variable and a predictor variable, considering star level as a factor.
    *   Utilizes spline regression with `statsmodels` and `patsy`.
    *   Provides model summaries and visualizations of predicted non-linear trends across star levels.
    *   Offers data downsampling option for faster computation with large datasets.


## How to Use

1.  **Installation:**
    *   Clone or download this repository.
    *   Navigate to the repository directory in your terminal.
    *   Create a virtual environment (optional but recommended): `python -m venv venv` (or `python3 -m venv venv` on some systems)
    *   Activate the virtual environment:
        *   On Windows: `venv\Scripts\activate`
        *   On macOS/Linux: `source venv/bin/activate`
    *   Install the required Python packages using pip: `pip install -r requirements.txt`

2.  **Running the Application:**
    *   In your terminal, within the activated virtual environment and repository directory, run: `streamlit run your_script_name.py` (replace `your_script_name.py` with the actual name of your Python script file).
    *   Streamlit will open the application in your web browser (usually at `http://localhost:8501`).

3.  **Using the Tool:**
    *   **Upload CSV Files:** Use the file uploader to upload your CSV files. Ensure your filenames contain star ratings (e.g., `one_star_data.csv`, `two_star_results.csv`) so the tool can automatically categorize the data.
    *   **Explore Combined Data:** Review the combined data table and use the checkbox to view the full dataset if needed.
    *   **Review Statistics:** Explore the "Combined Statistics" and "Statistics by Category" sections to understand descriptive measures and data distributions.
    *   **Statistical Significance Testing:** Analyze the pairwise t-test results for 'total\_time\_per\_meal' and 'satisfaction\_score' (or other specified columns) to identify significant differences between categories.
    *   **Advanced Statistical Tests:** Select a test from the "Advanced Statistical Tests" dropdown (Linear Regression, ANOVA, Chi-Square). Choose the relevant variables and click "Run" to perform the test and view the results and visualizations.
    *   **Interactive Plots:** Utilize the 2D and 3D scatter plot sections to visualize relationships between numerical variables. Experiment with different coordinate systems for 2D plots and select appropriate axes for all plot types.
    *   **Non-Linear Modeling:** Explore non-linear relationships using the "Non-Linear Modeling" section. Select dependent and predictor variables and run spline regression to analyze trends across star levels.

## Requirements

*   Python 3.7+
*   Libraries listed in `requirements.txt` (install using `pip install -r requirements.txt`)

    ```
    streamlit
    pandas
    numpy
    plotly
    scipy
    scikit-learn
    statsmodels
    patsy
    ```

## Data Input

*   **CSV Files:** The application expects CSV files as input.
*   **Filename-Based Categorization:** The tool infers the star category from the filename. Filenames should contain keywords like `one_star`, `two_star`, `three_star`, `four_star`, or `five_star` to be correctly categorized as '1-star', '2-star', '3-star', '4-star', or '5-star' respectively. Files without these keywords will be categorized as 'unknown'.
*   **Data Columns:** Ensure your CSV files contain numerical and categorical columns relevant to the analyses you wish to perform. The tool automatically cleans column names for easier processing.

## Statistical Tests and Visualizations

This tool provides a suite of statistical tests and visualizations commonly used in data analysis:

*   **Statistical Tests:** t-tests, ANOVA, Chi-Square test, Linear Regression, Spline Regression.
*   **Visualizations:** Histograms, Box Plots, Scatter Plots (2D Cartesian, 2D Polar, 3D), Heatmaps.

These tools are designed to help users explore their data, identify trends, assess statistical significance, and gain deeper insights from their datasets.

## Authors

*   David Pearl
*   Matthew Murphy

---

This README provides a guide to using the PCF Data Analysis and Visualization Tool. For any questions or issues, please refer to the code or contact the authors.
