# PCF Data Analysis and Visualization Tool

**By David Pearl, Matthew Murphy, and James Intriligator**

## Overview

This Streamlit application provides a comprehensive suite of tools for analyzing and visualizing datasets, with a particular focus on data categorized by star ratings (1-star to 5-star). Users can upload CSV files, perform a variety of statistical analyses, and generate insightful visualizations to explore data distributions, relationships, and significant differences across various categories.

## Features

*   **Data Upload and Preprocessing:**
    *   Upload multiple CSV files at once.
    *   Automatic categorization based on filenames (e.g., `one_star.csv`, `two_star.csv`).
    *   Data is combined into a single DataFrame for unified analysis.
    *   Column names are standardized for consistency.

*   **Descriptive Statistics:**
    *   View combined and category-specific descriptive statistics (Mean, Median, Min, Max, etc.).
    *   Generate histograms to visualize data distributions.

*   **Statistical Analysis:**
    *   **T-tests:** Assess statistical significance between categories.
    *   **Linear Regression:** Analyze relationships between numerical variables.
    *   **ANOVA:** Compare means across different categories.
    *   **Chi-Square Test:** Assess the independence of categorical variables.
    *   **Spline Regression:** Explore non-linear relationships.

*   **Interactive Visualizations:**
    *   2D Scatter Plots (Cartesian and Polar).
    *   3D Scatter Plots.
    *   Box Plots.
    *   Heatmaps.

## Getting Started

### Prerequisites

*   Python 3.7+
*   The libraries listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd PCF
    ```
3.  **Create and activate a virtual environment:**
    *   Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit app:**
    ```bash
    streamlit run Experiment_Analysis.py
    ```
2.  The application will open in your web browser, typically at `http://localhost:8501`.

## Data Input

*   **CSV Files:** The application accepts CSV files.
*   **Filename-Based Categorization:** For automatic categorization, filenames should contain keywords like `one_star`, `two_star`, etc.

