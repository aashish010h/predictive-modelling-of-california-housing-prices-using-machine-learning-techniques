# 🏡 California Housing Price Prediction

### 📋 Project Overview
This project is a machine learning regression task aimed at predicting median house values in California districts. Using data from the 1990 U.S. Census, I built and compared two predictive models to estimate property prices based on features like location, population, and median income.

The goal was to demonstrate the effectiveness of ensemble methods (Random Forest) over traditional linear baselines in capturing complex, non-linear market dynamics.

### 📊 Key Results
The **Random Forest Regressor** significantly outperformed the baseline, proving that "Location" and "Income" have non-linear impacts on price.

| Model | R² Score (Accuracy) | RMSE (Avg Error) |
| :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 64.01% | $68,506 |
| **Random Forest (Proposed)** | **81.45%** | **$49,183** |

### 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:**
    * `pandas` & `numpy` (Data Manipulation)
    * `scikit-learn` (Machine Learning)
    * `matplotlib` & `seaborn` (Visualization)

### 🧠 Methodology

#### 1. Data Preprocessing
* **Handling Missing Values:** Used Median Imputation to fill missing counts in `total_bedrooms` without losing data.
* **Feature Engineering:** Calculated "per household" metrics (e.g., Rooms per Household) to normalize the data.
* **Encoding:** Applied **One-Hot Encoding** to the `ocean_proximity` column to convert categorical text into machine-readable binary features.

#### 2. Exploratory Data Analysis (EDA)
I visualized the geographic distribution of prices to confirm that coastal proximity is a major driver of value.

![Geographic Heatmap](path_to_your_heatmap_image.png)
*(Figure: Heatmap showing high-value clusters along the California coast)*

#### 3. Model Comparison
I compared the "Actual vs. Predicted" values for both models. The Random Forest model (right) shows a much tighter fit to the ideal diagonal line compared to the wide variance of the Linear Regression (left).

![Actual vs Predicted](path_to_your_scatter_plot_image.png)

### 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/california-housing-prediction.git](https://github.com/yourusername/california-housing-prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook california_housing_project.ipynb
    ```

### 🔮 Future Improvements
* **Hyperparameter Tuning:** Implement GridSearch to optimize the Random Forest depth and estimators.
* **Modern Data:** Retrain the model on current housing market data (2024-2025) to make it applicable to today's market.
* **Feature Expansion:** Integrate crime rates or school district data for more granular predictions.

---
**Author:** Aashish Giri
