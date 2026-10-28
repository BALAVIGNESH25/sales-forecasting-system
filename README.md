# Sales Forecasting System

## 📌 Project Overview
A machine learning-based Sales Forecasting System that predicts future sales using historical data. The project compares multiple machine learning models and provides an interactive dashboard built with Streamlit for visualizing sales forecasts.

## 🚀 Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Time series analysis and feature engineering
- Sales prediction using XGBoost and Random Forest
- Model evaluation and performance comparison
- Interactive Streamlit dashboard for sales forecasting

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** XGBoost, Random Forest
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit

## 📊 Model Performance

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| XGBoost | 1782.46 | 5375.72 | 7.07% |
| Random Forest | 2069.19 | 6403.97 | 8.20% |
| Naive Baseline | 3868.20 | 11724.69 | 15.33% |

**Best Model:** XGBoost achieved the lowest error, with a WAPE of 7.07% on the validation data.

## ⚙️ Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/BALAVIGNESH25/sales-forecasting-system.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sales-forecasting-system
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows (Git Bash):**

   ```bash
   source .venv/Scripts/activate
   ```

4. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
   ```

5. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with your actual Streamlit application filename if it is different.

## 📈 Results
The project demonstrates that XGBoost outperformed Random Forest and the naive baseline on the validation dataset. The Streamlit dashboard makes the forecasting results easier to explore and understand.

## 👨‍💻 Author
**Balavignesh M**

GitHub: [BALAVIGNESH25](https://github.com/BALAVIGNESH25)
