# 🧠 Ridge Regression

This project is an interactive visualization of **Ridge Regression** on polynomial data, built with **Streamlit**. It allows users to adjust the model's polynomial degree and regularization strength (`alpha`) to see how these parameters affect the regression results.

---

## 📊 Features

- Load and visualize training/testing data
- Adjust **polynomial degree** (1–10)
- Adjust **regularization strength** `alpha` (0.0–10.0)
- Real-time plot of predictions vs actual values
- Mean Squared Error display
- Option to download predictions as CSV

---

## 🗂 Files Included

- `app.py` – Streamlit app
- `Assignment1.ipynb` – Original notebook implementation
- `ridge_regression_train_data.csv` – Training data
- `ridge_regression_test_data.csv` – Test data
- `requirements.txt` – Python dependencies

---

## 📦 Installation (Run Locally)

1. Clone the repository:
   ```bash
   git clone https://github.com/JadEletry/ml-ridge-regression-demo.git
   cd ml-ridge-regression-demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```
