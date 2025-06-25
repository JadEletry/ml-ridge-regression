import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load data
train_data = pd.read_csv("ridge_regression_train_data.csv")
test_data = pd.read_csv("ridge_regression_test_data.csv")

X_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y'].values
X_test = test_data['x'].values.reshape(-1, 1)
y_test = test_data['y'].values

st.title("Ridge Regression (Polynomial) Interactive Demo")

# Sidebar controls
degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=10, value=3)
alpha = st.sidebar.slider("Regularization Strength (alpha)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Model training
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plotting
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='blue', label='Train Data')
ax.scatter(X_test, y_test, color='green', label='Test Data')
ax.plot(X_test, y_pred, color='red', label='Prediction')
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Ridge Regression (Degree={degree}, Alpha={alpha})")
st.pyplot(fig)

# Show metrics
mse = np.mean((y_pred - y_test) ** 2)
st.write(f"**Mean Squared Error on Test Set:** {mse:.2f}")

# Option to download predictions
result_df = pd.DataFrame({"x": X_test.flatten(), "Actual y": y_test, "Predicted y": y_pred})
st.download_button("Download Predictions as CSV", result_df.to_csv(index=False), "predictions.csv")
