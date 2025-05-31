# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Generate a larger synthetic dataset
np.random.seed(42)
num_samples = 50  # Increased data size

area = np.random.randint(1000, 6000, num_samples)  # Random area sizes
bedrooms = np.random.randint(2, 7, num_samples)  # Random number of bedrooms
age = np.random.randint(1, 30, num_samples)  # Random house age
price = area * 120 + bedrooms * 10000 - age * 500 + np.random.randint(-20000, 20000, num_samples)

# Create a DataFrame
df = pd.DataFrame({'Area': area, 'Bedrooms': bedrooms, 'Age': age, 'Price': price})

# Step 3: Define features and target variable
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Step 4: Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train both Linear Regression & Random Forest
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Step 8: Evaluate both models
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")

evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Step 9: Improved visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_linear, label="Linear Regression", color='blue')
plt.scatter(y_test, y_pred_rf, label="Random Forest", color='red', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="black", label="Perfect Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
