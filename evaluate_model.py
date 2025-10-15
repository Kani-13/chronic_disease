import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load test dataset
df = pd.read_csv('data/chronic.csv')

# Specify the target column
target_column = 'DataValue'

# Drop rows with missing target
df = df.dropna(subset=[target_column])

# Prepare features and target
# Select numeric columns only for features (excluding target)
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column])
y = df[target_column]

# Load trained model
model = joblib.load('model/xgb_model.pkl')

# Predict on test data
y_pred = model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Calculate approximate accuracy from R-squared
accuracy = r2 * 100

# Print metrics and approximate accuracy
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Approximate Accuracy: {accuracy:.2f}%")
