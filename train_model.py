import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load dataset
df = pd.read_csv('data/chronic.csv')

# Show available columns to choose target
print("Available columns in dataset:")
print(df.columns)

# Set the target column — change if needed
target_column = 'DataValue'

# Drop rows with missing target
df = df.dropna(subset=[target_column])

# Fill missing numeric values
df = df.fillna(df.median(numeric_only=True))

# Select numerical columns only for features
X = df.select_dtypes(include='number').drop(target_column, axis=1)
y = df[target_column]

# Save feature names for use during prediction
os.makedirs('model', exist_ok=True)
joblib.dump(X.columns.tolist(), 'model/features.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/xgb_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Model Mean Squared Error: {mse:.2f}')
