import sys
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model/xgb_model.pkl')

# The features your model expects, in the correct order
feature_names = [
    'YearEnd',
    'DataValueAlt',
    'LowConfidenceLimit',
    'HighConfidenceLimit',
    'StratificationCategory2',
    'Stratification2',
    'StratificationCategory3',
    'Stratification3',
    'ResponseID',
    'LocationID'
]

# Function to parse command line args into a dataframe row
def parse_args(args):
    if len(args) != len(feature_names):
        print(f"Usage: python predict_model.py {' '.join(feature_names)}")
        sys.exit(1)
    # Build a dict from args
    data = dict(zip(feature_names, args))
    # Convert numeric features to float, keep categorical as string
    # Assuming Stratification* and others are categorical (strings)
    # Adjust if needed
    for col in ['YearEnd', 'DataValueAlt', 'LowConfidenceLimit', 'HighConfidenceLimit', 'ResponseID', 'LocationID']:
        data[col] = float(data[col])
    return pd.DataFrame([data])

if __name__ == "__main__":
    # Skip script name, get the args
    input_args = sys.argv[1:]
    new_data = parse_args(input_args)

    prediction = model.predict(new_data)
    print(f"Predicted DataValue: {prediction[0]:.2f}")
