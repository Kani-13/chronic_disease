from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model/xgb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form (make sure the order matches your model input)
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f'Predicted Class: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
