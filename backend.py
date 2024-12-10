from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and expected number of features
model, expected_num_features = joblib.load('stroke_predictor.pkl')

print(f"Model loaded. Expected number of features: {expected_num_features}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Ensure all required keys are present in the request
        required_keys = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                         'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing required key: {key}'}), 400

        # Preprocess input data
        gender = [1] if data['gender'] == 'Male' else [0]
        ever_married = [1] if data['ever_married'] == 'Yes' else [0]
        work_type = [0, 0, 0, 0, 0]
        if data['work_type'] == 'Private':
            work_type = [1, 0, 0, 0, 0]
        elif data['work_type'] == 'Self-employed':
            work_type = [0, 1, 0, 0, 0]
        elif data['work_type'] == 'Govt_job':
            work_type = [0, 0, 1, 0, 0]
        elif data['work_type'] == 'children':
            work_type = [0, 0, 0, 1, 0]
        elif data['work_type'] == 'Never_worked':
            work_type = [0, 0, 0, 0, 1]
        residence_type = [1] if data['Residence_type'] == 'Urban' else [0]
        smoking_status = [0, 0, 0]
        if data['smoking_status'] == 'formerly smoked':
            smoking_status = [1, 0, 0]
        elif data['smoking_status'] == 'never smoked':
            smoking_status = [0, 1, 0]
        elif data['smoking_status'] == 'smokes':
            smoking_status = [0, 0, 1]

        # Create the feature vector
        input_features = np.array([
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['avg_glucose_level'],
            data['bmi']
        ] + gender + ever_married + work_type + residence_type + smoking_status).reshape(1, -1)

        # Log the input for debugging
        print(f"Input features: {input_features}")
        print(f"Expected number of features: {expected_num_features}")

        # Validate feature vector shape
        if input_features.shape[1] != expected_num_features:
            return jsonify({'error': f"Expected {expected_num_features} features, but got {input_features.shape[1]}."}), 400

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Return prediction result as boolean
        return jsonify({'id': data['id'], 'stroke_risk': bool(prediction)})

    except Exception as e:
        # Return error details
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
