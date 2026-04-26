from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
from .model import NoShowModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model instance
model = None

def initialize_model(model_path: str = "model.joblib") -> None:
    """
    Initialize and load the fitted model. 
    Crucial for passing tests after running run.py.
    """
    global model
    model = NoShowModel()
    
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
            logger.info(f"✅ Model successfully loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
    else:
        logger.warning(f"⚠️ No model found at {model_path}. Predict will fail until fitted.")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API is operational'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # This will now succeed because the model is loaded from joblib
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_show': float(probability[1]),
            'probability_show': float(probability[0])
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features'])
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'prediction': int(pred),
                'probability_no_show': float(probabilities[i, 1]),
                'probability_show': float(probabilities[i, 0])
            })
        return jsonify({'predictions': results})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def run(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    initialize_model('model.joblib')
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run(debug=True)