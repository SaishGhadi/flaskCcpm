from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Laravel access

# Load the Carbon Credit Price model
model = None
try:
    model = joblib.load('cc_price_model.pkl')
    logger.info("Carbon Credit Price Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        'message': 'Carbon Credit Price Prediction API is running',
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0',
        'service': 'carbon-credit-price-prediction'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'message': 'Carbon Credit Price Prediction API is running',
        'model_loaded': model is not None,
        'service': 'carbon-credit-price-prediction',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Carbon Credit Price prediction endpoint"""
    try:
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        
        if not data:
            logger.warning("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400

        # Log the incoming request for debugging
        logger.info(f"Prediction request received: {data}")

        # Create DataFrame from input data (same as your original logic)
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert to float to ensure JSON serialization
        predicted_price = float(prediction[0])
        
        # Log the prediction for monitoring
        logger.info(f"Prediction made: {predicted_price} for input: {data}")
        
        response = {
            'predicted_price': predicted_price,
            'status': 'success',
            'currency': 'USD',  # Adjust if needed
            'model': 'carbon_credit_price_model',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Prediction failed'
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple carbon credit prices"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        
        if not data or 'batch_data' not in data:
            return jsonify({
                'error': 'No batch data provided', 
                'expected_format': {'batch_data': [{'feature1': 'value1'}, {'feature2': 'value2'}]}
            }), 400

        batch_data = data['batch_data']
        
        if not isinstance(batch_data, list):
            return jsonify({'error': 'batch_data must be a list'}), 400

        if len(batch_data) == 0:
            return jsonify({'error': 'batch_data cannot be empty'}), 400

        # Create DataFrame from batch data
        input_df = pd.DataFrame(batch_data)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Convert to list of floats
        predictions_list = [float(pred) for pred in predictions]
        
        logger.info(f"Batch prediction made for {len(batch_data)} records")
        
        return jsonify({
            'predicted_prices': predictions_list,
            'count': len(predictions_list),
            'status': 'success',
            'currency': 'USD',
            'model': 'carbon_credit_price_model',
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Batch prediction failed'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': type(model).__name__,
            'model_name': 'Carbon Credit Price Prediction Model',
            'status': 'loaded',
            'service': 'carbon-credit-price-prediction',
            'version': '1.0.0'
        }
        
        # Add more model-specific info if available
        if hasattr(model, 'feature_names_in_'):
            info['features'] = model.feature_names_in_.tolist()
            info['feature_count'] = len(model.feature_names_in_)
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        
        # Try to get additional model information
        try:
            if hasattr(model, 'get_params'):
                info['model_parameters'] = model.get_params()
        except:
            pass
            
        return jsonify(info)
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': f'Could not get model info: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """Simplified prediction endpoint that mimics your original exactly"""
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint for Laravel integration"""
    return jsonify({
        'api_status': 'online',
        'service': 'carbon-credit-price-prediction',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'endpoints': [
            {'path': '/', 'method': 'GET', 'description': 'Home page'},
            {'path': '/health', 'method': 'GET', 'description': 'Health check'},
            {'path': '/predict', 'method': 'POST', 'description': 'Single prediction'},
            {'path': '/predict-batch', 'method': 'POST', 'description': 'Batch prediction'},
            {'path': '/predict-simple', 'method': 'POST', 'description': 'Simple prediction (original format)'},
            {'path': '/model-info', 'method': 'GET', 'description': 'Model information'},
            {'path': '/api/status', 'method': 'GET', 'description': 'API status'}
        ],
        'timestamp': pd.Timestamp.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error',
        'available_endpoints': [
            '/', '/health', '/predict', '/predict-batch', 
            '/predict-simple', '/model-info', '/api/status'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'status': 'error',
        'message': 'Check the HTTP method for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error',
        'message': 'Something went wrong on the server'
    }), 500

# Add a catch-all route for debugging
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({
        'error': f'Path /{path} not found',
        'status': 'error',
        'available_endpoints': [
            '/', '/health', '/predict', '/predict-batch', 
            '/predict-simple', '/model-info', '/api/status'
        ]
    }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    host = '0.0.0.0'
    
    logger.info(f"Starting Carbon Credit Price Prediction API on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model loaded: {model is not None}")
    
    app.run(host=host, port=port, debug=debug)