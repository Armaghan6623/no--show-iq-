import sys
from pathlib import Path
import pytest
import json

# Ensure the src directory is in the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noshow_iq.api import app, initialize_model


@pytest.fixture
def client():
    """Setup the Flask test client and ensure the model is loaded."""
    app.config['TESTING'] = True
    # Crucial: Load the model that run.py just created
    initialize_model('model.joblib')
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert json.loads(response.data)['status'] == 'healthy'


def test_prediction_endpoint(client):
    """Test single prediction with fitted model."""
    payload = {
        "features": [30, 1, 0, 0, 0, 1, 5]  # Match 7-feature structure
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability_no_show' in data


def test_batch_prediction(client):
    """Test multiple predictions."""
    payload = {
        "features": [
            [25, 0, 0, 0, 0, 0, 2],
            [70, 1, 1, 0, 0, 1, 10]
        ]
    }
    response = client.post('/predict_batch',
                           data=json.dumps(payload),
                           content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['predictions']) == 2


def test_invalid_input(client):
    response = client.post('/predict', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400


def test_404_error(client):
    response = client.get('/missing_route')
    assert response.status_code == 404
