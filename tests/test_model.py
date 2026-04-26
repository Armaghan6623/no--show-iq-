import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the 'src' directory to the system path so we can import the noshow_iq package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noshow_iq.model import NoShowModel, train_and_evaluate, split_data

class TestModel:
    """Test suite for the Machine Learning model components."""

    @pytest.fixture
    def mock_features(self):
        """Creates dummy training data with the 7 features + target."""
        np.random.seed(42)
        n_samples = 100
        data = {
            'Age': np.random.randint(0, 90, n_samples),
            'Hypertension': np.random.choice([0, 1], n_samples),
            'Diabetes': np.random.choice([0, 1], n_samples),
            'Alcoholism': np.random.choice([0, 1], n_samples),
            'Handicap': np.random.choice([0, 1], n_samples),
            'SMS_received': np.random.choice([0, 1], n_samples),
            'days_in_advance': np.random.randint(0, 30, n_samples),
            'no_show': np.random.choice([0, 1], n_samples)
        }
        return pd.DataFrame(data)

    def test_model_initialization(self):
        """Test if different model types initialize correctly."""
        lr_model = NoShowModel(model_type='logistic_regression')
        rf_model = NoShowModel(model_type='random_forest')
        
        assert lr_model.model_type == 'logistic_regression'
        assert rf_model.model_type == 'random_forest'
        
        with pytest.raises(ValueError):
            NoShowModel(model_type='invalid_model_name')

    def test_data_splitting(self, mock_features):
        """Test the stratify split logic."""
        X = mock_features.drop('no_show', axis=1)
        y = mock_features['no_show']
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80

    def test_training_and_metrics(self, mock_features):
        """Test if the model trains and returns the required 5 metrics."""
        X = mock_features.drop('no_show', axis=1)
        y = mock_features['no_show']
        
        model_obj, metrics = train_and_evaluate(X, y, model_type='logistic_regression')
        
        # Check if all midterm-required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    def test_model_prediction_format(self, mock_features):
        """Ensure predict returns 0/1 and predict_proba returns probabilities."""
        X = mock_features.drop('no_show', axis=1)
        y = mock_features['no_show']
        
        model_obj = NoShowModel(model_type='random_forest')
        model_obj.train(X.values, y.values)
        
        preds = model_obj.predict(X.values[:5])
        probs = model_obj.predict_proba(X.values[:5])
        
        assert len(preds) == 5
        assert preds.dtype in [np.int64, int]
        assert probs.shape == (5, 2) # Binary classification shape

    def test_model_persistence(self, tmp_path, mock_features):
        """Test if the model can be saved and reloaded."""
        X = mock_features.drop('no_show', axis=1)
        y = mock_features['no_show']
        
        model_obj = NoShowModel(model_type='logistic_regression')
        model_obj.train(X.values, y.values)
        
        # Save to temporary directory
        model_path = tmp_path / "test_model.joblib"
        model_obj.save_model(str(model_path))
        
        assert os.path.exists(model_path)
        
        # Load into a new object
        new_model = NoShowModel()
        new_model.load_model(str(model_path))
        
        # Check if it can still predict
        new_preds = new_model.predict(X.values[:1])
        assert len(new_preds) == 1

if __name__ == "__main__":
    pytest.main([__file__])