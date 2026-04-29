"""
Configuration module for NoShowIQ.

This module contains all configuration settings for the application
including paths, model parameters, and API settings.
"""

import os
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# Data configuration
class DataConfig:
    """Data-related configuration."""

    DATA_PATH = DATA_DIR / 'KaggleV2-May-2016.csv'
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42

    # Missing value handling
    MISSING_VALUE_STRATEGY = 'drop'  # 'drop', 'mean', 'median'

    # Feature scaling
    SCALE_FEATURES = True

    # Categorical encoding
    ENCODE_CATEGORICAL = True


# Model configuration
class ModelConfig:
    """Model-related configuration."""

    # Model type: 'logistic_regression', 'random_forest', 'gradient_boosting'
    MODEL_TYPE = 'logistic_regression'

    # Logistic Regression parameters
    LR_MAX_ITER = 1000
    LR_C = 1.0

    # Random Forest parameters
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1

    # Gradient Boosting parameters
    GB_N_ESTIMATORS = 100
    GB_LEARNING_RATE = 0.1
    GB_MAX_DEPTH = 3

    # Model paths
    MODEL_SAVE_PATH = MODELS_DIR / 'model.pkl'
    MODEL_WEIGHTS_PATH = MODELS_DIR / 'model_weights.pkl'


# API configuration
class APIConfig:
    """API-related configuration."""

    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

    # CORS settings
    CORS_ORIGINS = ['*']

    # Request/Response settings
    JSON_SORT_KEYS = False
    JSON_INDENT = 2
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size


# Logging configuration
class LogConfig:
    """Logging-related configuration."""

    LOG_LEVEL = 'INFO'
    LOG_FILE = LOGS_DIR / 'app.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Rotating file handler settings
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5


# Training configuration
class TrainingConfig:
    """Training-related configuration."""

    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'binary_crossentropy'

    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Model checkpointing
    SAVE_BEST_MODEL = True
    CHECKPOINT_PATH = MODELS_DIR / 'checkpoint.pkl'


# Evaluation configuration
class EvalConfig:
    """Evaluation-related configuration."""

    # Metrics to track
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Threshold for classification
    CLASSIFICATION_THRESHOLD = 0.5

    # Cross-validation folds
    CV_FOLDS = 5


# Environment-specific configuration
class DevelopmentConfig:
    """Development environment configuration."""

    DEBUG = True
    TESTING = False
    API_CONFIG = APIConfig()
    API_CONFIG.DEBUG = True
    API_CONFIG.PORT = 5000


class ProductionConfig:
    """Production environment configuration."""

    DEBUG = False
    TESTING = False
    API_CONFIG = APIConfig()
    API_CONFIG.DEBUG = False
    API_CONFIG.PORT = 5000


class TestingConfig:
    """Testing environment configuration."""

    DEBUG = False
    TESTING = True
    API_CONFIG = APIConfig()
    API_CONFIG.DEBUG = False
    API_CONFIG.PORT = 5001


class ConfigFactory:
    """Configuration factory."""

    def __init__(self):
        pass


# Configuration factory
def get_config(env: str = None) -> 'DevelopmentConfig':
    """
    Get configuration based on environment.

    Args:
        env: Environment name ('development', 'production', 'testing')
             If None, uses NOSHOW_ENV environment variable or defaults to 'development'

    Returns:
        Configuration object
    """
    if env is None:
        env = os.getenv('NOSHOW_ENV', 'development').lower()

    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }

    config_class = configs.get(env, DevelopmentConfig)
    return config_class()


if __name__ == '__main__':
    CONFIG = get_config()

# Default configuration instance
else:
    CONFIG = get_config()


if __name__ == '__main__':
    # Print configuration for debugging
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print("\\nData Config:")
    print(f"  Data Path: {DataConfig.DATA_PATH}")
    print("\\nAPI Config:")
    print(f"  Host: {APIConfig.HOST}")
    print(f"  Port: {APIConfig.PORT}")
    print("\\nModel Config:")
    print(f"  Model Type: {ModelConfig.MODEL_TYPE}")
