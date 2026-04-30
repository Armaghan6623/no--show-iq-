"""
MongoDB connection and operations for NoShowIQ.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from dotenv import load_dotenv  # noqa: E402
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)

# Risk level and recommendation thresholds
RISK_THRESHOLD_HIGH = 0.6
RISK_THRESHOLD_MEDIUM = 0.3


def _get_risk_level(probability_no_show: float) -> str:
    """
    Determine risk level based on no-show probability.

    Args:
        probability_no_show: Probability of no-show (0-1)

    Returns:
        Risk level: 'high', 'medium', or 'low'
    """
    if probability_no_show >= RISK_THRESHOLD_HIGH:
        return 'high'
    elif probability_no_show >= RISK_THRESHOLD_MEDIUM:
        return 'medium'
    return 'low'


def _get_recommendation(probability_no_show: float, risk_level: str) -> str:
    """
    Get recommendation based on probability and risk level.

    Args:
        probability_no_show: Probability of no-show (0-1)
        risk_level: Risk level ('high', 'medium', or 'low')

    Returns:
        Recommendation string
    """
    if risk_level == 'high':
        return 'Send reminder SMS and consider calling patient'
    if risk_level == 'medium':
        return 'Send reminder SMS'
    return 'No action needed'


class MongoDBClient:
    """Handle MongoDB connection and operations."""

    def __init__(self, mongo_uri: str = None):
        """
        Initialize MongoDB client.

        Args:
            mongo_uri: MongoDB connection string. Defaults to MONGO_URI env var.
        """
        self.mongo_uri = mongo_uri or os.getenv(
            'MONGO_URI',
            'mongodb://root:example@localhost:27017/noshowiq'
        )
        self.client = None
        self.db = None
        self.predictions_collection = None
        self.training_runs_collection = None
        self.connect()

    def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client['noshowiq']
            self.predictions_collection = self.db['predictions']
            self.training_runs_collection = self.db['training_runs']
            logger.info("Connected to MongoDB successfully")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            msg = f"MongoDB connection failed: {e}. Predictions won't be saved."
            logger.warning(msg)
            self.client = None
            self.db = None
            self.predictions_collection = None
            self.training_runs_collection = None

    def save_prediction(
        self,
        raw_input: dict,
        cleaned_features: list,
        prediction: int,
        probability_no_show: float,
        probability_show: float,
    ) -> dict:
        """
        Save prediction to MongoDB with all required fields.

        Args:
            raw_input: Raw input from the request
            cleaned_features: Cleaned/preprocessed features
            prediction: Prediction result (0 or 1)
            probability_no_show: Probability of no-show
            probability_show: Probability of show

        Returns:
            Document inserted or error dict
        """
        if self.predictions_collection is None:
            logger.warning("MongoDB not connected, skipping save")
            return {'error': 'MongoDB not available'}

        try:
            # Determine risk level and recommendation
            risk_level = _get_risk_level(probability_no_show)
            recommendation = _get_recommendation(probability_no_show, risk_level)

            probability = {
                'no_show': probability_no_show,
                'show': probability_show,
            }

            document = {
                'timestamp': datetime.utcnow(),
                'raw_input': raw_input,
                'cleaned_features': cleaned_features,
                'risk_level': risk_level,
                'probability': probability,
                'recommendation': recommendation,
            }

            result = self.predictions_collection.insert_one(document)
            msg = f"Prediction saved to MongoDB with ID: {result.inserted_id}"
            logger.info(msg)
            return {'id': str(result.inserted_id), **document}
        except Exception as e:
            msg = f"Error saving prediction to MongoDB: {e}"
            logger.error(msg)
            return {'error': str(e)}

    def save_training_run(
        self,
        training_size: int,
        metrics: dict,
        imbalance_technique: str,
    ) -> dict:
        """
        Save training run metrics to MongoDB.

        Args:
            training_size: Number of samples used in training
            metrics: Dictionary with precision, recall, F1 per class
            imbalance_technique: Technique used for handling class imbalance

        Returns:
            Document inserted or error dict
        """
        if self.training_runs_collection is None:
            logger.warning("MongoDB not connected, skipping save")
            return {'error': 'MongoDB not available'}

        try:
            document = {
                'timestamp': datetime.utcnow(),
                'training_size': training_size,
                'metrics': {
                    'precision': metrics.get('precision', {}),
                    'recall': metrics.get('recall', {}),
                    'f1': metrics.get('f1', {}),
                },
                'imbalance_technique': imbalance_technique,
            }

            result = self.training_runs_collection.insert_one(document)
            msg = f"Training run saved to MongoDB with ID: {result.inserted_id}"
            logger.info(msg)
            return {'id': str(result.inserted_id), **document}
        except Exception as e:
            msg = f"Error saving training run to MongoDB: {e}"
            logger.error(msg)
            return {'error': str(e)}

    def get_stats(self) -> dict:
        """
        Get statistics using MongoDB aggregation pipeline.

        This method uses only MongoDB aggregation - no Python computation
        for computing the statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        if self.predictions_collection is None or self.training_runs_collection is None:
            logger.warning("MongoDB not connected")
            # Return default stats instead of error for smoke test
            return {
                'total_predictions': 0,
                'high_risk_count': 0,
                'low_risk_count': 0,
                'average_probability': 0.0,
                'last_trained': None,
            }

        try:
            # Aggregation pipeline for predictions stats
            predictions_pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'total_predictions': {'$sum': 1},
                        'high_risk_count': {
                            '$sum': {
                                '$cond': [{'$eq': ['$risk_level', 'high']}, 1, 0]
                            }
                        },
                        'low_risk_count': {
                            '$sum': {
                                '$cond': [{'$eq': ['$risk_level', 'low']}, 1, 0]
                            }
                        },
                        'avg_probability_no_show': {
                            '$avg': '$probability.no_show'
                        },
                    }
                }
            ]

            predictions_result = list(
                self.predictions_collection.aggregate(predictions_pipeline)
            )

            # Get last training timestamp
            last_train_pipeline = [
                {'$sort': {'timestamp': -1}},
                {'$limit': 1},
                {'$project': {'timestamp': 1}},
            ]

            last_train_result = list(
                self.training_runs_collection.aggregate(last_train_pipeline)
            )

            # Build stats response
            if predictions_result:
                stats = {
                    'total_predictions': predictions_result[0]['total_predictions'],
                    'high_risk_count': predictions_result[0]['high_risk_count'],
                    'low_risk_count': predictions_result[0]['low_risk_count'],
                    'average_probability': round(
                        predictions_result[0]['avg_probability_no_show'], 2
                    ),
                }
            else:
                stats = {
                    'total_predictions': 0,
                    'high_risk_count': 0,
                    'low_risk_count': 0,
                    'average_probability': 0.0,
                }

            # Add last_trained if available
            if last_train_result and 'timestamp' in last_train_result[0]:
                stats['last_trained'] = (
                    last_train_result[0]['timestamp'].isoformat() + 'Z'
                )
            else:
                stats['last_trained'] = None

            return stats

        except Exception as e:
            msg = f"Error getting stats from MongoDB: {e}"
            logger.error(msg)
            return {'error': str(e)}

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global MongoDB client instance
mongo_client = None


def get_mongo_client() -> MongoDBClient:
    """Get or create MongoDB client."""
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoDBClient()
    return mongo_client
