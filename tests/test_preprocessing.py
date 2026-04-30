import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402

from noshow_iq.preprocessing import (  # noqa: E402
    DataPreprocessor,
    clean_data,
    preprocess_pipeline,
)


class TestPreprocessing:
    """Test suite for preprocessing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample raw data for testing."""
        return pd.DataFrame({
            'PatientId': ['123456789', '987654321'],
            'AppointmentID': ['111111', '222222'],
            'Gender': ['F', 'M'],
            'ScheduledDay': ['2016-04-29T18:38:08Z', '2016-04-29T16:08:27Z'],
            'AppointmentDay': ['2016-04-29T00:00:00Z', '2016-04-29T00:00:00Z'],
            'Age': [62, 56],
            'Neighbourhood': ['JARDIM DA PENHA', 'JARDIM DA PENHA'],
            'Scholarship': [0, 0],
            'Hipertension': [1, 0],
            'Diabetes': [0, 0],
            'Alcoholism': [0, 0],
            'Handcap': [0, 0],
            'SMS_received': [0, 0],
            'No-show': ['No', 'No']
        })

    @pytest.fixture
    def data_path(self, tmp_path, sample_data):
        """Create a temporary CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        return csv_path

    def test_column_name_corrections(self, sample_data):
        result = clean_data(sample_data.copy())
        assert 'Hypertension' in result.columns
        assert 'Handicap' in result.columns
        assert 'no_show' in result.columns

    def test_data_type_conversions(self, sample_data):
        result = clean_data(sample_data.copy())
        assert pd.api.types.is_datetime64_any_dtype(result['ScheduledDay'])
        assert result['no_show'].dtype in [int, 'int64']

    def test_days_in_advance_calculation(self, sample_data):
        result = clean_data(sample_data.copy())
        assert 'days_in_advance' in result.columns
        assert all(result['days_in_advance'] >= 0)

    def test_negative_age_handling(self):
        data = pd.DataFrame({
            'Age': [-5, 25],
            'ScheduledDay': ['2016-04-29'] * 2,
            'AppointmentDay': ['2016-04-29'] * 2,
            'No-show': ['No'] * 2
        })
        result = clean_data(data)
        assert all(result['Age'] >= 0)

    def test_target_encoding(self, sample_data):
        result = clean_data(sample_data.copy())
        assert set(result['no_show'].unique()).issubset({0, 1})

    def test_comprehensive_preprocessing_pipeline(self, data_path):
        result = preprocess_pipeline(str(data_path))
        assert isinstance(result, pd.DataFrame)
        assert 'days_in_advance' in result.columns

    def test_comprehensive_preprocessor_class(self, data_path):
        # REMOVED SKIP: Now testing the class
        preprocessor = DataPreprocessor(str(data_path))
        df, report = preprocessor.preprocess()
        assert isinstance(df, pd.DataFrame)
        assert 'no_show_rate' in report

    def test_missing_value_handling(self, sample_data):
        result = clean_data(sample_data.copy())
        assert not result['no_show'].isnull().any()

    def test_feature_engineering_comprehensive(self, sample_data):
        # REMOVED SKIP: Now testing extra features
        preprocessor = DataPreprocessor("dummy")
        result = preprocessor.clean_data(sample_data.copy())
        assert 'age_group' in result.columns
        assert 'appointment_weekday' in result.columns

    def test_data_quality_validation(self, data_path):
        # REMOVED SKIP: Now testing reporting
        preprocessor = DataPreprocessor(str(data_path))
        _, report = preprocessor.preprocess()
        assert 'total_rows' in report
        assert 'age_distribution' in report


if __name__ == "__main__":
    pytest.main([__file__])
