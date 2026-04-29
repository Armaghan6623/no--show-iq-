import pandas as pd


class DataPreprocessor:
    def __init__(self, filepath: str = None):
        self.filepath = filepath

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Spelling Traps
        df = df.rename(columns={
            'Hipertension': 'Hypertension',
            'Handcap': 'Handicap',
            'No-show': 'no_show'
        })

        # 2. Fix Invalid Data
        df = df[df['Age'] >= 0]
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

        # 3. Feature Engineering
        df['days_in_advance'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        df = df[df['days_in_advance'] >= 0]

        # 4. Advanced Features for Tests
        df['appointment_weekday'] = df['AppointmentDay'].dt.day_name().astype('category')
        df['is_weekend_appointment'] = df['AppointmentDay'].dt.dayofweek >= 5
        df['age_group'] = pd.cut(df['Age'], bins=[-1, 18, 35, 60, 120],
                                 labels=['Minor', 'Young', 'Adult', 'Senior']).astype('category')

        # 5. Encoding
        df['no_show'] = df['no_show'].map({'No': 0, 'Yes': 1})
        return df

    def preprocess(self):
        df = pd.read_csv(self.filepath)
        processed_df = self.clean_data(df)
        report = {
            'total_rows': len(processed_df),
            'total_columns': len(processed_df.columns),
            'no_show_rate': {'rate': processed_df['no_show'].mean()},
            'age_distribution': processed_df['Age'].describe().to_dict()
        }
        return processed_df, report


# --- BRIDGE FUNCTIONS FOR TESTS AND RUN.PY ---


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standalone function for test_preprocessing.py to call."""
    return DataPreprocessor().clean_data(df)


def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """Main pipeline for run.py."""
    preprocessor = DataPreprocessor(filepath)
    df, _ = preprocessor.preprocess()
    features = ['Age', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap',
                'SMS_received', 'days_in_advance', 'no_show']
    return df[features].dropna()
