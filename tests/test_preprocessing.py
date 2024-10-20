# File: test_preprocessing.py

import unittest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Define the paths to the datasets
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_scaled.csv"
credit_card_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/credit_card_scaled.csv"
fraud_encoded_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_encoded.csv"
credit_card_encoded_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/credit_card_encoded.csv"

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Load the datasets before each test"""
        self.fraud_df = pd.read_csv(fraud_data_path)
        self.credit_card_df = pd.read_csv(credit_card_data_path)
        self.fraud_encoded_df = pd.read_csv(fraud_encoded_path)
        self.credit_card_encoded_df = pd.read_csv(credit_card_encoded_path)

    def test_fraud_data_scaling(self):
        """Test if fraud dataset is properly scaled using MinMaxScaler"""
        features = ['purchase_value', 'transaction_count', 'transaction_velocity', 'hour_of_day', 'day_of_week']
        
        # Check that all values are between 0 and 1
        for feature in features:
            self.assertTrue(self.fraud_df[feature].between(0, 1).all(), f"{feature} is not properly scaled")

    def test_credit_card_data_scaling(self):
        """Test if credit card dataset is properly standardized using StandardScaler"""
        features = [f"V{i}" for i in range(1, 29)] + ['Amount']
        
        # Check that the mean of each feature is approximately 0 and the std is 1
        for feature in features:
            mean = self.credit_card_df[feature].mean()
            std = self.credit_card_df[feature].std()
            self.assertAlmostEqual(mean, 0, places=1, msg=f"{feature} mean is not approximately 0")
            self.assertAlmostEqual(std, 1, places=1, msg=f"{feature} std is not approximately 1")

    def test_fraud_data_encoding(self):
        """Test if fraud dataset categorical features are properly one-hot encoded"""
        encoded_columns = ['source', 'browser', 'sex', 'country']  # Adjust according to your actual columns
        
        for column in encoded_columns:
            encoded_columns_exist = any(col.startswith(column) for col in self.fraud_encoded_df.columns)
            self.assertTrue(encoded_columns_exist, f"{column} is not properly encoded in fraud dataset")
        
    def test_credit_card_data_label_encoding(self):
        """Test if 'Class' in credit card dataset is properly label encoded"""
        # Check that the 'Class' column has been label encoded (0 or 1)
        self.assertIn('Class', self.credit_card_encoded_df.columns, "'Class' column is missing in credit card dataset")
        unique_values = self.credit_card_encoded_df['Class'].unique()
        self.assertTrue(set(unique_values).issubset({0, 1}), "Class column is not properly label encoded")
        
    def test_fraud_data_existence(self):
        """Test if the scaled and encoded fraud dataset is saved successfully"""
        # Test if the saved file exists and is not empty
        self.assertGreater(self.fraud_encoded_df.shape[0], 0, "Encoded fraud dataset is empty")
        self.assertGreater(self.fraud_encoded_df.shape[1], 0, "Encoded fraud dataset has no columns")

    def test_credit_card_data_existence(self):
        """Test if the scaled and encoded credit card dataset is saved successfully"""
        # Test if the saved file exists and is not empty
        self.assertGreater(self.credit_card_encoded_df.shape[0], 0, "Encoded credit card dataset is empty")
        self.assertGreater(self.credit_card_encoded_df.shape[1], 0, "Encoded credit card dataset has no columns")


if __name__ == "__main__":
    unittest.main()
