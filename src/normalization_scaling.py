# File: normalization_scaling.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set paths to cleaned data files
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_with_features.csv"
credit_card_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/creditcard_cleaned.csv"

# Load the datasets
fraud_df = pd.read_csv(fraud_data_path)
credit_card_df = pd.read_csv(credit_card_data_path)

# Select features to scale
# Assuming relevant features for scaling from fraud_df (change as necessary)
fraud_features = ['purchase_value', 'transaction_count', 'transaction_velocity', 'hour_of_day', 'day_of_week']

# Normalize and scale features for fraud_df
scaler_min_max = MinMaxScaler()
fraud_df[fraud_features] = scaler_min_max.fit_transform(fraud_df[fraud_features])

# Assuming relevant features for scaling from credit_card_df (change as necessary)
credit_card_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 
                        'V27', 'V28', 'Amount']

# Standardize features for credit_card_df
scaler_standard = StandardScaler()
credit_card_df[credit_card_features] = scaler_standard.fit_transform(credit_card_df[credit_card_features])

# Save the scaled datasets
scaled_fraud_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_scaled.csv"
scaled_credit_card_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/credit_card_scaled.csv"

fraud_df.to_csv(scaled_fraud_output_path, index=False)
credit_card_df.to_csv(scaled_credit_card_output_path, index=False)

print(f"Scaled fraud dataset saved to: {scaled_fraud_output_path}")
print(f"Scaled credit card dataset saved to: {scaled_credit_card_output_path}")
