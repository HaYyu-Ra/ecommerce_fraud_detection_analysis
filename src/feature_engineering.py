# File: feature_engineering.py

# Import necessary libraries
import pandas as pd

# Set paths to cleaned data files
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_cleaned.csv"
merged_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_with_country.csv"

# Load the datasets
fraud_df = pd.read_csv(fraud_data_path)
merged_df = pd.read_csv(merged_data_path)

# Convert 'purchase_time' and 'signup_time' to datetime
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])

# Feature Engineering
# 1. Transaction Frequency: Number of transactions per user
fraud_df['transaction_count'] = fraud_df.groupby('user_id')['user_id'].transform('count')

# 2. Transaction Velocity: Average time between transactions for each user
fraud_df = fraud_df.sort_values(by=['user_id', 'purchase_time'])
fraud_df['time_diff'] = fraud_df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
fraud_df['transaction_velocity'] = fraud_df['time_diff'].fillna(0)

# 3. Time-Based Features
# i. Hour of Day
fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour

# ii. Day of Week (Monday=0, Sunday=6)
fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek

# Save the updated dataset with new features
feature_engineered_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_with_features.csv"
fraud_df.to_csv(feature_engineered_output_path, index=False)

print(f"Feature engineered dataset saved to: {feature_engineered_output_path}")
