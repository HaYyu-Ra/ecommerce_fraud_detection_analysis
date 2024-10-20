# File: encode_categorical_features.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Set paths to cleaned and scaled data files
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_scaled.csv"
credit_card_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/credit_card_scaled.csv"

# Load the datasets
fraud_df = pd.read_csv(fraud_data_path)
credit_card_df = pd.read_csv(credit_card_data_path)

# Encoding categorical features for the fraud dataset
# Identify categorical features
categorical_features_fraud = ['source', 'browser', 'sex', 'country']  # Add or adjust features as necessary

# Check for the existence of categorical features in the fraud DataFrame
existing_categorical_features_fraud = [feature for feature in categorical_features_fraud if feature in fraud_df.columns]

# One-hot encoding for categorical features in fraud_df
if existing_categorical_features_fraud:
    fraud_df_encoded = pd.get_dummies(fraud_df, columns=existing_categorical_features_fraud, drop_first=True)
else:
    fraud_df_encoded = fraud_df.copy()  # No categorical features to encode

# For credit card data, if there are categorical features, apply encoding (assuming 'Class' is a categorical variable)
# Label encoding for 'Class' in credit_card_df (if needed)
if 'Class' in credit_card_df.columns:
    label_encoder = LabelEncoder()
    credit_card_df['Class'] = label_encoder.fit_transform(credit_card_df['Class'])

# Check if there are any categorical features in credit_card_df (if any categorical features need encoding)
# You might need to add more categorical columns to this list if available
categorical_features_credit_card = []  # Update this list with actual categorical features if any exist

# One-hot encoding for categorical features in credit_card_df (if applicable)
if categorical_features_credit_card:
    existing_categorical_features_credit_card = [feature for feature in categorical_features_credit_card if feature in credit_card_df.columns]
    if existing_categorical_features_credit_card:
        credit_card_df_encoded = pd.get_dummies(credit_card_df, columns=existing_categorical_features_credit_card, drop_first=True)
    else:
        credit_card_df_encoded = credit_card_df.copy()  # No categorical features to encode
else:
    credit_card_df_encoded = credit_card_df.copy()  # No categorical features to encode

# Save the encoded datasets
encoded_fraud_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_encoded.csv"
encoded_credit_card_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/credit_card_encoded.csv"

fraud_df_encoded.to_csv(encoded_fraud_output_path, index=False)
credit_card_df_encoded.to_csv(encoded_credit_card_output_path, index=False)

print(f"Encoded fraud dataset saved to: {encoded_fraud_output_path}")
print(f"Encoded credit card dataset saved to: {encoded_credit_card_output_path}")
