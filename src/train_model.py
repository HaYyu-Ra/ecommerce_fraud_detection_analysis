import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load datasets
data_merged_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\merged_data.csv'
data_credit_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv'

data_merged = pd.read_csv(data_merged_path)
data_credit = pd.read_csv(data_credit_path)

# Sample data to a maximum of 50,000 rows
data_merged = data_merged.sample(n=min(50000, len(data_merged)), random_state=42)
data_credit = data_credit.sample(n=min(50000, len(data_credit)), random_state=42)

# Combine datasets and drop unnecessary columns
data_combined = pd.concat([data_merged, data_credit], ignore_index=True)
data_combined.drop(columns=['user_id', 'signup_time', 'purchase_time'], errors='ignore', inplace=True)

# Handle missing values
data_combined.fillna(data_combined.select_dtypes(include=[np.number]).median(), inplace=True)
for col in data_combined.select_dtypes(include=['object']).columns:
    data_combined[col].fillna(data_combined[col].mode()[0], inplace=True)

# Encode categorical columns
data_combined = pd.get_dummies(data_combined, columns=['source', 'browser', 'sex'], drop_first=True)

# Separate features and target variable
X = data_combined.drop(columns=['Class'], errors='ignore')
y = data_combined['Class']

# Keep only numeric features and normalize the data
X_numeric = X.select_dtypes(include=[np.number])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Convert back to DataFrame for further processing
X = pd.DataFrame(X_scaled, columns=X_numeric.columns)

# Display initial information
print("Summary of features before filtering:")
print(X.describe())

# Filter based on positive 'purchase_value' feature if it exists
if 'purchase_value' in X.columns:
    print("Filtering rows with 'purchase_value' > 0")
    initial_row_count = X.shape[0]
    X = X[X['purchase_value'] > 0]
    y = y.loc[X.index]  # Ensure y matches X after filtering
    print(f"Rows filtered: {initial_row_count - X.shape[0]}")
else:
    print("Warning: 'purchase_value' column not found for filtering.")

# Display the number of rows after filtering
print(f"Number of rows after filtering: {X.shape[0]}")

# Proceed only if there are valid rows left
if len(X) > 0:
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model accuracy
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")
else:
    print("No valid rows available for training the model.")
