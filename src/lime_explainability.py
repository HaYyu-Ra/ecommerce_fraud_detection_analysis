# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load datasets
data_merged_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\merged_data.csv'

# Read the merged data
merged_data = pd.read_csv(data_merged_path)

# Step 1: Check for NaN values in the merged data
print("NaN values in each column before dropping:")
print(merged_data.isna().sum())

# Decide how to handle NaN values: 
# 1. Drop columns with more than 50% missing values
threshold = 0.5
merged_data = merged_data.loc[:, merged_data.isnull().mean() < threshold]

# 2. Fill remaining NaN values with the mean or drop rows if necessary
merged_data.fillna(merged_data.mean(numeric_only=True), inplace=True)

# Print shape after preprocessing
print("Shape of merged_data after handling NaN values:", merged_data.shape)

# Step 2: Convert datetime columns to numeric features
for col in ['signup_time', 'purchase_time']:
    if col in merged_data.columns:
        merged_data[col] = pd.to_datetime(merged_data[col], errors='coerce')
        merged_data[col] = merged_data[col].astype('int64') // 10**9  # Convert to Unix timestamp in seconds

# Step 3: Define features and target variable
X = merged_data.drop(columns=['class'], errors='ignore')  # Exclude target variable, ignore if 'class' doesn't exist
y = merged_data['class'] if 'class' in merged_data.columns else None

# Remove any remaining non-numeric columns
X = X.select_dtypes(include=[float, int])

# Check if X is empty
if X.empty:
    raise ValueError("Feature set X is empty after preprocessing. Please check your data cleaning steps.")

# Print shape of features
print("Shape of features (X):", X.shape)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Using LIME for Explainability ---

# Create LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=['Non-Fraud', 'Fraud'],
    mode='classification'
)

# Step 6: Explain a specific prediction
instance_index = 0  # Index of the instance you want to explain (adjust as needed)
exp = lime_explainer.explain_instance(
    X_test.values[instance_index],  # The instance to explain
    model.predict_proba,            # Model prediction function
    num_features=10                  # Number of features to show
)

# Step 7: LIME Feature Importance Plot
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation for Instance {}".format(instance_index))
plt.show()

# Optionally, save the LIME plot as an image
# plt.savefig('lime_feature_importance_plot.png')
