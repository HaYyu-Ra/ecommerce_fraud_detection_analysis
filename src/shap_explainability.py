# Import necessary libraries
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load datasets
data_merged_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\merged_data.csv'
data_credit_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv'

merged_data = pd.read_csv(data_merged_path)
credit_data = pd.read_csv(data_credit_path)

# Step 1: Check for NaN values and drop columns or rows accordingly
print("Check for NaN values in features:")
nan_counts = merged_data.isna().sum()
print(nan_counts[nan_counts > 0])

# Drop columns with too many missing values (more than 50%)
threshold = 0.5
merged_data = merged_data.loc[:, merged_data.isnull().mean() < threshold]

# Drop rows with remaining NaN values
merged_data.dropna(inplace=True)

# Step 2: Convert datetime columns to numeric features
merged_data['signup_time'] = pd.to_datetime(merged_data['signup_time'], errors='coerce')
merged_data['purchase_time'] = pd.to_datetime(merged_data['purchase_time'], errors='coerce')

# Extract relevant date components
merged_data['purchase_year'] = merged_data['purchase_time'].dt.year
merged_data['purchase_month'] = merged_data['purchase_time'].dt.month
merged_data['purchase_day'] = merged_data['purchase_time'].dt.day

# Drop original datetime columns if no longer needed
merged_data.drop(columns=['signup_time', 'purchase_time'], inplace=True)

# Step 3: Check unique counts for categorical features
unique_counts = merged_data.select_dtypes(include=['object', 'category']).nunique()
print("Unique counts for categorical features:")
print(unique_counts)

# Step 4: Use target encoding for high-cardinality categorical features
target_encoding_cols = ['source', 'browser']  # Adjust as necessary
for col in target_encoding_cols:
    if unique_counts[col] > 50:  # Set a threshold for high cardinality
        means = merged_data.groupby(col)['class'].mean()
        merged_data[col] = merged_data[col].map(means)

# Convert remaining categorical variables to dummies
low_cardinality_cols = unique_counts[unique_counts < 100].index.tolist()
merged_data = pd.get_dummies(merged_data[low_cardinality_cols + ['class']], drop_first=True, sparse=True)

# Print the shape of the DataFrame after cleaning
print("Shape of the DataFrame after cleaning:", merged_data.shape)

# Step 5: Define the features and target variable
X = merged_data.drop(columns=['class'])  # Exclude target variable
y = merged_data['class']

# Check if the feature set is empty
if X.empty:
    raise ValueError("Feature set X is empty. Please check your preprocessing steps.")

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Using SHAP for Explainability ---

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Step 8: SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

# Step 9: SHAP Force Plot for a specific instance
shap.initjs()  # Initialize JS for displaying plots in Jupyter

# Ensure index is valid and use the correct parameters
if X_test.shape[0] > 0:
    # Correct parameter order for force plot
    shap.plots.force(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])

# Step 10: SHAP Dependence Plot for a specific feature
feature_name = "purchase_value"  # Replace with an actual feature in your dataset
if feature_name in X_test.columns:
    shap.dependence_plot(feature_name, shap_values[1], X_test)
else:
    print(f"Feature '{feature_name}' not found in dataset.")

# --- Using LIME for Explainability ---

# Create LIME explainer
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['Non-Fraud', 'Fraud'], mode='classification')

# Explain a prediction
i = 0  # Index of the instance you want to explain (adjust as needed)
exp = lime_explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=10)

# LIME Feature Importance Plot
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation for Instance {}".format(i))
plt.show()

# Optionally, save SHAP plots as images if needed
# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig('shap_summary_plot.png')
