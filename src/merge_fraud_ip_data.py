# File: merge_fraud_ip_data.py

# Import necessary libraries
import pandas as pd

# Set paths to cleaned data files
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_cleaned.csv"
ip_country_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/ip_country_cleaned.csv"

# Load the datasets
fraud_df = pd.read_csv(fraud_data_path)
ip_country_df = pd.read_csv(ip_country_data_path)

# Convert IP addresses to integer format
fraud_df['ip_address'] = fraud_df['ip_address'].apply(lambda x: int(x))
ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].apply(lambda x: int(x))
ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].apply(lambda x: int(x))

# Merge fraud dataset with IP-country dataset based on IP address range
# We perform a left join, merging on the condition that the IP address falls within the lower and upper bound of the IP range
merged_df = pd.merge(fraud_df, ip_country_df, how='left',
                     left_on='ip_address',
                     right_on='lower_bound_ip_address')

# Filter out only those rows where ip_address falls between the lower and upper bound IP ranges
merged_df = merged_df[(merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) & 
                      (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])]

# Save the merged dataset for further analysis
merged_output_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_with_country.csv"
merged_df.to_csv(merged_output_path, index=False)

print(f"Merged dataset saved to: {merged_output_path}")
