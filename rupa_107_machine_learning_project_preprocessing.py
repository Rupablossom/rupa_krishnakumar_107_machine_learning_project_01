import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose

# Define file names
file1 = "Appointment.csv"
file2 = "Billing.csv"
file3 = "Doctor.csv"
file4 = "Medical procedure.csv"
file5 = "Patient.csv"

# Read CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)

# Ensure all DataFrames have the same number of rows before merging
min_rows = min(len(df1), len(df2), len(df3), len(df4), len(df5))
df1, df2, df3, df4, df5 = [df.iloc[:min_rows] for df in [df1, df2, df3, df4, df5]]

# Merge all DataFrames column-wise
merged_df = pd.concat([df1, df2, df3, df4, df5], axis=1)

# Save the merged file
merged_df.to_csv("rupa_107_decision_tree.csv", index=False)

# Display result
print(merged_df.head())

# Show summary statistics
data_detailing = merged_df.describe()
print(data_detailing)

# Show dataset info
print("\nDataset Info:")
merged_df.info()



"""below code will actually return True in that speicif cell if it is empty"""
df = pd.read_csv("rupa_107_decision_tree.csv")
finding_null_data = df.isna()
print(finding_null_data)

finding_null_data = df.isna().sum()
print(finding_null_data)

finding_null_data = df.isna().sum().sum()
print(finding_null_data)

# Fill null values in the 'Items' column with a 0
df['Items'].fillna('0', inplace=True)

finding_null_data = df.isna().sum()
print(finding_null_data)

# Load merged CSV
df = pd.read_csv("rupa_107_decision_tree.csv")

# Convert 'Date' column to datetime if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['DoctorName', 'Specialization', 'firstname', 'lastname', 'email', 'DoctorContact', 'ProcedureName','Items','InvoiceID','Date']

for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Drop 'Time' column if it exists
if 'Time' in df.columns:
    df.drop(columns=['Time'], inplace=True)

print("Updated Data Types:")
print(df.dtypes)

# Save the updated DataFrame to a new CSV file
df.to_csv("rupa_107_machine_learning_project.csv", index=False)

print("Updated file saved successfully as 'rupa_107_machine_learning_project'")






