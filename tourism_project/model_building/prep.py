# =============================================================================
# DATA PREPARATION SCRIPT
# This script handles data loading, preprocessing, and train-test split
# =============================================================================

# Import pandas for data manipulation and analysis
import pandas as pd

# Import numpy for numerical operations and array manipulations
import numpy as np

# Import sklearn for machine learning utilities and version checking
import sklearn

# Import os for file and environment variable operations
import os

# Import train_test_split for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LabelEncoder for encoding categorical variables to numerical values
from sklearn.preprocessing import LabelEncoder

# Import Hugging Face API for uploading processed data to Hub
from huggingface_hub import HfApi

# Initialize Hugging Face API client with authentication token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the path to the dataset on Hugging Face Hub
DATASET_PATH = "hf://datasets/vksfdc2024/tourism-package-dataset/tourism.csv"

# Load the tourism dataset from Hugging Face into a pandas DataFrame
print("Loading dataset from Hugging Face Hub...")
df = pd.read_csv(DATASET_PATH)

# Print confirmation with dataset shape (rows, columns)
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Display first few rows to understand the data structure
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values in each column of the dataset
print("\nMissing values in each column:")
print(df.isnull().sum())

# =============================================================================
# DATA PREPROCESSING STEPS
# =============================================================================

# Step 1: Drop the CustomerID column as it's just an identifier
# CustomerID doesn't contribute to prediction and might cause data leakage
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)  # axis=1 means drop column
    print("\nDropped 'CustomerID' column - it's just an identifier")

# Step 2: Handle missing values
# Get list of numerical columns (excluding target variable)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove target variable 'ProdTaken' from numerical columns list if present
if 'ProdTaken' in numerical_cols:
    numerical_cols.remove('ProdTaken')

# Fill missing values in numerical columns with median (robust to outliers)
for col in numerical_cols:
    if df[col].isnull().sum() > 0:  # Check if column has missing values
        median_value = df[col].median()  # Calculate median of non-null values
        df[col].fillna(median_value, inplace=True)  # Fill missing with median
        print(f"Filled missing values in '{col}' with median: {median_value}")

# Get list of categorical columns (object type)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values in categorical columns with mode (most frequent value)
for col in categorical_cols:
    if df[col].isnull().sum() > 0:  # Check if column has missing values
        mode_value = df[col].mode()[0]  # Get the most frequent value
        df[col].fillna(mode_value, inplace=True)  # Fill missing with mode
        print(f"Filled missing values in '{col}' with mode: {mode_value}")

# Step 3: Clean Gender column - standardize values
# Handle inconsistent values like 'Fe Male' which should be 'Female'
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip()  # Remove leading/trailing spaces
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})  # Fix typo
    print("\nCleaned Gender column - standardized values")

# Step 4: Encode categorical variables using LabelEncoder
print("\nEncoding categorical variables...")

# Initialize LabelEncoder for converting text categories to numbers
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                       'MaritalStatus', 'Designation']

# Encode each categorical column by fitting and transforming
for col in categorical_features:
    if col in df.columns:  # Check if column exists in dataframe
        # Fit the encoder on column values and transform to numerical labels
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        print(f"Encoded '{col}' column to numerical values")

# Step 5: Define target variable column name
target_col = 'ProdTaken'

# Step 6: Separate features (X) and target variable (y)
# X contains all columns except the target variable (independent variables)
X = df.drop(columns=[target_col])

# y contains only the target variable (dependent variable)
y = df[target_col]

# Print shapes to verify separation
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Step 7: Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  # Features and target
    test_size=0.2,  # 20% of data reserved for testing
    random_state=42,  # Set seed for reproducibility
    stratify=y  # Maintain class distribution in train and test sets
)

# Print training and testing set sizes
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 8: Save the processed datasets as CSV files locally
# Save training features
X_train.to_csv("X_train.csv", index=False)
print("Saved X_train.csv")

# Save testing features
X_test.to_csv("X_test.csv", index=False)
print("Saved X_test.csv")

# Save training target
y_train.to_csv("y_train.csv", index=False)
print("Saved y_train.csv")

# Save testing target
y_test.to_csv("y_test.csv", index=False)
print("Saved y_test.csv")

# Step 9: Upload all processed files to Hugging Face Hub
# Define list of files to upload
files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

print("\nUploading processed files to Hugging Face Hub...")

# Loop through each file and upload to the dataset repository
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,  # Local file path to upload
        path_in_repo=file_path.split("/")[-1],  # Extract filename for repo
        repo_id="vksfdc2024/tourism-package-dataset",  # Repository ID
        repo_type="dataset",  # Repository type
    )
    print(f"Uploaded {file_path} to Hugging Face Hub")

# Print final success message
print("\nData preparation completed successfully!")
