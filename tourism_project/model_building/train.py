# =============================================================================
# MODEL TRAINING SCRIPT (PRODUCTION ENVIRONMENT)
# This script trains the final model with MLflow tracking for production use
# =============================================================================

# Import necessary libraries for data manipulation
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations

# Import preprocessing utilities from scikit-learn
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.compose import make_column_transformer  # For preprocessing pipeline
from sklearn.pipeline import make_pipeline  # For model pipeline

# Import XGBoost for gradient boosting classification
import xgboost as xgb  # For XGBoost classifier

# Import model selection and evaluation utilities
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score  # For evaluation

# Import joblib for model serialization (saving/loading)
import joblib  # For model serialization

# Import os for environment variable access
import os  # For environment variables

# Import Hugging Face utilities for model upload
from huggingface_hub import HfApi, create_repo  # For Hugging Face operations
from huggingface_hub.utils import RepositoryNotFoundError  # For error handling

# Import mlflow for experiment tracking
import mlflow  # For experiment tracking

# Set MLflow tracking URI to local server (started by GitHub Actions)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment name for organizing MLflow runs
mlflow.set_experiment("tourism-model-training")

# Initialize Hugging Face API with authentication token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define paths to training and testing data on Hugging Face Hub
X_train_path = "hf://datasets/vksfdc2024/tourism-package-dataset/X_train.csv"
X_test_path = "hf://datasets/vksfdc2024/tourism-package-dataset/X_test.csv"
y_train_path = "hf://datasets/vksfdc2024/tourism-package-dataset/y_train.csv"
y_test_path = "hf://datasets/vksfdc2024/tourism-package-dataset/y_test.csv"

# Load preprocessed data from Hugging Face Hub
print("Loading data from Hugging Face Hub...")
X_train = pd.read_csv(X_train_path)  # Load training features
X_test = pd.read_csv(X_test_path)  # Load testing features
y_train = pd.read_csv(y_train_path).values.ravel()  # Load training target, convert to 1D array
y_test = pd.read_csv(y_test_path).values.ravel()  # Load testing target, convert to 1D array
print(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Calculate class weight for handling imbalanced data
class_counts = np.bincount(y_train)  # Count samples in each class
scale_pos_weight = class_counts[0] / class_counts[1]  # Ratio for weighting
print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

# Get list of numeric features for preprocessing
numeric_features = X_train.columns.tolist()

# Create preprocessing pipeline with StandardScaler
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)  # Apply scaling to all features
)

# Initialize XGBoost classifier with class weight for imbalance handling
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42,  # For reproducibility
    eval_metric='logloss'  # Use log loss for evaluation
)

# Define comprehensive hyperparameter grid for production tuning
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],  # Number of trees
    'xgbclassifier__max_depth': [3, 5, 7],  # Maximum tree depth
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9],  # Feature sampling per tree
    'xgbclassifier__subsample': [0.7, 0.8, 0.9],  # Sample sampling per tree
    'xgbclassifier__reg_lambda': [0.1, 0.5, 1.0],  # L2 regularization
}

# Create model pipeline combining preprocessing and classifier
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run for tracking the training experiment
with mlflow.start_run():
    print("\nStarting hyperparameter tuning with GridSearchCV...")
    
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        model_pipeline,  # Pipeline to optimize
        param_grid,  # Parameter grid
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',  # Optimize for ROC-AUC
        n_jobs=-1,  # Use all CPU cores
        verbose=2  # Show detailed progress
    )
    
    # Fit grid search on training data
    grid_search.fit(X_train, y_train)
    
    # Log all parameter combinations as nested runs for comparison
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        
        # Create nested run for each parameter combination
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)  # Log parameters
            mlflow.log_metric("mean_cv_roc_auc", mean_score)  # Log mean score
            mlflow.log_metric("std_cv_roc_auc", std_score)  # Log std score
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    
    # Log best parameters to main run
    print("\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    mlflow.log_params(grid_search.best_params_)
    
    # Make predictions on train and test sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Get classification report for detailed metrics
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Log all metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc,
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1_score": test_report['1']['f1-score']
    })
    
    # Print performance results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE RESULTS")
    print("="*50)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test Precision: {test_report['1']['precision']:.4f}")
    print(f"Test Recall: {test_report['1']['recall']:.4f}")
    print(f"Test F1-Score: {test_report['1']['f1-score']:.4f}")
    
    # Save model locally using joblib
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved locally as {model_path}")
    
    # Log model as artifact in MLflow for versioning
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model logged to MLflow as artifact")
    
    # Upload model to Hugging Face Hub for deployment
    repo_id = "vksfdc2024/tourism-package-model"
    repo_type = "model"
    
    # Check if model repository exists, create if not
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"\nModel repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"\nCreating model repository '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print("Model repository created successfully.")
    
    # Upload model file to Hugging Face Hub
    api.upload_file(
        path_or_fileobj=model_path,  # Local model file
        path_in_repo="best_tourism_model_v1.joblib",  # Path in repository
        repo_id=repo_id,  # Repository ID
        repo_type=repo_type,  # Repository type
    )
    print(f"Model uploaded to Hugging Face Hub: {repo_id}")

print("\nTraining completed successfully!")
