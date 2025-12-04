# =============================================================================
# DATA REGISTRATION SCRIPT
# This script registers and uploads the tourism dataset to Hugging Face Hub
# =============================================================================

# Import RepositoryNotFoundError for handling cases when repository doesn't exist
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Import HfApi for API operations and create_repo for repository creation
from huggingface_hub import HfApi, create_repo

# Import os module for accessing environment variables (HF_TOKEN)
import os

# Define the repository ID where the dataset will be stored on Hugging Face
# This uses your Hugging Face username from the configuration
repo_id = "vksfdc2024/tourism-package-dataset"

# Specify the type of repository - can be 'dataset', 'model', or 'space'
repo_type = "dataset"

# Initialize the Hugging Face API client with authentication token from environment variable
# The HF_TOKEN should be set in GitHub Actions secrets for CI/CD pipeline
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository already exists on Hugging Face
try:
    # Attempt to get repository information - this will succeed if repo exists
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    # If successful, repository already exists
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    # If RepositoryNotFoundError is raised, create a new repository
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    # Create a new public dataset repository on Hugging Face
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    # Print success message after repository creation
    print(f"Dataset repository '{repo_id}' created successfully.")

# Step 2: Upload the entire data folder to the Hugging Face repository
# This uploads all files in the local data folder to the remote repository
api.upload_folder(
    folder_path="tourism_project/data",  # Local path to the folder containing tourism.csv
    repo_id=repo_id,  # Target repository ID on Hugging Face Hub
    repo_type=repo_type,  # Type of repository (dataset)
)

# Print final success message confirming dataset upload
print(f"Dataset uploaded successfully to {repo_id}")
