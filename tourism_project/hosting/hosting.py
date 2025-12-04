# =============================================================================
# HOSTING SCRIPT
# This script uploads the deployment files to Hugging Face Space for hosting
# =============================================================================

# Import Hugging Face API for uploading files to the Space
from huggingface_hub import HfApi

# Import os for accessing environment variables
import os

# Initialize Hugging Face API with authentication token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the entire deployment folder to Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="vksfdc2024/tourism-package-prediction",
    repo_type="space",
    path_in_repo="",
)

# Print success message confirming upload
print("Deployment files uploaded successfully to Hugging Face Space!")
print("Your app will be available at: https://huggingface.co/spaces/vksfdc2024/tourism-package-prediction")
