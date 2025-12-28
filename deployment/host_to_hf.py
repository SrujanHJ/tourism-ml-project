from huggingface_hub import HfApi
import os

# Initialize the API
api = HfApi()

# Your Hugging Face username and Space name
space_repo = "srujanhj/tourism-wellness-prediction"

# Folder to upload (relative to this script)
folder_to_upload = os.path.join(os.path.dirname(__file__))  # this points to deployment/

# Ensure your HF_TOKEN is set as an environment variable
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set the HF_TOKEN environment variable before running this script.")

# Upload the folder
api.upload_folder(
    folder_path=folder_to_upload,
    repo_id=space_repo,
    repo_type="space",
    token=hf_token
)

print("Deployment files pushed to Hugging Face Space successfully!")
