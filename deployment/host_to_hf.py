from huggingface_hub import HfApi
import os

api = HfApi()

space_repo = "srujanhj/tourism-wellness-space"

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=space_repo,
    repo_type="space",
    token=os.getenv("HF_TOKEN")
)

print("Deployment files pushed to Hugging Face Space successfully")
