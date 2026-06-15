import os
from huggingface_hub import HfApi

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USERNAME  = "balamanian82"
DATASET_REPO = f"{HF_USERNAME}/predictive-maintenance-engine"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
print(f"Dataset repo ready: {DATASET_REPO}")

api.upload_file(
    path_or_fileobj="predictive_maintenance/data/raw/engine_data.csv",
    path_in_repo="data/raw/engine_data.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)
print("Raw data uploaded to HF Hub.")
