import os
from huggingface_hub import HfApi

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "balamanian82"
SPACE_REPO  = f"{HF_USERNAME}/predictive-maintenance-app"

api = HfApi(token=HF_TOKEN)
api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    token=HF_TOKEN
)
print(f"Space repo ready: {SPACE_REPO}")

base = "predictive_maintenance/deployment"
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    fpath = f"{base}/{fname}"
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=SPACE_REPO,
            repo_type="space",
            token=HF_TOKEN
        )
        print(f"Uploaded: {fname}")
    else:
        print(f"WARNING: {fname} not found at {fpath}, skipping.")

print(f"App deployed at: https://huggingface.co/spaces/{SPACE_REPO}")
